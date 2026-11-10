"""
NETRA-UGV 4-Tier Deterministic Failsafe Manager
=================================================
Implements the 4-Level Deterministic Failsafe Hierarchy:
  - Level 0: NOMINAL (Full visual-inertial autonomy)
  - Level 1: VISION_DEGRADED (Glare/dust washout, speed cap 0.8 m/s)
  - Level 2: VISION_CRITICAL (Mud/smoke, 1.5s air purge, limp-to-halt)
  - Level 3: CRITICAL_SENSOR_BLACKOUT (Complete sensor blackout, emergency safe halt)

Subscribes to:
  /netra/tamper_alert           (std_msgs/Bool or GPIO monitor)
  /camera/lens_metrics          (std_msgs/Float32 lens obscuration)

Publishes:
  /netra/failsafe_state         (netra_msgs/FailsafeState at 10 Hz)

Reference:
  MASTER_PROJECT_REPORT.md §4 & TECHNICAL_ARCHITECTURE.md §4.3
"""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Fallback imports if rclpy or ROS 2 environment is not installed
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

    _HAS_RCLPY = True
except ImportError:
    _HAS_RCLPY = False

    class Node:
        def __init__(self, *args, **kwargs):
            pass

        def declare_parameter(self, *args, **kwargs):
            pass

        def get_parameter(self, *args, **kwargs):
            class _P:
                def get_parameter_value(self):
                    class _V:
                        double_value = 1.0
                        string_value = ""

                    return _V()

            return _P()

        def create_publisher(self, *args, **kwargs):
            pass

        def create_subscription(self, *args, **kwargs):
            pass

        def create_timer(self, *args, **kwargs):
            pass

        def get_logger(self):
            return logger

        def get_clock(self):
            class _C:
                def now(self):
                    class _T:
                        def to_msg(self):
                            return None

                    return _T()

            return _C()

    class QoSProfile:
        def __init__(self, *args, **kwargs):
            pass

    class ReliabilityPolicy:
        RELIABLE = 1
        BEST_EFFORT = 2

    class HistoryPolicy:
        KEEP_LAST = 1


try:
    from std_msgs.msg import Bool, Float32, Header

    from netra_msgs.msg import FailsafeState
except ImportError:

    class Header:
        def __init__(self):
            self.stamp = None
            self.frame_id = ""

    class Bool:
        def __init__(self):
            self.data = False

    class Float32:
        def __init__(self):
            self.data = 0.0

    class FailsafeState:
        def __init__(self):
            self.header = Header()
            self.current_mode = 0
            self.camera_lens_obscuration = 0.0
            self.v_max_allowed = 1.5
            self.air_purge_firing = False
            self.zeroization_engaged = False


# Constants matching netra_msgs/FailsafeState
MODE_NOMINAL = 0
MODE_VISION_DEGRADED = 1
MODE_VISION_CRITICAL = 2
MODE_SECURITY_TAMPER = 3


class FailsafeStateMachine:
    """
    Pure algorithmic finite state machine logic for failsafe transitions.
    """

    def __init__(
        self,
        obscuration_degraded_thresh: float = 0.35,
        obscuration_critical_thresh: float = 0.70,
        obscuration_recovery_thresh: float = 0.20,
        air_purge_duration_sec: float = 1.5,
        recovery_dwell_sec: float = 1.0,
    ):
        self.degraded_thresh = obscuration_degraded_thresh
        self.critical_thresh = obscuration_critical_thresh
        self.recovery_thresh = obscuration_recovery_thresh
        self.air_purge_duration = air_purge_duration_sec
        self.recovery_dwell = recovery_dwell_sec

        self.current_mode: int = MODE_NOMINAL
        self.lens_obscuration: float = 0.0
        self.air_purge_active: bool = False
        self.zeroization_engaged: bool = False

        self._purge_timer: float = 0.0
        self._recovery_timer: float = 0.0
        self._critical_dwell: float = 0.0

    def update(
        self,
        lens_obscuration: float,
        tamper_detected: bool,
        dt: float = 0.1,
    ) -> Tuple[int, float, bool, bool]:
        """
        Cycle the state machine.

        Args:
            lens_obscuration: Current estimate of optical blockage [0.0 - 1.0]
            tamper_detected:  Flag from physical hull breach monitor
            dt:               Time elapsed since last update in seconds

        Returns:
            Tuple of:
              - current_mode:            0, 1, 2, or 3
              - v_max_allowed:           Dynamic maximum speed ceiling
              - air_purge_active:        Active state of pneumatic lens nozzle
              - zeroization_engaged:     Active state of cryptographic crowbar circuit
        """
        self.lens_obscuration = float(np_clip(lens_obscuration, 0.0, 1.0))

        # --- LEVEL 3: CRITICAL SENSOR BLACKOUT / TAMPER (Emergency Safe Halt) ---
        if tamper_detected or self.zeroization_engaged:
            self.current_mode = MODE_SECURITY_TAMPER
            self.zeroization_engaged = True
            self.air_purge_active = False
            return MODE_SECURITY_TAMPER, 0.0, False, True

        # --- LEVEL 2: VISION CRITICAL ---
        if self.lens_obscuration >= self.critical_thresh:
            self._critical_dwell += dt
            if self._critical_dwell >= 0.5:  # Confirmed for 500 ms
                if self.current_mode != MODE_VISION_CRITICAL:
                    logger.warning("Failsafe: Entering LEVEL 2 (VISION CRITICAL) — Firing air purge")
                    self.current_mode = MODE_VISION_CRITICAL
                    self._purge_timer = self.air_purge_duration
                    self.air_purge_active = True

        # Service active air purge pulse
        if self.air_purge_active:
            self._purge_timer -= dt
            if self._purge_timer <= 0.0:
                self.air_purge_active = False
                logger.info("Failsafe: Air purge pulse cycle finished")

        # --- LEVEL 1: VISION DEGRADED ---
        if self.current_mode == MODE_NOMINAL:
            if self.lens_obscuration >= self.degraded_thresh and self.lens_obscuration < self.critical_thresh:
                logger.info("Failsafe: Entering LEVEL 1 (VISION DEGRADED) — Capping speed at 0.8 m/s")
                self.current_mode = MODE_VISION_DEGRADED
                self._recovery_timer = 0.0

        # --- RECOVERY LOGIC (From Degraded or Critical back toward Nominal) ---
        if self.current_mode == MODE_VISION_DEGRADED:
            if self.lens_obscuration < self.recovery_thresh:
                self._recovery_timer += dt
                if self._recovery_timer >= self.recovery_dwell:
                    logger.info("Failsafe: Recovered to LEVEL 0 (NOMINAL OPERATION)")
                    self.current_mode = MODE_NOMINAL
                    self._recovery_timer = 0.0
            else:
                self._recovery_timer = 0.0

        elif self.current_mode == MODE_VISION_CRITICAL:
            if self.lens_obscuration < self.degraded_thresh and not self.air_purge_active:
                logger.info("Failsafe: Vision partially cleared, stepping down to LEVEL 1")
                self.current_mode = MODE_VISION_DEGRADED
                self._critical_dwell = 0.0

        # Compute dynamic velocity ceiling based on mode
        if self.current_mode == MODE_NOMINAL:
            v_max = 1.5
        elif self.current_mode == MODE_VISION_DEGRADED:
            v_max = 0.8
        elif self.current_mode == MODE_VISION_CRITICAL:
            v_max = 0.0  # Controlled halt
        else:
            v_max = 0.0

        return self.current_mode, v_max, self.air_purge_active, self.zeroization_engaged


def np_clip(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


class FailsafeNode(Node):
    """
    ROS 2 Node publishing the system-wide failsafe state at 10 Hz.
    """

    def __init__(self):
        super().__init__("failsafe_manager")

        # Declare parameters
        self.declare_parameter("obscuration_degraded_thresh", 0.35)
        self.declare_parameter("obscuration_critical_thresh", 0.70)
        self.declare_parameter("air_purge_duration_sec", 1.5)
        self.declare_parameter("publish_rate_hz", 10.0)

        degraded_th = self.get_parameter("obscuration_degraded_thresh").get_parameter_value().double_value
        critical_th = self.get_parameter("obscuration_critical_thresh").get_parameter_value().double_value
        purge_dur = self.get_parameter("air_purge_duration_sec").get_parameter_value().double_value

        self.fsm = FailsafeStateMachine(
            obscuration_degraded_thresh=degraded_th,
            obscuration_critical_thresh=critical_th,
            air_purge_duration_sec=purge_dur,
        )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Publisher
        self.state_pub = self.create_publisher(FailsafeState, "/netra/failsafe_state", qos)

        # Subscribers
        self.tamper_sub = self.create_subscription(Bool, "/netra/tamper_alert", self._tamper_cb, qos)
        self.zero_sub = self.create_subscription(Bool, "/netra/zeroization_status", self._tamper_cb, qos)
        self.lens_sub = self.create_subscription(Float32, "/camera/lens_metrics", self._lens_cb, qos)

        self._tamper_flag = False
        self._lens_obscuration = 0.0

        # Periodic timer (10 Hz)
        rate = self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        self.timer = self.create_timer(1.0 / rate, self._timer_cb)

        self.get_logger().info("Failsafe Manager initialized. Nominal state active.")

    def _tamper_cb(self, msg: Bool):
        if msg.data:
            self._tamper_flag = True
            self.get_logger().critical("TAMPER ALERT RECEIVED IN FAILSAFE MANAGER!")

    def _lens_cb(self, msg: Float32):
        self._lens_obscuration = msg.data

    def _timer_cb(self):
        mode, v_max, purge_on, zeroized = self.fsm.update(self._lens_obscuration, self._tamper_flag, dt=0.1)

        msg = FailsafeState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.current_mode = mode
        msg.camera_lens_obscuration = float(self._lens_obscuration)
        msg.v_max_allowed = float(v_max)
        msg.air_purge_firing = purge_on
        msg.zeroization_engaged = zeroized

        self.state_pub.publish(msg)


def main(args=None):
    if not _HAS_RCLPY:
        logger.error("rclpy not installed in current environment.")
        return
    rclpy.init(args=args)
    node = FailsafeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
