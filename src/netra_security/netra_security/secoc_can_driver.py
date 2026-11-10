"""
NETRA-UGV CAN-FD Motor Authentication Driver

Implements Secure On-Board Communication (SecOC) per AUTOSAR specification for
CAN-FD motor command authentication on industrial robotic platforms.

This is an OPTIONAL security layer for hardened BEL industrial deployments.
For standard outdoor navigation demonstrations, /cmd_vel dispatch is used directly.

Reference: MASTER_PROJECT_REPORT.md §1.4, TECHNICAL_ARCHITECTURE.md §4.4
  POSIX Priority: 80 (SCHED_FIFO) on Cores 0-1
  Frequency: 50 Hz dispatch rate
"""

import logging
import struct

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from netra_msgs.msg import FailsafeState, SecuredTwist
from netra_security.crypto_utils import (
    load_secoc_key,
    verify_secoc_payload,
)

logger = logging.getLogger(__name__)


class SecOCCanDriver(Node):
    """
    SecOC CAN-FD authentication and dispatch node.

    In production, this node interfaces with the physical isolated
    CAN-FD transceiver (2.5 kV galvanic isolation) connected to
    the ODrive v3.6 / VESC 6 motor drivers.

    In simulation mode, authenticated commands are logged to stdout
    and published as a standard geometry_msgs/Twist for Gazebo.
    """

    def __init__(self):
        super().__init__("secoc_can_driver")

        # --- Parameters ---
        self.declare_parameter("simulation_mode", True)
        self.declare_parameter("secoc_key_path", "")
        self.declare_parameter("can_bus_interface", "can0")
        self.declare_parameter("can_arbitration_id", 0x100)
        self.declare_parameter("dispatch_rate_hz", 50.0)

        self.simulation_mode = self.get_parameter("simulation_mode").value
        key_path = self.get_parameter("secoc_key_path").value or None
        self.can_interface = self.get_parameter("can_bus_interface").value
        self.can_arb_id = self.get_parameter("can_arbitration_id").value

        # --- Load SecOC Key ---
        self.secoc_key = load_secoc_key(key_path)
        self.last_valid_counter = 0

        # --- CAN Bus Interface (production only) ---
        self.can_bus = None
        if not self.simulation_mode:
            try:
                import can

                self.can_bus = can.interface.Bus(
                    channel=self.can_interface,
                    bustype="socketcan",
                    fd=True,
                )
                self.get_logger().info(f"CAN-FD bus opened on {self.can_interface}")
            except Exception as e:
                self.get_logger().error(f"Failed to open CAN-FD bus: {e}. " "Falling back to simulation mode.")
                self.simulation_mode = True

        # --- Failsafe State ---
        self.current_failsafe_mode = FailsafeState.MODE_NOMINAL

        # --- Subscribers ---
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.cmd_sub = self.create_subscription(
            SecuredTwist,
            "/cmd_vel_secured",
            self._on_secured_twist,
            qos_reliable,
        )

        self.failsafe_sub = self.create_subscription(
            FailsafeState,
            "/netra/failsafe_state",
            self._on_failsafe_state,
            qos_reliable,
        )

        # --- Statistics ---
        self.frames_dispatched = 0
        self.frames_rejected = 0

        self.get_logger().info(
            f"SecOC CAN-FD driver initialized " f"[sim={self.simulation_mode}, arb_id=0x{self.can_arb_id:03X}]"
        )

    def _on_failsafe_state(self, msg: FailsafeState):
        """Track the current failsafe mode to enforce actuation lockout."""
        self.current_failsafe_mode = msg.current_mode

    def _on_secured_twist(self, msg: SecuredTwist):
        """
        Process an incoming SecuredTwist command.

        1. Verify CMAC authentication tag.
        2. Verify freshness counter (anti-replay).
        3. Check failsafe state (Level 3 = lockout).
        4. Dispatch to CAN-FD bus or log.
        """
        # --- Level 3 Tamper Lockout ---
        if self.current_failsafe_mode == FailsafeState.MODE_SECURITY_TAMPER:
            self.get_logger().error(
                "TAMPER LOCKOUT: All actuation commands rejected. " "System is in zeroization state."
            )
            return

        # --- Verify SecOC Authentication ---
        received_mac = bytes(msg.aes_cmac)

        is_valid = verify_secoc_payload(
            linear_vel=msg.linear_velocity_x,
            angular_vel=msg.angular_velocity_z,
            counter_val=msg.freshness_counter,
            received_mac=received_mac,
            key=self.secoc_key,
            last_valid_counter=self.last_valid_counter,
        )

        if not is_valid:
            self.frames_rejected += 1
            self.get_logger().warn(f"SecOC frame REJECTED (total rejected: {self.frames_rejected})")
            return

        # --- Update anti-replay counter ---
        self.last_valid_counter = msg.freshness_counter
        self.frames_dispatched += 1

        # --- Dispatch to CAN-FD bus ---
        if self.simulation_mode:
            self.get_logger().debug(
                f"[SIM] CAN-FD dispatch: v={msg.linear_velocity_x:.3f} m/s, "
                f"ω={msg.angular_velocity_z:.3f} rad/s, "
                f"fc={msg.freshness_counter}, "
                f"total={self.frames_dispatched}"
            )
        else:
            self._dispatch_can_frame(msg)

    def _dispatch_can_frame(self, msg: SecuredTwist):
        """
        Construct and send the authenticated CAN-FD frame.

        CAN-FD payload layout (20 bytes):
          [0:4]   linear_velocity_x  (float32, big-endian)
          [4:8]   angular_velocity_z (float32, big-endian)
          [8:12]  freshness_counter  (uint32, big-endian)
          [12:20] aes_cmac           (8 bytes, truncated)
        """
        import can

        payload = (
            struct.pack(">ff", msg.linear_velocity_x, msg.angular_velocity_z)
            + struct.pack(">I", msg.freshness_counter)
            + bytes(msg.aes_cmac)
        )

        can_msg = can.Message(
            arbitration_id=self.can_arb_id,
            data=payload,
            is_fd=True,
            is_extended_id=False,
        )

        try:
            self.can_bus.send(can_msg)
        except can.CanError as e:
            self.get_logger().error(f"CAN-FD send failed: {e}")

    def destroy_node(self):
        """Clean shutdown: close CAN bus interface."""
        if self.can_bus is not None:
            self.can_bus.shutdown()
            self.get_logger().info("CAN-FD bus interface closed.")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SecOCCanDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
