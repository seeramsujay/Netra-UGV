"""
NETRA-UGV Cryptographic Utilities
=================================
AES-128 CMAC generation for SecOC CAN-FD frame authentication,
monotonic freshness counter management, and key loading utilities.

Reference: AUTOSAR SecOC Specification & MASTER_PROJECT_REPORT.md §1.4
  CAN Payload = [v, ω, FreshnessCounter, MAC]
  MAC = AES-128 CMAC (truncated to 64-bit) over [v || ω || FreshnessCounter]
"""

import hashlib
import hmac
import logging
import os
import struct
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AES-128 CMAC Implementation
# ---------------------------------------------------------------------------
# Uses the 'cryptography' library when available (production),
# falls back to HMAC-SHA256 truncated for simulation environments.

try:
    from cryptography.hazmat.primitives.ciphers.algorithms import AES
    from cryptography.hazmat.primitives.cmac import CMAC

    _HAS_CRYPTO_LIB = True
except ImportError:
    _HAS_CRYPTO_LIB = False
    logger.warning("python3-cryptography not available. " "Falling back to HMAC-SHA256 truncated for simulation.")


class FreshnessCounter:
    """
    Monotonically incrementing anti-replay freshness counter.

    In production, this counter is backed by battery-retained SRAM
    and survives power cycles. In simulation, it resets on restart.

    The counter is a 32-bit unsigned integer that wraps at 2^32 - 1,
    at which point the system triggers a key rotation event.
    """

    MAX_VALUE = 0xFFFFFFFF  # 2^32 - 1

    def __init__(self, initial_value: int = 0):
        self._counter = initial_value

    @property
    def value(self) -> int:
        return self._counter

    def increment(self) -> int:
        """Increment and return the new counter value."""
        if self._counter >= self.MAX_VALUE:
            logger.critical("Freshness counter overflow! Key rotation required.")
            raise OverflowError("Freshness counter exhausted — rotate keys")
        self._counter += 1
        return self._counter

    def to_bytes(self) -> bytes:
        """Pack counter as 4-byte big-endian unsigned integer."""
        return struct.pack(">I", self._counter)


def load_secoc_key(key_path: Optional[str] = None) -> bytes:
    """
    Load the 128-bit (16-byte) AES pre-shared key for SecOC CMAC.

    In production, this key is stored in the Infineon Optiga TPM 2.0
    secure element and never leaves the hardware boundary.

    In simulation, we load from a YAML-configured file path or
    generate a deterministic test key.

    Args:
        key_path: Filesystem path to the raw 16-byte key file.
                  If None, a deterministic test key is generated.

    Returns:
        16-byte AES-128 key.
    """
    if key_path and os.path.exists(key_path):
        with open(key_path, "rb") as f:
            key = f.read(16)
        if len(key) != 16:
            raise ValueError(f"SecOC key file must be exactly 16 bytes, got {len(key)}")
        logger.info("SecOC AES-128 key loaded from TPM-backed keystore.")
        return key

    # Deterministic simulation key (NOT for production use)
    test_key = hashlib.sha256(b"NETRA_UGV_SECOC_SIM_KEY_v1").digest()[:16]
    logger.warning("Using deterministic SIMULATION SecOC key. " "DO NOT deploy to production hardware.")
    return test_key


def compute_aes128_cmac(key: bytes, data: bytes) -> bytes:
    """
    Compute AES-128 CMAC over the input data.

    Returns the full 16-byte CMAC tag. The caller is responsible
    for truncating to the required 8-byte (64-bit) tag for CAN-FD.

    Args:
        key:  16-byte AES-128 key.
        data: Arbitrary-length input data to authenticate.

    Returns:
        16-byte CMAC tag.
    """
    if _HAS_CRYPTO_LIB:
        c = CMAC(AES(key))
        c.update(data)
        return c.finalize()
    else:
        # Fallback: HMAC-SHA256 truncated to 16 bytes (simulation only)
        mac = hmac.new(key, data, hashlib.sha256).digest()[:16]
        return mac


def build_secoc_payload(
    linear_vel: float,
    angular_vel: float,
    freshness: FreshnessCounter,
    key: bytes,
) -> Tuple[int, bytes]:
    """
    Build a SecOC-authenticated CAN-FD payload.

    Constructs the data frame:
      [linear_vel (f32) | angular_vel (f32) | freshness_counter (u32)]
    Then computes the AES-128 CMAC and truncates to 64-bit (8 bytes).

    Reference: MASTER_PROJECT_REPORT.md §1.4
      CAN Payload = [v, ω, FreshnessCounter, MAC]

    Args:
        linear_vel:  Target forward velocity in m/s.
        angular_vel: Target yaw rate in rad/s.
        freshness:   FreshnessCounter instance.
        key:         16-byte AES-128 pre-shared key.

    Returns:
        Tuple of (freshness_counter_value, truncated_8byte_cmac).
    """
    counter_val = freshness.increment()

    # Pack the authenticated data region
    auth_data = struct.pack(">ff", linear_vel, angular_vel) + freshness.to_bytes()

    # Compute full 16-byte CMAC, then truncate to 8 bytes for CAN-FD
    full_mac = compute_aes128_cmac(key, auth_data)
    truncated_mac = full_mac[:8]

    return counter_val, truncated_mac


def verify_secoc_payload(
    linear_vel: float,
    angular_vel: float,
    counter_val: int,
    received_mac: bytes,
    key: bytes,
    last_valid_counter: int,
) -> bool:
    """
    Verify a received SecOC CAN-FD payload.

    Checks:
      1. Freshness counter is strictly greater than the last accepted value
         (anti-replay protection).
      2. Truncated CMAC matches the recomputed value.

    Args:
        linear_vel:         Received forward velocity.
        angular_vel:        Received yaw rate.
        counter_val:        Received freshness counter.
        received_mac:       Received 8-byte truncated CMAC.
        key:                16-byte AES-128 pre-shared key.
        last_valid_counter: Last successfully verified counter value.

    Returns:
        True if the payload is authentic and fresh, False otherwise.
    """
    # Anti-replay check
    if counter_val <= last_valid_counter:
        logger.warning(
            f"SecOC REPLAY ATTACK detected! " f"Received counter {counter_val} <= last valid {last_valid_counter}"
        )
        return False

    # Recompute MAC
    auth_data = struct.pack(">ff", linear_vel, angular_vel) + struct.pack(">I", counter_val)
    full_mac = compute_aes128_cmac(key, auth_data)
    expected_mac = full_mac[:8]

    if not hmac.compare_digest(received_mac, expected_mac):
        logger.warning("SecOC MAC VERIFICATION FAILED — potential CAN injection attack!")
        return False

    return True
