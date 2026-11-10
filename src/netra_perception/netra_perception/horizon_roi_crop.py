"""
NETRA-UGV Dynamic Ground-Horizon ROI Crop
==========================================
Uses the vehicle's pitch angle from the onboard IMU to dynamically
crop the camera frame to the ground corridor, eliminating irrelevant
sky, mountain crests, and vehicle chassis.

Reference: MASTER_PROJECT_REPORT.md §3.1, TECHNICAL_ARCHITECTURE.md §5.1
  v_horizon = f_y * tan(θ_pitch) + c_y
  I_ROI = I[max(0, v_horizon - Δv) : H - δ_chassis, 0 : W]

This eliminates ~42% of the image surface area, reducing TensorRT INT8
inference latency from 4.2 ms → 2.6 ms.
"""

import math
from typing import Tuple

import numpy as np


class HorizonROICropper:
    """
    Dynamically crops the camera image to the traversable ground corridor
    using the vehicle's pitch angle from the IMU/VIO orientation.
    """

    def __init__(
        self,
        focal_length_y: float = 500.0,
        principal_point_y: float = 360.0,
        image_height: int = 720,
        image_width: int = 1280,
        horizon_margin_px: int = 50,
        chassis_crop_px: int = 40,
    ):
        """
        Args:
            focal_length_y:    Camera intrinsic f_y (pixels).
            principal_point_y: Camera intrinsic c_y (pixels).
            image_height:      Full frame height (pixels).
            image_width:       Full frame width (pixels).
            horizon_margin_px: Extra margin above horizon line (Δv pixels).
            chassis_crop_px:   Pixels to crop from bottom (vehicle hood/chassis).
        """
        self.f_y = focal_length_y
        self.c_y = principal_point_y
        self.H = image_height
        self.W = image_width
        self.delta_v = horizon_margin_px
        self.delta_chassis = chassis_crop_px

        # Cache for last computed ROI bounds
        self._last_row_start = 0
        self._last_row_end = image_height

    def compute_horizon_row(self, pitch_rad: float) -> int:
        """
        Compute the pixel row of the terrain horizon line.

        Formula: v_horizon = f_y * tan(θ_pitch) + c_y

        Args:
            pitch_rad: Vehicle pitch angle in radians (positive = nose up).

        Returns:
            Pixel row index of the horizon line.
        """
        v_horizon = self.f_y * math.tan(pitch_rad) + self.c_y
        return int(np.clip(v_horizon, 0, self.H - 1))

    def crop(
        self,
        image: np.ndarray,
        pitch_rad: float,
    ) -> Tuple[np.ndarray, int, int]:
        """
        Crop the image to the ground corridor ROI.

        Args:
            image:     Input image of shape (H, W, C) or (H, W).
            pitch_rad: Current vehicle pitch angle in radians.

        Returns:
            Tuple of:
              - Cropped image (ROI only)
              - row_start: Starting row index in the original frame
              - row_end:   Ending row index in the original frame
        """
        v_horizon = self.compute_horizon_row(pitch_rad)

        # ROI: from (horizon - margin) to (bottom - chassis)
        row_start = max(0, v_horizon - self.delta_v)
        row_end = self.H - self.delta_chassis

        # Safety: ensure valid range
        if row_start >= row_end:
            row_start = 0
            row_end = self.H

        self._last_row_start = row_start
        self._last_row_end = row_end

        cropped = image[row_start:row_end, :].copy()
        return cropped, row_start, row_end

    def get_crop_reduction_pct(self) -> float:
        """Return the percentage of image area eliminated by the crop."""
        roi_height = self._last_row_end - self._last_row_start
        total_pixels = self.H * self.W
        roi_pixels = roi_height * self.W
        return (1.0 - roi_pixels / total_pixels) * 100.0

    def map_to_full_frame(
        self,
        roi_row: int,
        roi_col: int,
    ) -> Tuple[int, int]:
        """
        Map pixel coordinates from the cropped ROI back to the full frame.

        Args:
            roi_row: Row in the cropped image.
            roi_col: Column in the cropped image.

        Returns:
            (full_row, full_col) in the original frame.
        """
        return (roi_row + self._last_row_start, roi_col)
