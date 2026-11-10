"""
NETRA-UGV FAST Corner Feature Extractor
=========================================
Grid-based FAST corner detection with adaptive thresholding.

Divides the frame into NxM cells and extracts the top-K strongest
FAST corners per cell to ensure uniform feature distribution across
the image, preventing feature clustering in textured regions.

Reference: MASTER_PROJECT_REPORT.md §3.3, TECHNICAL_ARCHITECTURE.md §2
  Uses ARM NEON-accelerated OpenCV FAST detector
  Execution: Sub-millisecond on ARM Cortex-A78AE
"""

from typing import List

import cv2
import numpy as np


class FASTFeatureExtractor:
    """
    Grid-based FAST corner feature extractor with adaptive thresholding.
    """

    def __init__(
        self,
        fast_threshold: int = 20,
        nonmax_suppression: bool = True,
        grid_rows: int = 4,
        grid_cols: int = 6,
        max_features_per_cell: int = 25,
        min_features_total: int = 100,
    ):
        """
        Args:
            fast_threshold:         FAST corner detection threshold.
            nonmax_suppression:     Enable non-maximum suppression.
            grid_rows:              Number of grid cell rows.
            grid_cols:              Number of grid cell columns.
            max_features_per_cell:  Maximum features to retain per cell.
            min_features_total:     Minimum total features; if below this,
                                    lower threshold and retry.
        """
        self.fast_threshold = fast_threshold
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.max_per_cell = max_features_per_cell
        self.min_total = min_features_total

        # Create FAST detector
        self.detector = cv2.FastFeatureDetector_create(
            threshold=fast_threshold,
            nonmaxSuppression=nonmax_suppression,
            type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16,
        )

    def detect(self, gray_image: np.ndarray) -> List[cv2.KeyPoint]:
        """
        Detect FAST corners with grid-based distribution.

        Args:
            gray_image: Grayscale image of shape (H, W), dtype uint8.

        Returns:
            List of cv2.KeyPoint objects, uniformly distributed across
            the image grid.
        """
        H, W = gray_image.shape[:2]
        cell_h = H // self.grid_rows
        cell_w = W // self.grid_cols

        all_keypoints = []

        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                # Extract cell region
                y_start = row * cell_h
                y_end = (row + 1) * cell_h if row < self.grid_rows - 1 else H
                x_start = col * cell_w
                x_end = (col + 1) * cell_w if col < self.grid_cols - 1 else W

                cell = gray_image[y_start:y_end, x_start:x_end]

                # Detect FAST corners in cell
                keypoints = self.detector.detect(cell)

                # Sort by response (strength) and keep top-K
                keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)
                keypoints = keypoints[: self.max_per_cell]

                # Offset keypoint coordinates to full-frame
                for kp in keypoints:
                    kp.pt = (kp.pt[0] + x_start, kp.pt[1] + y_start)

                all_keypoints.extend(keypoints)

        # Adaptive threshold: if too few features, lower threshold and retry
        if len(all_keypoints) < self.min_total:
            backup_detector = cv2.FastFeatureDetector_create(
                threshold=max(5, self.fast_threshold // 2),
                nonmaxSuppression=True,
            )
            backup_kps = backup_detector.detect(gray_image)
            backup_kps = sorted(backup_kps, key=lambda kp: kp.response, reverse=True)
            # Supplement with additional features
            needed = self.min_total - len(all_keypoints)
            all_keypoints.extend(backup_kps[:needed])

        return all_keypoints

    def extract_points(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Convenience method: detect and return as Nx2 numpy array.

        Returns:
            Nx2 float32 array of (x, y) pixel coordinates.
        """
        keypoints = self.detect(gray_image)
        if not keypoints:
            return np.empty((0, 2), dtype=np.float32)

        points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        return points
