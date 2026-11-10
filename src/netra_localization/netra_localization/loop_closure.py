"""
NETRA-UGV Asynchronous Loop Closure Module
============================================
Runs at 1 Hz in a low-priority background thread to detect and correct
drift accumulation by matching the current view against a database of
previously visited keyframes using ORB bag-of-words descriptors.

Reference: MASTER_PROJECT_REPORT.md §3.3
  1 Hz low-priority thread, keyframe place recognition
  Bounds long-term drift to < 1.2% over 500m

Reference: TECHNICAL_ARCHITECTURE.md §2
  POSIX Priority: 10 (SCHED_OTHER) — netra_loop thread
"""

import logging
import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Keyframe:
    """Represents a stored keyframe for loop closure matching."""

    def __init__(
        self,
        frame_id: int,
        descriptors: np.ndarray,
        keypoints: list,
        pose_R: np.ndarray,
        pose_p: np.ndarray,
        timestamp: float,
    ):
        self.frame_id = frame_id
        self.descriptors = descriptors
        self.keypoints = keypoints
        self.pose_R = pose_R.copy()
        self.pose_p = pose_p.copy()
        self.timestamp = timestamp


class LoopClosureDetector:
    """
    ORB-based loop closure detector with geometric verification.

    Maintains a sparse keyframe database and performs descriptor matching
    + fundamental matrix RANSAC verification to detect revisited locations.
    """

    def __init__(
        self,
        min_keyframe_distance: float = 2.0,
        min_time_gap: float = 10.0,
        match_threshold: float = 0.7,
        min_inliers: int = 20,
        max_keyframes: int = 200,
    ):
        """
        Args:
            min_keyframe_distance: Minimum distance (m) between keyframes.
            min_time_gap:          Minimum time (s) between query and candidate.
            match_threshold:       Lowe's ratio test threshold.
            min_inliers:           Minimum RANSAC inliers for a valid loop.
            max_keyframes:         Maximum keyframes in the database.
        """
        self.min_kf_distance = min_keyframe_distance
        self.min_time_gap = min_time_gap
        self.match_threshold = match_threshold
        self.min_inliers = min_inliers
        self.max_keyframes = max_keyframes

        # ORB detector and matcher
        self.orb = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Keyframe database
        self.keyframes: List[Keyframe] = []
        self._next_id = 0

        # Thread safety
        self._lock = threading.Lock()

    def should_add_keyframe(
        self,
        current_position: np.ndarray,
        current_time: float,
    ) -> bool:
        """Check if current pose is far enough from last keyframe."""
        if not self.keyframes:
            return True

        last_kf = self.keyframes[-1]
        distance = np.linalg.norm(current_position - last_kf.pose_p)
        time_gap = current_time - last_kf.timestamp

        return distance >= self.min_kf_distance or time_gap >= self.min_time_gap * 2

    def add_keyframe(
        self,
        gray_image: np.ndarray,
        pose_R: np.ndarray,
        pose_p: np.ndarray,
        timestamp: float,
    ) -> int:
        """
        Add a new keyframe to the database.

        Args:
            gray_image: Grayscale image (uint8).
            pose_R:     3x3 rotation matrix at this keyframe.
            pose_p:     3-vector position at this keyframe.
            timestamp:  Timestamp of the keyframe.

        Returns:
            Keyframe ID.
        """
        keypoints, descriptors = self.orb.detectAndCompute(gray_image, None)

        if descriptors is None or len(keypoints) < 10:
            return -1

        with self._lock:
            kf = Keyframe(
                frame_id=self._next_id,
                descriptors=descriptors,
                keypoints=keypoints,
                pose_R=pose_R,
                pose_p=pose_p,
                timestamp=timestamp,
            )
            self.keyframes.append(kf)
            self._next_id += 1

            # Limit database size
            if len(self.keyframes) > self.max_keyframes:
                # Remove oldest keyframes (keep first and last 50)
                self.keyframes = self.keyframes[:25] + self.keyframes[-self.max_keyframes + 25 :]

        return kf.frame_id

    def detect_loop(
        self,
        gray_image: np.ndarray,
        current_position: np.ndarray,
        current_time: float,
    ) -> Optional[Tuple[int, np.ndarray, np.ndarray]]:
        """
        Attempt to detect a loop closure against the keyframe database.

        Args:
            gray_image:       Current grayscale image.
            current_position: Current vehicle position (3-vector).
            current_time:     Current timestamp.

        Returns:
            If loop detected: (matched_kf_id, relative_R, relative_t)
            Otherwise: None
        """
        keypoints, descriptors = self.orb.detectAndCompute(gray_image, None)
        if descriptors is None or len(keypoints) < 10:
            return None

        best_match = None
        best_inliers = 0

        with self._lock:
            candidates = self.keyframes.copy()

        for kf in candidates:
            # Skip recent keyframes (must have sufficient temporal gap)
            time_gap = abs(current_time - kf.timestamp)
            if time_gap < self.min_time_gap:
                continue

            # Skip keyframes that are very close spatially (not a real loop)
            spatial_distance = np.linalg.norm(current_position - kf.pose_p)
            if spatial_distance > 50.0:  # Too far — unlikely to be same place
                continue

            # Descriptor matching with Lowe's ratio test
            try:
                matches = self.matcher.knnMatch(descriptors, kf.descriptors, k=2)
            except cv2.error:
                continue

            good_matches = []
            for m_pair in matches:
                if len(m_pair) == 2:
                    m, n = m_pair
                    if m.distance < self.match_threshold * n.distance:
                        good_matches.append(m)

            if len(good_matches) < self.min_inliers:
                continue

            # Geometric verification via fundamental matrix RANSAC
            src_pts = np.array(
                [keypoints[m.queryIdx].pt for m in good_matches],
                dtype=np.float32,
            )
            dst_pts = np.array(
                [kf.keypoints[m.trainIdx].pt for m in good_matches],
                dtype=np.float32,
            )

            F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 2.0)

            if mask is None:
                continue

            n_inliers = int(mask.sum())

            if n_inliers >= self.min_inliers and n_inliers > best_inliers:
                # Estimate relative pose (simplified using essential matrix)
                E = F  # Approximation without K correction for speed
                _, R_rel, t_rel, _ = cv2.recoverPose(
                    E,
                    src_pts[mask.flatten() == 1],
                    dst_pts[mask.flatten() == 1],
                )

                best_match = (kf.frame_id, R_rel, t_rel.flatten())
                best_inliers = n_inliers

        if best_match is not None:
            logger.info(f"Loop closure detected! Matched KF #{best_match[0]} " f"with {best_inliers} inliers.")

        return best_match

    @property
    def num_keyframes(self) -> int:
        with self._lock:
            return len(self.keyframes)
