"""
NETRA-UGV Multi-Scale KLT Sparse Optical Flow Tracker
=======================================================
Tracks FAST features across consecutive frames using the
Kanade-Lucas-Tomasi (KLT) pyramidal optical flow algorithm.

Manages feature track lifecycle:
  - Initiation of new tracks from FAST detections
  - Frame-to-frame tracking via OpenCV calcOpticalFlowPyrLK
  - Outlier rejection using fundamental matrix RANSAC
  - Track ID assignment and lifecycle management

Reference: MASTER_PROJECT_REPORT.md §3.3
  Multi-level KLT sparse optical flow using ARM NEON SIMD intrinsics
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FeatureTrack:
    """Represents a single feature track across multiple frames."""

    _next_id = 0

    def __init__(self, point: np.ndarray):
        self.id = FeatureTrack._next_id
        FeatureTrack._next_id += 1
        self.observations: List[np.ndarray] = [point.copy()]
        self.age = 1
        self.is_active = True

    def add_observation(self, point: np.ndarray):
        self.observations.append(point.copy())
        self.age += 1

    def last_point(self) -> np.ndarray:
        return self.observations[-1]


class KLTTracker:
    """
    Multi-scale KLT sparse optical flow tracker with lifecycle management.
    """

    def __init__(
        self,
        max_tracks: int = 200,
        min_tracks: int = 80,
        max_track_age: int = 30,
        lk_win_size: Tuple[int, int] = (21, 21),
        lk_max_level: int = 3,
        lk_criteria: Tuple = None,
        ransac_threshold: float = 1.0,
        min_track_length: int = 3,
    ):
        """
        Args:
            max_tracks:       Maximum number of simultaneous feature tracks.
            min_tracks:       Re-detect features when active tracks fall below this.
            max_track_age:    Maximum frames before a track is retired.
            lk_win_size:      KLT search window size.
            lk_max_level:     Number of pyramid levels for multi-scale tracking.
            lk_criteria:      KLT termination criteria.
            ransac_threshold: Fundamental matrix RANSAC inlier threshold (pixels).
            min_track_length: Minimum observations before a track is used for VIO.
        """
        self.max_tracks = max_tracks
        self.min_tracks = min_tracks
        self.max_age = max_track_age
        self.min_track_length = min_track_length

        self.lk_params = dict(
            winSize=lk_win_size,
            maxLevel=lk_max_level,
            criteria=lk_criteria or (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.ransac_thresh = ransac_threshold

        # Track storage
        self.active_tracks: Dict[int, FeatureTrack] = {}
        self.lost_tracks: List[FeatureTrack] = []

        # Previous frame cache
        self.prev_gray: Optional[np.ndarray] = None

    def process_frame(
        self,
        gray: np.ndarray,
        new_features: Optional[np.ndarray] = None,
    ) -> Dict[int, FeatureTrack]:
        """
        Process a new frame: track existing features and manage lifecycle.

        Args:
            gray:          Grayscale image (uint8, shape H×W).
            new_features:  Optional Nx2 array of new FAST keypoints to initiate.

        Returns:
            Dict of currently active tracks {track_id: FeatureTrack}.
        """
        if self.prev_gray is None:
            # First frame: initialize tracks from features
            self.prev_gray = gray.copy()
            if new_features is not None:
                self._initiate_tracks(new_features)
            return self.active_tracks

        # --- Track existing features via KLT ---
        if self.active_tracks:
            prev_points = np.array(
                [t.last_point() for t in self.active_tracks.values()],
                dtype=np.float32,
            ).reshape(-1, 1, 2)

            next_points, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, prev_points, None, **self.lk_params)

            # Backward check for robustness
            back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
                gray, self.prev_gray, next_points, None, **self.lk_params
            )

            # Filter by forward-backward consistency
            fb_error = np.linalg.norm(
                prev_points.reshape(-1, 2) - back_points.reshape(-1, 2),
                axis=1,
            )
            fb_valid = fb_error < 1.0  # pixels

            status = status.flatten().astype(bool) & back_status.flatten().astype(bool) & fb_valid

            # Update tracks
            track_ids = list(self.active_tracks.keys())
            tracks_to_remove = []

            for i, track_id in enumerate(track_ids):
                if status[i]:
                    self.active_tracks[track_id].add_observation(next_points[i].flatten())
                    # Retire old tracks
                    if self.active_tracks[track_id].age >= self.max_age:
                        self.active_tracks[track_id].is_active = False
                        self.lost_tracks.append(self.active_tracks[track_id])
                        tracks_to_remove.append(track_id)
                else:
                    # Track lost
                    self.active_tracks[track_id].is_active = False
                    self.lost_tracks.append(self.active_tracks[track_id])
                    tracks_to_remove.append(track_id)

            for tid in tracks_to_remove:
                del self.active_tracks[tid]

        # --- Outlier rejection via fundamental matrix RANSAC ---
        if len(self.active_tracks) >= 8:
            self._ransac_outlier_rejection()

        # --- Re-detect if below minimum ---
        if len(self.active_tracks) < self.min_tracks and new_features is not None:
            self._initiate_tracks(new_features)

        self.prev_gray = gray.copy()
        return self.active_tracks

    def _initiate_tracks(self, points: np.ndarray):
        """Create new tracks from detected feature points."""
        existing_points = set()
        for t in self.active_tracks.values():
            p = t.last_point()
            existing_points.add((int(p[0]), int(p[1])))

        count = 0
        for pt in points:
            if len(self.active_tracks) >= self.max_tracks:
                break
            key = (int(pt[0]), int(pt[1]))
            if key not in existing_points:
                track = FeatureTrack(pt)
                self.active_tracks[track.id] = track
                existing_points.add(key)
                count += 1

    def _ransac_outlier_rejection(self):
        """Remove outlier tracks using fundamental matrix RANSAC."""
        if len(self.active_tracks) < 8:
            return

        tracks_with_motion = [(tid, t) for tid, t in self.active_tracks.items() if t.age >= 2]

        if len(tracks_with_motion) < 8:
            return

        prev_pts = np.array(
            [t.observations[-2] for _, t in tracks_with_motion],
            dtype=np.float32,
        )
        curr_pts = np.array(
            [t.observations[-1] for _, t in tracks_with_motion],
            dtype=np.float32,
        )

        _, inlier_mask = cv2.findFundamentalMat(prev_pts, curr_pts, cv2.FM_RANSAC, self.ransac_thresh)

        if inlier_mask is not None:
            inlier_mask = inlier_mask.flatten().astype(bool)
            for i, (tid, track) in enumerate(tracks_with_motion):
                if not inlier_mask[i]:
                    track.is_active = False
                    self.lost_tracks.append(track)
                    if tid in self.active_tracks:
                        del self.active_tracks[tid]

    def get_mature_tracks(self) -> List[FeatureTrack]:
        """Return active tracks with sufficient observations for VIO update."""
        return [t for t in self.active_tracks.values() if t.age >= self.min_track_length]

    def pop_lost_tracks(self) -> List[FeatureTrack]:
        """Return and clear the list of recently lost tracks (for MSCKF update)."""
        lost = [t for t in self.lost_tracks if t.age >= self.min_track_length]
        self.lost_tracks.clear()
        return lost

    @property
    def num_active(self) -> int:
        return len(self.active_tracks)
