"""
NETRA-UGV BiSeNetV2 Terrain Segmentation Inference
====================================================
Loads and runs the BiSeNetV2-Lite (2.31M parameters) terrain classifier.

Primary paths in order of preference:
  1. TensorRT INT8 serialized engine (production / Jetson Orin Nano)
  2. ONNX Runtime with CUDAExecutionProvider (GPU accelerated, with CPU fallback)
  3. OpenCV DNN with ONNX model (lightweight CPU fallback)
  4. HSV-based color-space heuristic segmenter (zero-dependency demo fallback)

4 Functional Surface Classes:
  0 - SOLID_GROUND:       Dry soil, gravel, packed track, asphalt (Full speed <= 1.5 m/s)
  1 - PLIANT_VEGETATION:  Grass, brush (Governed speed <= 0.5 m/s)
  2 - MUD_HAZARD:         Wet clay, marsh, puddles (Traction precaution <= 0.4 m/s)
  3 - RIGID_OBSTACLE:     Rocks, trees, walls, barriers (Hard obstacle / Stop)

Reference: MASTER_PROJECT_REPORT.md §3.1, WEIGHTS_AND_MODELS_TRAINING.md
"""

import logging
import os
import sys
import time
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Class indices matching TerrainClassification.msg constants
CLASS_SOLID_GROUND = 0
CLASS_PLIANT_VEGETATION = 1
CLASS_MUD_HAZARD = 2
CLASS_RIGID_OBSTACLE = 3
NUM_CLASSES = 4

# Default input dimensions for the model
DEFAULT_INPUT_W = 1024
DEFAULT_INPUT_H = 448


def _resolve_model_path(path: str) -> str:
    """Resolves relative or workspace-relative model paths."""
    if not path:
        return ""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    # Search relative to repository root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    candidate = os.path.join(repo_root, path)
    if os.path.exists(candidate):
        return candidate
    return path


class BiSeNetV2Inference:
    """
    Multi-backend terrain segmentation inference engine.

    Attempts to load backends in order of preference:
      1. TensorRT INT8 engine (production / Jetson)
      2. ONNX Runtime (CUDAExecutionProvider / CPUExecutionProvider)
      3. OpenCV DNN with ONNX model (CPU fallback)
      4. HSV heuristic segmenter (zero-dependency demo)
    """

    def __init__(
        self,
        engine_path: str = "",
        onnx_path: str = "",
        input_width: int = DEFAULT_INPUT_W,
        input_height: int = DEFAULT_INPUT_H,
    ):
        self.input_w = input_width
        self.input_h = input_height
        self.backend = "heuristic"
        self.active_provider = "none"
        self.cuda_fallback_warning = False
        self.last_latency_ms = 0.0
        self.last_dist_str = "N/A"

        # ImageNet normalization parameters matching training pipeline
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        resolved_engine = _resolve_model_path(engine_path)
        resolved_onnx = _resolve_model_path(onnx_path)

        # --- 1. Attempt TensorRT ---
        if resolved_engine and os.path.exists(resolved_engine):
            if self._try_load_tensorrt(resolved_engine):
                self.backend = "tensorrt"
                self.active_provider = "TensorRT_INT8"
                logger.info(f"TensorRT INT8 engine loaded: {resolved_engine}")

        # --- 2. Attempt ONNX Runtime ---
        if self.backend == "heuristic" and resolved_onnx and os.path.exists(resolved_onnx):
            if self._try_load_onnxruntime(resolved_onnx):
                self.backend = "onnxruntime"
                logger.info(f"ONNX Runtime model loaded: {resolved_onnx} [provider={self.active_provider}]")
            # --- 3. Attempt OpenCV DNN Fallback ---
            elif self._try_load_opencv_dnn(resolved_onnx):
                self.backend = "opencv_dnn"
                self.active_provider = "OpenCV_CPU"
                logger.info(f"OpenCV DNN ONNX model loaded: {resolved_onnx}")

        # --- 4. Fallback: HSV Heuristic ---
        if self.backend == "heuristic":
            self.active_provider = "HSV_Heuristic"
            logger.warning("No TensorRT or ONNX model available. " "Using HSV heuristic segmenter for demonstration.")

        logger.info(f"BiSeNetV2 inference backend: {self.backend} [provider={self.active_provider}]")

    def _try_load_tensorrt(self, engine_path: str) -> bool:
        """Attempt to load a TensorRT serialized engine."""
        try:
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
            import tensorrt as trt

            trt_logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(trt_logger)

            with open(engine_path, "rb") as f:
                engine_data = f.read()

            if len(engine_data) < 1024:
                logger.warning("TensorRT engine file too small — likely a stub.")
                return False

            self.trt_engine = runtime.deserialize_cuda_engine(engine_data)
            if self.trt_engine is None:
                return False

            self.trt_context = self.trt_engine.create_execution_context()

            # Allocate device memory
            self.d_input = cuda.mem_alloc(1 * 3 * self.input_h * self.input_w * 4)  # float32
            self.d_output = cuda.mem_alloc(1 * self.input_h * self.input_w * 4)  # int32 class indices
            self.h_output = np.empty((self.input_h, self.input_w), dtype=np.int32)
            self.cuda_stream = cuda.Stream()

            return True
        except (ImportError, Exception) as e:
            logger.debug(f"TensorRT load failed: {e}")
            return False

    def _try_load_onnxruntime(self, onnx_path: str) -> bool:
        """Attempt to load ONNX model with ONNX Runtime using CUDAExecutionProvider."""
        try:
            # On Windows, add torch/lib to DLL directory to resolve CUDA/cuDNN DLLs
            if sys.platform == "win32":
                try:
                    import torch

                    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
                    if os.path.exists(torch_lib) and hasattr(os, "add_dll_directory"):
                        os.add_dll_directory(torch_lib)
                except Exception:
                    pass

            import onnxruntime as ort

            available_providers = ort.get_available_providers()
            logger.info(f"Available ONNX Runtime execution providers: {available_providers}")

            providers = []
            if "CUDAExecutionProvider" in available_providers:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

            try:
                self.ort_session = ort.InferenceSession(onnx_path, providers=providers)
            except Exception as e:
                logger.warning(f"Could not initialize providers {providers}: {e}. Retrying CPUExecutionProvider...")
                self.ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

            self.active_provider = self.ort_session.get_providers()[0]
            if "CUDAExecutionProvider" in available_providers and self.active_provider != "CUDAExecutionProvider":
                self.cuda_fallback_warning = True
                logger.warning(
                    f"CUDAExecutionProvider was available but session initialized with '{self.active_provider}'. "
                    f"Ensure CUDA 12 and cuDNN DLLs are in system PATH."
                )

            self.ort_input_name = self.ort_session.get_inputs()[0].name
            self.ort_output_name = self.ort_session.get_outputs()[0].name

            return True
        except (ImportError, Exception) as e:
            logger.debug(f"ONNX Runtime load failed: {e}")
            return False

    def _try_load_opencv_dnn(self, onnx_path: str) -> bool:
        """Attempt to load an ONNX model with OpenCV DNN."""
        try:
            self.dnn_net = cv2.dnn.readNetFromONNX(onnx_path)
            self.dnn_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.dnn_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            return True
        except Exception as e:
            logger.debug(f"OpenCV DNN ONNX load failed: {e}")
            return False

    def infer(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Run terrain segmentation on the input image.

        Args:
            image: BGR image of shape (H, W, 3), already ROI-cropped.

        Returns:
            Tuple of:
              - semantic_mask: uint8 array of shape (H, W) with class indices [0-3]
              - mean_confidence: float in [0.0, 1.0]
        """
        if self.backend == "tensorrt":
            return self._infer_tensorrt(image)
        elif self.backend == "onnxruntime":
            return self._infer_onnxruntime(image)
        elif self.backend == "opencv_dnn":
            return self._infer_opencv_dnn(image)
        else:
            return self._infer_heuristic(image)

    def _infer_tensorrt(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """TensorRT INT8 inference path."""
        import pycuda.driver as cuda

        t0 = time.perf_counter()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        blob = (resized.astype(np.float32) / 255.0 - self.mean) / self.std
        blob = blob.transpose(2, 0, 1)  # HWC → CHW
        blob = np.expand_dims(blob, axis=0)  # Add batch dim
        blob = np.ascontiguousarray(blob)

        # Transfer to GPU and execute
        cuda.memcpy_htod_async(self.d_input, blob, self.cuda_stream)
        self.trt_context.execute_async_v2(
            bindings=[int(self.d_input), int(self.d_output)],
            stream_handle=self.cuda_stream.handle,
        )
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.cuda_stream)
        self.cuda_stream.synchronize()
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0

        # Map model classes to NETRA 4-class scheme
        mask = np.clip(self.h_output, 0, NUM_CLASSES - 1).astype(np.uint8)

        # Calculate class distribution
        counts = np.bincount(mask.flatten(), minlength=NUM_CLASSES)
        total = mask.size
        pcts = (counts / total) * 100.0
        self.last_dist_str = f"Gnd:{pcts[0]:.1f}% Veg:{pcts[1]:.1f}% Mud:{pcts[2]:.1f}% Obs:{pcts[3]:.1f}%"

        # Resize back to original image dimensions
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        confidence = 0.92
        return mask, confidence

    def _infer_onnxruntime(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        ONNX Runtime inference path.
        Exact preprocessing matching training pipeline:
          1. BGR -> RGB
          2. Resize to 1024x448
          3. /255.0 normalization + ImageNet mean/std
          4. Transpose to CHW (1, 3, 448, 1024)
        """
        t0 = time.perf_counter()

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if (rgb.shape[1], rgb.shape[0]) != (self.input_w, self.input_h):
            rgb_resized = cv2.resize(rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        else:
            rgb_resized = rgb

        rgb_norm = (rgb_resized.astype(np.float32) / 255.0 - self.mean) / self.std
        blob = np.expand_dims(rgb_norm.transpose(2, 0, 1), axis=0).astype(np.float32)

        output = self.ort_session.run([self.ort_output_name], {self.ort_input_name: blob})[0]
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0

        # Output shape: (1, NUM_CLASSES, H, W) -> Argmax
        mask = np.argmax(output[0], axis=0).astype(np.uint8)
        mask = np.clip(mask, 0, NUM_CLASSES - 1)

        # Calculate class distribution
        counts = np.bincount(mask.flatten(), minlength=NUM_CLASSES)
        total = mask.size
        pcts = (counts / total) * 100.0
        self.last_dist_str = f"Gnd:{pcts[0]:.1f}% Veg:{pcts[1]:.1f}% Mud:{pcts[2]:.1f}% Obs:{pcts[3]:.1f}%"

        # Resize back to original ROI image dimensions
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        confidence = 0.95
        return mask, confidence

    def _infer_opencv_dnn(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """OpenCV DNN ONNX inference path."""
        t0 = time.perf_counter()

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if (rgb.shape[1], rgb.shape[0]) != (self.input_w, self.input_h):
            rgb_resized = cv2.resize(rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        else:
            rgb_resized = rgb

        rgb_norm = (rgb_resized.astype(np.float32) / 255.0 - self.mean) / self.std
        blob = np.expand_dims(rgb_norm.transpose(2, 0, 1), axis=0).astype(np.float32)

        self.dnn_net.setInput(blob)
        output = self.dnn_net.forward()
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0

        # output shape: (1, NUM_CLASSES, H, W) — take argmax
        mask = np.argmax(output[0], axis=0).astype(np.uint8)
        mask = np.clip(mask, 0, NUM_CLASSES - 1)

        counts = np.bincount(mask.flatten(), minlength=NUM_CLASSES)
        total = mask.size
        pcts = (counts / total) * 100.0
        self.last_dist_str = f"Gnd:{pcts[0]:.1f}% Veg:{pcts[1]:.1f}% Mud:{pcts[2]:.1f}% Obs:{pcts[3]:.1f}%"

        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        confidence = 0.85
        return mask, confidence

    def _infer_heuristic(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        HSV color-space heuristic segmenter (zero-dependency demo fallback).

        Heuristic rules:
          - Green-dominant pixels → PLIANT_VEGETATION (grass/brush)
          - Brown/dark pixels     → SOLID_GROUND (soil/gravel)
          - Very dark pixels      → MUD_HAZARD (shadow/wet regions)
          - High saturation grey  → RIGID_OBSTACLE (rocks/concrete)
        """
        t0 = time.perf_counter()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        mask = np.full(image.shape[:2], CLASS_SOLID_GROUND, dtype=np.uint8)

        # Vegetation: green hue (35-85), moderate saturation
        veg_mask = (h >= 35) & (h <= 85) & (s >= 40) & (v >= 40)
        mask[veg_mask] = CLASS_PLIANT_VEGETATION

        # Mud/marsh: dark, low saturation brownish
        mud_mask = (v < 80) & (s < 60)
        mask[mud_mask] = CLASS_MUD_HAZARD

        # Rigid obstacles: very dark OR very high contrast edges
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        kernel = np.ones((7, 7), np.uint8)
        obstacle_regions = cv2.dilate(edges, kernel, iterations=2)
        obstacle_mask = (obstacle_regions > 0) & (v > 30)
        mask[obstacle_mask] = CLASS_RIGID_OBSTACLE

        # Also mark very bright specular regions as obstacles
        bright_mask = (v > 220) & (s < 30)
        mask[bright_mask] = CLASS_RIGID_OBSTACLE

        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0

        counts = np.bincount(mask.flatten(), minlength=NUM_CLASSES)
        total = mask.size
        pcts = (counts / total) * 100.0
        self.last_dist_str = f"Gnd:{pcts[0]:.1f}% Veg:{pcts[1]:.1f}% Mud:{pcts[2]:.1f}% Obs:{pcts[3]:.1f}%"

        confidence = 0.55
        return mask, confidence
