"""
TensorRT INT8 Entropy Calibrator for BiSeNetV2
File: weights/scripts/calibrate_int8.py
Reference: docs/WEIGHTS_AND_MODELS_GUIDE.md §3, Step 2
"""

import glob
import os

import cv2
import numpy as np

try:
    import pycuda.driver as cuda
    import tensorrt as trt

    class RELLISEntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, calib_images_dir: str, cache_file: str, batch_size: int = 8):
            super().__init__()
            self.cache_file = cache_file
            self.batch_size = batch_size
            self.image_files = glob.glob(os.path.join(calib_images_dir, "*.jpg"))[:200]
            self.current_index = 0

            # Allocate CUDA memory for batch: [B, 3, 448, 1024]
            self.batch_shape = (self.batch_size, 3, 448, 1024)
            self.device_input = cuda.mem_alloc(int(np.prod(self.batch_shape) * 4))

        def get_batch_size(self):
            return self.batch_size

        def get_batch(self, names):
            if self.current_index + self.batch_size > len(self.image_files):
                return None

            batch_imgs = []
            for i in range(self.current_index, self.current_index + self.batch_size):
                img = cv2.imread(self.image_files[i])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (1024, 448))
                img = img.astype(np.float32) / 255.0
                # Normalize with ImageNet mean/std
                img -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
                img /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
                img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
                batch_imgs.append(img)

            batch_data = np.ascontiguousarray(np.stack(batch_imgs, axis=0))
            cuda.memcpy_htod(self.device_input, batch_data)
            self.current_index += self.batch_size
            return [int(self.device_input)]

        def read_calibration_cache(self):
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            with open(self.cache_file, "wb") as f:
                f.write(cache)

except ImportError:
    # When running on host without TensorRT/PyCUDA installed
    pass
