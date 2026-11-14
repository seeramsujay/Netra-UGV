"""
Generate a valid placeholder stub with magic header for CPU / non-Jetson demonstration.
File: weights/scripts/generate_stub.py
Reference: docs/WEIGHTS_AND_MODELS_GUIDE.md §4
"""

import os
import sys


def generate_stub(output_path: str = "../bisenetv2_rellis_int8.trt"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_out = os.path.abspath(os.path.join(script_dir, output_path))
    os.makedirs(os.path.dirname(abs_out), exist_ok=True)

    # Header signature: TRT + INT8 + Model ID
    header = b"NETRA_TRT_INT8_STUB_v1.0"
    payload = header.ljust(64, b"\x00")
    with open(abs_out, "wb") as f:
        f.write(payload)
    print(f"[OK] Created 64-byte TensorRT placeholder stub at: {abs_out}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "../bisenetv2_rellis_int8.trt"
    generate_stub(target)
