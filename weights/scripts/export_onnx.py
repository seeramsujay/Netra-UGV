"""
BiSeNetV2 PyTorch to ONNX Exporter for NETRA-UGV
File: weights/scripts/export_onnx.py
Reference: docs/WEIGHTS_AND_MODELS_GUIDE.md §3, Step 1
"""

import os
import sys

import torch

# Ensure bisenetv2_model can be imported
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
perception_pkg = os.path.join(repo_root, "src", "netra_perception", "netra_perception")
if perception_pkg not in sys.path:
    sys.path.insert(0, perception_pkg)

from bisenetv2_model import BiSeNetV2


def export_onnx(
    weights_path: str = "../bisenetv2_rellis_best.pth",
    output_onnx_path: str = "../bisenetv2_rellis.onnx",
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_weights = os.path.abspath(os.path.join(script_dir, weights_path))
    abs_output = os.path.abspath(os.path.join(script_dir, output_onnx_path))

    print(f"Loading checkpoint from: {abs_weights}")
    model = BiSeNetV2(num_classes=4, use_aux=False)
    checkpoint = torch.load(abs_weights, map_location="cpu")
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    filtered = {k: v for k, v in state_dict.items() if not k.startswith("aux")}
    model.load_state_dict(filtered, strict=False)
    model.eval()

    # Fixed input shape: [Batch=1, Channels=3, Height=448, Width=1024]
    dummy_input = torch.randn(1, 3, 448, 1024, dtype=torch.float32)

    print(f"Exporting to ONNX: {abs_output}")
    torch.onnx.export(
        model,
        dummy_input,
        abs_output,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_rgb"],
        output_names=["output_mask"],
    )
    print("[OK] ONNX export complete.")


if __name__ == "__main__":
    w_path = sys.argv[1] if len(sys.argv) > 1 else "../bisenetv2_rellis_best.pth"
    o_path = sys.argv[2] if len(sys.argv) > 2 else "../bisenetv2_rellis.onnx"
    export_onnx(w_path, o_path)
