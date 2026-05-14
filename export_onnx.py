# export_onnx.py

import argparse
import torch

from model import VesselNet


def load_model(model_path, device):
    model = VesselNet().to(device)

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pth model")
    parser.add_argument("--output", default="model.onnx", help="Output ONNX file")

    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--width", type=int, default=298)

    args = parser.parse_args()

    device = torch.device("cpu")  # ONNX export should be CPU

    print(f"Loading model from: {args.model}")
    model = load_model(args.model, device)

    # Dummy input (batch=1, channel=1, H, W)
    dummy_input = torch.randn(1, 1, args.height, args.width, device=device)

    print("Exporting to ONNX...")

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=["input"],
        output_names=["presence_logit", "column_logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "presence_logit": {0: "batch_size"},
            "column_logits": {0: "batch_size"},
        },
        opset_version=13,
    )

    print(f"✅ ONNX model saved to: {args.output}")


if __name__ == "__main__":
    main()