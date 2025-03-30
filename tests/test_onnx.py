import os

import torch
from models import create_model


def export_to_onnx(model: torch.nn.Module, dummy_input: torch.Tensor, save_path: str) -> bool:
    """
    Convert PyTorch model to ONNX format.
    """
    dynamic_axes = {
        name: {0: "batch_size", 2: "height", 3: "width"} for name in ["input", "output1", "output2", "output3"]
    }

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output1", "output2", "output3"],
        dynamic_axes=dynamic_axes,
    )


def get_file_size(file_path: str) -> str:
    """
    Get human-readable file size.
    """
    size_bytes = os.path.getsize(file_path)

    # Convert to appropriate unit
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def test_model_conversion(
    model_size: str, num_classes: int, input_size: tuple[int, int] = (640, 640), fp16: bool = False
) -> bool:
    """
    Test PyTorch to ONNX model conversion.
    """
    # Prepare model and input
    dtype = torch.float16 if fp16 else torch.float32
    dummy_input = torch.randn(1, 3, *input_size, dtype=dtype)
    model = create_model(model_size, num_classes)
    model.eval()

    # Prepare test results
    precision = "FP16" if fp16 else "FP32"
    model_params = sum(p.numel() for p in model.parameters())
    print(f"\nYOLO-{model_size} ({precision}):")
    print(f"✓ Model size: {model_params:,} parameters")

    # Try ONNX conversion
    onnx_path = f"test_{model_size}_{precision.lower()}.onnx"
    try:
        if fp16:
            with torch.autocast("cpu", dtype=torch.float16):
                export_to_onnx(model, dummy_input, onnx_path)
        else:
            export_to_onnx(model, dummy_input, onnx_path)

        file_size = get_file_size(onnx_path)
        print("✓ PyTorch → ONNX conversion")
        print(f"✓ ONNX file size: {file_size}")
        return True

    except Exception as e:
        print(f"✗ PyTorch → ONNX conversion: {str(e)}")
        return False

    finally:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)


def main():
    model_sizes = ["nano", "tiny", "small", "medium", "large", "xlarge"]
    num_classes = 80
    input_size = (640, 640)

    print("\nTesting YOLO model conversion:")
    print(f"- Input size: {input_size[0]}x{input_size[1]}")
    print(f"- Classes: {num_classes}")

    results = []
    for model_size in model_sizes:
        for fp16 in [False, True]:
            result = test_model_conversion(model_size, num_classes, input_size, fp16)
            results.append(result)

    if all(results):
        print("\n✓ All conversions successful!")
    else:
        print("\n✗ Some conversions failed!")


if __name__ == "__main__":
    main()
