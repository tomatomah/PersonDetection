import torch
from models import create_model


def test_model(model_type: str, num_classes: int, input_size: tuple[int, int] = (640, 640)) -> None:
    """
    Run yolox model test and display results.
    """
    print(f"\n{'='*20} Testing {model_type.upper()} Model {'='*20}")

    # Create input tensor
    x = torch.randn(1, 3, input_size[0], input_size[1])
    print(f"Input shape: {tuple(x.shape)}")

    # Create model and run inference
    model = create_model(model_type, num_classes)
    model.eval()

    with torch.no_grad():
        outputs = model(x)

    # Calculate and display parameter count
    params = sum(p.numel() for p in model.parameters())
    print("\nModel Statistics:")
    print(f"- Parameters: {params:,}")
    print(f"- Model type: yolox-{model_type}")
    print(f"- Number of classes: {num_classes}")

    # Display output shapes
    print("\nOutput shapes:")
    for i, output in enumerate(outputs, 1):
        print(f"- Feature map {i}: {tuple(output.shape)}")


def main():
    # Model types to test
    model_types = ["nano", "tiny", "small", "medium", "large", "xlarge"]
    num_classes = 80
    input_size = (640, 640)

    # Run tests for each model
    for model_type in model_types:
        test_model(model_type, num_classes, input_size)

    print("\nAll model tests completed!")


if __name__ == "__main__":
    main()
