import torch
from models import create_model


def test_model(model_size: str, num_classes: int, input_size: tuple[int, int] = (640, 640)) -> None:
    """
    Run yolo model test and display results.
    """
    print(f"\n{'='*20} Testing {model_size.upper()} Model {'='*20}")

    # Create input tensor
    x = torch.randn(1, 3, input_size[0], input_size[1])
    print(f"Input shape: {tuple(x.shape)}")

    # Create model and run inference
    model = create_model(model_size, num_classes)
    model.eval()

    with torch.no_grad():
        outputs = model(x)

    # Calculate and display parameter count
    params = sum(p.numel() for p in model.parameters())
    print("\nModel Statistics:")
    print(f"- Parameters: {params:,}")
    print(f"- Model size: yolo-{model_size}")
    print(f"- Number of classes: {num_classes}")

    # Display output shapes
    print("\nOutput shapes:")
    for i, output in enumerate(outputs, 1):
        print(f"- Feature map {i}: {tuple(output.shape)}")


def main():
    # Model sizes to test
    model_sizes = ["nano", "tiny", "small", "medium", "large", "xlarge"]
    num_classes = 80
    input_size = (640, 640)

    # Run tests for each model
    for model_size in model_sizes:
        test_model(model_size, num_classes, input_size)

    print("\nAll model tests completed!")


if __name__ == "__main__":
    main()
