# PersonDetection

PersonDetection is a project for detecting people in images and videos. This project uses the YOLOX object detection algorithm to achieve accurate and efficient detection in various environments. Using a large dataset of people, it predicts two classes: "head" and "person". Below are demo videos showing actual detection of "head" and "person" from videos. Both videos use the same deep learning model (nano model) for inference.

## Demo

https://github.com/user-attachments/assets/8466092e-7909-44be-a723-ed8579d34971

https://github.com/user-attachments/assets/1a5a6829-9850-4b32-89a7-6692df489b4e

## Features

- People detection in images or videos
- Dual-class classification (full body and head)
- Fast inference suitable for real-time applications (designed for CPU inference)

## Development Environment

This project's development environment uses Ubuntu 22.04.4 LTS. The Python version is 3.10.12, and the CUDA version is 12.1. The hardware specifications are as follows:

| Hardware |
|-------------|
| CPU: Intel Core i5-12600K (3.7GHz, 10 cores) |
| GPU: NVIDIA RTX 3060 (12GB) |

## Dataset
For training, we used the MSCOCO dataset and our proprietary dataset. It's important to note that we extended the MSCOCO dataset and added our own data, relabeling all images to suit our specific needs. We do not plan to publish these two datasets. The total number of training data and instances is as follows:
```
- Training data
  - Total: Approximately 320,000 images
  - Total: Approximately 1.7 million instances
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PersonDetection.git
cd PersonDetection

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Model Download

Pre-trained nano-sized ONNX models are available for release. The ONNX models come in four varieties: FP32 models with and without post-processing, and FP16 models with and without post-processing.

```bash
# 1. Download the ONNX model files from the GitHub release page
# 2. Create the output directory if it doesn't exist
mkdir -p outputs/onnx

# 3. Place the downloaded model files in the outputs/onnx/ directory
# Example:
# outputs/onnx/model_with_postprocessing.onnx
# outputs/onnx/model_without_postprocessing.onnx
# outputs/onnx/model_fp16_with_postprocessing.onnx
# outputs/onnx/model_fp16_without_postprocessing.onnx
```

### Basic Usage

The project settings are managed in the `config/config.yaml` file.

```bash
# Run detection
python src/detect.py

# Detection results are saved by default to ./outputs/detect/
# - Image detection results: ./outputs/detect/images/
# - Video detection results: ./outputs/detect/movie/
```

### Configuration Options

You can customize the following settings in the `config/config.yaml` file:

```yaml
detecting:
  # Detection class and display color settings
  id_to_class: {0: "head", 1: "person"}
  plot_colors: [[255, 128, 0],   # Color for head (orange)
                [0, 127, 255]]   # Color for person (blue)
  # Change input source
  input_type: "image"  # "image" or "movie"
  image_list_file: "./image_list.txt"  # List of image paths
  movie_path: "./assets/demo_video1.mp4"  # Video path

  # model file
  model_file: "model_with_postprocessing.onnx"

  # Adjust detection parameters
  detect_threshold: 0.25  # Detection threshold (0.0-1.0)
  iou_threshold: 0.35    # IoU threshold for NMS
  scale_list: [1.0]      # For multi-scale detection (e.g., [0.8, 1.0, 1.2])

  # Output directory for detection results
  save_dir: "./outputs"  # Directory to save results
```

### Processing Speed

- CPU: Intel Core i5-12600K (3.7GHz, 10 cores): Approximately 30FPS

*Note: Measured with image size 640×480, batch size 1*  
*Note: Inference performed on CPU (GPU is not used for inference)*

## How to Contribute

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## References
  - https://cocodataset.org/
  - https://github.com/Megvii-BaseDetection/YOLO
  - Video by Coverr from Pexels: https://www.pexels.com/video/black-and-white-video-of-people-853889/
  - Video by George Morina from Pexels: https://www.pexels.com/video/people-walking-by-bond-street-station-entry-gate-5325136/

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.