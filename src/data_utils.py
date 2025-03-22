import sys

from PIL import Image
import numpy as np
import torch


def imread(image_path: str) -> Image.Image:
    """
    Reads an image file and returns it as a Pillow Image object.
    """
    try:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        if image.width == 0 or image.height == 0:
            raise ValueError(f"Failed to load image: {image_path}")

        return image

    except Exception as e:
        print(e)
        sys.exit(1)


def cxcywh2xyxy(x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """
    Convert batches of bounding boxes from center format (cx, cy, width, height) to
    corner format (xmin, ymin, xmax, ymax).
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - (x[:, 2] / 2)  # top left x
    y[:, 1] = x[:, 1] - (x[:, 3] / 2)  # top left y
    y[:, 2] = x[:, 0] + (x[:, 2] / 2)  # bottom right x
    y[:, 3] = x[:, 1] + (x[:, 3] / 2)  # bottom right y
    return y


def xyxy2cxcywh(x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """
    Convert batches of bounding boxes from corner format (xmin, ymin, xmax, ymax) to
    center format (cx, cy, width, height).
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # center x
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # center y
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def compute_iou(box1: np.ndarray, box2: np.ndarray, eps=1e-7):
    """
    Compute Complete IoU (CIoU) between two sets of bounding boxes using NumPy.
    """
    # Unpack box coordinates
    b1_xmin, b1_ymin, b1_xmax, b1_ymax = np.split(box1, 4, axis=-1)
    b2_xmin, b2_ymin, b2_xmax, b2_ymax = np.split(box2, 4, axis=-1)

    # Calculate box dimensions with numerical stability
    w1, h1 = b1_xmax - b1_xmin + eps, b1_ymax - b1_ymin + eps
    w2, h2 = b2_xmax - b2_xmin + eps, b2_ymax - b2_ymin + eps

    # Compute intersection area
    intersection_xmin = np.maximum(b1_xmin, b2_xmin)
    intersection_ymin = np.maximum(b1_ymin, b2_ymin)
    intersection_xmax = np.minimum(b1_xmax, b2_xmax)
    intersection_ymax = np.minimum(b1_ymax, b2_ymax)

    intersection_width = np.maximum(0, intersection_xmax - intersection_xmin)
    intersection_height = np.maximum(0, intersection_ymax - intersection_ymin)
    intersection_area = intersection_width * intersection_height

    # Compute union area
    union_area = w1 * h1 + w2 * h2 - intersection_area + eps

    # Calculate IoU
    iou = intersection_area / union_area

    # Calculate enclosing box dimensions
    cw = np.maximum(b1_xmax, b2_xmax) - np.minimum(b1_xmin, b2_xmin)
    ch = np.maximum(b1_ymax, b2_ymax) - np.minimum(b1_ymin, b2_ymin)

    # Compute squared diagonal of enclosing box
    c2 = cw**2 + ch**2 + eps

    # Calculate box centers
    b1_center_x = (b1_xmin + b1_xmax) / 2
    b1_center_y = (b1_ymin + b1_ymax) / 2
    b2_center_x = (b2_xmin + b2_xmax) / 2
    b2_center_y = (b2_ymin + b2_ymax) / 2

    # Compute squared center point distance
    rho2 = (b2_center_x - b1_center_x) ** 2 + (b2_center_y - b1_center_y) ** 2

    # Calculate aspect ratio difference term
    v = (4 / (np.pi**2)) * (np.arctan(w2 / h2) - np.arctan(w1 / h1)) ** 2

    # Compute alpha term for balancing
    alpha = v / (v - iou + (1 + eps))

    # Calculate final CIoU
    ciou = iou - (rho2 / c2 + v * alpha)

    return ciou[:, 0]
