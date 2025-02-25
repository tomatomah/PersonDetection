import sys

import cv2
import numpy as np
import torch


def imread(image_path: str) -> np.ndarray:
    """
    Reads an image file and returns it as an RGB array.
    """
    try:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Failed to load image: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_rgb

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
