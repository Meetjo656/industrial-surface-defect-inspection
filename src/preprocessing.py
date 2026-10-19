"""Image preprocessing and dataset loader utilities for defect inspection."""

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from PIL import Image


def load_image(image_path: Path, target_size: Optional[Tuple[int, int]] = (256, 256)) -> np.ndarray:
    """Loads an image, optionally resizes, and normalizes it to [0, 1].

    Args:
        image_path: Path to the image file.
        target_size: Optional (width, height) tuple to resize.

    Returns:
        Normalized float32 numpy array of shape (H, W, C).
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        if target_size is not None:
            img = img.resize(target_size, Image.Resampling.BILINEAR)
        img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array


def load_mask(mask_path: Path, target_size: Optional[Tuple[int, int]] = (256, 256)) -> np.ndarray:
    """Loads a binary ground truth mask, resizes, and thresholds it.

    Args:
        mask_path: Path to the ground truth mask file.
        target_size: Optional (width, height) tuple to resize.

    Returns:
        Binary uint8 numpy array of shape (H, W) with values {0, 1}.
    """
    with Image.open(mask_path) as img:
        img = img.convert("L")
        if target_size is not None:
            img = img.resize(target_size, Image.Resampling.NEAREST)
        mask_array = (np.array(img, dtype=np.uint8) > 128).astype(np.uint8)
    return mask_array
