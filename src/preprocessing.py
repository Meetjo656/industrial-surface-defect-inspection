"""Image Preprocessing Module.

Extracts foreground region and centers surface image onto a standardized 256x256 canvas.
Supports both object and full-frame texture categories across MVTec AD.
"""

from pathlib import Path
from typing import Any, Dict, Tuple, Union
import cv2
import numpy as np

TEXTURE_CATEGORIES = {"carpet", "grid", "leather", "tile", "wood"}


def preprocess_image(
    image_input: Union[str, Path, np.ndarray],
    target_size: Tuple[int, int] = (256, 256),
    margin: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Crops foreground and normalizes image to target canvas size.

    Returns:
        (canvas_img, canvas_mask, transform):
            canvas_img: RGB array of shape (H, W, 3).
            canvas_mask: Binary uint8 array of shape (H, W).
            transform: Coordinate mapping dict to project back to original size.
    """
    if isinstance(image_input, (str, Path)):
        bgr = cv2.imread(str(image_input))
        if bgr is None:
            raise FileNotFoundError(f"Image not found: {image_input}")
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        path_str = str(image_input).lower()
    else:
        img = image_input.copy()
        path_str = ""

    orig_h, orig_w = img.shape[:2]
    tw, th = target_size

    # 1. Texture categories are full-frame surfaces (no background)
    is_texture = any(t in path_str for t in TEXTURE_CATEGORIES)
    if is_texture:
        canvas_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        canvas_mask = np.full((th, tw), 255, dtype=np.uint8)
        transform = {
            "orig_size": (orig_h, orig_w),
            "crop_box": (0, 0, orig_w, orig_h),
            "scaled_size": target_size,
            "offset": (0, 0),
        }
        return canvas_img, canvas_mask, transform

    # 2. Object categories: foreground thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if background is bright and object is dark
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    if border.mean() > 128:
        binary = cv2.bitwise_not(binary)

    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad_x = int(w * margin)
        pad_y = int(h * margin)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(orig_w, x + w + pad_x)
        y2 = min(orig_h, y + h + pad_y)
    else:
        x1, y1, x2, y2 = 0, 0, orig_w, orig_h

    crop_img = img[y1:y2, x1:x2]
    crop_mask = binary[y1:y2, x1:x2]

    # Aspect-ratio preserving resize onto target canvas
    scale = min(tw / max(1, x2 - x1), th / max(1, y2 - y1))
    nw, nh = max(1, int((x2 - x1) * scale)), max(1, int((y2 - y1) * scale))

    scaled_img = cv2.resize(crop_img, (nw, nh), interpolation=cv2.INTER_AREA)
    scaled_mask = cv2.resize(crop_mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

    ox, oy = (tw - nw) // 2, (th - nh) // 2
    canvas_img = np.zeros((th, tw, 3), dtype=np.uint8)
    canvas_mask = np.zeros((th, tw), dtype=np.uint8)

    canvas_img[oy:oy + nh, ox:ox + nw] = scaled_img
    canvas_mask[oy:oy + nh, ox:ox + nw] = scaled_mask

    transform = {
        "orig_size": (orig_h, orig_w),
        "crop_box": (x1, y1, x2, y2),
        "scaled_size": (nw, nh),
        "offset": (ox, oy),
    }
    return canvas_img, canvas_mask, transform
