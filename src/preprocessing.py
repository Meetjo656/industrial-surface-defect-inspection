"""Industrial Surface Defect Inspection - Preprocessing Module.

Hybrid foreground extraction and aspect-ratio-preserving canvas normalization.
Preserves original image pixel data alongside a separate binary foreground mask.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np


def preprocess_image(
    image_input: Union[str, Path, np.ndarray],
    target_size: Tuple[int, int] = (256, 256),
    margin: float = 0.02,
    canvas_fill: Optional[Union[int, Tuple[int, int, int]]] = None,
    return_transform: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """Hybrid foreground extraction and aspect-ratio-preserving canvas normalization.

    Args:
        image_input: File path (str/Path) or existing BGR/RGB numpy array (H, W, 3).
        target_size: Desired (width, height) output canvas dimensions.
        margin: Fractional margin around detected foreground bounding box.
        canvas_fill: Canvas fill color (RGB tuple or int). If None, automatically
                     matches the estimated image background color.

    Returns:
        Tuple of (normalized_image, foreground_mask):
            - normalized_image: RGB uint8 array (canvas_h, canvas_w, 3) with original
                                pixel intensities preserved inside the centered crop.
            - foreground_mask: Binary uint8 array (canvas_h, canvas_w) with values {0, 255}.
    """
    # 1. Load image (RGB)
    if isinstance(image_input, (str, Path)):
        img_bgr = cv2.imread(str(image_input), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image at: {image_input}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image_input.copy()
        if img_rgb.ndim == 2:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)

    h, w = img_rgb.shape[:2]
    target_w, target_h = target_size

    # 2. Hybrid Foreground Extraction: Intensity Contrast + Edge Gradient
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Estimate background color from outer border
    border_rgb = np.concatenate(
        [img_rgb[0, :, :], img_rgb[-1, :, :], img_rgb[:, 0, :], img_rgb[:, -1, :]],
        axis=0,
    )
    auto_bg_color = np.median(border_rgb, axis=0).astype(np.uint8)

    border_gray = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    bg_val = int(np.median(border_gray))

    # Intensity cue: difference from estimated background boundary
    diff = cv2.absdiff(blurred, bg_val)
    _, thresh_diff = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Edge cue: Canny edges dilated to bridge gaps
    edges = cv2.Canny(blurred, 40, 120)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_edges = cv2.dilate(edges, edge_kernel, iterations=2)

    # Hybrid fusion + morphological closure
    hybrid = cv2.bitwise_or(thresh_diff, dilated_edges)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(hybrid, cv2.MORPH_CLOSE, close_kernel)

    # Extract dominant foreground contour
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fg_mask = np.zeros((h, w), dtype=np.uint8)

    min_area = 0.01 * (h * w)
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    if valid_contours:
        cv2.drawContours(fg_mask, valid_contours, -1, 255, thickness=-1)
        # Compute combined bounding box across valid foreground contours
        x_min = min(cv2.boundingRect(c)[0] for c in valid_contours)
        y_min = min(cv2.boundingRect(c)[1] for c in valid_contours)
        x_max = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in valid_contours)
        y_max = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in valid_contours)
        bw, bh = x_max - x_min, y_max - y_min
    else:
        # Fallback to full image (e.g. continuous texture / flat surface)
        fg_mask.fill(255)
        x_min, y_min, bw, bh = 0, 0, w, h

    # Add margin padding
    pad_x = int(bw * margin)
    pad_y = int(bh * margin)
    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)
    x2 = min(w, x_min + bw + pad_x)
    y2 = min(h, y_min + bh + pad_y)

    # 3. Aspect-Ratio-Preserving Crop on Fixed-Size Canvas
    cropped_img = img_rgb[y1:y2, x1:x2]
    cropped_mask = fg_mask[y1:y2, x1:x2]
    crop_h, crop_w = cropped_img.shape[:2]

    scale = min(target_w / max(crop_w, 1), target_h / max(crop_h, 1))
    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))

    interp_img = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized_img = cv2.resize(cropped_img, (new_w, new_h), interpolation=interp_img)
    resized_mask = cv2.resize(cropped_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Create canvas preserving background
    fill_color = auto_bg_color if canvas_fill is None else np.array(canvas_fill, dtype=np.uint8)
    canvas_img = np.full((target_h, target_w, 3), fill_color, dtype=np.uint8)
    canvas_mask = np.zeros((target_h, target_w), dtype=np.uint8)

    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2

    canvas_img[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized_img
    canvas_mask[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized_mask

    if return_transform:
        transform = {
            "crop_box": (x1, y1, x2, y2),
            "offset": (offset_x, offset_y),
            "scaled_size": (new_w, new_h),
            "orig_size": (h, w),
            "scale": scale,
        }
        return canvas_img, canvas_mask, transform

    return canvas_img, canvas_mask


def load_image(image_path: Path, target_size: Optional[Tuple[int, int]] = (256, 256)) -> np.ndarray:
    """Loads an image with aspect-ratio-preserving canvas normalization to [0, 1]."""
    if target_size is None:
        img_bgr = cv2.imread(str(image_path))
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    norm_img, _ = preprocess_image(image_path, target_size=target_size)
    return norm_img.astype(np.float32) / 255.0


def load_mask(
    mask_path: Path,
    target_size: Optional[Tuple[int, int]] = (256, 256),
    transform: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Loads a ground truth mask with aspect-ratio-preserving canvas normalization to {0, 1}."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {mask_path}")
    bin_mask = (mask > 128).astype(np.uint8)

    if transform is not None and target_size is not None:
        target_w, target_h = target_size
        canvas_mask = np.zeros((target_h, target_w), dtype=np.uint8)
        x1, y1, x2, y2 = transform["crop_box"]
        offset_x, offset_y = transform["offset"]
        new_w, new_h = transform["scaled_size"]
        cropped = bin_mask[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        canvas_mask[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized
        return canvas_mask

    if target_size is not None:
        bin_mask = cv2.resize(bin_mask, target_size, interpolation=cv2.INTER_NEAREST)
    return bin_mask


def run_visualization_test(
    normal_path: Optional[Path] = None,
    defective_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """Runs visualization test on normal and defective samples, saving a comparison figure."""
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = output_dir or (repo_root / "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-discover sample images if not specified
    if normal_path is None or not normal_path.exists():
        candidates = [
            repo_root / "dataset" / "bottle" / "test" / "good" / "000.png",
            repo_root / "data" / "metal_nut" / "train" / "good" / "000.png",
        ]
        normal_path = next((p for p in candidates if p.exists()), None)
        if normal_path is None:
            normal_path = next(repo_root.glob("**/good/*.png"), None)

    if defective_path is None or not defective_path.exists():
        candidates = [
            repo_root / "dataset" / "bottle" / "test" / "broken_large" / "000.png",
            repo_root / "data" / "metal_nut" / "test" / "scratch" / "000.png",
        ]
        defective_path = next((p for p in candidates if p.exists()), None)
        if defective_path is None:
            defective_path = next(
                (p for p in repo_root.glob("**/test/*/*.png") if "good" not in p.parts),
                None,
            )

    if not normal_path or not defective_path:
        raise FileNotFoundError("Could not find normal and defective sample images in dataset.")

    # Process samples
    norm_img_normal, mask_normal = preprocess_image(normal_path)
    norm_img_defect, mask_defect = preprocess_image(defective_path)

    orig_normal = cv2.cvtColor(cv2.imread(str(normal_path)), cv2.COLOR_BGR2RGB)
    orig_defect = cv2.cvtColor(cv2.imread(str(defective_path)), cv2.COLOR_BGR2RGB)

    # Plot 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    samples = [
        ("Normal Sample", orig_normal, norm_img_normal, mask_normal),
        ("Defective Sample", orig_defect, norm_img_defect, mask_defect),
    ]

    for row_idx, (title, orig, norm_img, norm_mask) in enumerate(samples):
        # Original Image
        axes[row_idx, 0].imshow(orig)
        axes[row_idx, 0].set_title(f"{title} (Original {orig.shape[1]}x{orig.shape[0]})", fontsize=10)
        axes[row_idx, 0].axis("off")

        # Normalized Canvas Image (original pixel values preserved)
        axes[row_idx, 1].imshow(norm_img)
        axes[row_idx, 1].set_title(f"Normalized Canvas ({norm_img.shape[1]}x{norm_img.shape[0]})", fontsize=10)
        axes[row_idx, 1].axis("off")

        # Separate Foreground Mask
        axes[row_idx, 2].imshow(norm_mask, cmap="gray")
        axes[row_idx, 2].set_title("Extracted FG Mask (Separate)", fontsize=10)
        axes[row_idx, 2].axis("off")

    plt.tight_layout()
    save_path = out_dir / "preprocessing_test.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[OK] Preprocessing visualization saved to: {save_path}")
    print(f"     Normal:    {normal_path}")
    print(f"     Defective: {defective_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Surface defect preprocessing & visualization test.")
    parser.add_argument("--normal", type=Path, default=None, help="Path to sample normal image")
    parser.add_argument("--defective", type=Path, default=None, help="Path to sample defective image")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save test visualization")
    args = parser.parse_args()

    run_visualization_test(
        normal_path=args.normal,
        defective_path=args.defective,
        output_dir=args.output_dir,
    )
