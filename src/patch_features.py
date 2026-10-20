"""Patch Feature Extraction and Inspection Module.

Extracts local HSV color statistics and LBP texture features from fixed-size
patches on normalized images, recording spatial positions and filtering
background patches using the foreground mask.

NOTE: Defective test images are strictly for visualization and evaluation,
never for fitting the normal reference bank or tuning model parameters.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import local_binary_pattern

import sys

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from src.preprocessing import preprocess_image
except ImportError:
    from preprocessing import preprocess_image


@dataclass
class PatchRecord:
    """Stores a single patch's spatial metadata and extracted feature vector."""
    y: int
    x: int
    size: int
    center_norm: Tuple[float, float]  # (y_norm, x_norm) in [0, 1]
    grid_pos: Tuple[int, int]         # (grid_row, grid_col)
    fg_coverage: float               # Foreground pixel ratio [0, 1]
    features: np.ndarray             # 16-dim vector: 6 HSV stats + 10 LBP hist


def extract_single_patch_features(
    patch_rgb: np.ndarray,
    lbp_points: int = 8,
    lbp_radius: int = 1,
) -> np.ndarray:
    """Extracts local HSV color statistics and LBP texture features from one patch.

    Args:
        patch_rgb: Patch RGB array (P, P, 3).
        lbp_points: Number of circular points for LBP.
        lbp_radius: Radius for LBP.

    Returns:
        16-dimensional feature vector:
            - Indices 0..5:  H_mean, H_std, S_mean, S_std, V_mean, V_std (scaled [0, 1])
            - Indices 6..15: 10-bin normalized uniform LBP histogram
    """
    # 1. Local HSV Color Statistics
    hsv = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    h_norm = h / 180.0
    s_norm = s / 255.0
    v_norm = v / 255.0

    hsv_stats = [
        float(np.mean(h_norm)), float(np.std(h_norm)),
        float(np.mean(s_norm)), float(np.std(s_norm)),
        float(np.mean(v_norm)), float(np.std(v_norm)),
    ]

    # 2. Local LBP Texture Features
    gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, lbp_points, lbp_radius, method="uniform")
    n_bins = lbp_points + 2
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    return np.concatenate([hsv_stats, lbp_hist]).astype(np.float32)


def extract_image_patches(
    image_rgb: np.ndarray,
    fg_mask: np.ndarray,
    patch_size: int = 16,
    stride: Optional[int] = None,
    min_fg_ratio: float = 0.5,
) -> List[PatchRecord]:
    """Divides normalized image into patches, filters background, and extracts features.

    Args:
        image_rgb: Normalized RGB canvas image (H, W, 3).
        fg_mask: Corresponding foreground mask (H, W) with values {0, 255}.
        patch_size: Patch height and width in pixels.
        stride: Step size between patch origins. Defaults to patch_size (no overlap).
        min_fg_ratio: Minimum foreground fraction to retain a patch.

    Returns:
        List of PatchRecord objects for foreground-valid patches.
    """
    h, w = image_rgb.shape[:2]
    s = stride if stride is not None else patch_size
    records: List[PatchRecord] = []

    for y in range(0, h - patch_size + 1, s):
        grid_r = y // s
        cy_norm = (y + patch_size / 2.0) / h
        for x in range(0, w - patch_size + 1, s):
            grid_c = x // s
            cx_norm = (x + patch_size / 2.0) / w

            # Background policy: ignore patches predominantly outside foreground
            patch_mask = fg_mask[y:y + patch_size, x:x + patch_size]
            fg_cov = float(np.mean(patch_mask > 0))
            if fg_cov < min_fg_ratio:
                continue

            patch_rgb = image_rgb[y:y + patch_size, x:x + patch_size]
            feat = extract_single_patch_features(patch_rgb)

            records.append(
                PatchRecord(
                    y=y,
                    x=x,
                    size=patch_size,
                    center_norm=(cy_norm, cx_norm),
                    grid_pos=(grid_r, grid_c),
                    fg_coverage=fg_cov,
                    features=feat,
                )
            )

    return records


def visualize_patch_grid_and_features(
    normal_path: Path,
    defective_path: Path,
    output_path: Path,
    patch_size: int = 16,
    min_fg_ratio: float = 0.5,
) -> Path:
    """Visualizes patch grids and feature vector comparisons between normal and defective samples.

    RULE: Defective test sample is purely for inspection/visualization, not model fitting.
    """
    # 1. Load normalized images and masks
    norm_img_normal, mask_normal = preprocess_image(normal_path)
    norm_img_defect, mask_defect = preprocess_image(defective_path)

    # 2. Extract patch records
    normal_patches = extract_image_patches(
        norm_img_normal, mask_normal, patch_size=patch_size, min_fg_ratio=min_fg_ratio
    )
    defect_patches = extract_image_patches(
        norm_img_defect, mask_defect, patch_size=patch_size, min_fg_ratio=min_fg_ratio
    )

    # 3. Create grid overlay images
    overlay_normal = norm_img_normal.copy()
    for p in normal_patches:
        cv2.rectangle(
            overlay_normal,
            (p.x, p.y),
            (p.x + p.size, p.y + p.size),
            (0, 255, 0),
            1,
        )

    overlay_defect = norm_img_defect.copy()
    for p in defect_patches:
        cv2.rectangle(
            overlay_defect,
            (p.x, p.y),
            (p.x + p.size, p.y + p.size),
            (255, 140, 0),
            1,
        )

    # 4. Find sample patches to compare features (pick matching spatial coordinate if possible)
    defect_positions = {(p.grid_pos): p for p in defect_patches}
    normal_positions = {(p.grid_pos): p for p in normal_patches}
    common_pos = set(defect_positions.keys()).intersection(set(normal_positions.keys()))

    if common_pos:
        # Select patch with highest feature divergence
        best_pos = max(
            common_pos,
            key=lambda pos: np.linalg.norm(
                defect_positions[pos].features - normal_positions[pos].features
            ),
        )
        sample_norm_patch = normal_positions[best_pos]
        sample_def_patch = defect_positions[best_pos]
    else:
        sample_norm_patch = normal_patches[0]
        sample_def_patch = defect_patches[0]

    # Highlight inspected patches
    cv2.rectangle(
        overlay_normal,
        (sample_norm_patch.x, sample_norm_patch.y),
        (sample_norm_patch.x + sample_norm_patch.size, sample_norm_patch.y + sample_norm_patch.size),
        (255, 0, 0),
        2,
    )
    cv2.rectangle(
        overlay_defect,
        (sample_def_patch.x, sample_def_patch.y),
        (sample_def_patch.x + sample_def_patch.size, sample_def_patch.y + sample_def_patch.size),
        (255, 0, 0),
        2,
    )

    # 5. Plot visualization
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)

    # Normal Grid Overlay
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(overlay_normal)
    ax1.set_title(f"Normal Sample Grid ({len(normal_patches)} Valid Patches)", fontsize=11)
    ax1.axis("off")

    # Defective Grid Overlay
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(overlay_defect)
    ax2.set_title(f"Defective Sample Grid ({len(defect_patches)} Valid Patches)", fontsize=11)
    ax2.axis("off")

    # Foreground Masks with Grid Coverage
    ax3 = fig.add_subplot(gs[0, 1])
    ax3.imshow(mask_normal, cmap="gray")
    ax3.set_title("Normal Foreground Mask", fontsize=11)
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(mask_defect, cmap="gray")
    ax4.set_title("Defective Foreground Mask", fontsize=11)
    ax4.axis("off")

    # Feature Vector Inspection Comparison
    feature_labels = [
        "H_mean", "H_std", "S_mean", "S_std", "V_mean", "V_std",
        "LBP_0", "LBP_1", "LBP_2", "LBP_3", "LBP_4",
        "LBP_5", "LBP_6", "LBP_7", "LBP_8", "LBP_9"
    ]
    x_indices = np.arange(len(feature_labels))
    width = 0.38

    ax5 = fig.add_subplot(gs[:, 2])
    ax5.barh(
        x_indices - width / 2,
        sample_norm_patch.features,
        height=width,
        label="Normal Patch",
        color="#2ca02c",
        alpha=0.85,
    )
    ax5.barh(
        x_indices + width / 2,
        sample_def_patch.features,
        height=width,
        label="Defective Patch",
        color="#d62728",
        alpha=0.85,
    )
    ax5.set_yticks(x_indices)
    ax5.set_yticklabels(feature_labels, fontsize=9)
    ax5.invert_yaxis()
    ax5.set_xlabel("Normalized Feature Value")
    ax5.set_title(
        f"Feature Vector Inspection\nGrid Pos {sample_def_patch.grid_pos} (Blue Box)",
        fontsize=11,
    )
    ax5.legend(loc="lower right")
    ax5.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[OK] Patch inspection figure saved to: {output_path}")
    print(f"     Normal patches retained:    {len(normal_patches)}")
    print(f"     Defective patches retained: {len(defect_patches)}")
    print(f"     Inspected patch position:   {sample_def_patch.grid_pos}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Extract patches and inspect HSV + LBP feature vectors.")
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size (default 16)")
    parser.add_argument("--min-fg-ratio", type=float, default=0.5, help="Minimum foreground ratio per patch (default 0.5)")
    parser.add_argument("--normal", type=Path, default=None, help="Path to sample normal image")
    parser.add_argument("--defective", type=Path, default=None, help="Path to sample defective image")
    parser.add_argument("--output", type=Path, default=Path("outputs/patch_grid_inspection.png"), help="Output visualization path")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    # Auto-discover if not provided
    normal_path = args.normal
    if normal_path is None or not normal_path.exists():
        candidates = [
            repo_root / "dataset" / "bottle" / "test" / "good" / "000.png",
            repo_root / "data" / "metal_nut" / "train" / "good" / "000.png",
        ]
        normal_path = next((p for p in candidates if p.exists()), None)

    defective_path = args.defective
    if defective_path is None or not defective_path.exists():
        candidates = [
            repo_root / "dataset" / "bottle" / "test" / "broken_large" / "000.png",
            repo_root / "data" / "metal_nut" / "test" / "scratch" / "000.png",
        ]
        defective_path = next((p for p in candidates if p.exists()), None)

    if not normal_path or not defective_path:
        raise FileNotFoundError("Normal or defective sample image not found.")

    visualize_patch_grid_and_features(
        normal_path=normal_path,
        defective_path=defective_path,
        output_path=args.output,
        patch_size=args.patch_size,
        min_fg_ratio=args.min_fg_ratio,
    )


if __name__ == "__main__":
    main()
