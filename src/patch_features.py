"""Phase 3 - Patch-Level Feature Extraction Module.

Extracts local HSV color statistics and LBP texture features from fixed-size
patches on normalized images, filtering background-heavy patches with an
80% foreground coverage threshold.

Per-patch output schema:
1. Feature vector: 6 HSV statistics + 10 LBP histogram bins (16 dimensions).
2. Normalized grid position: (norm_row, norm_col) in [0.0, 1.0].
3. Foreground coverage: Fraction of foreground pixels in patch (>= 0.80).
4. Image and patch coordinates: (y, x, h, w) pixel bounding box and grid indices.

RULE: Only normal training images are used to construct the reference bank.
Defective test images are strictly reserved for visualization and evaluation.
"""

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import local_binary_pattern

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from src.preprocessing import preprocess_image
except ImportError:
    from preprocessing import preprocess_image


# ==============================================================================
# Phase 3 Configuration: Fixed Patch Geometry & Filtering Thresholds
# ==============================================================================
DEFAULT_CANVAS_SIZE: Tuple[int, int] = (256, 256)   # (width, height)
DEFAULT_PATCH_SIZE: int = 16                        # 16x16 pixel patches
DEFAULT_STRIDE: int = 16                            # Non-overlapping grid
DEFAULT_GRID_ROWS: int = DEFAULT_CANVAS_SIZE[1] // DEFAULT_STRIDE   # 16
DEFAULT_GRID_COLS: int = DEFAULT_CANVAS_SIZE[0] // DEFAULT_STRIDE   # 16
DEFAULT_MIN_FG_RATIO: float = 0.80                  # >= 80% foreground coverage


@dataclass
class PatchRecord:
    """Complete per-patch metadata and feature representation.

    Attributes:
        feature_vector: 16-dimensional descriptor (6 HSV statistics + 10-bin LBP histogram).
        normalized_grid_position: (norm_row, norm_col) center position in [0.0, 1.0].
        foreground_coverage: Ratio of foreground pixels in patch [0.0, 1.0].
        image_coords: Top-left origin (y, x) in normalized canvas pixels.
        patch_coords: Bounding box (y, x, patch_height, patch_width).
        grid_coords: Discrete grid index (grid_row, grid_col).
        image_id: Optional source image path or identifier.
    """
    feature_vector: np.ndarray
    normalized_grid_position: Tuple[float, float]
    foreground_coverage: float
    image_coords: Tuple[int, int]
    patch_coords: Tuple[int, int, int, int]
    grid_coords: Tuple[int, int]
    image_id: Optional[str] = None

    # Backward compatibility properties
    @property
    def features(self) -> np.ndarray:
        return self.feature_vector

    @property
    def appearance_descriptor(self) -> np.ndarray:
        return self.feature_vector

    @property
    def norm_coords(self) -> Tuple[float, float]:
        return self.normalized_grid_position

    @property
    def grid_pos(self) -> Tuple[int, int]:
        return self.grid_coords

    @property
    def y(self) -> int:
        return self.image_coords[0]

    @property
    def x(self) -> int:
        return self.image_coords[1]

    @property
    def size(self) -> int:
        return self.patch_coords[2]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes patch record to a dictionary."""
        return {
            "feature_vector": self.feature_vector.tolist(),
            "feature_dim": len(self.feature_vector),
            "normalized_grid_position": self.normalized_grid_position,
            "foreground_coverage": self.foreground_coverage,
            "image_coords": self.image_coords,
            "patch_coords": self.patch_coords,
            "grid_coords": self.grid_coords,
            "image_id": self.image_id,
        }


def extract_patch_features(
    patch_rgb: np.ndarray,
    lbp_points: int = 8,
    lbp_radius: int = 1,
) -> np.ndarray:
    """Extracts 16-dimensional feature vector: HSV statistics + LBP histogram.

    Args:
        patch_rgb: RGB image patch array of shape (P, P, 3).
        lbp_points: Number of circular points for uniform LBP (default 8).
        lbp_radius: Radius for uniform LBP (default 1).

    Returns:
        16-dimensional float32 feature vector:
            - [0]: Hue mean (scaled [0, 1])
            - [1]: Hue std (scaled [0, 1])
            - [2]: Saturation mean (scaled [0, 1])
            - [3]: Saturation std (scaled [0, 1])
            - [4]: Value mean (scaled [0, 1])
            - [5]: Value std (scaled [0, 1])
            - [6..15]: 10-bin normalized uniform LBP histogram (sums to 1.0)
    """
    if patch_rgb.ndim != 3 or patch_rgb.shape[2] != 3:
        raise ValueError(f"Expected 3-channel RGB patch, got shape {patch_rgb.shape}")

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
    n_bins = lbp_points + 2  # 10 bins for P=8 uniform LBP
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    lbp_hist = lbp_hist.astype(np.float32)

    features = np.concatenate([hsv_stats, lbp_hist]).astype(np.float32)
    return features


def extract_image_patches(
    image_rgb: np.ndarray,
    fg_mask: np.ndarray,
    patch_size: int = DEFAULT_PATCH_SIZE,
    stride: Optional[int] = DEFAULT_STRIDE,
    min_fg_ratio: float = DEFAULT_MIN_FG_RATIO,
    image_id: Optional[str] = None,
) -> List[PatchRecord]:
    """Divides normalized image into fixed patches and filters background-heavy patches.

    Args:
        image_rgb: Normalized RGB canvas image (H, W, 3).
        fg_mask: Separate foreground mask (H, W) with {0, 255}.
        patch_size: Fixed patch size in pixels (default 16).
        stride: Grid step size in pixels (default 16).
        min_fg_ratio: Foreground pixel threshold (default 0.80 = 80%).
        image_id: Optional image identifier or filename.

    Returns:
        List of PatchRecord objects meeting foreground coverage threshold.
    """
    h, w = image_rgb.shape[:2]
    s = stride if stride is not None else patch_size
    records: List[PatchRecord] = []

    for y in range(0, h - patch_size + 1, s):
        grid_r = y // s
        norm_r = float((y + patch_size / 2.0) / h)

        for x in range(0, w - patch_size + 1, s):
            grid_c = x // s
            norm_c = float((x + patch_size / 2.0) / w)

            # Check foreground coverage against 80% threshold
            patch_mask = fg_mask[y:y + patch_size, x:x + patch_size]
            fg_cov = float(np.mean(patch_mask > 0))
            if fg_cov < min_fg_ratio:
                continue

            patch_rgb = image_rgb[y:y + patch_size, x:x + patch_size]
            feat_vec = extract_patch_features(patch_rgb)

            record = PatchRecord(
                feature_vector=feat_vec,
                normalized_grid_position=(norm_r, norm_c),
                foreground_coverage=fg_cov,
                image_coords=(y, x),
                patch_coords=(y, x, patch_size, patch_size),
                grid_coords=(grid_r, grid_c),
                image_id=image_id,
            )
            records.append(record)

    return records


def find_compatible_reference_patches(
    query_patch: PatchRecord,
    reference_patches: List[PatchRecord],
    max_grid_dist: int = 1,
) -> List[PatchRecord]:
    """Filters reference patches for spatial compatibility using normalized grid coordinates."""
    qr, qc = query_patch.grid_coords
    return [
        ref for ref in reference_patches
        if max(abs(ref.grid_coords[0] - qr), abs(ref.grid_coords[1] - qc)) <= max_grid_dist
    ]


def build_normal_reference_bank(
    train_good_dir: Path,
    patch_size: int = DEFAULT_PATCH_SIZE,
    min_fg_ratio: float = DEFAULT_MIN_FG_RATIO,
    max_images: Optional[int] = None,
) -> List[PatchRecord]:
    """Builds normal patch reference bank strictly from normal training images.

    CRITICAL RULE: Rejects any test or defective images. Reference bank is
    constructed exclusively from normal/good training samples.
    """
    if not train_good_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_good_dir}")

    # Enforce strict normal training constraint
    parts_lower = [p.lower() for p in train_good_dir.parts]
    if "test" in parts_lower or any(k in train_good_dir.name.lower() for k in ["defect", "anomaly", "scratch", "broken"]):
        raise ValueError(
            f"VIOLATION: Attempted to build normal reference bank from non-training/defective dir: {train_good_dir}. "
            "Only normal training images may be used."
        )

    image_paths = sorted(list(train_good_dir.glob("*.png")) + list(train_good_dir.glob("*.jpg")))
    if max_images:
        image_paths = image_paths[:max_images]

    bank: List[PatchRecord] = []
    for p in image_paths:
        norm_img, mask = list(preprocess_image(p, target_size=DEFAULT_CANVAS_SIZE))
        patches = extract_image_patches(
            norm_img, mask, patch_size=patch_size, min_fg_ratio=min_fg_ratio, image_id=str(p.name)
        )
        bank.extend(patches)

    print(f"[OK] Normal reference bank built: {len(bank)} patches from {len(image_paths)} normal training images.")
    return bank


def test_feature_dimensions(sample_image_path: Path) -> bool:
    """Verifies feature vector dimensions, ranges, and patch record schema."""
    norm_img, mask = list(preprocess_image(sample_image_path, target_size=DEFAULT_CANVAS_SIZE))
    patches = extract_image_patches(
        norm_img, mask, patch_size=DEFAULT_PATCH_SIZE, min_fg_ratio=DEFAULT_MIN_FG_RATIO
    )

    assert len(patches) > 0, "No foreground patches extracted!"
    p0 = patches[0]

    # Validate feature dimensions
    assert p0.feature_vector.shape == (16,), f"Expected shape (16,), got {p0.feature_vector.shape}"
    assert p0.feature_vector.dtype == np.float32, f"Expected float32, got {p0.feature_vector.dtype}"

    # Validate HSV range and LBP density
    hsv_part = p0.feature_vector[:6]
    lbp_part = p0.feature_vector[6:]
    assert np.all(hsv_part >= 0.0) and np.all(hsv_part <= 1.0), "HSV stats out of [0, 1] range"
    assert len(lbp_part) == 10, f"Expected 10 LBP bins, got {len(lbp_part)}"

    # Validate normalized grid position in [0, 1]
    norm_r, norm_c = p0.normalized_grid_position
    assert 0.0 <= norm_r <= 1.0 and 0.0 <= norm_c <= 1.0, "Normalized grid coords out of bounds"

    # Validate foreground coverage threshold
    assert p0.foreground_coverage >= DEFAULT_MIN_FG_RATIO, f"Foreground ratio {p0.foreground_coverage} < 0.80"

    # Validate coordinates
    assert len(p0.image_coords) == 2, "image_coords must be (y, x)"
    assert len(p0.patch_coords) == 4, "patch_coords must be (y, x, h, w)"
    assert len(p0.grid_coords) == 2, "grid_coords must be (grid_r, grid_c)"

    print("[PASS] Feature dimensions test passed:")
    print(f"       Total dimensions: 16 (6 HSV stats + 10 LBP hist)")
    print(f"       Foreground threshold: >= {int(DEFAULT_MIN_FG_RATIO * 100)}%")
    print(f"       Valid patches extracted from test image: {len(patches)}")
    return True


def visualize_patch_grid_and_features(
    normal_path: Path,
    defective_path: Path,
    output_path: Path,
    patch_size: int = DEFAULT_PATCH_SIZE,
    min_fg_ratio: float = DEFAULT_MIN_FG_RATIO,
) -> Path:
    """Visualizes fixed-geometry patch grid and feature inspection for normal and defective samples.

    RULE: Defective test sample is purely for inspection/visualization, not model fitting.
    """
    norm_img_normal, mask_normal = list(preprocess_image(normal_path, target_size=DEFAULT_CANVAS_SIZE))
    norm_img_defect, mask_defect = list(preprocess_image(defective_path, target_size=DEFAULT_CANVAS_SIZE))

    normal_patches = extract_image_patches(
        norm_img_normal, mask_normal, patch_size=patch_size, min_fg_ratio=min_fg_ratio, image_id="normal"
    )
    defect_patches = extract_image_patches(
        norm_img_defect, mask_defect, patch_size=patch_size, min_fg_ratio=min_fg_ratio, image_id="defective"
    )

    overlay_normal = norm_img_normal.copy()
    for p in normal_patches:
        cv2.rectangle(overlay_normal, (p.x, p.y), (p.x + p.size, p.y + p.size), (0, 255, 0), 1)

    overlay_defect = norm_img_defect.copy()
    for p in defect_patches:
        cv2.rectangle(overlay_defect, (p.x, p.y), (p.x + p.size, p.y + p.size), (255, 140, 0), 1)

    defect_positions = {p.grid_coords: p for p in defect_patches}
    normal_positions = {p.grid_coords: p for p in normal_patches}
    common_pos = set(defect_positions.keys()).intersection(set(normal_positions.keys()))

    if common_pos:
        best_pos = max(
            common_pos,
            key=lambda pos: np.linalg.norm(
                defect_positions[pos].feature_vector - normal_positions[pos].feature_vector
            ),
        )
        sample_norm_patch = normal_positions[best_pos]
        sample_def_patch = defect_positions[best_pos]
    else:
        sample_norm_patch = normal_patches[0]
        sample_def_patch = defect_patches[0]

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

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(overlay_normal)
    ax1.set_title(
        f"Normal Sample Grid ({len(normal_patches)} Valid Patches, FG >= {int(min_fg_ratio*100)}%)",
        fontsize=10,
    )
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(overlay_defect)
    ax2.set_title(
        f"Defective Sample Grid ({len(defect_patches)} Valid Patches, FG >= {int(min_fg_ratio*100)}%)",
        fontsize=10,
    )
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 1])
    ax3.imshow(mask_normal, cmap="gray")
    ax3.set_title("Normal Foreground Mask", fontsize=10)
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(mask_defect, cmap="gray")
    ax4.set_title("Defective Foreground Mask", fontsize=10)
    ax4.axis("off")

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
        sample_norm_patch.feature_vector,
        height=width,
        label="Normal Patch",
        color="#2ca02c",
        alpha=0.85,
    )
    ax5.barh(
        x_indices + width / 2,
        sample_def_patch.feature_vector,
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
        f"16-dim Feature Inspection (Grid: {sample_def_patch.grid_coords})\n"
        f"Norm Pos: ({sample_def_patch.normalized_grid_position[0]:.2f}, {sample_def_patch.normalized_grid_position[1]:.2f}) | FG: {sample_def_patch.foreground_coverage:.1%}",
        fontsize=10,
    )
    ax5.legend(loc="lower right")
    ax5.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[OK] Patch inspection figure saved to: {output_path}")
    print(f"     Normal patches (>= {int(min_fg_ratio*100)}% FG):    {len(normal_patches)}")
    print(f"     Defective patches (>= {int(min_fg_ratio*100)}% FG): {len(defect_patches)}")
    print(f"     Inspected patch grid coords: {sample_def_patch.grid_coords}")
    print(f"     Inspected patch norm coords: ({sample_def_patch.normalized_grid_position[0]:.3f}, {sample_def_patch.normalized_grid_position[1]:.3f})")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Phase 3 - Patch-Level Feature Extraction & Inspection.")
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--min-fg-ratio", type=float, default=DEFAULT_MIN_FG_RATIO)
    parser.add_argument("--normal", type=Path, default=None)
    parser.add_argument("--defective", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/patch_grid_inspection.png"))
    parser.add_argument("--test-dims", action="store_true", default=True, help="Run feature dimension tests")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

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

    # 1. Feature Dimension Validation Test
    if args.test_dims:
        test_feature_dimensions(normal_path)

    # 2. Patch Grid & Feature Inspection Visualization
    visualize_patch_grid_and_features(
        normal_path=normal_path,
        defective_path=defective_path,
        output_path=args.output,
        patch_size=args.patch_size,
        min_fg_ratio=args.min_fg_ratio,
    )


if __name__ == "__main__":
    main()
