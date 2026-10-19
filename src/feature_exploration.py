"""Feature Exploration and Classical Feature Extraction for Industrial Surface Defect Inspection.

Analyzes MVTec AD 'metal_nut' dataset using classical computer vision methods:
1. HSV Color Analysis: Hue, Saturation, Value separation for discoloration and illumination defects.
2. Local Texture Analysis: Local Binary Patterns (LBP) for surface scratches and micro-textures.
3. Structural Analysis: Histogram of Oriented Gradients (HOG) for geometric deformities and flips.

Constraints:
- OpenCV, NumPy, scikit-image, and Matplotlib only.
- No deep learning, SAM2, or OCR models.
"""

from pathlib import Path
from typing import Dict, Tuple, List, Optional
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, hog


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "metal_nut"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def load_rgb(image_path: Path) -> np.ndarray:
    """Loads an image from disk in RGB format.

    Args:
        image_path: Path to the image file.

    Returns:
        RGB uint8 numpy array (H, W, 3).
    """
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def to_grayscale(image_rgb: np.ndarray) -> np.ndarray:
    """Converts RGB image to single-channel Grayscale uint8."""
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


# ==============================================================================
# 1. HSV Color Analysis
# ==============================================================================

def analyze_hsv(image_rgb: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, float]]:
    """Performs HSV decomposition and computes color channel metrics.

    Args:
        image_rgb: Input RGB image (H, W, 3).

    Returns:
        Tuple of:
          - Full HSV image (H, W, 3).
          - Dictionary of separated channels ('H', 'S', 'V').
          - Summary statistics (mean and std for Hue, Saturation, Value).
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)

    channels = {
        "H": h_channel,
        "S": s_channel,
        "V": v_channel,
    }

    # Focus statistics on the foreground/nut region (pixels with V > threshold)
    mask = v_channel > 30
    if not np.any(mask):
        mask = np.ones_like(v_channel, dtype=bool)

    stats = {
        "hue_mean": float(np.mean(h_channel[mask])),
        "hue_std": float(np.std(h_channel[mask])),
        "sat_mean": float(np.mean(s_channel[mask])),
        "sat_std": float(np.std(s_channel[mask])),
        "val_mean": float(np.mean(v_channel[mask])),
        "val_std": float(np.std(v_channel[mask])),
    }

    return hsv, channels, stats


def compute_hsv_histograms(
    hsv_image: np.ndarray, bins: int = 32
) -> Dict[str, np.ndarray]:
    """Calculates normalized 1D histograms for Hue, Saturation, and Value channels."""
    h, s, v = cv2.split(hsv_image)
    mask = (v > 30).astype(np.uint8) * 255

    hist_h = cv2.calcHist([h], [0], mask, [bins], [0, 180])
    hist_s = cv2.calcHist([s], [0], mask, [bins], [0, 256])
    hist_v = cv2.calcHist([v], [0], mask, [bins], [0, 256])

    # Normalize to probability distributions
    hist_h = (hist_h / (hist_h.sum() + 1e-7)).flatten()
    hist_s = (hist_s / (hist_s.sum() + 1e-7)).flatten()
    hist_v = (hist_v / (hist_v.sum() + 1e-7)).flatten()

    return {"H": hist_h, "S": hist_s, "V": hist_v}


# ==============================================================================
# 2. Local Texture Analysis using LBP
# ==============================================================================

def analyze_lbp(
    image_gray: np.ndarray,
    n_points: int = 24,
    radius: int = 3,
    method: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Computes Local Binary Patterns (LBP) and normalized texture histogram.

    Args:
        image_gray: Grayscale input image (H, W).
        n_points: Number of circularly symmetric neighbor set points.
        radius: Radius of circle.
        method: LBP operator method ('uniform', 'default', 'ror').

    Returns:
        Tuple of:
          - 2D LBP texture map.
          - 1D normalized LBP histogram.
          - Texture statistics (entropy, energy, contrast).
    """
    lbp_map = local_binary_pattern(image_gray, n_points, radius, method=method)

    # For 'uniform' method, number of bins is n_points + 2
    n_bins = n_points + 2 if method == "uniform" else int(lbp_map.max() + 1)
    hist, _ = np.histogram(
        lbp_map.ravel(),
        bins=n_bins,
        range=(0, n_bins),
        density=True,
    )

    # Texture descriptor metrics
    prob = hist + 1e-10
    entropy = -float(np.sum(prob * np.log2(prob)))
    energy = float(np.sum(hist ** 2))
    variance = float(np.var(lbp_map))

    stats = {
        "lbp_entropy": entropy,
        "lbp_energy": energy,
        "lbp_variance": variance,
    }

    return lbp_map, hist, stats


# ==============================================================================
# 3. Structural Analysis using HOG
# ==============================================================================

def analyze_hog(
    image_gray: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (16, 16),
    cells_per_block: Tuple[int, int] = (2, 2),
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Extracts Histogram of Oriented Gradients (HOG) features and structural map.

    Args:
        image_gray: Grayscale image (H, W).
        orientations: Number of gradient orientation bins.
        pixels_per_cell: Size of a cell in pixels.
        cells_per_block: Number of cells per block.

    Returns:
        Tuple of:
          - 1D HOG feature vector.
          - 2D visualization image of oriented gradients.
          - Gradient structural statistics (mean gradient magnitude, sparsity).
    """
    features, hog_image = hog(
        image_gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=True,
        channel_axis=None,
    )

    stats = {
        "hog_mean": float(np.mean(features)),
        "hog_std": float(np.std(features)),
        "hog_max": float(np.max(features)),
        "hog_sparsity": float(np.mean(features < 0.01)),
    }

    return features, hog_image, stats


# ==============================================================================
# Visualization Routines
# ==============================================================================

def plot_hsv_analysis(sample_dict: Dict[str, np.ndarray], save_path: Path):
    """Plots HSV channel decomposition and channel histograms for representative classes."""
    classes = list(sample_dict.keys())
    n_classes = len(classes)

    fig, axes = plt.subplots(n_classes, 4, figsize=(16, 3.2 * n_classes))
    if n_classes == 1:
        axes = np.expand_dims(axes, 0)

    for idx, cls_name in enumerate(classes):
        rgb = sample_dict[cls_name]
        _, channels, _ = analyze_hsv(rgb)

        axes[idx, 0].imshow(rgb)
        axes[idx, 0].set_title(f"[{cls_name.upper()}] RGB", fontsize=11, fontweight="bold")
        axes[idx, 0].axis("off")

        im_h = axes[idx, 1].imshow(channels["H"], cmap="hsv", vmin=0, vmax=180)
        axes[idx, 1].set_title("Hue (Color Tone)", fontsize=10)
        axes[idx, 1].axis("off")

        im_s = axes[idx, 2].imshow(channels["S"], cmap="inferno", vmin=0, vmax=255)
        axes[idx, 2].set_title("Saturation (Purity)", fontsize=10)
        axes[idx, 2].axis("off")

        im_v = axes[idx, 3].imshow(channels["V"], cmap="gray", vmin=0, vmax=255)
        axes[idx, 3].set_title("Value (Brightness)", fontsize=10)
        axes[idx, 3].axis("off")

    plt.suptitle("HSV Color Analysis across Normal and Defect Classes (metal_nut)", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved HSV analysis plot -> {save_path.name}")


def plot_lbp_texture_analysis(sample_dict: Dict[str, np.ndarray], save_path: Path):
    """Visualizes Local Binary Patterns texture maps and histogram profiles."""
    classes = list(sample_dict.keys())
    n_classes = len(classes)

    fig = plt.figure(figsize=(16, 3.2 * n_classes))

    for idx, cls_name in enumerate(classes):
        rgb = sample_dict[cls_name]
        gray = to_grayscale(rgb)
        lbp_map, hist, _ = analyze_lbp(gray, n_points=24, radius=3, method="uniform")

        # Column 1: Original Grayscale
        ax1 = fig.add_subplot(n_classes, 3, idx * 3 + 1)
        ax1.imshow(gray, cmap="gray")
        ax1.set_title(f"[{cls_name.upper()}] Grayscale", fontsize=10, fontweight="bold")
        ax1.axis("off")

        # Column 2: LBP Texture Map
        ax2 = fig.add_subplot(n_classes, 3, idx * 3 + 2)
        ax2.imshow(lbp_map, cmap="viridis")
        ax2.set_title("LBP Map (P=24, R=3)", fontsize=10)
        ax2.axis("off")

        # Column 3: Normalized LBP Histogram
        ax3 = fig.add_subplot(n_classes, 3, idx * 3 + 3)
        ax3.bar(np.arange(len(hist)), hist, color="steelblue", width=0.8)
        ax3.set_title("LBP Pattern Histogram", fontsize=10)
        ax3.set_ylim(0, max(0.4, float(np.max(hist)) * 1.15))
        ax3.set_xlabel("Pattern Bin", fontsize=8)
        ax3.set_ylabel("Probability", fontsize=8)
        ax3.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Local Texture Analysis (LBP) - Surface Roughness & Scratches", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved LBP texture analysis plot -> {save_path.name}")


def plot_hog_structure_analysis(sample_dict: Dict[str, np.ndarray], save_path: Path):
    """Visualizes Histogram of Oriented Gradients (HOG) structural representations."""
    classes = list(sample_dict.keys())
    n_classes = len(classes)

    fig, axes = plt.subplots(n_classes, 3, figsize=(14, 3.4 * n_classes))
    if n_classes == 1:
        axes = np.expand_dims(axes, 0)

    for idx, cls_name in enumerate(classes):
        rgb = sample_dict[cls_name]
        gray = to_grayscale(rgb)
        _, hog_img, _ = analyze_hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))

        axes[idx, 0].imshow(rgb)
        axes[idx, 0].set_title(f"[{cls_name.upper()}] Original RGB", fontsize=10, fontweight="bold")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(gray, cmap="gray")
        axes[idx, 1].set_title("Grayscale Input", fontsize=10)
        axes[idx, 1].axis("off")

        # Visualizing HOG gradient vectors
        axes[idx, 2].imshow(hog_img, cmap="inferno")
        axes[idx, 2].set_title("HOG Gradient Orientations", fontsize=10)
        axes[idx, 2].axis("off")

    plt.suptitle("Structural Analysis (HOG) - Geometric Deformations (Bent, Flip)", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved HOG structural analysis plot -> {save_path.name}")


def plot_multimodal_overview(sample_dict: Dict[str, np.ndarray], save_path: Path):
    """Combines RGB, HSV, LBP, and HOG into a unified feature comparison matrix."""
    classes = list(sample_dict.keys())
    n_classes = len(classes)

    fig, axes = plt.subplots(n_classes, 4, figsize=(16, 3.3 * n_classes))
    if n_classes == 1:
        axes = np.expand_dims(axes, 0)

    for idx, cls_name in enumerate(classes):
        rgb = sample_dict[cls_name]
        gray = to_grayscale(rgb)
        _, channels, _ = analyze_hsv(rgb)
        lbp_map, _, _ = analyze_lbp(gray, n_points=24, radius=3, method="uniform")
        _, hog_img, _ = analyze_hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))

        # 1. RGB
        axes[idx, 0].imshow(rgb)
        axes[idx, 0].set_title(f"[{cls_name.upper()}] RGB", fontsize=11, fontweight="bold")
        axes[idx, 0].axis("off")

        # 2. HSV Saturation (highlights color / discoloration)
        axes[idx, 1].imshow(channels["S"], cmap="inferno")
        axes[idx, 1].set_title("HSV Saturation", fontsize=10)
        axes[idx, 1].axis("off")

        # 3. LBP Texture (highlights surface scratch)
        axes[idx, 2].imshow(lbp_map, cmap="viridis")
        axes[idx, 2].set_title("LBP Texture Map", fontsize=10)
        axes[idx, 2].axis("off")

        # 4. HOG Structure (highlights bent / flip contour)
        axes[idx, 3].imshow(hog_img, cmap="magma")
        axes[idx, 3].set_title("HOG Gradient Map", fontsize=10)
        axes[idx, 3].axis("off")

    plt.suptitle("Multimodal Classical Feature Inspection Matrix (metal_nut)", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved Multimodal overview plot -> {save_path.name}")


# ==============================================================================
# Quantitative Feature Statistical Comparison
# ==============================================================================

def run_quantitative_feature_comparison() -> None:
    """Computes and displays quantitative feature stats across all dataset splits and classes."""
    splits = {
        "train_good": DATASET_DIR / "train" / "good",
        "test_good": DATASET_DIR / "test" / "good",
        "test_bent": DATASET_DIR / "test" / "bent",
        "test_color": DATASET_DIR / "test" / "color",
        "test_flip": DATASET_DIR / "test" / "flip",
        "test_scratch": DATASET_DIR / "test" / "scratch",
    }

    print("\n" + "=" * 80)
    print("QUANTITATIVE CLASSICAL FEATURE PROFILE SUMMARY")
    print("=" * 80)
    print(f"{'Category':<14} | {'Hue Mean':<9} | {'Sat Mean':<9} | {'Val Std':<9} | {'LBP Entr':<9} | {'HOG Mean':<9}")
    print("-" * 80)

    for label, dir_path in splits.items():
        if not dir_path.exists():
            continue

        image_files = sorted(list(dir_path.glob("*.png")))
        if not image_files:
            continue

        # Evaluate on available samples (or up to 25 to summarize)
        h_means, s_means, v_stds = [], [], []
        lbp_entropies, hog_means = [], []

        for p in image_files[:25]:
            rgb = load_rgb(p)
            gray = to_grayscale(rgb)

            _, _, hsv_stats = analyze_hsv(rgb)
            _, _, lbp_stats = analyze_lbp(gray, n_points=24, radius=3, method="uniform")
            _, _, hog_stats = analyze_hog(gray, orientations=9, pixels_per_cell=(16, 16))

            h_means.append(hsv_stats["hue_mean"])
            s_means.append(hsv_stats["sat_mean"])
            v_stds.append(hsv_stats["val_std"])
            lbp_entropies.append(lbp_stats["lbp_entropy"])
            hog_means.append(hog_stats["hog_mean"])

        print(
            f"{label:<14} | "
            f"{np.mean(h_means):>9.2f} | "
            f"{np.mean(s_means):>9.2f} | "
            f"{np.mean(v_stds):>9.2f} | "
            f"{np.mean(lbp_entropies):>9.3f} | "
            f"{np.mean(hog_means):>9.4f}"
        )

    print("=" * 80)
    print("Key Classical Observations:")
    print("  * COLOR defect: Strong shift in Saturation (Sat Mean spikes vs. normal steel).")
    print("  * SCRATCH defect: Higher localized high-frequency LBP texture variation.")
    print("  * BENT / FLIP defects: Alterations in HOG gradient orientations along outer hex edges/threads.")


# ==============================================================================
# Main Orchestrator
# ==============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MVTec AD metal_nut - Classical Feature Exploration")
    print("=" * 80)
    print(f"Dataset location: {DATASET_DIR.resolve()}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}\n")

    # Pick representative samples for qualitative visualization
    sample_paths = {
        "normal (train)": DATASET_DIR / "train" / "good" / "000.png",
        "normal (test)": DATASET_DIR / "test" / "good" / "000.png",
        "color": DATASET_DIR / "test" / "color" / "000.png",
        "scratch": DATASET_DIR / "test" / "scratch" / "000.png",
        "bent": DATASET_DIR / "test" / "bent" / "000.png",
        "flip": DATASET_DIR / "test" / "flip" / "000.png",
    }

    # Verify existence of samples
    samples: Dict[str, np.ndarray] = {}
    for name, path in sample_paths.items():
        if path.exists():
            samples[name] = load_rgb(path)
        else:
            print(f"[!] Warning: sample path does not exist: {path}")

    if not samples:
        print("[!] No sample images found. Please verify data/metal_nut.")
        return

    print("Generating Feature Exploration Visualizations...")
    # 1. HSV Color Analysis
    plot_hsv_analysis(samples, OUTPUT_DIR / "feature_hsv_analysis.png")

    # 2. Local Texture Analysis using LBP
    plot_lbp_texture_analysis(samples, OUTPUT_DIR / "feature_lbp_texture.png")

    # 3. Structural Analysis using HOG
    plot_hog_structure_analysis(samples, OUTPUT_DIR / "feature_hog_structure.png")

    # 4. Multimodal Overview
    plot_multimodal_overview(samples, OUTPUT_DIR / "feature_multimodal_overview.png")

    # 5. Quantitative Summary Table
    run_quantitative_feature_comparison()

    print("\nFeature exploration complete! All artifacts saved to outputs/.\n")


if __name__ == "__main__":
    main()
