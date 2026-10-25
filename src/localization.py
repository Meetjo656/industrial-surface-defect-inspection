"""Phase 5 - Industrial Surface Defect Localization and Evaluation Module.

Evaluates normal reference bank on test images, produces pixel-level anomaly heatmaps,
derives defect segmentations and bounding boxes, and calculates image- and pixel-level
metrics against ground truth masks.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from src.anomaly_detector import DetectorConfig, NormalReferenceBank
    from src.preprocessing import load_image, load_mask, preprocess_image
except ImportError:
    from anomaly_detector import DetectorConfig, NormalReferenceBank
    from preprocessing import load_image, load_mask, preprocess_image


# ==============================================================================
# Step 1: Defect Localizer Core Engine
# ==============================================================================

class DefectLocalizer:
    """Anomaly localization engine for defect segmentation and evaluation."""

    def __init__(
        self,
        threshold: float = 0.5,
        pixel_threshold: Optional[float] = None,
        min_region_area: int = 16,
    ):
        """Initializes localizer with image- and pixel-level thresholds.

        Args:
            threshold: Image-level anomaly score cut-off.
            pixel_threshold: Pixel-level heatmap anomaly score cut-off.
            min_region_area: Minimum connected-component area in pixels to keep.
        """
        self.threshold = float(threshold)
        self.pixel_threshold = float(pixel_threshold) if pixel_threshold is not None else float(threshold)
        self.min_region_area = int(min_region_area)

    def generate_heatmap(self, image: np.ndarray) -> np.ndarray:
        """Computes pixel-wise anomaly heatmap (fallback placeholder for API compatibility).

        Args:
            image: Normalized input image (H, W, C).

        Returns:
            Heatmap float array (H, W).
        """
        h, w = image.shape[:2]
        return np.zeros((h, w), dtype=np.float32)

    def segment_defect(
        self,
        heatmap: np.ndarray,
        threshold: Optional[float] = None,
        valid_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Thresholds anomaly heatmap to produce a binary defect segmentation mask.

        Args:
            heatmap: Float array of shape (H, W).
            threshold: Pixel anomaly threshold. If None, uses self.pixel_threshold.
            valid_mask: Optional boolean or uint8 mask of valid covered pixels.

        Returns:
            Binary mask (H, W) with {0, 1}.
        """
        th = threshold if threshold is not None else self.pixel_threshold
        pred = (heatmap >= th).astype(np.uint8)
        if valid_mask is not None:
            pred = np.where(valid_mask > 0, pred, 0).astype(np.uint8)
        return pred

    def extract_regions(
        self,
        pred_mask: np.ndarray,
        heatmap: Optional[np.ndarray] = None,
        min_area: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Identifies connected defect components and extracts bounding boxes and confidence.

        Args:
            pred_mask: Binary mask of shape (H, W) with {0, 1}.
            heatmap: Optional anomaly heatmap float array (H, W) for scoring regions.
            min_area: Optional minimum pixel area filter. Defaults to self.min_region_area.

        Returns:
            List of dicts containing bbox (x, y, w, h), area, mean_score, max_score.
        """
        cutoff_area = min_area if min_area is not None else self.min_region_area
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            pred_mask.astype(np.uint8), connectivity=8
        )

        regions: List[Dict[str, Any]] = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < cutoff_area:
                continue

            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])

            comp_mask = labels == i
            mean_sc = float(heatmap[comp_mask].mean()) if heatmap is not None else 1.0
            max_sc = float(heatmap[comp_mask].max()) if heatmap is not None else 1.0

            regions.append({
                "bbox": (x, y, w, h),
                "area": area,
                "mean_score": mean_sc,
                "max_score": max_sc,
            })

        return regions

    def evaluate_localization(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Calculates pixel-level TP, FP, FN, TN, IoU, Precision, Recall, and F1."""
        if valid_mask is not None:
            p = (pred_mask > 0) & (valid_mask > 0)
            g = (gt_mask > 0) & (valid_mask > 0)
            v = valid_mask > 0
        else:
            p = pred_mask > 0
            g = gt_mask > 0
            v = np.ones_like(p, dtype=bool)

        tp = int(np.logical_and(p, g).sum())
        fp = int(np.logical_and(p, ~g & v).sum())
        fn = int(np.logical_and(~p & v, g).sum())
        tn = int(np.logical_and(~p & v, ~g & v).sum())

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if g.sum() == 0 else 0.0)
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        intersection = tp
        union = tp + fp + fn
        if union == 0:
            iou = 1.0 if g.sum() == 0 else 0.0
        else:
            iou = float(intersection / union)

        dice = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else (1.0 if g.sum() == 0 else 0.0)

        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "dice": dice,
            "valid_pixels": int(v.sum()),
        }


# ==============================================================================
# Step 2: Coordinate Mapping & Coverage Derivation
# ==============================================================================

def map_canvas_to_original(
    canvas_map: np.ndarray,
    transform: Dict[str, Any],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Maps 2D canvas array (heatmap or mask) back to original image coordinate space.

    Args:
        canvas_map: 2D array on canvas dimensions (e.g. 256x256).
        transform: Transformation dictionary returned by preprocess_image.
        interpolation: cv2 interpolation flag (INTER_LINEAR for heatmaps, INTER_NEAREST for masks).

    Returns:
        2D array matching original image dimensions (e.g. 700x700).
    """
    orig_h, orig_w = transform["orig_size"]
    x1, y1, x2, y2 = transform["crop_box"]
    offset_x, offset_y = transform["offset"]
    new_w, new_h = transform["scaled_size"]
    crop_w = max(1, x2 - x1)
    crop_h = max(1, y2 - y1)

    canvas_crop = canvas_map[offset_y:offset_y + new_h, offset_x:offset_x + new_w]
    restored_crop = cv2.resize(canvas_crop, (crop_w, crop_h), interpolation=interpolation)

    orig_map = np.zeros((orig_h, orig_w), dtype=canvas_map.dtype)
    orig_map[y1:y2, x1:x2] = restored_crop
    return orig_map


def predict_with_coverage(
    bank: NormalReferenceBank,
    image_input: Union[str, Path, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, Dict[str, Any]]:
    """Predicts patch anomaly scores and derives heatmap with valid pixel coverage mask.

    Returns:
        Tuple of (canvas_img, heatmap, coverage_mask, image_score, transform):
            - canvas_img: Preprocessed canvas RGB image.
            - heatmap: 2D float32 anomaly map on canvas.
            - coverage_mask: 2D bool mask indicating valid patch coverage (count_map > 0).
            - image_score: Overall image anomaly score (95th percentile patch score).
            - transform: Preprocessing coordinate transformation metadata.
    """
    canvas_img, canvas_mask, transform = list(preprocess_image(
        image_input, target_size=bank.config.canvas_size, return_transform=True
    ))
    scored_patches = bank.predict_patch_scores(image_input)

    h, w = bank.config.canvas_size[1], bank.config.canvas_size[0]
    heatmap = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    p_size = bank.config.patch_size
    patch_scores = []

    for p, score in scored_patches:
        y, x = p.image_coords
        heatmap[y:y + p_size, x:x + p_size] += score
        count_map[y:y + p_size, x:x + p_size] += 1.0
        patch_scores.append(score)

    valid_pixels = count_map > 0
    heatmap[valid_pixels] /= count_map[valid_pixels]

    image_score = float(np.percentile(patch_scores, 95)) if patch_scores else 0.0
    return canvas_img, heatmap, valid_pixels, image_score, transform


# ==============================================================================
# Step 3: Threshold Selection on Held-out Normal Validation Split
# ==============================================================================

def select_thresholds_from_validation(
    bank: NormalReferenceBank,
    train_good_dir: Path,
    val_split_ratio: float = 0.10,
    image_percentile: float = 95.0,
    pixel_percentile: float = 99.0,
) -> Dict[str, Any]:
    """Derives image-level and pixel-level anomaly thresholds from held-out normal images.

    Strict protocol: Normal images only. Exactly matches the 198/22 training split used
    to fit the reference bank (indices 198 to 219 of sorted normal training set).
    """
    normal_files = sorted(list(train_good_dir.glob("*.png")) + list(train_good_dir.glob("*.jpg")))
    if not normal_files:
        raise FileNotFoundError(f"No normal training images found in: {train_good_dir}")

    split_idx = bank.num_train_images if bank.num_train_images > 0 else int(len(normal_files) * (1.0 - val_split_ratio))
    val_files = normal_files[split_idx:]

    if not val_files:
        print("[WARN] No held-out images outside reference bank; using last 10% of training set.")
        split_idx = int(len(normal_files) * (1.0 - val_split_ratio))
        val_files = normal_files[split_idx:]

    print(f"[*] Deriving thresholds on {len(val_files)} held-out normal validation images...")

    val_image_scores: List[float] = []
    val_pixel_scores: List[float] = []

    for f in val_files:
        _, hm, valid_cov, img_score, _ = predict_with_coverage(bank, f)
        val_image_scores.append(img_score)
        val_pixel_scores.extend(hm[valid_cov].flatten().tolist())

    img_th = float(np.percentile(val_image_scores, image_percentile))
    pix_th = float(np.percentile(val_pixel_scores, pixel_percentile))

    print(f"[OK] Thresholds derived from {len(val_files)} validation normal images:")
    print(f"     Image threshold (p{image_percentile:.1f}): {img_th:.4f}")
    print(f"     Pixel threshold (p{pixel_percentile:.1f}): {pix_th:.4f}")

    return {
        "num_val_images": len(val_files),
        "val_files": [f.name for f in val_files],
        "image_percentile": image_percentile,
        "pixel_percentile": pixel_percentile,
        "image_threshold": img_th,
        "pixel_threshold": pix_th,
        "val_image_scores_min": float(np.min(val_image_scores)),
        "val_image_scores_mean": float(np.mean(val_image_scores)),
        "val_image_scores_max": float(np.max(val_image_scores)),
        "val_pixel_scores_mean": float(np.mean(val_pixel_scores)),
        "val_pixel_scores_max": float(np.max(val_pixel_scores)),
    }


# ==============================================================================
# Step 4: Visualizations
# ==============================================================================

def save_per_image_figure(
    orig_img: np.ndarray,
    gt_mask: np.ndarray,
    heatmap: np.ndarray,
    pred_mask: np.ndarray,
    regions: List[Dict[str, Any]],
    image_score: float,
    image_threshold: float,
    pixel_threshold: float,
    category: str,
    stem: str,
    save_path: Path,
) -> None:
    """Generates 5-panel per-image visualization figure."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))

    # 1. Original Image
    axes[0].imshow(orig_img)
    axes[0].set_title(f"Original ({orig_img.shape[1]}x{orig_img.shape[0]})\nCat: {category} | {stem}", fontsize=9)
    axes[0].axis("off")

    # 2. Ground-Truth Mask
    if gt_mask.sum() > 0:
        axes[1].imshow(gt_mask, cmap="gray")
        axes[1].set_title(f"Ground Truth Mask\n({gt_mask.sum()} defect px)", fontsize=9)
    else:
        axes[1].imshow(np.zeros_like(gt_mask), cmap="gray")
        axes[1].set_title("Ground Truth\n(Normal / No Defect)", fontsize=9)
    axes[1].axis("off")

    # 3. Anomaly Heatmap
    im_hm = axes[2].imshow(heatmap, cmap="inferno", vmin=0.0, vmax=max(4.0, float(heatmap.max())))
    axes[2].set_title(f"Anomaly Heatmap\nScore: {image_score:.3f} (th={image_threshold:.3f})", fontsize=9)
    axes[2].axis("off")
    fig.colorbar(im_hm, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Predicted Binary Mask
    axes[3].imshow(pred_mask, cmap="gray")
    axes[3].set_title(f"Predicted Defect Mask\n(th={pixel_threshold:.3f}, {pred_mask.sum()} px)", fontsize=9)
    axes[3].axis("off")

    # 5. Original Image with Bounding Boxes
    annotated = orig_img.copy()
    for r in regions:
        bx, by, bw, bh = r["bbox"]
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (255, 40, 40), 2)
        cv2.putText(
            annotated,
            f"{r['mean_score']:.2f}",
            (bx, max(12, by - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 40, 40),
            1,
            cv2.LINE_AA,
        )

    is_anomaly = image_score >= image_threshold
    verdict = "DEFECT" if is_anomaly else "NORMAL"
    verdict_color = "red" if is_anomaly else "green"

    axes[4].imshow(annotated)
    axes[4].set_title(
        f"Defect Regions ({len(regions)})\nPred: {verdict}",
        fontsize=9,
        color=verdict_color,
        fontweight="bold",
    )
    axes[4].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def generate_dataset_plots(
    normal_scores: List[float],
    defect_scores: List[float],
    y_true_img: List[int],
    y_score_img: List[float],
    pixel_y_true: np.ndarray,
    pixel_y_score: np.ndarray,
    image_threshold: float,
    per_category_metrics: Dict[str, Dict[str, float]],
    out_dir: Path,
) -> None:
    """Generates score distribution, ROC curves, PR curves, and category comparisons."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Image Anomaly Score Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(normal_scores, bins=15, alpha=0.65, label=f"Normal (n={len(normal_scores)})", color="green", density=True)
    ax.hist(defect_scores, bins=25, alpha=0.65, label=f"Defective (n={len(defect_scores)})", color="crimson", density=True)
    ax.axvline(image_threshold, color="black", linestyle="--", linewidth=1.8, label=f"Threshold = {image_threshold:.3f}")
    ax.set_title("Image-Level Anomaly Score Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Anomaly Score (95th Percentile Patch Score)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "score_distribution.png", dpi=200)
    plt.close()

    # 2. Image and Pixel ROC Curves
    fig, ax = plt.subplots(figsize=(7, 6))
    if len(np.unique(y_true_img)) > 1:
        fpr_i, tpr_i, _ = roc_curve(y_true_img, y_score_img)
        auc_i = roc_auc_score(y_true_img, y_score_img)
        ax.plot(fpr_i, tpr_i, color="blue", lw=2, label=f"Image ROC (AUROC = {auc_i:.4f})")

    if len(np.unique(pixel_y_true)) > 1:
        # Sample for plotting speed if array is very large
        sample_size = min(len(pixel_y_true), 200000)
        idx = np.random.choice(len(pixel_y_true), sample_size, replace=False)
        fpr_p, tpr_p, _ = roc_curve(pixel_y_true[idx], pixel_y_score[idx])
        auc_p = roc_auc_score(pixel_y_true[idx], pixel_y_score[idx])
        ax.plot(fpr_p, tpr_p, color="darkorange", lw=2, linestyle="--", label=f"Pixel ROC (AUROC = {auc_p:.4f})")

    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=200)
    plt.close()

    # 3. Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(7, 6))
    if len(np.unique(y_true_img)) > 1:
        prec_i, rec_i, _ = precision_recall_curve(y_true_img, y_score_img)
        ap_i = average_precision_score(y_true_img, y_score_img)
        ax.plot(rec_i, prec_i, color="blue", lw=2, label=f"Image PR (AP = {ap_i:.4f})")

    if len(np.unique(pixel_y_true)) > 1:
        sample_size = min(len(pixel_y_true), 200000)
        idx = np.random.choice(len(pixel_y_true), sample_size, replace=False)
        prec_p, rec_p, _ = precision_recall_curve(pixel_y_true[idx], pixel_y_score[idx])
        ap_p = average_precision_score(pixel_y_true[idx], pixel_y_score[idx])
        ax.plot(rec_p, prec_p, color="darkorange", lw=2, linestyle="--", label=f"Pixel PR (AP = {ap_p:.4f})")

    ax.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "precision_recall_curve.png", dpi=200)
    plt.close()

    # 4. Per-Category Metric Comparison
    cats = [c for c in per_category_metrics.keys() if c != "all_defects"]
    if cats:
        x = np.arange(len(cats))
        width = 0.25

        img_aucs = [per_category_metrics[c].get("image_auroc", 0.0) for c in cats]
        pix_ious = [per_category_metrics[c].get("pixel_iou", 0.0) for c in cats]
        pix_f1s = [per_category_metrics[c].get("pixel_f1", 0.0) for c in cats]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - width, img_aucs, width, label="Image AUROC", color="royalblue")
        ax.bar(x, pix_f1s, width, label="Pixel F1", color="coral")
        ax.bar(x + width, pix_ious, width, label="Pixel IoU", color="mediumseagreen")

        ax.set_ylabel("Score", fontsize=10)
        ax.set_title("Per-Category Metric Comparison", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(cats, fontsize=10)
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc="upper right", frameon=True)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig(out_dir / "category_comparison.png", dpi=200)
        plt.close()


# ==============================================================================
# Step 5: Full Evaluation Pipeline
# ==============================================================================

def run_evaluation(
    model_path: Path,
    data_dir: Path,
    output_dir: Path,
    image_threshold: Optional[float] = None,
    pixel_threshold: Optional[float] = None,
    categories: Optional[List[str]] = None,
    limit: Optional[int] = None,
    min_region_area: int = 16,
    save_visualizations: bool = True,
    val_percentile: float = 95.0,
    pixel_percentile: float = 99.0,
) -> Dict[str, Any]:
    """Runs end-to-end defect localization and evaluation pipeline."""
    start_time = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Reference Bank
    print(f"[*] Loading reference bank from: {model_path}")
    bank = NormalReferenceBank.load(model_path)

    # 2. Derive or Assign Thresholds
    train_good_dir = data_dir / "train" / "good"
    if image_threshold is None or pixel_threshold is None:
        val_info = select_thresholds_from_validation(
            bank,
            train_good_dir,
            image_percentile=val_percentile,
            pixel_percentile=pixel_percentile,
        )
        final_img_th = image_threshold if image_threshold is not None else val_info["image_threshold"]
        final_pix_th = pixel_threshold if pixel_threshold is not None else val_info["pixel_threshold"]
    else:
        val_info = {"manual": True}
        final_img_th = float(image_threshold)
        final_pix_th = float(pixel_threshold)

    localizer = DefectLocalizer(
        threshold=final_img_th,
        pixel_threshold=final_pix_th,
        min_region_area=min_region_area,
    )

    # 3. Discover Test Images and Ground Truth Masks
    test_dir = data_dir / "test"
    gt_dir = data_dir / "ground_truth"

    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    all_cat_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    if categories is not None:
        target_cats = [d for d in all_cat_dirs if d.name in categories]
    else:
        target_cats = all_cat_dirs

    image_records: List[Dict[str, Any]] = []
    for cat_dir in target_cats:
        cname = cat_dir.name
        img_files = sorted(list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.jpg")))
        if limit is not None:
            img_files = img_files[:limit]

        for img_p in img_files:
            stem = img_p.stem
            # Match ground truth mask
            gt_mask_path: Optional[Path] = None
            if cname != "good":
                candidate_1 = gt_dir / cname / f"{stem}_mask.png"
                candidate_2 = gt_dir / cname / f"{stem}.png"
                if candidate_1.exists():
                    gt_mask_path = candidate_1
                elif candidate_2.exists():
                    gt_mask_path = candidate_2
                else:
                    print(f"[WARN] Missing ground truth mask for defective image: {img_p}")

            image_records.append({
                "path": img_p,
                "category": cname,
                "is_defect": 1 if cname != "good" else 0,
                "gt_mask_path": gt_mask_path,
            })

    print(f"[*] Total test images queued for evaluation: {len(image_records)}")

    # 4. Process Each Image
    per_image_results: List[Dict[str, Any]] = []
    normal_scores: List[float] = []
    defect_scores: List[float] = []
    y_true_img: List[int] = []
    y_score_img: List[float] = []

    # Flattened arrays for global pixel AUROC & AP
    pixel_y_true_list: List[np.ndarray] = []
    pixel_y_score_list: List[np.ndarray] = []

    # Aggregates for global pixel contingency
    global_pixel_tp = 0
    global_pixel_fp = 0
    global_pixel_fn = 0
    global_pixel_tn = 0

    category_pixel_tp: Dict[str, int] = {}
    category_pixel_fp: Dict[str, int] = {}
    category_pixel_fn: Dict[str, int] = {}
    category_pixel_tn: Dict[str, int] = {}

    failed_images: List[str] = []

    for idx, rec in enumerate(image_records):
        img_p = rec["path"]
        cname = rec["category"]
        is_defect = rec["is_defect"]
        gt_mask_p = rec["gt_mask_path"]

        try:
            # 1. Run detector with coverage derivation
            canvas_img, canvas_hm, canvas_cov, img_score, transform = predict_with_coverage(bank, img_p)

            # 2. Map heatmap and coverage mask to original image dimensions
            orig_img = cv2.cvtColor(cv2.imread(str(img_p)), cv2.COLOR_BGR2RGB)
            orig_hm = map_canvas_to_original(canvas_hm, transform, interpolation=cv2.INTER_LINEAR)
            orig_cov = map_canvas_to_original(canvas_cov.astype(np.uint8), transform, interpolation=cv2.INTER_NEAREST) > 0

            # 3. Load or generate ground truth mask in original image dimensions
            if gt_mask_p is not None and gt_mask_p.exists():
                gt_mask = (cv2.imread(str(gt_mask_p), cv2.IMREAD_GRAYSCALE) > 128).astype(np.uint8)
            else:
                gt_mask = np.zeros(orig_img.shape[:2], dtype=np.uint8)

            # 4. Predict binary mask and extract regions
            pred_mask = localizer.segment_defect(orig_hm, threshold=final_pix_th, valid_mask=orig_cov)
            regions = localizer.extract_regions(pred_mask, heatmap=orig_hm, min_area=min_region_area)

            # 5. Pixel-level contingency on valid covered pixels
            eval_dict = localizer.evaluate_localization(pred_mask, gt_mask, valid_mask=orig_cov)

            # Image-level verdict
            pred_label = 1 if img_score >= final_img_th else 0
            is_correct = bool(pred_label == is_defect)

            y_true_img.append(is_defect)
            y_score_img.append(img_score)

            if is_defect == 1:
                defect_scores.append(img_score)
            else:
                normal_scores.append(img_score)

            # Accumulate pixel contingencies
            global_pixel_tp += eval_dict["tp"]
            global_pixel_fp += eval_dict["fp"]
            global_pixel_fn += eval_dict["fn"]
            global_pixel_tn += eval_dict["tn"]

            if cname not in category_pixel_tp:
                category_pixel_tp[cname] = 0
                category_pixel_fp[cname] = 0
                category_pixel_fn[cname] = 0
                category_pixel_tn[cname] = 0

            category_pixel_tp[cname] += eval_dict["tp"]
            category_pixel_fp[cname] += eval_dict["fp"]
            category_pixel_fn[cname] += eval_dict["fn"]
            category_pixel_tn[cname] += eval_dict["tn"]

            # Collect valid pixel arrays for continuous AUROC/AP
            valid_gt = gt_mask[orig_cov].astype(np.uint8)
            valid_sc = orig_hm[orig_cov].astype(np.float32)
            pixel_y_true_list.append(valid_gt)
            pixel_y_score_list.append(valid_sc)

            res_entry = {
                "filename": img_p.name,
                "category": cname,
                "ground_truth_label": is_defect,
                "image_anomaly_score": round(img_score, 4),
                "predicted_image_label": pred_label,
                "prediction_correct": is_correct,
                "image_threshold": round(final_img_th, 4),
                "pixel_threshold": round(final_pix_th, 4),
                "pixel_precision": round(eval_dict["precision"], 4),
                "pixel_recall": round(eval_dict["recall"], 4),
                "pixel_f1": round(eval_dict["f1"], 4),
                "pixel_iou": round(eval_dict["iou"], 4),
                "defect_region_count": len(regions),
                "valid_pixel_count": eval_dict["valid_pixels"],
            }
            per_image_results.append(res_entry)

            # 6. Save per-image visualization
            if save_visualizations:
                cat_subfolder = "normal" if cname == "good" else cname
                vis_path = output_dir / "visualizations" / cat_subfolder / f"{img_p.stem}_localization.png"
                save_per_image_figure(
                    orig_img=orig_img,
                    gt_mask=gt_mask,
                    heatmap=orig_hm,
                    pred_mask=pred_mask,
                    regions=regions,
                    image_score=img_score,
                    image_threshold=final_img_th,
                    pixel_threshold=final_pix_th,
                    category=cname,
                    stem=img_p.stem,
                    save_path=vis_path,
                )

        except Exception as err:
            print(f"[ERR] Failed processing image {img_p}: {err}")
            failed_images.append(str(img_p))

    total_runtime = time.perf_counter() - start_time
    total_eval = len(per_image_results)
    print(f"[OK] Completed inference on {total_eval} images in {total_runtime:.2f}s ({total_runtime / max(1, total_eval):.3f}s/img)")

    # 5. Compute Image-Level Metrics
    y_true_arr = np.array(y_true_img, dtype=int)
    y_score_arr = np.array(y_score_img, dtype=float)
    y_pred_arr = (y_score_arr >= final_img_th).astype(int)

    has_both_classes = len(np.unique(y_true_arr)) > 1
    image_auroc = float(roc_auc_score(y_true_arr, y_score_arr)) if has_both_classes else 0.0
    image_ap = float(average_precision_score(y_true_arr, y_score_arr)) if has_both_classes else 0.0
    image_prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    image_rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    image_f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()

    image_level_metrics = {
        "total_test_images": total_eval,
        "normal_images": int((y_true_arr == 0).sum()),
        "defective_images": int((y_true_arr == 1).sum()),
        "image_threshold": round(final_img_th, 4),
        "auroc": round(image_auroc, 4),
        "average_precision": round(image_ap, 4),
        "precision": round(image_prec, 4),
        "recall": round(image_rec, 4),
        "f1_score": round(image_f1, 4),
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        },
    }

    # 6. Compute Pixel-Level Metrics
    all_pix_y_true = np.concatenate(pixel_y_true_list) if pixel_y_true_list else np.array([])
    all_pix_y_score = np.concatenate(pixel_y_score_list) if pixel_y_score_list else np.array([])

    pix_prec = float(global_pixel_tp / (global_pixel_tp + global_pixel_fp)) if (global_pixel_tp + global_pixel_fp) > 0 else 0.0
    pix_rec = float(global_pixel_tp / (global_pixel_tp + global_pixel_fn)) if (global_pixel_tp + global_pixel_fn) > 0 else 0.0
    pix_f1 = float(2 * pix_prec * pix_rec / (pix_prec + pix_rec)) if (pix_prec + pix_rec) > 0 else 0.0
    pix_iou = float(global_pixel_tp / (global_pixel_tp + global_pixel_fp + global_pixel_fn)) if (global_pixel_tp + global_pixel_fp + global_pixel_fn) > 0 else 0.0

    if len(all_pix_y_true) > 0 and len(np.unique(all_pix_y_true)) > 1:
        # Sample if large for efficient AUROC calculation
        sample_n = min(len(all_pix_y_true), 500000)
        idx_samp = np.random.choice(len(all_pix_y_true), sample_n, replace=False)
        pixel_auroc = float(roc_auc_score(all_pix_y_true[idx_samp], all_pix_y_score[idx_samp]))
        pixel_ap = float(average_precision_score(all_pix_y_true[idx_samp], all_pix_y_score[idx_samp]))
    else:
        pixel_auroc = 0.0
        pixel_ap = 0.0

    pixel_level_metrics = {
        "pixel_threshold": round(final_pix_th, 4),
        "total_evaluated_pixels": int(len(all_pix_y_true)),
        "global_tp": int(global_pixel_tp),
        "global_fp": int(global_pixel_fp),
        "global_fn": int(global_pixel_fn),
        "global_tn": int(global_pixel_tn),
        "precision": round(pix_prec, 4),
        "recall": round(pix_rec, 4),
        "f1_score": round(pix_f1, 4),
        "iou": round(pix_iou, 4),
        "auroc": round(pixel_auroc, 4),
        "average_precision": round(pixel_ap, 4),
    }

    # 7. Compute Per-Category Metrics
    per_category_metrics: Dict[str, Dict[str, float]] = {}
    categories_present = sorted(list(set(r["category"] for r in per_image_results)))

    for cat in categories_present:
        cat_entries = [r for r in per_image_results if r["category"] == cat]
        cat_scores = [r["image_anomaly_score"] for r in cat_entries]

        # Category image AUROC against normal test images
        if cat != "good":
            cat_y_true = [0] * len(normal_scores) + [1] * len(cat_scores)
            cat_y_score = normal_scores + cat_scores
            cat_auroc = float(roc_auc_score(cat_y_true, cat_y_score)) if len(set(cat_y_true)) > 1 else 0.0
            cat_ap = float(average_precision_score(cat_y_true, cat_y_score)) if len(set(cat_y_true)) > 1 else 0.0
        else:
            cat_auroc = 1.0
            cat_ap = 1.0

        c_tp = category_pixel_tp.get(cat, 0)
        c_fp = category_pixel_fp.get(cat, 0)
        c_fn = category_pixel_fn.get(cat, 0)

        c_prec = float(c_tp / (c_tp + c_fp)) if (c_tp + c_fp) > 0 else 0.0
        c_rec = float(c_tp / (c_tp + c_fn)) if (c_tp + c_fn) > 0 else (1.0 if cat == "good" else 0.0)
        c_f1 = float(2 * c_prec * c_rec / (c_prec + c_rec)) if (c_prec + c_rec) > 0 else 0.0
        c_iou = float(c_tp / (c_tp + c_fp + c_fn)) if (c_tp + c_fp + c_fn) > 0 else (1.0 if cat == "good" else 0.0)

        per_category_metrics[cat] = {
            "image_count": len(cat_entries),
            "mean_image_score": round(float(np.mean(cat_scores)), 4),
            "image_auroc": round(cat_auroc, 4),
            "image_ap": round(cat_ap, 4),
            "pixel_precision": round(c_prec, 4),
            "pixel_recall": round(c_rec, 4),
            "pixel_f1": round(c_f1, 4),
            "pixel_iou": round(c_iou, 4),
        }

    # 8. Save Machine-Readable Outputs
    # A. config.json
    config_dict = {
        "model_path": str(model_path),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "image_threshold": round(final_img_th, 4),
        "pixel_threshold": round(final_pix_th, 4),
        "min_region_area": min_region_area,
        "val_split_protocol": val_info,
        "total_test_images": total_eval,
        "runtime_seconds": round(total_runtime, 2),
        "failed_images": failed_images,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # B. image_level_metrics.json
    with open(output_dir / "image_level_metrics.json", "w") as f:
        json.dump(image_level_metrics, f, indent=2)

    # C. pixel_level_metrics.json
    with open(output_dir / "pixel_level_metrics.json", "w") as f:
        json.dump(pixel_level_metrics, f, indent=2)

    # D. per_image_results.csv
    csv_file = output_dir / "per_image_results.csv"
    if per_image_results:
        fieldnames = list(per_image_results[0].keys())
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_image_results)

    # E. per_category_metrics.csv
    cat_csv_file = output_dir / "per_category_metrics.csv"
    cat_rows = []
    for cat_name, metrics in per_category_metrics.items():
        row = {"category": cat_name}
        row.update(metrics)
        cat_rows.append(row)

    if cat_rows:
        with open(cat_csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(cat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(cat_rows)

    # 9. Generate Summary Figures
    generate_dataset_plots(
        normal_scores=normal_scores,
        defect_scores=defect_scores,
        y_true_img=y_true_img,
        y_score_img=y_score_img,
        pixel_y_true=all_pix_y_true,
        pixel_y_score=all_pix_y_score,
        image_threshold=final_img_th,
        per_category_metrics=per_category_metrics,
        out_dir=output_dir,
    )

    print(f"\n=======================================================")
    print(f"                  PHASE 5 RESULTS                      ")
    print(f"=======================================================")
    print(f" Images evaluated:     {total_eval} (Good: {len(normal_scores)}, Defect: {len(defect_scores)})")
    print(f" Image-level AUROC:    {image_auroc:.4f}")
    print(f" Image-level AP:       {image_ap:.4f}")
    print(f" Image Precision / Rec:{image_prec:.4f} / {image_rec:.4f} (F1: {image_f1:.4f})")
    print(f" Pixel-level AUROC:    {pixel_auroc:.4f}")
    print(f" Pixel-level AP:       {pixel_ap:.4f}")
    print(f" Pixel IoU / F1:       {pix_iou:.4f} / {pix_f1:.4f}")
    print(f" Output directory:     {output_dir}")
    print(f"=======================================================\n")

    return {
        "image_level": image_level_metrics,
        "pixel_level": pixel_level_metrics,
        "per_category": per_category_metrics,
        "total_runtime": total_runtime,
    }


# ==============================================================================
# Step 6: Command Line Interface
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 5 - Surface Defect Localization and Evaluation.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("outputs/metal_nut_reference_bank.pkl"),
        help="Path to serialized reference bank model",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/metal_nut"),
        help="Path to metal_nut dataset root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/localization"),
        help="Path to save evaluation metrics and visualizations",
    )
    parser.add_argument(
        "--image-threshold",
        type=float,
        default=None,
        help="Explicit image anomaly score threshold (default: derived from validation)",
    )
    parser.add_argument(
        "--pixel-threshold",
        type=float,
        default=None,
        help="Explicit pixel anomaly score threshold (default: derived from validation)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Subset of test categories to evaluate (e.g. good bent color)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on images per category for smoke testing",
    )
    parser.add_argument(
        "--min-region-area",
        type=int,
        default=16,
        help="Minimum connected defect region area in pixels",
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Disable per-image visualization rendering to save runtime",
    )
    parser.add_argument(
        "--val-percentile",
        type=float,
        default=95.0,
        help="Percentile on validation normal scores for image threshold",
    )
    parser.add_argument(
        "--pixel-percentile",
        type=float,
        default=99.0,
        help="Percentile on validation pixel scores for pixel threshold",
    )
    args = parser.parse_args()

    run_evaluation(
        model_path=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        image_threshold=args.image_threshold,
        pixel_threshold=args.pixel_threshold,
        categories=args.categories,
        limit=args.limit,
        min_region_area=args.min_region_area,
        save_visualizations=not args.no_visualizations,
        val_percentile=args.val_percentile,
        pixel_percentile=args.pixel_percentile,
    )


if __name__ == "__main__":
    main()
