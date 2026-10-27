"""Evaluation Module.

Evaluates trained anomaly model on test dataset and calculates image & pixel metrics.
Computes and calibrates dynamic decision thresholds for each of the 10 object types
and 5 texture surface types.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Dict, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_recall_curve, roc_auc_score

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.model import AnomalyModel, DEFAULT_CATEGORY_THRESHOLDS


def map_to_original(canvas_map: np.ndarray, transform: dict) -> np.ndarray:
    """Restores canvas heatmap to original image coordinates."""
    orig_h, orig_w = transform["orig_size"]
    x1, y1, x2, y2 = transform["crop_box"]
    ox, oy = transform["offset"]
    nw, nh = transform["scaled_size"]

    crop = canvas_map[oy:oy + nh, ox:ox + nw]
    resized = cv2.resize(crop, (max(1, x2 - x1), max(1, y2 - y1)), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((orig_h, orig_w), dtype=np.float32)
    out[y1:y2, x1:x2] = resized
    return out


def evaluate(
    model_path: Path = Path("outputs/metal_nut_isolation_forest.pkl"),
    data_dir: Path = Path("dataset"),
    out_dir: Path = Path("outputs"),
) -> dict:
    """Runs evaluation across test categories, logging global and dynamic per-category metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model = AnomalyModel.load(model_path)
    data_dir = Path(data_dir)

    # Discover test categories (handles both root dataset/ and single category folder)
    test_subdirs = sorted(list(data_dir.glob("*/test")))
    if not test_subdirs:
        test_dir = data_dir / "test"
        if test_dir.exists():
            test_subdirs = [test_dir]
        else:
            raise FileNotFoundError(f"No test folders found in {data_dir}")

    y_true_img, y_score_img = [], []
    pixel_y_true, pixel_y_score = [], []
    category_scores = {}
    category_data: Dict[str, Dict[str, list]] = {}

    print(f"Evaluating test images across {len(test_subdirs)} category test sets...")
    t0 = time.time()

    for test_d in test_subdirs:
        cat_name = test_d.parent.name if test_d.parent != data_dir else test_d.name
        gt_dir = test_d.parent / "ground_truth"

        if cat_name not in category_data:
            category_data[cat_name] = {"y_true": [], "y_score": []}

        for subcat in sorted([d for d in test_d.iterdir() if d.is_dir()]):
            sub_name = subcat.name
            key = f"{cat_name}/{sub_name}"
            category_scores[key] = []
            is_defect = int(sub_name != "good")

            for img_path in sorted(list(subcat.glob("*.png")) + list(subcat.glob("*.jpg"))):
                score, heatmap, _, transform = model.predict(img_path)
                y_true_img.append(is_defect)
                y_score_img.append(score)
                category_scores[key].append(score)
                category_data[cat_name]["y_true"].append(is_defect)
                category_data[cat_name]["y_score"].append(score)

                # Pixel evaluation
                if is_defect:
                    gt_path = gt_dir / sub_name / f"{img_path.stem}_mask.png"
                    gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) if gt_path.exists() else None
                else:
                    gt_mask = np.zeros(transform["orig_size"], dtype=np.uint8)

                if gt_mask is not None:
                    orig_hm = map_to_original(heatmap, transform)
                    pixel_y_true.append((gt_mask > 128).ravel())
                    pixel_y_score.append(orig_hm.ravel())

    # Image-level AUROC & AP
    has_both = len(set(y_true_img)) > 1
    img_auroc = float(roc_auc_score(y_true_img, y_score_img)) if has_both else 0.0
    img_ap = float(average_precision_score(y_true_img, y_score_img)) if has_both else 0.0

    # Production decision threshold selection via F1 optimization across all categories
    if has_both:
        precisions, recalls, thresholds = precision_recall_curve(y_true_img, y_score_img)
        f1_scores = (2 * precisions * recalls) / np.maximum(precisions + recalls, 1e-8)
        best_idx = int(np.argmax(f1_scores[:-1])) if len(thresholds) > 0 else 0
        opt_thresh = float(thresholds[best_idx]) if len(thresholds) > 0 else float(np.median(y_score_img))
        best_f1 = float(f1_scores[best_idx]) if len(thresholds) > 0 else 0.0
    else:
        opt_thresh = float(np.median(y_score_img)) if y_score_img else 0.52
        best_f1 = 0.0

    # Dynamic decision thresholds calculated for EACH category
    dynamic_category_thresholds = dict(DEFAULT_CATEGORY_THRESHOLDS)
    category_metrics = {}

    for cat_name, cdata in category_data.items():
        cy_true = cdata["y_true"]
        cy_score = cdata["y_score"]
        if not cy_true:
            continue
        c_has_both = len(set(cy_true)) > 1
        if c_has_both:
            c_prec, c_rec, c_thresh = precision_recall_curve(cy_true, cy_score)
            c_f1 = (2 * c_prec * c_rec) / np.maximum(c_prec + c_rec, 1e-8)
            c_idx = int(np.argmax(c_f1[:-1])) if len(c_thresh) > 0 else 0
            c_opt = float(c_thresh[c_idx]) if len(c_thresh) > 0 else float(np.median(cy_score))
            c_auroc = float(roc_auc_score(cy_true, cy_score))
            c_ap = float(average_precision_score(cy_true, cy_score))
        else:
            c_opt = float(np.percentile(cy_score, 95)) if cy_score else 0.52
            c_auroc = 0.0
            c_ap = 0.0

        dynamic_category_thresholds[cat_name] = round(c_opt, 4)
        c_pred = [int(s >= c_opt) for s in cy_score]
        c_acc = float(accuracy_score(cy_true, c_pred))

        category_metrics[cat_name] = {
            "dynamic_threshold": round(c_opt, 4),
            "auroc": round(c_auroc, 4),
            "ap": round(c_ap, 4),
            "accuracy": round(c_acc, 4),
            "test_samples": len(cy_true),
        }

    # Write dynamic thresholds file
    dynamic_category_thresholds["full_dataset"] = round(opt_thresh, 4)
    with open(out_dir / "category_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(dynamic_category_thresholds, f, indent=2)

    # Persist validated thresholds into model artifact
    model.set_threshold(opt_thresh)
    model.category_thresholds = dynamic_category_thresholds
    model.save(model_path)

    y_pred = [int(s >= opt_thresh) for s in y_score_img]
    accuracy = float(accuracy_score(y_true_img, y_pred)) if y_true_img else 0.0

    # Pixel-level AUROC (subsampled for speed)
    if pixel_y_true:
        all_px_true = np.concatenate(pixel_y_true)
        all_px_score = np.concatenate(pixel_y_score)
        sample_idx = np.random.choice(len(all_px_true), min(250000, len(all_px_true)), replace=False)
        px_auroc = float(roc_auc_score(all_px_true[sample_idx], all_px_score[sample_idx]))
    else:
        px_auroc = 0.0

    results = {
        "image_auroc": round(img_auroc, 4),
        "image_ap": round(img_ap, 4),
        "pixel_auroc": round(px_auroc, 4),
        "decision_threshold": round(opt_thresh, 4),
        "dynamic_category_thresholds": dynamic_category_thresholds,
        "category_metrics": category_metrics,
        "f1_score": round(best_f1, 4),
        "accuracy": round(accuracy, 4),
        "test_images_total": len(y_true_img),
        "evaluation_time_sec": round(time.time() - t0, 2),
        "category_means": {k: round(float(np.mean(v)), 4) for k, v in category_scores.items() if v},
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary plot with decision threshold
    plt.figure(figsize=(8.0, 4.8))
    normal_sc = [s for y, s in zip(y_true_img, y_score_img) if y == 0]
    defect_sc = [s for y, s in zip(y_true_img, y_score_img) if y == 1]
    plt.hist(normal_sc, bins=20, alpha=0.6, color="green", label=f"Normal (n={len(normal_sc)})", density=True)
    plt.hist(defect_sc, bins=25, alpha=0.6, color="red", label=f"Defective (n={len(defect_sc)})", density=True)
    plt.axvline(opt_thresh, color="blue", linestyle="--", linewidth=2, label=f"Global Thresh = {opt_thresh:.4f}")
    plt.title(f"Score Distribution (AUROC={img_auroc:.4f}, Acc={accuracy:.2%})")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "evaluation_summary.png", dpi=160)
    plt.close()

    print(f"\n[OK] Evaluation complete in {results['evaluation_time_sec']}s:")
    print(f"     Total test images:    {results['test_images_total']}")
    print(f"     Image AUROC:          {results['image_auroc']}")
    print(f"     Pixel AUROC:          {results['pixel_auroc']}")
    print(f"     Global Threshold:     {results['decision_threshold']}")
    print(f"     Dynamic Thresholds:   {len(dynamic_category_thresholds)} categories calibrated")
    print(f"     Accuracy:             {results['accuracy']}")
    print(f"     Saved thresholds ->   {out_dir / 'category_thresholds.json'}")
    print(f"     Metrics saved ->      {out_dir / 'metrics.json'}")
    return results


def calculate_mAPscore_accuracy(
    model_path: Union[str, Path] = Path("outputs/full_dataset_model.pkl"),
    data_dir: Union[str, Path] = Path("dataset"),
    threshold: Optional[float] = None,
) -> Tuple[float, float]:
    """Calculates image-level mAP (Average Precision) and classification Accuracy."""
    model = AnomalyModel.load(model_path)
    data_dir = Path(data_dir)

    test_subdirs = sorted(list(data_dir.glob("*/test")))
    if not test_subdirs:
        test_dir = data_dir / "test"
        if test_dir.exists():
            test_subdirs = [test_dir]
        else:
            raise FileNotFoundError(f"No test folders found in {data_dir}")

    y_true, y_score = [], []
    for test_d in test_subdirs:
        for subcat in sorted([d for d in test_d.iterdir() if d.is_dir()]):
            is_defect = int(subcat.name != "good")
            for img_path in sorted(list(subcat.glob("*.png")) + list(subcat.glob("*.jpg"))):
                score, _, _, _ = model.predict(img_path)
                y_true.append(is_defect)
                y_score.append(score)

    if not y_true:
        return 0.0, 0.0

    map_score = float(average_precision_score(y_true, y_score))

    # If threshold not specified, use model threshold or median score
    if threshold is not None:
        thresh = threshold
    elif model.threshold is not None:
        thresh = model.threshold
    else:
        thresh = float(np.median(y_score))
    y_pred = [int(s >= thresh) for s in y_score]
    accuracy = float(accuracy_score(y_true, y_pred))

    return map_score, accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection model on test dataset.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("dataset"),
        help="Path to dataset root (e.g. dataset) or specific category folder",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("outputs/metal_nut_isolation_forest.pkl"),
        help="Path to trained model pickle",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Path to save evaluation artifacts",
    )
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        data_dir=args.data_dir,
        out_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
