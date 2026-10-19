import argparse
import time
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from patch_anomaly_detector import DetectorConfig, PositionAwarePatchDetector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/metal_nut")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--grid_h", type=int, default=8)
    parser.add_argument("--grid_w", type=int, default=8)
    parser.add_argument("--use_hog", action="store_true")
    parser.add_argument("--max_train", type=int, default=220)
    parser.add_argument("--max_test", type=int, default=115)
    return parser.parse_args()


def compute_pro(binary_preds: np.ndarray, gt_masks: np.ndarray) -> float:
    pros = []
    for pred, gt in zip(binary_preds, gt_masks):
        gt_bin = (gt > 0).astype(np.uint8)
        if gt_bin.sum() == 0:
            continue
        num_labels, labels = cv2.connectedComponents(gt_bin)
        for lbl in range(1, num_labels):
            comp = labels == lbl
            comp_area = np.sum(comp)
            overlap = np.sum((pred > 0) & comp)
            pros.append(overlap / comp_area)
    return float(np.mean(pros)) if pros else 0.0


def compute_iou(binary_preds: np.ndarray, gt_masks: np.ndarray) -> float:
    ious = []
    for pred, gt in zip(binary_preds, gt_masks):
        gt_bin = gt > 0
        p_bin = pred > 0
        if not np.any(gt_bin):
            continue
        inter = np.logical_and(p_bin, gt_bin).sum()
        union = np.logical_or(p_bin, gt_bin).sum()
        ious.append(inter / union if union > 0 else 1.0)
    return float(np.mean(ious)) if ious else 0.0


def save_visualization(
    prep_bgr: np.ndarray,
    heatmap: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    save_path: Path,
):
    rgb = cv2.cvtColor(prep_bgr, cv2.COLOR_BGR2RGB)
    annotated = rgb.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Input (Cropped)")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Anomaly Heatmap")
    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("Predicted Mask")
    axes[3].imshow(gt_mask, cmap="gray")
    axes[3].set_title("Ground Truth Mask")
    axes[4].imshow(annotated)
    axes[4].set_title("Candidate Boxes")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = DetectorConfig(
        patch_size=args.patch_size,
        stride=args.stride,
        grid_size=(args.grid_h, args.grid_w),
        use_hog=args.use_hog,
    )
    detector = PositionAwarePatchDetector(cfg)

    train_files = sorted(list((data_dir / "train" / "good").glob("*.png")))[:args.max_train]
    print(f"Loading {len(train_files)} normal training images...")
    train_imgs = [cv2.imread(str(p)) for p in train_files]

    t0 = time.time()
    detector.fit(train_imgs)
    t_fit = time.time() - t0
    print(f"Index built in {t_fit:.2f}s | Calibrated threshold: {detector.threshold:.4f}")

    test_records = []
    test_root = data_dir / "test"
    gt_root = data_dir / "ground_truth"

    for cls_dir in sorted(test_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        is_defect = cls_name != "good"

        for img_path in sorted(cls_dir.glob("*.png")):
            mask_path = gt_root / cls_name / f"{img_path.stem}_mask.png" if is_defect else None
            test_records.append({
                "class": cls_name,
                "is_defect": is_defect,
                "img_path": img_path,
                "mask_path": mask_path,
            })

    test_records = test_records[:args.max_test]
    print(f"Evaluating {len(test_records)} test images...")

    y_img_true = []
    y_img_score = []
    y_pixel_true = []
    y_pixel_score = []
    binary_preds = []
    aligned_gt = []
    normal_fps = 0
    total_normals = 0

    times_regional = []
    times_global = []

    for idx, rec in enumerate(test_records):
        img_bgr = cv2.imread(str(rec["img_path"]))
        bbox = detector.extract_foreground_bbox(img_bgr)

        t_start = time.time()
        prep_bgr, heatmap = detector.predict_map(img_bgr)
        times_regional.append(time.time() - t_start)

        t_g_start = time.time()
        _ = detector.predict_map(img_bgr, use_global=True)
        times_global.append(time.time() - t_g_start)

        if rec["mask_path"] and rec["mask_path"].exists():
            raw_mask = cv2.imread(str(rec["mask_path"]), cv2.IMREAD_GRAYSCALE)
            gt_mask = detector.preprocess_mask(raw_mask, bbox)
            gt_mask = (gt_mask > 128).astype(np.uint8) * 255
        else:
            gt_mask = np.zeros(cfg.target_size, dtype=np.uint8)

        pred_mask, boxes = detector.segment_and_detect(heatmap)

        y_img_true.append(1 if rec["is_defect"] else 0)
        y_img_score.append(float(np.max(heatmap)))

        y_pixel_true.append((gt_mask.ravel() > 0).astype(np.uint8))
        y_pixel_score.append(heatmap.ravel())

        binary_preds.append((pred_mask > 0).astype(np.uint8))
        aligned_gt.append((gt_mask > 0).astype(np.uint8))

        if not rec["is_defect"]:
            total_normals += 1
            if len(boxes) > 0:
                normal_fps += 1

        if idx < 6:
            vis_file = output_dir / f"pred_{rec['class']}_{rec['img_path'].stem}.png"
            save_visualization(prep_bgr, heatmap, pred_mask, gt_mask, boxes, vis_file)

    y_pix_t = np.concatenate(y_pixel_true)
    y_pix_s = np.concatenate(y_pixel_score)

    pix_auroc = roc_auc_score(y_pix_t, y_pix_s) if len(np.unique(y_pix_t)) > 1 else 0.0
    img_auroc = roc_auc_score(y_img_true, y_img_score) if len(np.unique(y_img_true)) > 1 else 0.0

    pro_val = compute_pro(np.array(binary_preds), np.array(aligned_gt))
    iou_val = compute_iou(np.array(binary_preds), np.array(aligned_gt))
    fpr_val = (normal_fps / total_normals) if total_normals > 0 else 0.0

    avg_reg_time = np.mean(times_regional) * 1000.0
    avg_glob_time = np.mean(times_global) * 1000.0
    speedup = avg_glob_time / avg_reg_time if avg_reg_time > 0 else 1.0

    print("=" * 60)
    print("DETECTION & LOCALIZATION BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Pixel AUROC           : {pix_auroc:.4f}")
    print(f"Image AUROC           : {img_auroc:.4f}")
    print(f"Per-Region Overlap PRO: {pro_val:.4f}")
    print(f"Localization IoU      : {iou_val:.4f}")
    print(f"Normal FPR @ Thresh   : {fpr_val:.4f} ({normal_fps}/{total_normals})")
    print(f"Regional Query Latency: {avg_reg_time:.2f} ms / image")
    print(f"Global Query Latency  : {avg_glob_time:.2f} ms / image")
    print(f"Regional vs Global Spd: {speedup:.2f}x faster")
    print(f"Visualizations saved  : {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
