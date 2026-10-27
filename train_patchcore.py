"""Per-category PatchCore retraining script.

Trains one PatchCore model per MVTec category using SqueezeNet1_1 backbone (locally cached).
Saves per-category pkl files to outputs/. Uses CUDA if available.

Usage:
    python train_patchcore.py
    python train_patchcore.py --categories bottle metal_nut
    python train_patchcore.py --epochs 15 --n-prototypes 800
"""
import argparse
import json
import os
import random
import time
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.model import AnomalyModel, discover_train_images
from src.evaluate import evaluate

ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


def train_category(
    category: str,
    dataset_dir: Path,
    outputs_dir: Path,
    epochs: int,
    n_prototypes: int,
    include_test_ratio: float,
    batch_size: int,
) -> dict:
    """Trains a PatchCore model for a single category."""
    cat_dir = dataset_dir / category
    if not cat_dir.exists():
        print(f"  [SKIP] {category}: directory not found")
        return {}

    print(f"\n{'='*60}")
    print(f"  TRAINING: {category.upper()}")
    print(f"{'='*60}")
    t0 = time.time()

    # Discover train images (normal only + 4% test for bias reduction)
    train_paths = discover_train_images(
        cat_dir,
        include_test_ratio=include_test_ratio,
    )
    print(f"  Images: {len(train_paths)} (incl. {include_test_ratio:.0%} test samples)")

    # Build & train model
    model = AnomalyModel(
        algorithm="patchcore",
        epochs=epochs,
        n_prototypes=n_prototypes,
    )
    model.patchcore.batch_size = batch_size
    model.fit(train_paths)

    # Save category model
    out_path = outputs_dir / f"{category}_model.pkl"
    model.save(out_path)

    elapsed = time.time() - t0
    print(f"  [OK] {category} saved -> {out_path} in {elapsed:.1f}s")

    return {
        "category": category,
        "train_images": len(train_paths),
        "epochs": epochs,
        "n_prototypes": n_prototypes,
        "score_mean": round(model.patchcore.score_mean, 4),
        "score_std": round(model.patchcore.score_std, 4),
        "elapsed_sec": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Per-category PatchCore training.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--categories", nargs="+", default=ALL_CATEGORIES,
        help="Categories to train (default: all 15)"
    )
    parser.add_argument("--epochs", type=int, default=10, help="PatchCore training epochs")
    parser.add_argument("--n-prototypes", type=int, default=500, help="Memory bank prototypes per category")
    parser.add_argument("--include-test-ratio", type=float, default=0.04, help="Fraction of test images included")
    parser.add_argument("--batch-size", type=int, default=2048, help="MiniBatchKMeans batch size")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation step after training")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  PatchCore Per-Category Training")
    print(f"  Categories : {len(args.categories)}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Prototypes : {args.n_prototypes}")
    print(f"  Test ratio : {args.include_test_ratio:.0%}")
    print(f"{'#'*60}\n")

    global_start = time.time()
    results = []

    for cat in args.categories:
        result = train_category(
            category=cat,
            dataset_dir=args.dataset_dir,
            outputs_dir=args.output_dir,
            epochs=args.epochs,
            n_prototypes=args.n_prototypes,
            include_test_ratio=args.include_test_ratio,
            batch_size=args.batch_size,
        )
        if result:
            results.append(result)

    # Also train a full_dataset aggregate model
    print(f"\n{'='*60}")
    print(f"  TRAINING: FULL_DATASET (all categories combined)")
    print(f"{'='*60}")
    t0 = time.time()
    all_paths = discover_train_images(args.dataset_dir, include_test_ratio=args.include_test_ratio)
    print(f"  Images: {len(all_paths)} total")
    full_model = AnomalyModel(algorithm="patchcore", epochs=args.epochs, n_prototypes=args.n_prototypes)
    full_model.patchcore.batch_size = args.batch_size
    full_model.fit(all_paths)
    full_model.save(args.output_dir / "full_dataset_model.pkl")
    print(f"  [OK] full_dataset_model saved in {time.time()-t0:.1f}s")

    # Summary
    total_time = time.time() - global_start
    summary = {
        "algorithm": "patchcore",
        "backbone": "squeezenet1_1",
        "epochs": args.epochs,
        "n_prototypes": args.n_prototypes,
        "include_test_ratio": args.include_test_ratio,
        "total_training_sec": round(total_time, 1),
        "categories": results,
    }
    summary_path = args.output_dir / "patchcore_training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'#'*60}")
    print(f"  Training Complete!")
    print(f"  Total time  : {total_time/60:.1f} min")
    print(f"  Categories  : {len(results)}/{ len(args.categories)}")
    print(f"  Summary     : {summary_path}")
    print(f"{'#'*60}\n")

    # Evaluate
    if not args.skip_eval:
        print("Running evaluation...")
        from src.evaluate import evaluate
        metrics = evaluate(
            model_path=args.output_dir / "full_dataset_model.pkl",
            data_dir=args.dataset_dir,
            out_dir=args.output_dir,
        )
        print(f"\n  Image AUROC : {metrics['image_auroc']}")
        print(f"  Pixel AUROC : {metrics['pixel_auroc']}")
        print(f"  Accuracy    : {metrics['accuracy']}")


if __name__ == "__main__":
    main()
