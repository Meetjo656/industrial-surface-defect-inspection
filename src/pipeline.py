"""ZenML Pipeline for Industrial Surface Defect Inspection.

Orchestrates data discovery with test bias reduction (3-5% sampling),
PatchCore model training across epochs, and dynamic per-category threshold calibration.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Graceful ZenML step/pipeline fallback decorators
try:
    from zenml import pipeline, step
except ImportError:
    def step(func):
        return func

    def pipeline(func):
        return func

from src.evaluate import evaluate
from src.model import AnomalyModel, discover_train_images


@step
def load_data_step(
    dataset_dir: str = "dataset",
    include_test_ratio: float = 0.04,  # 3-5% test set to reduce bias
) -> List[str]:
    """Discovers normal training images and includes 3-5% test set samples to reduce bias."""
    data_path = Path(dataset_dir)
    image_paths = discover_train_images(data_path, include_test_ratio=include_test_ratio)
    print(
        f"[ZenML Data Step] Discovered {len(image_paths)} images "
        f"(including {include_test_ratio:.1%} test samples for bias mitigation) from {data_path}"
    )
    return [str(p) for p in image_paths]


@step
def train_model_step(
    train_image_paths: List[str],
    output_model_path: str = "outputs/full_dataset_model.pkl",
    algorithm: str = "patchcore",
    epochs: int = 10,
    n_estimators: int = 100,
    max_patches: int = 250000,
) -> str:
    """Fits AnomalyModel using PatchCore algorithm trained across epochs."""
    paths = [Path(p) for p in train_image_paths]
    model = AnomalyModel(
        algorithm=algorithm,
        epochs=epochs,
        n_estimators=n_estimators,
    )
    model.fit(paths, max_patches=max_patches)
    model.save(output_model_path)
    print(f"[ZenML Train Step] Model ({algorithm}, epochs={epochs}) saved to {output_model_path}")
    return output_model_path


@step
def evaluate_and_calibrate_step(
    model_path: str = "outputs/full_dataset_model.pkl",
    data_dir: str = "dataset",
    output_dir: str = "outputs",
) -> Dict[str, Any]:
    """Evaluates AUROC, mAP, and calibrates dynamic decision thresholds per category."""
    results = evaluate(
        model_path=Path(model_path),
        data_dir=Path(data_dir),
        out_dir=Path(output_dir),
    )
    print(f"[ZenML Eval Step] Global Threshold: {results.get('decision_threshold')}")
    print(f"[ZenML Eval Step] Dynamic Thresholds: {len(results.get('dynamic_category_thresholds', {}))} categories")
    print(f"[ZenML Eval Step] Image AUROC: {results.get('image_auroc')}")
    return results


@pipeline
def defect_inspection_pipeline(
    dataset_dir: str = "dataset",
    output_model_path: str = "outputs/full_dataset_model.pkl",
    output_dir: str = "outputs",
    algorithm: str = "patchcore",
    epochs: int = 10,
    include_test_ratio: float = 0.04,
):
    """End-to-end ZenML training and dynamic threshold calibration pipeline."""
    train_paths = load_data_step(
        dataset_dir=dataset_dir,
        include_test_ratio=include_test_ratio,
    )
    saved_model = train_model_step(
        train_image_paths=train_paths,
        output_model_path=output_model_path,
        algorithm=algorithm,
        epochs=epochs,
    )
    evaluate_and_calibrate_step(
        model_path=saved_model,
        data_dir=dataset_dir,
        output_dir=output_dir,
    )


def run_pipeline():
    print("[*] Launching ZenML defect inspection pipeline with PatchCore and dynamic thresholds...")
    defect_inspection_pipeline(
        dataset_dir="dataset",
        output_model_path="outputs/full_dataset_model.pkl",
        output_dir="outputs",
        algorithm="patchcore",
        epochs=10,
        include_test_ratio=0.04,
    )


if __name__ == "__main__":
    run_pipeline()
