"""ZenML Pipeline for Industrial Surface Defect Inspection.

Orchestrates data discovery, model training with persistent config,
and production threshold calibration & evaluation.
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
def load_data_step(dataset_dir: str = "dataset") -> List[str]:
    """Discovers and strictly validates normal training images."""
    data_path = Path(dataset_dir)
    image_paths = discover_train_images(data_path)
    print(f"[ZenML Data Step] Discovered {len(image_paths)} normal training images from {data_path}")
    return [str(p) for p in image_paths]


@step
def train_model_step(
    train_image_paths: List[str],
    output_model_path: str = "outputs/full_dataset_model.pkl",
    n_estimators: int = 100,
    max_patches: int = 250000,
) -> str:
    """Fits AnomalyModel with configuration persistence."""
    paths = [Path(p) for p in train_image_paths]
    model = AnomalyModel(n_estimators=n_estimators)
    model.fit(paths, max_patches=max_patches)
    model.save(output_model_path)
    print(f"[ZenML Train Step] Model saved to {output_model_path}")
    return output_model_path


@step
def evaluate_and_calibrate_step(
    model_path: str = "outputs/full_dataset_model.pkl",
    data_dir: str = "dataset",
    output_dir: str = "outputs",
) -> Dict[str, Any]:
    """Evaluates AUROC, mAP, and calibrates production decision threshold."""
    results = evaluate(
        model_path=Path(model_path),
        data_dir=Path(data_dir),
        out_dir=Path(output_dir),
    )
    print(f"[ZenML Eval Step] Decision Threshold: {results.get('decision_threshold')}")
    print(f"[ZenML Eval Step] Image AUROC: {results.get('image_auroc')}")
    return results


@pipeline
def defect_inspection_pipeline(
    dataset_dir: str = "dataset",
    output_model_path: str = "outputs/full_dataset_model.pkl",
    output_dir: str = "outputs",
):
    """End-to-end ZenML training and calibration pipeline."""
    train_paths = load_data_step(dataset_dir=dataset_dir)
    saved_model = train_model_step(
        train_image_paths=train_paths,
        output_model_path=output_model_path,
    )
    evaluate_and_calibrate_step(
        model_path=saved_model,
        data_dir=dataset_dir,
        output_dir=output_dir,
    )


def run_pipeline():
    print("[*] Launching ZenML defect inspection pipeline...")
    defect_inspection_pipeline(
        dataset_dir="dataset",
        output_model_path="outputs/full_dataset_model.pkl",
        output_dir="outputs",
    )


if __name__ == "__main__":
    run_pipeline()
