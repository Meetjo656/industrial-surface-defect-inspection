"""Model Training Module - Industrial Surface Defect Inspection.

Trains an Isolation Forest anomaly detection model on normal surface patches
from the MVTec AD metal_nut training set. Held-out normal training images are
used strictly to evaluate false-positive behavior. No test or defective images
are used for model fitting.
"""

import argparse
import json
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from src.preprocessing import preprocess_image
    from src.patch_features import (
        DEFAULT_CANVAS_SIZE,
        DEFAULT_MIN_FG_RATIO,
        DEFAULT_PATCH_SIZE,
        DEFAULT_STRIDE,
        PatchRecord,
        extract_image_patches,
    )
except ImportError:
    from preprocessing import preprocess_image
    from patch_features import (
        DEFAULT_CANVAS_SIZE,
        DEFAULT_MIN_FG_RATIO,
        DEFAULT_PATCH_SIZE,
        DEFAULT_STRIDE,
        PatchRecord,
        extract_image_patches,
    )


class IsolationForestAnomalyDetector:
    """Patch-based Isolation Forest anomaly detector for industrial surfaces."""

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: Union[float, str] = "auto",
        random_state: int = 42,
        canvas_size: Tuple[int, int] = DEFAULT_CANVAS_SIZE,
        patch_size: int = DEFAULT_PATCH_SIZE,
        stride: int = DEFAULT_STRIDE,
        min_fg_ratio: float = DEFAULT_MIN_FG_RATIO,
    ):
        self.n_estimators = int(n_estimators)
        self.contamination = contamination
        self.random_state = int(random_state)
        self.canvas_size = tuple(canvas_size)
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.min_fg_ratio = float(min_fg_ratio)

        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_dim: int = 0
        self.num_train_images: int = 0
        self.num_train_patches: int = 0
        self.decision_threshold: float = 0.0

    def fit(self, train_image_paths: List[Path]) -> "IsolationForestAnomalyDetector":
        """Extracts features strictly from normal training images and fits model."""
        forbidden_keywords = ["test", "ground_truth", "scratch", "defect", "broken", "bent", "color"]
        for p in train_image_paths:
            normalized_p = str(p).lower().replace("\\", "/")
            for kw in forbidden_keywords:
                if f"/{kw}" in normalized_p or f"_{kw}" in normalized_p:
                    raise ValueError(
                        f"VIOLATION: Non-training or defective path detected: {p}. "
                        "Only normal training images are permitted for model fitting."
                    )

        patch_features: List[np.ndarray] = []
        for img_path in train_image_paths:
            norm_img, mask = list(preprocess_image(img_path, target_size=self.canvas_size))
            patches = extract_image_patches(
                norm_img,
                mask,
                patch_size=self.patch_size,
                stride=self.stride,
                min_fg_ratio=self.min_fg_ratio,
                image_id=img_path.name,
            )
            for p in patches:
                patch_features.append(p.feature_vector)

        if not patch_features:
            raise ValueError("No valid foreground patches extracted from training images.")

        x_train = np.array(patch_features, dtype=np.float32)
        self.num_train_images = len(train_image_paths)
        self.num_train_patches = len(patch_features)
        self.feature_dim = x_train.shape[1]

        # Featurization order: Fit StandardScaler strictly on normal training patches
        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(x_train)

        # Fit Isolation Forest
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(x_train_scaled)
        return self

    def score_image(
        self,
        image_input: Union[str, Path, np.ndarray],
    ) -> Dict[str, Any]:
        """Scores an image using patch-level isolation forest anomaly scores.

        Anomaly score convention:
        Higher score = more anomalous (inverted from scikit-learn score_samples).
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model has not been fitted or loaded.")

        norm_img, mask = list(preprocess_image(image_input, target_size=self.canvas_size))
        patches = extract_image_patches(
            norm_img,
            mask,
            patch_size=self.patch_size,
            stride=self.stride,
            min_fg_ratio=self.min_fg_ratio,
        )

        if not patches:
            return {
                "num_patches": 0,
                "image_score": 0.0,
                "mean_patch_score": 0.0,
                "max_patch_score": 0.0,
                "outlier_patch_ratio": 0.0,
            }

        feats = np.array([p.feature_vector for p in patches], dtype=np.float32)
        feats_scaled = self.scaler.transform(feats)

        # In sklearn, score_samples returns opposite of anomaly score (lower is more anomalous)
        # We invert so higher score indicates higher degree of anomaly
        patch_anomaly_scores = -self.model.score_samples(feats_scaled)
        decision_scores = self.model.decision_function(feats_scaled)
        is_outlier_patch = decision_scores < 0

        # Aggregated image score: 95th percentile patch anomaly score
        image_score = float(np.percentile(patch_anomaly_scores, 95))
        mean_score = float(np.mean(patch_anomaly_scores))
        max_score = float(np.max(patch_anomaly_scores))
        outlier_ratio = float(np.mean(is_outlier_patch))

        return {
            "num_patches": len(patches),
            "image_score": round(image_score, 4),
            "mean_patch_score": round(mean_score, 4),
            "max_patch_score": round(max_score, 4),
            "outlier_patch_ratio": round(outlier_ratio, 4),
            "patch_scores": patch_anomaly_scores,
        }

    def save(self, output_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> Path:
        """Serializes trained model artifact, scaler, and parameters."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "config": {
                "n_estimators": self.n_estimators,
                "contamination": self.contamination,
                "random_state": self.random_state,
                "canvas_size": self.canvas_size,
                "patch_size": self.patch_size,
                "stride": self.stride,
                "min_fg_ratio": self.min_fg_ratio,
            },
            "training_stats": {
                "num_train_images": self.num_train_images,
                "num_train_patches": self.num_train_patches,
                "feature_dim": self.feature_dim,
            },
            "metadata": metadata or {},
        }

        with open(out_p, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        return out_p


def train_isolation_forest(
    train_dir: Path,
    output_model: Path,
    output_metadata: Path,
    val_ratio: float = 0.10,
    contamination: Union[float, str] = "auto",
    n_estimators: int = 100,
    random_state: int = 42,
    patch_size: int = DEFAULT_PATCH_SIZE,
    stride: int = DEFAULT_STRIDE,
    min_fg_ratio: float = DEFAULT_MIN_FG_RATIO,
    canvas_size: Tuple[int, int] = DEFAULT_CANVAS_SIZE,
) -> Dict[str, Any]:
    """Fits Isolation Forest on normal training images and evaluates false-positive rate."""
    start_time = time.perf_counter()
    train_dir = Path(train_dir)
    output_model = Path(output_model)
    output_metadata = Path(output_metadata)

    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    image_files = sorted(list(train_dir.glob("*.png")) + list(train_dir.glob("*.jpg")))
    if not image_files:
        raise FileNotFoundError(f"No image files found in {train_dir}")

    total_images = len(image_files)
    split_idx = int(total_images * (1.0 - val_ratio))
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    print(f"[*] Total normal images discovered: {total_images}")
    print(f"[*] Training images (fit split):     {len(train_files)}")
    print(f"[*] Held-out validation images:      {len(val_files)} ({val_ratio * 100:.1f}%)")

    # 1. Fit detector on training split
    detector = IsolationForestAnomalyDetector(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        canvas_size=canvas_size,
        patch_size=patch_size,
        stride=stride,
        min_fg_ratio=min_fg_ratio,
    )

    t_fit_start = time.perf_counter()
    detector.fit(train_files)
    fit_duration = time.perf_counter() - t_fit_start

    print(f"[OK] Fitted Isolation Forest in {fit_duration:.2f}s:")
    print(f"     Normal patches extracted: {detector.num_train_patches}")
    print(f"     Feature dimension:        {detector.feature_dim}")
    print(f"     Contamination parameter:  {contamination}")

    # 2. Evaluate False-Positive Behavior on Held-out Normal Validation Set
    val_image_scores: List[float] = []
    val_outlier_ratios: List[float] = []

    for vf in val_files:
        res = detector.score_image(vf)
        val_image_scores.append(res["image_score"])
        val_outlier_ratios.append(res["outlier_patch_ratio"])

    val_scores_arr = np.array(val_image_scores, dtype=np.float32)

    score_distribution = {
        "min": float(np.min(val_scores_arr)),
        "mean": float(np.mean(val_scores_arr)),
        "median": float(np.median(val_scores_arr)),
        "max": float(np.max(val_scores_arr)),
        "std": float(np.std(val_scores_arr)),
        "percentile_90": float(np.percentile(val_scores_arr, 90)),
        "percentile_95": float(np.percentile(val_scores_arr, 95)),
        "percentile_99": float(np.percentile(val_scores_arr, 99)),
    }

    # At decision boundary (chosen threshold = 95th percentile of normal validation scores)
    chosen_threshold = score_distribution["percentile_95"]
    false_positives = int(np.sum(val_scores_arr > chosen_threshold))
    fpr = float(false_positives / max(1, len(val_files)))

    # At default sklearn zero threshold on decision function
    fpr_default_zero = float(np.mean(np.array(val_outlier_ratios) > 0.05))

    val_summary = {
        "num_val_images": len(val_files),
        "score_distribution": score_distribution,
        "chosen_threshold": round(chosen_threshold, 4),
        "false_positive_rate_at_chosen_threshold": round(fpr, 4),
        "false_positive_count": false_positives,
        "validation_protocol": "Strict held-out normal split (no defect test data used)",
    }

    total_duration = time.perf_counter() - start_time

    # 3. Save Model Artifact
    detector.save(output_model, metadata=val_summary)
    print(f"[OK] Model artifact saved to: {output_model}")

    # 4. Save JSON Metadata
    metadata_payload = {
        "model_type": "IsolationForest",
        "contamination": contamination,
        "n_estimators": n_estimators,
        "random_state": random_state,
        "patch_size": patch_size,
        "stride": stride,
        "min_fg_ratio": min_fg_ratio,
        "canvas_size": list(canvas_size),
        "training_stats": {
            "total_normal_images": total_images,
            "train_normal_images": len(train_files),
            "num_patches_extracted": detector.num_train_patches,
            "feature_dimension": detector.feature_dim,
            "training_time_seconds": round(fit_duration, 2),
            "total_pipeline_seconds": round(total_duration, 2),
        },
        "validation_results": val_summary,
        "disclaimer": "Evaluated strictly on held-out normal training images. Defect classification accuracy cannot be claimed without an independently labeled evaluation set.",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_metadata, "w") as f:
        json.dump(metadata_payload, f, indent=2)
    print(f"[OK] Metadata saved to: {output_metadata}")

    print("\n=======================================================")
    print("           ISOLATION FOREST TRAINING RESULTS           ")
    print("=======================================================")
    print(f" Normal training images:    {len(train_files)}")
    print(f" Patches extracted:         {detector.num_train_patches}")
    print(f" Feature dimensions:        {detector.feature_dim}")
    print(f" Training time:             {fit_duration:.2f}s")
    print(f" Validation normal images:  {len(val_files)}")
    print(f" Validation score dist:     min={score_distribution['min']:.4f}, mean={score_distribution['mean']:.4f}, max={score_distribution['max']:.4f}")
    print(f" Chosen threshold (p95):    {chosen_threshold:.4f}")
    print(f" False-positive rate (FPR): {fpr * 100:.1f}% ({false_positives}/{len(val_files)})")
    print(" Note: Defect accuracy requires an independent labeled test set.")
    print("=======================================================\n")

    return metadata_payload


def parse_contamination(val: str) -> Union[float, str]:
    """Parses contamination argument into float or 'auto'."""
    val = val.strip()
    if val.lower() == "auto":
        return "auto"
    try:
        f_val = float(val)
        if 0.0 < f_val <= 0.5:
            return f_val
        raise argparse.ArgumentTypeError(f"Contamination must be in (0.0, 0.5] or 'auto', got {f_val}")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid contamination value: {val}")


def main():
    parser = argparse.ArgumentParser(
        description="Train an Isolation Forest anomaly detection model on normal surface images."
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("data/metal_nut/train/good"),
        help="Path to directory containing normal training images",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("outputs/metal_nut_isolation_forest.pkl"),
        help="Destination path for serialized model pickle",
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=Path("outputs/isolation_forest_metadata.json"),
        help="Destination path for metadata JSON file",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Ratio of normal images reserved for false-positive validation (default 0.10)",
    )
    parser.add_argument(
        "--contamination",
        type=parse_contamination,
        default="auto",
        help="Contamination parameter: float in (0.0, 0.5] or 'auto' (default: 'auto')",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of isolation trees (default: 100)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    train_isolation_forest(
        train_dir=args.train_dir,
        output_model=args.output_model,
        output_metadata=args.output_metadata,
        val_ratio=args.val_ratio,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
