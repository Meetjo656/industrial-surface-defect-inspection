"""Defect Inspection Model Module.

Extracts local appearance features (HSV color + LBP texture) from image patches
and trains an Isolation Forest anomaly detector on normal surface samples.
Supports single category folders or entire multi-category dataset roots.
"""

import argparse
import json
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.preprocessing import preprocess_image


def extract_patch_features(patch_rgb: np.ndarray) -> np.ndarray:
    """Computes 16-D descriptor: 6 HSV stats + 10-bin uniform LBP histogram."""
    hsv = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv_stats = np.concatenate([hsv.mean(axis=(0, 1)), hsv.std(axis=(0, 1))])

    gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)

    return np.concatenate([hsv_stats, lbp_hist]).astype(np.float32)


def extract_image_patches(
    img: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 16,
    stride: int = 16,
    min_fg: float = 0.5,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Extracts foreground patches and pixel coordinates from normalized canvas."""
    h, w = img.shape[:2]
    features, coords = [], []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch_mask = mask[y:y + patch_size, x:x + patch_size]
            if np.mean(patch_mask > 0) < min_fg:
                continue
            patch = img[y:y + patch_size, x:x + patch_size]
            features.append(extract_patch_features(patch))
            coords.append((y, x))

    feats_arr = np.array(features, dtype=np.float32) if features else np.empty((0, 16), dtype=np.float32)
    return feats_arr, coords


def discover_train_images(input_path: Path, max_per_cat: Optional[int] = None) -> List[Path]:
    """Discovers normal training images strictly following expected train/good structure."""
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: {p}")

    # Check if root containing multiple categories (e.g. dataset/ with bottle, metal_nut, ...)
    category_train_dirs = sorted(list(p.glob("*/train/good")))
    if category_train_dirs:
        all_files: List[Path] = []
        for cat_dir in category_train_dirs:
            cat_files = sorted(list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.jpg")))
            if max_per_cat:
                cat_files = cat_files[:max_per_cat]
            all_files.extend(cat_files)
        if all_files:
            return all_files

    # Single train/good directory
    if p.name == "good" and p.parent.name == "train":
        files = sorted(list(p.glob("*.png")) + list(p.glob("*.jpg")))
        if files:
            return files[:max_per_cat] if max_per_cat else files

    # Category directory (e.g. dataset/metal_nut)
    good_dir = p / "train" / "good"
    if good_dir.exists():
        files = sorted(list(good_dir.glob("*.png")) + list(good_dir.glob("*.jpg")))
        if files:
            return files[:max_per_cat] if max_per_cat else files

    raise ValueError(
        f"Directory '{p}' lacks expected 'train/good' structure. "
        "Strict verification requires normal images inside '<category>/train/good'."
    )


class AnomalyModel:
    """Isolation Forest anomaly detection on local surface patches with persistent config."""

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: Union[str, float] = "auto",
        patch_size: int = 16,
        stride: int = 16,
        min_fg: float = 0.5,
        threshold: Optional[float] = None,
    ):
        self.patch_size = patch_size
        self.stride = stride
        self.min_fg = min_fg
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.forest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )

    def set_threshold(self, threshold: float) -> "AnomalyModel":
        """Sets the production defect decision boundary."""
        self.threshold = float(threshold)
        return self

    def get_config(self) -> Dict[str, Any]:
        """Returns serializable configuration dictionary."""
        return {
            "patch_size": self.patch_size,
            "stride": self.stride,
            "min_fg": self.min_fg,
            "threshold": self.threshold,
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
        }

    def fit(self, normal_image_paths: List[Path], max_patches: int = 250000) -> "AnomalyModel":
        """Fits model strictly using normal training images."""
        all_feats: List[np.ndarray] = []
        total = len(normal_image_paths)

        for idx, p in enumerate(normal_image_paths):
            if (idx + 1) % 500 == 0 or (idx + 1) == total:
                print(f"[*] Processed {idx + 1}/{total} images...")
            img, mask, _ = preprocess_image(p)
            feats, _ = extract_image_patches(img, mask, self.patch_size, self.stride, self.min_fg)
            if len(feats) > 0:
                all_feats.append(feats)

        if not all_feats:
            raise ValueError("No patches extracted from input images.")

        x_train = np.vstack(all_feats)

        # Subsample if dataset is exceptionally large for fast training
        if len(x_train) > max_patches:
            idx_sub = np.random.choice(len(x_train), max_patches, replace=False)
            x_train = x_train[idx_sub]

        print(f"[*] Fitting Isolation Forest on {len(x_train):,} normal patches (dim={x_train.shape[1]})...")
        x_scaled = self.scaler.fit_transform(x_train)
        self.forest.fit(x_scaled)
        return self

    def predict(
        self,
        image_input: Union[str, Path, np.ndarray],
    ) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Scores image and generates anomaly heatmap.

        Returns:
            (image_score, heatmap, canvas_mask, transform)
        """
        img, mask, transform = preprocess_image(image_input)
        feats, coords = extract_image_patches(img, mask, self.patch_size, self.stride, self.min_fg)

        h, w = img.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        if len(feats) == 0:
            return 0.0, heatmap, mask, transform

        feats_scaled = self.scaler.transform(feats)
        patch_scores = -self.forest.score_samples(feats_scaled)

        for (y, x), score in zip(coords, patch_scores):
            heatmap[y:y + self.patch_size, x:x + self.patch_size] += score
            count_map[y:y + self.patch_size, x:x + self.patch_size] += 1.0

        valid = count_map > 0
        heatmap[valid] /= count_map[valid]
        image_score = float(np.percentile(patch_scores, 95))
        return image_score, heatmap, mask, transform

    def predict_decision(
        self,
        image_input: Union[str, Path, np.ndarray],
        custom_threshold: Optional[float] = None,
        fallback_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Performs inference and evaluates production binary decision."""
        score, heatmap, mask, transform = self.predict(image_input)
        if custom_threshold is not None:
            effective_threshold = custom_threshold
        elif self.threshold is not None:
            effective_threshold = self.threshold
        else:
            effective_threshold = fallback_threshold

        is_defect = bool(score >= effective_threshold)
        return {
            "is_defect": is_defect,
            "verdict": "DEFECTIVE" if is_defect else "NORMAL",
            "score": round(float(score), 4),
            "threshold": round(float(effective_threshold), 4),
            "heatmap": heatmap,
            "mask": mask,
            "transform": transform,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Saves model parameters and configuration to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        config = self.get_config()

        # Serialize model weights and config bundle
        with open(p, "wb") as f:
            pickle.dump({"scaler": self.scaler, "forest": self.forest, "config": config}, f)

        # Also write readable companion JSON config
        json_path = p.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AnomalyModel":
        """Loads model parameters and restored configuration."""
        p = Path(path)
        with open(p, "rb") as f:
            data = pickle.load(f)

        config = data.get("config", {})
        if not config:
            json_path = p.with_suffix(".json")
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    config = json.load(f)

        obj = cls(
            n_estimators=config.get("n_estimators", 100),
            contamination=config.get("contamination", "auto"),
            patch_size=config.get("patch_size", 16),
            stride=config.get("stride", 16),
            min_fg=config.get("min_fg", 0.5),
            threshold=config.get("threshold", None),
        )
        obj.scaler = data["scaler"]
        obj.forest = data["forest"]
        return obj


def main():
    parser = argparse.ArgumentParser(description="Train Isolation Forest anomaly model on normal dataset.")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("dataset"),
        help="Path to dataset root (e.g. dataset) or specific category train/good folder",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("outputs/metal_nut_isolation_forest.pkl"),
        help="Path to save trained model pickle",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of isolation trees (default 100)",
    )
    parser.add_argument(
        "--max-images-per-cat",
        type=int,
        default=None,
        help="Optional maximum normal images per category",
    )
    args = parser.parse_args()

    files = discover_train_images(args.train_dir, max_per_cat=args.max_images_per_cat)
    print(f"Training on {len(files):,} normal images from: {args.train_dir}")
    t0 = time.time()
    model = AnomalyModel(n_estimators=args.n_estimators).fit(files)
    model.save(args.output_model)
    print(f"[OK] Trained & saved in {time.time() - t0:.2f}s -> {args.output_model}")


if __name__ == "__main__":
    main()
