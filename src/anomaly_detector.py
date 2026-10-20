"""Phase 4 - Normal Reference Bank and Anomaly Detector.

Builds a position-aware normal reference bank using spatial KD-Trees (scikit-learn)
from normal training images only. Standardizes features using normal training
statistics and evaluates patch anomaly scores via nearest-neighbor distance.

RULE: Only normal training images (e.g. data/metal_nut/train/good/) are used
for fitting the reference bank and standardizer. Test images and defect masks
are strictly excluded from reference bank construction.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import pickle
import sys
import time
from typing import Dict, List, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from src.preprocessing import preprocess_image
    from src.patch_features import (
        DEFAULT_CANVAS_SIZE,
        DEFAULT_PATCH_SIZE,
        DEFAULT_STRIDE,
        DEFAULT_MIN_FG_RATIO,
        PatchRecord,
        extract_image_patches,
    )
except ImportError:
    from preprocessing import preprocess_image
    from patch_features import (
        DEFAULT_CANVAS_SIZE,
        DEFAULT_PATCH_SIZE,
        DEFAULT_STRIDE,
        DEFAULT_MIN_FG_RATIO,
        PatchRecord,
        extract_image_patches,
    )


@dataclass
class DetectorConfig:
    """Configuration for position-aware reference bank."""
    canvas_size: Tuple[int, int] = DEFAULT_CANVAS_SIZE
    patch_size: int = DEFAULT_PATCH_SIZE
    stride: int = DEFAULT_STRIDE
    min_fg_ratio: float = DEFAULT_MIN_FG_RATIO
    neighbor_window: int = 1  # Tolerance window on grid coordinates


class NormalReferenceBank:
    """Position-aware normal reference bank using regional scikit-learn KD-Trees."""

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self.regional_trees: Dict[Tuple[int, int], KDTree] = {}
        self.regional_data: Dict[Tuple[int, int], np.ndarray] = {}
        self.global_tree: Optional[KDTree] = None
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        self.num_train_images: int = 0
        self.total_patches: int = 0

    def fit(
        self,
        train_good_dir: Union[str, Path],
        max_images: Optional[int] = None,
    ) -> "NormalReferenceBank":
        """Fits reference bank strictly from normal training images.

        Args:
            train_good_dir: Directory containing normal training images.
            max_images: Optional cap on number of training images.
        """
        train_path = Path(train_good_dir)
        if not train_path.exists():
            raise FileNotFoundError(f"Training directory not found: {train_path}")

        # Strict validation: prevent any test or defect data from contaminating reference bank
        path_str = str(train_path).lower().replace("\\", "/")
        forbidden_keywords = ["test", "ground_truth", "scratch", "defect", "broken", "bent", "color"]
        for kw in forbidden_keywords:
            if f"/{kw}" in path_str or f"_{kw}" in path_str:
                raise ValueError(
                    f"VIOLATION: Attempted to build normal reference bank from non-training path: {train_path}. "
                    "Only normal training images may be used."
                )

        image_files = sorted(
            [f for f in train_path.glob("*.png")] + [f for f in train_path.glob("*.jpg")]
        )
        if not image_files:
            raise FileNotFoundError(f"No image files found in: {train_path}")

        if max_images is not None:
            image_files = image_files[:max_images]

        print(f"[*] Extracting normal patches from {len(image_files)} training images...")

        all_features_list: List[np.ndarray] = []
        raw_regional_features: Dict[Tuple[int, int], List[np.ndarray]] = {}

        for idx, img_file in enumerate(image_files):
            norm_img, mask = preprocess_image(img_file, target_size=self.config.canvas_size)
            patches = extract_image_patches(
                norm_img,
                mask,
                patch_size=self.config.patch_size,
                stride=self.config.stride,
                min_fg_ratio=self.config.min_fg_ratio,
                image_id=img_file.name,
            )

            for p in patches:
                all_features_list.append(p.feature_vector)
                g_pos = p.grid_coords
                if g_pos not in raw_regional_features:
                    raw_regional_features[g_pos] = []
                raw_regional_features[g_pos].append(p.feature_vector)

        if not all_features_list:
            raise ValueError("No valid foreground patches extracted from training images.")

        all_features_arr = np.array(all_features_list, dtype=np.float32)

        # Standardize features using statistics computed strictly from normal training data
        self.feature_mean = np.mean(all_features_arr, axis=0)
        self.feature_std = np.std(all_features_arr, axis=0) + 1e-6

        # Standardize and build regional KD-Trees
        self.regional_trees.clear()
        self.regional_data.clear()

        for g_pos, feats in raw_regional_features.items():
            feats_arr = np.array(feats, dtype=np.float32)
            std_feats = (feats_arr - self.feature_mean) / self.feature_std
            self.regional_data[g_pos] = std_feats
            self.regional_trees[g_pos] = KDTree(std_feats, metric="euclidean")

        # Global fallback tree
        std_all_features = (all_features_arr - self.feature_mean) / self.feature_std
        self.global_tree = KDTree(std_all_features, metric="euclidean")

        self.num_train_images = len(image_files)
        self.total_patches = len(all_features_list)

        print(f"[OK] Normal reference bank built successfully:")
        print(f"     Normal training images: {self.num_train_images}")
        print(f"     Total normal patches:   {self.total_patches}")
        print(f"     Active spatial regions: {len(self.regional_trees)}")
        print(f"     Feature dimension:      {len(self.feature_mean)}")
        return self

    def query_patch_distance(
        self,
        standardized_feat: np.ndarray,
        grid_pos: Tuple[int, int],
        use_neighbor_window: bool = True,
    ) -> float:
        """Queries nearest neighbor distance in compatible spatial regions."""
        gr, gc = grid_pos
        w = self.config.neighbor_window if use_neighbor_window else 0

        best_dist = float("inf")
        feat_row = standardized_feat.reshape(1, -1)

        for dr in range(-w, w + 1):
            nr = gr + dr
            for dc in range(-w, w + 1):
                nc = gc + dc
                neighbor_pos = (nr, nc)
                if neighbor_pos in self.regional_trees:
                    dists, _ = self.regional_trees[neighbor_pos].query(feat_row, k=1)
                    d = float(dists[0][0])
                    if d < best_dist:
                        best_dist = d

        # Fallback to global tree if no regional trees match
        if best_dist == float("inf"):
            if self.global_tree is not None:
                dists, _ = self.global_tree.query(feat_row, k=1)
                best_dist = float(dists[0][0])
            else:
                best_dist = 0.0

        return best_dist

    def predict_patch_scores(
        self,
        image_input: Union[str, Path, np.ndarray],
    ) -> List[Tuple[PatchRecord, float]]:
        """Computes anomaly distance score for each foreground patch in an image."""
        if self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Reference bank not fitted or loaded.")

        norm_img, mask = preprocess_image(image_input, target_size=self.config.canvas_size)
        patches = extract_image_patches(
            norm_img,
            mask,
            patch_size=self.config.patch_size,
            stride=self.config.stride,
            min_fg_ratio=self.config.min_fg_ratio,
        )

        scored_patches: List[Tuple[PatchRecord, float]] = []
        for p in patches:
            std_feat = (p.feature_vector - self.feature_mean) / self.feature_std
            score = self.query_patch_distance(std_feat, p.grid_coords, use_neighbor_window=True)
            scored_patches.append((p, score))

        return scored_patches

    def predict_anomaly_map(
        self,
        image_input: Union[str, Path, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Generates 2D anomaly heatmap on canvas from patch nearest-neighbor distances."""
        norm_img, mask = preprocess_image(image_input, target_size=self.config.canvas_size)
        scored_patches = self.predict_patch_scores(image_input)

        h, w = self.config.canvas_size[1], self.config.canvas_size[0]
        heatmap = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        p_size = self.config.patch_size
        patch_scores = []

        for p, score in scored_patches:
            y, x = p.image_coords
            heatmap[y:y + p_size, x:x + p_size] += score
            count_map[y:y + p_size, x:x + p_size] += 1.0
            patch_scores.append(score)

        valid_pixels = count_map > 0
        heatmap[valid_pixels] /= count_map[valid_pixels]

        # Overall image anomaly score (max or 95th percentile patch score)
        image_score = float(np.percentile(patch_scores, 95)) if patch_scores else 0.0
        return norm_img, heatmap, image_score

    def save(self, save_path: Union[str, Path]) -> Path:
        """Serializes reference bank model, regional KD-Trees, and metadata."""
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "config": self.config,
            "regional_trees": self.regional_trees,
            "regional_data": self.regional_data,
            "global_tree": self.global_tree,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "num_train_images": self.num_train_images,
            "total_patches": self.total_patches,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(out_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[OK] Reference bank saved to: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
        return out_path

    @classmethod
    def load(cls, model_path: Union[str, Path]) -> "NormalReferenceBank":
        """Loads serialized reference bank model and metadata."""
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")

        with open(p, "rb") as f:
            payload = pickle.load(f)

        bank = cls(config=payload["config"])
        bank.regional_trees = payload["regional_trees"]
        bank.regional_data = payload.get("regional_data", {})
        bank.global_tree = payload["global_tree"]
        bank.feature_mean = payload["feature_mean"]
        bank.feature_std = payload["feature_std"]
        bank.num_train_images = payload["num_train_images"]
        bank.total_patches = payload["total_patches"]
        print(f"[OK] Reference bank loaded: {bank.total_patches} patches from {bank.num_train_images} images.")
        return bank


def benchmark_kdtree_vs_bruteforce(
    bank: NormalReferenceBank,
    sample_img_path: Path,
) -> Dict[str, float]:
    """Compares scikit-learn KDTree vs brute-force search speed and distance consistency."""
    norm_img, mask = preprocess_image(sample_img_path, target_size=bank.config.canvas_size)
    patches = extract_image_patches(norm_img, mask, min_fg_ratio=bank.config.min_fg_ratio)
    std_feats = [
        (p.grid_coords, (p.feature_vector - bank.feature_mean) / bank.feature_std)
        for p in patches
    ]

    # 1. Benchmark KD-Tree search
    t0 = time.perf_counter()
    kdtree_dists = []
    for g_pos, feat in std_feats:
        d = bank.query_patch_distance(feat, g_pos, use_neighbor_window=True)
        kdtree_dists.append(d)
    kdtree_time_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Benchmark Brute-Force search
    t1 = time.perf_counter()
    bruteforce_dists = []
    w = bank.config.neighbor_window

    for (gr, gc), feat in std_feats:
        best_d = float("inf")
        for dr in range(-w, w + 1):
            nr = gr + dr
            for dc in range(-w, w + 1):
                nc = gc + dc
                neighbor_pos = (nr, nc)
                if neighbor_pos in bank.regional_data:
                    mat = bank.regional_data[neighbor_pos]
                    dists = np.linalg.norm(mat - feat.reshape(1, -1), axis=1)
                    min_d = float(np.min(dists))
                    if min_d < best_d:
                        best_d = min_d
        bruteforce_dists.append(best_d)
    bruteforce_time_ms = (time.perf_counter() - t1) * 1000.0

    max_diff = float(np.max(np.abs(np.array(kdtree_dists) - np.array(bruteforce_dists))))

    print("[BENCHMARK] KD-Tree vs Brute-Force:")
    print(f"            KD-Tree query time:     {kdtree_time_ms:.2f} ms ({len(std_feats)} patches)")
    print(f"            Brute-Force query time: {bruteforce_time_ms:.2f} ms ({len(std_feats)} patches)")
    print(f"            Max distance discrepancy: {max_diff:.8f}")

    return {
        "kdtree_ms": kdtree_time_ms,
        "bruteforce_ms": bruteforce_time_ms,
        "max_diff": max_diff,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 4 - Normal Reference Bank Construction.")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("data/metal_nut/train/good"),
        help="Path to normal training images (ONLY normal/good)",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("outputs/metal_nut_reference_bank.pkl"),
        help="Path to save serialized reference bank",
    )
    parser.add_argument(
        "--max-train-images",
        type=int,
        default=None,
        help="Optional max number of normal images to ingest",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=True,
        help="Run KD-Tree vs Brute-Force benchmark",
    )
    args = parser.parse_args()

    # 1. Build normal reference bank strictly from normal training images
    bank = NormalReferenceBank()
    bank.fit(args.train_dir, max_images=args.max_train_images)

    # 2. Save reference bank
    bank.save(args.output_model)

    # 3. Benchmark on a normal training image
    if args.benchmark:
        sample_img = next(args.train_dir.glob("*.png"))
        benchmark_kdtree_vs_bruteforce(bank, sample_img)

        # Sanity check: inference on normal sample
        _, heatmap, score = bank.predict_anomaly_map(sample_img)
        print(f"[TEST] Sanity check on normal sample: max_heatmap={heatmap.max():.4f}, image_score={score:.4f}")


if __name__ == "__main__":
    main()
