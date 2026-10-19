from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from skimage.feature import local_binary_pattern, hog


@dataclass
class DetectorConfig:
    target_size: Tuple[int, int] = (256, 256)
    patch_size: int = 16
    stride: int = 8
    grid_size: Tuple[int, int] = (8, 8)
    use_hog: bool = False
    hog_orientations: int = 8
    lbp_points: int = 8
    lbp_radius: int = 1
    gaussian_sigma: float = 4.0
    thresh_percentile: float = 98.0
    morph_kernel_size: int = 5
    min_box_area: int = 40
    neighbor_window: int = 1


class PositionAwarePatchDetector:
    def __init__(self, config: Optional[DetectorConfig] = None):
        self.cfg = config or DetectorConfig()
        self.regional_trees: Dict[Tuple[int, int], cKDTree] = {}
        self.global_tree: Optional[cKDTree] = None
        self.feat_mean: Optional[np.ndarray] = None
        self.feat_std: Optional[np.ndarray] = None
        self.threshold: float = 0.0

    @staticmethod
    def extract_foreground_bbox(img_bgr: np.ndarray, thresh_val: int = 30) -> Tuple[int, int, int, int]:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        pts = cv2.findNonZero(mask)
        if pts is None:
            return 0, 0, img_bgr.shape[1], img_bgr.shape[0]
        x, y, w, h = cv2.boundingRect(pts)
        pad = int(max(w, h) * 0.05)
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(img_bgr.shape[1], x + w + pad)
        y1 = min(img_bgr.shape[0], y + h + pad)
        return x0, y0, x1 - x0, y1 - y0

    def preprocess_image(self, img_bgr: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        if bbox is None:
            bbox = self.extract_foreground_bbox(img_bgr)
        x, y, w, h = bbox
        crop = img_bgr[y:y+h, x:x+w]
        resized = cv2.resize(crop, self.cfg.target_size, interpolation=cv2.INTER_AREA)
        return resized, bbox

    def preprocess_mask(self, mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = bbox
        crop = mask[y:y+h, x:x+w]
        return cv2.resize(crop, self.cfg.target_size, interpolation=cv2.INTER_NEAREST)

    def extract_patch_features(self, patch_rgb: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        h_norm = h / 180.0
        s_norm = s / 255.0
        v_norm = v / 255.0

        hsv_feats = [
            np.mean(h_norm), np.std(h_norm),
            np.mean(s_norm), np.std(s_norm),
            np.mean(v_norm), np.std(v_norm),
        ]

        gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray, self.cfg.lbp_points, self.cfg.lbp_radius, method="uniform")
        n_bins = self.cfg.lbp_points + 2
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

        feats = list(hsv_feats) + list(lbp_hist)

        if self.cfg.use_hog:
            hog_feat = hog(
                gray,
                orientations=self.cfg.hog_orientations,
                pixels_per_cell=(self.cfg.patch_size // 2, self.cfg.patch_size // 2),
                cells_per_block=(1, 1),
                visualize=False,
                channel_axis=None,
            )
            feats.extend(hog_feat.flatten())

        return np.array(feats, dtype=np.float32)

    def get_patch_coords(self) -> List[Tuple[int, int, int, int, Tuple[int, int]]]:
        h, w = self.cfg.target_size
        p = self.cfg.patch_size
        s = self.cfg.stride
        gh, gw = self.cfg.grid_size

        coords = []
        for y in range(0, h - p + 1, s):
            cy = min(gh - 1, int(((y + p / 2.0) / h) * gh))
            for x in range(0, w - p + 1, s):
                cx = min(gw - 1, int(((x + p / 2.0) / w) * gw))
                coords.append((y, x, p, p, (cy, cx)))
        return coords

    def fit(self, train_images_bgr: List[np.ndarray]):
        patch_coords = self.get_patch_coords()
        gh, gw = self.cfg.grid_size
        region_features: Dict[Tuple[int, int], List[np.ndarray]] = {
            (r, c): [] for r in range(gh) for c in range(gw)
        }
        all_features: List[np.ndarray] = []

        for img_bgr in train_images_bgr:
            prep_bgr, _ = self.preprocess_image(img_bgr)
            prep_rgb = cv2.cvtColor(prep_bgr, cv2.COLOR_BGR2RGB)

            for y, x, p, _, (cy, cx) in patch_coords:
                patch = prep_rgb[y:y+p, x:x+p]
                feat = self.extract_patch_features(patch)
                region_features[(cy, cx)].append(feat)
                all_features.append(feat)

        all_mat = np.array(all_features, dtype=np.float32)
        self.feat_mean = np.mean(all_mat, axis=0)
        self.feat_std = np.std(all_mat, axis=0) + 1e-6

        for (cy, cx), f_list in region_features.items():
            if f_list:
                normed = (np.array(f_list, dtype=np.float32) - self.feat_mean) / self.feat_std
                self.regional_trees[(cy, cx)] = cKDTree(normed)

        all_normed = (all_mat - self.feat_mean) / self.feat_std
        self.global_tree = cKDTree(all_normed)

        train_scores = []
        for img_bgr in train_images_bgr[:30]:
            _, scores_map = self.predict_map(img_bgr, use_global=False)
            train_scores.append(np.percentile(scores_map, self.cfg.thresh_percentile))
        self.threshold = float(np.mean(train_scores)) if train_scores else 1.0

    def predict_map(self, img_bgr: np.ndarray, use_global: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        prep_bgr, bbox = self.preprocess_image(img_bgr)
        prep_rgb = cv2.cvtColor(prep_bgr, cv2.COLOR_BGR2RGB)
        patch_coords = self.get_patch_coords()

        h, w = self.cfg.target_size
        acc_score = np.zeros((h, w), dtype=np.float32)
        acc_count = np.zeros((h, w), dtype=np.float32)

        p = self.cfg.patch_size
        win = self.cfg.neighbor_window
        gh, gw = self.cfg.grid_size

        for y, x, _, _, (cy, cx) in patch_coords:
            patch = prep_rgb[y:y+p, x:x+p]
            raw_f = self.extract_patch_features(patch)
            norm_f = (raw_f - self.feat_mean) / self.feat_std

            if use_global:
                dist, _ = self.global_tree.query(norm_f, k=1)
            else:
                best_d = float("inf")
                for dy in range(-win, win + 1):
                    ny = cy + dy
                    if 0 <= ny < gh:
                        for dx in range(-win, win + 1):
                            nx = cx + dx
                            if 0 <= nx < gw and (ny, nx) in self.regional_trees:
                                d, _ = self.regional_trees[(ny, nx)].query(norm_f, k=1)
                                if d < best_d:
                                    best_d = d
                dist = best_d if best_d != float("inf") else 0.0

            acc_score[y:y+p, x:x+p] += dist
            acc_count[y:y+p, x:x+p] += 1.0

        mask_nz = acc_count > 0
        acc_score[mask_nz] /= acc_count[mask_nz]
        heatmap = gaussian_filter(acc_score, sigma=self.cfg.gaussian_sigma)
        return prep_bgr, heatmap

    def segment_and_detect(
        self, heatmap: np.ndarray, threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        th = threshold if threshold is not None else self.threshold
        bin_mask = (heatmap >= th).astype(np.uint8) * 255

        k = self.cfg.morph_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        cleaned = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) >= self.cfg.min_box_area:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                boxes.append((bx, by, bw, bh))

        return cleaned, boxes
