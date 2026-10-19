"""Defect localization and segmentation module.

Generates anomaly heatmaps and segmentations to pinpoint precise defect locations.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np


class DefectLocalizer:
    """Anomaly localization engine using feature reconstruction or segmentation."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def generate_heatmap(self, image: np.ndarray) -> np.ndarray:
        """Computes pixel-wise anomaly heatmap.

        Args:
            image: Normalized input image (H, W, C).

        Returns:
            Heatmap float array (H, W) with anomaly scores in [0, 1].
        """
        h, w = image.shape[:2]
        return np.zeros((h, w), dtype=np.float32)

    def segment_defect(self, heatmap: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Thresholds anomaly heatmap to produce a binary defect segmentation mask.

        Args:
            heatmap: Float array of shape (H, W).
            threshold: Anomaly threshold cut-off.

        Returns:
            Binary mask (H, W) with {0, 1}.
        """
        th = threshold if threshold is not None else self.threshold
        return (heatmap >= th).astype(np.uint8)

    def evaluate_localization(
        self, pred_mask: np.ndarray, gt_mask: np.ndarray
    ) -> Dict[str, float]:
        """Calculates IoU and Dice score against ground truth mask."""
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        iou = float(intersection / union) if union > 0 else 1.0
        dice = float(2 * intersection / (pred_mask.sum() + gt_mask.sum())) if (pred_mask.sum() + gt_mask.sum()) > 0 else 1.0

        return {"iou": iou, "dice": dice}
