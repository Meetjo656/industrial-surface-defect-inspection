"""Gradio Web Deployment Application for Industrial Surface Defect Inspection."""

import argparse
import os
from pathlib import Path
import sys
from typing import Dict, Tuple, Union

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.model import AnomalyModel

HF_REPO_ID = os.getenv("HF_MODEL_REPO", "Meetjo656/industrial-defect-inspection-model")
HF_FILENAME = os.getenv("HF_MODEL_FILENAME", "full_dataset_model.pkl")


def ensure_model_artifact(model_path: Union[str, Path] = None) -> Path:
    """Resolves local artifact or downloads versioned model from Hugging Face Hub during startup."""
    if model_path:
        p = Path(model_path)
        if p.exists():
            return p

    candidates = [
        Path("outputs/full_dataset_model.pkl"),
        Path("outputs/metal_nut_isolation_forest.pkl"),
    ]
    for c in candidates:
        if c.exists():
            return c

    # Download from Hugging Face Model Hub during Space startup
    try:
        from huggingface_hub import hf_hub_download
        print(f"[*] Downloading model from Hugging Face repo '{HF_REPO_ID}/{HF_FILENAME}'...")
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            local_dir="outputs",
        )
        return Path(downloaded)
    except Exception as e:
        print(f"[!] Note: Hugging Face download fallback ({e})")
        return candidates[0]


def get_default_model_path() -> Path:
    return ensure_model_artifact()


class DefectInspectionApp:
    """Core inference engine for Gradio deployment."""

    def __init__(self, model_path: Union[str, Path], dataset_dir: Union[str, Path] = "dataset"):
        self.model_path = Path(model_path)
        self.dataset_dir = Path(dataset_dir)
        self.model = None
        if self.model_path.exists():
            self.model = AnomalyModel.load(self.model_path)
            if self.model.threshold is None:
                self.calibrate_threshold(n_samples=10)

    def calibrate_threshold(self, n_samples: int = 10) -> float:
        """Calibrates production threshold on a 10-image balanced validation set."""
        if self.model is None or not self.dataset_dir.exists():
            return 0.5

        normal_imgs = sorted(
            list(self.dataset_dir.glob("*/test/good/*.png")) + list(self.dataset_dir.glob("*/test/good/*.jpg"))
        )
        defect_imgs = sorted([
            p for p in (list(self.dataset_dir.glob("*/test/*/*.png")) + list(self.dataset_dir.glob("*/test/*/*.jpg")))
            if p.parent.name != "good"
        ])

        if not normal_imgs or not defect_imgs:
            self.model.set_threshold(0.5)
            return 0.5

        n_norm = n_samples // 2
        n_def = n_samples - n_norm
        val_imgs = normal_imgs[:n_norm] + defect_imgs[:n_def]
        y_true = [0] * len(val_imgs[:n_norm]) + [1] * len(val_imgs[n_norm:])

        y_score = []
        for p in val_imgs:
            score, _, _, _ = self.model.predict(p)
            y_score.append(score)

        from sklearn.metrics import precision_recall_curve
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
        f1_scores = (2 * precisions * recalls) / np.maximum(precisions + recalls, 1e-8)
        best_idx = int(np.argmax(f1_scores[:-1])) if len(thresholds) > 0 else 0
        calibrated_thresh = float(thresholds[best_idx]) if len(thresholds) > 0 else float(np.mean(y_score))

        self.model.set_threshold(calibrated_thresh)
        try:
            self.model.save(self.model_path)
            print(f"[OK] Calibrated threshold on {len(val_imgs)} validation images: {calibrated_thresh:.4f}")
        except Exception as e:
            print(f"[!] Warning: Could not save calibrated model: {e}")

        return calibrated_thresh

    def inspect(self, image: np.ndarray, custom_threshold: float = None):
        if image is None:
            return None, None, None, "Please upload or select an image."

        if self.model is None:
            return None, None, None, f"Model not found at {self.model_path}. Train model first."

        score, heatmap, mask, transform = self.model.predict(image)
        thresh = custom_threshold if custom_threshold is not None else (
            self.model.threshold if self.model.threshold is not None else 0.5
        )

        is_defect = bool(score >= thresh)
        verdict = "DEFECTIVE" if is_defect else "NORMAL"
        status_color = "#ef4444" if is_defect else "#22c55e"

        # Colorize anomaly heatmap
        norm_hm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        color_hm = cv2.applyColorMap(norm_hm, cv2.COLORMAP_JET)
        color_hm = cv2.cvtColor(color_hm, cv2.COLOR_BGR2RGB)

        canvas_h, canvas_w = heatmap.shape[:2]
        canvas_img = cv2.resize(image, (canvas_w, canvas_h))
        overlay = cv2.addWeighted(canvas_img, 0.6, color_hm, 0.4, 0)

        # Defect segmentation mask at decision threshold
        defect_mask = (heatmap >= thresh).astype(np.uint8) * 255

        summary_md = (
            f"### Verdict: <span style='color:{status_color}; font-size:1.4em; font-weight:bold;'>{verdict}</span>\n\n"
            f"- **Anomaly Score:** `{score:.4f}`\n"
            f"- **Decision Threshold:** `{thresh:.4f}`\n"
            f"- **Status:** {'Defect identified above threshold' if is_defect else 'Clean surface within normal limits'}"
        )

        return overlay, defect_mask, norm_hm, summary_md


def create_gradio_app(model_path: Union[str, Path] = None):
    import gradio as gr

    if model_path is None:
        model_path = get_default_model_path()

    app_engine = DefectInspectionApp(model_path)
    default_thresh = (
        app_engine.model.threshold
        if app_engine.model and app_engine.model.threshold is not None
        else 0.5
    )

    with gr.Blocks(title="Industrial Surface Defect Inspection") as demo:
        gr.Markdown(
            "# 🏭 Industrial Surface Defect Inspection\n"
            "Automated defect detection and localization using patch-based Isolation Forest anomaly modeling."
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(type="numpy", label="Input Surface Image")
                thresh_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=round(float(default_thresh), 4),
                    step=0.01,
                    label="Decision Threshold",
                    info="Threshold to flag an anomaly (calibrated on validation data).",
                )
                inspect_btn = gr.Button("Inspect Surface", variant="primary")

            with gr.Column(scale=1):
                result_markdown = gr.Markdown("### Verdict: Ready for inspection")
                with gr.Tab("Heatmap Overlay"):
                    output_overlay = gr.Image(label="Anomaly Heatmap Overlay")
                with gr.Tab("Defect Mask"):
                    output_mask = gr.Image(label="Segmented Defect Mask")
                with gr.Tab("Raw Heatmap"):
                    output_raw_hm = gr.Image(label="Patch Anomaly Intensity")

        inspect_btn.click(
            fn=app_engine.inspect,
            inputs=[input_img, thresh_slider],
            outputs=[output_overlay, output_mask, output_raw_hm, result_markdown],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch Gradio Defect Inspection App")
    parser.add_argument("--model-path", type=Path, default=get_default_model_path())
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    demo = create_gradio_app(args.model_path)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
