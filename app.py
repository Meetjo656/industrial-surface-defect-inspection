"""Gradio Web Deployment Application for Industrial Surface Defect Inspection."""

import argparse
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.model import AnomalyModel

HF_REPO_ID = os.getenv("HF_MODEL_REPO", "nerdyoutlier/Industrial_Surface_Defect_Inspection")
CATEGORIES = [
    "full_dataset",
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def resolve_category_model_path(category: str = "full_dataset") -> Path:
    """Finds local model or downloads category-specific model from Hugging Face."""
    filename = f"{category}_model.pkl"
    local_path = Path("outputs") / filename
    if local_path.exists():
        return local_path

    alt_path = Path(filename)
    if alt_path.exists():
        return alt_path

    # Hugging Face download fallback
    try:
        from huggingface_hub import hf_hub_download
        print(f"[*] Downloading {category} model from HF repo '{HF_REPO_ID}'...")
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f"outputs/{filename}",
        )
        return Path(downloaded)
    except Exception as e:
        print(f"[!] Warning: HF download error for {category} ({e})")
        return local_path


def ensure_model_artifact(model_path: Union[str, Path] = None) -> Path:
    """Resolves default model artifact."""
    if model_path:
        p = Path(model_path)
        if p.exists():
            return p
    return resolve_category_model_path("full_dataset")


def get_default_model_path() -> Path:
    return ensure_model_artifact()


class DefectInspectionApp:
    """Core multi-category inference engine for Gradio deployment."""

    def __init__(self, initial_model_path: Union[str, Path] = None):
        self.models: Dict[str, AnomalyModel] = {}
        if initial_model_path and Path(initial_model_path).exists():
            init_p = Path(initial_model_path)
            cat_name = init_p.stem.replace("_model", "")
            try:
                self.models[cat_name] = AnomalyModel.load(init_p)
            except Exception as e:
                print(f"[!] Error loading {init_p}: {e}")

    def get_or_load_model(self, category: str) -> Optional[AnomalyModel]:
        """Loads and caches category model on demand."""
        if category in self.models:
            return self.models[category]

        model_path = resolve_category_model_path(category)
        if model_path.exists():
            try:
                model = AnomalyModel.load(model_path)
                self.models[category] = model
                return model
            except Exception as e:
                print(f"[!] Error loading model {model_path}: {e}")
                return None
        return None

    def get_category_threshold(self, category: str) -> float:
        """Returns pre-calibrated threshold for selected category."""
        model = self.get_or_load_model(category)
        if model and model.threshold is not None:
            return round(float(model.threshold), 4)
        return 0.5

    def inspect(self, image: np.ndarray, category: str = "full_dataset", custom_threshold: float = None):
        if image is None:
            return None, None, None, "Please upload or select an image."

        # Fallback if category model not present
        if category not in self.models and "full_dataset" in self.models:
            category = "full_dataset"
        elif not self.models and category is None:
            category = "full_dataset"

        model = self.get_or_load_model(category)
        if model is None and self.models:
            model = next(iter(self.models.values()))
            category = next(iter(self.models.keys()))
        if model is None:
            return None, None, None, f"Model for '{category}' not found. Please train or upload model."

        score, heatmap, mask, transform = model.predict(image)
        thresh = custom_threshold if custom_threshold is not None else (
            model.threshold if model.threshold is not None else 0.5
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
            f"### Category: **{category.upper()}**\n\n"
            f"### Verdict: <span style='color:{status_color}; font-size:1.4em; font-weight:bold;'>{verdict}</span>\n\n"
            f"- **Anomaly Score:** `{score:.4f}`\n"
            f"- **Decision Threshold:** `{thresh:.4f}`\n"
            f"- **Status:** {'Defect identified above calibrated threshold' if is_defect else 'Clean surface within normal limits'}"
        )

        return overlay, defect_mask, norm_hm, summary_md


def create_gradio_app(model_path: Union[str, Path] = None):
    import gradio as gr

    app_engine = DefectInspectionApp(model_path)
    default_cat = "screw" if "screw" in CATEGORIES else "full_dataset"
    default_thresh = app_engine.get_category_threshold(default_cat)

    with gr.Blocks(title="Industrial Surface Defect Inspection") as demo:
        gr.Markdown(
            "# 🏭 Industrial Surface Defect Inspection\n"
            "Automated defect detection and localization using patch-based Isolation Forest anomaly modeling."
        )

        with gr.Row():
            with gr.Column(scale=1):
                category_dropdown = gr.Dropdown(
                    choices=CATEGORIES,
                    value=default_cat,
                    label="Material / Object Category",
                    info="Select target industrial surface category for dedicated calibrated anomaly model.",
                )
                input_img = gr.Image(type="numpy", label="Input Surface Image")
                thresh_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=default_thresh,
                    step=0.005,
                    label="Decision Threshold",
                    info="Threshold to flag an anomaly (auto-calibrated per category).",
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

        def update_thresh_on_category_change(cat):
            new_thresh = app_engine.get_category_threshold(cat)
            return gr.update(value=new_thresh)

        category_dropdown.change(
            fn=update_thresh_on_category_change,
            inputs=[category_dropdown],
            outputs=[thresh_slider],
        )

        inspect_btn.click(
            fn=app_engine.inspect,
            inputs=[input_img, category_dropdown, thresh_slider],
            outputs=[output_overlay, output_mask, output_raw_hm, result_markdown],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(
        description="Launch Gradio Defect Inspection App"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=get_default_model_path()
    )

    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", 7860))
    )

    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public Gradio link"
    )

    args = parser.parse_args()

    demo = create_gradio_app(args.model_path)

    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
