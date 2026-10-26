"""Deployment and Production Verification Tests.

Validates inference pipeline, image preprocessing, model config persistence,
strict training discovery, decision threshold classification, and Gradio app engine.
"""

from pathlib import Path
import sys
import tempfile

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app import DefectInspectionApp, ensure_model_artifact, get_default_model_path
from src.model import AnomalyModel, discover_train_images
from src.preprocessing import preprocess_image


@pytest.fixture
def dummy_surface_image():
    """Generates synthetic RGB surface image for headless testing."""
    img = np.ones((256, 256, 3), dtype=np.uint8) * 128
    # Add texture / defect pattern
    img[100:150, 100:150, :] = 40
    return img


def test_model_config_serialization():
    """Issue 1 test: Model configuration persists alongside pickle artifact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_model_path = Path(tmpdir) / "test_model.pkl"
        tmp_json_path = Path(tmpdir) / "test_model.json"

        model = AnomalyModel(
            n_estimators=10,
            patch_size=16,
            stride=16,
            min_fg=0.5,
            threshold=0.25,
        )

        # Create dummy fitted state
        dummy_feats = np.random.randn(20, 16).astype(np.float32)
        model.scaler.fit(dummy_feats)
        model.forest.fit(model.scaler.transform(dummy_feats))

        model.save(tmp_model_path)

        assert tmp_model_path.exists(), "Model pickle not saved"
        assert tmp_json_path.exists(), "Companion JSON config not saved alongside artifact"

        # Load and verify restoration
        loaded = AnomalyModel.load(tmp_model_path)
        assert loaded.threshold == 0.25
        assert loaded.patch_size == 16
        assert loaded.stride == 16
        assert loaded.min_fg == 0.5
        assert loaded.n_estimators == 10


def test_enforce_train_good_structure():
    """Issue 2 test: Permissive fallback is disabled; invalid paths raise ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        invalid_dir = Path(tmpdir) / "bad_dataset"
        invalid_dir.mkdir(parents=True)
        (invalid_dir / "random.png").touch()

        # Permissive fallback must NOT accept this folder
        with pytest.raises(ValueError, match="Expected folders matching '*/train/good'|lacks expected 'train/good' structure"):
            discover_train_images(invalid_dir)

        # Valid train/good structure must succeed
        valid_good_dir = invalid_dir / "train" / "good"
        valid_good_dir.mkdir(parents=True)
        (valid_good_dir / "normal_000.png").touch()

        found = discover_train_images(invalid_dir)
        assert len(found) == 1
        assert found[0].name == "normal_000.png"


def test_production_decision_threshold(dummy_surface_image):
    """Issue 3 test: Validated threshold partitions scores into binary decision."""
    model_path = get_default_model_path()
    if not model_path.exists():
        pytest.skip(f"Model checkpoint not found at {model_path}")

    model = AnomalyModel.load(model_path)

    # Test decision with low threshold -> DEFECTIVE
    res_defective = model.predict_decision(dummy_surface_image, custom_threshold=-1.0)
    assert res_defective["is_defect"] is True
    assert res_defective["verdict"] == "DEFECTIVE"
    assert "score" in res_defective
    assert "threshold" in res_defective

    # Test decision with high threshold -> NORMAL
    res_normal = model.predict_decision(dummy_surface_image, custom_threshold=100.0)
    assert res_normal["is_defect"] is False
    assert res_normal["verdict"] == "NORMAL"


def test_image_preprocessing_pipeline(dummy_surface_image):
    """Issue 6 test: Preprocessing normalizes canvas and maintains mapping transform."""
    canvas_img, canvas_mask, transform = preprocess_image(dummy_surface_image, target_size=(256, 256))

    assert canvas_img.shape == (256, 256, 3)
    assert canvas_mask.shape == (256, 256)
    assert "orig_size" in transform
    assert "scaled_size" in transform
    assert "crop_box" in transform


def test_gradio_app_engine_inference(dummy_surface_image):
    """Issue 4 test: Gradio app inference returns valid visual and metric outputs."""
    model_path = get_default_model_path()
    if not model_path.exists():
        pytest.skip(f"Model checkpoint not found at {model_path}")

    app = DefectInspectionApp(model_path)
    overlay, mask, raw_hm, summary_md = app.inspect(dummy_surface_image, custom_threshold=0.20)

    assert overlay is not None
    assert mask is not None
    assert raw_hm is not None
    assert "Verdict:" in summary_md
    assert overlay.shape[:2] == raw_hm.shape[:2]


def test_ensure_model_artifact_resolution():
    """Validates local artifact resolution and graceful fallback."""
    # Existing local file returns directly
    local_path = get_default_model_path()
    resolved = ensure_model_artifact(local_path)
    assert resolved == local_path

    # Unreachable path falls back gracefully
    fallback = ensure_model_artifact(Path("non_existent_model.pkl"))
    assert fallback is not None
