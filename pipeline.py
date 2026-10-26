"""Root entry point for ZenML defect inspection pipeline."""

from src.pipeline import defect_inspection_pipeline, run_pipeline

if __name__ == "__main__":
    run_pipeline()
