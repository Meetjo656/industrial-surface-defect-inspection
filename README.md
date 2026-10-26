---
title: Industrial Surface Defect Inspection
emoji: 🏭
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: mit
tags:
  - anomaly-detection
  - computer-vision
  - industrial-inspection
---

# Industrial Surface Defect Inspection

An end-to-end computer vision pipeline for automated surface defect detection and localization on industrial components (MVTec AD `metal_nut` dataset).

## Directory Structure

```text
industrial-surface-defect-inspection/
│
├── data/
│   └── metal_nut/             # Downloaded dataset
│       ├── train/
│       │   └── good/
│       ├── test/
│       │   ├── good/
│       │   └── ...
│       └── ground_truth/
│           └── ...
│
├── notebooks/
│   └── 01_dataset_exploration.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_validation.py
│   ├── preprocessing.py
│   ├── classification.py
│   └── localization.py
│
├── outputs/
├── requirements.txt
└── README.md
```

## Setup & Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Workflow

1. **Exploration**: Run `notebooks/01_dataset_exploration.ipynb` to analyze defect types and ground truth masks.
2. **Validation**: Run `python src/data_validation.py` to check dataset integrity.
3. **Classification & Localization**: Train and infer using modules under `src/`.
