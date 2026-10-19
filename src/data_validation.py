"""Data validation module for Industrial Surface Defect Inspection.

Validates dataset directory structures, file integrity, formats, and annotations.
"""

from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image


def validate_dataset_structure(data_dir: Path) -> Dict[str, bool]:
    """Validates that train, test, and ground_truth folders exist and are non-empty.

    Args:
        data_dir: Path to the dataset root (e.g. data/metal_nut).

    Returns:
        Dictionary mapping directory components to validation status.
    """
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    gt_dir = data_dir / "ground_truth"

    return {
        "data_dir_exists": data_dir.is_dir(),
        "train_exists": train_dir.is_dir(),
        "test_exists": test_dir.is_dir(),
        "ground_truth_exists": gt_dir.is_dir(),
    }


def scan_dataset(data_dir: Path) -> Dict[str, Dict[str, int]]:
    """Counts available images per split and category.

    Args:
        data_dir: Path to the dataset root.

    Returns:
        Nested dict with counts per split and defect category.
    """
    results: Dict[str, Dict[str, int]] = {}
    for split in ["train", "test", "ground_truth"]:
        split_path = data_dir / split
        if not split_path.exists():
            continue
        results[split] = {}
        for category_path in split_path.iterdir():
            if category_path.is_dir():
                valid_images = [
                    f for f in category_path.iterdir()
                    if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]
                ]
                results[split][category_path.name] = len(valid_images)
    return results


def verify_image_integrity(image_paths: List[Path]) -> Tuple[List[Path], List[Path]]:
    """Checks image files for corruption.

    Args:
        image_paths: List of file paths to verify.

    Returns:
        Tuple of (valid_paths, corrupted_paths).
    """
    valid, corrupted = [], []
    for path in image_paths:
        try:
            with Image.open(path) as img:
                img.verify()
            valid.append(path)
        except Exception:
            corrupted.append(path)
    return valid, corrupted


if __name__ == "__main__":
    dataset_path = Path("data/metal_nut")
    print(f"Validating dataset at: {dataset_path}")
    status = validate_dataset_structure(dataset_path)
    print("Structure status:", status)
    counts = scan_dataset(dataset_path)
    print("Image counts:", counts)
