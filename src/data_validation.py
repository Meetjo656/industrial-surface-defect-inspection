"""Data validation module for Industrial Surface Defect Inspection.

Validates dataset directory structures, file integrity, formats, and annotations.
"""

from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import cv2
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "metal_nut"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
def get_images(folder):
    """Return Image files inside a folder, excluding subdirectories."""
    if not folder.exists():
        return []

    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def inspect_image(path, grayscale=False):
    """Read an image and report whether OpenCV can decode it."""
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)

    if image is None:
        return None

    return image


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("METAL_NUT DATASET VALIDATION")
    print("=" * 60)
    print(f"Dataset path: {DATASET_DIR.resolve()}")

    # 1. Validate the essential directory structure.
    required_dirs = [
        DATASET_DIR / "train" / "good",
        DATASET_DIR / "test" / "good",
        DATASET_DIR / "ground_truth",
    ]

    missing_dirs = [str(p) for p in required_dirs if not p.is_dir()]

    if missing_dirs:
        print("\nERROR: Required directories are missing:")
        for folder in missing_dirs:
            print(f"  - {folder}")

        print(
            "\nCheck the dataset path. The downloaded archive may "
            "contain an extra nested metal_nut directory."
        )
        return

    print("\nDirectory structure: OK")

    # 2. Collect image inventory.
    records = []
    unreadable_images = []
    test_counts = {}
    mask_counts = {}

    for split in ("train", "test"):
        split_dir = DATASET_DIR / split

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            images = get_images(class_dir)

            if split == "test":
                test_counts[class_name] = len(images)

            for image_path in images:
                image = inspect_image(image_path)

                readable = image is not None
                if not readable:
                    unreadable_images.append(str(image_path))

                records.append({
                    "split": split,
                    "class": class_name,
                    "filename": image_path.name,
                    "path": str(image_path),
                    "readable": readable,
                    "width": image.shape[1] if readable else None,
                    "height": image.shape[0] if readable else None,
                })

    # 3. Inspect ground-truth mask directories.
    ground_truth_dir = DATASET_DIR / "ground_truth"

    for class_dir in sorted(ground_truth_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        masks = get_images(class_dir)
        mask_counts[class_dir.name] = len(masks)

        for mask_path in masks:
            mask = inspect_image(mask_path, grayscale=True)

            if mask is None:
                unreadable_images.append(str(mask_path))

    # 4. Print image counts.
    inventory = pd.DataFrame(records)

    print("\nTRAIN IMAGE COUNTS")
    if not inventory.empty:
        train_counts = inventory[inventory["split"] == "train"]
        print(train_counts.groupby("class").size().to_string())
    else:
        print("No images found.")

    print("\nTEST IMAGE COUNTS")
    if test_counts:
        for class_name, count in sorted(test_counts.items()):
            print(f"{class_name:15s}: {count}")
    else:
        print("No test class directories found.")

    print("\nGROUND-TRUTH MASK COUNTS")
    if mask_counts:
        for class_name, count in sorted(mask_counts.items()):
            print(f"{class_name:15s}: {count}")
    else:
        print("No mask subdirectories found.")

    # 5. Check that every defective test image has a mask.
    missing_masks = []

    for class_name in test_counts:
        if class_name == "good":
            continue

        image_dir = DATASET_DIR / "test" / class_name
        mask_dir = ground_truth_dir / class_name

        existing_masks = {
            p.stem.removesuffix("_mask")
            for p in get_images(mask_dir)
        }

        for image_path in get_images(image_dir):
            if image_path.stem not in existing_masks:
                missing_masks.append(
                    f"{class_name}/{image_path.name}"
                )

    print("\nVALIDATION SUMMARY")
    print(f"Total indexed images: {len(inventory)}")
    print(f"Unreadable images/masks: {len(unreadable_images)}")
    print(f"Defective images missing masks: {len(missing_masks)}")

    # 6. Save detailed reports.
    inventory.to_csv(OUTPUT_DIR / "dataset_inventory.csv", index=False)

    pd.DataFrame(
        {"unreadable_path": unreadable_images}
    ).to_csv(OUTPUT_DIR / "unreadable_files.csv", index=False)

    pd.DataFrame(
        {"image_without_mask": missing_masks}
    ).to_csv(OUTPUT_DIR / "missing_masks.csv", index=False)

    # 7. Plot image counts.
    if not inventory.empty:
        counts = (
            inventory.groupby(["split", "class"])
            .size()
            .reset_index(name="image_count")
        )

        counts["group"] = counts["split"] + "/" + counts["class"]

        plt.figure(figsize=(10, 5))
        plt.bar(counts["group"], counts["image_count"])
        plt.title("MVTec AD - metal_nut image distribution")
        plt.xlabel("Dataset split / category")
        plt.ylabel("Number of images")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "dataset_distribution.png", dpi=150)
        plt.close()

    # 8. Report the result without hiding any failures.
    passed = (
        len(unreadable_images) == 0
        and len(missing_masks) == 0
        and len(inventory) > 0
    )

    print("\nRESULT:", "PASS" if passed else "CHECK REQUIRED")
    print(f"Reports saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
