
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "metal_nut"
OUTPUT_DIR = PROJECT_ROOT / "outputs"



def load_rgb(path):
    """Load an image using OpenCV and convert BGR to RGB."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mask(path):
    """Load a ground-truth mask as a grayscale image."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Could not read mask: {path}")

    return mask


def main():
    # One normal image from the official training set.
    normal_path = DATASET_DIR / "train" / "good" / "000.png"

    # One defective test image and its ground-truth mask.
    defect_path = DATASET_DIR / "test" / "scratch" / "000.png"
    mask_path = DATASET_DIR / "ground_truth" / "scratch" / "000_mask.png"

    # Confirm these filenames exist before proceeding.
    for path in (normal_path, defect_path, mask_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Expected example not found: {path}\n"
                "Check filenames in the relevant folders and update "
                "the paths in this script."
            )

    normal = load_rgb(normal_path)
    defect = load_rgb(defect_path)
    mask = load_mask(mask_path)

    print("Normal image shape:", normal.shape)
    print("Defective image shape:", defect.shape)
    print("Ground-truth mask shape:", mask.shape)
    print("Mask unique pixel values:", np.unique(mask))

    if defect.shape[:2] != mask.shape:
        raise ValueError(
            "The defect image and ground-truth mask have "
            "different spatial dimensions."
        )

    # Display the three images side by side.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(normal)
    axes[0].set_title("Normal training image")

    axes[1].imshow(defect)
    axes[1].set_title("Defective test image")

    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("Ground-truth defect mask")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "dataset_visual_inspection.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()
