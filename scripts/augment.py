"""
Wiener Linien Fleet Spotter — Data Augmentation Pipeline
=========================================================
Applies offline augmentation to the training images to combat class imbalance
and increase dataset diversity. Augmented images + YOLO labels are saved
alongside the originals.

Usage:
    python scripts/augment.py [--input data/annotated/train] [--factor 3]
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent


# ── Augmentation pipeline ─────────────────────────────────────────────────────
def build_pipeline():
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.4),
        A.MotionBlur(blur_limit=7, p=0.3),   # simulates moving tram
        A.RandomRain(p=0.2),                  # Viennese weather
        A.RandomFog(p=0.15),
        A.RandomShadow(p=0.3),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.5,
                           border_mode=cv2.BORDER_REFLECT),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"],
                                min_visibility=0.3))


def augment_image(image_path: Path, label_path: Path, pipeline, out_img_dir: Path,
                  out_lbl_dir: Path, suffix: int):
    img = cv2.imread(str(image_path))
    if img is None:
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Parse YOLO labels
    bboxes, class_labels = [], []
    if label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) == 5:
                class_labels.append(int(parts[0]))
                bboxes.append([float(x) for x in parts[1:]])

    try:
        result = pipeline(image=img, bboxes=bboxes, class_labels=class_labels)
    except Exception:
        return

    out_name = f"{image_path.stem}_aug{suffix}"
    out_img = out_img_dir / f"{out_name}.jpg"
    out_lbl = out_lbl_dir / f"{out_name}.txt"

    aug_img = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_img), aug_img)

    with open(out_lbl, "w") as f:
        for cls, bbox in zip(result["class_labels"], result["bboxes"]):
            f.write(f"{cls} {' '.join(f'{v:.6f}' for v in bbox)}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=str(ROOT / "data" / "annotated" / "train"))
    parser.add_argument("--output", default=str(ROOT / "data" / "augmented" / "train"))
    parser.add_argument("--factor", type=int, default=3,
                        help="Number of augmented copies per original image")
    args = parser.parse_args()

    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    img_out = out_dir / "images"
    lbl_out = out_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    # Copy originals
    in_img = in_dir / "images"
    in_lbl = in_dir / "labels"
    for f in in_img.glob("*"):
        shutil.copy(f, img_out / f.name)
    for f in in_lbl.glob("*.txt"):
        shutil.copy(f, lbl_out / f.name)

    images = list(in_img.glob("*.jpg")) + list(in_img.glob("*.png"))
    pipeline = build_pipeline()

    print(f"\nAugmenting {len(images)} images x {args.factor} copies …")
    for img_path in tqdm(images):
        lbl_path = in_lbl / (img_path.stem + ".txt")
        for i in range(args.factor):
            augment_image(img_path, lbl_path, pipeline, img_out, lbl_out, suffix=i)

    total = len(list(img_out.glob("*")))
    print(f"Done — {total} images in {out_dir}")


if __name__ == "__main__":
    main()
