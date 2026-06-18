"""
Wiener Linien Fleet Spotter — Model Training
============================================
Trains a YOLOv8 model via Transfer Learning / Fine-Tuning on the
Wiener Linien vehicle dataset.

Usage:
    python scripts/train.py [--model yolov8n.pt] [--epochs 50] [--imgsz 640]
"""

import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO


# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_YAML   = ROOT / "data" / "dataset.yaml"
RUNS_DIR    = ROOT / "model" / "runs"
DATA_ROOT   = ROOT / "data" / "annotated"
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_dataset_images():
    """Total annotated images currently on disk across the train/valid/test splits."""
    return sum(
        1
        for split in ("train", "valid", "test")
        for p in (DATA_ROOT / split / "images").glob("*")
        if p.suffix.lower() in IMG_EXTS
    )


def build_run_name(label, epochs):
    """Convention: <label>-<epochs>ep-<images>img, e.g. quick-5ep-541img."""
    return f"{label}-{epochs}ep-{count_dataset_images()}img"


# ── Argument parsing ─────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train the Fleet Spotter YOLO model")
    parser.add_argument("--model",   default="yolov8n.pt",
                        help="Base YOLO model (e.g. yolov8n.pt / yolov8s.pt / yolov8m.pt)")
    parser.add_argument("--epochs",  type=int, default=50)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--batch",   type=int, default=16)
    parser.add_argument("--device",  default="",
                        help="cuda device (0,1,...) or 'cpu' — empty = auto")
    parser.add_argument("--label",   default="run",
                        help="Short run label; full name becomes <label>-<epochs>ep-<images>img")
    parser.add_argument("--name",    default=None,
                        help="Explicit run name, overriding the <label>-<epochs>ep-<images>img convention")
    parser.add_argument("--patience",type=int, default=15,
                        help="Early-stopping patience (epochs without improvement)")
    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    run_name = args.name or build_run_name(args.label, args.epochs)

    # Load dataset config to echo class names
    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)
    print(f"\n{'='*60}")
    print(f"  Wiener Linien Fleet Spotter — Training")
    print(f"{'='*60}")
    print(f"  Classes ({cfg['nc']}): {', '.join(cfg['names'].values())}")
    print(f"  Base model : {args.model}")
    print(f"  Epochs     : {args.epochs}  |  Batch: {args.batch}  |  ImgSz: {args.imgsz}")
    print(f"  Run name   : {run_name}")
    print(f"{'='*60}\n")

    # Load pre-trained YOLO (Transfer Learning base)
    model = YOLO(args.model)

    # Fine-tune on our dataset
    results = model.train(
        data      = str(DATA_YAML),
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        device    = args.device if args.device else None,
        project   = str(RUNS_DIR),
        name      = run_name,
        patience  = args.patience,
        # --- Augmentation (supplements albumentations pipeline) ---
        hsv_h     = 0.015,   # hue shift
        hsv_s     = 0.7,     # saturation
        hsv_v     = 0.4,     # brightness
        flipud    = 0.0,     # no vertical flip (trams don't fly)
        fliplr    = 0.5,     # horizontal flip OK
        mosaic    = 1.0,     # mosaic augmentation
        mixup     = 0.1,
        copy_paste= 0.1,
        # --- Logging ---
        plots     = True,
        save      = True,
        verbose   = True,
    )

    # Save best weights path for easy reference
    best = RUNS_DIR / run_name / "weights" / "best.pt"
    print(f"\nTraining complete!")
    print(f"   Best weights: {best}")
    print(f"   mAP50:        {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"   mAP50-95:     {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    print("\n   Run  `python scripts/evaluate.py`  to generate full evaluation report.")


if __name__ == "__main__":
    main()
