"""
Wiener Linien Fleet Spotter — Model Evaluation
===============================================
Runs the trained model on the test split and prints / saves a full report
including per-class AP, confusion matrix, and F1 curve.

Usage:
    python scripts/evaluate.py [--weights model/runs/fleet_spotter_baseline/weights/best.pt]
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


ROOT      = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "dataset.yaml"
DEFAULT_W = ROOT / "model" / "runs" / "fleet_spotter_baseline" / "weights" / "best.pt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=str(DEFAULT_W))
    p.add_argument("--imgsz",  type=int, default=640)
    p.add_argument("--conf",   type=float, default=0.25)
    p.add_argument("--iou",    type=float, default=0.6)
    return p.parse_args()


def main():
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        print(f"Error: weights not found: {weights}")
        print("   Train the model first:  python scripts/train.py")
        return

    model = YOLO(str(weights))

    print(f"\n{'='*60}")
    print(f"  Fleet Spotter — Evaluation on Test Set")
    print(f"{'='*60}")

    metrics = model.val(
        data   = str(DATA_YAML),
        split  = "test",
        imgsz  = args.imgsz,
        conf   = args.conf,
        iou    = args.iou,
        plots  = True,
        verbose= True,
    )

    # ── Summary table ────────────────────────────────────────────────────────
    class_names = list(metrics.names.values())
    print(f"\n{'─'*60}")
    print(f"  {'Class':<18} {'AP50':>8} {'AP50-95':>10} {'Precision':>10} {'Recall':>8}")
    print(f"{'─'*60}")
    for i, name in enumerate(class_names):
        ap50    = metrics.box.ap50[i]    if hasattr(metrics.box, 'ap50')    else 0
        ap5095  = metrics.box.ap[i]      if hasattr(metrics.box, 'ap')      else 0
        prec    = metrics.box.p[i]       if hasattr(metrics.box, 'p')       else 0
        rec     = metrics.box.r[i]       if hasattr(metrics.box, 'r')       else 0
        print(f"  {name:<18} {ap50:>8.4f} {ap5095:>10.4f} {prec:>10.4f} {rec:>8.4f}")

    mAP50   = metrics.box.map50
    mAP5095 = metrics.box.map
    f1      = metrics.box.f1.mean() if hasattr(metrics.box, 'f1') else 0

    print(f"{'─'*60}")
    print(f"  {'mAP50':<18} {mAP50:>8.4f}")
    print(f"  {'mAP50-95':<18} {mAP5095:>8.4f}")
    print(f"  {'mean F1':<18} {f1:>8.4f}")
    print(f"{'='*60}\n")

    # Save summary to text file
    report_path = weights.parent.parent / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Wiener Linien Fleet Spotter — Evaluation Report\n")
        f.write(f"Weights: {weights}\n\n")
        f.write(f"mAP50:    {mAP50:.4f}\n")
        f.write(f"mAP50-95: {mAP5095:.4f}\n")
        f.write(f"mean F1:  {f1:.4f}\n\n")
        f.write(f"Per-class AP50:\n")
        for i, name in enumerate(class_names):
            ap50 = metrics.box.ap50[i] if hasattr(metrics.box, 'ap50') else 0
            f.write(f"  {name}: {ap50:.4f}\n")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
