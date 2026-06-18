#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ ! -f venv/bin/python ]; then
    echo "[Fehler] Keine virtuelle Umgebung gefunden - bitte zuerst ./run_app.sh ausfuehren."
    exit 1
fi

echo "============================================"
echo "  Wiener Linien Fleet Spotter - Training"
echo "============================================"
echo ""
echo "  [1] Schnelltest  - yolov8n,  5 Epochen   (schneller Funktionscheck, CPU ok)"
echo "  [2] Standard     - yolov8n, 50 Epochen   (CPU / wenig VRAM)"
echo "  [3] Intensiv     - yolov8s, 100 Epochen, Batch 32  (GPU empfohlen)"
echo ""
read -p "Modus waehlen (1-3): " mode

case "$mode" in
    1) MODEL="yolov8n.pt"; EPOCHS=5;   BATCH=16; LABEL="quick"     ;;
    2) MODEL="yolov8n.pt"; EPOCHS=50;  BATCH=16; LABEL="standard"  ;;
    3) MODEL="yolov8s.pt"; EPOCHS=100; BATCH=32; LABEL="intensive" ;;
    *)
        echo "Ungueltige Auswahl: $mode"
        exit 1
        ;;
esac

echo ""
echo "Starte Training: $MODEL, $EPOCHS Epochen, Batch $BATCH ..."
echo "(Run-Name wird zu <label>-<epochs>ep-<bilder>img, z. B. ${LABEL}-${EPOCHS}ep-NNNimg)"
echo ""
venv/bin/python scripts/train.py --model "$MODEL" --epochs "$EPOCHS" --batch "$BATCH" --label "$LABEL"
