# 🚋 Wiener Linien Fleet Spotter

> **KI-gestützte Erkennung und Klassifizierung von Wiener Straßenbahnen & U-Bahnen**
>
> Universitätskurs „Machine Learning" · Abgabe 18. Juni 2026

---

## Projektübersicht

Der **Wiener Linien Fleet Spotter** ist ein prototypisches Computer-Vision-System, das Fahrzeuge des Wiener öffentlichen Nahverkehrs nicht nur erkennt, sondern bis auf die spezifische **Baureihe und Fahrzeuggeneration** klassifiziert. Das Modell basiert auf **YOLOv8 mit Transfer Learning** und wird auf einem eigens annotierten Datensatz von Wiener Linien-Fahrzeugen fine-getuned.

### Zielklassen

| Klasse | Typ | Beschreibung |
|---|---|---|
| `E2-Tram` | 🚃 Straßenbahn | Hochflur-Klassiker (Bj. 1978–1990), eckiges Design, oft mit Beiwagen |
| `ULF` | 🚋 Straßenbahn | Ultra Low Floor, charakteristische runde graue Front (ab 2000) |
| `Flexity` | 🚊 Straßenbahn | Modernes Niederflurfahrzeug mit LED-Anzeigen (ab 2018) |
| `Silberpfeil` | 🚇 U-Bahn | Unlackierte Aluminium-Optik, kantige Front (ab 1978) |
| `V-Wagen` | 🚄 U-Bahn | Rot-weiß, durchgängig begehbar (ab 2000) |
| `X-Wagen` | 🤖 U-Bahn | Vollautomatisch, L-förmige LED-Signatur (ab 2024) |

---

## Schnellstart

### Voraussetzungen

- Python 3.10+
- pip
- *(Optional für GPU-Training: CUDA 11.8+)*

### Installation

```bash
# 1. Repository klonen
git clone https://github.com/<your-org>/wiener-linien-fleet-spotter.git
cd wiener-linien-fleet-spotter

# 2. Virtuelle Umgebung erstellen (empfohlen)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Abhängigkeiten installieren
pip install -r requirements.txt
```

### Demo-App starten

```bash
streamlit run app/app.py
```

Browser öffnet sich automatisch auf `http://localhost:8501`.
Ein Bild hochladen → Fahrzeuge werden erkannt und klassifiziert.

> **Hinweis:** Ohne trainiertes Modell läuft die App im *Demo-Modus* mit einem allgemeinen YOLOv8-Modell. Die Wiener-Linien-Klassen werden erst nach dem Training erkannt.

---

## Vollständiger Workflow

### 1 · Datensatz vorbereiten

Bilder werden mit [Roboflow](https://roboflow.com) oder [CVAT](https://cvat.org) annotiert (Bounding Boxes + Klassen-Labels im YOLO-Format).

Verzeichnisstruktur nach der Annotation:

```
data/
├── annotated/
│   ├── train/
│   │   ├── images/   # *.jpg, *.png
│   │   └── labels/   # *.txt (YOLO format)
│   ├── val/
│   └── test/
└── dataset.yaml      # Klassen-Konfiguration
```

### 2 · Augmentierung (optional, empfohlen bei wenig Daten)

```bash
python scripts/augment.py --factor 3
# → verdreifacht den Trainingsdatensatz durch Helligkeits-,
#   Rotations-, Regen- und Nebeleffekte
```

### 3 · Modell trainieren

```bash
# Schnell (CPU / wenig VRAM): yolov8n
python scripts/train.py --model yolov8n.pt --epochs 50

# Besser (GPU empfohlen): yolov8s oder yolov8m
python scripts/train.py --model yolov8s.pt --epochs 100 --batch 32
```

Trainings-Artefakte (Gewichte, Metriken, Plots) landen in:
`model/runs/fleet_spotter_v1/`

### 4 · Evaluierung

```bash
python scripts/evaluate.py
# → druckt per-Klasse AP50, mAP50-95, F1-Score
# → speichert model/runs/fleet_spotter_v1/evaluation_report.txt
```

### 5 · EDA & visuelle Inspektion

```bash
jupyter notebook notebooks/eda_annotation_check.ipynb
```

---

## Projektstruktur

```
wiener-linien-fleet-spotter/
├── app/
│   └── app.py                  # Streamlit Web-UI
├── data/
│   ├── annotated/              # Annotierter Datensatz (gitignored)
│   ├── augmented/              # Augmentierte Bilder (gitignored)
│   └── dataset.yaml            # YOLO-Datensatz-Konfiguration
├── model/
│   └── runs/                   # Trainings-Output (gitignored)
├── notebooks/
│   └── eda_annotation_check.ipynb
├── scripts/
│   ├── train.py                # Training
│   ├── evaluate.py             # Evaluierung
│   └── augment.py              # Offline-Augmentierung
├── assets/                     # Bilder, Logos für Dokumentation
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Technischer Ansatz

| Aspekt | Details |
|---|---|
| **Backbone** | YOLOv8n / YOLOv8s (via `ultralytics`) |
| **Methode** | Transfer Learning + Fine-Tuning |
| **Augmentierung** | Helligkeit, Kontrast, Regen, Nebel, Schatten, horizontaler Flip, leichte Rotation |
| **Metriken** | mAP50, mAP50-95, Precision, Recall, F1 |
| **UI** | Streamlit – Upload → Echtzeit-Inferenz mit Bounding Boxes |

---

## Team

Vierergruppe, Universitätskurs „Machine Learning", Wintersemester 2025/26

---

## Lizenz

MIT License – Bildmaterial nur für akademische Zwecke, keine kommerzielle Nutzung.
