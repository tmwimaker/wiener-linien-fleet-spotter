"""
Wiener Linien Fleet Spotter — Streamlit Web App
================================================
Interactive demo: upload a photo → YOLO detects & classifies the vehicle.

Run:
    streamlit run app/app.py
"""

import csv
import io
import re
from datetime import datetime
from pathlib import Path

import yaml

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch

ROOT = Path(__file__).resolve().parent.parent

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wiener Linien Fleet Spotter",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Class metadata ────────────────────────────────────────────────────────────
CLASS_INFO = {
    "E2-Tram": {
        "color": "#E63946",
        "category": "Straßenbahn",
        "desc": "Hochflur-Straßenbahn, Bj. 1978–1990. Eckiges, traditionelles Design – oft mit Beiwagen unterwegs.",
        "fun_fact": "Die E2 ist der Klassiker der Wiener Bim und wird als fahrendes Denkmal geschätzt.",
    },
    "ULF": {
        "color": "#457B9D",
        "category": "Straßenbahn",
        "desc": "Ultra Low Floor Tram, ab 2000. Unverwechselbare abgerundete graue Frontpartie.",
        "fun_fact": "Der ULF war beim Launch eines der niedrigflursten Fahrzeuge weltweit – ideal für barrierefreies Einsteigen.",
    },
    "Flexity": {
        "color": "#2A9D8F",
        "category": "Straßenbahn",
        "desc": "Flexity Wien, ab 2018. Futuristisches Design mit markanten LED-Zielanzeigen.",
        "fun_fact": "Der Flexity ersetzt schrittweise die alten E2-Garnituren und bietet WLAN und USB-Ladebuchsen.",
    },
    "Silberpfeil": {
        "color": "#6C757D",
        "category": "U-Bahn",
        "desc": "U-Bahn Reihe U, unlackierte Aluminium-Optik, kantige Front. Ein Stück Wiener Technikgeschichte.",
        "fun_fact": "Der Silberpfeil ist seit 1978 im Einsatz – und war in den 1970ern hochmodern.",
    },
    "V-Wagen": {
        "color": "#C1121F",
        "category": "U-Bahn",
        "desc": "U-Bahn Reihe V, ab 2000. Rot-weiße Front, vollständig durchgängig begehbar.",
        "fun_fact": "Die V-Wagen sind die häufigsten U-Bahn-Fahrzeuge in Wien und decken fast alle Linien ab.",
    },
    "X-Wagen": {
        "color": "#7209B7",
        "category": "U-Bahn",
        "desc": "U-Bahn Reihe X, ab 2024. Vollautomatisch, fahrerlos, mit markanter L-förmiger LED-Scheinwerfersignatur.",
        "fun_fact": "Der X-Wagen ist Wiens erster vollautomatischer U-Bahn-Zug und wird zunächst auf der U5 eingesetzt.",
    },
}

# ── CSS Styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }
  h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
  }
  .hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #E63946;
    letter-spacing: -1px;
    margin-bottom: 0;
  }
  .hero-subtitle {
    font-size: 1.05rem;
    color: #6c757d;
    margin-top: 4px;
  }
  .detection-card {
    background: #f8f9fa;
    border-left: 4px solid #E63946;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
  }
  .class-badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    color: white;
    margin-bottom: 6px;
  }
  .conf-bar-bg {
    background: #e9ecef;
    border-radius: 6px;
    height: 8px;
    margin-top: 6px;
  }
  .stButton button {
    background-color: #E63946 !important;
    color: white !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.5rem 1.5rem !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Trained-run discovery ─────────────────────────────────────────────────────
RUNS_DIR = ROOT / "model" / "runs"


def list_trained_runs():
    """Available training runs with saved weights, newest first."""
    return sorted(
        RUNS_DIR.glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def completed_epochs(run_dir):
    """Number of epochs actually completed, from results.csv (None if unavailable)."""
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return None
    with open(results_csv) as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return None
    return int(float(rows[-1][0]))


def run_label(weights_path):
    run_dir = weights_path.parent.parent
    args_file = run_dir / "args.yaml"
    run_name = run_dir.name
    date_str = datetime.fromtimestamp(weights_path.stat().st_mtime).strftime("%d.%m.%Y")
    if not args_file.exists():
        return f"{run_name}  ·  {date_str}"
    with open(args_file) as f:
        a = yaml.safe_load(f)
    model = str(a.get("model", "?")).replace(".pt", "")
    configured_epochs = a.get("epochs", "?")
    done = completed_epochs(run_dir)
    epochs_label = f"{done}/{configured_epochs}" if done is not None and done != configured_epochs else configured_epochs
    batch = a.get("batch", "?")
    return f"{run_name}  ·  {model}, {epochs_label} ep, batch {batch}  ·  {date_str}"


def run_image_count(run_dir):
    """Total dataset images encoded in the run name (…-<N>img), or None if absent.

    The dataset size is part of the run name (naming convention
    <label>-<epochs>ep-<images>img), so it travels with the run on any machine —
    no git history or file-mtime guessing needed.
    """
    m = re.search(r"-(\d+)img$", run_dir.name)
    return int(m.group(1)) if m else None


# ── Grad-CAM detail-level options ─────────────────────────────────────────────
# Feature maps feeding the Detect head (strides 8/16/32 → P3/P4/P5).
CAM_LEVEL_OPTIONS = {
    "Kombiniert (P3+P4+P5)": (0, 1, 2),
    "Fein (P3, 80×80)": (0,),
    "Mittel (P4, 40×40)": (1,),
    "Grob (P5, 20×20)": (2,),
}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Einstellungen")

    trained_runs = list_trained_runs()

    runs_info = []
    for w in trained_runs:
        run_dir = w.parent.parent
        args_file = run_dir / "args.yaml"
        configured_epochs = 0
        if args_file.exists():
            with open(args_file) as f:
                configured_epochs = yaml.safe_load(f).get("epochs", 0)
        runs_info.append({
            "weights": w,
            "label": run_label(w),
            "epochs": configured_epochs,
            "total_imgs": run_image_count(run_dir),   # read straight from the run name
        })

    # Order by dataset size (encoded in the run name): current dataset first, then
    # by training length. The size is already part of each run name (…-<N>img), so
    # the option labels lead with the run name instead of repeating "N Bilder".
    sizes = sorted({r["total_imgs"] for r in runs_info if r["total_imgs"] is not None}, reverse=True)
    latest_imgs = sizes[0] if sizes else None

    def _is_latest(r):
        return latest_imgs is not None and r["total_imgs"] == latest_imgs

    latest_runs = sorted((r for r in runs_info if _is_latest(r)), key=lambda r: -r["epochs"])
    older_runs = sorted((r for r in runs_info if not _is_latest(r)), key=lambda r: -r["epochs"])

    run_choices = {}
    for r in latest_runs:
        run_choices[r["label"]] = r["weights"]
    run_choices["Demo (Basis-YOLOv8n, untrainiert)"] = None
    for r in older_runs:
        run_choices[r["label"]] = r["weights"]

    selected_run = st.selectbox(
        "Trainings-Run", list(run_choices.keys()),
        index=0,
        help="Wähle, welches trainierte Modell für die Erkennung verwendet werden soll.",
    )
    weights_path = run_choices[selected_run]

    st.divider()
    conf_thresh = st.slider("Konfidenz-Schwellwert", 0.1, 0.95, 0.30, 0.05,
                            help="Mindestvertrauen für eine Erkennung")
    iou_thresh = st.slider("IoU-Schwellwert (NMS)", 0.1, 0.9, 0.45, 0.05,
                           help="Non-Maximum Suppression Overlap-Grenze")
    show_labels = st.checkbox("Labels auf Bild anzeigen", value=True)
    show_conf = st.checkbox("Konfidenz anzeigen", value=True)
    show_cam = st.checkbox("Erklärbarkeits-Heatmap anzeigen", value=False,
                           help="Visualisiert per Grad-CAM, welche Bildbereiche eines "
                                "erkannten Fahrzeugs am stärksten für seine vorhergesagte "
                                "Klasse sprechen.")
    cam_alpha = st.slider("Heatmap-Intensität", 0.1, 0.9, 0.45, 0.05,
                          help="Deckkraft der Aktivierungs-Überlagerung",
                          disabled=not show_cam)
    cam_level_label = st.selectbox(
        "Heatmap-Detailgrad", list(CAM_LEVEL_OPTIONS.keys()),
        help="Welche Feature-Ebene(n) für die Heatmap verwendet werden. "
             "Feinere Ebenen liefern kleinteiligere, schärfere Hotspots; "
             "gröbere zeigen großflächigere Bereiche.",
        disabled=not show_cam,
    )
    cam_levels = CAM_LEVEL_OPTIONS[cam_level_label]

    st.divider()
    for name, info in CLASS_INFO.items():
        badge_color = info.get("color", "#333")
        badge_cat   = info.get("category", "")
        badge_desc  = info.get("desc", "")
        st.markdown(
            f"<span class='class-badge' style='background:{badge_color}'>"
            f"{name}</span>&nbsp;"
            f"<small style='color:#6c757d'>{badge_cat}</small><br>"
            f"<small>{badge_desc}</small>",
            unsafe_allow_html=True,
        )
        st.markdown("")

    st.divider()
    st.caption("Wiener Linien Fleet Spotter · ML-Kurs 2026")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<p class='hero-title'>Wiener Linien Fleet Spotter</p>", unsafe_allow_html=True)
st.markdown(
    "<p class='hero-subtitle'>Lade ein Foto hoch – das KI-Modell erkennt und klassifiziert "
    "Straßenbahnen & U-Bahnen der Wiener Linien.</p>",
    unsafe_allow_html=True,
)
st.divider()


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Modell wird geladen …")
def load_model(weights_path):
    from ultralytics import YOLO
    if weights_path is None:
        return YOLO("yolov8n.pt"), False   # demo mode
    return YOLO(str(weights_path)), True


model, is_trained = load_model(weights_path)


# ── Explainability (Grad-CAM) ─────────────────────────────────────────────────
# Feature maps feeding the Detect head (strides 8/16/32 → P3/P4/P5).
_GRADCAM_LAYERS = (15, 18, 21)

# How far the heatmap is allowed to reach beyond each detection box, as a
# fraction of the box's longer side. The gate is feathered (not a hard cut-off),
# so a margin of surrounding context stays visible and fades out smoothly.
CAM_CONTEXT_MARGIN = 0.18


def compute_gradcam(yolo_model, img_rgb, boxes_xyxy, class_ids, imgsz=640, levels=(0, 1, 2)):
    """Computes Grad-CAM heatmaps for each detected vehicle.

    For every detection, the predicted class score is backpropagated to the
    feature maps feeding the detection head. The resulting gradients show
    which pixels actually *increased the model's confidence in that vehicle's
    predicted class* — i.e. the real evidence behind the classification, not
    just generic feature activity. `levels` selects which of the three
    feature maps (0=P3/fine, 1=P4/medium, 2=P5/coarse) contribute. Each
    vehicle's map is restricted to its own bounding box; multiple vehicles
    are combined via a per-pixel maximum. Returns a smoothed map in [0, 1] at
    image resolution, or None if there is nothing to explain.
    """
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return None

    img_h, img_w = img_rgb.shape[:2]
    torch_model = yolo_model.model
    torch_model.eval()

    img_resized = cv2.resize(img_rgb, (imgsz, imgsz))
    tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor = tensor.to(next(torch_model.parameters()).device)
    tensor.requires_grad_(True)

    captured = {}

    def make_hook(level):
        def hook(_module, _inp, out):
            out.retain_grad()
            captured[level] = out
        return hook

    handles = [torch_model.model[layer_idx].register_forward_hook(make_hook(level))
               for level, layer_idx in enumerate(_GRADCAM_LAYERS)]

    # Anchors/strides cached from a prior inference-mode predict() call can't
    # be reused inside a graph that needs gradients - force a clean recompute.
    torch_model.model[-1].shape = None

    try:
        with torch.enable_grad():
            y, _ = torch_model(tensor)
    finally:
        for handle in handles:
            handle.remove()

    scale_x, scale_y = imgsz / img_w, imgsz / img_h
    anchor_cx, anchor_cy = y[0, 0, :], y[0, 1, :]

    cam = np.zeros((img_h, img_w), dtype=np.float32)

    for (x1, y1, x2, y2), cls_id in zip(boxes_xyxy, class_ids):
        bx1, by1, bx2, by2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y
        mask = (anchor_cx >= bx1) & (anchor_cx <= bx2) & (anchor_cy >= by1) & (anchor_cy <= by2)
        if not mask.any():
            continue

        score = y[0, 4 + int(cls_id), :][mask].sum()
        for t in captured.values():
            t.grad = None
        torch_model.zero_grad()
        score.backward(retain_graph=True)

        box_cam = np.zeros((img_h, img_w), dtype=np.float32)
        for level, t in captured.items():
            if level not in levels:
                continue
            act = t.detach()[0].numpy()
            grad = t.grad[0].numpy()
            weights = grad.mean(axis=(1, 2))
            level_cam = np.maximum((weights[:, None, None] * act).sum(axis=0), 0)
            level_cam -= level_cam.min()
            denom = level_cam.max()
            if denom > 1e-8:
                level_cam /= denom
            level_cam = cv2.resize(level_cam, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
            box_cam = np.maximum(box_cam, level_cam)

        # Soft, slightly-dilated gate around the box: rather than a hard cut-off
        # at the box edge, extend the mask by a margin and feather it, so a bit of
        # the surrounding context (rails, catenary, platform) stays visible and
        # fades out — instead of an abrupt rectangular boundary.
        bw, bh = x2 - x1, y2 - y1
        margin = max(CAM_CONTEXT_MARGIN * max(bw, bh), 6.0)
        ex1 = int(np.clip(x1 - margin, 0, img_w - 1))
        ey1 = int(np.clip(y1 - margin, 0, img_h - 1))
        ex2 = int(np.clip(round(x2 + margin), ex1 + 1, img_w))
        ey2 = int(np.clip(round(y2 + margin), ey1 + 1, img_h))
        box_mask = np.zeros((img_h, img_w), dtype=np.float32)
        box_mask[ey1:ey2, ex1:ex2] = 1.0
        box_mask = cv2.GaussianBlur(box_mask, (0, 0), sigmaX=margin * 0.5)
        box_cam *= box_mask
        if box_cam.max() > 1e-8:
            box_cam /= box_cam.max()

        cam = np.maximum(cam, box_cam)

    cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=max(img_w, img_h) / 200)
    return np.clip(cam, 0, 1)


def render_cam(img_rgb, cam, alpha=0.45, highlight_quantile=0.85):
    """Turns a raw CAM into a colorized activation map and an overlay with a
    contour drawn around the strongest-activation region for extra punch."""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_TURBO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_rgb, 1 - alpha, heatmap, alpha, 0)

    hot_mask = (cam >= np.quantile(cam, highlight_quantile)).astype(np.uint8)
    contours, _ = cv2.findContours(hot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

    return heatmap, overlay


def colorbar_legend(width=400, height=22):
    """Horizontal gradient bar matching the CAM colormap, used as a legend."""
    gradient = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    bar = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(bar, cv2.COLOR_BGR2RGB)

if not is_trained:
    st.warning(
        "**Demo-Modus**: Kein trainiertes Modell gefunden. "
        "Das Basis-YOLOv8n-Modell ist aktiv — Wiener-Linien-Klassen werden erst nach "
        "dem Training erkannt. Führe `python scripts/train.py` aus, um das Modell zu trainieren."
    )


# ── Upload & Detection ────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Bild hochladen (JPG, PNG)", type=["jpg", "jpeg", "png"],
    help="Lade ein Foto einer Wiener Straßenbahn oder U-Bahn hoch."
)

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    img_np  = np.array(pil_img)

    col_orig, col_result = st.columns(2, gap="large")

    with col_orig:
        st.markdown("#### Original")
        st.image(pil_img, use_column_width=True)

    with st.spinner("Erkenne Fahrzeuge …"):
        results = model.predict(
            source     = img_np,
            conf       = conf_thresh,
            iou        = iou_thresh,
            verbose    = False,
        )

    result  = results[0]
    boxes   = result.boxes
    ann_img = result.plot(labels=show_labels, conf=show_conf)  # BGR numpy
    ann_img = cv2.cvtColor(ann_img, cv2.COLOR_BGR2RGB)

    with col_result:
        st.markdown("#### Erkannte Fahrzeuge")
        st.image(ann_img, use_column_width=True)

    if show_cam:
        if boxes is not None and len(boxes):
            box_xyxy  = boxes.xyxy.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
        else:
            box_xyxy, class_ids = None, None

        with st.spinner("Berechne Erklärbarkeits-Heatmap …"):
            cam = compute_gradcam(model, img_np, box_xyxy, class_ids, levels=cam_levels)
        if cam is not None:
            heatmap_img, overlay_img = render_cam(img_np, cam, alpha=cam_alpha)

            st.divider()
            st.markdown("### Warum hat das Modell so entschieden?")

            cam_col1, cam_col2 = st.columns(2, gap="large")
            with cam_col1:
                st.image(overlay_img, use_column_width=True, caption="Entscheidungs-Overlay")
            with cam_col2:
                st.image(heatmap_img, use_column_width=True, caption="Reine Heatmap")

            _, legend_col, _ = st.columns([1, 2, 1])
            with legend_col:
                st.image(colorbar_legend(), use_column_width=True)
                label_a, label_b = st.columns(2)
                label_a.markdown(
                    "<span style='font-size:0.8rem;color:#6c757d'>weniger relevant</span>",
                    unsafe_allow_html=True,
                )
                label_b.markdown(
                    "<span style='font-size:0.8rem;color:#6c757d;float:right'>stark relevant</span>",
                    unsafe_allow_html=True,
                )

            st.caption(
                "Die Heatmap (Grad-CAM) zeigt, welche Bereiche am stärksten für die jeweils "
                "vorhergesagte Klasse gesprochen haben — also die tatsächliche Evidenz hinter "
                "der Entscheidung, nicht nur generische Bildaktivität. Der Fokus liegt auf dem "
                "erkannten Fahrzeug, bezieht aber einen weich auslaufenden Rand des umgebenden "
                "Kontexts (z. B. Gleise, Oberleitung) mit ein. Die weiß umrandete Kontur "
                "markiert die 15 % einflussreichsten Bereiche."
            )
        elif box_xyxy is not None:
            st.info(
                "Für die Heatmap konnte keine Erklärung berechnet werden "
                "(z. B. weil die erkannten Boxen außerhalb des Modellrasters liegen)."
            )

    st.divider()
    st.markdown("### Erkennungs-Details")

    if boxes is None or len(boxes) == 0:
        st.info("Keine Fahrzeuge erkannt. Probiere einen niedrigeren Konfidenz-Schwellwert.")
    else:
        det_cols = st.columns(min(len(boxes), 3))
        for i, box in enumerate(boxes):
            cls_id   = int(box.cls[0])
            conf_val = float(box.conf[0])
            cls_name = result.names[cls_id] if cls_id < len(result.names) else f"Klasse {cls_id}"
            info     = CLASS_INFO.get(cls_name, {})
            color    = info.get("color", "#333")

            with det_cols[i % 3]:
                bar_w = int(conf_val * 100)
                st.markdown(f"""
                <div class="detection-card">
                  <span class="class-badge" style="background:{color}">{cls_name}</span>
                  <div style="font-size:1.4rem;font-weight:700;font-family:'Space Mono',monospace">
                    {conf_val:.1%}
                  </div>
                  <div class="conf-bar-bg">
                    <div style="width:{bar_w}%;background:{color};height:8px;border-radius:6px"></div>
                  </div>
                  <p style="margin-top:8px;font-size:0.85rem;color:#555">
                    {info.get('desc','') }
                  </p>
                  <p style="font-size:0.8rem;color:#888;font-style:italic">
                    Wissenswert: {info.get('fun_fact','')}
                  </p>
                </div>
                """, unsafe_allow_html=True)

        # Download annotated image
        st.divider()
        buf = io.BytesIO()
        Image.fromarray(ann_img).save(buf, format="JPEG", quality=95)
        st.download_button(
            label    = "Annotiertes Bild herunterladen",
            data     = buf.getvalue(),
            file_name= "fleet_spotter_result.jpg",
            mime     = "image/jpeg",
        )

else:
    # Placeholder / instructions
    st.markdown("""
    <div style="text-align:center;padding:3rem;border:2px dashed #dee2e6;border-radius:12px;color:#adb5bd">
      <p style="font-size:1.1rem;margin-top:0">
        Bild hochladen, um die Erkennung zu starten
      </p>
      <p style="font-size:0.85rem">
        Unterstützte Klassen: E2-Tram · ULF · Flexity · Silberpfeil · V-Wagen · X-Wagen
      </p>
    </div>
    """, unsafe_allow_html=True)
