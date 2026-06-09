"""
Wiener Linien Fleet Spotter — Streamlit Web App
================================================
Interactive demo: upload a photo → YOLO detects & classifies the vehicle.

Run:
    streamlit run app/app.py
"""

import io
from datetime import datetime
from pathlib import Path

import yaml

import streamlit as st
from PIL import Image
import numpy as np
import cv2

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wiener Linien Fleet Spotter",
    page_icon="🚋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Class metadata ────────────────────────────────────────────────────────────
CLASS_INFO = {
    "E2-Tram": {
        "emoji": "🚃",
        "color": "#E63946",
        "category": "Straßenbahn",
        "desc": "Hochflur-Straßenbahn, Bj. 1978–1990. Eckiges, traditionelles Design – oft mit Beiwagen unterwegs.",
        "fun_fact": "Die E2 ist der Klassiker der Wiener Bim und wird als fahrendes Denkmal geschätzt.",
    },
    "ULF": {
        "emoji": "🚋",
        "color": "#457B9D",
        "category": "Straßenbahn",
        "desc": "Ultra Low Floor Tram, ab 2000. Unverwechselbare abgerundete graue Frontpartie.",
        "fun_fact": "Der ULF war beim Launch eines der niedrigflursten Fahrzeuge weltweit – ideal für barrierefreies Einsteigen.",
    },
    "Flexity": {
        "emoji": "🚊",
        "color": "#2A9D8F",
        "category": "Straßenbahn",
        "desc": "Flexity Wien, ab 2018. Futuristisches Design mit markanten LED-Zielanzeigen.",
        "fun_fact": "Der Flexity ersetzt schrittweise die alten E2-Garnituren und bietet WLAN und USB-Ladebuchsen.",
    },
    "Silberpfeil": {
        "emoji": "🚇",
        "color": "#6C757D",
        "category": "U-Bahn",
        "desc": "U-Bahn Reihe U, unlackierte Aluminium-Optik, kantige Front. Ein Stück Wiener Technikgeschichte.",
        "fun_fact": "Der Silberpfeil ist seit 1978 im Einsatz – und war in den 1970ern hochmodern.",
    },
    "V-Wagen": {
        "emoji": "🚄",
        "color": "#C1121F",
        "category": "U-Bahn",
        "desc": "U-Bahn Reihe V, ab 2000. Rot-weiße Front, vollständig durchgängig begehbar.",
        "fun_fact": "Die V-Wagen sind die häufigsten U-Bahn-Fahrzeuge in Wien und decken fast alle Linien ab.",
    },
    "X-Wagen": {
        "emoji": "🤖",
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
RUNS_DIR = Path(__file__).parent.parent / "model" / "runs"


def list_trained_runs():
    """Available training runs with saved weights, newest first."""
    return sorted(
        RUNS_DIR.glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def run_label(weights_path):
    args_file = weights_path.parent.parent / "args.yaml"
    run_name = weights_path.parent.parent.name
    date_str = datetime.fromtimestamp(weights_path.stat().st_mtime).strftime("%d.%m.%Y")
    if args_file.exists():
        with open(args_file) as f:
            a = yaml.safe_load(f)
        model = str(a.get("model", "?")).replace(".pt", "")
        epochs = a.get("epochs", "?")
        batch = a.get("batch", "?")
        return f"{run_name}  ·  {model}, {epochs} ep, batch {batch}  ·  {date_str}"
    return f"{run_name}  ·  {date_str}"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Einstellungen")

    trained_runs = list_trained_runs()
    run_choices = {run_label(w): w for w in trained_runs}
    demo_label = "🧪 Demo (Basis-YOLOv8n, untrainiert)"
    run_labels = [demo_label] + list(run_choices.keys())
    selected_run = st.selectbox(
        "Trainings-Run", run_labels,
        index=1 if run_choices else 0,
        help="Wähle, welches trainierte Modell für die Erkennung verwendet werden soll.",
    )
    weights_path = run_choices.get(selected_run)

    st.divider()
    conf_thresh = st.slider("Konfidenz-Schwellwert", 0.1, 0.95, 0.30, 0.05,
                            help="Mindestvertrauen für eine Erkennung")
    iou_thresh = st.slider("IoU-Schwellwert (NMS)", 0.1, 0.9, 0.45, 0.05,
                           help="Non-Maximum Suppression Overlap-Grenze")
    show_labels = st.checkbox("Labels auf Bild anzeigen", value=True)
    show_conf = st.checkbox("Konfidenz anzeigen", value=True)
    show_cam = st.checkbox("🔥 Erklärbarkeits-Heatmap anzeigen", value=False,
                           help="Visualisiert per EigenCAM, welche Bildregionen die "
                                "stärksten Aktivierungen im Modell auslösen.")

    st.divider()
    for name, info in CLASS_INFO.items():
        badge_color = info.get("color", "#333")
        badge_emoji = info.get("emoji", "🚌")
        badge_cat   = info.get("category", "")
        badge_desc  = info.get("desc", "")
        st.markdown(
            f"<span class='class-badge' style='background:{badge_color}'>"
            f"{badge_emoji} {name}</span>&nbsp;"
            f"<small style='color:#6c757d'>{badge_cat}</small><br>"
            f"<small>{badge_desc}</small>",
            unsafe_allow_html=True,
        )
        st.markdown("")

    st.divider()
    st.caption("Wiener Linien Fleet Spotter · ML-Kurs 2026")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<p class='hero-title'>🚋 Wiener Linien Fleet Spotter</p>", unsafe_allow_html=True)
st.markdown(
    "<p class='hero-subtitle'>Lade ein Foto hoch – das KI-Modell erkennt und klassifiziert "
    "Straßenbahnen & U-Bahnen der Wiener Linien.</p>",
    unsafe_allow_html=True,
)
st.divider()


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🧠 Modell wird geladen …")
def load_model(weights_path):
    from ultralytics import YOLO
    if weights_path is None:
        return YOLO("yolov8n.pt"), False   # demo mode
    return YOLO(str(weights_path)), True


model, is_trained = load_model(weights_path)


# ── Explainability (EigenCAM) ─────────────────────────────────────────────────
def compute_eigencam(yolo_model, img_rgb, layer_index=-14):
    """Highlights the regions that drive the backbone's strongest feature
    activations via the principal component of its activation maps (EigenCAM).
    Needs no backprop, which makes it robust for detection models like YOLO.
    """
    target_layer = yolo_model.model.model[layer_index]
    captured = []

    def hook(_module, _inp, out):
        captured.append(out.detach().cpu().numpy())

    handle = target_layer.register_forward_hook(hook)
    try:
        yolo_model.predict(source=img_rgb, verbose=False)
    finally:
        handle.remove()

    if not captured:
        return None

    fmap = captured[0][0]                          # (C, H, W)
    flat = fmap.reshape(fmap.shape[0], -1).T       # (H*W, C)
    flat = flat - flat.mean(axis=0)
    _, _, vt = np.linalg.svd(flat, full_matrices=True)
    cam = (flat @ vt[0]).reshape(fmap.shape[1:])   # (H, W)

    cam -= cam.min()
    cam /= (cam.max() + 1e-8)
    cam = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 0.55, heatmap, 0.45, 0)

if not is_trained:
    st.warning(
        "⚠️ **Demo-Modus**: Kein trainiertes Modell gefunden. "
        "Das Basis-YOLOv8n-Modell ist aktiv — Wiener-Linien-Klassen werden erst nach "
        "dem Training erkannt. Führe `python scripts/train.py` aus, um das Modell zu trainieren.",
        icon="⚠️",
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
        st.markdown("#### 📷 Original")
        st.image(pil_img, use_column_width=True)

    with st.spinner("🔍 Erkenne Fahrzeuge …"):
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
        st.markdown("#### 🎯 Erkannte Fahrzeuge")
        st.image(ann_img, use_column_width=True)

    if show_cam:
        with st.spinner("🧠 Berechne Erklärbarkeits-Heatmap …"):
            cam_img = compute_eigencam(model, img_np)
        if cam_img is not None:
            st.divider()
            st.markdown("### 🔥 Worauf achtet das Modell?")
            st.image(cam_img, use_column_width=True)
            st.caption(
                "Die Heatmap (EigenCAM) zeigt, welche Bildregionen die stärksten "
                "Aktivierungen im Backbone des Modells auslösen – ein Blick in die "
                "„Gedanken“ des neuronalen Netzes."
            )

    st.divider()
    st.markdown("### 📋 Erkennungs-Details")

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
            emoji    = info.get("emoji", "🚌")

            with det_cols[i % 3]:
                bar_w = int(conf_val * 100)
                st.markdown(f"""
                <div class="detection-card">
                  <span class="class-badge" style="background:{color}">{emoji} {cls_name}</span>
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
                    💡 {info.get('fun_fact','')}
                  </p>
                </div>
                """, unsafe_allow_html=True)

        # Download annotated image
        st.divider()
        buf = io.BytesIO()
        Image.fromarray(ann_img).save(buf, format="JPEG", quality=95)
        st.download_button(
            label    = "💾 Annotiertes Bild herunterladen",
            data     = buf.getvalue(),
            file_name= "fleet_spotter_result.jpg",
            mime     = "image/jpeg",
        )

else:
    # Placeholder / instructions
    st.markdown("""
    <div style="text-align:center;padding:3rem;border:2px dashed #dee2e6;border-radius:12px;color:#adb5bd">
      <div style="font-size:3rem">🚋</div>
      <p style="font-size:1.1rem;margin-top:1rem">
        Bild hochladen, um die Erkennung zu starten
      </p>
      <p style="font-size:0.85rem">
        Unterstützte Klassen: E2-Tram · ULF · Flexity · Silberpfeil · V-Wagen · X-Wagen
      </p>
    </div>
    """, unsafe_allow_html=True)
