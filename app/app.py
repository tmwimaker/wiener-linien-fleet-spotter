"""
Wiener Linien Fleet Spotter — Streamlit Web App
================================================
Interactive demo: upload a photo → YOLO detects & classifies the vehicle.

Run:
    streamlit run app/app.py
"""

import io
from pathlib import Path

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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Einstellungen")
    conf_thresh = st.slider("Konfidenz-Schwellwert", 0.1, 0.95, 0.30, 0.05,
                            help="Mindestvertrauen für eine Erkennung")
    iou_thresh  = st.slider("IoU-Schwellwert (NMS)", 0.1, 0.9, 0.45, 0.05,
                            help="Non-Maximum Suppression Overlap-Grenze")
    show_labels = st.checkbox("Labels auf Bild anzeigen", value=True)
    show_conf   = st.checkbox("Konfidenz anzeigen", value=True)

    st.divider()
    st.markdown("### 🚋 Zielklassen")
    for name, info in CLASS_INFO.items():
        st.markdown(
            f"<span class='class-badge' style='background:{info[\"color\"]}'>"
            f"{info['emoji']} {name}</span>&nbsp;"
            f"<small style='color:#6c757d'>{info['category']}</small><br>"
            f"<small>{info['desc']}</small>",
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
def load_model():
    from ultralytics import YOLO
    # Try trained weights first, fall back to base model for demo
    trained = (
        Path(__file__).parent.parent
        / "model" / "runs" / "fleet_spotter_v1" / "weights" / "best.pt"
    )
    if trained.exists():
        return YOLO(str(trained)), True
    else:
        return YOLO("yolov8n.pt"), False   # demo mode


model, is_trained = load_model()

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
        st.image(pil_img, use_container_width=True)

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
        st.image(ann_img, use_container_width=True)

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
