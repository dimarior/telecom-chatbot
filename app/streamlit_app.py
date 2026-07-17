"""
app/streamlit_app.py — GAIA Telecom Colombia
"""
import base64
import uuid
import os
import requests
from pathlib import Path
from PIL import Image
import streamlit as st

MODELO_ACTIVO  = "mistral-small-latest"
API_URL        = "http://127.0.0.1:8082/ask/graph"
API_AUDIO_URL  = "http://127.0.0.1:8082/ask/audio"
API_IMAGE_URL  = "http://127.0.0.1:8082/ask/image"
API_DOC_URL    = "http://127.0.0.1:8082/ask/document"
HEALTH_URL     = "http://127.0.0.1:8082/health"
ASSETS         = Path("app/assets")

OPERADORES = {
    "todos": {
        "nombre": "Todos los operadores",
        "color": "#4FE3E0",
        "emoji": "📡",
        "logo": None,
        "placeholder": "Pregunta sobre Claro, Movistar o Tigo...",
    },
    "claro": {
        "nombre": "Claro",
        "color": "#4FE3E0",
        "emoji": "🔴",
        "logo": "app/assets/claro-logo.png",
        "placeholder": "Pregunta sobre planes, soporte o servicios de Claro...",
    },
    "movistar": {
        "nombre": "Movistar",
        "color": "#5DA8FF",
        "emoji": "🔵",
        "logo": "app/assets/movistar-logo.png",
        "placeholder": "Pregunta sobre planes, soporte o servicios de Movistar...",
    },
    "tigo": {
        "nombre": "Tigo",
        "color": "#7B61FF",
        "emoji": "🟡",
        "logo": "app/assets/tigo-logo.png",
        "placeholder": "Pregunta sobre planes, soporte o servicios de Tigo...",
    },
}


def img_to_base64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""


def get_img_tag(path: str, width: str = "auto", extra_style: str = "") -> str:
    b64 = img_to_base64(path)
    if not b64:
        return ""
    if path.endswith((".jpg", ".jpeg")):
        ext = "jpg"
    elif path.endswith(".webp"):
        ext = "webp"
    else:
        ext = "png"
    return f'<img src="data:image/{ext};base64,{b64}" style="width:{width};{extra_style}" />'


try:
    favicon = Image.open(str(ASSETS / "gaia-favicon.ico"))
    st.set_page_config(
        page_title="GAIA - Inteligencia Conversacional",
        page_icon=favicon,
        layout="wide",
    )
except Exception:
    st.set_page_config(
        page_title="GAIA - Inteligencia Conversacional",
        page_icon="✦",
        layout="wide",
    )

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())[:8]
if "sesiones" not in st.session_state:
    sid = st.session_state["session_id"]
    st.session_state["sesiones"] = {sid: {"nombre": "Nueva conversación", "historial": []}}
    st.session_state["sesion_actual"] = sid
if "pregunta_actual" not in st.session_state:
    st.session_state["pregunta_actual"] = ""
if "limpiar_textarea" not in st.session_state:
    st.session_state["limpiar_textarea"] = False
if "operador_actual" not in st.session_state:
    st.session_state["operador_actual"] = "todos"
if "archivo_adjunto" not in st.session_state:
    st.session_state["archivo_adjunto"] = None
if "tipo_adjunto" not in st.session_state:
    st.session_state["tipo_adjunto"] = None
if "mostrar_uploader" not in st.session_state:
    st.session_state["mostrar_uploader"] = None
if "textarea_key" not in st.session_state:
    st.session_state["textarea_key"] = 0
if "texto_transcrito" not in st.session_state:
    st.session_state["texto_transcrito"] = ""

operador = st.session_state["operador_actual"]
color_op = OPERADORES[operador]["color"]

# ── ESTILOS ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@300;400;500;600;700;800&display=swap');
    * {{ font-family: 'Inter Tight', sans-serif; }}

    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    section[data-testid="stMain"] > div {{ background-color: #07111F !important; }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stDeployButton {{display: none !important;}}
    header[data-testid="stHeader"] {{display: none !important;}}
    [data-testid="collapsedControl"] {{display: none !important;}}

    [data-testid="stSidebar"] {{
        background-color: #02060B !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }}
    [data-testid="stSidebar"] * {{ color: #B7C2D0 !important; }}

    .sidebar-section {{
        font-size: 0.65rem; font-weight: 600;
        color: rgba(183,194,208,0.5) !important;
        text-transform: uppercase; letter-spacing: 2.5px;
        margin: 1.2rem 0 0.5rem;
    }}

    [data-testid="stSidebar"] .stButton > button {{
        background: rgba(255,255,255,0.03) !important;
        color: #B7C2D0 !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 8px !important;
        font-size: 0.82rem !important;
        transition: all 0.35s ease !important;
        text-align: left !important;
        box-shadow: none !important;
        outline: none !important;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        background: rgba(79,227,224,0.08) !important;
        border-color: rgba(79,227,224,0.3) !important;
        color: #4FE3E0 !important;
    }}
    [data-testid="stSidebar"] .stButton > button:focus,
    [data-testid="stSidebar"] .stButton > button:focus-visible {{
        outline: none !important;
        box-shadow: none !important;
        border-color: rgba(79,227,224,0.2) !important;
    }}

    .op-logo-box {{
        display: flex; justify-content: center; align-items: center;
        height: 48px; margin-bottom: 0px;
    }}
    .op-btn-activo .stButton > button {{
        border-color: var(--op-color, rgba(79,227,224,0.6)) !important;
        background: var(--op-bg, rgba(79,227,224,0.08)) !important;
        box-shadow: 0 0 18px var(--op-glow, rgba(79,227,224,0.2)) !important;
        color: #F8FBFF !important;
        font-weight: 700 !important;
    }}

    .stButton > button {{
        background: rgba(255,255,255,0.04) !important;
        color: #B7C2D0 !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 8px !important;
        font-size: 0.82rem !important;
        transition: all 0.35s ease !important;
        text-align: left !important;
    }}
    .stButton > button:hover {{
        background: rgba(79,227,224,0.08) !important;
        border-color: rgba(79,227,224,0.3) !important;
        color: #4FE3E0 !important;
    }}
    .stButton > button:focus,
    .stButton > button:focus-visible {{
        outline: none !important;
        box-shadow: none !important;
    }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(90deg, #4FE3E0 0%, #5DA8FF 45%, #7B61FF 100%) !important;
        color: white !important;
        border: none !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        border-radius: 99px !important;
        box-shadow: 0 4px 24px rgba(79,227,224,0.25) !important;
        text-align: center !important;
        opacity: 1 !important;
        height: auto !important;
        min-height: auto !important;
        margin-top: 0 !important;
    }}

    .stTextArea textarea {{
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
        color: #F8FBFF !important;
        font-size: 0.9rem !important;
    }}
    .stTextArea textarea:focus {{
        border-color: rgba(79,227,224,0.4) !important;
        box-shadow: 0 0 0 3px rgba(79,227,224,0.1) !important;
    }}
    .stTextArea textarea::placeholder {{
        color: rgba(183,194,208,0.4) !important;
    }}

    .hero-section {{ padding: 0.5rem 0 1.5rem; text-align: center; }}
    .hero-section h1 {{
        color: #F8FBFF; font-size: 2rem; font-weight: 800;
        margin: 0 0 0.3rem; letter-spacing: -0.5px;
    }}
    .hero-gradient-text {{
        background: linear-gradient(90deg, #4FE3E0 0%, #5DA8FF 45%, #7B61FF 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; font-size: 2rem; font-weight: 800;
    }}
    .hero-section p {{ color: #B7C2D0; font-size: 0.9rem; margin-top: 0.5rem; }}

    .status-ok {{
        background: rgba(79,227,224,0.08); border: 1px solid rgba(79,227,224,0.2);
        border-left: 3px solid #4FE3E0; border-radius: 10px;
        padding: 0.55rem 1rem; color: #4FE3E0; font-size: 0.82rem;
        font-weight: 500; margin-bottom: 1rem;
    }}
    .status-err {{
        background: rgba(255,80,80,0.08); border: 1px solid rgba(255,80,80,0.2);
        border-left: 3px solid #FF5050; border-radius: 10px;
        padding: 0.55rem 1rem; color: #FF8080; font-size: 0.82rem;
        margin-bottom: 1rem;
    }}

    .section-title {{
        font-size: 0.75rem; font-weight: 600; color: rgba(183,194,208,0.6);
        text-transform: uppercase; letter-spacing: 2px; margin: 1.5rem 0 0.8rem;
    }}

    .chat-user {{
        background: rgba(79,227,224,0.08); border: 1px solid rgba(79,227,224,0.15);
        border-radius: 12px 12px 2px 12px; padding: 0.75rem 1rem;
        margin: 0.5rem 0; font-size: 0.9rem; color: #F8FBFF !important;
        text-align: right;
    }}
    .chat-user-badge {{
        font-size: 0.65rem; color: rgba(79,227,224,0.6);
        margin-bottom: 0.3rem; text-align: right;
    }}
    .chat-bot {{
        background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
        border-left: 3px solid {color_op}; border-radius: 2px 12px 12px 12px;
        padding: 0.75rem 1rem; margin: 0.5rem 0; font-size: 0.9rem;
        color: #B7C2D0; line-height: 1.8;
    }}
    .chat-meta {{ font-size: 0.68rem; color: rgba(183,194,208,0.4); margin-top: 0.4rem; }}

    .respuesta-box {{
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
        border-top: 2px solid {color_op}; border-radius: 16px;
        padding: 1.5rem; margin-top: 1rem; box-shadow: 0 8px 40px rgba(0,0,0,0.3);
    }}
    .respuesta-label {{
        font-size: 0.65rem; font-weight: 700; color: {color_op};
        text-transform: uppercase; letter-spacing: 2.5px; margin-bottom: 0.75rem;
    }}
    .respuesta-text {{ color: #F8FBFF; line-height: 1.9; font-size: 0.92rem; }}

    .metrics-row {{
        display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 8px; margin-top: 1rem;
    }}
    .metric-card {{
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px; padding: 0.75rem; text-align: center;
    }}
    .metric-val {{ font-size: 0.95rem; font-weight: 700; color: {color_op}; display: block; }}
    .metric-lbl {{
        font-size: 0.65rem; color: rgba(183,194,208,0.5); margin-top: 2px;
        display: block; text-transform: uppercase; letter-spacing: 1px;
    }}

    .custom-divider {{
        height: 1px;
        background: linear-gradient(to right, transparent, rgba(255,255,255,0.08), transparent);
        margin: 1.5rem 0;
    }}

    .adjunto-preview {{
        display: flex; align-items: center; gap: 10px;
        background: rgba(79,227,224,0.06);
        border: 1px solid rgba(79,227,224,0.2);
        border-radius: 10px; padding: 0.6rem 1rem;
        margin-bottom: 0.5rem; font-size: 0.82rem; color: #4FE3E0;
    }}
    .adjunto-preview span {{ flex: 1; }}

    .modal-btn-icon {{
        text-align: center;
        margin-bottom: 2px;
        height: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .modal-btn-icon img {{
        height: 22px;
        object-fit: contain;
        filter: drop-shadow(0 0 4px rgba(79,227,224,0.2));
    }}

    .footer {{
        text-align: center; background: rgba(2,6,11,0.8);
        border: 1px solid rgba(255,255,255,0.06); color: rgba(183,194,208,0.5);
        font-size: 0.8rem; margin-top: 1.5rem; padding: 1.2rem;
        border-radius: 12px; letter-spacing: 0.5px;
    }}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    logo_tag = get_img_tag(
        str(ASSETS / "gaia-logo.png"),
        width="180px",
        extra_style="margin-bottom:1.5rem;display:block;"
    )
    if logo_tag:
        st.markdown(logo_tag, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-size:2rem;font-weight:800;
            background:linear-gradient(90deg,#4FE3E0,#7B61FF);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            margin-bottom:1.5rem;">✦ GAIA</div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Conversaciones</div>', unsafe_allow_html=True)

    if st.button("+ Nueva conversación", use_container_width=True, key="nueva_sidebar"):
        nuevo_sid = str(uuid.uuid4())[:8]
        st.session_state["sesiones"][nuevo_sid] = {
            "nombre": "Nueva conversación", "historial": []
        }
        st.session_state["sesion_actual"] = nuevo_sid
        st.session_state["session_id"] = nuevo_sid
        st.session_state["pregunta_actual"] = ""
        st.session_state["textarea_principal"] = ""
        st.session_state["operador_actual"] = "todos"
        st.session_state["archivo_adjunto"] = None
        st.session_state["tipo_adjunto"] = None
        st.session_state["mostrar_uploader"] = None
        st.rerun()

    for sid, datos in list(st.session_state["sesiones"].items()):
        if not datos["historial"]:
            continue
        if st.button(datos["nombre"], key=f"ses_{sid}", use_container_width=True):
            st.session_state["sesion_actual"] = sid
            st.session_state["session_id"] = sid
            st.rerun()

    st.markdown("---")
    st.markdown('<div class="sidebar-section">Operador</div>', unsafe_allow_html=True)

    for key, op in OPERADORES.items():
        activo       = st.session_state["operador_actual"] == key
        border_color = f"{op['color']}90" if activo else "rgba(255,255,255,0.10)"
        bg_color     = f"{op['color']}12" if activo else "rgba(255,255,255,0.03)"
        glow_shadow  = f"0 0 16px {op['color']}30" if activo else "none"
        font_weight  = "700" if activo else "500"
        text_color   = op['color'] if activo else "#B7C2D0"

        logo_inner = ""
        if op.get("logo"):
            b64 = img_to_base64(op["logo"])
            if b64:
                logo_inner = (
                    f'<img src="data:image/png;base64,{b64}" '
                    f'style="height:26px;object-fit:contain;max-width:75px;" />'
                )
        if not logo_inner:
            logo_inner = f'<span style="font-size:1.3rem;">{op["emoji"]}</span>'

        st.markdown(f"""
        <div style="display:flex;justify-content:center;align-items:center;
                    height:36px;margin-bottom:2px;">
            {logo_inner}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <style>
            div[data-testid="stSidebar"]
            div[data-testid="stButton"]:has(button[data-testid="baseButton-secondary"][aria-label="{op['nombre']}"])
            button {{
                border-color: {border_color} !important;
                background: {bg_color} !important;
                box-shadow: {glow_shadow} !important;
                color: {text_color} !important;
                font-weight: {font_weight} !important;
                text-align: center !important;
                border-radius: 8px !important;
                font-size: 0.82rem !important;
            }}
        </style>
        """, unsafe_allow_html=True)

        if st.button(op["nombre"], key=f"op_{key}", use_container_width=True):
            st.session_state["operador_actual"] = key
            st.session_state["pregunta_actual"] = ""
            st.rerun()

        st.markdown("<div style='margin-bottom:6px'></div>", unsafe_allow_html=True)

# ── MAIN ──────────────────────────────────────────────────────────────────────
op_info = OPERADORES[st.session_state["operador_actual"]]

# Banner
banner_tag = get_img_tag(
    str(ASSETS / "gaia-banner.png"),
    width="100%",
    extra_style="display:block;width:100%;height:auto;"
)
if banner_tag:
    st.markdown(f"""
    <div style="border-radius:16px;overflow:hidden;margin-bottom:1.5rem;
        box-shadow:0 8px 40px rgba(0,0,0,0.4);">{banner_tag}</div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#07111F 0%,#0D1F3C 60%,#07111F 100%);
        border:1px solid rgba(79,227,224,0.15);border-radius:16px;
        padding:4rem 3rem;margin-bottom:1.5rem;overflow:hidden;
        box-shadow:0 8px 40px rgba(0,0,0,0.4);">
        <p style="font-size:0.7rem;font-weight:600;color:#4FE3E0;
            text-transform:uppercase;letter-spacing:3px;margin-bottom:1rem;">
            ✦ IA CONVERSACIONAL CENTRADA EN EL USUARIO
        </p>
        <h2 style="color:#F8FBFF;font-size:2.5rem;font-weight:800;margin:0 0 0.2rem;">
            Conversaciones más humanas.
        </h2>
        <h2 style="background:linear-gradient(90deg,#4FE3E0,#5DA8FF,#7B61FF);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            font-size:2.5rem;font-weight:800;margin:0 0 1.2rem;">
            Soluciones más inteligentes.
        </h2>
        <p style="color:#B7C2D0;font-size:0.95rem;">
            Plataforma conversacional impulsada por GenAI y RAG multimodal
        </p>
    </div>
    """, unsafe_allow_html=True)

# Hero
st.markdown(f"""
<div class="hero-section">
    <h1>Asistente <span class="hero-gradient-text">GAIA</span></h1>
    <p>Inteligencia conversacional · {op_info['nombre']}</p>
</div>
""", unsafe_allow_html=True)

# Status API
try:
    health = requests.get(HEALTH_URL, timeout=3)
    if health.status_code == 200:
        st.markdown('<div class="status-ok">✦ Sistema conectado y operando correctamente</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-err">⚠ Estado inesperado de la API</div>',
                    unsafe_allow_html=True)
except Exception:
    st.markdown(f"""
    <div class="status-err">⚡ API no disponible —
        <code>uvicorn api.main:app --reload --host 127.0.0.1 --port 8082</code>
    </div>""", unsafe_allow_html=True)

# Historial
historial_actual = st.session_state["sesiones"].get(
    st.session_state["sesion_actual"], {}).get("historial", [])

if historial_actual:
    st.markdown('<div class="section-title">Conversación</div>', unsafe_allow_html=True)
    for msg in historial_actual:
        if msg["role"] == "user":
            modalidad = msg.get("modalidad", "texto")
            badge = ""
            if modalidad == "audio":
                badge = '<div class="chat-user-badge">🎤 Voz transcrita</div>'
            elif modalidad == "imagen":
                badge = f'<div class="chat-user-badge">🖼 Imagen: {msg.get("archivo","")}</div>'
            elif modalidad == "documento":
                badge = f'<div class="chat-user-badge">📎 PDF: {msg.get("archivo","")}</div>'
            st.markdown(
                f'<div class="chat-user">{badge}{msg["content"]}</div>',
                unsafe_allow_html=True)
        else:
            modal_icons = {"texto": "✦", "audio": "🎤", "imagen": "🖼", "documento": "📎"}
            modal_icon  = modal_icons.get(msg.get("modalidad", "texto"), "✦")
            st.markdown(
                f'<div class="chat-bot">{msg["content"]}'
                f'<div class="chat-meta">{modal_icon} ⏱ {msg.get("latencia","")}s · {MODELO_ACTIVO} · GAIA</div>'
                f'</div>', unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# Preguntas sugeridas
PREGUNTAS = {
    "todos": [
        "¿Cómo hago la portabilidad numérica?",
        "¿Cuáles son los planes de internet hogar disponibles?",
        "¿Cómo reporto una falla técnica?",
        "¿Cómo pago mi factura en línea?",
        "¿Qué es la autogestión y cómo funciona?",
        "¿Cuáles son los canales de atención al cliente?",
    ],
    "claro": [
        "¿Cómo me registro en Mi Claro?",
        "¿Cómo activo el roaming en Claro?",
        "¿Dónde pago mi factura Claro?",
        "¿Cómo hago portabilidad a Claro?",
        "¿Qué planes de fibra óptica tiene Claro?",
        "¿Cómo reporto una avería con Claro?",
    ],
    "movistar": [
        "¿Cómo accedo a Mi Movistar?",
        "¿Cómo solicito asistencia técnica en Movistar?",
        "¿Qué canales de TV tiene Movistar?",
        "¿Cómo hago portabilidad a Movistar?",
        "¿Cómo activo el buzón de voz en Movistar?",
        "¿Cuáles son los procesos de autogestión de Movistar?",
    ],
    "tigo": [
        "¿Cómo recargo mi línea Tigo?",
        "¿Cómo contacto al soporte de Tigo?",
        "¿Qué planes prepago tiene Tigo?",
        "¿Cómo hago portabilidad a Tigo?",
        "¿Cómo consulto mi saldo en Tigo?",
        "¿Cuáles son las preguntas frecuentes de Tigo?",
    ],
}

st.markdown('<div class="section-title">Preguntas frecuentes</div>', unsafe_allow_html=True)
preguntas = PREGUNTAS[st.session_state["operador_actual"]]
cols = st.columns(3)
for i, ejemplo in enumerate(preguntas):
    if cols[i % 3].button(ejemplo, key=f"ej_{i}", use_container_width=True):
        st.session_state["pregunta_actual"] = ejemplo
        st.session_state["textarea_principal"] = ejemplo

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Tu consulta</div>', unsafe_allow_html=True)

# Limpiar textarea o cargar texto transcrito
if st.session_state.get("limpiar_textarea", False):
    st.session_state["pregunta_actual"] = ""
    st.session_state["texto_transcrito"] = ""
    st.session_state["textarea_key"] += 1
    st.session_state["limpiar_textarea"] = False

_texto_inicial = st.session_state.get("texto_transcrito", "") or st.session_state.get("pregunta_actual", "")

# ── Textarea ──────────────────────────────────────────────────────────────────
pregunta = st.text_area(
    label="Pregunta",
    placeholder=op_info["placeholder"],
    height=110,
    label_visibility="collapsed",
    key=f"textarea_principal_{st.session_state['textarea_key']}",
    value=_texto_inicial,
)

# ── Botones multimodal: ícono arriba + botón abajo ────────────────────────────
icon_img_b64 = img_to_base64(str(ASSETS / "icon-imagen.png"))
icon_pdf_b64 = img_to_base64(str(ASSETS / "icon-pdf.png"))
icon_aud_b64 = img_to_base64(str(ASSETS / "icon-audio.png"))

col_b1, col_b2, col_b3, col_spacer = st.columns([1, 1, 1, 9])

with col_b1:
    if icon_img_b64:
        st.markdown(
            f'<div class="modal-btn-icon"><img src="data:image/png;base64,{icon_img_b64}"/></div>',
            unsafe_allow_html=True
        )
    if st.button("Imagen", key="btn_imagen", use_container_width=True, help="Adjuntar imagen"):
        st.session_state["mostrar_uploader"] = (
            None if st.session_state["mostrar_uploader"] == "imagen" else "imagen"
        )
        st.session_state["archivo_adjunto"] = None
        st.session_state["tipo_adjunto"] = None

with col_b2:
    if icon_pdf_b64:
        st.markdown(
            f'<div class="modal-btn-icon"><img src="data:image/png;base64,{icon_pdf_b64}"/></div>',
            unsafe_allow_html=True
        )
    if st.button("PDF", key="btn_doc", use_container_width=True, help="Adjuntar documento PDF"):
        st.session_state["mostrar_uploader"] = (
            None if st.session_state["mostrar_uploader"] == "documento" else "documento"
        )
        st.session_state["archivo_adjunto"] = None
        st.session_state["tipo_adjunto"] = None

with col_b3:
    if icon_aud_b64:
        st.markdown(
            f'<div class="modal-btn-icon"><img src="data:image/png;base64,{icon_aud_b64}"/></div>',
            unsafe_allow_html=True
        )
    if st.button("Voz", key="btn_audio", use_container_width=True, help="Grabar mensaje de voz"):
        st.session_state["mostrar_uploader"] = (
            None if st.session_state["mostrar_uploader"] == "audio" else "audio"
        )
        st.session_state["archivo_adjunto"] = None
        st.session_state["tipo_adjunto"] = None

# Uploaders dinámicos
if st.session_state["mostrar_uploader"] == "imagen":
    archivo = st.file_uploader(
        "Selecciona una imagen",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="uploader_imagen",
        label_visibility="collapsed",
    )
    if archivo:
        if len(archivo.getvalue()) > 5 * 1024 * 1024:
            st.error("La imagen supera el límite de 5 MB. Por favor usa una imagen más pequeña.")
        else:
            st.session_state["archivo_adjunto"] = archivo
            st.session_state["tipo_adjunto"] = "imagen"

elif st.session_state["mostrar_uploader"] == "documento":
    archivo = st.file_uploader(
        "Selecciona un PDF (máximo 5 MB)",
        type=["pdf"],
        key="uploader_doc",
        label_visibility="collapsed",
    )
    if archivo:
        if len(archivo.getvalue()) > 5 * 1024 * 1024:
            st.error("El archivo supera el límite de 5 MB. Por favor usa un PDF más pequeño.")
        else:
            st.session_state["archivo_adjunto"] = archivo
            st.session_state["tipo_adjunto"] = "documento"

elif st.session_state["mostrar_uploader"] == "audio":
    audio_grabado = st.audio_input(
        "Graba tu mensaje de voz",
        key="grabador_audio",
    )
    if audio_grabado:
        # Transcripción automática al terminar de grabar
        with st.spinner("Transcribiendo tu mensaje de voz..."):
            try:
                audio_grabado.seek(0)
                files = {"audio": ("audio.wav", audio_grabado.read(), "audio/wav")}
                data  = {"session_id": st.session_state["session_id"]}
                resp  = requests.post(
                    "http://127.0.0.1:8082/transcribe",
                    files=files, data=data, timeout=180
                )
                result = resp.json()
                if result.get("success") and result.get("text"):
                    st.session_state["texto_transcrito"] = result["text"]
                    st.session_state["textarea_key"] += 1
                    st.session_state["mostrar_uploader"] = None
                    st.rerun()
                else:
                    st.error(f"No se pudo transcribir: {result.get('error', 'Audio no reconocido')}")
            except Exception as e:
                st.error(f"Error al transcribir: {str(e)}")

# Preview del archivo adjunto (solo imagen y PDF, no audio)
if st.session_state["archivo_adjunto"]:
    archivo_adj_prev = st.session_state["archivo_adjunto"]
    tipo_prev = st.session_state["tipo_adjunto"]

    if tipo_prev in ("imagen", "documento"):
        icono_badge = {"imagen": "🖼", "documento": "📎"}.get(tipo_prev, "📎")
        col_prev, col_rm = st.columns([10, 1])
        with col_prev:
            st.markdown(f"""
            <div class="adjunto-preview">
                <span>{icono_badge}</span>
                <span>{archivo_adj_prev.name}</span>
                <span style="color:rgba(79,227,224,0.5);font-size:0.7rem;">
                    {round(len(archivo_adj_prev.getvalue()) / 1024, 1)} KB
                </span>
            </div>
            """, unsafe_allow_html=True)
            if tipo_prev == "imagen":
                try:
                    img_preview = Image.open(archivo_adj_prev)
                    st.image(img_preview, width=200)
                    archivo_adj_prev.seek(0)
                except Exception:
                    pass
        with col_rm:
            if st.button("✕", key="rm_adj", help="Quitar archivo"):
                st.session_state["archivo_adjunto"] = None
                st.session_state["tipo_adjunto"] = None
                st.session_state["mostrar_uploader"] = None
                st.rerun()

# ── Botón principal ───────────────────────────────────────────────────────────
consultar = st.button("Consultar con GAIA ✦", type="primary", use_container_width=True)

# ── Lógica de envío ───────────────────────────────────────────────────────────
archivo_adj = st.session_state.get("archivo_adjunto")
tipo_adj    = st.session_state.get("tipo_adjunto")
hay_texto   = bool(pregunta and pregunta.strip())
hay_archivo = archivo_adj is not None

if consultar and not hay_texto and not hay_archivo:
    st.warning("Escribe una pregunta o adjunta un archivo antes de consultar.")

elif consultar and (hay_texto or hay_archivo):
    operador_sel = st.session_state["operador_actual"]
    session_id   = st.session_state["session_id"]

    with st.spinner("GAIA está procesando tu consulta..."):
        try:
            result = None
            modalidad_enviada = "texto"
            contenido_usuario = ""

            if hay_archivo and tipo_adj == "audio":
                archivo_adj.seek(0)
                files  = {"audio": (archivo_adj.name, archivo_adj.read(), "audio/mpeg")}
                data   = {"session_id": session_id}
                resp   = requests.post(API_AUDIO_URL, files=files, data=data, timeout=180)
                result = resp.json()
                modalidad_enviada = "audio"
                contenido_usuario = f"[Audio: {archivo_adj.name}]"
                if hay_texto:
                    contenido_usuario += f" {pregunta.strip()}"

            elif hay_archivo and tipo_adj == "imagen":
                archivo_adj.seek(0)
                files  = {"imagen": (archivo_adj.name, archivo_adj.read(), "image/jpeg")}
                data   = {"session_id": session_id, "pregunta_adicional": pregunta.strip()}
                resp   = requests.post(API_IMAGE_URL, files=files, data=data, timeout=180)
                result = resp.json()
                modalidad_enviada = "imagen"
                contenido_usuario = f"[Imagen: {archivo_adj.name}]"
                if hay_texto:
                    contenido_usuario += f" {pregunta.strip()}"

            elif hay_archivo and tipo_adj == "documento":
                archivo_adj.seek(0)
                files  = {"documento": (archivo_adj.name, archivo_adj.read(), "application/pdf")}
                data   = {"session_id": session_id, "pregunta_adicional": pregunta.strip()}
                resp   = requests.post(API_DOC_URL, files=files, data=data, timeout=180)
                result = resp.json()
                modalidad_enviada = "documento"
                contenido_usuario = f"[PDF: {archivo_adj.name}]"
                if hay_texto:
                    contenido_usuario += f" {pregunta.strip()}"

            else:
                pregunta_enriquecida = pregunta.strip()
                if operador_sel != "todos":
                    pregunta_enriquecida = f"[{OPERADORES[operador_sel]['nombre']}] {pregunta.strip()}"
                resp   = requests.post(
                    API_URL,
                    json={"pregunta": pregunta_enriquecida, "session_id": session_id},
                    timeout=120,
                )
                result = resp.json()
                modalidad_enviada = "texto"
                contenido_usuario = pregunta.strip()

            if result and "respuesta" in result:
                sid = st.session_state["sesion_actual"]
                if sid not in st.session_state["sesiones"]:
                    st.session_state["sesiones"][sid] = {"nombre": "Nueva conversación", "historial": []}

                nombre_archivo = archivo_adj.name if archivo_adj else ""
                st.session_state["sesiones"][sid]["historial"].append({
                    "role": "user",
                    "content": contenido_usuario,
                    "modalidad": modalidad_enviada,
                    "archivo": nombre_archivo,
                })
                st.session_state["sesiones"][sid]["historial"].append({
                    "role": "assistant",
                    "content": result["respuesta"],
                    "latencia": result["latencia_segundos"],
                    "modalidad": modalidad_enviada,
                })
                if len(st.session_state["sesiones"][sid]["historial"]) == 2:
                    nombre_conv = contenido_usuario[:30] if contenido_usuario else nombre_archivo[:30]
                    st.session_state["sesiones"][sid]["nombre"] = nombre_conv

                st.session_state["limpiar_textarea"] = True
                st.session_state["texto_transcrito"] = ""
                st.session_state["archivo_adjunto"] = None
                st.session_state["tipo_adjunto"] = None
                st.session_state["mostrar_uploader"] = None

                modal_icons = {"texto": "✦", "audio": "🎤", "imagen": "🖼", "documento": "📎"}
                modal_icon  = modal_icons.get(modalidad_enviada, "✦")
                modal_label = modalidad_enviada.upper()

                st.markdown(f"""
                <div class="respuesta-box">
                    <div class="respuesta-label">{modal_icon} GAIA · {op_info['nombre']} · {modal_label}</div>
                    <div class="respuesta-text">{result['respuesta']}</div>
                </div>
                <div class="metrics-row">
                    <div class="metric-card">
                        <span class="metric-val">{result['latencia_segundos']}s</span>
                        <span class="metric-lbl">Latencia</span>
                    </div>
                    <div class="metric-card">
                        <span class="metric-val">{modal_icon}</span>
                        <span class="metric-lbl">{modal_label}</span>
                    </div>
                    <div class="metric-card">
                        <span class="metric-val">3</span>
                        <span class="metric-lbl">Operadores</span>
                    </div>
                    <div class="metric-card">
                        <span class="metric-val">RAG</span>
                        <span class="metric-lbl">Motor</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.rerun()
            else:
                error_msg = result.get("detail", result.get("error", "Error desconocido")) if result else "Sin respuesta"
                st.error(f"Error: {error_msg}")

        except requests.exceptions.ConnectionError:
            st.error("❌ No se pudo conectar con la API en :8082")
        except requests.exceptions.Timeout:
            st.warning("⏳ La consulta tardó demasiado. Intenta de nuevo.")
        except Exception as e:
            st.error(f"Error inesperado: {str(e)}")

# Footer brands
brands_path = str(ASSETS / "gaia-footer-brands.png")
try:
    with open(brands_path, "rb") as f:
        b64_brands = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <div style="border-radius:16px;overflow:hidden;margin-top:2rem;
        box-shadow:0 4px 30px rgba(0,0,0,0.3);">
        <img src="data:image/png;base64,{b64_brands}" style="width:100%;display:block;"/>
    </div>
    """, unsafe_allow_html=True)
except FileNotFoundError:
    st.markdown("""
    <div style="background:rgba(2,6,11,0.8);border:1px solid rgba(255,255,255,0.06);
        border-radius:16px;padding:1.5rem 2rem;margin-top:2rem;text-align:center;">
        <p style="color:rgba(183,194,208,0.4);font-size:0.7rem;
            text-transform:uppercase;letter-spacing:2px;margin-bottom:1rem;">
            Empresas que confían en GAIA
        </p>
        <p style="color:rgba(183,194,208,0.6);font-size:0.9rem;">
            Tigo · WOM · Movistar · DirecTV · Claro · ETB · y muchas más
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    ® GAIA - Inteligencia Conversacional · Colombia · 2026
</div>
""", unsafe_allow_html=True)