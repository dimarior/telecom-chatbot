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

MODELO_ACTIVO = "mistral-small-latest"
API_URL = "http://127.0.0.1:8082/ask/graph"
HEALTH_URL = "http://127.0.0.1:8082/health"
ASSETS = Path("app/assets")

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
        page_title="GAIA — Inteligencia Conversacional",
        page_icon=favicon,
        layout="wide",
    )
except Exception:
    st.set_page_config(
        page_title="GAIA — Inteligencia Conversacional",
        page_icon="✦",
        layout="wide",
    )

# ── SESSION STATE INIT ────────────────────────────────────────────────────────
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

operador = st.session_state["operador_actual"]
color_op = OPERADORES[operador]["color"]

# ── ESTILOS GLOBALES ──────────────────────────────────────────────────────────
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

    /* ── Botones de conversación en sidebar ── */
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

    /* ── Logo de operador centrado sobre el botón ── */
    .op-logo-box {{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 48px;
        margin-bottom: 0px;
    }}
    .op-logo-box img {{
        height: 28px;
        object-fit: contain;
        max-width: 80px;
        pointer-events: none;
        filter: drop-shadow(0 0 6px rgba(255,255,255,0.15));
    }}

    /* Botones de operador con estado activo via clase CSS */
    .op-btn-activo .stButton > button {{
        border-color: var(--op-color, rgba(79,227,224,0.6)) !important;
        background: var(--op-bg, rgba(79,227,224,0.08)) !important;
        box-shadow: 0 0 18px var(--op-glow, rgba(79,227,224,0.2)) !important;
        color: #F8FBFF !important;
        font-weight: 700 !important;
    }}

    /* ── Botón principal ── */
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
        display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-top: 1rem;
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

    # ── Selector de operadores ────────────────────────────────────────────────
    # Logo: imagen decorativa pura, sin bordes ni interacción.
    # Botón: solo el texto del nombre, con estilo activo/inactivo.
    for key, op in OPERADORES.items():
        activo      = st.session_state["operador_actual"] == key
        border_color = f"{op['color']}90" if activo else "rgba(255,255,255,0.10)"
        bg_color     = f"{op['color']}12" if activo else "rgba(255,255,255,0.03)"
        glow_shadow  = f"0 0 16px {op['color']}30" if activo else "none"
        font_weight  = "700" if activo else "500"
        text_color   = op['color'] if activo else "#B7C2D0"

        # Logo/emoji — puro HTML decorativo, sin bordes ni fondo
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

        # CSS para este botón específico (activo/inactivo)
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
            st.markdown(f'<div class="chat-user">{msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="chat-bot">{msg["content"]}'
                f'<div class="chat-meta">⏱ {msg.get("latencia","")}s · {MODELO_ACTIVO} · GAIA</div>'
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

# Si venimos de un envío exitoso, limpiar el textarea
if st.session_state.get("limpiar_textarea", False):
    st.session_state["pregunta_actual"] = ""
    st.session_state["textarea_principal"] = ""
    st.session_state["limpiar_textarea"] = False

pregunta = st.text_area(
    label="Pregunta",
    placeholder=op_info["placeholder"],
    height=110,
    label_visibility="collapsed",
    key="textarea_principal",
)

consultar = st.button("Consultar con GAIA ✦", type="primary", use_container_width=True)

if consultar and pregunta.strip():
    operador_sel = st.session_state["operador_actual"]
    pregunta_enriquecida = pregunta.strip()
    if operador_sel != "todos":
        pregunta_enriquecida = f"[{OPERADORES[operador_sel]['nombre']}] {pregunta.strip()}"

    with st.spinner("GAIA está procesando tu consulta..."):
        try:
            response = requests.post(
                API_URL,
                json={
                    "pregunta": pregunta_enriquecida,
                    "session_id": st.session_state["session_id"],
                },
                timeout=120,
            )
            result = response.json()
            if "error" in result:
                st.error(f"{result['error']}")
            else:
                sid = st.session_state["sesion_actual"]
                if sid not in st.session_state["sesiones"]:
                    st.session_state["sesiones"][sid] = {
                        "nombre": "Nueva conversación", "historial": []
                    }
                st.session_state["sesiones"][sid]["historial"].append({
                    "role": "user", "content": pregunta.strip()
                })
                st.session_state["sesiones"][sid]["historial"].append({
                    "role": "assistant",
                    "content": result["respuesta"],
                    "latencia": result["latencia_segundos"],
                })
                if len(st.session_state["sesiones"][sid]["historial"]) == 2:
                    st.session_state["sesiones"][sid]["nombre"] = pregunta.strip()[:30]

                st.session_state["limpiar_textarea"] = True

                st.markdown(f"""
                <div class="respuesta-box">
                    <div class="respuesta-label">✦ GAIA · {op_info['nombre']}</div>
                    <div class="respuesta-text">{result['respuesta']}</div>
                </div>
                <div class="metrics-row">
                    <div class="metric-card">
                        <span class="metric-val">{result['latencia_segundos']}s</span>
                        <span class="metric-lbl">Latencia</span>
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

        except requests.exceptions.ConnectionError:
            st.error("❌ No se pudo conectar con la API en :8082")
        except requests.exceptions.Timeout:
            st.warning("⏳ La consulta tardó demasiado. Intenta de nuevo.")

elif consultar and not pregunta.strip():
    st.warning("Escribe una pregunta antes de consultar.")

# Footer brands
brands_path = str(ASSETS / "gaia-footer-brands.png")
try:
    with open(brands_path, "rb") as f:
        b64_brands = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <div style="border-radius:16px;overflow:hidden;margin-top:2rem;
        box-shadow:0 4px 30px rgba(0,0,0,0.3);">
        <img src="data:image/png;base64,{b64_brands}"
             style="width:100%;display:block;"/>
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