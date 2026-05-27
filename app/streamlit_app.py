"""
app/streamlit_app.py
"""

import base64
import uuid
import requests
from pathlib import Path
from PIL import Image
import streamlit as st

MODELO_ACTIVO = "mistral-small-latest"

API_URL = "http://127.0.0.1:8081/ask"
HEALTH_URL = "http://127.0.0.1:8081/health"

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
    ext = "jpg" if path.endswith(".jpg") else "png"
    return f'<img src="data:image/{ext};base64,{b64}" style="width:{width};{extra_style}" />'

ASSETS      = Path("app/assets")
LOGO_SALONIN = str(ASSETS / "salonin-logo.png")
BANNER      = str(ASSETS / "banner-salonin.png")

try:
    favicon = Image.open(str(ASSETS / "recamierco-favicon.ico"))
    st.set_page_config(
        page_title="Asistente de Productos — Recamier",
        page_icon=favicon,
        layout="wide",
    )
except Exception:
    st.set_page_config(
        page_title="Asistente de Productos — Recamier",
        page_icon="💇",
        layout="wide",
    )

# ---------------------------------------------------------------------------
# Inicializar estados
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())[:8]

if "sesiones" not in st.session_state:
    sid = st.session_state["session_id"]
    st.session_state["sesiones"] = {sid: {"nombre": "Nueva conversación", "historial": []}}
    st.session_state["sesion_actual"] = sid

if "pregunta_actual" not in st.session_state:
    st.session_state["pregunta_actual"] = ""

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    section[data-testid="stMain"] > div {
        background-color: #FFFFFF !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none !important;}
    header[data-testid="stHeader"] {display: none !important;}
    [data-testid="stHeader"] {display: none !important;}
    [data-testid="collapsedControl"] {display: none !important;}

    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    [data-testid="stSidebar"] * {
        color: #2D3748 !important;
    }
    .sidebar-section {
        font-size: 0.7rem;
        font-weight: 600;
        color: #A0AEC0 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 1rem 0 0.5rem;
    }

    [data-testid="stSidebar"] .stButton > button {
        background: #F7FAFC !important;
        color: #2D3748 !important;
        border: 1.5px solid #E2E8F0 !important;
        border-radius: 8px !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        transition: all 0.2s !important;
        box-shadow: none !important;
        padding: 0.5rem 0.8rem !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #EBF4FF !important;
        border-color: #87CEEB !important;
        color: #1A6B8A !important;
    }

    .banner-wrap {
        border-radius: 16px;
        overflow: hidden;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    .banner-wrap img {
        width: 100%;
        display: block;
        max-height: 260px;
        object-fit: contain;
    }

    .hero-section {
        background: transparent;
        padding: 0.5rem 0 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .hero-section h1 {
        color: #1A1A2E;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 1rem;
    }
    .hero-logo { display: flex; justify-content: center; margin-top: 0.5rem; }

    .status-ok {
        background: #F0FFF4;
        border: 1px solid #9AE6B4;
        border-left: 4px solid #38A169;
        border-radius: 10px;
        padding: 0.55rem 1rem;
        color: #276749;
        font-size: 0.82rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    .status-err {
        background: #FFF5F5;
        border: 1px solid #FEB2B2;
        border-left: 4px solid #E53E3E;
        border-radius: 10px;
        padding: 0.55rem 1rem;
        color: #C53030;
        font-size: 0.82rem;
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1A1A2E;
        margin: 1.2rem 0 0.6rem;
    }

    .stButton > button {
        background: white !important;
        color: #4A5568 !important;
        border: 1.5px solid #E2E8F0 !important;
        border-radius: 6px !important;
        font-size: 0.82rem !important;
        font-weight: 400 !important;
        transition: all 0.2s !important;
        text-align: left !important;
        line-height: 1.4 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }
    .stButton > button:hover {
        background: #F7FAFC !important;
        border-color: #87CEEB !important;
        color: #1A6B8A !important;
    }

    .stButton > button[kind="primary"] {
        background: #87CEEB !important;
        color: #1A3A5C !important;
        border: none !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        border-radius: 99px !important;
        box-shadow: 0 3px 12px rgba(135,206,235,0.4) !important;
        transition: background 0.2s, transform 0.2s !important;
        padding: 0.55rem 1rem !important;
        text-align: center !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #7C4DFF !important;
        color: white !important;
        transform: translateY(-1px) !important;
    }

    .stTextArea textarea {
        border: 1.5px solid #E2E8F0 !important;
        border-radius: 12px !important;
        background: #FAFAFA !important;
        font-size: 0.9rem !important;
        color: #2D3748 !important;
        box-shadow: none !important;
    }
    .stTextArea textarea:focus {
        border-color: #87CEEB !important;
        box-shadow: 0 0 0 3px rgba(135,206,235,0.15) !important;
        background: white !important;
    }

    .chat-user {
        background: #E8F4FD;
        border-radius: 12px 12px 2px 12px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #1A3A5C;
        text-align: right;
    }
    .chat-bot {
        background: #F8F9FA;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #87CEEB;
        border-radius: 2px 12px 12px 12px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #2D3748;
        line-height: 1.7;
    }
    .chat-meta {
        font-size: 0.7rem;
        color: #A0AEC0;
        margin-top: 0.3rem;
    }

    .respuesta-box {
        background: white;
        border: 1.5px solid #E2E8F0;
        border-top: 4px solid #87CEEB;
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }
    .respuesta-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #87CEEB;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.75rem;
    }
    .respuesta-text { color: #2D3748; line-height: 1.8; font-size: 0.95rem; }

    .metrics-row {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 8px;
        margin-top: 1rem;
    }
    .metric-card {
        background: #FAFAFA;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0.75rem;
        text-align: center;
    }
    .metric-val { font-size:1rem; font-weight:700; color:#1A6B8A; display:block; }
    .metric-lbl { font-size:0.68rem; color:#A0AEC0; margin-top:2px; display:block; }

    .custom-divider {
        height: 1px;
        background: linear-gradient(to right, transparent, #E2E8F0, transparent);
        margin: 1.2rem 0;
    }

    .footer {
        text-align: center;
        background: #1A1A1A;
        color: #FFFFFF;
        font-size: 1rem;
        font-weight: 500;
        margin-top: 2rem;
        padding: 1.2rem;
        border-radius: 10px;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    logo_rec_tag = get_img_tag(
        str(ASSETS / "recamier-logo.png"),
        width="140px",
        extra_style="filter:brightness(0); margin-bottom:1.5rem; display:block;"
    )
    if logo_rec_tag:
        st.markdown(logo_rec_tag, unsafe_allow_html=True)
    else:
        st.markdown("**Recamier**")

    st.markdown('<div class="sidebar-section">Conversaciones</div>',
                unsafe_allow_html=True)

    if st.button("+ Nueva conversación", use_container_width=True, key="nueva_sidebar"):
        nuevo_sid = str(uuid.uuid4())[:8]
        st.session_state["sesiones"][nuevo_sid] = {
            "nombre": "Nueva conversación",
            "historial": []
        }
        st.session_state["sesion_actual"] = nuevo_sid
        st.session_state["session_id"] = nuevo_sid
        st.session_state["pregunta_actual"] = ""
        st.rerun()

    st.markdown("---")

    for sid, datos in list(st.session_state["sesiones"].items()):
        if not datos["historial"]:
            continue
        if st.button(datos["nombre"], key=f"ses_{sid}", use_container_width=True):
            st.session_state["sesion_actual"] = sid
            st.session_state["session_id"] = sid
            st.rerun()

# ---------------------------------------------------------------------------
# CONTENIDO PRINCIPAL
# ---------------------------------------------------------------------------
banner_tag = get_img_tag(BANNER, width="100%",
                         extra_style="max-height:300px;object-fit:contain;display:block;background:#fff;")
if banner_tag:
    st.markdown(f'<div class="banner-wrap">{banner_tag}</div>',
                unsafe_allow_html=True)

logo_salonin_tag = get_img_tag(LOGO_SALONIN, width="auto",
                                extra_style="max-width:100%; display:block;")
st.markdown(f"""
<div class="hero-section">
    <h1>Asistente de Productos</h1>
    <div class="hero-logo">{logo_salonin_tag}</div>
</div>
""", unsafe_allow_html=True)

# Estado API
try:
    health = requests.get(HEALTH_URL, timeout=3)
    if health.status_code == 200:
        st.markdown(
            '<div class="status-ok">Sistema conectado y funcionando correctamente</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="status-err">El sistema respondió con un estado inesperado</div>',
            unsafe_allow_html=True)
except Exception:
    st.markdown("""
    <div class="status-err">
        No se puede conectar con la API —
        Ejecuta: <code>uvicorn api.main:app --reload --host 127.0.0.1 --port 8081</code>
    </div>""", unsafe_allow_html=True)

# Historial
historial_actual = st.session_state["sesiones"].get(
    st.session_state["sesion_actual"], {}).get("historial", [])

if historial_actual:
    st.markdown('<div class="section-title">Conversación</div>',
                unsafe_allow_html=True)
    for msg in historial_actual:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user">🧑 {msg["content"]}</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="chat-bot">{msg["content"]}'
                f'<div class="chat-meta">⏱️ {msg.get("latencia", "")}s · {MODELO_ACTIVO}</div>'
                f'</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# Preguntas frecuentes
st.markdown('<div class="section-title">Preguntas frecuentes</div>',
            unsafe_allow_html=True)

ejemplos = [
    "¿Qué productos de Recamier son buenos para el cabello seco?",
    "¿Cómo funciona la keratina de Recamier?",
    "¿Qué diferencia hay entre Recamier y Salon In Professional?",
    "¿Qué productos recomiendas para el cabello teñido?",
    "¿Tienen productos para el crecimiento del cabello?",
    "¿Cómo puedo hidratar mi cabello en casa?",
]

cols = st.columns(3)
for i, ejemplo in enumerate(ejemplos):
    if cols[i % 3].button(ejemplo, key=f"ej_{i}", use_container_width=True):
        st.session_state["pregunta_actual"] = ejemplo

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# Nueva consulta
st.markdown('<div class="section-title">Nueva consulta</div>',
            unsafe_allow_html=True)

pregunta = st.text_area(
    label="Pregunta",
    value=st.session_state.get("pregunta_actual", ""),
    placeholder="Escribe tu pregunta sobre productos Recamier o Salon In Professional...",
    height=110,
    label_visibility="collapsed",
)

consultar = st.button("Consultar", type="primary", use_container_width=True)

if consultar and pregunta.strip():
    with st.spinner("Analizando y generando respuesta..."):
        try:
            response = requests.post(
                API_URL,
                json={
                    "pregunta": pregunta.strip(),
                    "session_id": st.session_state["session_id"]
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
                        "nombre": "Nueva conversación",
                        "historial": []
                    }
                st.session_state["sesiones"][sid]["historial"].append({
                    "role": "user", "content": pregunta.strip()
                })
                st.session_state["sesiones"][sid]["historial"].append({
                    "role": "assistant",
                    "content": result["respuesta"],
                    "latencia": result["latencia_segundos"]
                })
                if len(st.session_state["sesiones"][sid]["historial"]) == 2:
                    st.session_state["sesiones"][sid]["nombre"] = pregunta.strip()[:30]

                st.session_state["pregunta_actual"] = ""

                st.markdown(f"""
                <div class="respuesta-box">
                    <div class="respuesta-label">Respuesta del Asistente</div>
                    <div class="respuesta-text">{result['respuesta']}</div>
                </div>
                <div class="metrics-row">
                    <div class="metric-card">
                        <span class="metric-val">{result['latencia_segundos']}s</span>
                        <span class="metric-lbl">Tiempo de respuesta</span>
                    </div>
                    <div class="metric-card">
                        <span class="metric-val">2 sitios</span>
                        <span class="metric-lbl">Fuentes indexadas</span>
                    </div>
                    <div class="metric-card">
                        <span class="metric-val">{MODELO_ACTIVO}</span>
                        <span class="metric-lbl">Modelo activo</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.rerun()

        except requests.exceptions.ConnectionError:
            st.error("No se pudo conectar con la API en :8081")
        except requests.exceptions.Timeout:
            st.warning("La consulta tardó demasiado. Intenta de nuevo.")

elif consultar and not pregunta.strip():
    st.warning("Escribe una pregunta antes de consultar.")

# Footer
st.markdown("""
<div class="footer">
    Recamier · Salon In Professional · Transformación Digital · 2026
</div>
""", unsafe_allow_html=True)
