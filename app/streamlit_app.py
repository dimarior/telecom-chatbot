"""
app/streamlit_app.py — Telecom Chatbot Colombia
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
ASSETS = Path("app/assets")

OPERADORES = {
    "todos": {
        "nombre": "Todos los operadores",
        "color": "#2D3748",
        "emoji": "📡",
        "placeholder": "Pregunta sobre Claro, Movistar o Tigo...",
    },
    "claro": {
        "nombre": "Claro",
        "color": "#E8002D",
        "emoji": "🔴",
        "placeholder": "Pregunta sobre planes, soporte o servicios de Claro...",
    },
    "movistar": {
        "nombre": "Movistar",
        "color": "#009BDE",
        "emoji": "🔵",
        "placeholder": "Pregunta sobre planes, soporte o servicios de Movistar...",
    },
    "tigo": {
        "nombre": "Tigo",
        "color": "#00377B",
        "emoji": "🟡",
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
    ext = "jpg" if path.endswith(".jpg") else "png"
    return f'<img src="data:image/{ext};base64,{b64}" style="width:{width};{extra_style}" />'


try:
    favicon = Image.open(str(ASSETS / "telecom-favicon.ico"))
    st.set_page_config(
        page_title="Asistente Telecom Colombia",
        page_icon=favicon,
        layout="wide",
    )
except Exception:
    st.set_page_config(
        page_title="Asistente Telecom Colombia",
        page_icon="📡",
        layout="wide",
    )

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())[:8]
if "sesiones" not in st.session_state:
    sid = st.session_state["session_id"]
    st.session_state["sesiones"] = {sid: {"nombre": "Nueva conversación", "historial": []}}
    st.session_state["sesion_actual"] = sid
if "pregunta_actual" not in st.session_state:
    st.session_state["pregunta_actual"] = ""
if "operador_actual" not in st.session_state:
    st.session_state["operador_actual"] = "todos"

operador = st.session_state["operador_actual"]
color_op = OPERADORES[operador]["color"]

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * {{ font-family: 'Inter', sans-serif; }}
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    section[data-testid="stMain"] > div {{ background-color: #FFFFFF !important; }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stDeployButton {{display: none !important;}}
    header[data-testid="stHeader"] {{display: none !important;}}
    [data-testid="collapsedControl"] {{display: none !important;}}
    [data-testid="stSidebar"] {{
        background-color: #FAFAFA !important;
        border-right: 1px solid #E2E8F0 !important;
    }}
    [data-testid="stSidebar"] * {{ color: #2D3748 !important; }}
    .sidebar-section {{
        font-size: 0.7rem; font-weight: 600; color: #A0AEC0 !important;
        text-transform: uppercase; letter-spacing: 2px; margin: 1rem 0 0.5rem;
    }}
    [data-testid="stSidebar"] .stButton > button {{
        background: #F7FAFC !important; color: #2D3748 !important;
        border: 1.5px solid #E2E8F0 !important; border-radius: 8px !important;
        font-size: 0.85rem !important; font-weight: 500 !important;
        text-align: left !important; transition: all 0.2s !important;
    }}
    .operador-card {{
        border: 2px solid #E2E8F0; border-radius: 12px; padding: 0.8rem;
        text-align: center; cursor: pointer; transition: all 0.2s;
        background: white; margin-bottom: 0.5rem;
    }}
    .operador-card.activo {{
        border-color: {color_op}; background: {color_op}15;
    }}
    .operador-nombre {{ font-size: 0.9rem; font-weight: 600; color: #2D3748; }}
    .hero-section {{
        padding: 0.5rem 0 1rem; text-align: center; margin-bottom: 0.5rem;
    }}
    .hero-section h1 {{ color: #1A1A2E; font-size: 1.8rem; font-weight: 700; margin: 0; }}
    .hero-section p {{ color: #718096; font-size: 0.9rem; margin-top: 0.3rem; }}
    .status-ok {{
        background: #F0FFF4; border: 1px solid #9AE6B4;
        border-left: 4px solid #38A169; border-radius: 10px;
        padding: 0.55rem 1rem; color: #276749; font-size: 0.82rem;
        font-weight: 500; margin-bottom: 1rem;
    }}
    .status-err {{
        background: #FFF5F5; border: 1px solid #FEB2B2;
        border-left: 4px solid #E53E3E; border-radius: 10px;
        padding: 0.55rem 1rem; color: #C53030; font-size: 0.82rem; margin-bottom: 1rem;
    }}
    .section-title {{
        font-size: 0.95rem; font-weight: 600; color: #1A1A2E; margin: 1.2rem 0 0.6rem;
    }}
    .stButton > button {{
        background: white !important; color: #4A5568 !important;
        border: 1.5px solid #E2E8F0 !important; border-radius: 6px !important;
        font-size: 0.82rem !important; transition: all 0.2s !important;
        text-align: left !important; box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }}
    .stButton > button[kind="primary"] {{
        background: {color_op} !important; color: white !important;
        border: none !important; font-size: 0.95rem !important;
        font-weight: 600 !important; border-radius: 99px !important;
        box-shadow: 0 3px 12px {color_op}40 !important;
        text-align: center !important;
    }}
    .stTextArea textarea {{
        border: 1.5px solid #E2E8F0 !important; border-radius: 12px !important;
        background: #FAFAFA !important; font-size: 0.9rem !important;
    }}
    .stTextArea textarea:focus {{
        border-color: {color_op} !important;
        box-shadow: 0 0 0 3px {color_op}25 !important;
    }}
    .chat-user {{
        background: #F7FAFC; border-radius: 12px 12px 2px 12px;
        padding: 0.75rem 1rem; margin: 0.5rem 0;
        font-size: 0.9rem; color: #2D3748; text-align: right;
    }}
    .chat-bot {{
        background: #F8F9FA; border: 1px solid #E2E8F0;
        border-left: 4px solid {color_op};
        border-radius: 2px 12px 12px 12px; padding: 0.75rem 1rem;
        margin: 0.5rem 0; font-size: 0.9rem; color: #2D3748; line-height: 1.7;
    }}
    .chat-meta {{ font-size: 0.7rem; color: #A0AEC0; margin-top: 0.3rem; }}
    .respuesta-box {{
        background: white; border: 1.5px solid #E2E8F0;
        border-top: 4px solid {color_op}; border-radius: 16px;
        padding: 1.5rem; margin-top: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }}
    .respuesta-label {{
        font-size: 0.7rem; font-weight: 600; color: {color_op};
        text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.75rem;
    }}
    .respuesta-text {{ color: #2D3748; line-height: 1.8; font-size: 0.95rem; }}
    .metrics-row {{
        display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-top: 1rem;
    }}
    .metric-card {{
        background: #FAFAFA; border: 1px solid #E2E8F0;
        border-radius: 12px; padding: 0.75rem; text-align: center;
    }}
    .metric-val {{ font-size:1rem; font-weight:700; color:{color_op}; display:block; }}
    .metric-lbl {{ font-size:0.68rem; color:#A0AEC0; margin-top:2px; display:block; }}
    .custom-divider {{
        height: 1px;
        background: linear-gradient(to right, transparent, #E2E8F0, transparent);
        margin: 1.2rem 0;
    }}
    .footer {{
        text-align: center; background: #1A1A1A; color: #FFFFFF;
        font-size: 0.9rem; margin-top: 2rem; padding: 1.2rem;
        border-radius: 10px;
    }}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Telecom Colombia")
    st.markdown('<div class="sidebar-section">Selecciona operador</div>', unsafe_allow_html=True)

    for key, op in OPERADORES.items():
        activo = "activo" if st.session_state["operador_actual"] == key else ""
        if st.button(
            f"{op['emoji']} {op['nombre']}",
            key=f"op_{key}",
            use_container_width=True,
        ):
            st.session_state["operador_actual"] = key
            st.session_state["pregunta_actual"] = ""
            st.rerun()

    st.markdown("---")
    st.markdown('<div class="sidebar-section">Conversaciones</div>', unsafe_allow_html=True)

    if st.button("+ Nueva conversación", use_container_width=True, key="nueva_sidebar"):
        nuevo_sid = str(uuid.uuid4())[:8]
        st.session_state["sesiones"][nuevo_sid] = {"nombre": "Nueva conversación", "historial": []}
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

# ── MAIN ──────────────────────────────────────────────────────────────────────
op_info = OPERADORES[st.session_state["operador_actual"]]

st.markdown(f"""
<div class="hero-section">
    <h1>{op_info['emoji']} Asistente Telecom Colombia</h1>
    <p>Servicio al cliente · {op_info['nombre']}</p>
</div>
""", unsafe_allow_html=True)

try:
    health = requests.get(HEALTH_URL, timeout=3)
    if health.status_code == 200:
        st.markdown('<div class="status-ok">✅ Sistema conectado y funcionando</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-err">⚠️ Estado inesperado de la API</div>', unsafe_allow_html=True)
except Exception:
    st.markdown(f"""
    <div class="status-err">
        ⚡ API no disponible —
        <code>uvicorn api.main:app --reload --host 127.0.0.1 --port 8081</code>
    </div>""", unsafe_allow_html=True)

# Historial
historial_actual = st.session_state["sesiones"].get(
    st.session_state["sesion_actual"], {}).get("historial", [])

if historial_actual:
    st.markdown('<div class="section-title">Conversación</div>', unsafe_allow_html=True)
    for msg in historial_actual:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="chat-bot">{msg["content"]}'
                f'<div class="chat-meta">⏱️ {msg.get("latencia", "")}s · {MODELO_ACTIVO}</div>'
                f'</div>', unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# Preguntas sugeridas por operador
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

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Tu consulta</div>', unsafe_allow_html=True)

pregunta = st.text_area(
    label="Pregunta",
    value=st.session_state.get("pregunta_actual", ""),
    placeholder=op_info["placeholder"],
    height=110,
    label_visibility="collapsed",
)

consultar = st.button("Consultar ✨", type="primary", use_container_width=True)

if consultar and pregunta.strip():
    # Enriquecer pregunta con contexto del operador
    operador_sel = st.session_state["operador_actual"]
    pregunta_enriquecida = pregunta.strip()
    if operador_sel != "todos":
        pregunta_enriquecida = f"[{OPERADORES[operador_sel]['nombre']}] {pregunta.strip()}"

    with st.spinner("Buscando información y generando respuesta..."):
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
                    st.session_state["sesiones"][sid] = {"nombre": "Nueva conversación", "historial": []}

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

                st.session_state["pregunta_actual"] = ""

                st.markdown(f"""
                <div class="respuesta-box">
                    <div class="respuesta-label">Respuesta — {op_info['nombre']}</div>
                    <div class="respuesta-text">{result['respuesta']}</div>
                </div>
                <div class="metrics-row">
                    <div class="metric-card">
                        <span class="metric-val">{result['latencia_segundos']}s</span>
                        <span class="metric-lbl">Tiempo de respuesta</span>
                    </div>
                    <div class="metric-card">
                        <span class="metric-val">3 operadores</span>
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
            st.error("❌ No se pudo conectar con la API en :8081")
        except requests.exceptions.Timeout:
            st.warning("⏳ La consulta tardó demasiado. Intenta de nuevo.")

elif consultar and not pregunta.strip():
    st.warning("Escribe una pregunta antes de consultar.")

st.markdown("""
<div class="footer">
    Claro · Movistar · Tigo · Servicio al Cliente · Colombia · 2026
</div>
""", unsafe_allow_html=True)
