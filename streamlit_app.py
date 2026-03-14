import sys
import os
import base64
import numpy as np
import cv2
import pandas as pd
import streamlit as st
import pydeck as pdk
from PIL import Image

# Core module imports
try:
    from ai.predict import predict_crack
    from backend.risk_assessment import risk_level
    from utils.image_processing import get_stress_heatmap, get_canny_edges
except ImportError as e:
    st.error(f"Import Error: {e}")

# --- ASSETS & CONFIG ---
LOGO_PATH = os.path.join("frontend", "assets", "logo.png")
BG_IMAGE_PATH = os.path.join("frontend", "assets", "building.jpg")

st.set_page_config(
    page_title="InfraGuard AI",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🏗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Session State
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Overview"

# --- UTILS ---
def get_base64_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def get_high_intensity_heatmap(image: Image.Image):
    img_cv = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    dist_input = cv2.bitwise_not(edges)
    dist_transform = cv2.distanceTransform(dist_input, cv2.DIST_L2, 5)
    influence_radius = 60.0
    stress_field = np.clip(1.0 - (dist_transform / influence_radius), 0, 1)
    stress_field = np.power(stress_field, 0.7) 
    influence_mask = (stress_field * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(influence_mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_cv, 0.4, heatmap_color, 0.6, 0)
    return overlay

# --- THEME STYLING ---
def apply_custom_styles(is_login=False):
    bg_style = ""
    if is_login and os.path.exists(BG_IMAGE_PATH):
        try:
            bin_str = get_base64_bin_file(BG_IMAGE_PATH)
            bg_style = f"background-image: url('data:image/jpg;base64,{bin_str}'); background-size: cover; background-position: center; background-attachment: fixed;"
        except Exception:
            bg_style = "background-color: #f1f5f9;"
    else:
        bg_style = "background-color: #f8fafc;"

    css_content = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        .stApp {{ {bg_style} color: #1e293b; font-family: 'Plus Jakarta Sans', sans-serif; }}
        {" .stApp::before { content: ''; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.45); z-index: -1; }" if is_login else ""}
        header, #stDecoration, footer, [data-testid="stSidebar"] {{ display: none !important; visibility: hidden !important; }}
        .stMainBlockContainer {{ padding-top: 0 !important; padding-bottom: 2rem !important; }}
        .login-header {{ text-align: center; width: 100%; }}
        .dash-card {{ background: white; padding: 2.5rem; border-radius: 20px; border: 1px solid #e2e8f0; margin-bottom: 2rem; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05); }}
        .metric-box {{ background: #ffffff; padding: 1.5rem; border-radius: 15px; text-align: center; border: 1px solid #e2e8f0; box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.05); }}
        .risk-tag {{ padding: 0.7rem 2.5rem; border-radius: 50px; font-weight: 800; font-size: 1rem; display: inline-block; margin-top: 15px; letter-spacing: 0.8px; }}
        .risk-HIGH {{ background: #fee2e2; color: #dc2626; border: 1px solid #dc2626; }}
        .risk-MODERATE {{ background: #fef3c7; color: #d97706; border: 1px solid #d97706; }}
        .risk-LOW {{ background: #dcfce7; color: #16a34a; border: 1px solid #16a34a; }}
        h1, h2, h3, h4 {{ margin-bottom: 1.2rem !important; color: #1e1b4b; }}
        p {{ line-height: 1.8; color: #475569; font-size: 1.1rem; }}
        .maintenance-panel {{ background: #ffffff; border-left: 8px solid #dc2626; padding: 2.5rem; border-radius: 0 12px 12px 0; margin-top: 2rem; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05); }}
        .measure-item {{ margin-bottom: 1.2rem; font-weight: 500; display: flex; align-items: center; gap: 1rem; }}
        .measure-bullet {{ color: #dc2626; font-size: 1.8rem; }}
    </style>
    """
    st.markdown(css_content, unsafe_allow_html=True)

# --- LOGIN PAGE ---
def render_login():
    apply_custom_styles(is_login=True)
    st.markdown("<div style='height: 20vh;'></div>", unsafe_allow_html=True)
    st.markdown(\"\"\"
    <div class='login-header'>
        <h1 style='color: #dc2626; font-size: 7rem; font-weight: 900; margin: 0; text-align: center;'>InfraGuard AI</h1>
        <p style='color: #000000; font-size: 2rem; font-weight: 700; margin-top: 0.5rem; margin-bottom: 4rem; text-align: center;'>Infrastructure Monitoring Portal</p>
    </div>
    \"\"\", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        email = st.text_input("Email", placeholder="Email Address", label_visibility="collapsed")
        password = st.text_input("Password", type="password", placeholder="Password", label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("PORTAL ACCESS", use_container_width=True):
            st.session_state.authenticated = True
            st.rerun()

# --- NAVIGATION ---
def render_navbar():
    l, r = st.columns([1.5, 3])
    with l:
        st.markdown("<h2 style='margin:10px 0; color:#1e1b4b; font-weight:800;'>🏗️ InfraGuard AI</h2>", unsafe_allow_html=True)
    with r:
        n1, n2, n3, n4, n5 = st.columns([1, 1, 1, 1, 1])
        if n1.button("Overview", use_container_width=True): st.session_state.current_page = "Overview"
        if n2.button("Terminal", use_container_width=True): st.session_state.current_page = "Terminal"
        if n3.button("Surveillance", use_container_width=True): st.session_state.current_page = "Surveillance"
        if n4.button("GeoWatch", use_container_width=True): st.session_state.current_page = "GeoWatch"
        if n5.button("Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()
    st.markdown("---")

# --- PAGES ---

def page_overview():
    st.title("System Methodology & Documentation")
    st.markdown(\"\"\"
    <div class='dash-card'>
        <h2>InfraGuard AI Operating Paradigm</h2>
        <p>InfraGuard AI represents a critical advancement in autonomous structural health monitoring (SHM). Our platform provides high-fidelity diagnostic data for bridge spans, industrial foundations, and high-load civil assets.</p>
    </div>
    \"\"\", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(\"\"\"
        <div class='dash-card'>
            <h3>1. Neural Crack Detection</h3>
            <p>Our inference engine utilizes state-of-the-art CNN architectures specifically trained on macro and micro-fracture datasets.</p>
        </div>
        \"\"\", unsafe_allow_html=True)
        st.markdown(\"\"\"
        <div class='dash-card'>
            <h3>2. Predictive Risk Modeling</h3>
            <p>The system computes risk coefficients by analyzing fracture density and orientation thresholds.</p>
        </div>
        \"\"\", unsafe_allow_html=True)

    with c2:
        st.markdown(\"\"\"
        <div class='dash-card'>
            <h3>3. Visual Auditing Techniques</h3>
            <p>We provide <b>Edge Mapping</b> for exact path recovery and <b>Radiance Heatmapping</b> to visualize the structural influence zone.</p>
        </div>
        \"\"\", unsafe_allow_html=True)
        st.markdown(\"\"\"
        <div class='dash-card'>
            <h3>4. Catastrophic Failure Prevention</h3>
            <p>Early identified micro-fatigue precursors allow for localized intervention, protecting both public safety and long-term asset fiscality.</p>
        </div>
        \"\"\", unsafe_allow_html=True)

def page_terminal():
    st.title("Diagnostic Analysis Terminal")
    st.markdown("Initiate structural audit by providing high-resolution asset captures.")
    st.markdown("---")
    file_up = st.file_uploader("Imagery Feed", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if file_up:
        img = Image.open(file_up)
        with st.spinner("Processing structural diagnostics..."):
            _, conf, crack_prob = predict_crack(img)
            h_score = int(max(0, min(100, (1.0 - float(crack_prob)) * 100.0)))
            if h_score > 75: risk_tier = "LOW"
            elif 40 <= h_score <= 75: risk_tier = "MODERATE"
            else: risk_tier = "HIGH"
            edges = get_canny_edges(img)
            damage_viz = get_high_intensity_heatmap(img)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Asset Capture Feed"); st.image(img, use_container_width=True)
            st.subheader("Structural Path (Edge Map)"); st.image(edges, use_container_width=True, caption="Fracture Geometry Recovery")
        with col2:
            st.subheader("High-Intensity Damage Highlight"); st.image(damage_viz, use_container_width=True, caption="Highlighted Affected Areas")
            st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:#64748b; font-weight:700; margin:0;'>Estimated Structural Health</p>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='margin:0; color:#1e1b4b; font-size:4.5rem;'>{h_score}%</h1>", unsafe_allow_html=True)
            st.progress(h_score / 100)
            st.markdown(f"<span class='risk-tag risk-{risk_tier}'>{risk_tier} RISK ASSESSMENT</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---"); st.subheader("Maintenance Directives / Preventive Measures")
        if h_score > 75: m_color = "#16a34a"; m_list = ["Quarterly visual audit.", "Protective sealant application.", "Log harmonic baseline.", "Drainage verification.", "Annual sensor calibration."]
        elif 40 <= h_score <= 75: m_color = "#d97706"; m_list = ["Epoxy path node injection.", "Audit fracture depth profile.", "Transit load restrictions.", "Material resurfacing.", "Monthly diagnostice cycle."]
        else: m_color = "#dc2626"; m_list = ["RESTRICT ACCESS IMMEDIATELY.", "Support shoring deployment.", "Core-drilling material audit.", "Real-time strain grid install.", "Architectural reinforcement."]
        st.markdown(f\"\"\"
        <div class='maintenance-panel' style='border-left-color: {m_color};'>
            <div class='measure-item'><span class='measure-bullet'>•</span> <span>{m_list[0]}</span></div>
            <div class='measure-item'><span class='measure-bullet'>•</span> <span>{m_list[1]}</span></div>
            <div class='measure-item'><span class='measure-bullet'>•</span> <span>{m_list[2]}</span></div>
            <div class='measure-item'><span class='measure-bullet'>•</span> <span>{m_list[3]}</span></div>
            <div class='measure-item'><span class='measure-bullet'>•</span> <span>{m_list[4]}</span></div>
        </div>\"\"\", unsafe_allow_html=True)

def page_surveillance():
    st.title("Direct Surveillance & Monitoring")
    cam_in = st.camera_input("Optical Sensor Node-01")
    if cam_in:
        st.success("High-Resolution Asset Capture Received.")
        if st.button("SEND TO ANALYSIS TERMINAL", use_container_width=True):
            st.session_state.captured_image = Image.open(cam_in); st.session_state.current_page = "Terminal"; st.rerun()

def page_geowatch():
    st.title("GeoWatch Global Monitor")
    data = pd.DataFrame({'name': ['Hudson Span', 'Metro Viaduct'], 'lat': [40.7128, 40.7829], 'lon': [-74.0060, -73.9654], 'risk': ['LOW', 'LOW'], 'color': [[22, 163, 74, 200], [22, 163, 74, 200]]})
    view = pdk.ViewState(latitude=40.75, longitude=-73.97, zoom=10, pitch=45)
    layer = pdk.Layer("ScatterplotLayer", data, get_position=["lon", "lat"], get_color="color", get_radius=800, pickable=True)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view))

# --- BOOTSTRAP ---
if not st.session_state.authenticated:
    render_login()
else:
    apply_custom_styles(is_login=False)
    render_navbar()
    if st.session_state.current_page == "Overview": page_overview()
    elif st.session_state.current_page == "Terminal": page_terminal()
    elif st.session_state.current_page == "Surveillance": page_surveillance()
    elif st.session_state.current_page == "GeoWatch": page_geowatch()
