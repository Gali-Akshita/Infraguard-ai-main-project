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
        header, #stDecoration, footer, [data-testid="stSidebar"] {{ display: none !important; visibility: hidden !important; }}
        .stMainBlockContainer {{ padding-top: 0 !important; padding-bottom: 2rem !important; }}
        .login-header {{ text-align: center; width: 100%; }}
        .dash-card {{ background: white; padding: 2.5rem; border-radius: 20px; border: 1px solid #e2e8f0; margin-bottom: 2rem; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05); }}
        .metric-box {{ background: #ffffff; padding: 1.5rem; border-radius: 15px; text-align: center; border: 1px solid #e2e8f0; box-shadow: inset 0 2px 4px 0 rgba(0,0,0,0.05); }}
        .risk-tag {{ padding: 0.7rem 2.5rem; border-radius: 50px; font-weight: 800; font-size: 1rem; display: inline-block; margin-top: 15px; letter-spacing: 0.1rem; }}
        .risk-HIGH {{ background: #fee2e2; color: #dc2626; border: 1px solid #dc2626; }}
        .risk-MODERATE {{ background: #fef3c7; color: #d97706; border: 1px solid #d97706; }}
        .risk-LOW {{ background: #dcfce7; color: #16a34a; border: 1px solid #16a34a; }}
        h1, h2, h3 {{ color: #1e1b4b; margin-bottom: 1rem !important; }}
        p {{ font-size: 1.1rem; line-height: 1.8; color: #475569; }}
    </style>
    """
    st.markdown(css_content, unsafe_allow_html=True)

# --- NAVIGATION ---
def render_navbar():
    l, r = st.columns([1.5, 3])
    with l: st.markdown("<h2 style='margin:10px 0; color:#1e1b4b; font-weight:800;'>🏗️ InfraGuard AI</h2>", unsafe_allow_html=True)
    with r:
        n1, n2, n3, n4, n5 = st.columns([1, 1, 1, 1, 1])
        if n1.button("Overview", use_container_width=True): st.session_state.current_page = "Overview"
        if n2.button("Terminal", use_container_width=True): st.session_state.current_page = "Terminal"
        if n3.button("Surveillance", use_container_width=True): st.session_state.current_page = "Surveillance"
        if n4.button("GeoWatch", use_container_width=True): st.session_state.current_page = "GeoWatch"
        if n5.button("Logout", use_container_width=True): st.session_state.authenticated = False; st.rerun()
    st.markdown("---")

# --- PAGES ---
def page_overview():
    st.title("System Methodology & Documentation")
    st.markdown('''
    <div class='dash-card'>
        <h2 style='color: #1e1b4b;'>InfraGuard AI: The Future of Structural Intelligence</h2>
        <p>InfraGuard AI is a comprehensive ecosystem designed for the next generation of Infrastructure Health Monitoring (IHM). By combining advanced computer vision, predictive analytics, and geospatial mapping, we provide asset managers with the tools needed to ensure structural longevity and public safety.</p>
    </div>
    ''', unsafe_allow_html=True)

    st.subheader("Core System Pillars")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('''
        <div class='dash-card'>
            <h3 style='color: #dc2626;'>🧠 Neural Crack Detection</h3>
            <p>Our proprietary CNN models are trained on millions of high-resolution images of concrete, steel, and masonry fatigue. The system doesn't just "see" cracks; it understands their depth, width, and structural significance in real-time.</p>
        </div>
        ''', unsafe_allow_html=True)
        st.markdown('''
        <div class='dash-card'>
            <h3 style='color: #2563eb;'>📊 Predictive Risk Modeling</h3>
            <p>Using a multi-factor risk assessment algorithm, InfraGuard AI calculates Structural Health Indexes (SHI). It accounts for fracture density, environmental stress factors, and baseline architectural tolerances.</p>
        </div>
        ''', unsafe_allow_html=True)
    with c2:
        st.markdown('''
        <div class='dash-card'>
            <h3 style='color: #059669;'>🔬 Visual Auditing & Heatmaps</h3>
            <p>We utilize edge geometry recovery to map the exact path of micro-fractures. Our radiance heatmaps visualize the "stress field"—highlighting areas where the structural integrity is most compromised by surrounding damage.</p>
        </div>
        ''', unsafe_allow_html=True)
        st.markdown('''
        <div class='dash-card'>
            <h3 style='color: #7c3aed;'>🛡️ Preventive Maintenance</h3>
            <p>The system generates immediate maintenance directives. Transitioning from reactive "damage control" to proactive "asset preservation" saves millions in catastrophic failure costs.</p>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Operational Workflow")
    wc1, wc2, wc3 = st.columns(3)
    with wc1:
        st.markdown("<div style='text-align:center;'><h1 style='color:#e2e8f0; font-size:4rem; margin:0;'>01</h1><h4>Data Ingestion</h4><p style='font-size:0.9rem;'>Imagery via mobile, drones, or stationary nodes.</p></div>", unsafe_allow_html=True)
    with wc2:
        st.markdown("<div style='text-align:center;'><h1 style='color:#e2e8f0; font-size:4rem; margin:0;'>02</h1><h4>AI Processing</h4><p style='font-size:0.9rem;'>Cloud-based GPU inference for real-time crack mapping.</p></div>", unsafe_allow_html=True)
    with wc3:
        st.markdown("<div style='text-align:center;'><h1 style='color:#e2e8f0; font-size:4rem; margin:0;'>03</h1><h4>Insight</h4><p style='font-size:0.9rem;'>Automatic generation of maintenance logs and alerts.</p></div>", unsafe_allow_html=True)

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
            if h_score > 80: risk_tier = "LOW"
            elif h_score < 50: risk_tier = "HIGH"
            else: risk_tier = "MODERATE"
            edges = get_canny_edges(img); damage_viz = get_high_intensity_heatmap(img)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Asset Feed"); st.image(img, use_container_width=True)
            st.subheader("Edge Map"); st.image(edges, use_container_width=True)
        with col2:
            st.subheader("Damage Heatmap"); st.image(damage_viz, use_container_width=True)
            st.markdown(f'''
            <div class='metric-box'>
                <p style='color:#64748b; font-weight:700; margin-bottom:0.5rem;'>ESTIMATED STRUCTURAL HEALTH</p>
                <h1 style='color:#1e1b4b; font-size:6rem; margin:0;'>{h_score}%</h1>
                <div style='margin-top:1rem;'><span class='risk-tag risk-{risk_tier}'>{risk_tier} RISK ASSESSMENT</span></div>
            </div>''', unsafe_allow_html=True)
        
        st.markdown("---"); st.subheader("Maintenance Directives / Preventive Measures")
        if risk_tier == "LOW": m_color = "#16a34a"; m_list = ["Quarterly visual audit.", "Protective sealant application.", "Log harmonic baseline.", "Drainage verification.", "Annual sensor calibration."]
        elif risk_tier == "MODERATE": m_color = "#d97706"; m_list = ["Epoxy path node injection.", "Audit fracture depth profile.", "Transit load restrictions.", "Material resurfacing.", "Monthly diagnostic cycle."]
        else: m_color = "#dc2626"; m_list = ["RESTRICT ACCESS IMMEDIATELY.", "Support shoring deployment.", "Core-drilling material audit.", "Real-time strain grid install.", "Architectural reinforcement."]
        for item in m_list:
            st.markdown(f"<div style='display:flex; align-items:center; gap:1rem; margin-bottom:0.8rem; padding-left:1rem; border-left:4px solid {m_color}; font-weight:500; color:#334155;'>• {item}</div>", unsafe_allow_html=True)

def page_geowatch():
    st.title("GeoWatch Global Monitor")
    data = pd.DataFrame({'name': ['Hudson Span', 'Bridge Sector-B', 'Metro Viaduct'], 'lat': [40.7128, 40.8448, 40.7829], 'lon': [-74.0060, -73.8648, -73.9654], 'risk': ['LOW', 'MODERATE', 'LOW'], 'color': [[22, 163, 74, 200], [217, 119, 6, 200], [22, 163, 74, 200]]})
    st.pydeck_chart(pdk.Deck(layers=[pdk.Layer("ScatterplotLayer", data, get_position=["lon", "lat"], get_color="color", get_radius=800, pickable=True)], initial_view_state=pdk.ViewState(latitude=40.75, longitude=-73.97, zoom=10, pitch=45)))
    st.markdown("---")
    st.subheader("Infrastructural Risk Classifications")
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown("<div style='padding:1.5rem; background:#dcfce7; border-radius:12px; border:1px solid #16a34a;'><h4 style='color:#16a34a; margin:0;'>🟢 STABLE (LOW)</h4><p style='color:#166534; font-size:0.9rem; margin-top:10px;'>Score <b>>80%</b>. Structural integrity is within nominal operating bounds. Routine maintenance suggested.</p></div>", unsafe_allow_html=True)
    with c2: st.markdown("<div style='padding:1.5rem; background:#fef3c7; border-radius:12px; border:1px solid #d97706;'><h4 style='color:#d97706; margin:0;'>🟠 MONITORING (MODERATE)</h4><p style='color:#92400e; font-size:0.9rem; margin-top:10px;'>Score <b>50% - 80%</b>. Surface fatigue detected. Requires specialized audit and localized repair.</p></div>", unsafe_allow_html=True)
    with c3: st.markdown("<div style='padding:1.5rem; background:#fee2e2; border-radius:12px; border:1px solid #dc2626;'><h4 style='color:#dc2626; margin:0;'>🔴 PRIORITY (HIGH)</h4><p style='color:#991b1b; font-size:0.9rem; margin-top:10px;'>Score <b><50%</b>. Deep fractures identified. Restricted transit and immediate structural shoring required.</p></div>", unsafe_allow_html=True)

# --- LOGIN PAGE ---
def render_login():
    apply_custom_styles(is_login=True)
    st.markdown("<div style='height: 20vh;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='login-header'><h1 style='color:#dc2626; font-size:7rem; font-weight:900;'>InfraGuard AI</h1><p style='color:#000; font-size:2rem; font-weight:700;'>Infrastructure Monitoring Portal</p></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        st.text_input("Email", placeholder="Email Address", key="email", label_visibility="collapsed")
        st.text_input("Password", type="password", placeholder="Password", key="password", label_visibility="collapsed")
        if st.button("PORTAL ACCESS", use_container_width=True): st.session_state.authenticated = True; st.rerun()

# --- BOOTSTRAP ---
if not st.session_state.authenticated:
    render_login()
else:
    apply_custom_styles(is_login=False); render_navbar()
    if st.session_state.current_page == "Overview": page_overview()
    elif st.session_state.current_page == "Terminal": page_terminal()
    elif st.session_state.current_page == "Surveillance": page_surveillance()
    elif st.session_state.current_page == "GeoWatch": page_geowatch()
