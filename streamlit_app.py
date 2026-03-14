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
        .metric-box {{ background: #ffffff; padding: 1.5rem; border-radius: 15px; text-align: center; border: 1px solid #e2e8f0; box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.05); }}
        .risk-tag {{ padding: 0.7rem 2.5rem; border-radius: 50px; font-weight: 800; font-size: 1rem; display: inline-block; margin-top: 15px; letter-spacing: 0.8px; }}
        .risk-HIGH {{ background: #fee2e2; color: #dc2626; border: 1px solid #dc2626; }}
        .risk-MODERATE {{ background: #fef3c7; color: #d97706; border: 1px solid #d97706; }}
        .risk-LOW {{ background: #dcfce7; color: #16a34a; border: 1px solid #16a34a; }}
        h1, h2, h3, h4 {{ margin-bottom: 1.2rem !important; color: #1e1b4b; }}
        p {{ line-height: 1.8; color: #475569; font-size: 1.1rem; }}
        .maintenance-panel {{ background: #ffffff; border-left: 8px solid #dc2626; padding: 2.5rem; border-radius: 0 12px 12px 0; margin-top: 2rem; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05); }}
    </style>
    """
    st.markdown(css_content, unsafe_allow_html=True)

# --- LOGIN PAGE ---
def render_login():
    apply_custom_styles(is_login=True)
    st.markdown("<div style='height: 20vh;'></div>", unsafe_allow_html=True)
    st.markdown('''
    <div class='login-header'>
        <h1 style='color: #dc2626; font-size: 7rem; font-weight: 900; margin: 0; text-align: center;'>InfraGuard AI</h1>
        <p style='color: #000000; font-size: 2rem; font-weight: 700; margin-top: 0.5rem; margin-bottom: 4rem; text-align: center;'>Infrastructure Monitoring Portal</p>
    </div>
    ''', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        st.text_input("Email", placeholder="Email Address", key="email", label_visibility="collapsed")
        st.text_input("Password", type="password", placeholder="Password", key="password", label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("PORTAL ACCESS", use_container_width=True):
            st.session_state.authenticated = True
            st.rerun()

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
def page_terminal():
    st.title("Diagnostic Analysis Terminal")
    st.markdown("Initiate structural audit by providing high-resolution asset captures.")
    file_up = st.file_uploader("Imagery Feed", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if file_up:
        img = Image.open(file_up)
        with st.spinner("Processing structural diagnostics..."):
            _, conf, crack_prob = predict_crack(img)
            h_score = int(max(0, min(100, (1.0 - float(crack_prob)) * 100.0)))
            if h_score > 75: risk_tier = "LOW"
            elif 40 <= h_score <= 75: risk_tier = "MODERATE"
            else: risk_tier = "HIGH"
            edges = get_canny_edges(img); damage_viz = get_high_intensity_heatmap(img)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Asset Feed"); st.image(img, use_container_width=True)
            st.subheader("Edge Map"); st.image(edges, use_container_width=True)
        with col2:
            st.subheader("Damage Heatmap"); st.image(damage_viz, use_container_width=True)
            st.markdown(f'''
            <div class='metric-box'>
                <p style='color:#64748b; font-weight:700;'>Estimated Structural Health</p>
                <h1 style='color:#1e1b4b; font-size:4.5rem;'>{h_score}%</h1>
                <p><span class='risk-tag risk-{risk_tier}'>{risk_tier} RISK</span></p>
            </div>''', unsafe_allow_html=True)

def page_overview():
    st.title("System Methodology")
    st.markdown("<div class='dash-card'><h2>Operating Paradigm</h2><p>High-fidelity structural health monitoring.</p></div>", unsafe_allow_html=True)

def page_surveillance():
    st.title("Direct Surveillance"); cam_in = st.camera_input("Optical Sensor")
    if cam_in and st.button("SEND TO ANALYSIS"): 
        st.session_state.captured_image = Image.open(cam_in); st.session_state.current_page = "Terminal"; st.rerun()

def page_geowatch():
    st.title("GeoWatch"); data = pd.DataFrame({'lat': [40.7128], 'lon': [-74.0060]})
    st.pydeck_chart(pdk.Deck(layers=[pdk.Layer("ScatterplotLayer", data, get_position=["lon", "lat"], get_radius=500)], initial_view_state=pdk.ViewState(latitude=40.75, longitude=-73.97, zoom=10)))

# --- BOOTSTRAP ---
if not st.session_state.authenticated:
    render_login()
else:
    apply_custom_styles(is_login=False); render_navbar()
    if st.session_state.current_page == "Overview": page_overview()
    elif st.session_state.current_page == "Terminal": page_terminal()
    elif st.session_state.current_page == "Surveillance": page_surveillance()
    elif st.session_state.current_page == "GeoWatch": page_geowatch()
