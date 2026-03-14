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
# Paths are now relative to the root entry point
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
        header, #stDecoration, footer, [data-testid="stSidebar"] {{ display: none !important; }}
        .dash-card {{ background: white; padding: 2.5rem; border-radius: 20px; border: 1px solid #e2e8f0; margin-bottom: 2rem; }}
        .risk-tag {{ padding: 0.7rem 2.5rem; border-radius: 50px; font-weight: 800; }}
        .risk-HIGH {{ background: #fee2e2; color: #dc2626; }}
        .risk-MODERATE {{ background: #fef3c7; color: #d97706; }}
        .risk-LOW {{ background: #dcfce7; color: #16a34a; }}
    </style>
    """
    st.markdown(css_content, unsafe_allow_html=True)

# --- LOGIN PAGE ---
def render_login():
    apply_custom_styles(is_login=True)
    st.markdown("<h1 style='color: #dc2626; font-size: 5rem; text-align: center;'>InfraGuard AI</h1>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        st.text_input("Email", placeholder="Email Address")
        st.text_input("Password", type="password", placeholder="Password")
        if st.button("PORTAL ACCESS", use_container_width=True):
            st.session_state.authenticated = True
            st.rerun()

# --- NAVIGATION ---
def render_navbar():
    l, r = st.columns([1.5, 3])
    with l: st.markdown("## 🏗️ InfraGuard AI")
    with r:
        n1, n2, n3, n4, n5 = st.columns(5)
        if n1.button("Overview"): st.session_state.current_page = "Overview"
        if n2.button("Terminal"): st.session_state.current_page = "Terminal"
        if n5.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()

# --- PAGES ---
def page_terminal():
    st.title("Diagnostic Analysis Terminal")
    file_up = st.file_uploader("Imagery Feed", type=["jpg", "png"])
    if file_up:
        img = Image.open(file_up)
        _, _, crack_prob = predict_crack(img)
        h_score = int((1.0 - float(crack_prob)) * 100)
        st.image(img)
        st.metric("Structural Health", f"{h_score}%")

# --- BOOTSTRAP ---
if not st.session_state.authenticated:
    render_login()
else:
    render_navbar()
    if st.session_state.current_page == "Overview": st.write("Welcome to InfraGuard AI Dashboard.")
    elif st.session_state.current_page == "Terminal": page_terminal()
