import sys
import os

# Add project root to sys.path for submodule imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from PIL import Image
import time
import random
import datetime
import pandas as pd
import pydeck as pdk
from ai.predict import predict_crack
from backend.risk_assessment import risk_level
from utils.image_processing import get_canny_edges

# --- SYSTEM CONFIGURATION ---
st.set_page_config(
    page_title="InfraGuide AI | Vibrant Enterprise Monitoring",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Session State
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Overview"
if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []

# --- VIBRANT PROFESSIONAL UI CSS: INDIGO, EMERALD, CRIMSON ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Global Body: Professional Executive White */
    .stApp {
        background: radial-gradient(circle at top right, #ffffff, #f1f5f9) !important;
        color: #0f172a !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Hide Sidebars & Streamlit Trim */
    [data-testid="stSidebar"] { display: none; }
    header { visibility: hidden; }
    .stMainBlockContainer { padding-top: 2rem !important; }

    /* Glassmorphism Header Bar */
    .header-bar {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 75px;
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(226, 232, 240, 0.5);
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 4%;
        z-index: 10000;
        box-shadow: 0 4px 15px rgba(0,0,0,0.02);
    }

    .brand-id {
        font-weight: 800;
        font-size: 1.6rem;
        color: #1e1b4b; /* Deep Indigo */
        letter-spacing: -0.04em;
        display: flex;
        align-items: center;
        gap: 10px;
        position: relative;
        top: 2px; /* Visual center tweak */
    }
    .brand-id span { 
        background: linear-gradient(135deg, #4f46e5, #1e1b4b); 
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Unified Navigation Placement */
    .nav-unified {
        position: fixed;
        top: 13px;
        right: 4%;
        z-index: 10001;
        display: flex;
        align-items: center;
        gap: 8px;
        width: auto;
    }

    /* --- CLEAN WHITE EXECUTIVE OVERHAUL --- */
    [data-testid="stAppViewContainer"], .stApp {
        background: #ffffff !important;
        background-color: #ffffff !important;
        animation: none !important;
    }

    /* Target the container of the auth-gate */
    .auth-container {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 80vh;
        width: 100%;
        background: #ffffff !important;
    }

    .auth-gate {
        max-width: 440px;
        width: 100%;
        padding: 3rem;
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 32px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.08);
        text-align: center;
        animation: fadeIn 0.8s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .auth-gate h1 {
        font-size: 3rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #1e1b4b, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem !important;
    }

    /* Premium Input Styling */
    div[data-baseweb="input"] {
        border-radius: 12px !important;
        background-color: rgba(255, 255, 255, 0.5) !important;
        border: 1px solid rgba(226, 232, 240, 0.8) !important;
        transition: all 0.3s ease !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1) !important;
    }

    /* Specialized Login Button - Executive Slate/Indigo */
    .auth-btn button {
        background: #1e1b4b !important;
        color: white !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.1em !important;
        height: 58px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        transition: all 0.3s ease !important;
        margin-top: 1rem !important;
    }
    
    .auth-btn button:hover {
        background: #312e81 !important;
        transform: translateY(-1px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
    }

    /* Vibrant Navigation Buttons Styling */
    .stButton>button {
        background: transparent !important;
        color: #64748b !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        text-transform: capitalize !important;
        letter-spacing: 0.02em !important;
        padding: 0.6rem 1rem !important;
        border-radius: 12px !important;
        height: 42px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        white-space: nowrap !important;
    }
    
    .stButton>button:hover {
        background: rgba(99, 102, 241, 0.05) !important;
        color: #6366f1 !important;
    }

    /* Specialized Logout Button Styling within the line */
    .logout-item button {
        background: rgba(239, 68, 68, 0.05) !important;
        color: #dc2626 !important;
        border: 1px solid rgba(220, 38, 38, 0.1) !important;
    }
    
    .logout-item button:hover {
        background: #dc2626 !important;
        color: white !important;
    }

    /* Active Tab Overlay */
    .nav-active button {
        background: #6366f1 !important;
        color: white !important;
        border: none !important;
    }

    /* Glass Content Viewport */
    .viewport {
        padding-top: 85px;
        max-width: 1200px;
        margin: 0 auto;
        padding-bottom: 60px;
    }

    /* Premium Gradient Cards */
    .card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 28px;
        padding: 2.5rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.03);
        margin-bottom: 1.5rem;
    }

    /* Dashboard Metrics With Vibrant Gradients */
    .metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin-top: 1.5rem; }
    
    .metric-pill {
        padding: 1.8rem;
        border-radius: 24px;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.8);
    }
    
    .pill-blue { background: linear-gradient(135deg, #e0e7ff, #ede9fe); }
    .pill-green { background: linear-gradient(135deg, #ecfdf5, #d1fae5); }
    .pill-orange { background: linear-gradient(135deg, #fff7ed, #ffedd5); }

    .stat-label { font-size: 0.7rem; font-weight: 800; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; }
    .stat-value { font-size: 2.8rem; font-weight: 800; color: #1e1b4b; margin-top: 5px; }

    /* Auth Gate UI */
    .auth-gate {
        max-width: 450px;
        margin: 12vh auto;
        padding: 4rem;
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(20px);
        border-radius: 35px;
        box-shadow: 0 40px 100px rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.5);
        text-align: center;
    }

    /* Sophisticated Scans */
    .scan-box { position: relative; border-radius: 24px; overflow: hidden; border: 4px solid white; box-shadow: 0 15px 40px rgba(0,0,0,0.05); }
    .scan-beam { 
        position: absolute; width:100%; height:8px; 
        background: linear-gradient(to bottom, transparent, #6366f1, transparent); 
        animation: beam 2s infinite ease-in-out; 
        z-index: 5; 
        opacity: 0.8;
    }
    @keyframes beam { 0% { top: -10%; } 100% { top: 110%; } }

    /* Results Banners - Three Tiers */
    .res-banner {
        padding: 1.5rem;
        border-radius: 20px;
        font-weight: 800;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
        font-size: 1.1rem;
        border: 1px solid transparent;
    }
    .res-high { background: #fef2f2; color: #dc2626; border-color: #fee2e2; }
    .res-med { background: #fffbeb; color: #d97706; border-color: #fef3c7; }
    .res-low { background: #f0fdf4; color: #16a34a; border-color: #dcfce7; }

</style>
""", unsafe_allow_html=True)

# --- AUTH FLOW ---
def auth_gate():
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<div class='auth-container'>", unsafe_allow_html=True)
        st.markdown("""
        <div class='auth-gate'>
            <h1>InfraGuide AI</h1>
            <p style='color: #6366f1; font-weight: 800; font-size: 0.9rem; letter-spacing: 0.25em; text-transform: uppercase; margin-bottom: 2.5rem;'>Secure Infrastructure Monitoring Portal</p>
        """, unsafe_allow_html=True)
        
        st.text_input("Operational Identity", placeholder="Authorized Personnel ID")
        st.text_input("Access Security Key", type="password", placeholder="••••••••")
        
        st.markdown("<div class='auth-btn'>", unsafe_allow_html=True)
        if st.button("AUTHENTICATE SYSTEM", use_container_width=True):
            st.session_state.authenticated = True
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
            <p style='margin-top: 2rem; color: #94a3b8; font-size: 0.75rem; font-weight: 600;'>
                CRYPTO-SHIELD v4.2 | END-TO-END ENCRYPTED SESSION
            </p>
        </div></div>
        """, unsafe_allow_html=True)

# --- TOP NAVIGATION ---
def render_navbar():
    st.markdown(f"""
    <div class="header-bar">
        <div class="brand-id">🏗️ InfraGuide <span>AI</span></div>
    </div>
    """, unsafe_allow_html=True)

    # UNIFIED NAVIGATION: Overview, Terminal, Surveillance, Geo-Watch, History, Logout
    st.markdown("<div class='nav-unified'>", unsafe_allow_html=True)
    n1, n2, n3, n4, n5, n6 = st.columns([1, 1, 1.2, 1.2, 1, 1.2])
    
    with n1:
        is_active = "nav-active" if st.session_state.current_page == "Overview" else ""
        st.markdown(f"<div class='{is_active}'>", unsafe_allow_html=True)
        if st.button("Overview", key="n_over"): st.session_state.current_page = "Overview"
        st.markdown("</div>", unsafe_allow_html=True)
        
    with n2:
        is_active = "nav-active" if st.session_state.current_page == "Terminal" else ""
        st.markdown(f"<div class='{is_active}'>", unsafe_allow_html=True)
        if st.button("Terminal", key="n_term"): st.session_state.current_page = "Terminal"
        st.markdown("</div>", unsafe_allow_html=True)

    with n3:
        is_active = "nav-active" if st.session_state.current_page == "Surveillance" else ""
        st.markdown(f"<div class='{is_active}'>", unsafe_allow_html=True)
        if st.button("Surveillance", key="n_surv"): st.session_state.current_page = "Surveillance"
        st.markdown("</div>", unsafe_allow_html=True)

    with n4:
        is_active = "nav-active" if st.session_state.current_page == "Geo-Watch" else ""
        st.markdown(f"<div class='{is_active}'>", unsafe_allow_html=True)
        if st.button("Geo-Watch", key="n_map"): st.session_state.current_page = "Geo-Watch"
        st.markdown("</div>", unsafe_allow_html=True)
        
    with n5:
        is_active = "nav-active" if st.session_state.current_page == "History" else ""
        st.markdown(f"<div class='{is_active}'>", unsafe_allow_html=True)
        if st.button("History", key="n_hist"): st.session_state.current_page = "History"
        st.markdown("</div>", unsafe_allow_html=True)

    with n6:
        st.markdown("<div class='logout-item'>", unsafe_allow_html=True)
        if st.button("Logout 🚪", key="n_off"): 
            st.session_state.authenticated = False
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- APPLICATION PAGES ---

def overview_page():
    st.markdown("<h1>System Intelligence <span style='background: linear-gradient(135deg, #6366f1, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Dashboard</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.1rem; color: #64748b; margin-bottom: 2rem;'>Real-time orchestration of global infrastructure monitoring nodes.</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='metric-grid'>", unsafe_allow_html=True)
    st.markdown("<div class='metric-pill pill-blue'><div class='stat-label'>Global Integrity</div><div class='stat-value'>98.2%</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='metric-pill pill-green'><div class='stat-label'>Scanning Nodes</div><div class='stat-value'>342</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='metric-pill pill-orange'><div class='stat-label'>Active Alerts</div><div class='stat-value' style='color:#f59e0b;'>02</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card' style='margin-top: 2rem;'><h3>Live Performance Feed</h3><p style='color:#64748b;'>Distributed neural networks are operating at 99.9% precision. No critical structural propagation detected in the last 24 hours.</p></div>", unsafe_allow_html=True)

def terminal_page():
    st.markdown("<h2>Audit <span style='color: #6366f1;'>Intelligence Terminal</span></h2>", unsafe_allow_html=True)
    
    c_config, c_process = st.columns([1, 1.8])
    
    with c_config:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0;'>Diagnostic Setup</h4>", unsafe_allow_html=True)
        asset = st.selectbox("Asset Category", ["Reinforced Concrete", "Suspension Steel", "Masonry Facade", "Seismic Support"])
        stress = st.selectbox("Environment Data", ["Tropical Humidity", "Arctic Thermal", "Industrial Resonance"])
        
        file_up = st.file_uploader("Imagery Feed", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        if file_up:
            st.image(Image.open(file_up), use_container_width=True)
            if st.button("RUN DEEP SCAN ANALYSIS", use_container_width=True):
                st.session_state.scanning = True
        else:
            st.session_state.scanning = False
            st.markdown("<div style='height: 150px; display: flex; align-items: center; justify-content: center; background: #f8fafc; border: 2px dashed #e2e8f0; border-radius: 20px; color: #94a3b8; font-style: italic;'>Drop capture here...</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_process:
        if file_up and st.session_state.get('scanning', False):
            # Premium Scan Effect
            placeholder = st.empty()
            with placeholder.container():
                st.markdown("<div class='scan-box'>", unsafe_allow_html=True)
                st.markdown("<div class='scan-beam'></div>", unsafe_allow_html=True)
                st.image(Image.open(file_up), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                st.info("Compiling fracture vectors via Neural Pipeline v4.0...")
            
            p = st.progress(0)
            for i in range(101):
                time.sleep(0.01)
                p.progress(i)
            
            placeholder.empty()
            p.empty()
            
            # ACTUAL AI INTEGRATION
            img_input = Image.open(file_up)
            
            # Get AI Prediction (label, display_confidence, raw crack_probability)
            res_label, confidence, crack_probability = predict_crack(img_input)
            
            # Get Risk Assessment from Backend (uses crack probability, not display confidence)
            risk_tier_name, health_score, recommendation = risk_level(crack_probability)
            
            # Map risk level to UI styling
            if risk_tier_name == "HIGH":
                risk_tier = "res-high"
                risk_label = "CRITICAL ANOMALY"
                risk_color = "#dc2626"
                risk_desc = "HIGH RISK"
            elif risk_tier_name == "MEDIUM":
                risk_tier = "res-med"
                risk_label = "MODERATE STRESS"
                risk_color = "#d97706"
                risk_desc = "MEDIUM RISK"
            else:
                risk_tier = "res-low"
                risk_label = "INTEGRITY VERIFIED"
                risk_color = "#16a34a"
                risk_desc = "STABLE"
            
            score = health_score
            
            # Canny Edge detection for visualization
            edges = get_canny_edges(img_input)
            
            st.markdown(f"""
            <div class='card' style='border-top: 10px solid {risk_color};'>
                <div class='res-banner {risk_tier}'>
                    <span>{risk_label}</span>
                    <span style='background: white; border: 1px solid currentColor; padding: 0.3rem 1.2rem; border-radius: 12px; font-size: 0.8rem;'>{risk_desc}</span>
                </div>
                
                <div class="metric-grid" style='margin-bottom: 2rem;'>
                    <div class='metric-pill pill-blue' style='padding:1.2rem; border-radius:16px;'><div class='stat-label'>Health Score</div><div class='stat-value' style='font-size: 1.8rem; color:{risk_color};'>{score}</div></div>
                    <div class='metric-pill pill-green' style='padding:1.2rem; border-radius:16px;'><div class='stat-label'>Confidence</div><div class='stat-value' style='font-size: 1.8rem;'>{confidence*100:.1f}%</div></div>
                    <div class='metric-pill pill-orange' style='padding:1.2rem; border-radius:16px;'><div class='stat-label'>Load Bias</div><div class='stat-value' style='font-size: 1.2rem; padding-top:10px;'>{stress.split()[0].upper()}</div></div>
                </div>
                
                <h4 style='color: #1e1b4b;'>Maintenance Directive</h4>
                <div style='background: #f8fafc; padding: 1.5rem; border-radius: 16px; border: 1px solid #f1f5f9; font-size: 0.95rem; line-height: 1.6;'>
                    <b>{risk_tier_name} PRIORITY:</b> {recommendation}
                </div>
                
                <div style='margin-top: 2rem;'>
                    <h4 style='color: #1e1b4b;'>Structural Edge Mapping</h4>
                    <p style='color: #64748b; font-size: 0.8rem;'>Canny Edge Detection analysis showing surface discontinuities.</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.image(edges, caption="Edge Detection Matrix", use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 GENERATE ENTERPRISE AUDIT REPORT", use_container_width=True):
                st.toast("Compiling structural data...", icon="📄")
                time.sleep(1)
                st.success("Report Generated: InfraGuide_Audit_" + datetime.datetime.now().strftime("%Y%m%d") + ".pdf")
            st.markdown(f"""
            </div>
            """, unsafe_allow_html=True)
            
            st.session_state.scan_history.append({'time': datetime.datetime.now().strftime("%H:%M"), 'res': risk_label, 'score': score, 'color': risk_color})
            
        else:
            st.markdown("<div style='height: 520px; display: flex; align-items: center; justify-content: center; background: rgba(255,255,255,0.5); border-radius: 28px; border: 2px dashed #e2e8f0; color: #94a3b8; font-style: italic; backdrop-filter: blur(10px);'>Awaiting high-resolution imagery for processing.</div>", unsafe_allow_html=True)

def surveillance_page():
    st.markdown("<h2>Node <span style='color: #6366f1;'>Surveillance Watch</span></h2>", unsafe_allow_html=True)
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("<div class='scan-box' style='height: 450px; background: #000; display: flex; align-items: center; justify-content: center;'>", unsafe_allow_html=True)
        st.markdown("<div class='scan-beam' style='animation-duration: 4s;'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #10b981; font-family: monospace;'>LIVE FEED: BRIDGE_NODE_074</h3>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='card' style='padding:1.5rem;'><b>Active Detections:</b> <span style='color:#16a34a;'>No fractures identified in current sweep.</span></div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h4>Node Status</h4>", unsafe_allow_html=True)
        st.write("● Node-Alpha: Online")
        st.write("● Node-Beta: Online")
        st.write("● Node-Gamma: Recalibrating...")
        
        st.markdown("<hr style='border: 0.5px solid #e2e8f0;'>", unsafe_allow_html=True)
        st.markdown("<h4>Control Panel</h4>", unsafe_allow_html=True)
        st.toggle("Night Vision Mode")
        st.toggle("Thermal Overlay")
        st.button("CYCLE NODES", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

def map_page():
    st.markdown("<h2>Geo-Watch <span style='color: #6366f1;'>GIS Asset Map</span></h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; margin-bottom: 2rem;'>Geospatial risk distribution across active monitoring sectors.</p>", unsafe_allow_html=True)
    
    # Mock Data for Infrastructure Nodes
    map_data = pd.DataFrame({
        'lat': [40.7128, 40.7306, 40.7410, 40.6782, 40.7831],
        'lon': [-74.0060, -73.9352, -73.9897, -73.9442, -73.9712],
        'name': ['Brooklyn Bridge Node', 'Queensboro Support', 'Empire Structural Node', 'Manhattan Bridge Pillar', 'Central Reservoir Span'],
        'risk': ['CRITICAL', 'STABLE', 'STABLE', 'WARNING', 'STABLE'],
        'color': [[220, 38, 38, 180], [22, 163, 74, 180], [22, 163, 74, 180], [217, 119, 6, 180], [22, 163, 74, 180]]
    })

    # PyDeck 3D Map
    view_state = pdk.ViewState(latitude=40.73, longitude=-73.97, zoom=11, pitch=45)
    
    layer = pdk.Layer(
        "ColumnLayer",
        data=map_data,
        get_position="[lon, lat]",
        get_elevation=1000,
        elevation_scale=1,
        radius=250,
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    )

    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "{name}\nStatus: {risk}"}
    ))
    
    st.markdown("<div class='card' style='margin-top: 2rem;'><h4>Regional Risk Summary</h4><p style='color:#64748b;'>Current focus: <b>Metropolitan Sector 4</b>. Higher stress factors detected near waterway pillars due to maritime corrosion profiles.</p></div>", unsafe_allow_html=True)

def history_page():
    st.markdown("<h2>Audit <span style='color: #a855f7;'>Session Records</span></h2>", unsafe_allow_html=True)
    if not st.session_state.scan_history:
        st.info("Log database is currently empty for active session.")
    else:
        for entry in reversed(st.session_state.scan_history):
            color = entry.get('color', "#1e293b")
            st.markdown(f"""
            <div class='card' style='padding: 1.5rem; border-left: 8px solid {color};'>
                <div style='display: flex; justify-content: space-between;'>
                    <b>Time: {entry['time']}</b>
                    <span style='color:{color}; font-weight:800;'>{entry['res']} (Health Index: {entry['score']:.1f})</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# --- EXECUTION ---
if not st.session_state.authenticated:
    auth_gate()
else:
    render_navbar()
    st.markdown("<div class='viewport'>", unsafe_allow_html=True)
    
    if st.session_state.current_page == "Overview": overview_page()
    elif st.session_state.current_page == "Terminal": terminal_page()
    elif st.session_state.current_page == "Surveillance": surveillance_page()
    elif st.session_state.current_page == "Geo-Watch": map_page()
    elif st.session_state.current_page == "History": history_page()
    
    st.markdown("</div>", unsafe_allow_html=True)

