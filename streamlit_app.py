import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import folium
from streamlit_folium import st_folium

# ─── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="Solar Rooftop Scout",
    layout="wide",
    page_icon=None
)

# ─── CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background-color: #F0F4F8;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Page wrapper ── */
.block-container {
    padding: 2rem 2.5rem 3rem 2.5rem !important;
    max-width: 1200px;
}

/* ── Header ── */
.page-header {
    border-left: 4px solid #F59E0B;
    padding-left: 1rem;
    margin-bottom: 2rem;
}
.page-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #1E293B;
    letter-spacing: -0.02em;
    margin: 0 0 0.25rem 0;
    line-height: 1.2;
}
.page-subtitle {
    font-size: 0.95rem;
    color: #64748B;
    margin: 0;
    font-weight: 400;
}

/* ── Section label ── */
.section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #94A3B8;
    margin: 2rem 0 0.75rem 0;
}

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 0.5rem;
}
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    flex: 1;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #F59E0B, #FCD34D);
}
.metric-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: #1E293B;
    line-height: 1;
}
.metric-unit {
    font-size: 0.85rem;
    font-weight: 500;
    color: #94A3B8;
    margin-left: 0.25rem;
}

/* ── Info box ── */
.info-box {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
}
.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid #F1F5F9;
    font-size: 0.9rem;
}
.info-row:last-child { border-bottom: none; }
.info-key { color: #64748B; font-weight: 400; }
.info-val { color: #1E293B; font-weight: 600; }
.highlight-val { color: #D97706; font-weight: 700; }

/* ── Empty state ── */
.empty-state {
    background: #FFFFFF;
    border: 1px dashed #CBD5E1;
    border-radius: 16px;
    padding: 4rem 2rem;
    text-align: center;
    margin-top: 1rem;
}
.empty-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #334155;
    margin-bottom: 0.5rem;
}
.empty-sub {
    font-size: 0.9rem;
    color: #94A3B8;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #1E293B !important;
}
[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: #F59E0B;
    color: #1E293B !important;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.88rem;
    letter-spacing: 0.03em;
    padding: 0.6rem 1rem;
    width: 100%;
    transition: background 0.2s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #FCD34D;
}
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stTextInput label {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94A3B8 !important;
}
[data-testid="stSidebar"] input {
    background-color: #334155 !important;
    border: 1px solid #475569 !important;
    color: #F1F5F9 !important;
    border-radius: 8px !important;
}
.sidebar-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #F1F5F9;
    letter-spacing: 0.03em;
    margin-bottom: 1.5rem;
}
.sidebar-divider {
    border: none;
    border-top: 1px solid #334155;
    margin: 1.25rem 0;
}

/* ── Map container ── */
.map-wrapper {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    overflow: hidden;
}

/* ── Chart ── */
.stLineChart > div {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "lat" not in st.session_state:
    st.session_state.lat = 18.7883
if "lng" not in st.session_state:
    st.session_state.lng = 98.9853

# ─── Load Scaler ───────────────────────────────────────────
@st.cache_resource
def load_scaler():
    if os.path.exists("scaler.pkl"):
        try:
            with open("scaler.pkl", "rb") as f:
                return pickle.load(f), None
        except Exception as e:
            return None, str(e)
    return None, None

scaler, scaler_err = load_scaler()

# ─── Simulation Functions ──────────────────────────────────
def predict_roof_area():
    mask = np.zeros((256, 256, 1))
    mask[50:200, 40:210, 0] = 1
    area = float(np.sum(mask) * 0.25)
    return area

def predict_solar_irradiance(scaler_obj):
    base = np.array([4.8, 5.2, 5.5, 5.8, 5.1, 4.3, 4.1, 4.4, 4.6, 4.9, 5.0, 4.7])
    if scaler_obj:
        try:
            return scaler_obj.inverse_transform(
                scaler_obj.transform(base.reshape(-1, 1))
            ).flatten()
        except:
            return base
    return base

# ─── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">Solar Rooftop Scout</div>', unsafe_allow_html=True)
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    lat = st.number_input("Latitude", value=st.session_state.lat, format="%.4f")
    lng = st.number_input("Longitude", value=st.session_state.lng, format="%.4f")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Analyze Location"):
        st.session_state.analyzed = True
        st.session_state.lat = lat
        st.session_state.lng = lng

    if st.session_state.analyzed:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Reset"):
            st.session_state.analyzed = False
            st.rerun()

# ─── Page Header ───────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="page-title">Solar Rooftop Analysis</div>
    <div class="page-subtitle">AI-powered solar potential assessment for industrial rooftops</div>
</div>
""", unsafe_allow_html=True)

# ─── Main Content ──────────────────────────────────────────
if st.session_state.analyzed:

    area = predict_roof_area()
    forecast = predict_solar_irradiance(scaler)

    capacity_kwp = area / 10
    avg = float(np.mean(forecast))
    annual = capacity_kwp * avg * 0.75 * 365
    best_month = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][int(np.argmax(forecast))]

    # ─── Metrics ──────────────────────────────────────────
    st.markdown('<div class="section-label">Results Summary</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Roof Area</div>
            <div class="metric-value">{area:,.0f}<span class="metric-unit">m²</span></div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">System Capacity</div>
            <div class="metric-value">{capacity_kwp:,.1f}<span class="metric-unit">kWp</span></div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Annual Output</div>
            <div class="metric-value">{annual:,.0f}<span class="metric-unit">kWh</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Map + Info ───────────────────────────────────────
    st.markdown('<div class="section-label">Location</div>', unsafe_allow_html=True)

    col_map, col_info = st.columns([3, 1])

    with col_map:
        st.markdown('<div class="map-wrapper">', unsafe_allow_html=True)
        m = folium.Map(
            location=[st.session_state.lat, st.session_state.lng],
            zoom_start=17,
            tiles="CartoDB positron"
        )
        folium.CircleMarker(
            location=[st.session_state.lat, st.session_state.lng],
            radius=10,
            color="#F59E0B",
            fill=True,
            fill_color="#F59E0B",
            fill_opacity=0.7,
            popup=f"Lat: {st.session_state.lat}, Lng: {st.session_state.lng}"
        ).add_to(m)
        st_folium(m, width=None, height=360, returned_objects=[])
        st.markdown('</div>', unsafe_allow_html=True)

    with col_info:
        st.markdown(f"""
        <div class="info-box">
            <div class="section-label" style="margin-top:0">Site Details</div>
            <div class="info-row">
                <span class="info-key">Latitude</span>
                <span class="info-val">{st.session_state.lat:.4f}</span>
            </div>
            <div class="info-row">
                <span class="info-key">Longitude</span>
                <span class="info-val">{st.session_state.lng:.4f}</span>
            </div>
            <div class="info-row">
                <span class="info-key">Avg. Irradiance</span>
                <span class="info-val">{avg:.2f} kWh/m²</span>
            </div>
            <div class="info-row">
                <span class="info-key">Peak Month</span>
                <span class="highlight-val">{best_month}</span>
            </div>
            <div class="info-row">
                <span class="info-key">System Factor</span>
                <span class="info-val">0.75 PR</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ─── Chart ────────────────────────────────────────────
    st.markdown('<div class="section-label">Monthly Irradiance Forecast</div>', unsafe_allow_html=True)

    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    df = pd.DataFrame({
        "Irradiance (kWh/m²/day)": forecast
    }, index=months)

    st.line_chart(df, use_container_width=True, height=240)

else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-title">No location selected</div>
        <div class="empty-sub">Enter coordinates in the sidebar and click Analyze Location to begin.</div>
    </div>
    """, unsafe_allow_html=True)
