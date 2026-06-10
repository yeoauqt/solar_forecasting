import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import folium
from streamlit_folium import st_folium

# ─── CONFIG & SETUP ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Industrial Solar Rooftop Scout",
    layout="wide",
    page_icon="☀️"
)

# CSS Theme
st.markdown("""
    <style>
    .main-title { font-size: 36px; font-weight: 800; color: #1E3A8A; margin-bottom: 5px; }
    .sub-title { font-size: 16px; color: #4B5563; margin-bottom: 30px; }
    .section-header { font-size: 22px; font-weight: 700; color: #1E3A8A; border-bottom: 2px solid #E5E7EB; padding-bottom: 8px; margin-top: 25px; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────────────────
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'saved_lat' not in st.session_state:
    st.session_state.saved_lat = 18.7883
if 'saved_lng' not in st.session_state:
    st.session_state.saved_lng = 98.9853

# ─── LOAD SCALER ───────────────────────────────────────────────────────────
@st.cache_resource
def load_scaler():
    scaler_file = 'scaler.pkl'
    if os.path.exists(scaler_file):
        try:
            with open(scaler_file, 'rb') as f:
                return pickle.load(f), None
        except Exception as e:
            return None, f"❌ โหลด Scaler ไม่สำเร็จ: {str(e)}"
    return None, "⚠️ ไม่พบไฟล์ scaler.pkl (ใช้ mock mode)"

with st.spinner("🔄 กำลังเตรียมระบบ..."):
    scaler, scaler_err = load_scaler()

# ─── FUNCTIONS ─────────────────────────────────────────────────────────────
def get_satellite_image(lat, lng):
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[50:200, 40:210, :] = [140, 145, 150]
    return img

def predict_roof_area(image):
    mask = np.zeros((256, 256, 1), dtype=np.uint8)
    mask[50:200, 40:210, 0] = 1
    pixel_resolution = 0.25
    roof_pixels = np.sum(mask)
    area = roof_pixels * pixel_resolution
    return mask, area

def predict_solar_irradiance(scaler_obj):
    base = np.array([4.8, 5.2, 5.5, 5.8, 5.1, 4.3, 4.1, 4.4, 4.6, 4.9, 5.0, 4.7])
    if scaler_obj is not None:
        try:
            scaled = scaler_obj.transform(base.reshape(-1, 1))
            return scaler_obj.inverse_transform(scaled).flatten()
        except:
            return base
    return base

# ─── SIDEBAR ───────────────────────────────────────────────────────────────
st.sidebar.title("☀️ Control Panel")

lat_input = st.sidebar.number_input("Latitude", value=st.session_state.saved_lat, format="%.6f")
lng_input = st.sidebar.number_input("Longitude", value=st.session_state.saved_lng, format="%.6f")

if st.sidebar.button("🚀 Analyze"):
    st.session_state.analyzed = True
    st.session_state.saved_lat = lat_input
    st.session_state.saved_lng = lng_input

if st.sidebar.button("🔄 Reset"):
    st.session_state.analyzed = False
    st.rerun()

# ─── HEADER ────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">☀️ Solar Rooftop Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI วิเคราะห์ศักยภาพโซลาร์บนหลังคาโรงงาน</div>', unsafe_allow_html=True)

if scaler_err:
    st.caption(scaler_err)

# ─── MAIN ──────────────────────────────────────────────────────────────────
if st.session_state.analyzed:

    sat_img = get_satellite_image(st.session_state.saved_lat, st.session_state.saved_lng)
    roof_mask, roof_area = predict_roof_area(sat_img)
    forecast = predict_solar_irradiance(scaler)

    potential_kwp = roof_area / 10
    avg_irradiance = np.mean(forecast)
    annual_energy = potential_kwp * avg_irradiance * 0.75 * 365

    months = ["ม.ค.","ก.พ.","มี.ค.","เม.ย.","พ.ค.","มิ.ย.","ก.ค.","ส.ค.","ก.ย.","ต.ค.","พ.ย.","ธ.ค."]
    df = pd.DataFrame({
        "เดือน": months,
        "irradiance": forecast
    }).set_index("เดือน")

    # ─── METRICS ─────────────────────────────
    st.markdown('<div class="section-header">Summary</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Roof Area", f"{roof_area:,.2f} m²", "U-Net mock")

    with c2:
        st.metric("Capacity", f"{potential_kwp:,.2f} kWp")

    with c3:
        st.metric("Annual Energy", f"{annual_energy:,.2f} kWh")

    # ─── MAP + IMAGE ─────────────────────────
    st.markdown('<div class="section-header">Map & Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        m = folium.Map(location=[st.session_state.saved_lat, st.session_state.saved_lng], zoom_start=17)
        folium.Marker([st.session_state.saved_lat, st.session_state.saved_lng]).add_to(m)

        st_folium(m, width=None, height=300)

    with col2:
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))

        ax[0].imshow(sat_img)
        ax[0].set_title("Satellite")
        ax[0].axis("off")

        ax[1].imshow(roof_mask.squeeze(), cmap="hot")
        ax[1].set_title("Mask")
        ax[1].axis("off")

        st.pyplot(fig)

    # ─── FORECAST ────────────────────────────
    st.markdown('<div class="section-header">Forecast</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.line_chart(df)

    with col2:
        st.info(f"""
Average Irradiance: {avg_irradiance:.2f}

Best season: Mar–Apr

Recommendation: Suitable for solar investment
""")

else:
    st.info("กรุณากรอกพิกัดและกด Analyze")
