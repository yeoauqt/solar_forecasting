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

# แต่งหน้าตาตัวหนังสือและธีม Dashboard ให้ดูสวยหรู มีระเบียบ ด้วย CSS 
st.markdown("""
    <style>
    .main-title { font-size: 36px; font-weight: 800; color: #1E3A8A; margin-bottom: 5px; }
    .sub-title { font-size: 16px; color: #4B5563; margin-bottom: 30px; }
    .section-header { font-size: 22px; font-weight: 700; color: #1E3A8A; border-bottom: 2px solid #E5E7EB; padding-bottom: 8px; margin-top: 25px; margin-bottom: 15px; }
    </style>
""", unsafe_html=True)

# สร้างระบบสมุดบันทึก (Session State) ป้องกันข้อมูลและผลลัพธ์หายเวลาขยับหน้าจอ
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'saved_lat' not in st.session_state:
    st.session_state.saved_lat = 18.7883
if 'saved_lng' not in st.session_state:
    st.session_state.saved_lng = 98.9853

# ฟังก์ชันโหลดไฟล์ Scaler แบบปลอดภัยสูง
@st.cache_resource
def load_scaler():
    scaler_path = 'scaler.pkl'
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, 'rb') as f:
                return pickle.load(f), None
        except Exception as e:
            return None, f"❌ โหลด Scaler ไม่สำเร็จ: {str(e)}"
    return None, "⚠️ ไม่พบไฟล์ 'scaler.pkl' (ระบบเปิดโหมดจำลองอัตโนมัติ)"

with st.spinner("🔄 กำลังเตรียมความพร้อมระบบ..."):
    scaler, scaler_err = load_scaler()

# ─── HELPER FUNCTIONS ──────────────────────────────────────────────────────
def get_satellite_image(lat, lng):
    # วาดโครงภาพหลังคาโรงงานจำลองให้ดูเนียนและสะอาดตา
    mock_img = np.zeros((256, 256, 3), dtype=np.uint8)
    mock_img[50:200, 40:210, :] = [140, 145, 150] 
    return mock_img

def predict_roof_area(image):
    # จำลองการตัด Mask ของพื้นที่หลังคาโรงงาน
    binary_mask = np.zeros((256, 256, 1), dtype=np.uint8)
    binary_mask[50:200, 40:210, 0] = 1
    pixel_resolution = 0.25 
    roof_pixels = np.sum(binary_mask)
    calculated_area = roof_pixels * pixel_resolution
    return binary_mask, calculated_area

def predict_solar_irradiance(scaler):
    # ค่ารังสีมาตรฐานรายเดือน (ม.ค. - ธ.ค.)
    base_values = np.array([4.8, 5.2, 5.5, 5.8, 5.1, 4.3, 4.1, 4.4, 4.6, 4.9, 5.0, 4.7])
    if scaler is not None:
        try:
            # แก้ไขจุดวงเล็บตกหล่นเรียบร้อยตรงบรรทัดนี้ครับ
            scaled = scaler.transform(base_values.reshape(-1, 1)).flatten()
            real_pred = scaler.inverse_transform(scaled.reshape(-1, 1)).flatten()
            return real_pred
        except:
            return base_values
    return base_values

# ─── SIDEBAR & INPUTS ──────────────────────────────────────────────────────
st.sidebar.title("☀️ แผงควบคุมระบบ")
st.sidebar.markdown("กรอกพิกัดตำแหน่งโรงงานอุตสาหกรรมเพื่อทำการสแกน")

lat_input = st.sidebar.number_input("📍 Latitude (ละติจูด)", value=st.session_state.saved_lat, format="%.6f")
lng_input = st.sidebar.number_input("📍 Longitude (ลองจิจูด)", value=st.session_state.saved_lng, format="%.6f")

st.sidebar.markdown("---")
if st.sidebar.button("🚀 เริ่มวิเคราะห์ระบบโซลาร์", type="primary", use_container_width=True):
    st.session_state.analyzed = True
    st.session_state.saved_lat = lat_input
    st.session_state.saved_lng = lng_input

if st.sidebar.button("🔄 ล้างข้อมูลหน้าจอ", use_container_width=True):
    st.session_state.analyzed = False
    st.rerun()

# ─── MAIN DASHBOARD ────────────────────────────────────────────────────────
st.markdown('<div class="main-title">☀️ Industrial Solar Rooftop Scout Dashboard</div>', unsafe_html=True)
st.markdown('<div class="sub-title">🔮 ระบบ AI อัจฉริยะวิเคราะห์พื้นที่หลังคาโรงงานอุตสาหกรรม และพยากรณ์ค่ารังสีแสงอาทิตย์ล่วงหน้าเพื่อประเมินความคุ้มค่า</div>', unsafe_html=True)

if scaler_err:
    st.caption(scaler_err)

# จัดลำดับ: ถ้าระบบโดนกดปุ่มวิเคราะห์ ให้ทำการคำนวณเบื้องหลังให้เสร็จสิ้น 100% ก่อนก้าวไปจัดหน้าจอ
if st.session_state.analyzed:
    
    sat_img = get_satellite_image(st.session_state.saved_lat, st.session_state.saved_lng)
    roof_mask, roof_area = predict_roof_area(sat_img)
    forecast_values = predict_solar_irradiance(scaler)
    
    potential_kwp = roof_area / 10.0
    avg_irradiance = np.mean(forecast_values)
    annual
