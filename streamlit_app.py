import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import folium
from streamlit_folium import st_folium

# ─── CONFIG & SETUP ────────────────────────────────────────────────────────
st.set_page_config(page_title="Industrial Solar Rooftop Scout", layout="wide", page_icon="☀️")

# 1. ⚙️ สร้างสมุดจดบันทึก (Session State) เพื่อล็อกให้ผลลัพธ์อยู่ถาวร
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False  # บันทึกว่ากดปุ่มหรือยัง
if 'saved_lat' not in st.session_state:
    st.session_state.saved_lat = 18.7883
if 'saved_lng' not in st.session_state:
    st.session_state.saved_lng = 98.9853

# ฟังก์ชันโหลดไฟล์ Scaler แบบปลอดภัย
@st.cache_resource
def load_scaler():
    scaler_path = 'scaler.pkl'
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, 'rb') as f:
                return pickle.load(f), None
        except Exception as e:
            return None, f"❌ โหลด Scaler ไม่สำเร็จ: {str(e)}"
    return None, "⚠️ ไม่พบไฟล์ 'scaler.pkl' บน GitHub (ระบบกำลังใช้โหมดจำลอง)"

with st.spinner("🔄 กำลังเตรียมความพร้อมระบบ Dashboard..."):
    scaler, scaler_err = load_scaler()

# ─── HELPER FUNCTIONS ──────────────────────────────────────────────────────
def get_satellite_image(lat, lng):
    """ จำลองการดึงภาพดาวเทียมรอบพิกัดโรงงาน """
    mock_img = np.zeros((256, 256, 3), dtype=np.uint8)
    mock_img[60:190, 60:190, :] = [180, 180, 180] # วาดหลังคาจำลองสีเทา
    return mock_img

def predict_roof_area(image):
    """ ระบบวิเคราะห์พื้นที่หลังคาโรงงาน """
    binary_mask = np.zeros((256, 256, 1), dtype=np.uint8)
    binary_mask[60:190, 60:190, 0] = 1
    pixel_resolution = 0.25 
    roof_pixels = np.sum(binary_mask)
    calculated_area = roof_pixels * pixel_resolution
    return binary_mask, calculated_area

def predict_solar_irradiance(scaler):
    """ ระบบพยากรณ์ค่ารังสีแสงอาทิตย์ล่วงหน้า 12 เดือน """
    base_values = np.array([4.8, 5.2, 5.5, 5.8, 5.1, 4.3, 4.1, 4.4, 4.6, 4.9, 5.0, 4.7])
    if scaler is not None:
        try:
            scaled = scaler.transform(base_values.reshape(-1, 1)).flatten()
            real_pred = scaler.inverse_transform(scaled.reshape(-1, 1)).flatten()
            return real_pred
        except:
            return base_values
    return base_values

# ─── SIDEBAR & INPUTS ──────────────────────────────────────────────────────
st.sidebar.header("📍 ระบุข้อมูลโรงงาน")
lat_input = st.sidebar.number_input("Latitude (ละติจูด)", value=st.session_state.saved_lat, format="%.6f")
lng_input = st.sidebar.number_input("Longitude (ลองจิจูด)", value=st.session_state.saved_lng, format="%.6f")

# เมื่อกดปุ่ม เริ่มวิเคราะห์ จะทำการจดบันทึกลงสมุดว่า "กดแล้ว"
if st.sidebar.button("🚀 เริ่มวิเคราะห์ระบบโซลาร์", type="primary"):
    st.session_state.analyzed = True
    st.session_state.saved_lat = lat_input
    st.session_state.saved_lng = lng_input

# ปุ่มสำหรับรีเซ็ตหน้าจอ เผื่อป้าอยากล้างหน้าไพ่เริ่มใหม่
if st.sidebar.button("🔄 ล้างข้อมูลผลลัพธ์"):
    st.session_state.analyzed = False
    st.rerun()

# ─── MAIN DASHBOARD ────────────────────────────────────────────────────────
st.title("🛰️ Industrial Solar Rooftop Scout Dashboard")
st.write("ระบบ AI วิเคราะห์ศักยภาพและประเมินความคุ้มค่าการติดตั้งโซลาร์เซลล์สำหรับโรงงานอุตสาหกรรม")

if scaler_err:
    st.caption(scaler_err)

# 2. 🔏 ตรวจสอบสมุดบันทึก: ถ้าเคยกดปุ่มวิเคราะห์แล้ว ให้แสดงผลค้างไว้ตลอดกาลจนกว่าจะกดรีเซ็ต
if st.session_state.analyzed:
    with st.spinner("กำลังประมวลผลพิกัดและข้อมูลดาวเทียม..."):
        col1, col2 = st.columns([1, 1])
        
        # 1. แสดงแผนที่พิกัดโรงงาน (ดึงจากค่าที่เซฟไว้ในสมุดบันทึก)
        with col1:
            st.subheader("🗺️ ที่ตั้งโรงงานอุตสาหกรรม")
            m = folium.Map(location=[st.session_state.saved_lat, st.session_state.saved_lng], zoom_start=17)
            folium.Marker([st.session_state.saved_lat, st.session_state.saved_lng], popup="โรงงานเป้าหมาย").add_to(m)
            st_folium(m, width=500, height=300, key="solar_map") # ใส่ key ดักไว้ไม่ให้รันซ้ำ

        # 2. ประมวลผลพื้นที่หลังคา
        sat_img = get_satellite_image(st.session_state.saved_lat, st.session_state.saved_lng)
        roof_mask, roof_area = predict_roof_area(sat_img)
