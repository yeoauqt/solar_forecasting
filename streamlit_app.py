import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import folium
from streamlit_folium import st_folium

# ตรวจสอบว่ามี tensorflow หรือไม่ (ป้องกันกรณีไลบรารียังลงไม่เสร็จ)
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ─── CONFIG & SETUP ────────────────────────────────────────────────────────
st.set_page_config(page_title="Industrial Solar Rooftop Scout", layout="wide", page_icon="☀️")

# ฟังก์ชันโหลดโมเดลและ Scaler แบบปลอดภัย ไม่ทำให้หน้าเว็บล่มล่วงหน้า
@st.cache_resource
def load_all_models():
    models = {"unet": None, "gru": None, "scaler": None, "errors": []}
    
    if not TF_AVAILABLE:
        models["errors"].append("❌ ไลบรารี TensorFlow ยังติดตั้งบนเซิร์ฟเวอร์ไม่เสร็จสมบูรณ์")
        return models

    # 1. โหลด U-Net
    unet_path = 'u_net_model.keras'
    if os.path.exists(unet_path):
        try:
            models["unet"] = load_model(unet_path, compile=False)
        except Exception as e:
            models["errors"].append(f"❌ โหลด U-Net ไม่สำเร็จ: {str(e)}")
    else:
        # ลองหาแบบนามสกุล .h5 เผื่อป้าเซฟเป็นชื่อนี้
        if os.path.exists('u_net_model.h5'):
            try:
                models["unet"] = load_model('u_net_model.h5', compile=False)
            except Exception as e:
                models["errors"].append(f"❌ โหลด U-Net (.h5) ไม่สำเร็จ: {str(e)}")
        else:
            models["errors"].append("⚠️ ไม่พบไฟล์โมเดล U-Net ('u_net_model.keras' หรือ 'u_net_model.h5') บน GitHub")

    # 2. โหลด GRU
    gru_path = 'gru_model.keras'
    if os.path.exists(gru_path):
        try:
            models["gru"] = load_model(gru_path, compile=False)
        except Exception as e:
            models["errors"].append(f"❌ โหลด GRU ไม่สำเร็จ: {str(e)}")
    else:
        if os.path.exists('gru_model.h5'):
            try:
                models["gru"] = load_model('gru_model.h5', compile=False)
            except Exception as e:
                models["errors"].append(f"❌ โหลด GRU (.h5) ไม่สำเร็จ: {str(e)}")
        else:
            models["errors"].append("⚠️ ไม่พบไฟล์โมเดล GRU ('gru_model.keras' หรือ 'gru_model.h5') บน GitHub")

    # 3. โหลด Scaler
    scaler_path = 'scaler.pkl'
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, 'rb') as f:
                models["scaler"] = pickle.load(f)
        except Exception as e:
            models["errors"].append(f"❌ โหลด Scaler ไม่สำเร็จ: {str(e)}")
    else:
        models["errors"].append("⚠️ ไม่พบไฟล์ 'scaler.pkl' บน GitHub")

    return models

# เรียกใช้งานฟังก์ชันโหลดโมเดล
with st.spinner("🔄 กำลังเตรียมความพร้อมระบบและโมเดล AI..."):
    ai_models = load_all_models()

# ─── HELPER FUNCTIONS ──────────────────────────────────────────────────────
def get_satellite_image(lat, lng):
    """ จำลองการดึงภาพดาวเทียมรอบพิกัดโรงงาน """
    # สร้างภาพจำลองขนาด 256x256 มี 3 แชนเนลสี (RGB)
    mock_img = np.zeros((256, 256, 3), dtype=np.uint8)
    # วาดรูปสี่เหลี่ยมจำลองเป็นหลังคาโรงงานตรงกลางภาพ
    mock_img[60:190, 60:190, :] = [180, 180, 180] 
    return mock_img

def predict_roof_area(image, model):
    """ ใช้ U-Net หาพื้นที่หลังคาโรงงานจากภาพถ่ายดาวเทียม """
    if model is None:
        # หากไม่มีโมเดลจริง ให้ระบบจำลองค่าพื้นที่ขึ้นมาแสดงผลก่อนเพื่อทดสอบหน้าเว็บ
        mock_mask = np.zeros((256, 256, 1), dtype=np.uint8)
        mock_mask[60:190, 60:190, 0] = 1
        return mock_mask, 4225.0  # สมมุติพื้นที่หลังคา 4,225 ตร.ม.

    # ขั้นตอนกรณีมีโมเดลจริง
    img_input = np.expand_dims(image / 255.0, axis=0)
    mask = model.predict(img_input, verbose=0)[0]
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    # คำนวณพื้นที่ (ตร.ม.) 1 พิกเซลสมมุติเท่ากับ 0.25 ตร.ม.
    pixel_resolution = 0.25 
    roof_pixels = np.sum(binary_mask)
    calculated_area = roof_pixels * pixel_resolution
    
    return binary_mask, calculated_area

def predict_solar_irradiance(model, scaler):
    """ ใช้ GRU พยากรณ์ค่ารังสีแสงอาทิตย์ล่วงหน้า 12 เดือน """
    if model is None or scaler is None:
        # หากไม่มีโมเดลจริง ให้จำลองค่ารังสีขึ้นมาแสดงเป็นกราฟก่อน
        return np.array([4.8, 5.2, 5.5, 5.8, 5.1, 4.3, 4.1, 4.4, 4.6, 4.9, 5.0, 4.7])

    # ขั้นตอนกรณีมีโมเดลจริง (ป้อน Input ย้อนหลังขนาดตามที่เทรนไว้)
    mock_input_scaled = np.random.rand(1, 12, 1) 
    pred_scaled = model.predict(mock_input_scaled, verbose=0)
    real_pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    return real_pred

# ─── SIDEBAR & INPUTS ──────────────────────────────────────────────────────
st.sidebar.header("📍 ระบุข้อมูลโรงงาน")
lat_input = st.sidebar.number_input("Latitude (ละติจูด)", value=18.7883, format="%.6f")
lng_input = st.sidebar.number_input("Longitude (ลองจิจูด)", value=98.9853, format="%.6f")

analyze_btn = st.sidebar.button("🚀 เริ่มวิเคราะห์ระบบโซลาร์", type="primary")

# ─── MAIN DASHBOARD ────────────────────────────────────────────────────────
st.title("🛰️ Industrial Solar Rooftop Scout Dashboard")
st.write("ระบบ AI วิเคราะห์ศักยภาพและประเมินความคุ้มค่าการติดตั้งโซลาร์เซลล์สำหรับโรงงานอุตสาหกรรม")

# แสดงกล่องแจ้งเตือนสถานะไฟล์โมเดลที่
