import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import folium
from streamlit_folium import st_folium

# ─── ตั้งค่าหน้าเว็บ ─────────────────────────────────────
st.set_page_config(
    page_title="Industrial Solar Rooftop Scout",
    layout="wide",
    page_icon="☀️"
)

# ─── CSS ตกแต่ง ──────────────────────────────────────────
st.markdown("""
    <style>
    .main-title { font-size: 36px; font-weight: 800; color: #1E3A8A; }
    .sub-title { font-size: 16px; color: #4B5563; margin-bottom: 20px; }
    .section-header { font-size: 22px; font-weight: 700; color: #1E3A8A; margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)

# ─── SESSION STATE (เก็บค่าระหว่างใช้งาน) ───────────────
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "lat" not in st.session_state:
    st.session_state.lat = 18.7883
if "lng" not in st.session_state:
    st.session_state.lng = 98.9853

# ─── โหลด scaler (ถ้ามีไฟล์) ───────────────────────────
@st.cache_resource
def load_scaler():
    if os.path.exists("scaler.pkl"):
        try:
            with open("scaler.pkl", "rb") as f:
                return pickle.load(f), None
        except Exception as e:
            return None, f"โหลด scaler ไม่ได้: {e}"
    return None, "ไม่พบ scaler.pkl (ใช้โหมดจำลอง)"

scaler, scaler_err = load_scaler()

# ─── ฟังก์ชันจำลอง AI ───────────────────────────────────
def get_satellite_image(lat, lng):
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[50:200, 40:210] = [140, 145, 150]
    return img

def predict_roof_area(image):
    mask = np.zeros((256, 256, 1))
    mask[50:200, 40:210, 0] = 1
    area = np.sum(mask) * 0.25
    return mask, area

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

# ─── SIDEBAR ─────────────────────────────────────────────
st.sidebar.title("☀️ แผงควบคุม")

lat = st.sidebar.number_input("ละติจูด", value=st.session_state.lat)
lng = st.sidebar.number_input("ลองจิจูด", value=st.session_state.lng)

if st.sidebar.button("🚀 เริ่มวิเคราะห์"):
    st.session_state.analyzed = True
    st.session_state.lat = lat
    st.session_state.lng = lng

if st.sidebar.button("🔄 รีเซ็ต"):
    st.session_state.analyzed = False
    st.rerun()

# ─── หัวเรื่อง ───────────────────────────────────────────
st.markdown('<div class="main-title">☀️ Solar Rooftop Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">ระบบ AI วิเคราะห์ศักยภาพโซลาร์บนหลังคาโรงงาน</div>', unsafe_allow_html=True)

if scaler_err:
    st.caption(scaler_err)

# ─── ถ้ากดเริ่มวิเคราะห์ ────────────────────────────────
if st.session_state.analyzed:

    img = get_satellite_image(st.session_state.lat, st.session_state.lng)
    mask, area = predict_roof_area(img)
    forecast = predict_solar_irradiance(scaler)

    capacity_kwp = area / 10
    avg = np.mean(forecast)
    annual = capacity_kwp * avg * 0.75 * 365

    months = ["ม.ค.","ก.พ.","มี.ค.","เม.ย.","พ.ค.","มิ.ย.",
              "ก.ค.","ส.ค.","ก.ย.","ต.ค.","พ.ย.","ธ.ค."]

    df = pd.DataFrame({
        "เดือน": months,
        "รังสีแสงอาทิตย์": forecast
    }).set_index("เดือน")

    # ─── สรุปผล ─────────────────────────────
    st.markdown('<div class="section-header">📊 สรุปผล</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("พื้นที่หลังคา", f"{area:,.2f} ตร.ม.")

    with c2:
        st.metric("กำลังผลิต", f"{capacity_kwp:,.2f} kWp")

    with c3:
        st.metric("พลังงานต่อปี", f"{annual:,.2f} kWh")

    # ─── แผนที่ + ภาพ ───────────────────────
    st.markdown('<div class="section-header">🛰️ แผนที่และภาพวิเคราะห์</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        m = folium.Map(location=[st.session_state.lat, st.session_state.lng], zoom_start=17)
        folium.Marker([st.session_state.lat, st.session_state.lng]).add_to(m)
        st_folium(m, width=None, height=300)

    with col2:
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))

        ax[0].imshow(img)
        ax[0].set_title("ภาพดาวเทียม")
        ax[0].axis("off")

        ax[1].imshow(mask.squeeze(), cmap="hot")
        ax[1].set_title("Mask หลังคา")
        ax[1].axis("off")

        st.pyplot(fig)

    # ─── กราฟพยากรณ์ ───────────────────────
    st.markdown('<div class="section-header">🌤️ พยากรณ์แสงอาทิตย์</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.line_chart(df)

    with col2:
        st.info(f"""
ค่าเฉลี่ย: {avg:.2f} kWh/m²/วัน

ช่วงดีที่สุด: มี.ค. - เม.ย.

เหมาะสำหรับติดตั้งโซลาร์เซลล์
""")

else:
    st.info("กรุณากรอกพิกัดแล้วกดปุ่ม เริ่มวิเคราะห์")
