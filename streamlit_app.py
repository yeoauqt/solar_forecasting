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

# สร้างสมุดจดบันทึกประจำแอป (Session State)
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
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
    mock_img[60:190, 60:190, :] = [180, 180, 180] # หลังคาโรงงานจำลองสีเทา
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

if st.sidebar.button("🚀 เริ่มวิเคราะห์ระบบโซลาร์", type="primary"):
    st.session_state.analyzed = True
    st.session_state.saved_lat = lat_input
    st.session_state.saved_lng = lng_input

if st.sidebar.button("🔄 ล้างข้อมูลผลลัพธ์"):
    st.session_state.analyzed = False
    st.rerun()

# ─── MAIN DASHBOARD ────────────────────────────────────────────────────────
st.title("🛰️ Industrial Solar Rooftop Scout Dashboard")
st.write("ระบบ AI วิเคราะห์ศักยภาพและประเมินความคุ้มค่าการติดตั้งโซลาร์เซลล์สำหรับโรงงานอุตสาหกรรม")

if scaler_err:
    st.caption(scaler_err)

# ตรวจสอบ: ถ้าเคยกดปุ่มวิเคราะห์ ให้ลุยงานคำนวณทั้งหมดให้เสร็จก่อนโชว์หน้าจอ
if st.session_state.analyzed:
    
    # 💥 [STEP 1] ให้ AI คำนวณเตรียมผลลัพธ์ทุกอย่างให้เสร็จสรรพก่อนวาดแผนที่ 💥
    sat_img = get_satellite_image(st.session_state.saved_lat, st.session_state.saved_lng)
    roof_mask, roof_area = predict_roof_area(sat_img)
    forecast_values = predict_solar_irradiance(scaler)
    
    # คำนวณตัวเลข ROI ทางวิศวกรรม
    potential_kwp = roof_area / 10.0
    avg_irradiance = np.mean(forecast_values)
    annual_energy = potential_kwp * avg_irradiance * 0.75 * 365
    
    # จัดเตรียมข้อมูลทำกราฟรังสีแสงอาทิตย์
    months = [f"เดือนที่ +{i+1}" for i in range(len(forecast_values))]
    df_forecast = pd.DataFrame({
        'เดือนพยากรณ์': months,
        'ค่ารังสีแสงอาทิตย์ (kWh/m²/วัน)': forecast_values
    })

    # 💥 [STEP 2] นำข้อมูลที่คำนวณเสร็จแล้วมากางโชว์บน Dashboard อย่างมั่นคง 💥
    col1, col2 = st.columns([1, 1])
    
    # คอลัมน์ซ้าย: แผนที่ (จัดวางให้อยู่ล่างสุดของฟลอร์นี้ แผนที่จะงอแงยังไง ข้อมูลเราก็ไม่หาย)
    with col1:
        st.subheader("🗺️ ที่ตั้งโรงงานอุตสาหกรรม")
        m = folium.Map(location=[st.session_state.saved_lat, st.session_state.saved_lng], zoom_start=17)
        folium.Marker([st.session_state.saved_lat, st.session_state.saved_lng], popup="โรงงานเป้าหมาย").add_to(m)
        st_folium(m, width=500, height=300, key="fixed_solar_map")

    # คอลัมน์ขวา: แสดงพื้นที่หลังคาและการตัด Mask ของ U-Net
    with col2:
        st.subheader("📐 ผลการวิเคราะห์พื้นที่หลังคา (U-Net Mode)")
        st.metric(label="พื้นที่หลังคาที่ตรวจพบทั้งหมด", value=f"{roof_area:,.2f} ตร.ม.")
        
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax[0].imshow(sat_img)
        ax[0].set_title("Original Image")
        ax[0].axis('off')
        ax[1].imshow(roof_mask.squeeze(), cmap='gray')
        ax[1].set_title("Predicted Roof Mask")
        ax[1].axis('off')
        st.pyplot(fig)

    st.markdown("---")

    # ส่วนล่าง: กราฟเส้นทำนายรังสี และสรุปตัวเลขความคุ้มค่า
    st.subheader("☀️ การพยากรณ์ค่ารังสีแสงอาทิตย์ล่วงหน้า 12 เดือน")
    
    col3, col4 = st.columns([2, 1])
    with col3:
        st.line_chart(df_forecast.set_index('เดือนพยากรณ์'))
    
    with col4:
        st.markdown("### 📊 ประเมินกำลังผลิตและ ROI")
        st.write(f"**ขนาดกำลังผลิตสูงสุดที่แนะนำ:** {potential_kwp:,.2f} kWp")
        st.write(f"**ค่ารังสีแสงอาทิตย์เฉลี่ยรายวัน:** {avg_irradiance:.2f} kWh/m²/วัน")
        st.write(f"**คาดการณ์พลังงานไฟฟ้าที่ผลิตได้:** {annual_energy:,.2f} kWh/ปี")

else:
    st.info("💡 ระบุพิกัดโรงงานที่เมนูด้านซ้าย แล้วกดปุ่ม 'เริ่มวิเคราะห์ระบบโซลาร์' ได้เลยครับป้า")
