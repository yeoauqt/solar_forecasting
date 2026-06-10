import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import folium
from streamlit_folium import st_folium

# ─── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="วิเคราะห์ศักยภาพโซลาร์หลังคาโรงงาน",
    layout="wide",
    page_icon=None
)

# ─── CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Sarabun', sans-serif;
}

.stApp {
    background-color: #F0F4F8;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 2rem 2.5rem 3rem 2.5rem !important;
    max-width: 1200px;
}

/* ── Header ── */
.page-header {
    display: flex;
    align-items: flex-start;
    gap: 1.25rem;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #E2E8F0;
}
.page-header-accent {
    width: 4px;
    height: 52px;
    background: linear-gradient(180deg, #F59E0B, #FCD34D);
    border-radius: 2px;
    flex-shrink: 0;
    margin-top: 2px;
}
.page-title {
    font-family: 'Space Grotesk', 'Sarabun', sans-serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: #1E293B;
    letter-spacing: -0.01em;
    margin: 0 0 0.3rem 0;
    line-height: 1.2;
}
.page-subtitle {
    font-size: 0.92rem;
    color: #64748B;
    margin: 0;
    font-weight: 400;
    line-height: 1.6;
}

/* ── Section label ── */
.section-label {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94A3B8;
    margin: 1.75rem 0 0.75rem 0;
}

/* ── Metric cards ── */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.25rem 1.5rem 1.35rem 1.5rem;
    position: relative;
    overflow: hidden;
    height: 100%;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #F59E0B, #FCD34D);
}
.metric-label {
    font-size: 0.8rem;
    font-weight: 500;
    color: #94A3B8;
    margin-bottom: 0.6rem;
}
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #1E293B;
    line-height: 1;
}
.metric-unit {
    font-size: 0.82rem;
    font-weight: 500;
    color: #94A3B8;
    margin-left: 0.3rem;
    font-family: 'Sarabun', sans-serif;
}

/* ── Info box ── */
.info-box {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    height: 100%;
    box-sizing: border-box;
}
.info-box-title {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94A3B8;
    margin: 0 0 0.75rem 0;
}
.info-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.55rem 0;
    border-bottom: 1px solid #F1F5F9;
    font-size: 0.88rem;
    gap: 0.5rem;
}
.info-row:last-child { border-bottom: none; }
.info-key { color: #64748B; font-weight: 400; white-space: nowrap; }
.info-val { color: #1E293B; font-weight: 600; text-align: right; }
.highlight-val { color: #D97706; font-weight: 700; text-align: right; }

/* ── Map ── */
.map-wrapper {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    overflow: hidden;
}

/* ── Chart card ── */
.chart-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.25rem 1.5rem 0.75rem 1.5rem;
    margin-top: 0.25rem;
}
.chart-title {
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 0.5rem;
}

/* ── Empty state ── */
.empty-state {
    background: #FFFFFF;
    border: 1px dashed #CBD5E1;
    border-radius: 16px;
    padding: 5rem 2rem;
    text-align: center;
    margin-top: 1rem;
}
.empty-title {
    font-family: 'Space Grotesk', 'Sarabun', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #334155;
    margin-bottom: 0.5rem;
}
.empty-sub {
    font-size: 0.9rem;
    color: #94A3B8;
    line-height: 1.7;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #1E293B !important;
}
[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
    font-family: 'Sarabun', sans-serif !important;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: #F59E0B;
    color: #1E293B !important;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.9rem;
    padding: 0.65rem 1rem;
    width: 100%;
    transition: background 0.2s;
    margin-top: 0.25rem;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #FCD34D;
}
[data-testid="stSidebar"] .stNumberInput label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #94A3B8 !important;
}
[data-testid="stSidebar"] input {
    background-color: #334155 !important;
    border: 1px solid #475569 !important;
    color: #F1F5F9 !important;
    border-radius: 8px !important;
}
.sidebar-brand {
    font-family: 'Space Grotesk', 'Sarabun', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 0.25rem;
}
.sidebar-brand-sub {
    font-size: 0.78rem;
    color: #64748B;
    margin-bottom: 1.25rem;
    line-height: 1.5;
}
.sidebar-divider {
    border: none;
    border-top: 1px solid #334155;
    margin: 1.25rem 0;
}
.sidebar-hint {
    font-size: 0.78rem;
    color: #475569;
    line-height: 1.6;
    margin-top: 1rem;
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
    return float(np.sum(mask) * 0.25)

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
    st.markdown('<div class="sidebar-brand">Solar Rooftop Scout</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-brand-sub">ต้นแบบระบบวิเคราะห์ศักยภาพโซลาร์เซลล์บนหลังคาโรงงาน</div>', unsafe_allow_html=True)
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    lat = st.number_input("ละติจูด", value=st.session_state.lat, format="%.4f")
    lng = st.number_input("ลองจิจูด", value=st.session_state.lng, format="%.4f")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("วิเคราะห์พื้นที่นี้"):
        st.session_state.analyzed = True
        st.session_state.lat = lat
        st.session_state.lng = lng

    if st.session_state.analyzed:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("ล้างผลลัพธ์"):
            st.session_state.analyzed = False
            st.rerun()

    st.markdown('<div class="sidebar-hint">กรอกพิกัดของโรงงานที่ต้องการประเมิน แล้วกด "วิเคราะห์พื้นที่นี้" เพื่อดูผลการคำนวณศักยภาพพลังงานแสงอาทิตย์</div>', unsafe_allow_html=True)

# ─── Page Header ───────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="page-header-accent"></div>
    <div>
        <div class="page-title">วิเคราะห์ศักยภาพโซลาร์หลังคาโรงงาน</div>
        <div class="page-subtitle">ประเมินพื้นที่หลังคา กำลังผลิต และพลังงานที่คาดการณ์ได้ต่อปี<br>เพื่อสนับสนุนการตัดสินใจติดตั้งระบบโซลาร์เซลล์ในโรงงานอุตสาหกรรม</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Main Content ──────────────────────────────────────────
if st.session_state.analyzed:

    area = predict_roof_area()
    forecast = predict_solar_irradiance(scaler)

    capacity_kwp = area / 10
    avg = float(np.mean(forecast))
    annual = capacity_kwp * avg * 0.75 * 365
    month_names_th = ["ม.ค.","ก.พ.","มี.ค.","เม.ย.","พ.ค.","มิ.ย.","ก.ค.","ส.ค.","ก.ย.","ต.ค.","พ.ย.","ธ.ค."]
    best_month = month_names_th[int(np.argmax(forecast))]

    # ─── Metrics ──────────────────────────────────────────
    st.markdown('<div class="section-label">ผลการวิเคราะห์</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">พื้นที่หลังคาที่ประเมินได้</div>
            <div class="metric-value">{area:,.0f}<span class="metric-unit">ตร.ม.</span></div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">กำลังผลิตติดตั้งได้</div>
            <div class="metric-value">{capacity_kwp:,.1f}<span class="metric-unit">kWp</span></div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">พลังงานที่ผลิตได้ต่อปี</div>
            <div class="metric-value">{annual:,.0f}<span class="metric-unit">kWh</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Map + Info ───────────────────────────────────────
    st.markdown('<div class="section-label">ตำแหน่งพื้นที่</div>', unsafe_allow_html=True)

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
            fill_opacity=0.75,
            popup=f"{st.session_state.lat:.4f}, {st.session_state.lng:.4f}"
        ).add_to(m)
        st_folium(m, width=None, height=360, returned_objects=[])
        st.markdown('</div>', unsafe_allow_html=True)

    with col_info:
        st.markdown(f"""
        <div class="info-box">
            <div class="info-box-title">รายละเอียดพิกัด</div>
            <div class="info-row">
                <span class="info-key">ละติจูด</span>
                <span class="info-val">{st.session_state.lat:.4f}</span>
            </div>
            <div class="info-row">
                <span class="info-key">ลองจิจูด</span>
                <span class="info-val">{st.session_state.lng:.4f}</span>
            </div>
            <div class="info-row">
                <span class="info-key">รังสีเฉลี่ย</span>
                <span class="info-val">{avg:.2f} kWh/m²</span>
            </div>
            <div class="info-row">
                <span class="info-key">เดือนที่ดีที่สุด</span>
                <span class="highlight-val">{best_month}</span>
            </div>
            <div class="info-row">
                <span class="info-key">Performance Ratio</span>
                <span class="info-val">0.75</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ─── Chart ────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="chart-card">
        <div class="chart-title">พยากรณ์รังสีแสงอาทิตย์รายเดือน (kWh/m²/วัน)</div>
    """, unsafe_allow_html=True)

    months_th = ["ม.ค.","ก.พ.","มี.ค.","เม.ย.","พ.ค.","มิ.ย.","ก.ค.","ส.ค.","ก.ย.","ต.ค.","พ.ย.","ธ.ค."]
    df = pd.DataFrame({
        "รังสีแสงอาทิตย์ (kWh/m²/วัน)": forecast
    }, index=months_th)

    st.line_chart(df, use_container_width=True, height=220)
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-title">ยังไม่ได้เลือกพื้นที่</div>
        <div class="empty-sub">กรอกพิกัดละติจูดและลองจิจูดของโรงงานในแถบด้านซ้าย<br>แล้วกด "วิเคราะห์พื้นที่นี้" เพื่อดูผลการประเมินศักยภาพโซลาร์เซลล์</div>
    </div>
    """, unsafe_allow_html=True)
