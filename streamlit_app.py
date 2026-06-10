# 📊 SECTION 1: สรุปตัวเลขสำคัญ (Key Metrics) แบบตารางสี่เหลี่ยมสวยๆ ด้านบน
    st.markdown('<div class="section-header">📊 สรุปภาพรวมการประเมินศักยภาพ (Dashboard Summary)</div>', unsafe_html=True)
    
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        metric_box1 = st.container(border=True) # <-- เปลี่ยนชื่อตัวแปรตรงนี้
        metric_box1.metric(label="📐 พื้นที่หลังคาโรงงานทั้งหมด", value=f"{roof_area:,.2f} ตร.ม.", delta="ตรวจพบโดย U-Net")
    with m_col2:
        metric_box2 = st.container(border=True) # <-- เปลี่ยนชื่อตัวแปรตรงนี้
        metric_box2.metric(label="🔌 ขนาดกำลังผลิตที่ติดตั้งได้", value=f"{potential_kwp:,.2f} kWp", delta="คำนวณจากพื้นที่จริง")
    with m_col3:
        metric_box3 = st.container(border=True) # <-- เปลี่ยนชื่อตัวแปรตรงนี้
        metric_box3.metric(label="🔋 คาดการณ์พลังงานที่ผลิตได้ต่อปี", value=f"{annual_energy:,.2f} kWh/ปี", delta="ความคุ้มค่าเกรดสูง")
