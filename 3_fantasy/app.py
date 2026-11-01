import streamlit as st
from PIL import Image
import json

from pydantic import ValidationError

from parsing import team_json_for_prompt
from schemas import FPLTeamState
from vision_extractor import extract_fpl_data_from_image
from fpl_crew import run_fpl_crew

st.set_page_config(page_title="FPL AI Squad Planner", layout="wide", page_icon="⚽")

st.title("⚽ FPL Squad Planner with CrewAI & DeepSeek")
st.caption("จัดทีม Fantasy Premier League อัตโนมัติจากภาพ Screenshot หน้า Transfers")

# แบ่งหน้าจอเป็น 2 คอลัมน์หลัก
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1. อัปโหลดภาพ Screenshot ประจำสัปดาห์")
    uploaded_file = st.file_uploader("เลือกรูปภาพหน้าจัดทีม FPL", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="รูปภาพที่อัปโหลด", use_container_width=True)
        
        if st.button("🔍 ดึงข้อมูลจากรูปภาพ (Extract to JSON)", type="primary"):
            with st.spinner("กำลังอ่านข้อมูลจากภาพด้วย Vision Model..."):
                try:
                    extracted_data = extract_fpl_data_from_image(image)
                    st.session_state["fpl_state"] = extracted_data.model_dump()
                    st.session_state["fpl_json_editor"] = json.dumps(
                        st.session_state["fpl_state"], indent=2, ensure_ascii=False
                    )
                    st.session_state["fpl_extract_round"] = (
                        st.session_state.get("fpl_extract_round", 0) + 1
                    )
                    st.session_state.pop("crew_result", None)
                    st.success("สกัดข้อมูลสำเร็จ!")
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการอ่านภาพ: {str(e)}")

with col_right:
    st.subheader("2. ตรวจสอบข้อมูลทีม & รัน CrewAI")
    
    if "fpl_state" in st.session_state:
        state = st.session_state["fpl_state"]
        
        # แสดงสรุปสถานะทีม
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Gameweek", state.get("gameweek", "-"))
        m2.metric("เงินในธนาคาร", f"£{state.get('budget_itb', 0.0)}m")
        m3.metric("Free Transfers", state.get("free_transfers", 0))
        m4.metric("Chips", f"WC: {'✅' if state.get('wildcard_available') else '❌'}")
        
        # ตัวแก้ไข JSON หากผู้ใช้ต้องการแก้ไขค่าก่อนประมวลผล
        # ใช้ key ที่ผูกกับเลขรอบการสกัด เพื่อให้ค่าใหม่ทับค่าเดิมได้จริงเมื่อสกัดรูปใหม่
        editor_key = f"fpl_json_editor_{st.session_state.get('fpl_extract_round', 0)}"
        with st.expander("ตรวจสอบและแก้ไขข้อมูล JSON ดิบ"):
            editable_json = st.text_area(
                "JSON Data",
                value=st.session_state.get(
                    "fpl_json_editor",
                    json.dumps(state, indent=2, ensure_ascii=False),
                ),
                height=250,
                key=editor_key,
            )
            
        if st.button("🚀 วางแผนจัดทีมด้วย CrewAI & DeepSeek", type="primary"):
            try:
                team_state = FPLTeamState.model_validate_json(editable_json)
            except ValidationError as e:
                st.error(f"JSON ไม่ถูกต้องตามโครงสร้าง FPLTeamState: {str(e).splitlines()[0]}")
            else:
                with st.spinner("AI Agents กำลังวิเคราะห์สถิติ ค้นหานักเตะ และวางแผนจัดทีม..."):
                    try:
                        result = run_fpl_crew(team_json_for_prompt(team_state))
                        st.session_state["crew_result"] = str(result)
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดระหว่างรัน CrewAI: {str(e)}")
    else:
        st.info("กรุณาอัปโหลดรูปภาพและกดปุ่มสกัดข้อมูลด้านซ้ายเพื่อเริ่มต้น")

# แสดงผลลัพธ์การจัดทีม
if "crew_result" in st.session_state:
    st.divider()
    st.subheader("📋 แผนการจัดทีมและคำแนะนำจาก AI")
    st.markdown(st.session_state["crew_result"])