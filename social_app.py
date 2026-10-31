from __future__ import annotations

import json
import logging
import threading
import time
from queue import Empty, Queue

import streamlit as st

from social_pipeline import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    STAGES,
    SocialContentPack,
    run_pipeline,
    to_markdown,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


# -------------------------------------------------------------------
# Page
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Social Content Studio",
    page_icon="🎬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border-radius: 12px;
        padding: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


DEMO_BRIEF = """อธิบายวิดีโอเกี่ยวกับ...
"""


if "job" not in st.session_state:
    st.session_state.job = None


def is_running() -> bool:
    job = st.session_state.job
    return bool(job and job["running"])


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------

with st.sidebar:
    st.header("ตั้งค่าการทำงาน")

    model = st.text_input(
        "CrewAI model",
        value=DEFAULT_MODEL,
        disabled=is_running(),
    )

    base_url = st.text_input(
        "Ollama Base URL",
        value=DEFAULT_BASE_URL,
        disabled=is_running(),
    )



    st.divider()

    st.markdown("**Workflow**")

    for index, stage in enumerate(STAGES, start=1):
        st.write(f"{index}. {stage['title']}")

    st.caption(
        "หน้าจอแสดงสถานะงานและผลงานที่ Agent ส่งกลับ "
        "ไม่ใช่กระบวนการคิดภายในของโมเดล"
    )


# -------------------------------------------------------------------
# User form
# -------------------------------------------------------------------

st.title("🎬 Social Content Studio")
st.write(
    "อธิบายวิดีโอครั้งเดียว "
    "ให้ทีม AI เตรียมข้อความสำหรับ YouTube, Facebook และ TikTok"
)

with st.form("video_form"):
    video_brief = st.text_area(
        "วิดีโอนี้เกี่ยวกับอะไร?",
        value=DEMO_BRIEF,
        height=270,
        disabled=is_running(),
    )

    left, right = st.columns(2)

    with left:
        audience = st.text_input(
            "กลุ่มเป้าหมาย",
            value="ผู้เริ่มต้น Python และคนที่สนใจสร้าง AI Agent",
            disabled=is_running(),
        )

        tone = st.selectbox(
            "น้ำเสียง",
            options=[
                "เป็นกันเอง เข้าใจง่าย ไม่โอเวอร์",
                "มืออาชีพ กระชับ น่าเชื่อถือ",
                "สนุก ชวนติดตาม แต่ไม่กล่าวเกินจริง",
                "สอนแบบจับมือทำ เหมาะกับมือใหม่",
            ],
            disabled=is_running(),
        )

        language = st.selectbox(
            "ภาษาของโพสต์",
            options=["ไทย", "English"],
            disabled=is_running(),
        )

    with right:
        brand_name = st.text_input(
            "ชื่อช่อง / แบรนด์ — ไม่บังคับ",
            value="",
            disabled=is_running(),
        )

        cta = st.text_input(
            "อยากชวนคนดูทำอะไร?",
            value="บันทึกคลิปไว้ทำตาม และคอมเมนต์สิ่งที่อยากให้สอนต่อ",
            disabled=is_running(),
        )

        st.caption(
            "ค่าเริ่มต้น: YouTube วิดีโอปกติ, "
            "Facebook Feed และ TikTok แนวตั้ง"
        )

    cloud_consent = st.checkbox(
        "ยินยอมส่งข้อมูลที่กรอกไปประมวลผลผ่าน Ollama Cloud",
        value=False,
        disabled=is_running(),
    )

    submitted = st.form_submit_button(
        "🚀 เริ่มสร้างชุดคอนเทนต์",
        type="primary",
        disabled=is_running(),
        use_container_width=True,
    )


# -------------------------------------------------------------------
# Start background worker
# -------------------------------------------------------------------

if submitted:
    errors = []

    if not video_brief.strip():
        errors.append("กรุณาอธิบายเนื้อหาวิดีโอ")

    if not model.strip().startswith("ollama/"):
        errors.append("ตัวอย่างนี้ต้องใช้ชื่อโมเดลที่ขึ้นต้นด้วย ollama/")

    if not base_url.strip():
        errors.append("กรุณาระบุ Ollama Base URL")

    if not cloud_consent:
        errors.append("กรุณายืนยันการส่งข้อมูลไปประมวลผลบน Cloud")

    if errors:
        for error in errors:
            st.error(error)

    else:
        event_queue = Queue()

        inputs = {
            "video_brief": video_brief.strip(),
            "audience": audience.strip(),
            "tone": tone,
            "language": language,
            "brand_name": brand_name.strip(),
            "cta": cta.strip(),
        }

        job = {
            "running": True,
            "started_at": time.monotonic(),
            "finished_at": None,
            "events": event_queue,
            "connection": None,
            "error": None,
            "result": None,
            "model": model.strip(),
            "stages": {
                stage["id"]: {
                    "status": "pending",
                    "started_at": None,
                    "elapsed": None,
                    "output": None,
                }
                for stage in STAGES
            },
        }

        worker = threading.Thread(
            target=run_pipeline,
            kwargs={
                "user_inputs": inputs,
                "model": model.strip(),
                "base_url": base_url.strip(),
                "events": event_queue,
            },
            daemon=True,
            name="social-content-worker",
        )

        job["thread"] = worker
        st.session_state.job = job

        worker.start()
        st.rerun()


# -------------------------------------------------------------------
# UI helpers
# -------------------------------------------------------------------

STATUS_LABELS = {
    "pending": "⚪ รอทำงาน",
    "running": "🟡 กำลังทำงาน",
    "completed": "🟢 เสร็จแล้ว",
    "failed": "🔴 ไม่สำเร็จ",
    "skipped": "⚫ ไม่ได้ทำงาน",
}


def copy_box(label: str, value: str) -> None:
    st.markdown(f"**{label}**")
    st.code(value, language=None, wrap_lines=True)


def render_result(data: dict) -> None:
    pack = SocialContentPack.model_validate(data)

    st.header("ชุดคอนเทนต์พร้อมคัดลอก")

    st.caption(
        "ชี้เมาส์ที่กล่องข้อความแล้วกดไอคอน Copy มุมขวาบน "
        "ควรตรวจทานก่อนเผยแพร่"
    )

    tabs = st.tabs(["YouTube", "Facebook", "TikTok"])

    for tab, platform in zip(
        tabs,
        ("youtube", "facebook", "tiktok"),
    ):
        item = getattr(pack, platform)

        with tab:
            content_col, image_col = st.columns([1.15, 1])

            with content_col:
                st.subheader("ข้อความสำหรับโพสต์")

                for index, title in enumerate(item.titles, start=1):
                    suffix = " — แนะนำ" if index == 1 else ""

                    copy_box(
                        f"Title / Hook {index}{suffix}",
                        title,
                    )

                copy_box(
                    "Description / Caption",
                    item.description,
                )

                copy_box(
                    "Hashtags",
                    " ".join(item.hashtags),
                )

                copy_box(
                    "Caption รวม Hashtags — คัดลอกครั้งเดียว",
                    item.description.rstrip()
                    + "\n\n"
                    + " ".join(item.hashtags),
                )

            with image_col:
                st.subheader("Prompt สำหรับภาพปก")

                st.info(
                    "ส่วนนี้สร้างข้อความ Prompt เท่านั้น "
                    "ให้นำไปใช้กับเครื่องมือสร้างภาพ"
                )

                copy_box(
                    "Text-to-image Prompt",
                    item.image_prompt,
                )

                copy_box(
                    "Negative Prompt",
                    item.negative_prompt,
                )

                copy_box(
                    "ข้อความบนภาพ — นำไปวางภายหลัง",
                    item.cover_text,
                )

                copy_box(
                    "สัดส่วนภาพที่แนะนำ",
                    item.aspect_ratio,
                )

                st.caption(
                    "เครื่องมือสร้างภาพบางตัวไม่มีช่อง Negative Prompt "
                    "และอาจต้องตั้งสัดส่วนภาพในหน้าตั้งค่าของเครื่องมือนั้น"
                )

            if item.review_notes:
                with st.expander("ข้อควรตรวจสอบของแพลตฟอร์มนี้"):
                    for note in item.review_notes:
                        st.write(f"- {note}")

    if pack.review_notes:
        st.warning("มีข้อมูลที่ควรตรวจสอบก่อนเผยแพร่")

        for note in pack.review_notes:
            st.write(f"- {note}")

    st.divider()
    st.subheader("ดาวน์โหลดผลงาน")

    json_data = json.dumps(
        pack.model_dump(),
        ensure_ascii=False,
        indent=2,
    )

    markdown_data = to_markdown(pack)

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "ดาวน์โหลด JSON",
            data=json_data,
            file_name="social_content.json",
            mime="application/json",
            use_container_width=True,
        )

    with col2:
        st.download_button(
            "ดาวน์โหลด Markdown",
            data=markdown_data,
            file_name="social_content.md",
            mime="text/markdown",
            use_container_width=True,
        )


# -------------------------------------------------------------------
# Live progress panel
# -------------------------------------------------------------------

@st.fragment(run_every=0.5)
def live_panel() -> None:
    job = st.session_state.job

    if job is None:
        st.info(
            "กรอกบรีฟแล้วกดเริ่ม "
            "หรือใช้บรีฟตัวอย่างด้านบนสำหรับทดลอง Demo"
        )
        return

    terminal_event_received = False

    # Read all available events. Never block the Streamlit thread.
    while True:
        try:
            event = job["events"].get_nowait()
        except Empty:
            break

        kind = event["kind"]

        if kind == "connection_ok":
            job["connection"] = event["version"]

        elif kind == "stage_started":
            state = job["stages"][event["stage_id"]]
            state["status"] = "running"
            state["started_at"] = event["started_at"]

        elif kind == "stage_completed":
            state = job["stages"][event["stage_id"]]
            state["status"] = "completed"
            state["elapsed"] = event["elapsed"]
            state["output"] = event["output"]

        elif kind == "finished":
            job["running"] = False
            job["finished_at"] = time.monotonic()
            job["result"] = event["result"]
            terminal_event_received = True

        elif kind == "failed":
            job["running"] = False
            job["finished_at"] = time.monotonic()
            job["error"] = event
            terminal_event_received = True

            stage_id = event.get("stage_id")

            if stage_id and stage_id in job["stages"]:
                state = job["stages"][stage_id]
                state["status"] = "failed"

                if state["started_at"] is not None:
                    state["elapsed"] = (
                        time.monotonic() - state["started_at"]
                    )

            for state in job["stages"].values():
                if state["status"] == "pending":
                    state["status"] = "skipped"

    # A full rerun updates disabled form controls when the job finishes.
    if terminal_event_received:
        st.rerun()

    st.divider()
    st.header("Agent Workflow")

    if job["connection"]:
        st.caption(
            f"Ollama {job['connection']} • {job['model']}"
        )
    elif job["running"]:
        st.caption("กำลังตรวจการเชื่อมต่อ Ollama...")

    completed = sum(
        state["status"] == "completed"
        for state in job["stages"].values()
    )

    active = next(
        (
            stage
            for stage in STAGES
            if job["stages"][stage["id"]]["status"] == "running"
        ),
        None,
    )

    now = job["finished_at"] or time.monotonic()
    elapsed = now - job["started_at"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("ขั้นตอนที่เสร็จ", f"{completed} / {len(STAGES)}")

    with col2:
        st.metric("เวลารวม", f"{elapsed:.1f} วินาที")

    with col3:
        if active:
            current_label = active["title"]
        elif job["result"]:
            current_label = "พร้อมใช้งาน"
        elif job["error"]:
            current_label = "งานหยุด"
        else:
            current_label = "เตรียมระบบ"

        st.metric("สถานะปัจจุบัน", current_label)

    progress = completed / len(STAGES)

    st.progress(
        progress,
        text=f"เสร็จแล้ว {completed} จาก {len(STAGES)} ขั้นตอน",
    )

    st.caption(
        "เปอร์เซ็นต์คำนวณจากขั้นตอนที่เสร็จ "
        "แต่ละขั้นอาจใช้เวลาไม่เท่ากัน"
    )

    for row_start in range(0, len(STAGES), 3):
        columns = st.columns(3)

        for column, stage in zip(
            columns,
            STAGES[row_start:row_start + 3],
        ):
            state = job["stages"][stage["id"]]

            with column:
                with st.container(border=True):
                    st.markdown(f"**{stage['title']}**")
                    st.caption(stage["agent"])
                    st.write(STATUS_LABELS[state["status"]])

                    if state["status"] == "running":
                        stage_elapsed = (
                            time.monotonic() - state["started_at"]
                        )
                        st.caption(
                            f"กำลังประมวลผล {stage_elapsed:.1f} วินาที"
                        )

                    elif state["elapsed"] is not None:
                        st.caption(
                            f"ใช้เวลา {state['elapsed']:.1f} วินาที"
                        )

    completed_outputs = [
        stage
        for stage in STAGES
        if job["stages"][stage["id"]]["output"] is not None
    ]

    if completed_outputs:
        with st.expander("ดูผลงานระหว่างทางของแต่ละ Agent"):
            st.caption(
                "ข้อมูลนี้เป็นผลงานของ Task "
                "ก่อน Publishing Editor ตรวจรวมในขั้นสุดท้าย"
            )

            for stage in completed_outputs:
                st.markdown(f"**{stage['title']}**")
                st.json(
                    job["stages"][stage["id"]]["output"],
                    expanded=False,
                )

    if job["running"]:
        st.info(
            "ทีม AI กำลังทำงาน "
            "เปิดดูผลงานของขั้นที่เสร็จแล้วได้ระหว่างรอ"
        )

    elif job["error"]:
        error = job["error"]

        st.error(error["message"])
        st.caption(f"ประเภทข้อผิดพลาด: {error['error_type']}")

        st.write(
            "ผลลัพธ์ของขั้นที่เสร็จแล้วยังดูได้ด้านบน "
            "เมื่อแก้ปัญหาแล้ว กดเริ่มเพื่อรันงานใหม่ทั้งชุด"
        )


live_panel()


# Render final results outside the polling fragment.
job = st.session_state.job

if job and job["result"]:
    st.success("สร้างชุดคอนเทนต์ครบแล้ว")
    render_result(job["result"])