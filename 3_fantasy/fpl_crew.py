import requests
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# 1. เชื่อมต่อโมเดล deepseek-v4.1-flash:cloud
fpl_llm = LLM(
    model="ollama/deepseek-v4.1-flash:cloud",
    base_url="http://localhost:11434",
    temperature=0.2
)

# 2. Tool ค้นหาตัวเลือกนักเตะจาก FPL Official API
@tool("Query FPL Replacements")
def fetch_fpl_market_data(position: str, max_price: float) -> str:
    """ดึงข้อมูลนักเตะจาก FPL API ตามตำแหน่ง (GKP, DEF, MID, FWD) ที่ราคาไม่เกิน max_price และพร้อมลงเล่น"""
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    try:
        res = requests.get(url, timeout=10)
        data = res.json()
        
        pos_map = {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
        target_pos = pos_map.get(position.upper(), 4)
        teams = {t["id"]: t["short_name"] for t in data["teams"]}
        
        candidates = []
        for p in data["elements"]:
            price = p["now_cost"] / 10.0
            if p["element_type"] == target_pos and price <= max_price and p["status"] == "a":
                candidates.append({
                    "name": p["web_name"],
                    "team": teams.get(p["team"], "UNK"),
                    "price": price,
                    "form": float(p["form"]),
                    "ep_next": float(p["ep_next"] or 0),
                    "selected_by": f"{p['selected_by_percent']}%"
                })
        
        top = sorted(candidates, key=lambda x: x["ep_next"], reverse=True)[:8]
        return str(top)
    except Exception as e:
        return f"Error fetching FPL data: {str(e)}"

# 3. กำหนด Agents
scout = Agent(
    role="FPL Data Analyst & Scout",
    goal="ตรวจเช็คสถานะความพร้อมของทีม ค้นหานักเตะที่มีปัญหาบาดเจ็บ และเสนอชื่อตัวแทนที่ดีที่สุดจาก FPL API",
    backstory="ผู้เชี่ยวชาญสถิติ FPL ที่แม่นยำด้าน Expected Points และโปรแกรมการแข่งขัน",
    tools=[fetch_fpl_market_data],
    llm=fpl_llm,
    verbose=True
)

manager = Agent(
    role="Elite FPL Manager",
    goal="ตัดสินใจเลือกย้ายตัว จัด 11 ตัวจริง ลำดับตัวสำรอง และเลือกกัปตันทีม",
    backstory="กุนซือ FPL อันดับท็อปที่มีแนวทางอนุรักษ์แต้ม ไม่เปลี่ยนตัวโดยไม่จำเป็น และเลือกตัวทำแต้มสูงสุด",
    llm=fpl_llm,
    verbose=True
)

def run_fpl_crew(team_data_json: str):
    task1 = Task(
        description=f"""
        วิเคราะห์สภาพทีมจากข้อมูล JSON นี้:
        {team_data_json}
        
        1. ตรวจสอบนักเตะที่ติดสถานะ RED (เจ็บ) หรือ YELLOW (เช็คฟิต)
        2. เช็คว่ามีงบเหลือในธนาคาร (budget_itb) เท่าใด และหากขายตัวเจ็บออก จะมีงบสูงสุดเท่าใด
        3. ใช้เครื่องมือ Query FPL Replacements ค้นหาตัวเลือกทดแทนที่ฟอร์มดีที่สุดและมี ep_next สูงสุด
        """,
        expected_output="สรุปรายชื่อนักเตะที่มีปัญหา และตัวเลือกเข้าใหม่ 3-5 คนพร้อมราคาและสถิติ",
        agent=scout
    )

    task2 = Task(
        description=f"""
        ข้อมูลทีมปัจจุบัน (JSON):
        {team_data_json}

        จากผลการวิเคราะห์ของ Scout และข้อมูลทีมข้างต้น:
        1. สรุปคำแนะนำการเปลี่ยนตัว (ย้ายออก - ย้ายเข้า หรือ Roll Transfer เก็บโควตา)
        2. แนะนำการใช้ชิป (Wildcard / Free Hit) หากสถานการณ์ทีมวิกฤต
        3. จัดทีมตัวจริง 11 คน (Starting XI) พร้อมจัดรูปแบบแผน (เช่น 3-4-3 หรือ 3-5-2)
        4. จัดลำดับตัวสำรอง 4 คน (GK สำรอง + ม้านั่งลำดับ 1, 2, 3)
        5. ฟันธง Captain (C) และ Vice-Captain (VC) พร้อมเหตุผลสั้นๆ
        """,
        expected_output="แผนปฏิบัติการจัดทีม FPL รายสัปดาห์ในรูปแบบ Markdown อ่านง่าย ชัดเจน",
        agent=manager,
        context=[task1]
    )

    crew = Crew(
        agents=[scout, manager],
        tasks=[task1, task2],
        process=Process.sequential
    )
    return crew.kickoff()