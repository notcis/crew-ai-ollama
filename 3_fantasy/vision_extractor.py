import base64
import json
from io import BytesIO

from PIL import Image
from ollama import Client
from pydantic import ValidationError

from parsing import parse_team_state
from schemas import FPLTeamState

DEFAULT_MODEL = "deepseek-v4.1-flash:cloud"

REPAIR_SYSTEM = (
    "You convert scouting notes into strict JSON. Output raw JSON only - no prose, "
    "no markdown. Use the JSON Schema's top-level keys directly, do not nest them."
)


def build_prompt(schema_json: str) -> str:
    """ฝัง JSON Schema ไว้ใน prompt เพราะโมเดลไม่ทำตามพารามิเตอร์ format= เพียงอย่างเดียว"""
    return f"""Extract the Fantasy Premier League (FPL) squad from this screenshot.

Return ONLY a raw JSON object matching this JSON Schema. Use the schema's top-level keys
directly - never nest the values under "header" or "players":

{schema_json}

Rules:
- gameweek: string, e.g. "Gameweek 9".
- deadline: string exactly as printed, e.g. "Sat 31 Oct, 18:00".
- budget_itb: float in millions, e.g. 1.3 (strip the pound sign and the "m").
- free_transfers and cost_pts: integers.
- wildcard_available / free_hit_available: true when the chip still shows "Play".
- goalkeepers (2), defenders (5), midfielders (5), forwards (3): pitch rows top to bottom.
- position: exactly one of GKP, DEF, MID, FWD.
- price: float in millions.
- fixture: the fixture text on the card, e.g. "BHA (H)".
- status_flag: "RED" for a red exclamation mark, "YELLOW" for a yellow warning
  triangle, otherwise "NONE".

Output raw JSON only. No prose, no markdown, no code fences."""


def encode_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def extract_fpl_data_from_image(
    image: Image.Image,
    ollama_host: str = "http://localhost:11434",
    model: str = DEFAULT_MODEL,
) -> FPLTeamState:
    client = Client(host=ollama_host)
    schema = FPLTeamState.model_json_schema()
    schema_json = json.dumps(schema, ensure_ascii=False)
    image_b64 = encode_image_to_base64(image)

    response = client.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": build_prompt(schema_json),
                "images": [image_b64],
            }
        ],
        format=schema,
        options={"temperature": 0.0},
    )
    raw_text = response.message.content or ""

    try:
        return parse_team_state(raw_text)
    except (ValueError, ValidationError):
        pass

    # Pass 2: บางครั้งโมเดลตอบเป็น Markdown ให้ส่งกลับไปแปลงเป็น JSON แบบข้อความล้วน
    repair = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": REPAIR_SYSTEM},
            {
                "role": "user",
                "content": (
                    "Convert these FPL notes into JSON matching this JSON Schema exactly "
                    f"(use its top-level keys, do not nest them):\n{schema_json}\n\nNOTES:\n{raw_text}"
                ),
            },
        ],
        format=schema,
        options={"temperature": 0.0},
    )
    return parse_team_state(repair.message.content or "")