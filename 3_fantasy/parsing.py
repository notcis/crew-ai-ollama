"""ตัวช่วยแปลงคำตอบของ LLM ให้เป็น FPLTeamState ที่ถูกต้อง

deepseek-v4.1-flash:cloud บน Ollama มักไม่ทำตาม `format=` JSON Schema แล้วตอบเป็น
ข้อความ Markdown แทน ทำให้ `FPLTeamState.model_validate_json()` ล้มด้วย
"Invalid JSON: expected value at line 1 column 1" โมดูลนี้จึงแกะ JSON ออกมาจาก
ข้อความและปรับชนิดข้อมูลให้ตรงกับ schema ก่อน validate เสมอ
"""

import json
import re

from pydantic import ValidationError
from schemas import FPLTeamState

ROWS = {"goalkeepers": "GKP", "defenders": "DEF", "midfielders": "MID", "forwards": "FWD"}
POS_ALIASES = {
    "GK": "GKP", "GKP": "GKP", "GOALKEEPER": "GKP",
    "DEF": "DEF", "DEFENDER": "DEF",
    "MID": "MID", "MIDFIELDER": "MID",
    "FWD": "FWD", "FORWARD": "FWD", "ST": "FWD", "ATT": "FWD",
}
FLAG_KEYS = ("status_flag", "flag", "status", "warning", "injury_status", "alert")


def strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def first_json_object(text: str) -> dict:
    """ดึง JSON object ก้อนแรกออกมา ตัด prose หรือโค้ดบล็อกที่โมเดลห่อมาให้"""
    text = strip_code_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        raise ValueError("model output contains no JSON object")

    depth, in_string, escaped = 0, False, False
    for i, char in enumerate(text[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("model output has an unterminated JSON object")


def to_number(value, default=0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if match:
            return float(match.group())
    return default


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "yes", "y", "available", "play", "1"}


def to_flag(value) -> str:
    text = str(value or "").upper()
    if "RED" in text or "INJUR" in text:
        return "RED"
    if "YELLOW" in text or "DOUBT" in text or "WARN" in text or "FIT" in text:
        return "YELLOW"
    return "NONE"


def to_gameweek(value) -> str:
    match = re.search(r"\d+", str(value or ""))
    return f"Gameweek {match.group()}" if match else str(value or "").strip()


def flatten_state(raw) -> dict:
    """รองรับกรณีโมเดลห่อข้อมูลเป็น {header: ..., players: {...}} แทน schema แบบแบน"""
    if not isinstance(raw, dict):
        raise ValueError("model output is not a JSON object")
    if "header" not in raw and "players" not in raw:
        return raw

    flat = dict(raw.get("header") or {})
    players = raw.get("players") or {}
    for key, code in ROWS.items():
        if key in players:
            flat.setdefault(key, players[key])
        if code in players:
            flat.setdefault(key, players[code])

    for key in ("wildcard_available", "free_hit_available"):
        chip = flat.get(key)
        if isinstance(chip, dict):
            flat[key] = chip.get("status") or chip.get("available")
        if flat.get(key) is None:
            alias = key.replace("_available", "")
            if alias in flat:
                chip = flat[alias]
                flat[key] = (chip.get("status") or chip.get("available")) if isinstance(chip, dict) else chip

    for alias in ("bank", "budget", "itb", "budget_in_the_bank"):
        if "budget_itb" not in flat and alias in flat:
            flat["budget_itb"] = flat[alias]
    if "gw" in flat and "gameweek" not in flat:
        flat["gameweek"] = flat["gw"]

    return flat


def normalize_state(raw) -> dict:
    """ปรับชนิดข้อมูลให้ตรงกับ FPLTeamState เช่น "£1.3m" -> 1.3 และ "GW9" -> "Gameweek 9" """
    flat = flatten_state(raw)
    state = {
        "gameweek": to_gameweek(flat.get("gameweek")),
        "deadline": str(flat.get("deadline") or "").strip(),
        "budget_itb": round(to_number(flat.get("budget_itb")), 1),
        "free_transfers": int(to_number(flat.get("free_transfers"))),
        "cost_pts": int(to_number(flat.get("cost_pts"))),
        "wildcard_available": to_bool(flat.get("wildcard_available")),
        "free_hit_available": to_bool(flat.get("free_hit_available")),
    }

    for key, default_position in ROWS.items():
        players = flat.get(key) or []
        if isinstance(players, dict):
            players = list(players.values())
        state[key] = [
            {
                "name": str(player.get("name") or "").strip(),
                "position": POS_ALIASES.get(
                    str(player.get("position") or "").strip().upper(), default_position
                ),
                "price": round(to_number(player.get("price")), 1),
                "fixture": str(player.get("fixture") or "").strip(),
                "status_flag": to_flag(
                    next((player[k] for k in FLAG_KEYS if player.get(k) is not None), None)
                ),
            }
            for player in players
            if isinstance(player, dict)
        ]

    return state


def parse_team_state(text: str) -> FPLTeamState:
    return FPLTeamState.model_validate(normalize_state(first_json_object(text)))


def team_json_for_prompt(state: FPLTeamState) -> str:
    return json.dumps(state.model_dump(), indent=2, ensure_ascii=False)


__all__ = [
    "first_json_object",
    "flatten_state",
    "normalize_state",
    "parse_team_state",
    "team_json_for_prompt",
    "ValidationError",
]
