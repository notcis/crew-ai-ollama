from pydantic import BaseModel, Field
from typing import List, Optional

class PlayerInfo(BaseModel):
    name: str = Field(description="ชื่อนักเตะที่ปรากฏบนการ์ด เช่น Haaland, Calafiori, João Pedro")
    position: str = Field(description="ตำแหน่ง: GKP, DEF, MID, FWD")
    price: float = Field(description="ราคาขายที่แสดงบนการ์ด เช่น 15.6, 7.6, 5.7")
    fixture: str = Field(description="คู่แข่งและสถานะเหย้าเยือน เช่น BHA (H), MCI (A)")
    status_flag: Optional[str] = Field(default="NONE", description="สถานะเตือน: RED (ธงแดงเจ็บ), YELLOW (ธงเหลืองเช็คฟิต), หรือ NONE")

class FPLTeamState(BaseModel):
    gameweek: str = Field(description="Gameweek ปัจจุบัน เช่น Gameweek 9")
    deadline: str = Field(description="เส้นตายการจัดทีม เช่น Sat 31 Oct, 18:00")
    budget_itb: float = Field(description="เงินคงเหลือในธนาคาร (In The Bank) เป็นหน่วยล้านปอนด์ เช่น 1.3")
    free_transfers: int = Field(description="จำนวน Free Transfers ที่มี เช่น 1")
    cost_pts: int = Field(default=0, description="แต้มที่โดนหักจากการย้ายตัวปัจจุบัน เช่น 0")
    wildcard_available: bool = Field(description="ปุ่ม Wildcard พร้อมใช้งานหรือไม่")
    free_hit_available: bool = Field(description="ปุ่ม Free Hit พร้อมใช้งานหรือไม่")
    goalkeepers: List[PlayerInfo] = Field(description="ผู้รักษาประตู 2 คน")
    defenders: List[PlayerInfo] = Field(description="กองหลัง 5 คน")
    midfielders: List[PlayerInfo] = Field(description="กองกลาง 5 คน")
    forwards: List[PlayerInfo] = Field(description="กองหน้า 3 คน")