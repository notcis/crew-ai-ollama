## install
### Open PowerShell and install Python 3.12 
winget install Python.Python.3.12

### Using the Windows Python Launcher to target 3.12:
py -3.12 -m venv .venv-crewai

### Activate the virtual environment
source .venv-crewai/Scripts/activate

---

### or use the requirements file
pip install -r requirements.txt

---

### Deactivate the virtual environment when done
deactivate

---

ollama serve

streamlit run app.py

---

# ถ้า error crewai เวอร์ชัน 1.15 ไม่ได้ใช้ litellm แล้ว มันเรียก Ollama ผ่านทาง OpenAI-compatible
## ให้ทำตามนี้
### ปิด Ollama เดิมก่อน (ปิดจากถาด system tray หรือหยุด process ollama)
### เปิด PowerShell ใหม่แล้วตั้งค่า context length ใหม่ แล้วใส่คำสั่งต่อไปนี้
- $env:OLLAMA_CONTEXT_LENGTH="16384"
### เรียก Ollama ใหม่ด้วย
- ollama serve