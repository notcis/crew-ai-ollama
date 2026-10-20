# คู่มือการติดตั้งและใช้งาน (Installation & Setup Guide)

---

## 1. ติดตั้ง Python 3.12 (Prerequisites)

เปิด **PowerShell** แล้วติดตั้ง Python 3.12 ผ่าน winget:

```powershell
winget install Python.Python.3.12
```

---

## 2. สร้างและเปิดใช้งาน Virtual Environment

### 2.1 สร้าง Virtual Environment ด้วย Python 3.12
```powershell
py -3.12 -m venv .venv-crewai
```

### 2.2 เปิดใช้งาน (Activate) Virtual Environment
- **บน PowerShell:**
  ```powershell
  .\.venv-crewai\Scripts\Activate.ps1
  ```
- **บน Command Prompt (CMD):**
  ```cmd
  .venv-crewai\Scripts\activate.bat
  ```
- **บน Git Bash / WSL:**
  ```bash
  source .venv-crewai/Scripts/activate
  ```

---

## 3. ติดตั้ง Dependencies

ติดตั้งแพ็กเกจที่จำเป็นจาก `requirements.txt`:

```powershell
pip install -r requirements.txt
```

---

## 4. เริ่มการทำงานของระบบ (Run Application)

### 4.1 เปิดบริการ Ollama
เปิด terminal เพื่อรัน Ollama Server:

```powershell
ollama serve
```

### 4.2 รันแอปพลิเคชัน Streamlit
เปิด terminal อีกหน้าต่าง (ที่ activate virtual environment แล้ว) และรัน:

```powershell
streamlit run app.py
```

---

## 5. การออกจาก Virtual Environment

เมื่อใช้งานเสร็จเรียบร้อย ให้ใช้คำสั่ง:

```powershell
deactivate
```

---

## 💡 Troubleshooting / การแก้ปัญหา

### กรณีพบ Error กับ CrewAI v1.15+ (เรียก Ollama ผ่าน OpenAI-compatible)
ตั้งแต่ CrewAI 1.15 เป็นต้นไป มีการเปลี่ยนผ่านการเรียก Ollama เป็นแบบ OpenAI-compatible ซึ่งอาจติดปัญหาเรื่อง Context Length ให้ทำตามขั้นตอนดังนี้:

1. **ปิด Ollama เดิมก่อน:**
   - ปิดจาก System Tray หรือปิด Process ของ Ollama ที่ทำงานอยู่
2. **เปิด PowerShell ใหม่ และกำหนดค่า Context Length:**
   ```powershell
   $env:OLLAMA_CONTEXT_LENGTH="16384"
   ```
3. **เริ่ม Ollama ใหม่อีกครั้ง:**
   ```powershell
   ollama serve
   ```
