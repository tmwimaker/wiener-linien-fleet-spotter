@echo off
cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
    echo Erstelle virtuelle Umgebung in .\venv ...
    python -m venv venv
    echo Installiere Abhaengigkeiten aus requirements.txt ...
    venv\Scripts\python.exe -m pip install --upgrade pip
    venv\Scripts\python.exe -m pip install -r requirements.txt
)

venv\Scripts\python.exe -m streamlit run app\app.py
