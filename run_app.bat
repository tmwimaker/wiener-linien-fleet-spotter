@echo off
setlocal
cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
    rem torch<2.6 (siehe requirements.txt) hat nur Wheels fuer Python 3.10-3.12.
    python -c "import sys; raise SystemExit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)"
    if errorlevel 1 (
        echo [Fehler] Diese Python-Version wird nicht unterstuetzt - bitte Python 3.10, 3.11 oder 3.12 verwenden.
        echo          ^(torch^<2.6 hat fuer Python 3.13+ kein passendes Wheel.^)
        python --version
        pause
        exit /b 1
    )
    echo Erstelle virtuelle Umgebung in .\venv ...
    python -m venv venv
    echo Installiere Abhaengigkeiten aus requirements.txt ...
    venv\Scripts\python.exe -m pip install --upgrade pip
    venv\Scripts\python.exe -m pip install -r requirements.txt
)

venv\Scripts\python.exe -m streamlit run app\app.py
