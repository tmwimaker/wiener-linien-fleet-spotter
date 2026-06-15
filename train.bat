@echo off
setlocal
cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
    echo [Fehler] Keine virtuelle Umgebung gefunden - bitte zuerst run_app.bat ausfuehren.
    pause
    exit /b 1
)

echo ============================================
echo   Wiener Linien Fleet Spotter - Training
echo ============================================
echo.
echo   [1] Schnelltest  - yolov8n,  5 Epochen   (schneller Funktionscheck, CPU ok)
echo   [2] Standard     - yolov8n, 50 Epochen   (CPU / wenig VRAM)
echo   [3] Intensiv     - yolov8s, 100 Epochen, Batch 32  (GPU empfohlen)
echo.
set /p mode=Modus waehlen (1-3):

if "%mode%"=="1" goto quick
if "%mode%"=="2" goto standard
if "%mode%"=="3" goto intensive

echo.
echo Ungueltige Auswahl: %mode%
pause
exit /b 1

:quick
set MODEL=yolov8n.pt
set EPOCHS=5
set BATCH=16
set NAME=fleet_spotter_quick
goto run

:standard
set MODEL=yolov8n.pt
set EPOCHS=50
set BATCH=16
set NAME=fleet_spotter_standard
goto run

:intensive
set MODEL=yolov8s.pt
set EPOCHS=100
set BATCH=32
set NAME=fleet_spotter_intensive
goto run

:run
echo.
echo Starte Training: %MODEL%, %EPOCHS% Epochen, Batch %BATCH% ...
echo.
venv\Scripts\python.exe scripts\train.py --model %MODEL% --epochs %EPOCHS% --batch %BATCH% --name %NAME%

pause
