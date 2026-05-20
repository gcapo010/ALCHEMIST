@echo off
REM One-time calibration. Run AS ADMINISTRATOR.
cd /d "%~dp0"
python main.py --calibrate
echo.
echo --- calibration finished ---
pause
