@echo off
REM Launch the ALCHEMIST control panel (GUI). Run AS ADMINISTRATOR.
cd /d "%~dp0"
python main.py --gui
echo.
echo --- gui closed ---
pause
