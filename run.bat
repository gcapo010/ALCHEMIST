@echo off
REM Launch the ALCHEMIST bot and keep the console open on errors.
REM Run this AS ADMINISTRATOR (right-click -> Run as administrator).
cd /d "%~dp0"
python main.py %*
echo.
echo --- bot has exited ---
pause
