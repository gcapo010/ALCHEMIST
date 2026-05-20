@echo off
REM Diagnostics mode - prints what the bot detects but never clicks.
REM Useful for verifying calibration. Run AS ADMINISTRATOR.
cd /d "%~dp0"
python main.py --probe --debug
echo.
echo --- probe ended ---
pause
