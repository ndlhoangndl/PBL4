@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
 echo Please run setup.cmd first.
 pause
 exit /b 1
)
.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
pause
