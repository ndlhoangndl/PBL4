@echo off
cd /d "%~dp0"
py -3.11 -m venv .venv
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
.venv\Scripts\python.exe verify_models.py
if errorlevel 1 (
 echo Model files are missing or invalid. Downloading the pinned files...
 .venv\Scripts\python.exe download_models.py
 if errorlevel 1 goto failed
 .venv\Scripts\python.exe verify_models.py
 if errorlevel 1 goto failed
)
echo Setup completed. Run start.cmd then open http://127.0.0.1:8000
pause
exit /b 0
:failed
echo Setup failed. Read the error above. Python 3.11 64-bit is required.
pause
exit /b 1
