@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Collect-Faces.ps1"
if errorlevel 1 pause
