param([string]$Subject = 'person-001')
$ErrorActionPreference='Stop'
$smartDoorRoot=Split-Path $PSScriptRoot -Parent
$capturePython=Join-Path $smartDoorRoot '.venv-ai\Scripts\python.exe'
if(!(Test-Path -LiteralPath $capturePython)) { throw 'Chạy scripts\Setup-AI.ps1 trước để cài OpenCV.' }
$env:PYTHONUTF8='1'
& $capturePython -B (Join-Path $PSScriptRoot 'collect_faces.py') --subject $Subject
if($LASTEXITCODE -ne 0) { throw 'Công cụ thu ảnh gặp lỗi; xem thông báo ở trên.' }
