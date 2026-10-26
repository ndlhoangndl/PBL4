param([string]$AiDirectory)
$ErrorActionPreference='Stop'
$smartDoorRoot=Split-Path $PSScriptRoot -Parent
if(!$AiDirectory) { $AiDirectory=Join-Path $smartDoorRoot 'ai\runtime' }
if(!(Test-Path "$AiDirectory\requirements.txt")) { throw 'Thiếu SmartDoor\ai\runtime\requirements.txt. Kiểm tra lại bộ mã AI.' }
python -m venv "$smartDoorRoot\.venv-ai"
if($LASTEXITCODE-ne0) { throw 'Không tạo được venv; cần Python x64 3.11/3.12.' }
& "$smartDoorRoot\.venv-ai\Scripts\python.exe" -m pip install -r "$AiDirectory\requirements.txt"
if($LASTEXITCODE-ne0) { throw 'Cài thư viện Python thất bại.' }
