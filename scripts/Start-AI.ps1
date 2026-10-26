param([string]$AiDirectory,[int]$Port=8000)
$ErrorActionPreference='Stop'
$pbl4Root=Split-Path $PSScriptRoot -Parent
$pbl4Python="$pbl4Root\.venv-ai\Scripts\python.exe"
if(!(Test-Path $pbl4Python)) { throw 'Chạy scripts\Setup-AI.ps1 trước.' }
$pbl4Args=@("$PSScriptRoot\start_ai.py",'--port',"$Port")
if($AiDirectory) { $pbl4Args+=@('--ai-dir',$AiDirectory) }
& $pbl4Python @pbl4Args
if($LASTEXITCODE-ne0) { throw 'AI khởi động thất bại; xem lỗi ở trên.' }
