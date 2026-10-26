param()
$ErrorActionPreference='Stop'
$trainingRoot=Join-Path (Split-Path $PSScriptRoot -Parent) 'ai\training'
$voicePython=Join-Path $trainingRoot '.venv-training\Scripts\python.exe'
if(!(Test-Path -LiteralPath $voicePython)) { throw 'Missing .venv-training. See ai/training/VOICE_CAPTURE.md.' }
$env:PYTHONUTF8='1'
Push-Location $trainingRoot
try { & $voicePython -B -m pbl4_training.collect_voice; if($LASTEXITCODE -ne 0) { throw 'Voice collection failed.' } }
finally { Pop-Location }
