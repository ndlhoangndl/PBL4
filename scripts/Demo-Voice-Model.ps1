param([string]$AudioFile,[string]$CheckpointDirectory)
$ErrorActionPreference='Stop'
$smartDoorRoot=Split-Path $PSScriptRoot -Parent
$trainingRoot=Join-Path $smartDoorRoot 'ai\training'
$demoPython=Join-Path $trainingRoot '.venv-training\Scripts\python.exe'
if(!$CheckpointDirectory){$CheckpointDirectory=Join-Path $trainingRoot 'runs\whisper-tiny-otp-pilot-20261019-v1\final-model'}
if(!(Test-Path -LiteralPath (Join-Path $CheckpointDirectory 'model.safetensors'))){throw 'Fine-tuned model not found; see ai/training/WHISPER_PILOT_DEMO.md.'}
if(!$AudioFile){
    Add-Type -AssemblyName System.Windows.Forms
    $audioDialog=New-Object System.Windows.Forms.OpenFileDialog
    $audioDialog.Title='Choose a WAV recording for the fine-tuned model'
    $audioDialog.Filter='WAV audio (*.wav)|*.wav'
    $audioDialog.InitialDirectory=Join-Path $smartDoorRoot 'ai\datasets\audio'
    try { if($audioDialog.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK){exit 0};$AudioFile=$audioDialog.FileName }
    finally {$audioDialog.Dispose()}
}
$AudioFile=(Resolve-Path -LiteralPath $AudioFile).Path
$CheckpointDirectory=(Resolve-Path -LiteralPath $CheckpointDirectory).Path
$env:PYTHONUTF8='1'
Push-Location $trainingRoot
try { & $demoPython -B -m pbl4_training.predict_whisper_pilot --checkpoint $CheckpointDirectory --audio $AudioFile; if($LASTEXITCODE -ne 0){throw 'Voice demo failed.'} }
finally {Pop-Location}
