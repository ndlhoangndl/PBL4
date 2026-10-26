param([string]$SessionDirectory)
$ErrorActionPreference='Stop'
$smartDoorRoot=Split-Path $PSScriptRoot -Parent
$capturePython=Join-Path $smartDoorRoot '.venv-ai\Scripts\python.exe'
$reviewPython=Join-Path $smartDoorRoot 'ai\training\.venv-training\Scripts\python.exe'
if(!(Test-Path -LiteralPath $capturePython) -or !(Test-Path -LiteralPath $reviewPython)) { throw 'Missing inference or training venv; see ai/training/FACE_CAPTURE.md.' }
if(!$SessionDirectory) {
    $manifest=Get-ChildItem (Join-Path $smartDoorRoot 'ai\datasets\face-capture') -Filter captures.jsonl -Recurse | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if(!$manifest) { throw 'No captured session found.' }
    $SessionDirectory=$manifest.DirectoryName
}
$SessionDirectory=(Resolve-Path -LiteralPath $SessionDirectory).Path
$env:PYTHONUTF8='1'
& $capturePython -B (Join-Path $PSScriptRoot 'review_faces.py') --session $SessionDirectory
if($LASTEXITCODE -ne 0) { throw 'Capture audit failed.' }
Push-Location (Join-Path $smartDoorRoot 'ai\training')
try { & $reviewPython -B -m pbl4_training.review_face_regions --session $SessionDirectory; if($LASTEXITCODE -ne 0) { throw 'Region editor failed.' } }
finally { Pop-Location }
