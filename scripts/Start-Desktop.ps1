$ErrorActionPreference='Stop'
. "$PSScriptRoot\Find-DotNet.ps1"
$pbl4Root=Split-Path $PSScriptRoot -Parent
& $pbl4Dotnet build "$pbl4Root\SmartDoor.sln" -m:1
if($LASTEXITCODE-ne0) { throw 'Build thất bại.' }
& $pbl4Dotnet "$pbl4Root\src\SmartDoor.Desktop\bin\Debug\net8.0-windows\SmartDoor.Desktop.dll"
