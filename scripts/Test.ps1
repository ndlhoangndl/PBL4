param([switch]$SqlIntegration)
$ErrorActionPreference='Stop'
. "$PSScriptRoot\Find-DotNet.ps1"
$pbl4Root=Split-Path $PSScriptRoot -Parent
if($SqlIntegration-and!$env:PBL4_TEST_SQL_BASE)
{
    $env:PBL4_TEST_SQL_BASE='Server=.\SQLEXPRESS;Integrated Security=true;TrustServerCertificate=true;Connect Timeout=3'
}
& $pbl4Dotnet test "$pbl4Root\tests\SmartDoor.Tests\SmartDoor.Tests.csproj" -m:1 --logger 'trx;LogFileName=tests.trx' --results-directory "$pbl4Root\TestResults"
if($LASTEXITCODE-ne0) { throw 'Kiểm thử thất bại.' }
