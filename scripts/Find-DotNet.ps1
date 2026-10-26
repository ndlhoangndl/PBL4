$pbl4DotnetCommand=Get-Command dotnet -ErrorAction SilentlyContinue
if($pbl4DotnetCommand)
{
    $pbl4Dotnet=$pbl4DotnetCommand.Source
}
elseif(Test-Path "$env:USERPROFILE\.dotnet\dotnet.exe")
{
    $pbl4Dotnet="$env:USERPROFILE\.dotnet\dotnet.exe"
}
else
{
    throw 'Cần .NET 8 SDK x64 và Desktop Runtime. Xem README.'
}
