# Registers a Windows Task Scheduler job to run paper refresh every 8 hours.
# Run from PowerShell:  .\scripts\install-scheduled-task.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BatPath = Join-Path $ProjectRoot "scripts\scheduled-refresh.bat"

if (-not (Test-Path $BatPath)) {
    Write-Error "Not found: $BatPath"
}

$TaskName = "SportsPredictor-Refresh"
$Description = "Poll odds, log +EV paper picks, sync scores, settle bets (Sports EV system)"

# Remove existing task so re-run is idempotent
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

$Action = New-ScheduledTaskAction -Execute $BatPath -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).Date.AddMinutes(5) `
    -RepetitionInterval (New-TimeSpan -Hours 8) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description $Description | Out-Null

Write-Host ""
Write-Host "Scheduled task installed: $TaskName"
Write-Host "  Runs every 8 hours: $BatPath"
Write-Host "  Log file: $ProjectRoot\logs\scheduled-refresh.log"
Write-Host ""
Write-Host "To run once now:"
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host ""
Write-Host "To remove:"
Write-Host "  Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"
Write-Host ""
