# Creates a desktop shortcut that launches the SportsPredictor dashboard.
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BatPath = Join-Path $ProjectRoot "scripts\launch-dashboard.bat"
$Desktop = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $Desktop "SportsPredictor.lnk"

if (-not (Test-Path $BatPath)) {
    Write-Error "Launcher not found: $BatPath"
    exit 1
}

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $BatPath
$Shortcut.WorkingDirectory = $ProjectRoot
$Shortcut.Description = "Launch SportsPredictor paper betting dashboard"
$Shortcut.Save()

Write-Host ""
Write-Host "Desktop shortcut created:"
Write-Host "  $ShortcutPath"
Write-Host ""
Write-Host "Double-click SportsPredictor on your desktop to open the betting UI."
