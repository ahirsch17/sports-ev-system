@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0.."

if not exist "logs" mkdir logs
set "LOG=logs\scheduled-refresh.log"

echo. >> "%LOG%"
echo ============================================================ >> "%LOG%"
echo [%date% %time%] scheduled-refresh start >> "%LOG%"

call "%~dp0ensure-docker-postgres.bat" >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [%date% %time%] ABORT: Docker/Postgres not ready >> "%LOG%"
    exit /b 1
)

call .venv\Scripts\activate.bat

echo [%date% %time%] refresh-cycle... >> "%LOG%"
python -m sports_ev.cli refresh-cycle --all-sports --days 3 --book draftkings --sync-probables --capture-closing --score-lookback-days 14 >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"

echo [%date% %time%] paper-report... >> "%LOG%"
python -m sports_ev.cli paper-report --days 7 >> "%LOG%" 2>&1

echo [%date% %time%] scheduled-refresh done exit=%RC% >> "%LOG%"
exit /b %RC%
