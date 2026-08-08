@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0.."

if not exist ".venv\Scripts\activate.bat" (
    echo Virtual environment not found at .venv
    exit /b 1
)

call .venv\Scripts\activate.bat

where docker >nul 2>nul
if errorlevel 1 (
    echo Docker CLI not found. Install Docker Desktop.
    exit /b 1
)

call :ensure_docker_running
if errorlevel 1 exit /b 1

echo Starting Postgres if needed...
docker compose up -d
if errorlevel 1 (
    echo docker compose up failed
    exit /b 1
)

call :wait_for_postgres
if errorlevel 1 (
    echo Postgres not responding on port 5433
    exit /b 1
)

python -m sports_ev.cli migrate-db >nul 2>nul
exit /b 0

:ensure_docker_running
docker info >nul 2>nul
if not errorlevel 1 exit /b 0

echo Docker is not running. Starting Docker Desktop...
set "DOCKER_DESKTOP="
if exist "%ProgramFiles%\Docker\Docker\Docker Desktop.exe" (
    set "DOCKER_DESKTOP=%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
)
if not defined DOCKER_DESKTOP if exist "%LocalAppData%\Programs\Docker\Docker\Docker Desktop.exe" (
    set "DOCKER_DESKTOP=%LocalAppData%\Programs\Docker\Docker\Docker Desktop.exe"
)
if not defined DOCKER_DESKTOP (
    echo Could not find Docker Desktop.exe
    exit /b 1
)
start "" "!DOCKER_DESKTOP!"

set /a DOCKER_TRIES=0
:docker_wait_loop
set /a DOCKER_TRIES+=1
if !DOCKER_TRIES! GTR 40 exit /b 1
docker info >nul 2>nul
if not errorlevel 1 (
    echo Docker is ready.
    exit /b 0
)
if !DOCKER_TRIES!==1 echo Waiting for Docker to start...
timeout /t 3 /nobreak >nul
goto docker_wait_loop

:wait_for_postgres
echo Waiting for Postgres on port 5433...
set /a PG_TRIES=0
:postgres_wait_loop
set /a PG_TRIES+=1
if !PG_TRIES! GTR 30 exit /b 1
python -c "from sports_ev.config import get_settings; from sports_ev.db.session import create_db_engine; from sqlalchemy import text; e=create_db_engine(get_settings()); conn=e.connect(); conn.execute(text('select 1')); conn.close()" >nul 2>nul
if not errorlevel 1 (
    echo Postgres is ready.
    exit /b 0
)
timeout /t 2 /nobreak >nul
goto postgres_wait_loop
