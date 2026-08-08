@echo off

setlocal EnableDelayedExpansion

cd /d "%~dp0.."



if not exist ".venv\Scripts\activate.bat" (

    echo.

    echo  Virtual environment not found at .venv

    echo  From sports-ev-system run:

    echo    python -m venv .venv

    echo    .venv\Scripts\activate

    echo    pip install -e .

    echo.

    pause

    exit /b 1

)



call .venv\Scripts\activate.bat



call "%~dp0ensure-docker-postgres.bat"

if errorlevel 1 (

    echo.

    echo  Docker/Postgres not ready. Open Docker Desktop if needed, then retry.

    echo.

    pause

    exit /b 1

)



echo.

echo Launching SportsPredictor EV Control Room...

echo Close the browser tab to stop the server, or press Ctrl+C here.

echo.

python -u -m sports_ev.dashboard



set "DASH_EXIT=%ERRORLEVEL%"

if "%DASH_EXIT%"=="0" (

    echo.

    echo SportsPredictor stopped. This window will close.

    timeout /t 2 /nobreak >nul

    exit /b 0

)



echo.

echo Dashboard exited with an error ^(code %DASH_EXIT%^).

pause

exit /b %DASH_EXIT%

