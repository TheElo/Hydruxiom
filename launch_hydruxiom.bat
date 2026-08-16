@echo off
setlocal
cd /d "%~dp0"

echo Launching Hydruxiom - 3D Tag Space Explorer...

REM Create venv on first run
if not exist ".venv\Scripts\python.exe" (
    echo First run: creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Could not create venv. Is Python on PATH?
        pause
        exit /b 1
    )
)

REM Install deps if requirements changed
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe -m pip install --quiet --disable-pip-version-check -r requirements.txt
)

.venv\Scripts\python.exe main.py

REM Close automatically on normal exit; keep the window open only if the app
REM crashed, so the traceback stays readable.
if errorlevel 1 (
    echo.
    echo Hydruxiom exited with an error (code %errorlevel%). See output above.
    pause
)
