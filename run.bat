@echo off
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 main.py
) else (
    python main.py
)

if errorlevel 1 (
    echo.
    echo Something went wrong - see the error above.
    pause
)
