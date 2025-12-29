@echo off
REM Tactical Map Generator - Launcher (Windows)
REM Double-click this file to start the map generator

title Tactical Map Generator

echo ========================================
echo   Tactical Map Generator
echo ========================================
echo.

REM Check if venv exists
if not exist "venv" (
    echo Virtual environment not found.
    echo Running setup first...
    echo.
    call setup.bat
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Start the server (browser opens automatically)
echo Starting server...
echo Browser will open to http://localhost:8080
echo.
echo Use the 'Stop Server' button in the browser when done,
echo or close this window to stop the server.
echo.
echo ========================================
echo.

python map_server.py

REM If we get here, server was stopped
echo.
echo Server stopped. You can close this window.
pause
