@echo off
REM Tactical Map Generator - Setup Script (Windows)
REM This script creates a virtual environment and installs all dependencies

echo ========================================
echo Tactical Map Generator - Setup
echo ========================================
echo.

REM Check for Python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python is required but not found.
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

python --version

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo.
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
pip install --upgrade pip

REM Install dependencies
echo.
echo Installing dependencies (this may take a few minutes)...
pip install numpy pandas geopandas rasterio shapely pyproj svgwrite mgrs requests pillow flask

REM Create directory structure
echo.
echo Creating directory structure...
if not exist "data" mkdir data
if not exist "output" mkdir output

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To use the tool:
echo.
echo 1. Activate the environment:
echo    venv\Scripts\activate.bat
echo.
echo 2. Open map_config.html in your browser to configure a map
echo.
echo 3. Download data for a region (e.g., Philippines 51P TT):
echo    python download_mgrs_data.py 51P TT
echo.
echo 4. Generate the map:
echo    python tactical_map.py
echo.
echo 5. Find your output in: output\{region}\{name}\
echo.
pause
