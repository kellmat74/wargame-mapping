#!/bin/bash
# Tactical Map Generator - Setup Script (Mac/Linux)
# This script creates a virtual environment and installs all dependencies

set -e

echo "========================================"
echo "Tactical Map Generator - Setup"
echo "========================================"
echo ""

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is required but not found."
    echo "Please install Python 3.9+ from https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies (this may take a few minutes)..."
pip install numpy pandas geopandas rasterio shapely pyproj svgwrite mgrs requests pillow

# Create directory structure
echo ""
echo "Creating directory structure..."
mkdir -p data
mkdir -p output

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To use the tool:"
echo ""
echo "1. Activate the environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Open map_config.html in your browser to configure a map"
echo ""
echo "3. Download data for a region (e.g., Philippines 51P TT):"
echo "   python download_mgrs_data.py 51P TT"
echo ""
echo "4. Generate the map:"
echo "   python tactical_map.py"
echo ""
echo "5. Find your output in: output/{region}/{name}/"
echo ""
