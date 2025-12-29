#!/bin/bash
# Tactical Map Generator - Launcher (Mac)
# Double-click this file to start the map generator

# Change to the script's directory
cd "$(dirname "$0")"

# Clear screen and show header
clear
echo "========================================"
echo "  Tactical Map Generator"
echo "========================================"
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found."
    echo "Running setup first..."
    echo ""
    ./setup.sh
fi

# Activate virtual environment
source venv/bin/activate

# Start the server (browser opens automatically)
echo "Starting server..."
echo "Browser will open to http://localhost:8080"
echo ""
echo "Use the 'Stop Server' button in the browser when done,"
echo "or close this window to stop the server."
echo ""
echo "========================================"
echo ""

python map_server.py

# If we get here, server was stopped
echo ""
echo "Server stopped. You can close this window."
