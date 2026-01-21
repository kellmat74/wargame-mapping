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

# Check if port 8080 is already in use
PORT_PID=$(lsof -ti :8080 2>/dev/null)
if [ -n "$PORT_PID" ]; then
    echo "Port 8080 is already in use (PID: $PORT_PID)"
    echo ""
    echo "This usually means the server is already running."
    echo ""
    read -p "Kill existing process and restart? (y/n): " choice
    if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
        echo "Stopping existing server..."
        kill -9 $PORT_PID 2>/dev/null
        sleep 1
        echo ""
    else
        echo ""
        echo "Opening browser to existing server..."
        open "http://localhost:8080"
        echo ""
        echo "Server is already running at http://localhost:8080"
        exit 0
    fi
fi

# Start the server (browser opens automatically)
echo "Starting server..."
echo "Browser will open to http://localhost:8080"
echo ""
echo "Use the 'Restart Server' button in the browser if needed,"
echo "or close this window to stop the server."
echo ""
echo "========================================"
echo ""

python map_server.py
EXIT_CODE=$?

# Show appropriate message based on exit
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "Server stopped. You can close this window."
else
    echo "Server exited with an error (code: $EXIT_CODE)."
    echo "Check the output above for details."
fi
