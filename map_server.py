#!/usr/bin/env python3
"""
Tactical Map Generator - Web Server

A simple Flask server that:
1. Serves the map configuration page
2. Accepts config via POST and runs the map generator
3. Streams progress back to the browser

Usage:
    python map_server.py

Then open http://localhost:5000 in your browser.
"""

import json
import os
import subprocess
import sys
import threading
import queue
from pathlib import Path
from flask import Flask, send_file, request, jsonify, Response

app = Flask(__name__)

# Store for generation status
generation_status = {
    'running': False,
    'output': [],
    'error': None,
    'complete': False
}
status_lock = threading.Lock()
output_queue = queue.Queue()


@app.route('/')
def index():
    """Serve the map configuration page."""
    return send_file('map_config.html')


@app.route('/api/config', methods=['POST'])
def save_config():
    """Save configuration to map_config.json."""
    try:
        config = request.get_json()
        if not config:
            return jsonify({'error': 'No configuration provided'}), 400

        config_path = Path(__file__).parent / 'map_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        return jsonify({'success': True, 'message': 'Configuration saved'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate', methods=['POST'])
def generate_map():
    """Start map generation."""
    global generation_status

    with status_lock:
        if generation_status['running']:
            return jsonify({'error': 'Generation already in progress'}), 409

    # Save config first
    try:
        config = request.get_json()
        if config:
            config_path = Path(__file__).parent / 'map_config.json'
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
    except Exception as e:
        return jsonify({'error': f'Failed to save config: {e}'}), 500

    # Get region from config
    region = config.get('region', '') if config else ''

    # Reset status
    with status_lock:
        generation_status = {
            'running': True,
            'output': [],
            'error': None,
            'complete': False
        }

    # Clear the queue
    while not output_queue.empty():
        try:
            output_queue.get_nowait()
        except queue.Empty:
            break

    # Start generation in background thread (with region for auto-download)
    thread = threading.Thread(target=run_generation, args=(region,))
    thread.daemon = True
    thread.start()

    return jsonify({'success': True, 'message': 'Generation started'})


def check_data_exists(region):
    """Check if required MGRS data exists for a region."""
    if not region:
        return False, "No region specified"

    # Parse region (e.g., "51P/TT" or "51P TT")
    region_clean = region.replace(' ', '/').replace('//', '/')
    data_path = Path(__file__).parent / 'data' / region_clean

    # Check for required files
    required_files = ['roads.geojson', 'buildings.geojson', 'landuse.geojson']

    if not data_path.exists():
        return False, f"Data directory not found: {data_path}"

    missing = [f for f in required_files if not (data_path / f).exists()]
    if missing:
        return False, f"Missing files: {', '.join(missing)}"

    return True, "Data exists"


def run_subprocess(cmd, description):
    """Run a subprocess and stream output to the queue."""
    output_queue.put(f">>> {description}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(Path(__file__).parent)
    )

    for line in iter(process.stdout.readline, ''):
        line = line.rstrip()
        if line:
            with status_lock:
                generation_status['output'].append(line)
            output_queue.put(line)

    process.wait()
    return process.returncode


def run_generation(region):
    """Download data if needed, then run tactical_map.py."""
    global generation_status

    try:
        base_path = Path(__file__).parent

        # Check if data exists, download if not
        data_exists, msg = check_data_exists(region)
        if not data_exists:
            output_queue.put(f"Data not found for region '{region}' - downloading...")

            # Run download script
            download_script = base_path / 'download_mgrs_data.py'
            # Convert region format: "51P/TT" -> "51P TT"
            region_arg = region.replace('/', ' ')

            returncode = run_subprocess(
                [sys.executable, str(download_script), region_arg],
                f"Downloading data for {region_arg}..."
            )

            if returncode != 0:
                with status_lock:
                    generation_status['running'] = False
                    generation_status['error'] = f'Data download failed with code {returncode}'
                    generation_status['complete'] = True
                output_queue.put(None)
                return

            output_queue.put("Download complete. Starting map generation...")
        else:
            output_queue.put(f"Data found for region '{region}'")

        # Run map generation
        script_path = base_path / 'tactical_map.py'
        returncode = run_subprocess(
            [sys.executable, str(script_path)],
            "Generating tactical map..."
        )

        with status_lock:
            generation_status['running'] = False
            generation_status['complete'] = True
            if returncode != 0:
                generation_status['error'] = f'Map generation failed with code {returncode}'
            output_queue.put(None)  # Signal completion

    except Exception as e:
        with status_lock:
            generation_status['running'] = False
            generation_status['error'] = str(e)
            generation_status['complete'] = True
        output_queue.put(None)


@app.route('/api/progress')
def progress_stream():
    """Stream generation progress via Server-Sent Events."""
    def generate():
        while True:
            try:
                line = output_queue.get(timeout=30)
                if line is None:
                    # Generation complete
                    with status_lock:
                        if generation_status['error']:
                            yield f"data: {json.dumps({'type': 'error', 'message': generation_status['error']})}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'complete', 'message': 'Map generation complete!'})}\n\n"
                    break
                else:
                    yield f"data: {json.dumps({'type': 'output', 'message': line})}\n\n"
            except queue.Empty:
                # Send keepalive
                yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"

                # Check if generation is still running
                with status_lock:
                    if not generation_status['running'] and generation_status['complete']:
                        break

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/status')
def get_status():
    """Get current generation status."""
    with status_lock:
        return jsonify({
            'running': generation_status['running'],
            'complete': generation_status['complete'],
            'error': generation_status['error'],
            'output_lines': len(generation_status['output'])
        })


@app.route('/api/check-data', methods=['POST'])
def check_data():
    """Check if required MGRS data exists."""
    try:
        config = request.get_json()
        region = config.get('region', '')

        if not region:
            return jsonify({'exists': False, 'message': 'No region specified'})

        # Parse region (e.g., "51P/TT" or "51P TT")
        region_clean = region.replace(' ', '/').replace('//', '/')
        data_path = Path(__file__).parent / 'data' / region_clean

        # Check for required files
        required_files = ['roads.geojson', 'buildings.geojson', 'landuse.geojson']
        existing = []
        missing = []

        for f in required_files:
            if (data_path / f).exists():
                existing.append(f)
            else:
                missing.append(f)

        if not missing:
            return jsonify({
                'exists': True,
                'message': f'Data found for {region}',
                'path': str(data_path)
            })
        else:
            return jsonify({
                'exists': False,
                'message': f'Missing data for {region}: {", ".join(missing)}',
                'missing': missing,
                'download_command': f'python download_mgrs_data.py "{region}"'
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    PORT = 8080  # Using 8080 to avoid conflict with AirPlay on macOS

    print("=" * 50)
    print("Tactical Map Generator - Web Server")
    print("=" * 50)
    print()
    print(f"Open your browser to: http://localhost:{PORT}")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 50)

    # Open browser automatically (optional)
    import webbrowser
    webbrowser.open(f'http://localhost:{PORT}')

    app.run(debug=False, port=PORT, threaded=True)
