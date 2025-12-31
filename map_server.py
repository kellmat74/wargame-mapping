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
import signal
import subprocess
import sys
import threading
import queue
from pathlib import Path
from flask import Flask, send_file, request, jsonify, Response

# Import region registry for auto-discovery of available PBF files
from region_registry import detect_region_for_coords, get_available_regions

app = Flask(__name__)


def detect_geofabrik_region(lat: float, lon: float) -> str:
    """Detect which Geofabrik region contains the given coordinates.

    Uses auto-discovered regions from cached PBF files.
    Returns the region name or None if not found.
    """
    return detect_region_for_coords(lat, lon)

# Store for generation status
generation_status = {
    'running': False,
    'output': [],
    'error': None,
    'complete': False
}
status_lock = threading.Lock()
output_queue = queue.Queue()

# Track active subprocess for clean shutdown
active_process = None
process_lock = threading.Lock()


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


@app.route('/api/defaults', methods=['GET'])
def get_defaults():
    """Get map defaults from map_defaults.json."""
    try:
        defaults_path = Path(__file__).parent / 'map_defaults.json'
        if defaults_path.exists():
            with open(defaults_path) as f:
                return jsonify(json.load(f))
        else:
            # Return empty defaults structure
            return jsonify({
                "grid": {"hex_size_m": 250, "grid_width": 47, "grid_height": 26},
                "contours": {"contour_interval_m": 20, "index_contour_interval_m": 100},
                "print": {"trim_width_in": 34.0, "trim_height_in": 22.0, "bleed_in": 0.125, "data_margin_in": 1.25},
                "play_margins": {"top_in": 0.0, "bottom_in": 0.0, "left_in": 0.0, "right_in": 0.0},
                "mgrs": {"grid_interval_m": 1000}
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/defaults', methods=['POST'])
def save_defaults():
    """Save map defaults to map_defaults.json."""
    try:
        defaults = request.get_json()
        if not defaults:
            return jsonify({'error': 'No defaults provided'}), 400

        defaults_path = Path(__file__).parent / 'map_defaults.json'
        with open(defaults_path, 'w') as f:
            json.dump(defaults, f, indent=2)

        return jsonify({'success': True, 'message': 'Defaults saved. Restart may be needed for changes to take effect.'})
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
    required_files = ['roads.geojson', 'buildings.geojson', 'landcover.geojson']

    if not data_path.exists():
        return False, f"Data directory not found: {data_path}"

    missing = [f for f in required_files if not (data_path / f).exists()]
    if missing:
        return False, f"Missing files: {', '.join(missing)}"

    return True, "Data exists"


def run_subprocess(cmd, description):
    """Run a subprocess and stream output to the queue."""
    global active_process

    output_queue.put(f">>> {description}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(Path(__file__).parent)
    )

    # Track this process for clean shutdown
    with process_lock:
        active_process = process

    try:
        for line in iter(process.stdout.readline, ''):
            line = line.rstrip()
            if line:
                with status_lock:
                    generation_status['output'].append(line)
                output_queue.put(line)

        process.wait()
        return process.returncode
    finally:
        with process_lock:
            active_process = None


def run_generation(region):
    """Download data if needed, then run tactical_map.py."""
    global generation_status

    try:
        base_path = Path(__file__).parent

        # Check if data exists, download if not
        data_exists, msg = check_data_exists(region)
        if not data_exists:
            output_queue.put(f"Data not found for region '{region}' - downloading...")

            # Read config to get coordinates for auto-detecting Geofabrik region
            config_path = base_path / 'map_config.json'
            geofabrik_region = None
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    lat = config.get('center_lat')
                    lon = config.get('center_lon')
                    if lat and lon:
                        geofabrik_region = detect_geofabrik_region(lat, lon)
                        if geofabrik_region:
                            output_queue.put(f"Auto-detected Geofabrik region: {geofabrik_region}")

            if not geofabrik_region:
                output_queue.put("WARNING: Could not auto-detect Geofabrik region from coordinates")
                output_queue.put("Falling back to Overpass API download (may be incomplete)")
                # Fall back to old script
                download_script = base_path / 'download_mgrs_data.py'
            else:
                # Use new Osmium-based script
                download_script = base_path / 'download_mgrs_data_osmium.py'

            # Convert region format: "51P/TT" -> "51P TT"
            region_arg = region.replace('/', ' ')

            # Build command based on which script we're using
            if geofabrik_region:
                cmd = [sys.executable, str(download_script), '--region', geofabrik_region, region_arg]
            else:
                cmd = [sys.executable, str(download_script), region_arg]

            returncode = run_subprocess(
                cmd,
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
        lat = config.get('center_lat')
        lon = config.get('center_lon')

        if not region:
            return jsonify({'exists': False, 'message': 'No region specified'})

        # Auto-detect Geofabrik region from coordinates
        geofabrik_region = None
        if lat and lon:
            geofabrik_region = detect_geofabrik_region(lat, lon)

        # Parse region (e.g., "51P/TT" or "51P TT")
        region_clean = region.replace(' ', '/').replace('//', '/')
        data_path = Path(__file__).parent / 'data' / region_clean

        # Check for required files
        required_files = ['roads.geojson', 'buildings.geojson', 'landcover.geojson']
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
                'path': str(data_path),
                'geofabrik_region': geofabrik_region
            })
        else:
            # Build download command based on detected region
            region_arg = region.replace('/', ' ')
            if geofabrik_region:
                download_cmd = f'python download_mgrs_data_osmium.py --region {geofabrik_region} "{region_arg}"'
            else:
                download_cmd = f'python download_mgrs_data.py "{region_arg}"'

            return jsonify({
                'exists': False,
                'message': f'Missing data for {region}: {", ".join(missing)}',
                'missing': missing,
                'geofabrik_region': geofabrik_region,
                'download_command': download_cmd
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/shutdown', methods=['POST'])
def shutdown():
    """Shut down the server and any running subprocesses."""
    global generation_status

    def stop_server():
        # Give time for response to be sent
        time.sleep(0.5)

        # Kill any active subprocess first
        with process_lock:
            if active_process is not None:
                try:
                    active_process.terminate()
                    active_process.wait(timeout=2)
                except:
                    try:
                        active_process.kill()
                    except:
                        pass

        # Update status
        with status_lock:
            generation_status['running'] = False
            generation_status['complete'] = True
            generation_status['error'] = 'Server shutdown requested'

        # Kill the server
        os.kill(os.getpid(), signal.SIGTERM)

    thread = threading.Thread(target=stop_server)
    thread.daemon = True
    thread.start()

    return jsonify({'success': True, 'message': 'Server shutting down...'})


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
