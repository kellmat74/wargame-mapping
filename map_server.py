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
from region_registry import (
    detect_region_for_coords,
    get_available_regions,
    get_regions_by_continent,
    get_geofabrik_url,
    derive_display_name,
    GEOFABRIK_DIR,
    GEOFABRIK_CONTINENTS,
)

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


@app.route('/api/version')
def get_version():
    """Get the current generator version."""
    # Read VERSION from tactical_map.py without importing it
    # (importing triggers pyproj which breaks subprocess forking)
    try:
        version_file = Path(__file__).parent / 'tactical_map.py'
        with open(version_file) as f:
            for line in f:
                if line.startswith('VERSION'):
                    version = line.split('=')[1].strip().strip('"\'')
                    return jsonify({'version': version})
        return jsonify({'version': 'unknown'})
    except Exception as e:
        return jsonify({'version': 'unknown', 'error': str(e)})


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


# === Game Map Conversion Routes ===

# Store for game map conversion status (separate from tactical map generation)
game_map_status = {
    'running': False,
    'output': [],
    'error': None,
    'complete': False
}
game_map_lock = threading.Lock()
game_map_queue = queue.Queue()


@app.route('/game-map')
def game_map_page():
    """Serve the game map conversion page."""
    return send_file('game_map_config.html')


@app.route('/api/detailed-maps')
def list_detailed_maps():
    """List available detailed maps for conversion."""
    try:
        output_dir = Path(__file__).parent / 'output'
        maps = []

        if output_dir.exists():
            # Scan for map folders with hexdata.json
            for country_dir in output_dir.iterdir():
                if country_dir.is_dir():
                    for map_dir in country_dir.iterdir():
                        if map_dir.is_dir():
                            # Check for versioned folders (timestamp format)
                            for version_dir in map_dir.iterdir():
                                if version_dir.is_dir():
                                    hexdata = list(version_dir.glob('*_hexdata.json'))
                                    svg = list(version_dir.glob('*_tactical.svg'))
                                    if hexdata and svg:
                                        # Load metadata
                                        with open(hexdata[0]) as f:
                                            metadata = json.load(f).get('metadata', {})

                                        maps.append({
                                            'path': str(version_dir),
                                            'country': country_dir.name,
                                            'name': map_dir.name,
                                            'version': version_dir.name,
                                            'render_version': metadata.get('version'),
                                            'center_lat': metadata.get('center_lat'),
                                            'center_lon': metadata.get('center_lon'),
                                            'has_game_map': (version_dir / 'game_map').exists()
                                        })

                            # Also check map_dir directly (for older maps without version folders)
                            hexdata = list(map_dir.glob('*_hexdata.json'))
                            svg = list(map_dir.glob('*_tactical.svg'))
                            if hexdata and svg:
                                with open(hexdata[0]) as f:
                                    metadata = json.load(f).get('metadata', {})

                                maps.append({
                                    'path': str(map_dir),
                                    'country': country_dir.name,
                                    'name': map_dir.name,
                                    'version': None,
                                    'render_version': metadata.get('version'),
                                    'center_lat': metadata.get('center_lat'),
                                    'center_lon': metadata.get('center_lon'),
                                    'has_game_map': (map_dir / 'game_map').exists()
                                })

        return jsonify({'maps': maps})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/convert-game-map', methods=['POST'])
def convert_game_map():
    """Start game map conversion for the specified detailed map."""
    global game_map_status

    try:
        config = request.get_json()
        map_path = config.get('map_path')

        if not map_path:
            return jsonify({'error': 'No map path specified'}), 400

        # Check if already running
        with game_map_lock:
            if game_map_status['running']:
                return jsonify({'error': 'Conversion already in progress'}), 400

            # Reset status
            game_map_status = {
                'running': True,
                'output': [],
                'error': None,
                'complete': False
            }

        # Clear the queue
        while not game_map_queue.empty():
            try:
                game_map_queue.get_nowait()
            except queue.Empty:
                break

        # Start conversion in background thread
        def run_conversion():
            try:
                from game_map_converter import convert_to_game_map

                # Redirect print output to queue
                import io
                import contextlib

                class QueueWriter(io.StringIO):
                    def write(self, s):
                        if s.strip():
                            game_map_queue.put(s.rstrip())
                        return len(s)

                writer = QueueWriter()
                with contextlib.redirect_stdout(writer):
                    convert_config = {
                        'terrain_style': config.get('terrain_style', 'auto'),
                        'elevation_intensity': config.get('elevation_intensity', 1.0),
                        'hillside_intensity': config.get('hillside_intensity', 1.0),
                        'hillside_color': config.get('hillside_color'),
                        'frame_color': config.get('frame_color', '#6B2D2D'),
                        'terrain_colors': config.get('terrain_colors'),
                        'elevation_band_opacities': config.get('elevation_band_opacities'),
                        'label_rows': config.get('label_rows', [1, 5, 10, 15, 20, 25]),
                        'label_color': config.get('label_color', '#5e5959'),
                        'label_opacity': config.get('label_opacity', 0.7),
                    }
                    convert_to_game_map(Path(map_path), convert_config)

                with game_map_lock:
                    game_map_status['running'] = False
                    game_map_status['complete'] = True
                game_map_queue.put(None)  # Signal completion

            except Exception as e:
                with game_map_lock:
                    game_map_status['running'] = False
                    game_map_status['error'] = str(e)
                    game_map_status['complete'] = True
                game_map_queue.put(None)

        thread = threading.Thread(target=run_conversion)
        thread.daemon = True
        thread.start()

        return jsonify({'success': True, 'message': 'Conversion started'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/game-map-progress')
def game_map_progress_stream():
    """Stream game map conversion progress via Server-Sent Events."""
    def generate():
        while True:
            try:
                line = game_map_queue.get(timeout=30)
                if line is None:
                    # Conversion complete
                    with game_map_lock:
                        if game_map_status['error']:
                            yield f"data: {json.dumps({'type': 'error', 'message': game_map_status['error']})}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'complete', 'message': 'Game map conversion complete!'})}\n\n"
                    break
                else:
                    yield f"data: {json.dumps({'type': 'output', 'message': line})}\n\n"
            except queue.Empty:
                # Send keepalive
                yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"

                # Check if conversion is still running
                with game_map_lock:
                    if not game_map_status['running'] and game_map_status['complete']:
                        break

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/game-map-status')
def get_game_map_status():
    """Get current game map conversion status."""
    with game_map_lock:
        return jsonify({
            'running': game_map_status['running'],
            'complete': game_map_status['complete'],
            'error': game_map_status['error'],
            'output_lines': len(game_map_status['output'])
        })


@app.route('/api/rerender-map', methods=['POST'])
def rerender_map():
    """Re-render a detailed map with the current version of the rendering engine.

    Loads config from map_config.json or reconstructs from hexdata.json,
    replaces the existing folder with a new render.
    """
    global generation_status
    import shutil
    import re
    import mgrs

    # Read VERSION from tactical_map.py without importing it
    # (importing triggers pyproj which breaks subprocess forking)
    version_str = "v1.0.0"  # fallback
    try:
        version_file = Path(__file__).parent / 'tactical_map.py'
        with open(version_file) as f:
            for line in f:
                if line.startswith('VERSION'):
                    # Parse: VERSION = "v1.0.0"
                    version_str = line.split('=')[1].strip().strip('"\'')
                    break
    except:
        pass

    with status_lock:
        if generation_status['running']:
            return jsonify({'error': 'Generation already in progress'}), 409

    try:
        data = request.get_json()
        map_path = data.get('map_path')

        if not map_path:
            return jsonify({'error': 'No map path specified'}), 400

        map_dir = Path(map_path)
        if not map_dir.exists():
            return jsonify({'error': f'Map directory not found: {map_path}'}), 404

        # Try to load config from map_config.json first
        config_file = map_dir / 'map_config.json'
        hexdata_files = list(map_dir.glob('*_hexdata.json'))

        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
        elif hexdata_files:
            # Reconstruct config from hexdata.json metadata
            with open(hexdata_files[0]) as f:
                hexdata = json.load(f)
                metadata = hexdata.get('metadata', {})

            config = {
                'name': metadata.get('name', map_dir.parent.name),
                'center_lat': metadata.get('center_lat'),
                'center_lon': metadata.get('center_lon'),
                'region': metadata.get('region'),
                'country': metadata.get('country', ''),
                'rotation_deg': metadata.get('rotation_deg', 0),
            }

            if not config['center_lat'] or not config['center_lon']:
                return jsonify({'error': 'Cannot reconstruct config: missing coordinates in hexdata'}), 400
        else:
            return jsonify({'error': 'No map_config.json or hexdata.json found'}), 404

        # Compute MGRS region from coordinates if missing
        if not config.get('region') and config.get('center_lat') and config.get('center_lon'):
            m = mgrs.MGRS()
            mgrs_str = m.toMGRS(config['center_lat'], config['center_lon'], MGRSPrecision=0)
            # mgrs_str format: "36UUA" -> "36U/UA"
            zone = mgrs_str[:3] if mgrs_str[2].isalpha() else mgrs_str[:2]
            square = mgrs_str[len(zone):len(zone)+2]
            config['region'] = f"{zone}/{square}"

        # Extract base timestamp from folder name (strip version suffix if present)
        folder_name = map_dir.name
        # Match: YYYY-MM-DD_HH-MM or YYYY-MM-DD_HH-MM_vX.Y.Z
        match = re.match(r'^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})(?:_v[\d.]+)?$', folder_name)
        if match:
            base_timestamp = match.group(1)
        else:
            # Folder doesn't match expected pattern, use current timestamp
            from datetime import datetime
            base_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

        # Create new timestamp with current version
        new_timestamp = f"{base_timestamp}_{version_str}"
        config['timestamp'] = new_timestamp

        # Determine new folder path
        parent_dir = map_dir.parent
        new_folder = parent_dir / new_timestamp

        # If re-rendering to same version, delete old folder first
        if new_folder == map_dir:
            # Same folder - clear contents
            for item in map_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        elif new_folder.exists():
            # Different version folder already exists - clear it
            shutil.rmtree(new_folder)

        # If folder name is changing (version upgrade), we'll let the old folder remain
        # until new generation succeeds, then can clean up manually

        # Save config for the generator
        main_config_path = Path(__file__).parent / 'map_config.json'
        with open(main_config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Get region from config
        region = config.get('region', '')

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

        # Add info about re-render
        output_queue.put(f"Re-rendering map with {version_str}")
        output_queue.put(f"Source: {map_path}")
        output_queue.put(f"New timestamp: {new_timestamp}")

        # Track old folder for cleanup after successful generation
        # But DON'T delete if old folder is a parent/ancestor of where new files will go
        # (this happens when old map had no timestamp subfolder)
        old_folder_to_delete = None
        expected_new_output = Path(__file__).parent / 'output' / config.get('country', '') / config.get('name', '') / new_timestamp

        if new_folder != map_dir and map_dir.exists():
            # Check if the old folder is a parent of where new output will go
            try:
                expected_new_output.relative_to(map_dir)
                # If we get here, map_dir is a parent of expected output - DON'T delete it
                output_queue.put(f"Note: Old folder '{folder_name}' contains new output location, will not be deleted")
            except ValueError:
                # map_dir is not a parent, safe to delete
                old_folder_to_delete = map_dir
                output_queue.put(f"Note: Old folder '{folder_name}' will be deleted after successful generation")

        # Start generation in background thread with cleanup callback
        def run_generation_with_cleanup():
            run_generation(region)
            # Check if generation succeeded before deleting old folder
            with status_lock:
                if generation_status['complete'] and not generation_status['error']:
                    if old_folder_to_delete and old_folder_to_delete.exists():
                        try:
                            shutil.rmtree(old_folder_to_delete)
                            output_queue.put(f"Cleaned up old folder: {old_folder_to_delete.name}")
                        except Exception as e:
                            output_queue.put(f"Warning: Could not delete old folder: {e}")

        thread = threading.Thread(target=run_generation_with_cleanup)
        thread.daemon = True
        thread.start()

        return jsonify({
            'success': True,
            'message': 'Re-render started',
            'new_timestamp': new_timestamp
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/shutdown', methods=['POST'])
def shutdown():
    """Shut down the server and any running subprocesses."""
    global generation_status

    def stop_server():
        import time
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


@app.route('/api/restart', methods=['POST'])
def restart_server():
    """Restart the server to pick up code changes."""
    def do_restart():
        import time
        time.sleep(0.5)  # Give time for response to be sent
        # Spawn new server process (with --no-browser to avoid opening new tab)
        subprocess.Popen(
            [sys.executable, __file__, '--no-browser'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            start_new_session=True
        )
        time.sleep(0.3)  # Give new server time to start
        os._exit(0)  # Exit current process

    threading.Thread(target=do_restart, daemon=True).start()
    return jsonify({'success': True, 'message': 'Server restarting...'})


# === Geofabrik Data Download Routes ===

# Store for Geofabrik download status
geofabrik_status = {
    'running': False,
    'output': [],
    'error': None,
    'complete': False,
    'region': None,
}
geofabrik_lock = threading.Lock()
geofabrik_queue = queue.Queue()


@app.route('/data-download')
def data_download_page():
    """Serve the data download page."""
    return send_file('data_download_config.html')


@app.route('/api/geofabrik/regions')
def get_geofabrik_regions():
    """Get available Geofabrik regions grouped by continent with subregions.

    Returns hierarchical structure:
    {
        "continent": {
            "display_name": "Continent Name",
            "countries": {
                "country": {
                    "display_name": "Country Name",
                    "subregions": [{"name": "...", "display_name": "..."}]
                }
            }
        }
    }
    """
    try:
        return jsonify(get_regions_by_continent())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/geofabrik/downloaded')
def get_downloaded_regions():
    """Get list of downloaded Geofabrik PBF files with metadata."""
    try:
        downloaded = []

        if GEOFABRIK_DIR.exists():
            for pbf_file in GEOFABRIK_DIR.glob('*-latest.osm.pbf'):
                region_name = pbf_file.stem.replace('-latest.osm', '')
                stat = pbf_file.stat()

                # Get continent from registry
                continent = GEOFABRIK_CONTINENTS.get(region_name, 'unknown')

                downloaded.append({
                    'name': region_name,
                    'display_name': derive_display_name(region_name),
                    'continent': continent,
                    'size_bytes': stat.st_size,
                    'size_mb': round(stat.st_size / (1024 * 1024), 1),
                    'modified': stat.st_mtime,
                    'path': str(pbf_file),
                })

        # Sort by display name
        downloaded.sort(key=lambda x: x['display_name'])

        return jsonify({'downloaded': downloaded})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/geofabrik/download', methods=['POST'])
def start_geofabrik_download():
    """Start downloading a Geofabrik region PBF file.

    Request body:
        region: Region/subregion name (required)
        continent: Continent name (optional, for subregions)
        country: Country name (optional, for subregions)

    Examples:
        {"region": "ukraine"} - Download country
        {"region": "california", "continent": "north-america", "country": "us"} - Download subregion
    """
    global geofabrik_status

    try:
        data = request.get_json()
        region = data.get('region')
        continent = data.get('continent')  # Optional for subregions
        country = data.get('country')      # Optional for subregions

        if not region:
            return jsonify({'error': 'No region specified'}), 400

        # Build display name for status
        display_name = region
        if country:
            display_name = f"{country}/{region}"

        # Check if already running
        with geofabrik_lock:
            if geofabrik_status['running']:
                return jsonify({'error': 'Download already in progress'}), 409

            # Reset status
            geofabrik_status = {
                'running': True,
                'output': [],
                'error': None,
                'complete': False,
                'region': region,
            }

        # Clear the queue
        while not geofabrik_queue.empty():
            try:
                geofabrik_queue.get_nowait()
            except queue.Empty:
                break

        # Start download in background thread
        def run_download():
            try:
                import urllib.request
                import time as time_module

                url = get_geofabrik_url(region, continent, country)
                GEOFABRIK_DIR.mkdir(parents=True, exist_ok=True)
                output_path = GEOFABRIK_DIR / f"{region}-latest.osm.pbf"

                geofabrik_queue.put(f"Downloading {display_name} from Geofabrik...")
                geofabrik_queue.put(f"URL: {url}")

                # Throttled progress reporting to prevent UI freezing
                last_update_time = [0]  # Use list to allow modification in nested function
                last_percent = [-1]

                def report_progress(block_num, block_size, total_size):
                    if total_size <= 0:
                        return

                    now = time_module.time()
                    percent = int(min(100, block_num * block_size * 100 / total_size))

                    # Only update if: 1 second passed OR percent changed by 5+
                    if now - last_update_time[0] >= 1.0 or percent >= last_percent[0] + 5:
                        last_update_time[0] = now
                        last_percent[0] = percent
                        size_mb = block_num * block_size / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        geofabrik_queue.put(f"Progress: {percent}% ({size_mb:.1f} / {total_mb:.1f} MB)")

                # Use urlretrieve for progress reporting
                urllib.request.urlretrieve(url, output_path, reporthook=report_progress)

                # Verify file exists and has content
                if output_path.exists() and output_path.stat().st_size > 0:
                    size_mb = output_path.stat().st_size / (1024 * 1024)
                    geofabrik_queue.put(f"Download complete: {size_mb:.1f} MB")

                    # Update region registry
                    geofabrik_queue.put("Updating region registry...")
                    get_available_regions(force_rescan=True)
                    geofabrik_queue.put("Registry updated")
                else:
                    raise Exception("Download failed - file is empty or missing")

                with geofabrik_lock:
                    geofabrik_status['running'] = False
                    geofabrik_status['complete'] = True
                geofabrik_queue.put(None)  # Signal completion

            except Exception as e:
                with geofabrik_lock:
                    geofabrik_status['running'] = False
                    geofabrik_status['error'] = str(e)
                    geofabrik_status['complete'] = True
                geofabrik_queue.put(None)

        thread = threading.Thread(target=run_download)
        thread.daemon = True
        thread.start()

        return jsonify({'success': True, 'message': f'Download started for {region}'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/geofabrik/progress')
def geofabrik_progress_stream():
    """Stream Geofabrik download progress via Server-Sent Events."""
    def generate():
        while True:
            try:
                line = geofabrik_queue.get(timeout=30)
                if line is None:
                    # Download complete
                    with geofabrik_lock:
                        if geofabrik_status['error']:
                            yield f"data: {json.dumps({'type': 'error', 'message': geofabrik_status['error']})}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'complete', 'message': 'Download complete!'})}\n\n"
                    break
                else:
                    yield f"data: {json.dumps({'type': 'output', 'message': line})}\n\n"
            except queue.Empty:
                # Send keepalive
                yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"

                # Check if download is still running
                with geofabrik_lock:
                    if not geofabrik_status['running'] and geofabrik_status['complete']:
                        break

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/geofabrik/status')
def get_geofabrik_status():
    """Get current Geofabrik download status."""
    with geofabrik_lock:
        return jsonify({
            'running': geofabrik_status['running'],
            'complete': geofabrik_status['complete'],
            'error': geofabrik_status['error'],
            'region': geofabrik_status['region'],
        })


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

    # Open browser automatically (optional - skip if restarting)
    if '--no-browser' not in sys.argv:
        import webbrowser
        webbrowser.open(f'http://localhost:{PORT}')

    app.run(debug=False, port=PORT, threaded=True)
