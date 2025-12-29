"""
download_region_data.py - Download map data for a region

Downloads:
- Reference tiles (OpenTopoMap) for artist reference layer
- Enhanced OSM features via Overpass API
"""

import json
import math
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
import time

# Tile settings
TILE_ZOOM = 15  # Good detail level (~4m/pixel)
TILE_SIZE = 256
TILE_URL_TEMPLATE = "https://tile.opentopomap.org/{z}/{x}/{y}.png"
# Alternative: ESRI World Topo
# TILE_URL_TEMPLATE = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}"

# Overpass API
OVERPASS_URL = "https://lz4.overpass-api.de/api/interpreter"


def lat_lon_to_tile(lat, lon, zoom):
    """Convert lat/lon to tile coordinates."""
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(lat_rad)) / math.pi) / 2 * n)
    return x, y


def tile_to_lat_lon(x, y, zoom):
    """Convert tile coordinates to lat/lon (NW corner of tile)."""
    n = 2 ** zoom
    lon = x / n * 360 - 180
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def download_reference_tiles(center_lat, center_lon, width_km, height_km, output_path, zoom=TILE_ZOOM):
    """
    Download and stitch tiles covering the map area.

    Args:
        center_lat, center_lon: Map center coordinates
        width_km, height_km: Map dimensions in kilometers
        output_path: Path to save stitched image
        zoom: Tile zoom level
    """
    print(f"\nDownloading reference tiles at zoom {zoom}...")

    # Calculate bounds with buffer
    # Roughly: 1 degree lat = 111km, 1 degree lon = 111km * cos(lat)
    lat_buffer = (height_km / 2 + 1) / 111
    lon_buffer = (width_km / 2 + 1) / (111 * math.cos(math.radians(center_lat)))

    min_lat = center_lat - lat_buffer
    max_lat = center_lat + lat_buffer
    min_lon = center_lon - lon_buffer
    max_lon = center_lon + lon_buffer

    # Get tile range
    min_x, max_y = lat_lon_to_tile(min_lat, min_lon, zoom)
    max_x, min_y = lat_lon_to_tile(max_lat, max_lon, zoom)

    # Ensure proper ordering
    if min_x > max_x:
        min_x, max_x = max_x, min_x
    if min_y > max_y:
        min_y, max_y = max_y, min_y

    num_tiles_x = max_x - min_x + 1
    num_tiles_y = max_y - min_y + 1
    total_tiles = num_tiles_x * num_tiles_y

    print(f"  Tile range: X={min_x}-{max_x}, Y={min_y}-{max_y}")
    print(f"  Total tiles: {total_tiles} ({num_tiles_x} x {num_tiles_y})")

    # Create stitched image
    stitched_width = num_tiles_x * TILE_SIZE
    stitched_height = num_tiles_y * TILE_SIZE
    stitched = Image.new('RGB', (stitched_width, stitched_height))

    # Download tiles
    downloaded = 0
    failed = 0
    headers = {'User-Agent': 'TacticalMapGenerator/1.0 (wargame map creation tool)'}

    for ty in range(min_y, max_y + 1):
        for tx in range(min_x, max_x + 1):
            url = TILE_URL_TEMPLATE.format(z=zoom, x=tx, y=ty)

            try:
                response = requests.get(url, headers=headers, timeout=30)
                if response.status_code == 200:
                    tile_img = Image.open(BytesIO(response.content))
                    # Calculate position in stitched image
                    pos_x = (tx - min_x) * TILE_SIZE
                    pos_y = (ty - min_y) * TILE_SIZE
                    stitched.paste(tile_img, (pos_x, pos_y))
                    downloaded += 1
                else:
                    failed += 1
                    print(f"    Failed: {url} ({response.status_code})")
            except Exception as e:
                failed += 1
                print(f"    Error: {url} ({e})")

            # Rate limiting - be nice to tile servers
            time.sleep(0.1)

            if (downloaded + failed) % 10 == 0:
                print(f"  Progress: {downloaded + failed}/{total_tiles} tiles")

    print(f"  Downloaded: {downloaded}, Failed: {failed}")

    # Save stitched image
    stitched.save(output_path, 'PNG', optimize=True)
    print(f"  Saved to: {output_path}")

    # Return georeferencing info
    nw_lat, nw_lon = tile_to_lat_lon(min_x, min_y, zoom)
    se_lat, se_lon = tile_to_lat_lon(max_x + 1, max_y + 1, zoom)

    return {
        'path': str(output_path),
        'width': stitched_width,
        'height': stitched_height,
        'bounds': {
            'north': nw_lat,
            'south': se_lat,
            'east': se_lon,
            'west': nw_lon,
        },
        'zoom': zoom,
    }


def download_enhanced_osm_features(center_lat, center_lon, width_km, height_km, data_dir):
    """
    Download detailed OSM features via Overpass API.

    Downloads separate GeoJSON files for:
    - streams (waterway=stream, ditch, drain, canal)
    - paths (highway=footway, path, cycleway, track)
    - barriers (barrier=fence, wall, hedge, etc.)
    - powerlines (power=line, minor_line)
    - bridges (bridge=yes on ways)
    - tree_rows (natural=tree_row)
    - detailed_landcover (more landuse/natural tags)
    """
    print(f"\nDownloading enhanced OSM features...")

    # Calculate bounds with buffer
    lat_buffer = (height_km / 2 + 2) / 111
    lon_buffer = (width_km / 2 + 2) / (111 * math.cos(math.radians(center_lat)))

    south = center_lat - lat_buffer
    north = center_lat + lat_buffer
    west = center_lon - lon_buffer
    east = center_lon + lon_buffer

    bbox = f"{south},{west},{north},{east}"
    print(f"  Bounds: {south:.4f},{west:.4f} to {north:.4f},{east:.4f}")

    # Define feature queries
    feature_queries = {
        'streams': '''
            way["waterway"~"stream|ditch|drain|canal"]({bbox});
        ''',
        'paths': '''
            way["highway"~"footway|path|cycleway|bridleway|steps"]({bbox});
        ''',
        'barriers': '''
            way["barrier"~"fence|wall|hedge|retaining_wall|city_wall"]({bbox});
        ''',
        'powerlines': '''
            way["power"~"line|minor_line|cable"]({bbox});
        ''',
        'bridges': '''
            way["bridge"="yes"]({bbox});
            way["man_made"="bridge"]({bbox});
        ''',
        'tree_rows': '''
            way["natural"="tree_row"]({bbox});
        ''',
        'railways': '''
            way["railway"~"rail|narrow_gauge|light_rail"]({bbox});
        ''',
        'waterways_area': '''
            way["natural"="water"]({bbox});
            relation["natural"="water"]({bbox});
            way["waterway"="riverbank"]({bbox});
            way["landuse"="reservoir"]({bbox});
        ''',
        'farmland': '''
            way["landuse"~"farmland|paddy|vineyard|orchard|meadow"]({bbox});
            relation["landuse"~"farmland|paddy|vineyard|orchard|meadow"]({bbox});
        ''',
        'cliffs': '''
            way["natural"~"cliff|earth_bank"]({bbox});
            way["man_made"~"embankment|cutting"]({bbox});
        ''',
    }

    data_dir = Path(data_dir)
    results = {}

    for feature_name, query_body in feature_queries.items():
        print(f"  Downloading {feature_name}...")

        query = f'''
            [out:json][timeout:180];
            (
                {query_body.format(bbox=bbox)}
            );
            out body;
            >;
            out skel qt;
        '''

        try:
            response = requests.post(OVERPASS_URL, data={'data': query}, timeout=180)

            if response.status_code != 200:
                print(f"    Failed: HTTP {response.status_code}")
                continue

            data = response.json()
            elements = data.get('elements', [])
            print(f"    Received {len(elements)} elements")

            # Convert to GeoJSON
            nodes = {}
            ways = []
            relations = []

            for el in elements:
                if el['type'] == 'node':
                    nodes[el['id']] = (el['lon'], el['lat'])
                elif el['type'] == 'way':
                    ways.append(el)
                elif el['type'] == 'relation':
                    relations.append(el)

            features = []

            # Process ways
            for way in ways:
                coords = []
                for node_id in way.get('nodes', []):
                    if node_id in nodes:
                        coords.append(nodes[node_id])

                if len(coords) >= 2:
                    # Determine geometry type
                    tags = way.get('tags', {})
                    is_area = (
                        tags.get('area') == 'yes' or
                        way['nodes'][0] == way['nodes'][-1] and len(coords) >= 4 and
                        any(k in tags for k in ['landuse', 'natural', 'waterway']) and
                        tags.get('waterway') not in ['stream', 'ditch', 'drain', 'canal']
                    )

                    if is_area and len(coords) >= 4:
                        geom = {
                            "type": "Polygon",
                            "coordinates": [coords]
                        }
                    else:
                        geom = {
                            "type": "LineString",
                            "coordinates": coords
                        }

                    features.append({
                        "type": "Feature",
                        "properties": tags,
                        "geometry": geom
                    })

            # Save GeoJSON
            geojson = {
                "type": "FeatureCollection",
                "features": features
            }

            output_path = data_dir / f"{feature_name}.geojson"
            with open(output_path, 'w') as f:
                json.dump(geojson, f)

            results[feature_name] = {
                'path': str(output_path),
                'count': len(features)
            }
            print(f"    Saved {len(features)} features to {output_path}")

            # Rate limiting
            time.sleep(1)

        except Exception as e:
            print(f"    Error: {e}")

    return results


def main():
    """Download data for the current map config."""

    # Load config
    config_path = Path("map_config.json")
    if not config_path.exists():
        print("No map_config.json found!")
        return

    with open(config_path) as f:
        config = json.load(f)

    print(f"Downloading data for: {config['name']}")
    print(f"Center: {config['center_lat']:.4f}, {config['center_lon']:.4f}")

    # Map dimensions (from tactical_map.py constants)
    width_km = 10.2
    height_km = 6.5

    # Data directory
    data_dir = Path("data") / config['region']
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download reference tiles
    tiles_path = data_dir / f"reference_tiles_{config['name']}.png"
    tile_info = download_reference_tiles(
        config['center_lat'],
        config['center_lon'],
        width_km,
        height_km,
        tiles_path
    )

    # Save tile georeferencing info
    tile_info_path = data_dir / f"reference_tiles_{config['name']}.json"
    with open(tile_info_path, 'w') as f:
        json.dump(tile_info, f, indent=2)
    print(f"  Tile info saved to: {tile_info_path}")

    # Download enhanced OSM features
    osm_results = download_enhanced_osm_features(
        config['center_lat'],
        config['center_lon'],
        width_km,
        height_km,
        data_dir
    )

    print("\n" + "="*60)
    print("Download complete!")
    print("="*60)
    print(f"\nReference tiles: {tiles_path}")
    print(f"\nOSM features downloaded:")
    for name, info in osm_results.items():
        print(f"  {name}: {info['count']} features")


if __name__ == "__main__":
    main()
