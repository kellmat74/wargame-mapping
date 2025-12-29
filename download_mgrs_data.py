#!/usr/bin/env python3
"""
download_mgrs_data.py - Download map data for an MGRS 100km square

Downloads elevation (DEM), roads, buildings, and other OSM features
for a specified MGRS 100km square (e.g., "51R TG").

Data is organized by Grid Zone Designator and square:
  data/51R/TG/elevation.tif
  data/51R/TG/roads.geojson
  etc.

Usage:
  python download_mgrs_data.py "51R TG"
  python download_mgrs_data.py 51RTG
"""

import sys
import json
import time
import requests
from pathlib import Path
from typing import Tuple, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import mgrs

# === Configuration ===
DATA_DIR = Path("data")
OVERPASS_URL = "https://lz4.overpass-api.de/api/interpreter"
OPENTOPOGRAPHY_API_KEY = "137877e41e9d540cf80cc3601dc2230a"

# Delay between Overpass API requests (seconds)
API_DELAY = 3  # Increased to be gentler on the API


def parse_mgrs_square(square_str: str) -> Tuple[str, str]:
    """
    Parse MGRS square designator into GZD and 100km square ID.

    Accepts formats like:
      "51R TG", "51RTG", "51R-TG"

    Returns:
      (gzd, square) e.g., ("51R", "TG")
    """
    # Remove spaces, dashes
    cleaned = square_str.replace(" ", "").replace("-", "").upper()

    # MGRS format: 2 digits (zone) + 1 letter (band) + 2 letters (square)
    # e.g., "51RTG"
    if len(cleaned) < 5:
        raise ValueError(f"Invalid MGRS square: {square_str}")

    # Zone is 1-2 digits, band is 1 letter, square is 2 letters
    # Find where the letters start
    zone_end = 0
    for i, c in enumerate(cleaned):
        if c.isalpha():
            zone_end = i
            break

    if zone_end == 0:
        raise ValueError(f"Invalid MGRS square: {square_str}")

    zone = cleaned[:zone_end]
    band = cleaned[zone_end]
    square = cleaned[zone_end + 1:zone_end + 3]

    gzd = f"{zone}{band}"

    if len(square) != 2:
        raise ValueError(f"Invalid MGRS square: {square_str}")

    return (gzd, square)


def get_mgrs_square_bounds(gzd: str, square: str) -> Tuple[float, float, float, float]:
    """
    Get the WGS84 bounding box for an MGRS 100km square.

    Returns:
      (min_lon, min_lat, max_lon, max_lat)
    """
    m = mgrs.MGRS()

    # MGRS precision 0 = 100km square
    # To get bounds, we convert corners with high precision coordinates
    base = f"{gzd}{square}"

    # Sample multiple points within the square to find true bounds
    # (MGRS squares are not perfectly rectangular in lat/lon)
    lats = []
    lons = []

    # Sample a grid of points
    for easting in range(0, 100000, 10000):
        for northing in range(0, 100000, 10000):
            mgrs_str = f"{base}{easting:05d}{northing:05d}"
            try:
                lat, lon = m.toLatLon(mgrs_str)
                lats.append(lat)
                lons.append(lon)
            except:
                pass  # Some corners may be invalid

    # Also sample the corners more precisely
    corners = [
        (0, 0), (99999, 0), (99999, 99999), (0, 99999),
        (50000, 0), (50000, 99999), (0, 50000), (99999, 50000)
    ]
    for e, n in corners:
        mgrs_str = f"{base}{e:05d}{n:05d}"
        try:
            lat, lon = m.toLatLon(mgrs_str)
            lats.append(lat)
            lons.append(lon)
        except:
            pass

    if not lats:
        raise ValueError(f"Could not determine bounds for {gzd} {square}")

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    # Add small buffer to ensure complete coverage
    buffer = 0.01  # ~1km buffer
    min_lat -= buffer
    max_lat += buffer
    min_lon -= buffer
    max_lon += buffer

    return (min_lon, min_lat, max_lon, max_lat)


def download_osm_chunk(
    bounds: Tuple[float, float, float, float],
    query_body: str
) -> List[dict]:
    """Download OSM features for a single chunk."""
    min_lon, min_lat, max_lon, max_lat = bounds
    bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"

    query = f"""
    [out:json][timeout:300][bbox:{bbox}];
    {query_body}
    out body geom;
    """

    response = requests.post(
        OVERPASS_URL,
        data={"data": query},
        timeout=600
    )

    if response.status_code == 429:
        print(f"      Rate limited, waiting 60 seconds...")
        time.sleep(60)
        response = requests.post(
            OVERPASS_URL,
            data={"data": query},
            timeout=600
        )

    if response.status_code != 200:
        raise Exception(f"Error {response.status_code}")

    return response.json().get("elements", [])


def count_osm_features(
    bounds: Tuple[float, float, float, float],
    query_body: str
) -> int:
    """
    Count features in an OSM area using Overpass API count query.

    Args:
        bounds: (min_lon, min_lat, max_lon, max_lat)
        query_body: The Overpass query body (elements to count)

    Returns:
        Total count of features
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"

    # Use [out:json][count] to just get counts without data
    query = f"""[out:json][timeout:300];
{query_body.replace('{{bbox}}', bbox)}
out count;"""

    time.sleep(API_DELAY)

    try:
        response = requests.post(
            OVERPASS_URL,
            data={"data": query},
            timeout=300
        )

        if response.status_code == 429:
            print(f"  Rate limited, waiting 60 seconds...")
            time.sleep(60)
            response = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=300
            )

        if response.status_code != 200:
            return -1  # Unknown count

        data = response.json()
        # Count query returns elements with tags containing counts
        elements = data.get("elements", [])
        if elements:
            # Sum all counts from the query
            total = 0
            for elem in elements:
                tags = elem.get("tags", {})
                total += int(tags.get("total", 0))
            return total
        return 0
    except Exception as e:
        print(f"  Error counting features: {e}")
        return -1


def verify_downloaded_features(
    bounds: Tuple[float, float, float, float],
    output_dir: Path,
    feature_queries: Dict[str, str]
) -> Dict[str, dict]:
    """
    Verify downloaded features by comparing counts against OSM.

    Returns a dict of {filename: {"downloaded": N, "osm_count": N, "status": "ok"|"mismatch"|"unknown"}}
    """
    results = {}

    print("\nVerifying downloaded features against OSM...")

    for filename, query_body in feature_queries.items():
        filepath = output_dir / filename

        # Count downloaded features
        downloaded_count = 0
        if filepath.exists():
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    features = data.get("features", [])
                    downloaded_count = len(features)
            except Exception as e:
                print(f"  Error reading {filename}: {e}")
                downloaded_count = -1

        # Count OSM features
        print(f"  Checking {filename}...", end=" ", flush=True)
        osm_count = count_osm_features(bounds, query_body)

        # Determine status
        if osm_count < 0 or downloaded_count < 0:
            status = "unknown"
            symbol = "?"
        elif downloaded_count >= osm_count:
            status = "ok"
            symbol = "✓"
        else:
            status = "mismatch"
            symbol = "✗"

        results[filename] = {
            "downloaded": downloaded_count,
            "osm_count": osm_count,
            "status": status
        }

        print(f"{symbol} downloaded={downloaded_count}, osm={osm_count}")

    # Summary
    ok_count = sum(1 for r in results.values() if r["status"] == "ok")
    mismatch_count = sum(1 for r in results.values() if r["status"] == "mismatch")
    unknown_count = sum(1 for r in results.values() if r["status"] == "unknown")

    print(f"\nVerification summary: {ok_count} OK, {mismatch_count} mismatches, {unknown_count} unknown")

    if mismatch_count > 0:
        print("\nMismatched files (may need re-download with --force):")
        for filename, info in results.items():
            if info["status"] == "mismatch":
                diff = info["osm_count"] - info["downloaded"]
                print(f"  {filename}: missing ~{diff} features")

    return results


def download_osm_features(
    bounds: Tuple[float, float, float, float],
    output_dir: Path,
    feature_queries: Dict[str, str],
    chunk_size: float = 0.25,  # degrees (~25km chunks to reduce API calls)
    force: bool = False  # Force re-download even if file exists
) -> None:
    """
    Download OSM features using Overpass API, chunking large areas.

    Args:
        bounds: (min_lon, min_lat, max_lon, max_lat)
        output_dir: Directory to save GeoJSON files
        feature_queries: Dict of {filename: overpass_query_body}
        chunk_size: Size of chunks in degrees (default 0.25 = ~25km)
        force: If True, re-download even if file already exists
    """
    min_lon, min_lat, max_lon, max_lat = bounds

    # Calculate chunks needed
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    num_lon_chunks = max(1, int(lon_range / chunk_size) + 1)
    num_lat_chunks = max(1, int(lat_range / chunk_size) + 1)
    total_chunks = num_lon_chunks * num_lat_chunks

    # Generate chunk bounds
    chunks = []
    for i in range(num_lon_chunks):
        for j in range(num_lat_chunks):
            chunk_min_lon = min_lon + i * chunk_size
            chunk_max_lon = min(min_lon + (i + 1) * chunk_size, max_lon)
            chunk_min_lat = min_lat + j * chunk_size
            chunk_max_lat = min(min_lat + (j + 1) * chunk_size, max_lat)
            chunks.append((chunk_min_lon, chunk_min_lat, chunk_max_lon, chunk_max_lat))

    # Number of parallel workers (be respectful to the API)
    MAX_WORKERS = 2  # Reduced from 4 to avoid rate limits
    print_lock = threading.Lock()

    def download_chunk_task(args):
        """Download a single chunk - used by thread pool."""
        idx, chunk_bounds, query_body = args
        try:
            elements = download_osm_chunk(chunk_bounds, query_body)
            return (idx, chunk_bounds, elements, None)
        except Exception as e:
            return (idx, chunk_bounds, None, str(e))

    for filename, query_body in feature_queries.items():
        output_file = output_dir / filename

        if output_file.exists() and not force:
            print(f"  {filename} already exists, skipping")
            continue

        if output_file.exists() and force:
            print(f"  Downloading {filename}... (force re-download)")
        else:
            print(f"  Downloading {filename}...")

        if total_chunks > 1:
            print(f"    Using {total_chunks} chunks with {MAX_WORKERS} parallel workers...")

        all_elements = []
        seen_ids = set()  # Deduplicate elements that span chunks
        failed_chunks = []
        completed = 0

        # Prepare tasks
        tasks = [(idx, chunk_bounds, query_body) for idx, chunk_bounds in enumerate(chunks)]

        # Download chunks in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(download_chunk_task, task): task[0] for task in tasks}

            for future in as_completed(futures):
                idx, chunk_bounds, elements, error = future.result()
                completed += 1

                with print_lock:
                    if error:
                        print(f"      Chunk {idx + 1}/{total_chunks}: error - {error}")
                        failed_chunks.append((idx, chunk_bounds))
                    else:
                        # Deduplicate by element ID
                        new_count = 0
                        for elem in elements:
                            elem_id = (elem.get("type"), elem.get("id"))
                            if elem_id not in seen_ids:
                                seen_ids.add(elem_id)
                                all_elements.append(elem)
                                new_count += 1
                        print(f"      Chunk {idx + 1}/{total_chunks}: {new_count} features ({completed}/{total_chunks} done)")

        # Retry failed chunks sequentially with exponential backoff
        if failed_chunks:
            print(f"    Retrying {len(failed_chunks)} failed chunks...")
            for retry in range(3):  # Up to 3 retries
                if not failed_chunks:
                    break

                wait_time = (retry + 1) * 5  # 5s, 10s, 15s
                print(f"      Retry {retry + 1}/3 (waiting {wait_time}s)...")
                time.sleep(wait_time)

                still_failed = []
                for idx, chunk_bounds in failed_chunks:
                    print(f"        Chunk {idx + 1}...", end=" ", flush=True)
                    try:
                        elements = download_osm_chunk(chunk_bounds, query_body)
                        new_count = 0
                        for elem in elements:
                            elem_id = (elem.get("type"), elem.get("id"))
                            if elem_id not in seen_ids:
                                seen_ids.add(elem_id)
                                all_elements.append(elem)
                                new_count += 1
                        print(f"{new_count} new features")
                        time.sleep(API_DELAY)
                    except Exception as e:
                        print(f"error: {e}")
                        still_failed.append((idx, chunk_bounds))
                        time.sleep(API_DELAY)

                failed_chunks = still_failed

            if failed_chunks:
                print(f"    WARNING: {len(failed_chunks)} chunks failed after retries")

        if all_elements:
            geojson = osm_to_geojson({"elements": all_elements})
            with open(output_file, 'w') as f:
                json.dump(geojson, f)
            print(f"    Saved {len(geojson['features'])} features")
        else:
            print(f"    No features downloaded")


def osm_to_geojson(osm_data: dict) -> dict:
    """Convert Overpass JSON to GeoJSON."""
    features = []

    for element in osm_data.get("elements", []):
        geom = None
        props = element.get("tags", {})

        if element["type"] == "node":
            geom = {
                "type": "Point",
                "coordinates": [element["lon"], element["lat"]]
            }
        elif element["type"] == "way":
            if "geometry" in element:
                coords = [[p["lon"], p["lat"]] for p in element["geometry"]]
                if len(coords) >= 2:
                    # Check if closed polygon
                    if coords[0] == coords[-1] and len(coords) >= 4:
                        geom = {"type": "Polygon", "coordinates": [coords]}
                    else:
                        geom = {"type": "LineString", "coordinates": coords}
        elif element["type"] == "relation":
            # Handle multipolygon relations
            if element.get("tags", {}).get("type") == "multipolygon":
                if "members" in element:
                    outer_rings = []
                    for member in element["members"]:
                        if member.get("role") == "outer" and "geometry" in member:
                            coords = [[p["lon"], p["lat"]] for p in member["geometry"]]
                            if len(coords) >= 4:
                                outer_rings.append(coords)
                    if outer_rings:
                        if len(outer_rings) == 1:
                            geom = {"type": "Polygon", "coordinates": outer_rings}
                        else:
                            geom = {"type": "MultiPolygon", "coordinates": [[r] for r in outer_rings]}

        if geom:
            features.append({
                "type": "Feature",
                "properties": props,
                "geometry": geom
            })

    return {"type": "FeatureCollection", "features": features}


def extract_elevation_from_existing(
    bounds: Tuple[float, float, float, float],
    output_file: Path,
    source_file: Path
) -> bool:
    """
    Extract elevation data from an existing DEM file.

    Returns True if successful, False otherwise.
    """
    try:
        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.transform import from_bounds as transform_from_bounds
        import numpy as np

        min_lon, min_lat, max_lon, max_lat = bounds

        with rasterio.open(source_file) as src:
            # Check if source covers our bounds
            src_bounds = src.bounds
            if (src_bounds.left > min_lon or src_bounds.right < max_lon or
                src_bounds.bottom > min_lat or src_bounds.top < max_lat):
                print(f"    Source DEM doesn't fully cover requested bounds")
                return False

            # Get window for our bounds
            window = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)

            # Read the data
            data = src.read(1, window=window)

            # Calculate new transform
            new_transform = src.window_transform(window)

            # Write output
            profile = src.profile.copy()
            profile.update(
                width=data.shape[1],
                height=data.shape[0],
                transform=new_transform
            )

            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(data, 1)

            print(f"    Extracted {data.shape[1]}x{data.shape[0]} elevation grid")
            return True

    except Exception as e:
        print(f"    Error extracting from existing DEM: {e}")
        return False


def download_elevation(
    bounds: Tuple[float, float, float, float],
    output_dir: Path
) -> None:
    """
    Download elevation data for the bounds.

    First tries to extract from existing taiwan DEM, then falls back to OpenTopography.
    """
    output_file = output_dir / "elevation.tif"

    if output_file.exists():
        print("  elevation.tif already exists, skipping")
        return

    min_lon, min_lat, max_lon, max_lat = bounds

    print("  Preparing elevation data...")
    print(f"    Bounds: {min_lat:.4f},{min_lon:.4f} to {max_lat:.4f},{max_lon:.4f}")

    # First, try to extract from existing Taiwan DEM if available
    taiwan_dem = DATA_DIR / "taiwan" / "elevation.tif"
    if taiwan_dem.exists():
        print("    Trying to extract from existing Taiwan DEM...")
        if extract_elevation_from_existing(bounds, output_file, taiwan_dem):
            return

    # Fall back to OpenTopography API
    print("    Requesting from OpenTopography...")
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": "SRTMGL1",  # SRTM 30m
        "south": min_lat,
        "north": max_lat,
        "west": min_lon,
        "east": max_lon,
        "outputFormat": "GTiff",
        "API_Key": OPENTOPOGRAPHY_API_KEY,
    }

    try:
        response = requests.get(url, params=params, timeout=600)

        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"    Saved elevation.tif ({len(response.content) / 1024 / 1024:.1f} MB)")
        elif response.status_code == 401:
            print(f"    Error {response.status_code}: Invalid or missing API key")
            print("    Update OPENTOPOGRAPHY_API_KEY in download_mgrs_data.py")
        else:
            print(f"    Error {response.status_code}: {response.text[:200]}")

    except Exception as e:
        print(f"    Error downloading elevation: {e}")


def download_reference_tiles(
    bounds: Tuple[float, float, float, float],
    output_dir: Path,
    name: str,
    zoom: int = 14
) -> None:
    """Download and stitch OpenTopoMap tiles for reference."""
    import math
    from io import BytesIO

    try:
        from PIL import Image
    except ImportError:
        print("  PIL not available, skipping reference tiles")
        return

    output_image = output_dir / f"reference_tiles.png"
    output_info = output_dir / f"reference_tiles.json"

    if output_image.exists():
        print("  Reference tiles already exist, skipping")
        return

    min_lon, min_lat, max_lon, max_lat = bounds

    print(f"  Downloading reference tiles (zoom {zoom})...")

    def lat_lon_to_tile(lat, lon, zoom):
        n = 2 ** zoom
        x = int((lon + 180) / 360 * n)
        y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
        return x, y

    def tile_to_lat_lon(x, y, zoom):
        n = 2 ** zoom
        lon = x / n * 360 - 180
        lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
        return lat, lon

    # Get tile range
    x_min, y_min = lat_lon_to_tile(max_lat, min_lon, zoom)
    x_max, y_max = lat_lon_to_tile(min_lat, max_lon, zoom)

    num_tiles = (x_max - x_min + 1) * (y_max - y_min + 1)
    print(f"    Downloading {num_tiles} tiles...")

    if num_tiles > 500:
        print(f"    Too many tiles ({num_tiles}), reducing zoom level")
        return download_reference_tiles(bounds, output_dir, name, zoom - 1)

    # Download tiles
    tiles = {}
    tile_size = 256

    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            url = f"https://tile.opentopomap.org/{zoom}/{x}/{y}.png"
            try:
                response = requests.get(url, timeout=30, headers={
                    'User-Agent': 'WargameMapping/1.0'
                })
                if response.status_code == 200:
                    tiles[(x, y)] = Image.open(BytesIO(response.content))
                time.sleep(0.1)  # Be nice to the tile server
            except Exception as e:
                print(f"      Error fetching tile {x},{y}: {e}")

    if not tiles:
        print("    No tiles downloaded")
        return

    # Stitch tiles
    width = (x_max - x_min + 1) * tile_size
    height = (y_max - y_min + 1) * tile_size
    stitched = Image.new('RGB', (width, height))

    for (x, y), tile in tiles.items():
        px = (x - x_min) * tile_size
        py = (y - y_min) * tile_size
        stitched.paste(tile, (px, py))

    stitched.save(output_image)

    # Save bounds info
    nw_lat, nw_lon = tile_to_lat_lon(x_min, y_min, zoom)
    se_lat, se_lon = tile_to_lat_lon(x_max + 1, y_max + 1, zoom)

    info = {
        "zoom": zoom,
        "width": width,
        "height": height,
        "bounds": {
            "north": nw_lat,
            "south": se_lat,
            "west": nw_lon,
            "east": se_lon
        }
    }
    with open(output_info, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"    Saved {width}x{height} reference image")


def download_mgrs_square(mgrs_square: str, force: bool = False) -> None:
    """Download all data for an MGRS 100km square.

    Args:
        mgrs_square: MGRS square designator (e.g., "51R TG")
        force: If True, re-download all features even if files exist
    """

    # Parse the MGRS square
    gzd, square = parse_mgrs_square(mgrs_square)
    print(f"\n{'='*60}")
    print(f"Downloading data for MGRS square: {gzd} {square}")
    print(f"{'='*60}")

    # Get bounds
    print("\nCalculating bounds...")
    bounds = get_mgrs_square_bounds(gzd, square)
    min_lon, min_lat, max_lon, max_lat = bounds
    print(f"  SW: {min_lat:.4f}°N, {min_lon:.4f}°E")
    print(f"  NE: {max_lat:.4f}°N, {max_lon:.4f}°E")
    print(f"  Size: ~{(max_lat - min_lat) * 111:.0f}km x ~{(max_lon - min_lon) * 111 * 0.9:.0f}km")

    # Create output directory
    output_dir = DATA_DIR / gzd / square
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Save bounds info
    bounds_file = output_dir / "bounds.json"
    with open(bounds_file, 'w') as f:
        json.dump({
            "gzd": gzd,
            "square": square,
            "mgrs": f"{gzd}{square}",
            "bounds": {
                "min_lon": min_lon,
                "min_lat": min_lat,
                "max_lon": max_lon,
                "max_lat": max_lat
            }
        }, f, indent=2)

    # Download elevation
    print("\nDownloading elevation data...")
    download_elevation(bounds, output_dir)

    # Define OSM feature queries
    print("\nDownloading OSM features...")

    feature_queries = {
        # Core features
        "roads.geojson": """(
            way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|track|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|living_street"];
        );""",

        "buildings.geojson": """(
            way["building"];
            relation["building"];
        );""",

        "landcover.geojson": """(
            way["landuse"~"forest|residential|commercial|industrial|farmland|meadow|orchard|vineyard|retail"];
            relation["landuse"~"forest|residential|commercial|industrial|farmland|meadow|orchard|vineyard|retail"];
            way["natural"~"wood|water|wetland|scrub|grassland|beach|sand"];
            relation["natural"~"wood|water|wetland|scrub|grassland|beach|sand"];
        );""",

        # Water features
        "waterways.geojson": """(
            way["waterway"~"river|stream|canal|drain|ditch"];
        );""",

        "waterways_area.geojson": """(
            way["natural"="water"];
            relation["natural"="water"];
            way["waterway"="riverbank"];
            relation["waterway"="riverbank"];
            way["landuse"="reservoir"];
            relation["landuse"="reservoir"];
        );""",

        # Coastline (land/sea boundary)
        "coastline.geojson": """(
            way["natural"="coastline"];
        );""",

        # Infrastructure
        "railways.geojson": """(
            way["railway"~"rail|light_rail|subway|tram"];
        );""",

        "powerlines.geojson": """(
            way["power"="line"];
        );""",

        "bridges.geojson": """(
            way["bridge"="yes"];
            way["man_made"="bridge"];
        );""",

        # Paths and trails
        "paths.geojson": """(
            way["highway"~"footway|path|cycleway|bridleway|steps"];
        );""",

        # Barriers and boundaries
        "barriers.geojson": """(
            way["barrier"~"fence|wall|hedge|retaining_wall"];
        );""",

        # Vegetation
        "tree_rows.geojson": """(
            way["natural"="tree_row"];
        );""",

        # Terrain features
        "cliffs.geojson": """(
            way["natural"~"cliff|earth_bank"];
            way["man_made"="embankment"];
        );""",

        # Agricultural
        "farmland.geojson": """(
            way["landuse"~"farmland|paddy|orchard|vineyard"];
            relation["landuse"~"farmland|paddy|orchard|vineyard"];
        );""",

        # Mangrove and wetlands
        "mangrove.geojson": """(
            way["natural"="mangrove"];
            relation["natural"="mangrove"];
        );""",

        "wetland.geojson": """(
            way["natural"~"marsh|swamp|wetland|reedbed|saltmarsh|mud"];
            relation["natural"~"marsh|swamp|wetland|reedbed|saltmarsh|mud"];
            way["wetland"];
            relation["wetland"];
        );""",

        # Heath and scrubland (beyond what's in landcover)
        "heath.geojson": """(
            way["natural"="heath"];
            relation["natural"="heath"];
        );""",

        # Rocky terrain
        "rocky_terrain.geojson": """(
            way["natural"~"bare_rock|scree|shingle|rock"];
            relation["natural"~"bare_rock|scree|shingle|rock"];
        );""",

        # Sand and dunes
        "sand.geojson": """(
            way["natural"~"sand|dune"];
            relation["natural"~"sand|dune"];
        );""",

        # Military areas
        "military.geojson": """(
            way["landuse"="military"];
            relation["landuse"="military"];
            way["military"];
            relation["military"];
            node["military"];
        );""",

        # Quarries and mines
        "quarries.geojson": """(
            way["landuse"="quarry"];
            relation["landuse"="quarry"];
            node["man_made"="mineshaft"];
            way["man_made"="mineshaft"];
        );""",

        # Cemeteries
        "cemeteries.geojson": """(
            way["landuse"="cemetery"];
            relation["landuse"="cemetery"];
            way["amenity"="grave_yard"];
            relation["amenity"="grave_yard"];
        );""",

        # Places (settlements)
        "places.geojson": """(
            node["place"~"city|town|village|hamlet|isolated_dwelling|locality"];
        );""",

        # Peaks and terrain features
        "peaks.geojson": """(
            node["natural"~"peak|saddle|volcano|hill"];
        );""",

        # Caves
        "caves.geojson": """(
            node["natural"="cave_entrance"];
            way["natural"="cave_entrance"];
        );""",

        # Dams
        "dams.geojson": """(
            way["waterway"="dam"];
            node["waterway"="dam"];
            way["waterway"="weir"];
        );""",

        # Airfields and aviation
        "airfields.geojson": """(
            way["aeroway"~"aerodrome|runway|helipad|taxiway|apron"];
            node["aeroway"~"aerodrome|helipad"];
            relation["aeroway"="aerodrome"];
        );""",

        # Ports and maritime
        "ports.geojson": """(
            way["waterway"~"dock|boatyard"];
            way["man_made"~"pier|breakwater|groyne|quay"];
            node["man_made"~"pier|lighthouse"];
            way["landuse"="port"];
            relation["landuse"="port"];
            node["seamark:type"];
        );""",

        # Towers and antennas (observation/communication)
        "towers.geojson": """(
            node["man_made"~"tower|mast|communications_tower|antenna"];
            way["man_made"~"tower|mast"];
        );""",

        # Fuel and energy infrastructure
        "fuel_infrastructure.geojson": """(
            node["amenity"="fuel"];
            way["man_made"="storage_tank"];
            way["man_made"="pipeline"];
            node["power"="plant"];
            way["power"="plant"];
        );""",
    }

    download_osm_features(bounds, output_dir, feature_queries, force=force)

    # Rename waterways to streams for compatibility
    waterways_file = output_dir / "waterways.geojson"
    streams_file = output_dir / "streams.geojson"
    if waterways_file.exists() and not streams_file.exists():
        waterways_file.rename(streams_file)
        print("  Renamed waterways.geojson -> streams.geojson")

    # Download reference tiles (lower zoom for 100km square)
    print("\nDownloading reference tiles...")
    download_reference_tiles(bounds, output_dir, f"{gzd}_{square}", zoom=12)

    print(f"\n{'='*60}")
    print(f"Download complete: {output_dir}")
    print(f"{'='*60}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download map data for an MGRS 100km square",
        epilog="""Examples:
  python download_mgrs_data.py "51R TG"        # Download data for Taichung region
  python download_mgrs_data.py 51RTG --verify  # Verify downloaded data
  python download_mgrs_data.py 51RTG --force   # Re-download all features

Available MGRS squares for Taiwan area:
  51R TG - Taichung region
  51R UH - Loudong/Yilan region
  51R TH - Northern Taiwan
  51R TF - Southern Taiwan
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mgrs_square", nargs="*", help="MGRS square (e.g., '51R TG' or '51RTG')")
    parser.add_argument("--verify", action="store_true",
                        help="Verify downloaded data against OSM counts")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download of all features (ignore existing files)")

    args = parser.parse_args()

    if not args.mgrs_square:
        parser.print_help()
        sys.exit(1)

    mgrs_square = " ".join(args.mgrs_square)

    if args.verify:
        # Verification mode - just check counts, don't download
        gzd, square = parse_mgrs_square(mgrs_square)
        bounds = get_mgrs_square_bounds(gzd, square)
        output_dir = DATA_DIR / gzd / square

        if not output_dir.exists():
            print(f"Error: No data found at {output_dir}")
            print(f"Run without --verify first to download data.")
            sys.exit(1)

        # Get the feature queries from the download function
        # We need to define them here too for verification
        feature_queries = {
            "roads.geojson": """(way["highway"];);""",
            "buildings.geojson": """(way["building"];relation["building"];);""",
            "landcover.geojson": """(way["natural"~"wood|grassland|scrub|wetland|beach"];way["landuse"~"forest|farmland|meadow|residential|industrial|commercial|orchard|vineyard"];relation["natural"~"wood|grassland"];relation["landuse"~"forest|farmland"];);""",
            "waterways.geojson": """(way["waterway"~"stream|river|canal|drain|ditch"];);""",
            "waterways_area.geojson": """(way["natural"="water"];relation["natural"="water"];way["waterway"~"riverbank|dock"];way["landuse"="reservoir"];);""",
            "coastline.geojson": """(way["natural"="coastline"];);""",
        }

        verify_downloaded_features(bounds, output_dir, feature_queries)
    else:
        download_mgrs_square(mgrs_square, force=args.force)


if __name__ == "__main__":
    main()
