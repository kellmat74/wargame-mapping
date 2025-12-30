#!/usr/bin/env python3
"""
download_mgrs_data_osmium.py - Download map data for an MGRS 100km square using Osmium

Uses Geofabrik regional PBF extracts + osmium tool for reliable, complete OSM data.
Falls back to Overpass API only for coastlines (which are handled specially by Geofabrik).

Data is organized by Grid Zone Designator and square:
  data/51R/TG/elevation.tif
  data/51R/TG/roads.geojson
  etc.

Prerequisites:
  - osmium-tool: brew install osmium-tool
  - Python packages: mgrs, requests, rasterio (optional)

Usage:
  python download_mgrs_data_osmium.py "51R TG"
  python download_mgrs_data_osmium.py --region philippines "50P PS"
"""

import sys
import json
import time
import subprocess
import requests
from pathlib import Path
from typing import Tuple, Dict, Optional
import mgrs
import tempfile
import shutil

# === Configuration ===
DATA_DIR = Path("data")
GEOFABRIK_DIR = DATA_DIR / "geofabrik"
OPENTOPOGRAPHY_API_KEY = "137877e41e9d540cf80cc3601dc2230a"
OVERPASS_URL = "https://lz4.overpass-api.de/api/interpreter"

# Geofabrik region URLs (add more as needed)
GEOFABRIK_REGIONS = {
    "philippines": "https://download.geofabrik.de/asia/philippines-latest.osm.pbf",
    "taiwan": "https://download.geofabrik.de/asia/taiwan-latest.osm.pbf",
    "japan": "https://download.geofabrik.de/asia/japan-latest.osm.pbf",
    "south-korea": "https://download.geofabrik.de/asia/south-korea-latest.osm.pbf",
    "indonesia": "https://download.geofabrik.de/asia/indonesia-latest.osm.pbf",
    "malaysia-singapore-brunei": "https://download.geofabrik.de/asia/malaysia-singapore-brunei-latest.osm.pbf",
    "vietnam": "https://download.geofabrik.de/asia/vietnam-latest.osm.pbf",
    "thailand": "https://download.geofabrik.de/asia/thailand-latest.osm.pbf",
}

# Feature definitions: tag filters for osmium tags-filter
# Format: (filename, osmium_filter)
# osmium_filter uses the osmium tags-filter syntax: n/r/w for node/relation/way
# Using broader filters (tag keys without value restrictions) for complete data
# All geometry types (point,linestring,polygon) are exported by default
OSM_FEATURES = {
    # Core features - use broad filters for completeness
    "roads.geojson": "w/highway",  # All highway types
    "buildings.geojson": "nwr/building",  # All buildings
    "landcover.geojson": "nwr/landuse nwr/natural",  # All landuse and natural features

    # Water features
    "waterways.geojson": "w/waterway",  # All waterway types
    "waterways_area.geojson": "wr/natural=water wr/waterway=riverbank wr/landuse=reservoir",

    # Infrastructure
    "railways.geojson": "nwr/railway",  # All railway types
    "powerlines.geojson": "nwr/power",  # All power infrastructure
    "bridges.geojson": "nwr/bridge w/man_made=bridge",

    # Paths and trails (subset of highway - kept for backward compatibility)
    "paths.geojson": "w/highway=footway,path,cycleway,bridleway,steps,pedestrian",

    # Barriers and boundaries
    "barriers.geojson": "nwr/barrier",  # All barrier types

    # Vegetation
    "tree_rows.geojson": "nwr/natural=tree_row nwr/natural=tree",

    # Terrain features
    "cliffs.geojson": "w/natural=cliff,earth_bank w/man_made=embankment",

    # Agricultural
    "farmland.geojson": "wr/landuse=farmland,paddy,orchard,vineyard,farm,greenhouse_horticulture",

    # Wetlands and mangrove
    "mangrove.geojson": "wr/natural=mangrove",
    "wetland.geojson": "nwr/wetland nwr/natural=marsh,swamp,wetland,reedbed,saltmarsh,mud,bog,fen",

    # Heath and scrubland
    "heath.geojson": "wr/natural=heath,scrub,fell",

    # Rocky terrain
    "rocky_terrain.geojson": "wr/natural=bare_rock,scree,shingle,rock,stone",

    # Sand and dunes
    "sand.geojson": "wr/natural=sand,dune,beach",

    # Military areas
    "military.geojson": "nwr/landuse=military nwr/military",

    # Quarries and mines
    "quarries.geojson": "nwr/landuse=quarry nwr/man_made=mineshaft,adit,mine",

    # Cemeteries
    "cemeteries.geojson": "wr/landuse=cemetery wr/amenity=grave_yard",

    # Places (settlements)
    "places.geojson": "n/place",  # All place types

    # Peaks and terrain features
    "peaks.geojson": "n/natural=peak,saddle,volcano,hill,ridge",

    # Caves
    "caves.geojson": "nwr/natural=cave_entrance,cave",

    # Dams
    "dams.geojson": "nwr/waterway=dam,weir,lock_gate nwr/man_made=dam",

    # Airfields and aviation
    "airfields.geojson": "nwr/aeroway",  # All aeroway types

    # Ports and maritime
    "ports.geojson": (
        "nwr/waterway=dock,boatyard "
        "nwr/man_made=pier,breakwater,groyne,quay,lighthouse,jetty "
        "nwr/landuse=port "
        "nwr/harbour "
        "n/seamark:type"
    ),

    # Towers and antennas
    "towers.geojson": "nwr/man_made=tower,mast,communications_tower,antenna,monitoring_station",

    # Fuel and energy infrastructure
    "fuel_infrastructure.geojson": (
        "nwr/amenity=fuel "
        "nwr/man_made=storage_tank,pipeline,petroleum_well "
        "nwr/power=plant,generator,substation"
    ),
}


def parse_mgrs_square(square_str: str) -> Tuple[str, str]:
    """Parse MGRS square designator into GZD and 100km square ID."""
    cleaned = square_str.replace(" ", "").replace("-", "").upper()

    if len(cleaned) < 5:
        raise ValueError(f"Invalid MGRS square: {square_str}")

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
    """Get the WGS84 bounding box for an MGRS 100km square."""
    m = mgrs.MGRS()
    base = f"{gzd}{square}"

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
                pass

    # Sample corners
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

    # Add buffer
    buffer = 0.01
    return (min_lon - buffer, min_lat - buffer, max_lon + buffer, max_lat + buffer)


def check_osmium_installed() -> bool:
    """Check if osmium is installed and accessible."""
    try:
        result = subprocess.run(
            ["osmium", "--version"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def download_geofabrik_pbf(region: str) -> Path:
    """Download a Geofabrik regional PBF file if not already present."""
    GEOFABRIK_DIR.mkdir(parents=True, exist_ok=True)

    if region not in GEOFABRIK_REGIONS:
        available = ", ".join(GEOFABRIK_REGIONS.keys())
        raise ValueError(f"Unknown region: {region}. Available: {available}")

    url = GEOFABRIK_REGIONS[region]
    filename = f"{region}-latest.osm.pbf"
    output_file = GEOFABRIK_DIR / filename

    if output_file.exists():
        print(f"  Using cached {filename}")
        return output_file

    print(f"  Downloading {filename} from Geofabrik...")
    print(f"    URL: {url}")

    response = requests.get(url, stream=True, timeout=3600)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0

    with open(output_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = downloaded * 100 / total_size
                mb = downloaded / 1024 / 1024
                print(f"\r    Downloaded: {mb:.1f} MB ({pct:.1f}%)", end="", flush=True)

    print(f"\n    Saved {output_file} ({downloaded / 1024 / 1024:.1f} MB)")
    return output_file


def extract_region_pbf(
    source_pbf: Path,
    bounds: Tuple[float, float, float, float],
    output_pbf: Path
) -> bool:
    """Extract a bounding box region from a PBF file using osmium."""
    min_lon, min_lat, max_lon, max_lat = bounds
    bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"

    cmd = [
        "osmium", "extract",
        f"--bbox={bbox}",
        str(source_pbf),
        "-o", str(output_pbf),
        "--overwrite"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    Error extracting region: {result.stderr}")
        return False

    return True


def filter_and_export_features(
    source_pbf: Path,
    output_geojson: Path,
    tag_filter: str
) -> int:
    """Filter OSM features and export to GeoJSON using osmium.

    Args:
        source_pbf: Path to the source PBF file
        output_geojson: Path for output GeoJSON file
        tag_filter: Space-separated osmium tag filter expressions

    Returns:
        Number of features exported
    """

    # Create temp file for filtered PBF
    with tempfile.NamedTemporaryFile(suffix=".osm.pbf", delete=False) as tmp:
        filtered_pbf = Path(tmp.name)

    try:
        # Split tag filter into individual filters
        # osmium tags-filter expects each filter as a separate argument
        filters = tag_filter.split()

        # Run osmium tags-filter
        cmd = ["osmium", "tags-filter", str(source_pbf)] + filters + [
            "-o", str(filtered_pbf),
            "--overwrite"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"      Filter error: {result.stderr}")
            return 0

        # Export to GeoJSON with OSM IDs (all geometry types)
        cmd = [
            "osmium", "export",
            str(filtered_pbf),
            "--add-unique-id=type_id",
            "-o", str(output_geojson),
            "--overwrite"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"      Export error: {result.stderr}")
            return 0

        # Count features
        if output_geojson.exists():
            with open(output_geojson) as f:
                data = json.load(f)
                return len(data.get("features", []))

        return 0

    finally:
        # Clean up temp file
        if filtered_pbf.exists():
            filtered_pbf.unlink()


def download_coastline_overpass(
    bounds: Tuple[float, float, float, float],
    output_file: Path
) -> bool:
    """Download coastline using Overpass API (Geofabrik doesn't include coastlines)."""
    min_lon, min_lat, max_lon, max_lat = bounds
    bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"

    query = f"""
    [out:json][timeout:300][bbox:{bbox}];
    (
        way["natural"="coastline"];
    );
    out body geom;
    """

    try:
        response = requests.post(
            OVERPASS_URL,
            data={"data": query},
            timeout=600
        )

        if response.status_code == 429:
            print("      Rate limited, waiting 60 seconds...")
            time.sleep(60)
            response = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=600
            )

        if response.status_code != 200:
            print(f"      Error {response.status_code}")
            return False

        # Convert to GeoJSON
        elements = response.json().get("elements", [])
        features = []

        for element in elements:
            if element["type"] == "way" and "geometry" in element:
                coords = [[p["lon"], p["lat"]] for p in element["geometry"]]
                if len(coords) >= 2:
                    features.append({
                        "type": "Feature",
                        "id": f"w{element['id']}",
                        "properties": element.get("tags", {}),
                        "geometry": {
                            "type": "LineString",
                            "coordinates": coords
                        }
                    })

        geojson = {"type": "FeatureCollection", "features": features}
        with open(output_file, 'w') as f:
            json.dump(geojson, f)

        return True

    except Exception as e:
        print(f"      Error: {e}")
        return False


def download_elevation(
    bounds: Tuple[float, float, float, float],
    output_dir: Path
) -> None:
    """Download elevation data for the bounds."""
    output_file = output_dir / "elevation.tif"

    if output_file.exists():
        print("  elevation.tif already exists, skipping")
        return

    min_lon, min_lat, max_lon, max_lat = bounds

    print("  Downloading elevation data...")
    print(f"    Bounds: {min_lat:.4f},{min_lon:.4f} to {max_lat:.4f},{max_lon:.4f}")

    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": "SRTMGL1",
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
        else:
            print(f"    Error {response.status_code}: {response.text[:200]}")
    except Exception as e:
        print(f"    Error downloading elevation: {e}")


def download_reference_tiles(
    bounds: Tuple[float, float, float, float],
    output_dir: Path,
    name: str,
    zoom: int = 12
) -> None:
    """Download and stitch OpenTopoMap tiles for reference."""
    import math
    from io import BytesIO

    try:
        from PIL import Image
    except ImportError:
        print("  PIL not available, skipping reference tiles")
        return

    output_image = output_dir / "reference_tiles.png"
    output_info = output_dir / "reference_tiles.json"

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

    x_min, y_min = lat_lon_to_tile(max_lat, min_lon, zoom)
    x_max, y_max = lat_lon_to_tile(min_lat, max_lon, zoom)

    num_tiles = (x_max - x_min + 1) * (y_max - y_min + 1)
    print(f"    Downloading {num_tiles} tiles...")

    if num_tiles > 500:
        print(f"    Too many tiles ({num_tiles}), reducing zoom level")
        return download_reference_tiles(bounds, output_dir, name, zoom - 1)

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
                time.sleep(0.1)
            except Exception as e:
                print(f"      Error fetching tile {x},{y}: {e}")

    if not tiles:
        print("    No tiles downloaded")
        return

    width = (x_max - x_min + 1) * tile_size
    height = (y_max - y_min + 1) * tile_size
    stitched = Image.new('RGB', (width, height))

    for (x, y), tile in tiles.items():
        px = (x - x_min) * tile_size
        py = (y - y_min) * tile_size
        stitched.paste(tile, (px, py))

    stitched.save(output_image)

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


def download_mgrs_square_osmium(
    mgrs_square: str,
    region: str,
    force: bool = False
) -> None:
    """Download all data for an MGRS 100km square using Osmium."""

    # Check osmium is installed
    if not check_osmium_installed():
        print("ERROR: osmium-tool is not installed.")
        print("Install with: brew install osmium-tool")
        sys.exit(1)

    # Parse the MGRS square
    gzd, square = parse_mgrs_square(mgrs_square)
    print(f"\n{'='*60}")
    print(f"Downloading data for MGRS square: {gzd} {square}")
    print(f"Using Geofabrik region: {region}")
    print(f"{'='*60}")

    # Get bounds
    print("\nCalculating bounds...")
    bounds = get_mgrs_square_bounds(gzd, square)
    min_lon, min_lat, max_lon, max_lat = bounds
    print(f"  SW: {min_lat:.4f}N, {min_lon:.4f}E")
    print(f"  NE: {max_lat:.4f}N, {max_lon:.4f}E")
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
            },
            "source": {
                "type": "geofabrik",
                "region": region
            }
        }, f, indent=2)

    # Download elevation
    print("\nDownloading elevation data...")
    download_elevation(bounds, output_dir)

    # Download Geofabrik PBF
    print("\nPreparing OSM data...")
    regional_pbf = download_geofabrik_pbf(region)

    # Extract region PBF
    print(f"  Extracting region from {region} PBF...")
    region_pbf = GEOFABRIK_DIR / f"{gzd}_{square}.osm.pbf"

    if region_pbf.exists() and not force:
        print(f"    Using cached {region_pbf.name}")
    else:
        if not extract_region_pbf(regional_pbf, bounds, region_pbf):
            print("    ERROR: Failed to extract region")
            sys.exit(1)

        # Get file info
        result = subprocess.run(
            ["osmium", "fileinfo", "-e", str(region_pbf)],
            capture_output=True,
            text=True
        )
        # Extract counts from output
        for line in result.stdout.split('\n'):
            if 'Number of nodes:' in line:
                print(f"    {line.strip()}")
            elif 'Number of ways:' in line:
                print(f"    {line.strip()}")
            elif 'Number of relations:' in line:
                print(f"    {line.strip()}")

    # Extract features
    print("\nExtracting OSM features...")

    for filename, tag_filter in OSM_FEATURES.items():
        output_file = output_dir / filename

        if output_file.exists() and not force:
            with open(output_file) as f:
                data = json.load(f)
                count = len(data.get("features", []))
            print(f"  {filename}: {count} features (cached)")
            continue

        print(f"  Extracting {filename}...", end=" ", flush=True)
        count = filter_and_export_features(
            region_pbf,
            output_file,
            tag_filter
        )
        print(f"{count} features")

    # Coastline needs special handling (not in Geofabrik regional extracts)
    coastline_file = output_dir / "coastline.geojson"
    if coastline_file.exists() and not force:
        with open(coastline_file) as f:
            data = json.load(f)
            count = len(data.get("features", []))
        print(f"  coastline.geojson: {count} features (cached)")
    else:
        print("  Downloading coastline.geojson (via Overpass)...", end=" ", flush=True)
        if download_coastline_overpass(bounds, coastline_file):
            with open(coastline_file) as f:
                data = json.load(f)
                count = len(data.get("features", []))
            print(f"{count} features")
        else:
            print("failed")

    # Download reference tiles
    print("\nDownloading reference tiles...")
    download_reference_tiles(bounds, output_dir, f"{gzd}_{square}", zoom=12)

    # Summary
    print(f"\n{'='*60}")
    print(f"Download complete: {output_dir}")
    print(f"{'='*60}")

    # Count total features
    total_features = 0
    for filename in list(OSM_FEATURES.keys()) + ["coastline.geojson"]:
        filepath = output_dir / filename
        if filepath.exists():
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    total_features += len(data.get("features", []))
            except:
                pass

    print(f"\nTotal OSM features: {total_features:,}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download map data for an MGRS 100km square using Osmium",
        epilog="""Examples:
  python download_mgrs_data_osmium.py --region philippines "50P PS"
  python download_mgrs_data_osmium.py --region taiwan "51R TG"
  python download_mgrs_data_osmium.py --region taiwan "51R TG" --force

Available regions:
  philippines, taiwan, japan, south-korea, indonesia,
  malaysia-singapore-brunei, vietnam, thailand
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mgrs_square", nargs="*", help="MGRS square (e.g., '51R TG' or '51RTG')")
    parser.add_argument("--region", required=True, help="Geofabrik region name")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download of all features (ignore existing files)")
    parser.add_argument("--list-regions", action="store_true",
                        help="List available Geofabrik regions")

    args = parser.parse_args()

    if args.list_regions:
        print("Available Geofabrik regions:")
        for name, url in sorted(GEOFABRIK_REGIONS.items()):
            print(f"  {name}: {url}")
        sys.exit(0)

    if not args.mgrs_square:
        parser.print_help()
        sys.exit(1)

    mgrs_square = " ".join(args.mgrs_square)
    download_mgrs_square_osmium(mgrs_square, args.region, force=args.force)


if __name__ == "__main__":
    main()
