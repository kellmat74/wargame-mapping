#!/usr/bin/env python3
"""
Region Registry - Auto-discovery of available Geofabrik PBF files.

Scans the cache directory for downloaded PBF files and extracts metadata
(bounding boxes, display names) automatically. No hardcoded region lists needed.

Usage:
    from region_registry import get_available_regions, detect_region_for_coords

    # Get all available regions
    regions = get_available_regions()
    # Returns: {'ukraine': {'bounds': (22.1, 44.2, 40.2, 52.4), 'display_name': 'Ukraine', ...}, ...}

    # Detect region for coordinates
    region = detect_region_for_coords(lat=50.0, lon=36.5)
    # Returns: 'ukraine' or None
"""

import json
import subprocess
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

# Directories where PBF files may be stored
# Primary location used by download_mgrs_data_osmium.py
GEOFABRIK_DIR = Path(__file__).parent / "data" / "geofabrik"
# Secondary location for manual downloads
CACHE_DIR = Path(__file__).parent / "cache"
# Registry file stored in the geofabrik directory
REGISTRY_FILE = GEOFABRIK_DIR / "regions.json"

# Geofabrik base URLs by continent
GEOFABRIK_CONTINENTS = {
    # Europe
    "ukraine": "europe",
    "poland": "europe",
    "germany": "europe",
    "france": "europe",
    "spain": "europe",
    "italy": "europe",
    "united-kingdom": "europe",
    "romania": "europe",
    "belarus": "europe",
    "moldova": "europe",
    "hungary": "europe",
    "czech-republic": "europe",
    "slovakia": "europe",
    "austria": "europe",
    "switzerland": "europe",
    "netherlands": "europe",
    "belgium": "europe",
    "sweden": "europe",
    "norway": "europe",
    "finland": "europe",
    "denmark": "europe",
    "greece": "europe",
    "turkey": "europe",
    "russia": "europe",  # Geofabrik has it under europe
    # Asia
    "philippines": "asia",
    "taiwan": "asia",
    "japan": "asia",
    "south-korea": "asia",
    "indonesia": "asia",
    "malaysia-singapore-brunei": "asia",
    "vietnam": "asia",
    "thailand": "asia",
    "cambodia": "asia",
    "laos": "asia",
    "myanmar": "asia",
    "china": "asia",
    "india": "asia",
    "bangladesh": "asia",
    "pakistan": "asia",
    "iran": "asia",
    "iraq": "asia",
    "syria": "asia",
    # Africa
    "egypt": "africa",
    "south-africa": "africa",
    "morocco": "africa",
    "algeria": "africa",
    "libya": "africa",
    "ethiopia": "africa",
    "kenya": "africa",
    "nigeria": "africa",
    # Americas
    "us-midwest": "north-america",
    "us-northeast": "north-america",
    "us-south": "north-america",
    "us-west": "north-america",
    "canada": "north-america",
    "mexico": "north-america",
    "brazil": "south-america",
    "argentina": "south-america",
    "colombia": "south-america",
    "peru": "south-america",
    "chile": "south-america",
    # Oceania
    "australia": "australia-oceania",
    "new-zealand": "australia-oceania",
}


def get_pbf_bounds(pbf_path: Path) -> Optional[Tuple[float, float, float, float]]:
    """
    Extract bounding box from a PBF file using osmium fileinfo.

    Returns: (min_lon, min_lat, max_lon, max_lat) or None if extraction fails.
    """
    try:
        result = subprocess.run(
            ["osmium", "fileinfo", "-e", str(pbf_path)],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"  Warning: osmium fileinfo failed for {pbf_path.name}")
            return None

        # Parse output for bounding box
        # Format: "    (min_lon,min_lat,max_lon,max_lat)"
        for line in result.stdout.split('\n'):
            match = re.search(r'\(([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)\)', line)
            if match:
                min_lon = float(match.group(1))
                min_lat = float(match.group(2))
                max_lon = float(match.group(3))
                max_lat = float(match.group(4))
                return (min_lon, min_lat, max_lon, max_lat)

        print(f"  Warning: No bounding box found in osmium output for {pbf_path.name}")
        return None

    except subprocess.TimeoutExpired:
        print(f"  Warning: Timeout extracting bounds from {pbf_path.name}")
        return None
    except FileNotFoundError:
        print("  Warning: osmium not found. Install with: brew install osmium-tool")
        return None
    except Exception as e:
        print(f"  Warning: Error extracting bounds: {e}")
        return None


def derive_display_name(region_name: str) -> str:
    """Convert region filename to display name."""
    # Handle special cases
    special_names = {
        "malaysia-singapore-brunei": "Malaysia",
        "south-korea": "South Korea",
        "united-kingdom": "United Kingdom",
        "new-zealand": "New Zealand",
        "south-africa": "South Africa",
        "czech-republic": "Czech Republic",
    }

    if region_name in special_names:
        return special_names[region_name]

    # Default: capitalize and replace hyphens with spaces
    return region_name.replace("-", " ").title()


def get_geofabrik_url(region_name: str) -> str:
    """Get the Geofabrik download URL for a region."""
    continent = GEOFABRIK_CONTINENTS.get(region_name, "asia")  # Default to asia
    return f"https://download.geofabrik.de/{continent}/{region_name}-latest.osm.pbf"


def scan_pbf_directories() -> Dict[str, dict]:
    """
    Scan both geofabrik and cache directories for PBF files and extract metadata.

    Prefers geofabrik directory (primary) over cache directory (secondary).
    Returns dict of region_name -> {bounds, display_name, pbf_file, pbf_path, url}
    """
    regions = {}

    # Scan both directories, preferring geofabrik over cache
    directories = []
    if GEOFABRIK_DIR.exists():
        directories.append(("geofabrik", GEOFABRIK_DIR))
    if CACHE_DIR.exists():
        directories.append(("cache", CACHE_DIR))

    if not directories:
        GEOFABRIK_DIR.mkdir(parents=True, exist_ok=True)
        return regions

    # Find all PBF files from both directories
    pbf_files = []
    for dir_name, dir_path in directories:
        for pbf_path in dir_path.glob("*-latest.osm.pbf"):
            region_name = pbf_path.stem.replace("-latest.osm", "")
            # Only add if not already found (geofabrik takes priority)
            if region_name not in [r[0] for r in pbf_files]:
                pbf_files.append((region_name, pbf_path, dir_name))

    if not pbf_files:
        return regions

    print(f"Scanning {len(pbf_files)} PBF file(s)...")

    for region_name, pbf_path, dir_name in pbf_files:
        print(f"  Processing {region_name} (from {dir_name})...")

        # Get bounding box
        bounds = get_pbf_bounds(pbf_path)
        if bounds is None:
            print(f"    Skipping {region_name} - could not extract bounds")
            continue

        regions[region_name] = {
            "bounds": bounds,
            "display_name": derive_display_name(region_name),
            "pbf_file": pbf_path.name,
            "pbf_path": str(pbf_path),
            "url": get_geofabrik_url(region_name),
        }

        print(f"    Bounds: {bounds}")

    return regions


def save_registry(regions: Dict[str, dict]) -> None:
    """Save region registry to JSON file."""
    GEOFABRIK_DIR.mkdir(parents=True, exist_ok=True)

    # Convert tuples to lists for JSON serialization
    serializable = {}
    for name, data in regions.items():
        serializable[name] = {
            "bounds": list(data["bounds"]),
            "display_name": data["display_name"],
            "pbf_file": data["pbf_file"],
            "pbf_path": data.get("pbf_path", ""),
            "url": data["url"],
        }

    with open(REGISTRY_FILE, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"Saved registry to {REGISTRY_FILE}")


def load_registry() -> Dict[str, dict]:
    """Load region registry from JSON file."""
    if not REGISTRY_FILE.exists():
        return {}

    try:
        with open(REGISTRY_FILE) as f:
            data = json.load(f)

        # Convert lists back to tuples for bounds
        regions = {}
        for name, info in data.items():
            regions[name] = {
                "bounds": tuple(info["bounds"]),
                "display_name": info["display_name"],
                "pbf_file": info["pbf_file"],
                "pbf_path": info.get("pbf_path", ""),
                "url": info["url"],
            }
        return regions
    except Exception as e:
        print(f"Warning: Could not load registry: {e}")
        return {}


def get_available_regions(force_rescan: bool = False) -> Dict[str, dict]:
    """
    Get all available regions with their metadata.

    Uses cached registry if available, otherwise scans PBF directories.
    Set force_rescan=True to always rescan.

    Returns dict of region_name -> {bounds, display_name, pbf_file, pbf_path, url}
    """
    # Check if any new PBF files exist that aren't in registry
    registry = load_registry() if not force_rescan else {}

    # Get list of PBF files from both directories
    pbf_files = set()
    for dir_path in [GEOFABRIK_DIR, CACHE_DIR]:
        if dir_path.exists():
            for p in dir_path.glob("*-latest.osm.pbf"):
                pbf_files.add(p.stem.replace("-latest.osm", ""))

    # Check if registry is up to date
    registry_regions = set(registry.keys())

    if pbf_files != registry_regions or force_rescan:
        # Rescan needed
        registry = scan_pbf_directories()
        if registry:
            save_registry(registry)

    return registry


def detect_region_for_coords(lat: float, lon: float) -> Optional[str]:
    """
    Detect which available region contains the given coordinates.

    Returns region name or None if not found.
    """
    regions = get_available_regions()

    for region_name, info in regions.items():
        min_lon, min_lat, max_lon, max_lat = info["bounds"]
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            return region_name

    return None


def get_region_url(region_name: str) -> Optional[str]:
    """Get the download URL for a region (from registry or generated)."""
    regions = get_available_regions()

    if region_name in regions:
        return regions[region_name]["url"]

    # Not in registry, generate URL
    return get_geofabrik_url(region_name)


def get_region_display_name(region_name: str) -> str:
    """Get the display name for a region."""
    regions = get_available_regions()

    if region_name in regions:
        return regions[region_name]["display_name"]

    return derive_display_name(region_name)


def get_region_pbf_path(region_name: str) -> Optional[Path]:
    """Get the path to the cached PBF file for a region."""
    regions = get_available_regions()

    if region_name in regions:
        pbf_path = regions[region_name].get("pbf_path")
        if pbf_path:
            return Path(pbf_path)
        # Fallback to geofabrik directory
        return GEOFABRIK_DIR / regions[region_name]["pbf_file"]

    # Check if file exists even if not in registry (check both directories)
    for dir_path in [GEOFABRIK_DIR, CACHE_DIR]:
        expected_path = dir_path / f"{region_name}-latest.osm.pbf"
        if expected_path.exists():
            return expected_path

    return None


# CLI for testing/manual operations
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Region Registry - manage available map regions")
    parser.add_argument("--scan", action="store_true", help="Force rescan of cache directory")
    parser.add_argument("--list", action="store_true", help="List available regions")
    parser.add_argument("--detect", nargs=2, type=float, metavar=("LAT", "LON"),
                        help="Detect region for coordinates")

    args = parser.parse_args()

    if args.scan:
        print("Forcing rescan of cache directory...")
        regions = get_available_regions(force_rescan=True)
        print(f"\nFound {len(regions)} region(s)")

    if args.list or (not args.scan and not args.detect):
        regions = get_available_regions()
        if regions:
            print("\nAvailable regions:")
            for name, info in sorted(regions.items()):
                bounds = info["bounds"]
                print(f"  {name}:")
                print(f"    Display name: {info['display_name']}")
                print(f"    Bounds: {bounds[0]:.2f}°E to {bounds[2]:.2f}°E, {bounds[1]:.2f}°N to {bounds[3]:.2f}°N")
                print(f"    PBF file: {info['pbf_file']}")
        else:
            print("\nNo regions available. Download PBF files to cache/ directory.")
            print("Example: curl -o cache/ukraine-latest.osm.pbf https://download.geofabrik.de/europe/ukraine-latest.osm.pbf")

    if args.detect:
        lat, lon = args.detect
        region = detect_region_for_coords(lat, lon)
        if region:
            print(f"\nCoordinates ({lat}, {lon}) are in region: {region}")
        else:
            print(f"\nNo available region contains coordinates ({lat}, {lon})")
