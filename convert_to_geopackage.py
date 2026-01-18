#!/usr/bin/env python3
"""
Convert GeoJSON files to GeoPackage format for faster loading.

GeoPackage (SQLite-based) provides:
- Faster loading times (3-5x improvement)
- Built-in spatial indexing
- Efficient partial reads (only load data within bounds)

Usage:
    # Convert all MGRS squares
    python convert_to_geopackage.py

    # Convert specific square
    python convert_to_geopackage.py 34U/FF

    # Convert and delete original GeoJSON files
    python convert_to_geopackage.py --delete-originals

    # Dry run (show what would be converted)
    python convert_to_geopackage.py --dry-run
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import geopandas as gpd

DATA_DIR = Path(__file__).parent / "data"

# Files to convert (these are the large ones that benefit most)
CONVERTIBLE_FILES = [
    "buildings.geojson",
    "landcover.geojson",
    "farmland.geojson",
    "roads.geojson",
    "paths.geojson",
    "streams.geojson",
    "barriers.geojson",
    "powerlines.geojson",
    "railways.geojson",
    "waterways.geojson",
    "waterways_area.geojson",
    "coastline.geojson",
    "tree_rows.geojson",
    "cliffs.geojson",
    "bridges.geojson",
    "wetland.geojson",
    "heath.geojson",
    "rocky_terrain.geojson",
    "sand.geojson",
    "mangrove.geojson",
    "military.geojson",
    "quarries.geojson",
    "cemeteries.geojson",
    "places.geojson",
    "peaks.geojson",
    "caves.geojson",
    "dams.geojson",
    "airfields.geojson",
    "ports.geojson",
    "towers.geojson",
    "fuel_infrastructure.geojson",
]


def get_mgrs_directories() -> List[Path]:
    """Find all MGRS data directories."""
    mgrs_dirs = []

    if not DATA_DIR.exists():
        return mgrs_dirs

    # Pattern: data/{zone}/{square}/ e.g., data/34U/FF/
    for zone_dir in DATA_DIR.iterdir():
        if zone_dir.is_dir() and zone_dir.name not in ['geofabrik', 'land-polygons']:
            for square_dir in zone_dir.iterdir():
                if square_dir.is_dir():
                    # Check if it has geojson files
                    if any(square_dir.glob("*.geojson")):
                        mgrs_dirs.append(square_dir)

    return sorted(mgrs_dirs)


def convert_geojson_to_gpkg(
    geojson_path: Path,
    delete_original: bool = False,
    dry_run: bool = False
) -> Optional[Path]:
    """
    Convert a GeoJSON file to GeoPackage format.

    Args:
        geojson_path: Path to the GeoJSON file
        delete_original: If True, delete the original after successful conversion
        dry_run: If True, only print what would be done

    Returns:
        Path to the created GeoPackage, or None if skipped/failed
    """
    gpkg_path = geojson_path.with_suffix('.gpkg')

    # Skip if already converted
    if gpkg_path.exists():
        return None

    # Skip empty files
    if geojson_path.stat().st_size < 100:
        return None

    if dry_run:
        size_mb = geojson_path.stat().st_size / (1024 * 1024)
        print(f"  Would convert: {geojson_path.name} ({size_mb:.1f} MB)")
        return gpkg_path

    try:
        # Load GeoJSON
        gdf = gpd.read_file(geojson_path)

        if gdf.empty:
            return None

        # Save as GeoPackage with spatial index
        gdf.to_file(gpkg_path, driver="GPKG", spatial_index=True)

        # Verify the conversion
        if gpkg_path.exists() and gpkg_path.stat().st_size > 0:
            size_orig = geojson_path.stat().st_size / (1024 * 1024)
            size_new = gpkg_path.stat().st_size / (1024 * 1024)
            print(f"  Converted: {geojson_path.name} ({size_orig:.1f} MB → {size_new:.1f} MB)")

            if delete_original:
                geojson_path.unlink()
                print(f"    Deleted original: {geojson_path.name}")

            return gpkg_path
        else:
            print(f"  ERROR: Conversion failed for {geojson_path.name}")
            return None

    except Exception as e:
        print(f"  ERROR converting {geojson_path.name}: {e}")
        return None


def convert_directory(
    mgrs_dir: Path,
    delete_originals: bool = False,
    dry_run: bool = False
) -> int:
    """
    Convert all GeoJSON files in an MGRS directory to GeoPackage.

    Returns number of files converted.
    """
    converted = 0

    for filename in CONVERTIBLE_FILES:
        geojson_path = mgrs_dir / filename
        if geojson_path.exists():
            result = convert_geojson_to_gpkg(
                geojson_path,
                delete_original=delete_originals,
                dry_run=dry_run
            )
            if result:
                converted += 1

    return converted


def main():
    parser = argparse.ArgumentParser(
        description="Convert GeoJSON files to GeoPackage for faster loading"
    )
    parser.add_argument(
        "squares",
        nargs="*",
        help="Specific MGRS squares to convert (e.g., '34U/FF'). If none, converts all."
    )
    parser.add_argument(
        "--delete-originals",
        action="store_true",
        help="Delete original GeoJSON files after successful conversion"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be converted without actually converting"
    )

    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN - no files will be modified\n")

    # Find directories to process
    if args.squares:
        # Convert specific squares
        mgrs_dirs = []
        for square in args.squares:
            square_path = DATA_DIR / square.replace(" ", "/")
            if square_path.exists():
                mgrs_dirs.append(square_path)
            else:
                print(f"Warning: Directory not found: {square_path}")
    else:
        # Convert all
        mgrs_dirs = get_mgrs_directories()

    if not mgrs_dirs:
        print("No MGRS data directories found.")
        return 1

    print(f"Found {len(mgrs_dirs)} MGRS directories to process\n")

    total_converted = 0

    for mgrs_dir in mgrs_dirs:
        rel_path = mgrs_dir.relative_to(DATA_DIR)
        print(f"Processing {rel_path}...")

        converted = convert_directory(
            mgrs_dir,
            delete_originals=args.delete_originals,
            dry_run=args.dry_run
        )

        if converted > 0:
            total_converted += converted
            print(f"  → {converted} files converted")
        else:
            print(f"  → No files to convert (already done or empty)")
        print()

    print(f"Total: {total_converted} files converted")

    if not args.dry_run and total_converted > 0:
        print("\nGeoPackage files created. The map generator will automatically")
        print("use .gpkg files when available, falling back to .geojson if not.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
