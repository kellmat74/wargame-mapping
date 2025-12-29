"""
fetch_elevation.py - Download SRTM elevation data from AWS Terrain Tiles (free, no auth)

Downloads HGT tiles from AWS S3 and merges them into a single GeoTIFF.
"""

import gzip
import io
import struct
from pathlib import Path
import requests
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_bounds

# Taiwan bounding box
BBOX = {
    "south": 21,
    "north": 26,  # Round up to include N25 tile
    "west": 120,
    "east": 123,  # Round up to include E122 tile
}

DATA_DIR = Path("data/raw")
TILES_DIR = DATA_DIR / "srtm_tiles"
OUTPUT_FILE = DATA_DIR / "taiwan_elevation.tif"

# AWS Terrain Tiles (free, no authentication required)
AWS_BUCKET_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi"


def get_tiles_needed():
    """Get list of SRTM tile names needed for the bounding box."""
    tiles = []
    for lat in range(BBOX["south"], BBOX["north"]):
        for lon in range(BBOX["west"], BBOX["east"]):
            # Format: N22E120
            lat_str = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
            lon_str = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
            tiles.append(f"{lat_str}{lon_str}")
    return tiles


def download_tile(tile_name: str) -> Path | None:
    """Download a single SRTM tile from AWS."""
    # URL format: https://s3.amazonaws.com/elevation-tiles-prod/skadi/N22/N22E120.hgt.gz
    lat_folder = tile_name[:3]  # e.g., "N22"
    url = f"{AWS_BUCKET_URL}/{lat_folder}/{tile_name}.hgt.gz"

    output_path = TILES_DIR / f"{tile_name}.hgt"

    if output_path.exists():
        print(f"  {tile_name}: already downloaded")
        return output_path

    try:
        print(f"  {tile_name}: downloading from AWS...")
        response = requests.get(url, timeout=60)

        if response.status_code == 200:
            # Decompress gzip
            decompressed = gzip.decompress(response.content)
            with open(output_path, "wb") as f:
                f.write(decompressed)
            print(f"  {tile_name}: saved ({len(decompressed) / 1_000_000:.1f} MB)")
            return output_path
        elif response.status_code == 404:
            print(f"  {tile_name}: not found (ocean tile)")
            return None
        else:
            print(f"  {tile_name}: error {response.status_code}")
            return None
    except Exception as e:
        print(f"  {tile_name}: error - {e}")
        return None


def hgt_to_geotiff(hgt_path: Path) -> Path | None:
    """Convert HGT file to GeoTIFF."""
    tif_path = hgt_path.with_suffix(".tif")

    if tif_path.exists():
        return tif_path

    # Parse tile name to get coordinates
    tile_name = hgt_path.stem  # e.g., "N22E120"
    lat = int(tile_name[1:3])
    if tile_name[0] == 'S':
        lat = -lat
    lon = int(tile_name[4:7])
    if tile_name[3] == 'W':
        lon = -lon

    # Read HGT file (SRTM1 = 3601x3601, SRTM3 = 1201x1201)
    file_size = hgt_path.stat().st_size
    if file_size == 3601 * 3601 * 2:  # SRTM1 (1 arc-second)
        size = 3601
    elif file_size == 1201 * 1201 * 2:  # SRTM3 (3 arc-second)
        size = 1201
    else:
        print(f"  Unknown HGT size: {file_size} bytes")
        return None

    # Read elevation data (big-endian signed 16-bit integers)
    with open(hgt_path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=">i2").reshape((size, size))

    # Convert to native byte order for rasterio
    data = data.astype(np.int16)

    # HGT files have origin at top-left, data goes south
    # Transform: (west, north) to (east, south)
    transform = from_bounds(lon, lat, lon + 1, lat + 1, size, size)

    # Write GeoTIFF
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype=np.int16,
        crs="EPSG:4326",
        transform=transform,
        nodata=-32768,
    ) as dst:
        dst.write(data, 1)

    return tif_path


def merge_tiles(tif_files: list[Path]) -> bool:
    """Merge multiple GeoTIFF tiles into a single file."""
    print(f"\nMerging {len(tif_files)} tiles...")

    # Open all files
    src_files = [rasterio.open(f) for f in tif_files]

    try:
        # Merge
        mosaic, out_transform = merge(src_files)

        # Get metadata from first file
        out_meta = src_files[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
            "compress": "lzw",
        })

        # Write merged file
        with rasterio.open(OUTPUT_FILE, "w", **out_meta) as dst:
            dst.write(mosaic)

        return True
    finally:
        for src in src_files:
            src.close()


def main():
    print("=" * 60)
    print("SRTM Elevation Data Downloader (AWS Terrain Tiles)")
    print("=" * 60)
    print(f"\nBounding box: {BBOX}")
    print(f"Output: {OUTPUT_FILE}")

    # Create directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TILES_DIR.mkdir(parents=True, exist_ok=True)

    # Get tiles needed
    tiles = get_tiles_needed()
    print(f"\nTiles needed: {len(tiles)}")
    print(f"  {tiles}")

    # Download tiles
    print("\n--- Downloading tiles from AWS ---")
    downloaded_hgt = []
    for tile in tiles:
        hgt_path = download_tile(tile)
        if hgt_path:
            downloaded_hgt.append(hgt_path)

    if not downloaded_hgt:
        print("\nNo tiles downloaded!")
        return False

    print(f"\nDownloaded {len(downloaded_hgt)} tiles")

    # Convert to GeoTIFF
    print("\n--- Converting to GeoTIFF ---")
    tif_files = []
    for hgt_path in downloaded_hgt:
        tif_path = hgt_to_geotiff(hgt_path)
        if tif_path:
            tif_files.append(tif_path)
            print(f"  {hgt_path.stem}: converted to GeoTIFF")

    if not tif_files:
        print("\nNo tiles converted!")
        return False

    # Merge tiles
    if merge_tiles(tif_files):
        size_mb = OUTPUT_FILE.stat().st_size / 1_000_000
        print(f"\n✓ Elevation data saved: {OUTPUT_FILE} ({size_mb:.1f} MB)")
        return True
    else:
        print("\n✗ Merge failed")
        return False


if __name__ == "__main__":
    main()
