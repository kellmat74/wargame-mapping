"""
fetch_data.py - Download map data for Taiwan hex wargame map

Uses direct Overpass API queries for OSM data.
"""

import os
import json
import time
import requests
import geopandas as gpd
from shapely.geometry import shape, Point, LineString, Polygon, MultiPolygon
from pathlib import Path

# Taiwan bounding box (WGS84)
# Format: (south, west, north, east) for Overpass
TAIWAN_BBOX = (21.8, 120.0, 25.4, 122.1)
BBOX_STR = f"{TAIWAN_BBOX[0]},{TAIWAN_BBOX[1]},{TAIWAN_BBOX[2]},{TAIWAN_BBOX[3]}"

# Output directory
DATA_DIR = Path("data/raw")

# Overpass API endpoint
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def setup_directories():
    """Create data directories if they don't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {DATA_DIR.absolute()}")


def overpass_query(query: str, desc: str = "data") -> dict | None:
    """Execute an Overpass API query with retry logic."""
    print(f"  Querying Overpass API for {desc}...")

    for attempt in range(3):
        try:
            response = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=180,
            )
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print(f"    Rate limited, waiting 30s... (attempt {attempt + 1}/3)")
                time.sleep(30)
            else:
                print(f"    Error {response.status_code}: {response.text[:200]}")
                return None
        except requests.exceptions.Timeout:
            print(f"    Timeout, retrying... (attempt {attempt + 1}/3)")
            time.sleep(10)
        except Exception as e:
            print(f"    Error: {e}")
            return None

    return None


def osm_to_geodataframe(data: dict, geometry_type: str = "all") -> gpd.GeoDataFrame:
    """Convert Overpass JSON to GeoDataFrame."""
    features = []

    # Build node lookup for ways
    nodes = {}
    for element in data.get("elements", []):
        if element["type"] == "node":
            nodes[element["id"]] = (element["lon"], element["lat"])

    for element in data.get("elements", []):
        props = element.get("tags", {})
        props["osm_id"] = element["id"]
        props["osm_type"] = element["type"]

        geom = None

        if element["type"] == "node" and geometry_type in ["all", "point"]:
            geom = Point(element["lon"], element["lat"])

        elif element["type"] == "way" and geometry_type in ["all", "line", "polygon"]:
            coords = []
            for node_id in element.get("nodes", []):
                if node_id in nodes:
                    coords.append(nodes[node_id])

            if len(coords) >= 2:
                if geometry_type == "polygon" and coords[0] == coords[-1] and len(coords) >= 4:
                    geom = Polygon(coords)
                else:
                    geom = LineString(coords)

        if geom is not None:
            features.append({"geometry": geom, "properties": props})

    if not features:
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    return gdf


def download_coastline():
    """Download Taiwan coastline/boundary from OSM."""
    print("\n=== Downloading Coastline ===")

    try:
        import osmnx as ox
        taiwan = ox.geocode_to_gdf("Taiwan")
        taiwan.to_file(DATA_DIR / "taiwan_coastline.geojson", driver="GeoJSON")
        print(f"Saved coastline to {DATA_DIR / 'taiwan_coastline.geojson'}")
        return taiwan
    except Exception as e:
        print(f"Error with osmnx: {e}")
        print("Trying Overpass API...")

        # Get Taiwan admin boundary via Overpass
        query = f"""
        [out:json][timeout:120];
        relation["name:en"="Taiwan"]["admin_level"="2"];
        out geom;
        """

        data = overpass_query(query, "Taiwan boundary")
        if data:
            # For relations, we need special handling
            print("  Processing boundary relation...")
            # Save raw data for now
            with open(DATA_DIR / "taiwan_coastline_raw.json", "w") as f:
                json.dump(data, f)
            print(f"  Saved raw boundary data")

        return None


def download_roads():
    """Download major road network from OSM."""
    print("\n=== Downloading Roads ===")

    # Only fetch major roads suitable for 10km hex scale
    query = f"""
    [out:json][timeout:180];
    (
      way["highway"="motorway"]({BBOX_STR});
      way["highway"="trunk"]({BBOX_STR});
      way["highway"="primary"]({BBOX_STR});
      way["highway"="secondary"]({BBOX_STR});
    );
    out body;
    >;
    out skel qt;
    """

    data = overpass_query(query, "major roads")
    if not data:
        return None

    gdf = osm_to_geodataframe(data, geometry_type="line")
    if len(gdf) == 0:
        print("  No roads found")
        return None

    # Keep relevant columns
    cols = ["geometry", "highway", "name", "ref"]
    cols_available = [c for c in cols if c in gdf.columns]
    gdf = gdf[cols_available]

    gdf.to_file(DATA_DIR / "taiwan_roads.geojson", driver="GeoJSON")
    print(f"Saved {len(gdf)} road segments to {DATA_DIR / 'taiwan_roads.geojson'}")
    return gdf


def download_rivers():
    """Download major rivers from OSM."""
    print("\n=== Downloading Rivers ===")

    query = f"""
    [out:json][timeout:180];
    (
      way["waterway"="river"]({BBOX_STR});
      way["waterway"="canal"]({BBOX_STR});
    );
    out body;
    >;
    out skel qt;
    """

    data = overpass_query(query, "rivers")
    if not data:
        return None

    gdf = osm_to_geodataframe(data, geometry_type="line")
    if len(gdf) == 0:
        print("  No rivers found")
        return None

    # Keep relevant columns
    cols = ["geometry", "waterway", "name", "width"]
    cols_available = [c for c in cols if c in gdf.columns]
    gdf = gdf[cols_available]

    gdf.to_file(DATA_DIR / "taiwan_rivers.geojson", driver="GeoJSON")
    print(f"Saved {len(gdf)} river segments to {DATA_DIR / 'taiwan_rivers.geojson'}")
    return gdf


def download_places():
    """Download cities and towns from OSM."""
    print("\n=== Downloading Places ===")

    query = f"""
    [out:json][timeout:120];
    (
      node["place"="city"]({BBOX_STR});
      node["place"="town"]({BBOX_STR});
    );
    out body;
    """

    data = overpass_query(query, "cities and towns")
    if not data:
        return None

    gdf = osm_to_geodataframe(data, geometry_type="point")
    if len(gdf) == 0:
        print("  No places found")
        return None

    # Keep relevant columns
    cols = ["geometry", "place", "name", "name:en", "population"]
    cols_available = [c for c in cols if c in gdf.columns]
    gdf = gdf[cols_available]

    gdf.to_file(DATA_DIR / "taiwan_places.geojson", driver="GeoJSON")
    print(f"Saved {len(gdf)} places to {DATA_DIR / 'taiwan_places.geojson'}")
    return gdf


def download_infrastructure():
    """Download ports and airfields from OSM."""
    print("\n=== Downloading Infrastructure ===")

    # Airports
    query_airports = f"""
    [out:json][timeout:120];
    (
      node["aeroway"="aerodrome"]({BBOX_STR});
      way["aeroway"="aerodrome"]({BBOX_STR});
      node["aeroway"="airport"]({BBOX_STR});
      way["aeroway"="airport"]({BBOX_STR});
    );
    out body;
    >;
    out skel qt;
    """

    airports_data = overpass_query(query_airports, "airports")

    # Ports
    query_ports = f"""
    [out:json][timeout:120];
    (
      node["harbour"="yes"]({BBOX_STR});
      way["harbour"="yes"]({BBOX_STR});
      node["landuse"="port"]({BBOX_STR});
      way["landuse"="port"]({BBOX_STR});
    );
    out body;
    >;
    out skel qt;
    """

    ports_data = overpass_query(query_ports, "ports")

    all_features = []

    if airports_data:
        gdf = osm_to_geodataframe(airports_data, geometry_type="all")
        if len(gdf) > 0:
            gdf["infrastructure_type"] = "airport"
            # Convert polygons to centroids
            gdf["geometry"] = gdf.geometry.centroid
            all_features.append(gdf)
            print(f"  Found {len(gdf)} airports")

    if ports_data:
        gdf = osm_to_geodataframe(ports_data, geometry_type="all")
        if len(gdf) > 0:
            gdf["infrastructure_type"] = "port"
            gdf["geometry"] = gdf.geometry.centroid
            all_features.append(gdf)
            print(f"  Found {len(gdf)} ports")

    if not all_features:
        print("  No infrastructure found")
        return None

    import pandas as pd
    infra = gpd.GeoDataFrame(pd.concat(all_features, ignore_index=True), crs="EPSG:4326")

    # Keep relevant columns
    cols = ["geometry", "infrastructure_type", "name", "name:en"]
    cols_available = [c for c in cols if c in infra.columns]
    infra = infra[cols_available]

    infra.to_file(DATA_DIR / "taiwan_infrastructure.geojson", driver="GeoJSON")
    print(f"Saved {len(infra)} infrastructure features to {DATA_DIR / 'taiwan_infrastructure.geojson'}")
    return infra


def download_landcover():
    """Download land cover data from OSM."""
    print("\n=== Downloading Land Cover ===")
    print("  Note: Land cover from OSM is incomplete. Consider using ESA WorldCover instead.")

    query = f"""
    [out:json][timeout:300];
    (
      way["landuse"~"residential|commercial|industrial|farmland|forest"]({BBOX_STR});
      way["natural"~"wood|scrub|grassland|water"]({BBOX_STR});
    );
    out body;
    >;
    out skel qt;
    """

    data = overpass_query(query, "land cover")
    if not data:
        return None

    gdf = osm_to_geodataframe(data, geometry_type="polygon")
    if len(gdf) == 0:
        print("  No land cover polygons found")
        return None

    # Keep relevant columns
    cols = ["geometry", "landuse", "natural", "name"]
    cols_available = [c for c in cols if c in gdf.columns]
    gdf = gdf[cols_available]

    gdf.to_file(DATA_DIR / "taiwan_landcover.geojson", driver="GeoJSON")
    print(f"Saved {len(gdf)} land cover polygons to {DATA_DIR / 'taiwan_landcover.geojson'}")
    return gdf


def download_elevation():
    """Provide instructions for downloading elevation data."""
    print("\n=== Elevation Data ===")
    print("Elevation data (DEM) requires downloading from external sources.")
    print("\nRecommended: OpenTopography SRTM")
    print(f"  1. Go to: https://portal.opentopography.org/raster?opentopoID=OTSRTM.082015.4326.1")
    print(f"  2. Draw box around Taiwan or enter coordinates:")
    print(f"     South: {TAIWAN_BBOX[0]}, West: {TAIWAN_BBOX[1]}")
    print(f"     North: {TAIWAN_BBOX[2]}, East: {TAIWAN_BBOX[3]}")
    print(f"  3. Download as GeoTiff")
    print(f"  4. Save as: data/raw/taiwan_elevation.tif")

    instructions = {
        "note": "Download DEM from OpenTopography",
        "url": "https://portal.opentopography.org/raster?opentopoID=OTSRTM.082015.4326.1",
        "bbox": {
            "south": TAIWAN_BBOX[0],
            "west": TAIWAN_BBOX[1],
            "north": TAIWAN_BBOX[2],
            "east": TAIWAN_BBOX[3],
        },
        "output_file": "taiwan_elevation.tif",
    }

    with open(DATA_DIR / "elevation_instructions.json", "w") as f:
        json.dump(instructions, f, indent=2)

    print(f"\nSaved instructions to {DATA_DIR / 'elevation_instructions.json'}")


def main():
    """Download all map data."""
    print("=" * 60)
    print("Taiwan Map Data Downloader")
    print("=" * 60)

    setup_directories()

    # Download each dataset
    download_coastline()
    download_roads()
    download_rivers()
    download_places()
    download_infrastructure()
    download_landcover()
    download_elevation()

    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)

    # List downloaded files
    for f in sorted(DATA_DIR.glob("*")):
        size = f.stat().st_size
        if size > 1_000_000:
            size_str = f"{size / 1_000_000:.1f} MB"
        elif size > 1_000:
            size_str = f"{size / 1_000:.1f} KB"
        else:
            size_str = f"{size} bytes"
        print(f"  {f.name}: {size_str}")


if __name__ == "__main__":
    main()
