"""
fetch_landcover.py - Download land cover data in smaller chunks
"""

import time
import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from pathlib import Path

TAIWAN_BBOX = (21.8, 120.0, 25.4, 122.1)
BBOX_STR = f"{TAIWAN_BBOX[0]},{TAIWAN_BBOX[1]},{TAIWAN_BBOX[2]},{TAIWAN_BBOX[3]}"
DATA_DIR = Path("data/raw")
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def overpass_query(query, desc):
    print(f"  Querying {desc}...")
    for attempt in range(3):
        try:
            response = requests.post(OVERPASS_URL, data={"data": query}, timeout=180)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print(f"    Rate limited, waiting 30s...")
                time.sleep(30)
            else:
                print(f"    Error {response.status_code}")
                return None
        except requests.exceptions.Timeout:
            print(f"    Timeout, retrying...")
            time.sleep(10)
    return None


def osm_to_gdf(data):
    features = []
    nodes = {}
    for el in data.get("elements", []):
        if el["type"] == "node":
            nodes[el["id"]] = (el["lon"], el["lat"])

    for el in data.get("elements", []):
        if el["type"] != "way":
            continue
        props = el.get("tags", {})
        coords = [nodes[n] for n in el.get("nodes", []) if n in nodes]
        if len(coords) >= 4 and coords[0] == coords[-1]:
            features.append({"geometry": Polygon(coords), "properties": props})

    if not features:
        return gpd.GeoDataFrame()
    return gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")


def main():
    print("=== Downloading Land Cover (chunked) ===")

    all_gdfs = []

    # Query each landuse type separately
    landuse_queries = [
        ("urban", "residential|commercial|industrial"),
        ("forest", "forest"),
        ("farmland", "farmland|orchard|vineyard"),
    ]

    for name, type_str in landuse_queries:
        query = f"""
        [out:json][timeout:120];
        way["landuse"~"{type_str}"]({BBOX_STR});
        out body;
        >;
        out skel qt;
        """
        data = overpass_query(query, f"{name} landuse")
        if data:
            gdf = osm_to_gdf(data)
            if len(gdf) > 0:
                all_gdfs.append(gdf)
                print(f"    Found {len(gdf)} {name} polygons")
        time.sleep(2)

    # Natural features
    for ntype in ["wood", "scrub", "grassland", "water"]:
        query = f"""
        [out:json][timeout:120];
        way["natural"="{ntype}"]({BBOX_STR});
        out body;
        >;
        out skel qt;
        """
        data = overpass_query(query, ntype)
        if data:
            gdf = osm_to_gdf(data)
            if len(gdf) > 0:
                all_gdfs.append(gdf)
                print(f"    Found {len(gdf)} {ntype} polygons")
        time.sleep(2)

    if all_gdfs:
        landcover = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs="EPSG:4326")
        landcover.to_file(DATA_DIR / "taiwan_landcover.geojson", driver="GeoJSON")
        print(f"\nSaved {len(landcover)} total land cover polygons")
    else:
        print("No land cover data found")


if __name__ == "__main__":
    main()
