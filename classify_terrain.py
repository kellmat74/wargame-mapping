"""
classify_terrain.py - Phase 2: Terrain Classification

Assigns terrain types to each hex based on:
- Coastline (land vs water)
- Elevation (DEM)
- Land cover (OSM data)
"""

import json
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box, mapping, MultiPolygon
from shapely.affinity import rotate
from shapely import minimum_rotated_rectangle
import pyproj
from pyproj import Transformer

from hexgrid import HexGrid, HexData, render_svg, TERRAIN_COLORS

# === Configuration ===
DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("output")

# Rotation angle (degrees) - computed from minimum bounding rectangle
# Positive = counter-clockwise. Set to 0 to disable rotation.
ROTATION_ANGLE = 48.0  # Rotates Taiwan to be more vertical

# Padding around island (in hex units) to ensure water surrounds it
PADDING_HEXES = 2

# Grid parameters
HEX_SIZE_M = 10000  # 10 km
GRID_CRS = "EPSG:3826"
DATA_CRS = "EPSG:4326"

# Terrain priority (higher = takes precedence)
TERRAIN_PRIORITY = {
    "water": 7,
    "urban": 6,
    "mountain": 5,
    "forest": 4,
    "paddy": 3,
    "rough": 2,
    "clear": 1,
}

# Elevation bands
ELEVATION_BANDS = [
    (0, 100, 0, "lowland"),
    (100, 500, 1, "hills"),
    (500, 1000, 2, "highlands"),
    (1000, 2000, 3, "mountains"),
    (2000, 5000, 4, "high_mountains"),
]


def get_elevation_band(elevation: float) -> tuple[int, str]:
    """Get elevation band level and label."""
    for min_e, max_e, level, label in ELEVATION_BANDS:
        if min_e <= elevation < max_e:
            return level, label
    return 4, "high_mountains"


def get_main_island(gdf: gpd.GeoDataFrame) -> tuple:
    """Extract main Taiwan island and compute rotation center."""
    all_geom = gdf.union_all()

    if isinstance(all_geom, MultiPolygon):
        # Find largest polygon (main island)
        polygons = list(all_geom.geoms)
        polygons.sort(key=lambda p: p.area, reverse=True)
        main_island = polygons[0]
    else:
        main_island = all_geom

    return main_island, main_island.centroid


def rotate_geometry(geom, angle: float, origin):
    """Rotate geometry around origin point."""
    if angle == 0:
        return geom
    return rotate(geom, angle, origin=origin)


def rotate_geodataframe(gdf: gpd.GeoDataFrame, angle: float, origin) -> gpd.GeoDataFrame:
    """Rotate all geometries in a GeoDataFrame."""
    if angle == 0:
        return gdf
    rotated = gdf.copy()
    rotated['geometry'] = rotated['geometry'].apply(lambda g: rotate_geometry(g, angle, origin))
    return rotated


def create_grid(coastline: gpd.GeoDataFrame) -> tuple[HexGrid, list[tuple[int, int]], tuple]:
    """Create hex grid covering rotated Taiwan with padding."""
    print("Creating hex grid...")

    # Get main island and rotation center
    main_island, rotation_center = get_main_island(coastline)
    print(f"  Rotation center: ({rotation_center.x:.0f}, {rotation_center.y:.0f})")
    print(f"  Rotation angle: {ROTATION_ANGLE}°")

    # Rotate the main island to compute bounds
    rotated_island = rotate_geometry(main_island, ROTATION_ANGLE, rotation_center)

    # Get bounds of rotated island
    minx, miny, maxx, maxy = rotated_island.bounds
    width = maxx - minx
    height = maxy - miny
    print(f"  Rotated bounds: {width/1000:.0f} km x {height/1000:.0f} km")

    # Add padding (in meters)
    padding_m = PADDING_HEXES * HEX_SIZE_M
    minx -= padding_m
    miny -= padding_m
    maxx += padding_m
    maxy += padding_m

    # Create grid with origin at northwest corner
    grid = HexGrid(
        hex_size_m=HEX_SIZE_M,
        origin_x=minx,
        origin_y=maxy,
    )

    # Generate all hex coordinates
    hex_coords = list(grid.generate_grid(
        min_x=minx,
        min_y=miny,
        max_x=maxx,
        max_y=maxy,
    ))

    # Filter to positive coordinates only
    hex_coords = [(q, r) for q, r in hex_coords if q >= 0 and r >= 0]

    print(f"  Grid size: {len(hex_coords)} hexes")
    return grid, hex_coords, (rotation_center, ROTATION_ANGLE)


def load_coastline() -> gpd.GeoDataFrame:
    """Load Taiwan coastline polygon."""
    print("Loading coastline...")
    coastline = gpd.read_file(DATA_DIR / "taiwan_coastline.geojson")

    # Reproject to grid CRS
    coastline = coastline.to_crs(GRID_CRS)
    print(f"  Loaded {len(coastline)} polygon(s)")
    return coastline


def load_elevation() -> rasterio.DatasetReader:
    """Load elevation raster."""
    print("Loading elevation data...")
    dem = rasterio.open(DATA_DIR / "taiwan_elevation.tif")
    print(f"  Raster size: {dem.width}x{dem.height}")
    print(f"  CRS: {dem.crs}")
    return dem


def load_landcover() -> gpd.GeoDataFrame:
    """Load land cover polygons."""
    print("Loading land cover...")

    # Load main landcover
    landcover = gpd.read_file(DATA_DIR / "taiwan_landcover.geojson")
    print(f"  Main landcover: {len(landcover)} polygons")

    # Load urban separately if exists
    urban_path = DATA_DIR / "taiwan_urban.geojson"
    if urban_path.exists():
        print("  Loading urban data...")
        urban = gpd.read_file(urban_path)
        urban["terrain_type"] = "urban"
        print(f"  Urban: {len(urban)} polygons")

        # Ensure same CRS before concat
        if urban.crs != landcover.crs:
            urban = urban.to_crs(landcover.crs)

        landcover = gpd.GeoDataFrame(
            pd.concat([landcover, urban], ignore_index=True),
            crs=landcover.crs
        )

    # Reproject to grid CRS
    landcover = landcover.to_crs(GRID_CRS)
    print(f"  Total: {len(landcover)} polygons")
    return landcover


def load_rivers() -> gpd.GeoDataFrame:
    """Load river linestrings."""
    print("Loading rivers...")
    rivers = gpd.read_file(DATA_DIR / "taiwan_rivers.geojson")
    rivers = rivers.to_crs(GRID_CRS)
    print(f"  Loaded {len(rivers)} river segments")
    return rivers


def load_roads() -> gpd.GeoDataFrame:
    """Load road network."""
    print("Loading roads...")
    roads = gpd.read_file(DATA_DIR / "taiwan_roads.geojson")
    roads = roads.to_crs(GRID_CRS)

    # Count by type
    if 'highway' in roads.columns:
        for hwy_type in ['motorway', 'trunk', 'primary', 'secondary']:
            count = len(roads[roads['highway'] == hwy_type])
            if count > 0:
                print(f"    {hwy_type}: {count}")

    print(f"  Total: {len(roads)} road segments")
    return roads


def load_places() -> gpd.GeoDataFrame:
    """Load cities and towns."""
    print("Loading places...")
    places = gpd.read_file(DATA_DIR / "taiwan_places.geojson")
    places = places.to_crs(GRID_CRS)

    cities = len(places[places['place'] == 'city'])
    towns = len(places[places['place'] == 'town'])
    print(f"  Loaded {cities} cities, {towns} towns")
    return places


def load_infrastructure() -> gpd.GeoDataFrame:
    """Load airports and ports."""
    print("Loading infrastructure...")
    infra = gpd.read_file(DATA_DIR / "taiwan_infrastructure.geojson")
    infra = infra.to_crs(GRID_CRS)

    # Count types
    if 'infrastructure_type' in infra.columns:
        for infra_type in infra['infrastructure_type'].unique():
            count = len(infra[infra['infrastructure_type'] == infra_type])
            print(f"    {infra_type}: {count}")

    print(f"  Total: {len(infra)} features")
    return infra


def classify_landcover_type(row) -> str:
    """Map OSM landuse/natural tags to terrain types."""
    # Check if already classified (e.g., urban data)
    if "terrain_type" in row.index and pd.notna(row.get("terrain_type")):
        return row["terrain_type"]

    landuse = str(row.get("landuse", "") or "")
    natural = str(row.get("natural", "") or "")

    # Urban
    if landuse in ["residential", "commercial", "industrial", "retail"]:
        return "urban"

    # Forest
    if landuse == "forest" or natural == "wood":
        return "forest"

    # Water (from natural tag)
    if natural == "water" or landuse in ["reservoir", "basin", "aquaculture"]:
        return "water"

    # Paddy/Agriculture
    if landuse in ["farmland", "orchard", "vineyard", "meadow"]:
        return "paddy"

    # Rough terrain
    if natural in ["scrub", "grassland", "heath", "wetland"] or landuse == "grass":
        return "rough"

    return "clear"


def process_rivers(
    grid: HexGrid,
    hex_coords: list[tuple[int, int]],
    rivers: gpd.GeoDataFrame,
) -> dict[tuple[int, int], list[int]]:
    """
    Detect which hex edges are crossed by rivers.

    Returns:
        Dict mapping (q, r) to list of edge indices (0-5) with river crossings
    """
    print("\nProcessing rivers...")
    from shapely.geometry import LineString

    river_edges = {}

    # Build spatial index for rivers
    rivers_sindex = rivers.sindex

    total = len(hex_coords)
    for i, (q, r) in enumerate(hex_coords):
        if i % 200 == 0:
            print(f"  Processing hex {i}/{total}...")

        hex_poly = grid.hex_polygon(q, r)
        edge_midpoints = grid.hex_edge_midpoints(q, r)

        # Find rivers that might intersect this hex
        possible_idx = list(rivers_sindex.intersection(hex_poly.bounds))
        if not possible_idx:
            continue

        possible_rivers = rivers.iloc[possible_idx]
        intersecting = possible_rivers[possible_rivers.intersects(hex_poly)]

        if len(intersecting) == 0:
            continue

        # Check each edge for river crossings
        edges_crossed = []
        coords = list(hex_poly.exterior.coords)

        for edge_idx in range(6):
            # Create edge line segment
            p1 = coords[edge_idx]
            p2 = coords[(edge_idx + 1) % 6]
            edge_line = LineString([p1, p2])

            # Check if any river crosses this edge
            for _, river_row in intersecting.iterrows():
                if river_row.geometry.intersects(edge_line):
                    edges_crossed.append(edge_idx)
                    break

        if edges_crossed:
            river_edges[(q, r)] = edges_crossed

    print(f"  Found river crossings in {len(river_edges)} hexes")
    return river_edges


def process_roads(
    grid: HexGrid,
    hex_coords: list[tuple[int, int]],
    roads: gpd.GeoDataFrame,
) -> list[dict]:
    """
    Create road connections between adjacent hexes.

    Returns:
        List of road connections: {from: [q1, r1], to: [q2, r2], type: "highway"}
    """
    print("\nProcessing roads...")
    from hexgrid import hex_neighbors

    connections = []
    seen_pairs = set()

    # Build spatial index for roads
    roads_sindex = roads.sindex

    # Map highway types to our classification
    road_type_map = {
        'motorway': 'highway',
        'trunk': 'highway',
        'primary': 'major_road',
        'secondary': 'minor_road',
    }

    # Create lookup of hex coords
    hex_set = set(hex_coords)

    total = len(hex_coords)
    for i, (q, r) in enumerate(hex_coords):
        if i % 200 == 0:
            print(f"  Processing hex {i}/{total}...")

        hex_poly = grid.hex_polygon(q, r)
        hex_center = grid.axial_to_world(q, r)

        # Find roads that might intersect this hex
        possible_idx = list(roads_sindex.intersection(hex_poly.bounds))
        if not possible_idx:
            continue

        possible_roads = roads.iloc[possible_idx]
        intersecting = possible_roads[possible_roads.intersects(hex_poly)]

        if len(intersecting) == 0:
            continue

        # Check each neighbor
        neighbors = hex_neighbors(q, r)
        for neighbor_q, neighbor_r in neighbors:
            if (neighbor_q, neighbor_r) not in hex_set:
                continue

            # Create canonical pair to avoid duplicates
            pair = tuple(sorted([(q, r), (neighbor_q, neighbor_r)]))
            if pair in seen_pairs:
                continue

            neighbor_poly = grid.hex_polygon(neighbor_q, neighbor_r)

            # Check if any road connects these two hexes
            best_road_type = None
            for _, road_row in intersecting.iterrows():
                if road_row.geometry.intersects(neighbor_poly):
                    highway_type = road_row.get('highway', 'secondary')
                    road_class = road_type_map.get(highway_type, 'minor_road')

                    # Keep best road type (highway > major > minor)
                    if best_road_type is None:
                        best_road_type = road_class
                    elif road_class == 'highway':
                        best_road_type = 'highway'
                    elif road_class == 'major_road' and best_road_type != 'highway':
                        best_road_type = 'major_road'

            if best_road_type:
                seen_pairs.add(pair)
                connections.append({
                    'from': [q, r],
                    'to': [neighbor_q, neighbor_r],
                    'type': best_road_type,
                })

    # Count by type
    type_counts = {}
    for conn in connections:
        t = conn['type']
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"  Found {len(connections)} road connections:")
    for t, count in sorted(type_counts.items()):
        print(f"    {t}: {count}")

    return connections


def process_places(
    grid: HexGrid,
    hex_coords: list[tuple[int, int]],
    places: gpd.GeoDataFrame,
) -> dict[tuple[int, int], dict]:
    """
    Assign cities and towns to hexes.

    Returns:
        Dict mapping (q, r) to city info: {name, name_en, size, population}
    """
    print("\nProcessing places...")
    from shapely.geometry import Point

    hex_cities = {}

    for _, place_row in places.iterrows():
        # Get place location
        if place_row.geometry.geom_type == 'Point':
            px, py = place_row.geometry.x, place_row.geometry.y
        else:
            px, py = place_row.geometry.centroid.x, place_row.geometry.centroid.y

        # Find which hex contains this point
        q, r = grid.world_to_axial(px, py)

        # Verify hex is in our grid
        if (q, r) not in set(hex_coords):
            continue

        # Determine size
        place_type = place_row.get('place', 'town')
        population = place_row.get('population')
        if population:
            try:
                population = int(population)
            except:
                population = None

        if place_type == 'city' or (population and population > 500000):
            size = 'major'
        elif population and population > 100000:
            size = 'large'
        elif population and population > 50000:
            size = 'medium'
        else:
            size = 'small'

        city_info = {
            'name': place_row.get('name', ''),
            'name_en': place_row.get('name:en', ''),
            'size': size,
            'population': population,
        }

        # If hex already has a city, keep the larger one
        if (q, r) in hex_cities:
            existing = hex_cities[(q, r)]
            existing_pop = existing.get('population') or 0
            new_pop = population or 0
            if new_pop <= existing_pop:
                continue

        hex_cities[(q, r)] = city_info

    # Count by size
    size_counts = {}
    for info in hex_cities.values():
        s = info['size']
        size_counts[s] = size_counts.get(s, 0) + 1

    print(f"  Assigned {len(hex_cities)} places to hexes:")
    for s, count in sorted(size_counts.items()):
        print(f"    {s}: {count}")

    return hex_cities


def process_infrastructure(
    grid: HexGrid,
    hex_coords: list[tuple[int, int]],
    infrastructure: gpd.GeoDataFrame,
) -> dict[tuple[int, int], list[str]]:
    """
    Assign airports and ports to hexes.

    Returns:
        Dict mapping (q, r) to list of features: ['airfield', 'port']
    """
    print("\nProcessing infrastructure...")

    hex_features = {}
    hex_set = set(hex_coords)

    for _, infra_row in infrastructure.iterrows():
        # Get location
        geom = infra_row.geometry
        if geom.geom_type == 'Point':
            px, py = geom.x, geom.y
        else:
            px, py = geom.centroid.x, geom.centroid.y

        # Find hex
        q, r = grid.world_to_axial(px, py)

        if (q, r) not in hex_set:
            continue

        # Determine feature type
        infra_type = infra_row.get('infrastructure_type', '')
        if infra_type in ['airport', 'aerodrome']:
            feature = 'airfield'
        elif infra_type == 'port':
            feature = 'port'
        else:
            continue

        if (q, r) not in hex_features:
            hex_features[(q, r)] = []

        if feature not in hex_features[(q, r)]:
            hex_features[(q, r)].append(feature)

    # Count
    airfields = sum(1 for feats in hex_features.values() if 'airfield' in feats)
    ports = sum(1 for feats in hex_features.values() if 'port' in feats)
    print(f"  Found {airfields} hexes with airfields, {ports} hexes with ports")

    return hex_features


def calculate_hex_terrain(
    grid: HexGrid,
    hex_coords: list[tuple[int, int]],
    coastline: gpd.GeoDataFrame,
    dem: rasterio.DatasetReader,
    landcover: gpd.GeoDataFrame,
    rotation_center=None,
    rotation_angle: float = 0,
) -> list[HexData]:
    """Calculate terrain for each hex."""
    print("\nClassifying terrain...")

    # Create transformer for DEM (grid CRS -> WGS84 for sampling)
    transformer_to_wgs84 = Transformer.from_crs(GRID_CRS, DATA_CRS, always_xy=True)

    # Get land polygon (union of all coastline polygons)
    land_polygon = coastline.union_all()

    # Prepare landcover spatial index
    landcover["terrain_type"] = landcover.apply(classify_landcover_type, axis=1)
    lc_sindex = landcover.sindex

    hexes = []
    total = len(hex_coords)

    for i, (q, r) in enumerate(hex_coords):
        if i % 100 == 0:
            print(f"  Processing hex {i}/{total}...")

        # Get hex polygon
        hex_poly = grid.hex_polygon(q, r)

        # === 1. Land/Water classification ===
        intersection = hex_poly.intersection(land_polygon)
        land_fraction = intersection.area / hex_poly.area if hex_poly.area > 0 else 0

        if land_fraction < 0.5:
            # Mostly water
            hex_data = HexData(q=q, r=r, terrain="water", elevation=0)
            hexes.append(hex_data)
            continue

        # === 2. Elevation ===
        # Get hex center and inverse-rotate to original coordinates for DEM sampling
        cx, cy = grid.axial_to_world(q, r)

        # Inverse rotation: rotate by negative angle around same center
        if rotation_angle != 0 and rotation_center is not None:
            from shapely.geometry import Point
            original_point = rotate(Point(cx, cy), -rotation_angle, origin=rotation_center)
            cx_orig, cy_orig = original_point.x, original_point.y
        else:
            cx_orig, cy_orig = cx, cy

        lon, lat = transformer_to_wgs84.transform(cx_orig, cy_orig)

        try:
            # Sample DEM at original (non-rotated) hex center position
            row, col = dem.index(lon, lat)
            if 0 <= row < dem.height and 0 <= col < dem.width:
                elevation = dem.read(1)[row, col]
                if elevation == dem.nodata or elevation < -1000:
                    elevation = 0
            else:
                elevation = 0
        except:
            elevation = 0

        elevation = max(0, float(elevation))
        elev_level, elev_label = get_elevation_band(elevation)

        # === 3. Check land cover first ===
        terrain = "clear"  # Default
        terrain_from_landcover = None

        # Find landcover polygons that intersect this hex
        possible_matches_idx = list(lc_sindex.intersection(hex_poly.bounds))
        if possible_matches_idx:
            possible_matches = landcover.iloc[possible_matches_idx]
            intersecting = possible_matches[possible_matches.intersects(hex_poly)]

            if len(intersecting) > 0:
                # Calculate area of each terrain type in hex
                terrain_areas = {}
                for _, lc_row in intersecting.iterrows():
                    lc_terrain = lc_row["terrain_type"]
                    try:
                        inter_area = hex_poly.intersection(lc_row.geometry).area
                        terrain_areas[lc_terrain] = terrain_areas.get(lc_terrain, 0) + inter_area
                    except:
                        pass

                if terrain_areas:
                    # Get dominant terrain (by area)
                    dominant_terrain = max(terrain_areas, key=terrain_areas.get)
                    dominant_area = terrain_areas[dominant_terrain]

                    # Use if covers at least 15% of hex (lowered threshold)
                    if dominant_area > hex_poly.area * 0.15:
                        terrain_from_landcover = dominant_terrain
                        terrain = dominant_terrain

        # === 4. Apply elevation-based terrain ===
        # Mountain terrain from elevation (>1000m) takes priority over most types
        # but urban always wins
        if elevation >= 1000:
            if terrain_from_landcover != "urban":
                terrain = "mountain"
        elif elevation >= 500 and terrain == "clear":
            # High hills default to rough if no other landcover
            terrain = "rough"

        # === 5. Coastal edges ===
        coastal_edges = []
        if land_fraction < 1.0:
            # Check each edge for water
            edge_midpoints = grid.hex_edge_midpoints(q, r)
            for edge_idx, (ex, ey) in enumerate(edge_midpoints):
                from shapely.geometry import Point
                edge_point = Point(ex, ey)
                if not land_polygon.contains(edge_point):
                    coastal_edges.append(edge_idx)

        # Create hex data
        hex_data = HexData(
            q=q,
            r=r,
            terrain=terrain,
            elevation=int(elevation),
            coastal_edges=coastal_edges,
        )
        hexes.append(hex_data)

    print(f"  Classified {len(hexes)} hexes")
    return hexes


def generate_output(
    grid: HexGrid,
    hexes: list[HexData],
    road_connections: list[dict],
    rotation_angle: float = 0
):
    """Generate output files."""
    print("\nGenerating output...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Count terrain types
    terrain_counts = {}
    for h in hexes:
        terrain_counts[h.terrain] = terrain_counts.get(h.terrain, 0) + 1

    print("  Terrain distribution:")
    for terrain, count in sorted(terrain_counts.items(), key=lambda x: -x[1]):
        pct = count / len(hexes) * 100
        print(f"    {terrain}: {count} ({pct:.1f}%)")

    # Count features
    cities_count = sum(1 for h in hexes if h.city)
    rivers_count = sum(1 for h in hexes if h.river_edges)
    features_count = sum(1 for h in hexes if h.features)
    print(f"\n  Features:")
    print(f"    Hexes with cities: {cities_count}")
    print(f"    Hexes with rivers: {rivers_count}")
    print(f"    Hexes with infrastructure: {features_count}")
    print(f"    Road connections: {len(road_connections)}")

    # Generate SVG
    print("\n  Generating SVG...")
    render_svg(
        grid,
        hexes,
        str(OUTPUT_DIR / "taiwan_hexmap.svg"),
        show_coords=False,
        scale=0.003,
        margin_px=30,
        road_connections=road_connections,
    )

    # Generate JSON - hex data
    print("  Generating JSON...")
    output_data = {
        "metadata": {
            "region": "Taiwan",
            "hex_size_km": HEX_SIZE_M / 1000,
            "crs": GRID_CRS,
            "rotation_degrees": rotation_angle,
            "total_hexes": len(hexes),
        },
        "hexes": [
            {
                "coord": [h.q, h.r],
                "terrain": h.terrain,
                "elevation": h.elevation,
                "coastal_edges": h.coastal_edges,
                "river_edges": h.river_edges,
                "city": h.city,
                "features": h.features,
            }
            for h in hexes
        ],
    }

    with open(OUTPUT_DIR / "taiwan_hexdata.json", "w") as f:
        json.dump(output_data, f, indent=2)

    # Generate JSON - road connections
    connections_data = {
        "metadata": {
            "region": "Taiwan",
            "total_connections": len(road_connections),
        },
        "connections": road_connections,
    }

    with open(OUTPUT_DIR / "taiwan_connections.json", "w") as f:
        json.dump(connections_data, f, indent=2)

    print(f"\n  Output files:")
    print(f"    {OUTPUT_DIR / 'taiwan_hexmap.svg'}")
    print(f"    {OUTPUT_DIR / 'taiwan_hexdata.json'}")
    print(f"    {OUTPUT_DIR / 'taiwan_connections.json'}")


def main():
    print("=" * 60)
    print("Taiwan Hex Map Generator - Phases 2-4")
    print("=" * 60)

    # === Load all data ===
    # Load coastline first (needed for grid bounds calculation)
    coastline = load_coastline()

    # Create grid based on rotated coastline
    grid, hex_coords, rotation_info = create_grid(coastline)
    rotation_center, rotation_angle = rotation_info

    # Rotate coastline data
    print(f"\nRotating vector data by {rotation_angle}°...")
    coastline = rotate_geodataframe(coastline, rotation_angle, rotation_center)

    # Load and rotate terrain data
    dem = load_elevation()
    landcover = load_landcover()
    landcover = rotate_geodataframe(landcover, rotation_angle, rotation_center)

    # Load and rotate linear/point feature data
    rivers = load_rivers()
    rivers = rotate_geodataframe(rivers, rotation_angle, rotation_center)

    roads = load_roads()
    roads = rotate_geodataframe(roads, rotation_angle, rotation_center)

    places = load_places()
    places = rotate_geodataframe(places, rotation_angle, rotation_center)

    infrastructure = load_infrastructure()
    infrastructure = rotate_geodataframe(infrastructure, rotation_angle, rotation_center)

    # === Phase 2: Terrain Classification ===
    print("\n" + "=" * 60)
    print("Phase 2: Terrain Classification")
    print("=" * 60)
    hexes = calculate_hex_terrain(
        grid, hex_coords, coastline, dem, landcover,
        rotation_center, rotation_angle
    )

    # Close DEM (no longer needed)
    dem.close()

    # === Phase 3: Linear Features ===
    print("\n" + "=" * 60)
    print("Phase 3: Linear Features")
    print("=" * 60)

    # Process rivers
    river_edges_map = process_rivers(grid, hex_coords, rivers)

    # Process roads
    road_connections = process_roads(grid, hex_coords, roads)

    # === Phase 4: Point Features ===
    print("\n" + "=" * 60)
    print("Phase 4: Point Features")
    print("=" * 60)

    # Process places
    cities_map = process_places(grid, hex_coords, places)

    # Process infrastructure
    features_map = process_infrastructure(grid, hex_coords, infrastructure)

    # === Merge all data into hex objects ===
    print("\nMerging feature data into hexes...")
    hex_lookup = {(h.q, h.r): h for h in hexes}

    for (q, r), edges in river_edges_map.items():
        if (q, r) in hex_lookup:
            hex_lookup[(q, r)].river_edges = edges

    for (q, r), city_info in cities_map.items():
        if (q, r) in hex_lookup:
            hex_lookup[(q, r)].city = city_info

    for (q, r), features in features_map.items():
        if (q, r) in hex_lookup:
            hex_lookup[(q, r)].features = features

    # === Generate Output ===
    print("\n" + "=" * 60)
    print("Generating Output")
    print("=" * 60)
    generate_output(grid, hexes, road_connections, rotation_angle)

    print("\n" + "=" * 60)
    print("All Phases Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
