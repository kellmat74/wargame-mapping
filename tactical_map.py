"""
tactical_map.py - Tactical-scale hex map generator (250m/hex)

Generates US Army military-style maps with:
- Contour lines (20m interval, 100m index)
- Terrain classification (1:50,000 style)
- Roads, buildings, bridges
- Military grid reference system
"""

import json
import math
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

import numpy as np
import geopandas as gpd
import rasterio
import mgrs
from rasterio.warp import transform_bounds
from shapely.geometry import Point, LineString, Polygon, box, MultiLineString
from shapely.affinity import rotate
from shapely.ops import linemerge
from pyproj import Transformer, CRS
import svgwrite

# === Configuration ===
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
GEOFABRIK_DIR = DATA_DIR / "geofabrik"

# Grid parameters
HEX_SIZE_M = 250  # 250m hex (flat edge to flat edge)
GRID_WIDTH = 47   # hexes wide
GRID_HEIGHT = 26  # hexes tall

# Contour parameters
CONTOUR_INTERVAL = 20      # meters between contour lines
INDEX_CONTOUR_INTERVAL = 100  # meters between index (bold) contours

# Coordinate systems
GRID_CRS = "EPSG:3826"  # TWD97 / TM2 Taiwan (meters)
WGS84 = "EPSG:4326"

# UTM zone - will be set dynamically based on map center location
UTM_CRS = None  # Set by get_utm_crs() based on center coordinates


def get_utm_crs(lon: float, lat: float) -> str:
    """Calculate the correct UTM CRS based on longitude and latitude."""
    # UTM zones are 6 degrees wide, starting from zone 1 at 180°W
    zone = int((lon + 180) / 6) + 1
    # Northern or Southern hemisphere
    if lat >= 0:
        return f"EPSG:326{zone:02d}"  # Northern hemisphere
    else:
        return f"EPSG:327{zone:02d}"  # Southern hemisphere

# MGRS grid parameters
MGRS_GRID_INTERVAL = 1000  # 1km grid lines for 1:50,000 scale

# Terrain colors (OpenTopoMap style)
TERRAIN_COLORS = {
    "water": "#c6ecff",       # Light blue
    "urban": "#a0a0a0",       # Medium grey for built-up areas
    "forest": "#c8e6c8",      # Light green
    "orchard": "#d4e6c8",     # Lighter green with pattern
    "marsh": "#c6ecff",       # Light blue (with pattern overlay)
    "open": "#fffef0",        # Off-white/cream
    "sand": "#fffacd",        # Light yellow
}

# Terrain outlines (OpenTopoMap style)
TERRAIN_OUTLINES = {
    "water": ("#0066cc", 1),      # Dark blue outline, 1m width
    "urban": ("#666666", 1),      # Dark grey outline, 1m width
    "farmland": ("#8b4513", 1),   # Brown outline, 1m width
}

# Line styles - all widths in METERS (1 SVG unit = 1 meter)
CONTOUR_COLOR = "#8b4513"         # Brown
INDEX_CONTOUR_COLOR = "#654321"   # Darker brown
CONTOUR_WIDTH_M = 2               # Regular contour width in meters
INDEX_CONTOUR_WIDTH_M = 5         # Index contour width in meters
CONTOUR_LABEL_INTERVAL_M = 1000   # Distance between elevation labels on index contours
CONTOUR_LABEL_SIZE_M = 25         # Contour label font size (same as hex labels)

ROAD_COLORS = {
    "highway": "#666666",         # Dark grey (OpenTopoMap style)
    "major_road": "#666666",      # Dark grey
    "minor_road": "#666666",      # Dark grey
    "residential": "#ffffff",     # White - residential/unclassified
    "service": "#ffffff",         # White - service roads
    "track": "#996633",           # Brown - tracks/unpaved
}
ROAD_WIDTHS_M = {
    "highway": 10,                # All main roads 10m width
    "major_road": 10,             # All main roads 10m width
    "minor_road": 10,             # All main roads 10m width
    "residential": 10,            # Residential 10m width
    "service": 3,                 # Service road width
    "track": 3,                   # Track width in meters
}
ROAD_OUTLINES = {
    "highway": "#000000",         # Black outline, 2m
    "major_road": "#000000",      # Black outline, 2m
    "minor_road": "#000000",      # Black outline, 2m
    "residential": "#666666",     # Dark grey outline, 2m
    "service": "#888888",         # Light grey outline
    "track": None,                # No outline for tracks
}
ROAD_OUTLINE_WIDTH_M = 2          # Outline width for roads

BUILDING_COLOR = "#666666"        # Dark grey fill
BUILDING_OUTLINE_COLOR = "#000000" # Black outline
BUILDING_OUTLINE_WIDTH_M = 1      # 1m outline width
BRIDGE_COLOR = "#000000"
MGRS_GRID_COLOR = "#a8a8a8"       # Medium grey for MGRS grid
MGRS_EASTING_LABEL_COLOR = "#228b22"   # Light green for easting labels
MGRS_NORTHING_LABEL_COLOR = "#4169e1"  # Light blue for northing labels
MGRS_GRID_WIDTH_M = 3             # MGRS grid line width in meters
MGRS_LABEL_SIZE_M = 80            # MGRS label font size in meters

HEX_GRID_COLOR = "#5e5959"        # Grey for hex grid
HEX_GRID_WIDTH_M = 5              # Hex line width in meters
HEX_GRID_OPACITY = 0.5            # Hex grid opacity (50%)
HEX_LABEL_SIZE_M = 25             # Hex label font size in meters
HEX_MARKER_RADIUS_M = 12          # Hex center circle radius in meters

# Print layout settings (fixed document dimensions)
TRIM_WIDTH_IN = 34.0              # Trim width in inches (final printed width)
TRIM_HEIGHT_IN = 22.0             # Trim height in inches (final printed height)
PLAY_MARGIN_TOP_IN = 0.0          # Margin from trim to top hex row (inches)
PLAY_MARGIN_BOTTOM_IN = 0.0       # Margin from trim to bottom hex row (inches)
PLAY_MARGIN_LEFT_IN = 0.0         # Margin from trim to left hex points (inches)
PLAY_MARGIN_RIGHT_IN = 0.0        # Margin from trim to right hex points (inches)
DATA_MARGIN_IN = 1.25             # Margin outside bleed for data elements (inches)

# Print bleed settings
BLEED_INCHES = 0.125              # Standard bleed for professional printing (1/8")

# Legacy margin (will be calculated dynamically based on scale)
MARGIN_M = 300                    # Default margin around playable area in meters (recalculated)

# Data block styling (in meters)
DATA_FONT_SIZE_M = 35             # Data block font size in meters
DATA_LINE_HEIGHT_M = 45           # Line spacing in meters

# Enhanced feature styles (OpenTopoMap style)
STREAM_COLOR = "#0066cc"          # Blue for streams/ditches
STREAM_WIDTH_M = 10               # Stream line width in meters

COASTLINE_COLOR = "#0055aa"       # Darker blue for coastline
COASTLINE_WIDTH_M = 15            # Coastline width (prominent)

PATH_COLOR = "#666666"            # Dark grey for footpaths
PATH_WIDTH_M = 2                  # Path line width
PATH_DASH = "8,4"                 # Dashed pattern

BARRIER_COLOR = "#333333"         # Dark grey for fences/walls
BARRIER_WIDTH_M = 1.5             # Barrier line width

POWERLINE_COLOR = "#666666"       # Grey for power lines
POWERLINE_WIDTH_M = 1.5           # Power line width
POWERLINE_DASH = "2,6"            # Short dash, long gap

RAILWAY_COLOR = "#333333"         # Dark grey for railways
RAILWAY_WIDTH_M = 4               # Railway line width

TREE_ROW_COLOR = "#228b22"        # Forest green for tree rows
TREE_ROW_WIDTH_M = 3              # Tree row line width

CLIFF_COLOR = "#8b4513"           # Brown for cliffs/embankments
CLIFF_WIDTH_M = 2                 # Cliff line width

BRIDGE_OUTLINE_COLOR = "#000000"  # Black outline for bridges
BRIDGE_FILL_COLOR = "#ffffff"     # White fill for bridges
BRIDGE_WIDTH_M = 10               # Bridge width

FARMLAND_COLOR = "#f5f5dc"        # Beige for farmland
PADDY_COLOR = "#e6ffe6"           # Light green for rice paddies

# New tactical feature colors
MANGROVE_COLOR = "#006644"        # Dark green for mangroves
MANGROVE_OUTLINE = "#004d33"      # Darker outline

WETLAND_COLOR = "#aaddee"         # Light blue-grey for wetlands
WETLAND_OUTLINE = "#88bbcc"       # Slightly darker outline

HEATH_COLOR = "#c8b464"           # Yellow-brown for heath/scrubland
HEATH_OUTLINE = "#b0a050"         # Darker outline

ROCKY_COLOR = "#b0a090"           # Grey-brown for rocky terrain
ROCKY_OUTLINE = "#908070"         # Darker outline

SAND_COLOR = "#fffacd"            # Light yellow for sand/dunes
SAND_OUTLINE = "#e6e0a0"          # Slightly darker

MILITARY_COLOR = "#ff6666"        # Red for military areas
MILITARY_OUTLINE = "#cc0000"      # Dark red outline
MILITARY_WIDTH_M = 3              # Military boundary width

QUARRY_COLOR = "#d0d0d0"          # Light grey for quarries
QUARRY_OUTLINE = "#999999"        # Grey outline

CEMETERY_COLOR = "#aaccaa"        # Pale green for cemeteries
CEMETERY_OUTLINE = "#88aa88"      # Green outline

PLACE_LABEL_SIZE_M = 30           # Settlement name font size
PLACE_LABEL_COLOR = "#333333"     # Dark text for place names

PEAK_MARKER_SIZE_M = 8            # Peak marker size
PEAK_COLOR = "#8b4513"            # Brown for peak markers
PEAK_LABEL_SIZE_M = 20            # Peak elevation label size

CAVE_MARKER_SIZE_M = 10           # Cave marker size
CAVE_COLOR = "#444444"            # Dark grey for caves

DAM_COLOR = "#555555"             # Dark grey for dams
DAM_WIDTH_M = 6                   # Dam line width

AIRFIELD_COLOR = "#888888"        # Grey for airfield surfaces
AIRFIELD_OUTLINE = "#666666"      # Darker outline
RUNWAY_COLOR = "#404040"          # Dark grey for runways
RUNWAY_WIDTH_M = 20               # Runway width

PORT_COLOR = "#666699"            # Blue-grey for port areas
PORT_OUTLINE = "#444477"          # Darker outline
PIER_COLOR = "#8b4513"            # Brown for piers
PIER_WIDTH_M = 4                  # Pier width

TOWER_MARKER_SIZE_M = 6           # Tower marker size
TOWER_COLOR = "#333333"           # Dark grey for towers

FUEL_COLOR = "#cc6600"            # Orange for fuel facilities
FUEL_MARKER_SIZE_M = 8            # Fuel marker size


@dataclass
class TacticalHex:
    """Data for a single tactical hex cell."""
    q: int
    r: int
    terrain: str = "open"
    elevation_min: float = 0
    elevation_max: float = 0
    elevation_avg: float = 0
    has_building: bool = False
    has_bridge: bool = False
    road_types: list = field(default_factory=list)


@dataclass
class MapConfig:
    """Configuration for a tactical map."""
    name: str
    center_lat: float
    center_lon: float
    region: str  # e.g., "51R/TG" or "51R TG" - determines which data folder to use
    country: str = ""  # e.g., "Japan", "Taiwan", "Philippines" - for output folder organization
    rotation_deg: float = 0  # Map rotation in degrees (positive = clockwise)

    # Computed values (set by calculate_bounds)
    center_x: float = 0
    center_y: float = 0
    min_x: float = 0
    max_x: float = 0
    min_y: float = 0
    max_y: float = 0
    # Expanded bounds for data loading (accounts for rotation)
    data_min_x: float = 0
    data_max_x: float = 0
    data_min_y: float = 0
    data_max_y: float = 0
    # UTM CRS for this location (computed from center coordinates)
    utm_crs: str = ""

    def __post_init__(self):
        # Set UTM CRS based on center location
        self.utm_crs = get_utm_crs(self.center_lon, self.center_lat)
        self.calculate_bounds()
        # Normalize region path (handle "51R TG" -> "51R/TG")
        self.region = self._normalize_region_path(self.region)

    def _normalize_region_path(self, region: str) -> str:
        """
        Normalize region path to handle MGRS-style paths.

        Converts:
          "51R TG" -> "51R/TG"
          "51R-TG" -> "51R/TG"
          "51RTG"  -> "51R/TG" (if it looks like MGRS)
          "taiwan" -> "taiwan" (unchanged)
        """
        # Already has slash - use as-is
        if "/" in region:
            return region

        # Check if it looks like MGRS format (e.g., "51R TG", "51R-TG", "51RTG")
        cleaned = region.replace(" ", "").replace("-", "").upper()

        # MGRS format: 2 digits + 1 letter + 2 letters (e.g., "51RTG")
        if len(cleaned) >= 5 and cleaned[:2].isdigit():
            # Find where letters start
            for i, c in enumerate(cleaned):
                if c.isalpha():
                    # Zone + band (e.g., "51R")
                    gzd = cleaned[:i+1]
                    # Square ID (e.g., "TG")
                    square = cleaned[i+1:i+3]
                    if len(square) == 2 and square.isalpha():
                        return f"{gzd}/{square}"
                    break

        # Return as-is (e.g., "taiwan")
        return region

    def calculate_bounds(self):
        """Calculate map bounds from center point."""
        # Transform center to projected CRS
        transformer = Transformer.from_crs(WGS84, GRID_CRS, always_xy=True)
        self.center_x, self.center_y = transformer.transform(
            self.center_lon, self.center_lat
        )

        # Calculate hex geometry
        size = HEX_SIZE_M / math.sqrt(3)  # center to vertex
        col_spacing = 1.5 * size
        row_spacing = HEX_SIZE_M

        # Map dimensions in meters
        # Width: leftmost vertex to rightmost vertex
        width_m = (GRID_WIDTH - 1) * col_spacing + 2 * size
        # Height: For offset hex grids, odd columns are shifted down by row_spacing/2
        # So visual height = rows * row_spacing + row_spacing/2 (for the odd column extension)
        height_m = GRID_HEIGHT * row_spacing + row_spacing / 2

        # Bounds centered on center point
        self.min_x = self.center_x - width_m / 2
        self.max_x = self.center_x + width_m / 2
        self.min_y = self.center_y - height_m / 2
        self.max_y = self.center_y + height_m / 2

        # Calculate expanded bounds for data loading (accounts for rotation)
        if self.rotation_deg != 0:
            # When rotated, we need data for a larger area to fill the visible window
            # The diagonal of the map determines the maximum extent needed
            diagonal = math.sqrt(width_m**2 + height_m**2)
            # Use half diagonal as the radius from center, plus some buffer
            expand_buffer = (diagonal / 2) - min(width_m, height_m) / 2 + 500
            self.data_min_x = self.min_x - expand_buffer
            self.data_max_x = self.max_x + expand_buffer
            self.data_min_y = self.min_y - expand_buffer
            self.data_max_y = self.max_y + expand_buffer
        else:
            # No rotation - data bounds same as map bounds
            self.data_min_x = self.min_x
            self.data_max_x = self.max_x
            self.data_min_y = self.min_y
            self.data_max_y = self.max_y

    def shift_center(self, north_m: float = 0, east_m: float = 0):
        """Shift the map center by given meters."""
        self.center_x += east_m
        self.center_y += north_m

        # Recalculate bounds
        size = HEX_SIZE_M / math.sqrt(3)
        col_spacing = 1.5 * size
        row_spacing = HEX_SIZE_M

        width_m = (GRID_WIDTH - 1) * col_spacing + 2 * size
        # Height: For offset hex grids, odd columns are shifted down by row_spacing/2
        height_m = GRID_HEIGHT * row_spacing + row_spacing / 2

        self.min_x = self.center_x - width_m / 2
        self.max_x = self.center_x + width_m / 2
        self.min_y = self.center_y - height_m / 2
        self.max_y = self.center_y + height_m / 2

        # Recalculate expanded data bounds for rotation
        if self.rotation_deg != 0:
            diagonal = math.sqrt(width_m**2 + height_m**2)
            expand_buffer = (diagonal / 2) - min(width_m, height_m) / 2 + 500
            self.data_min_x = self.min_x - expand_buffer
            self.data_max_x = self.max_x + expand_buffer
            self.data_min_y = self.min_y - expand_buffer
            self.data_max_y = self.max_y + expand_buffer
        else:
            self.data_min_x = self.min_x
            self.data_max_x = self.max_x
            self.data_min_y = self.min_y
            self.data_max_y = self.max_y

        # Update lat/lon
        transformer = Transformer.from_crs(GRID_CRS, WGS84, always_xy=True)
        self.center_lon, self.center_lat = transformer.transform(
            self.center_x, self.center_y
        )

    @property
    def data_path(self) -> Path:
        return DATA_DIR / self.region

    @property
    def output_path(self) -> Path:
        if self.country:
            return OUTPUT_DIR / self.country / self.name
        return OUTPUT_DIR / self.name


class TacticalHexGrid:
    """Hex grid for tactical maps."""

    def __init__(self, config: MapConfig):
        self.config = config
        self.hex_size = HEX_SIZE_M
        self.size = HEX_SIZE_M / math.sqrt(3)  # center to vertex
        self.col_spacing = 1.5 * self.size
        self.row_spacing = HEX_SIZE_M

        # Origin at top-left of grid
        self.origin_x = config.min_x + self.size
        self.origin_y = config.max_y - self.row_spacing / 2

    def axial_to_world(self, q: int, r: int) -> Tuple[float, float]:
        """Convert axial (q, r) to world coordinates."""
        x = self.origin_x + q * self.col_spacing
        y = self.origin_y - r * self.row_spacing - (q % 2) * self.row_spacing * 0.5
        return (x, y)

    def world_to_axial(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to nearest axial hex."""
        q = round((x - self.origin_x) / self.col_spacing)
        adjusted_y = self.origin_y - y
        if q % 2 == 1:
            adjusted_y -= self.row_spacing * 0.5
        r = round(adjusted_y / self.row_spacing)
        return (int(q), int(r))

    def hex_polygon(self, q: int, r: int) -> Polygon:
        """Generate polygon for hex at (q, r)."""
        cx, cy = self.axial_to_world(q, r)
        radius = self.size

        vertices = []
        for i in range(6):
            angle = i * math.pi / 3  # Flat-top: 0°, 60°, 120°, etc.
            vx = cx + radius * math.cos(angle)
            vy = cy + radius * math.sin(angle)
            vertices.append((vx, vy))

        return Polygon(vertices)

    def generate_grid(self) -> List[Tuple[int, int]]:
        """Generate all hex coordinates for the grid."""
        return [(q, r) for q in range(GRID_WIDTH) for r in range(GRID_HEIGHT)]


def load_dem(config: MapConfig) -> rasterio.DatasetReader:
    """Load elevation raster."""
    dem_path = config.data_path / "elevation.tif"
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")
    return rasterio.open(dem_path)


def generate_contours(
    dem: rasterio.DatasetReader,
    config: MapConfig,
    interval: float = CONTOUR_INTERVAL,
) -> List[dict]:
    """
    Generate contour lines from DEM within map bounds.
    Uses expanded bounds for rotation.

    Returns list of {geometry: LineString, elevation: float, is_index: bool}
    """
    from rasterio.windows import from_bounds
    from skimage import measure

    print("Generating contours...")

    # Get window for our data bounds (in DEM's CRS, which is WGS84)
    # Uses expanded bounds to account for rotation
    transformer = Transformer.from_crs(GRID_CRS, dem.crs, always_xy=True)
    dem_min_x, dem_min_y = transformer.transform(config.data_min_x, config.data_min_y)
    dem_max_x, dem_max_y = transformer.transform(config.data_max_x, config.data_max_y)

    # Add buffer for edge contours
    buffer = 0.01  # degrees
    dem_min_x -= buffer
    dem_min_y -= buffer
    dem_max_x += buffer
    dem_max_y += buffer

    # Read DEM window
    try:
        window = from_bounds(
            dem_min_x, dem_min_y, dem_max_x, dem_max_y,
            dem.transform
        )
        data = dem.read(1, window=window)
        window_transform = dem.window_transform(window)
    except Exception as e:
        print(f"  Error reading DEM window: {e}")
        return []

    # Handle nodata
    if dem.nodata is not None:
        data = np.where(data == dem.nodata, np.nan, data)

    # Get elevation range
    valid_data = data[~np.isnan(data)]
    if len(valid_data) == 0:
        print("  No valid elevation data in bounds")
        return []

    elev_min = np.floor(np.nanmin(valid_data) / interval) * interval
    elev_max = np.ceil(np.nanmax(valid_data) / interval) * interval

    print(f"  Elevation range: {elev_min:.0f}m to {elev_max:.0f}m")

    # Generate contour levels
    levels = np.arange(elev_min, elev_max + interval, interval)

    contours = []
    transformer_to_grid = Transformer.from_crs(dem.crs, GRID_CRS, always_xy=True)

    for level in levels:
        # Find contours at this level
        try:
            contour_lines = measure.find_contours(data, level)
        except:
            continue

        is_index = (level % INDEX_CONTOUR_INTERVAL) == 0

        for line in contour_lines:
            if len(line) < 2:
                continue

            # Convert pixel coords to world coords
            coords = []
            for row, col in line:
                # Pixel to CRS coordinates
                x = window_transform.c + col * window_transform.a
                y = window_transform.f + row * window_transform.e

                # Transform to grid CRS
                gx, gy = transformer_to_grid.transform(x, y)
                coords.append((gx, gy))

            if len(coords) >= 2:
                contours.append({
                    "geometry": LineString(coords),
                    "elevation": level,
                    "is_index": is_index,
                })

    print(f"  Generated {len(contours)} contour segments")
    return contours


def classify_terrain(
    grid: TacticalHexGrid,
    hex_coords: List[Tuple[int, int]],
    dem: rasterio.DatasetReader,
    landcover: gpd.GeoDataFrame,
    config: MapConfig,
) -> List[TacticalHex]:
    """Classify terrain for each hex."""
    print("\nClassifying terrain...")

    transformer = Transformer.from_crs(GRID_CRS, dem.crs, always_xy=True)

    # Build landcover spatial index
    if not landcover.empty:
        lc_sindex = landcover.sindex
    else:
        lc_sindex = None

    hexes = []
    total = len(hex_coords)

    for i, (q, r) in enumerate(hex_coords):
        if i % 100 == 0:
            print(f"  Processing hex {i}/{total}...")

        hex_poly = grid.hex_polygon(q, r)
        cx, cy = grid.axial_to_world(q, r)

        # Sample elevation
        lon, lat = transformer.transform(cx, cy)
        try:
            row, col = dem.index(lon, lat)
            if 0 <= row < dem.height and 0 <= col < dem.width:
                elev = dem.read(1)[row, col]
                if elev == dem.nodata or elev < -1000:
                    elev = 0
            else:
                elev = 0
        except:
            elev = 0

        # Classify landcover
        terrain = "open"

        if lc_sindex is not None:
            possible_idx = list(lc_sindex.intersection(hex_poly.bounds))
            if possible_idx:
                possible = landcover.iloc[possible_idx]
                intersecting = possible[possible.intersects(hex_poly)]

                if len(intersecting) > 0:
                    # Find dominant landcover
                    terrain_areas = {}
                    for _, lc_row in intersecting.iterrows():
                        lc_type = classify_military_terrain(lc_row)
                        try:
                            area = hex_poly.intersection(lc_row.geometry).area
                            terrain_areas[lc_type] = terrain_areas.get(lc_type, 0) + area
                        except:
                            pass

                    if terrain_areas:
                        terrain = max(terrain_areas, key=terrain_areas.get)

        # Ocean detection: if no landcover data and elevation at/below sea level,
        # this is likely ocean (SRTM shows ~0m for water surfaces)
        if terrain == "open" and elev <= 1:
            terrain = "water"

        hexes.append(TacticalHex(
            q=q, r=r,
            terrain=terrain,
            elevation_avg=float(elev),
        ))

    print(f"  Classified {len(hexes)} hexes")
    return hexes


def merge_road_segments(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Merge connected road segments by road type to reduce fragment count.

    This uses shapely's linemerge to combine LineStrings that share endpoints,
    grouped by OSM highway type. Results in fewer, longer road segments that
    are easier to edit in vector graphics software.
    """
    if roads.empty:
        return roads

    # Get the highway column (OSM road type)
    if 'highway' not in roads.columns:
        print("    No 'highway' column found, skipping merge")
        return roads

    # Filter to only LineString and MultiLineString geometries
    # (OSM data sometimes includes Polygons for road areas which can't be merged)
    line_mask = roads.geometry.apply(
        lambda g: g is not None and g.geom_type in ('LineString', 'MultiLineString')
    )
    roads_lines = roads[line_mask].copy()

    if roads_lines.empty:
        print("    No LineString geometries to merge")
        return roads

    original_count = len(roads_lines)
    merged_rows = []

    # Group by highway type and merge each group
    for highway_type, group in roads_lines.groupby('highway'):
        if len(group) == 0:
            continue

        # Collect all geometries for this road type
        # Flatten MultiLineStrings into individual LineStrings for merging
        geometries = []
        for geom in group.geometry:
            if geom.geom_type == 'MultiLineString':
                geometries.extend(list(geom.geoms))
            else:
                geometries.append(geom)

        if not geometries:
            continue

        # Use linemerge to combine connected segments
        # linemerge returns a LineString if all segments connect into one line,
        # or a MultiLineString if there are disconnected groups
        try:
            merged = linemerge(geometries)
        except Exception as e:
            print(f"    Warning: Could not merge {highway_type}: {e}")
            # Keep original geometries on error
            for _, row in group.iterrows():
                merged_rows.append(row.to_dict())
            continue

        # Extract individual LineStrings from the result
        if merged.is_empty:
            continue
        elif merged.geom_type == 'LineString':
            result_geoms = [merged]
        elif merged.geom_type == 'MultiLineString':
            result_geoms = list(merged.geoms)
        else:
            # Fallback: keep original geometries
            for _, row in group.iterrows():
                merged_rows.append(row.to_dict())
            continue

        # Create a row for each merged segment
        # Copy attributes from first row of the group
        template_row = group.iloc[0].to_dict()
        for geom in result_geoms:
            row = template_row.copy()
            row['geometry'] = geom
            merged_rows.append(row)

    if not merged_rows:
        return roads

    # Create new GeoDataFrame with merged roads
    merged_gdf = gpd.GeoDataFrame(merged_rows, crs=roads.crs)

    print(f"    Merged {original_count} segments → {len(merged_gdf)} contiguous paths")

    return merged_gdf


def load_roads(config: MapConfig) -> gpd.GeoDataFrame:
    """Load roads within map bounds (uses expanded bounds for rotation)."""
    # Prefer detailed_roads.geojson if available (has all road types)
    detailed_path = config.data_path / "detailed_roads.geojson"
    roads_path = config.data_path / "roads.geojson"

    try:
        if detailed_path.exists():
            print("  Loading detailed roads...")
            roads = gpd.read_file(detailed_path, on_invalid="ignore")
        elif roads_path.exists():
            print("  Loading roads...")
            roads = gpd.read_file(roads_path, on_invalid="ignore")
        else:
            return gpd.GeoDataFrame()

        roads = roads.to_crs(GRID_CRS)

        # Remove invalid geometries
        roads = roads[roads.geometry.notna() & roads.geometry.is_valid]

        # Clip to data bounds (expanded for rotation) with buffer
        map_bounds = box(
            config.data_min_x - 500,
            config.data_min_y - 500,
            config.data_max_x + 500,
            config.data_max_y + 500
        )
        roads = roads[roads.intersects(map_bounds)]
        print(f"    Filtered to {len(roads)} road segments in bounds")

        # Merge connected segments by road type for easier editing
        roads = merge_road_segments(roads)

        return roads
    except Exception as e:
        print(f"    Warning: Error loading roads: {e}")
        return gpd.GeoDataFrame()


def generate_mgrs_grid(config: MapConfig) -> dict:
    """
    Generate MGRS grid lines for the map area.
    Uses expanded data bounds to ensure coverage after rotation.

    Returns dict with:
        - vertical_lines: list of (x_utm, y_start, y_end) for N-S lines
        - horizontal_lines: list of (y_utm, x_start, x_end) for E-W lines
        - labels: list of (x, y, text, orientation) for grid labels
    """
    print(f"\nGenerating MGRS grid (UTM zone: {config.utm_crs})...")

    # Transform data bounds to UTM (uses expanded bounds for rotation)
    transformer_to_utm = Transformer.from_crs(GRID_CRS, config.utm_crs, always_xy=True)
    transformer_from_utm = Transformer.from_crs(config.utm_crs, GRID_CRS, always_xy=True)

    # Get UTM bounds from expanded data bounds
    utm_min_x, utm_min_y = transformer_to_utm.transform(config.data_min_x, config.data_min_y)
    utm_max_x, utm_max_y = transformer_to_utm.transform(config.data_max_x, config.data_max_y)

    # Also transform corners to handle any rotation/skew
    corners_utm = [
        transformer_to_utm.transform(config.data_min_x, config.data_min_y),
        transformer_to_utm.transform(config.data_max_x, config.data_min_y),
        transformer_to_utm.transform(config.data_max_x, config.data_max_y),
        transformer_to_utm.transform(config.data_min_x, config.data_max_y),
    ]

    utm_min_x = min(c[0] for c in corners_utm) - MGRS_GRID_INTERVAL
    utm_max_x = max(c[0] for c in corners_utm) + MGRS_GRID_INTERVAL
    utm_min_y = min(c[1] for c in corners_utm) - MGRS_GRID_INTERVAL
    utm_max_y = max(c[1] for c in corners_utm) + MGRS_GRID_INTERVAL

    print(f"  UTM bounds: E{utm_min_x/1000:.0f}-{utm_max_x/1000:.0f}km, N{utm_min_y/1000:.0f}-{utm_max_y/1000:.0f}km")

    # Calculate grid line positions (round to MGRS_GRID_INTERVAL)
    first_easting = int(utm_min_x / MGRS_GRID_INTERVAL) * MGRS_GRID_INTERVAL
    last_easting = int(utm_max_x / MGRS_GRID_INTERVAL + 1) * MGRS_GRID_INTERVAL
    first_northing = int(utm_min_y / MGRS_GRID_INTERVAL) * MGRS_GRID_INTERVAL
    last_northing = int(utm_max_y / MGRS_GRID_INTERVAL + 1) * MGRS_GRID_INTERVAL

    # Generate vertical lines (constant easting)
    vertical_lines = []
    for easting in range(first_easting, last_easting + 1, MGRS_GRID_INTERVAL):
        # Transform line endpoints to map CRS
        start = transformer_from_utm.transform(easting, utm_min_y)
        end = transformer_from_utm.transform(easting, utm_max_y)
        vertical_lines.append({
            'utm_value': easting,
            'start': start,
            'end': end,
        })

    # Generate horizontal lines (constant northing)
    horizontal_lines = []
    for northing in range(first_northing, last_northing + 1, MGRS_GRID_INTERVAL):
        start = transformer_from_utm.transform(utm_min_x, northing)
        end = transformer_from_utm.transform(utm_max_x, northing)
        horizontal_lines.append({
            'utm_value': northing,
            'start': start,
            'end': end,
        })

    print(f"  Generated {len(vertical_lines)} E-W lines, {len(horizontal_lines)} N-S lines")

    return {
        'vertical_lines': vertical_lines,
        'horizontal_lines': horizontal_lines,
    }


def load_buildings(config: MapConfig) -> gpd.GeoDataFrame:
    """Load buildings within map bounds (uses expanded bounds for rotation)."""
    buildings_path = config.data_path / "buildings.geojson"
    if not buildings_path.exists():
        print("  No buildings data found")
        return gpd.GeoDataFrame()

    print("  Loading buildings...")
    try:
        buildings = gpd.read_file(buildings_path, on_invalid="ignore")
        buildings = buildings.to_crs(GRID_CRS)

        # Remove invalid geometries
        buildings = buildings[buildings.geometry.notna() & buildings.geometry.is_valid]

        # Clip to data bounds (expanded for rotation) with small buffer
        map_bounds = box(
            config.data_min_x - 100,
            config.data_min_y - 100,
            config.data_max_x + 100,
            config.data_max_y + 100
        )
        buildings = buildings[buildings.intersects(map_bounds)]
        print(f"    Filtered to {len(buildings)} buildings in bounds")

        return buildings
    except Exception as e:
        print(f"    Warning: Error loading buildings: {e}")
        return gpd.GeoDataFrame()


def load_optional_features(config: MapConfig, filename: str, feature_name: str) -> gpd.GeoDataFrame:
    """Load optional GeoJSON features within map bounds."""
    feature_path = config.data_path / filename
    if not feature_path.exists():
        return gpd.GeoDataFrame()

    print(f"  Loading {feature_name}...")
    try:
        # Use on_invalid="ignore" to skip malformed geometries
        features = gpd.read_file(feature_path, on_invalid="ignore")
        if len(features) == 0:
            return gpd.GeoDataFrame()

        features = features.to_crs(GRID_CRS)

        # Remove any rows with null/invalid geometries
        features = features[features.geometry.notna() & features.geometry.is_valid]

        # Clip to data bounds with buffer
        map_bounds = box(
            config.data_min_x - 100,
            config.data_min_y - 100,
            config.data_max_x + 100,
            config.data_max_y + 100
        )
        features = features[features.intersects(map_bounds)]
        print(f"    Filtered to {len(features)} {feature_name} in bounds")

        return features
    except Exception as e:
        print(f"    Warning: Error loading {feature_name}: {e}")
        return gpd.GeoDataFrame()


def load_reference_tile_info(config: MapConfig) -> dict:
    """Load reference tile image info for embedding in SVG."""
    import base64

    # Try different naming conventions:
    # 1. reference_tiles_{name}.json (old style, per-map)
    # 2. reference_tiles.json (new style, per-MGRS-square)
    tile_info_path = None
    tile_image_path = None

    # Try map-specific first
    path1_info = config.data_path / f"reference_tiles_{config.name}.json"
    path1_image = config.data_path / f"reference_tiles_{config.name}.png"
    if path1_info.exists() and path1_image.exists():
        tile_info_path = path1_info
        tile_image_path = path1_image

    # Then try generic (MGRS-square level)
    if tile_info_path is None:
        path2_info = config.data_path / "reference_tiles.json"
        path2_image = config.data_path / "reference_tiles.png"
        if path2_info.exists() and path2_image.exists():
            tile_info_path = path2_info
            tile_image_path = path2_image

    if tile_info_path is None:
        print("  No reference tiles found")
        return None

    print("  Loading reference tiles...")
    with open(tile_info_path) as f:
        tile_info = json.load(f)

    # Read image and encode as base64
    with open(tile_image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    tile_info['image_data'] = image_data
    print(f"    Loaded {tile_info['width']}x{tile_info['height']} tile image")

    return tile_info


def download_highres_reference_tiles(config: 'MapConfig', zoom: int = 15) -> dict:
    """Download high-resolution reference tiles for the specific map area.

    Args:
        config: MapConfig with calculated bounds
        zoom: Tile zoom level (15 = ~4.7m/pixel, max for OpenTopoMap outside Europe)

    Returns:
        Tile info dict with embedded base64 image, or None if download fails
    """
    import base64
    import time
    from io import BytesIO

    try:
        import requests
        from PIL import Image
    except ImportError:
        print("  requests or PIL not available, skipping high-res tiles")
        return None

    # Create output directory if needed
    config.output_path.mkdir(parents=True, exist_ok=True)

    output_image = config.output_path / f"{config.name}_reference_highres.png"
    output_info = config.output_path / f"{config.name}_reference_highres.json"

    # Transform map bounds from UTM to WGS84 to check coverage
    transformer = Transformer.from_crs(GRID_CRS, WGS84, always_xy=True)
    needed_min_lon, needed_min_lat = transformer.transform(config.data_min_x, config.data_min_y)
    needed_max_lon, needed_max_lat = transformer.transform(config.data_max_x, config.data_max_y)

    # Check if already downloaded AND covers the needed area
    if output_image.exists() and output_info.exists():
        with open(output_info) as f:
            tile_info = json.load(f)

        # Check if stored bounds cover the needed area (with small tolerance)
        tolerance = 0.001  # ~100m
        stored = tile_info.get('bounds', {})
        bounds_ok = (
            stored.get('north', 0) >= needed_max_lat - tolerance and
            stored.get('south', 0) <= needed_min_lat + tolerance and
            stored.get('east', 0) >= needed_max_lon - tolerance and
            stored.get('west', 0) <= needed_min_lon + tolerance
        )

        if bounds_ok:
            print(f"  Loading existing high-res reference tiles...")
            with open(output_image, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            tile_info['image_data'] = image_data
            print(f"    Loaded {tile_info['width']}x{tile_info['height']} high-res tile image (zoom {tile_info['zoom']})")
            return tile_info
        else:
            print(f"  Stored reference tiles don't cover current bounds, re-downloading...")

    print(f"  Downloading high-res reference tiles (zoom {zoom})...")

    # Use the already computed bounds
    min_lon, min_lat = needed_min_lon, needed_min_lat
    max_lon, max_lat = needed_max_lon, needed_max_lat

    def lat_lon_to_tile(lat, lon, z):
        n = 2 ** z
        x = int((lon + 180) / 360 * n)
        y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
        return x, y

    def tile_to_lat_lon(x, y, z):
        n = 2 ** z
        lon = x / n * 360 - 180
        lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
        return lat, lon

    # Get tile range
    x_min, y_min = lat_lon_to_tile(max_lat, min_lon, zoom)
    x_max, y_max = lat_lon_to_tile(min_lat, max_lon, zoom)

    num_tiles = (x_max - x_min + 1) * (y_max - y_min + 1)
    print(f"    Need {num_tiles} tiles for map area...")

    # Reduce zoom if too many tiles
    if num_tiles > 400:
        print(f"    Too many tiles, reducing zoom level to {zoom - 1}")
        return download_highres_reference_tiles(config, zoom - 1)

    # Download tiles from OpenTopoMap
    tiles = {}
    tile_size = 256
    downloaded = 0
    failed = 0
    not_found = 0

    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            url = f"https://tile.opentopomap.org/{zoom}/{x}/{y}.png"
            try:
                response = requests.get(url, timeout=30, headers={
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                    'Accept': 'image/png,image/*',
                })
                if response.status_code == 200:
                    tiles[(x, y)] = Image.open(BytesIO(response.content))
                    downloaded += 1
                elif response.status_code == 404:
                    not_found += 1
                else:
                    failed += 1
                    if failed <= 3:
                        print(f"      Tile {x},{y}: HTTP {response.status_code}")
                time.sleep(0.1)  # Be nice to tile server
            except requests.exceptions.Timeout:
                failed += 1
                if failed <= 3:
                    print(f"      Tile {x},{y}: timeout")
            except Exception as e:
                failed += 1
                if failed <= 3:
                    print(f"      Tile {x},{y}: {e}")

    # If most tiles are 404, try lower zoom (coverage doesn't exist at this zoom)
    if not_found > num_tiles * 0.5 and zoom > 12:
        print(f"    {not_found}/{num_tiles} tiles not found, trying zoom {zoom - 1}...")
        return download_highres_reference_tiles(config, zoom - 1)

    print(f"    Downloaded {downloaded}/{num_tiles} tiles")

    if not tiles:
        print("    No tiles downloaded, falling back to MGRS tiles")
        return None

    # Stitch tiles
    width = (x_max - x_min + 1) * tile_size
    height = (y_max - y_min + 1) * tile_size
    stitched = Image.new('RGB', (width, height))

    for (x, y), tile in tiles.items():
        px = (x - x_min) * tile_size
        py = (y - y_min) * tile_size
        stitched.paste(tile, (px, py))

    # Save image
    stitched.save(output_image, quality=95)

    # Calculate bounds of the stitched tiles
    nw_lat, nw_lon = tile_to_lat_lon(x_min, y_min, zoom)
    se_lat, se_lon = tile_to_lat_lon(x_max + 1, y_max + 1, zoom)

    tile_info = {
        "zoom": zoom,
        "width": width,
        "height": height,
        "bounds": {
            "north": nw_lat,
            "south": se_lat,
            "west": nw_lon,
            "east": se_lon
        },
        "source": "OpenTopoMap"
    }

    # Save info
    with open(output_info, 'w') as f:
        json.dump(tile_info, f, indent=2)

    # Encode for embedding
    with open(output_image, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    tile_info['image_data'] = image_data

    print(f"    Saved {width}x{height} high-res tile image")

    return tile_info


def classify_road_type(highway: str) -> str:
    """Map OSM highway types to military road classification."""
    if highway in ["motorway", "motorway_link", "trunk", "trunk_link"]:
        return "highway"
    elif highway in ["primary", "primary_link", "secondary", "secondary_link"]:
        return "major_road"
    elif highway in ["tertiary", "tertiary_link"]:
        return "minor_road"
    elif highway in ["residential", "unclassified", "living_street"]:
        return "residential"
    elif highway in ["service"]:
        return "service"
    elif highway in ["track"]:
        return "track"
    return "residential"  # Default to residential for unknown types


def classify_military_terrain(row) -> str:
    """Map OSM landuse/natural to military terrain types."""
    landuse = str(row.get("landuse", "") or "")
    natural = str(row.get("natural", "") or "")

    # Water
    if natural == "water" or landuse in ["reservoir", "basin"]:
        return "water"

    # Urban/built-up
    if landuse in ["residential", "commercial", "industrial", "retail"]:
        return "urban"

    # Forest
    if landuse == "forest" or natural == "wood":
        return "forest"

    # Orchard
    if landuse in ["orchard", "vineyard"]:
        return "orchard"

    # Marsh
    if natural in ["wetland", "marsh", "swamp"]:
        return "marsh"

    # Farmland -> open
    if landuse in ["farmland", "meadow", "grass"]:
        return "open"

    return "open"


def render_tactical_svg(
    grid: TacticalHexGrid,
    hexes: List[TacticalHex],
    contours: List[dict],
    roads: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    landcover: gpd.GeoDataFrame,
    mgrs_grid: dict,
    config: MapConfig,
    output_path: Path,
    enhanced_features: dict = None,
    dem: rasterio.DatasetReader = None,
) -> None:
    """Render tactical map to SVG with 1 SVG unit = 1 meter."""
    print("\nRendering SVG...")

    # === NEW LAYOUT: Fixed 34" x 22" trim with data elements outside bleed ===

    # Hex grid dimensions in meters (at current HEX_SIZE_M)
    # config.min/max already define the VISUAL bounds (including vertices)
    # calculated in calculate_bounds() as: width_m = (GRID_WIDTH-1)*col_spacing + 2*size
    # So we don't need additional vertex extensions - they're already included.
    hex_vertex_extend_h = 0  # No additional extension needed - bounds include vertices
    hex_vertex_extend_v = 0  # No additional extension needed - bounds include vertices

    # Grid dimensions based on hex CENTERS (for scaling - maximizes hex size)
    hex_grid_center_width = config.max_x - config.min_x
    hex_grid_center_height = config.max_y - config.min_y

    # Visual bounds including vertex extensions (for positioning only)
    hex_grid_width = hex_grid_center_width + 2 * hex_vertex_extend_h
    hex_grid_height = hex_grid_center_height + 2 * hex_vertex_extend_v

    # Play area (where hexes fit) = trim minus top/bottom margins
    play_width_in = TRIM_WIDTH_IN - PLAY_MARGIN_LEFT_IN - PLAY_MARGIN_RIGHT_IN
    play_height_in = TRIM_HEIGHT_IN - PLAY_MARGIN_TOP_IN - PLAY_MARGIN_BOTTOM_IN

    print(f"  Trim size: {TRIM_WIDTH_IN}\" x {TRIM_HEIGHT_IN}\"")
    print(f"  Play area: {play_width_in}\" x {play_height_in}\"")

    # Calculate scale to fit hex grid into play area (uniform scaling)
    # We want to maximize fill while keeping hexes undistorted
    scale_by_width = play_width_in / (hex_grid_width / 25.4)   # meters to inches
    scale_by_height = play_height_in / (hex_grid_height / 25.4)

    # Actually, let's calculate meters_per_inch directly
    # meters_per_inch = hex_grid_dimension_m / play_dimension_in
    meters_per_inch_by_width = hex_grid_width / play_width_in
    meters_per_inch_by_height = hex_grid_height / play_height_in

    # Use the larger scale (more meters per inch = smaller hexes) to ensure fit
    meters_per_inch = max(meters_per_inch_by_width, meters_per_inch_by_height)

    # Calculate actual hex grid size in inches at this scale
    actual_grid_width_in = hex_grid_width / meters_per_inch
    actual_grid_height_in = hex_grid_height / meters_per_inch

    # Calculate centering offsets (to center hex grid in play area)
    center_offset_x_in = (play_width_in - actual_grid_width_in) / 2
    center_offset_y_in = (play_height_in - actual_grid_height_in) / 2

    print(f"  Hex grid: {actual_grid_width_in:.2f}\" x {actual_grid_height_in:.2f}\"")
    print(f"  Scale: 1\" = {meters_per_inch:.1f}m")
    print(f"  Centering offset: ({center_offset_x_in:.3f}\", {center_offset_y_in:.3f}\")")

    # Convert layout dimensions to meters (for SVG viewBox, which uses 1 unit = 1 meter)
    trim_width_m = TRIM_WIDTH_IN * meters_per_inch
    trim_height_m = TRIM_HEIGHT_IN * meters_per_inch
    bleed_m = BLEED_INCHES * meters_per_inch
    data_margin_m = DATA_MARGIN_IN * meters_per_inch
    play_margin_top_m = PLAY_MARGIN_TOP_IN * meters_per_inch
    play_margin_bottom_m = PLAY_MARGIN_BOTTOM_IN * meters_per_inch
    play_margin_left_m = PLAY_MARGIN_LEFT_IN * meters_per_inch
    play_margin_right_m = PLAY_MARGIN_RIGHT_IN * meters_per_inch
    center_offset_x_m = center_offset_x_in * meters_per_inch
    center_offset_y_m = center_offset_y_in * meters_per_inch

    # viewBox dimensions: trim area only (data elements are outside, not in viewBox)
    # Actually, we need viewBox to include bleed + data margin for the full document
    viewbox_width = trim_width_m
    viewbox_height = trim_height_m

    # Full document dimensions including bleed and data margin
    doc_width_in = TRIM_WIDTH_IN + 2 * BLEED_INCHES + 2 * DATA_MARGIN_IN
    doc_height_in = TRIM_HEIGHT_IN + 2 * BLEED_INCHES + 2 * DATA_MARGIN_IN
    doc_width_m = doc_width_in * meters_per_inch
    doc_height_m = doc_height_in * meters_per_inch

    # viewBox includes everything (data margin + bleed + trim)
    viewbox_width_with_bleed = doc_width_m
    viewbox_height_with_bleed = doc_height_m

    # Offsets for positioning content within the viewBox
    # Content origin starts after data_margin + bleed
    content_offset_m = data_margin_m + bleed_m

    print(f"  Full document: {doc_width_in:.2f}\" x {doc_height_in:.2f}\"")
    print(f"    (includes {DATA_MARGIN_IN}\" data margin + {BLEED_INCHES}\" bleed on each side)")

    # Create SVG with exact dimensions (data margin + bleed + trim + bleed + data margin)
    dwg = svgwrite.Drawing(
        str(output_path),
        size=(f"{doc_width_in}in", f"{doc_height_in}in"),
        viewBox=f"0 0 {viewbox_width_with_bleed} {viewbox_height_with_bleed}",
    )
    # No aspect ratio adjustment needed - dimensions match exactly
    dwg["preserveAspectRatio"] = "none"

    # Create a clipPath to constrain map content to trim area (excludes bleed and data margin)
    # Trim area starts at (data_margin + bleed) from document edge
    clip_path = dwg.defs.add(dwg.clipPath(id="viewbox-clip"))
    clip_path.add(dwg.rect((content_offset_m, content_offset_m), (trim_width_m, trim_height_m)))

    # Create an expanded clip-path for rotated reference content (covers rotation corners)
    if config.rotation_deg != 0:
        rotation_buffer = max(trim_width_m, trim_height_m) * 0.5
        clip_path_rotated = dwg.defs.add(dwg.clipPath(id="viewbox-clip-rotated"))
        clip_path_rotated.add(dwg.rect(
            (content_offset_m - rotation_buffer, content_offset_m - rotation_buffer),
            (trim_width_m + 2 * rotation_buffer, trim_height_m + 2 * rotation_buffer)
        ))

    # Create raster-based patterns for better Affinity Designer compatibility
    # Generate PNG textures and embed as base64
    import base64
    from io import BytesIO
    try:
        from PIL import Image, ImageDraw

        # Tree pattern - green circles on light green background
        tree_size = 64
        tree_img = Image.new('RGBA', (tree_size, tree_size), (200, 230, 200, 255))  # Light green
        tree_draw = ImageDraw.Draw(tree_img)
        # Draw tree circles
        tree_draw.ellipse([8, 8, 24, 24], fill=(34, 139, 34, 255))  # Forest green
        tree_draw.ellipse([40, 40, 56, 56], fill=(34, 139, 34, 255))
        tree_draw.ellipse([8, 40, 20, 52], fill=(46, 139, 46, 255))  # Slightly different green
        tree_draw.ellipse([44, 8, 56, 20], fill=(46, 139, 46, 255))

        # Convert to base64
        tree_buffer = BytesIO()
        tree_img.save(tree_buffer, format='PNG')
        tree_base64 = base64.b64encode(tree_buffer.getvalue()).decode('utf-8')

        # Create SVG pattern with embedded PNG
        tree_pattern = dwg.pattern(
            id="tree_pattern",
            insert=(0, 0),
            size=(tree_size, tree_size),
            patternUnits="userSpaceOnUse"
        )
        tree_pattern.add(dwg.image(
            href=f"data:image/png;base64,{tree_base64}",
            insert=(0, 0),
            size=(tree_size, tree_size)
        ))
        dwg.defs.add(tree_pattern)

        # Farmland pattern - horizontal lines on beige background
        farm_size = 64
        farm_img = Image.new('RGBA', (farm_size, farm_size), (245, 245, 220, 255))  # Beige
        farm_draw = ImageDraw.Draw(farm_img)
        # Draw horizontal lines
        farm_draw.line([(0, 20), (farm_size, 20)], fill=(196, 180, 152, 255), width=3)
        farm_draw.line([(0, 42), (farm_size, 42)], fill=(196, 180, 152, 255), width=3)

        # Convert to base64
        farm_buffer = BytesIO()
        farm_img.save(farm_buffer, format='PNG')
        farm_base64 = base64.b64encode(farm_buffer.getvalue()).decode('utf-8')

        # Create SVG pattern with embedded PNG
        farmland_pattern = dwg.pattern(
            id="farmland_pattern",
            insert=(0, 0),
            size=(farm_size, farm_size),
            patternUnits="userSpaceOnUse"
        )
        farmland_pattern.add(dwg.image(
            href=f"data:image/png;base64,{farm_base64}",
            insert=(0, 0),
            size=(farm_size, farm_size)
        ))
        dwg.defs.add(farmland_pattern)

        patterns_available = True
        print("  Created raster-based patterns for forest/farmland")

    except ImportError:
        print("  PIL not available, skipping pattern creation")
        patterns_available = False

    def to_svg(x: float, y: float) -> Tuple[float, float]:
        """Convert world coordinates (meters) to SVG coordinates.

        SVG Y-axis is inverted (increases downward), so we flip Y.
        Coordinates are offset to position hex grid within the play area.

        Layout (from edge of document):
        - data_margin_m: space for reference data (outside print area)
        - bleed_m: print bleed
        - play_margin: space between trim edge and hex grid
        - center_offset: centering within play area

        Note: Vertex extensions are already accounted for in the scaling calculation
        (hex_grid_width/height include them), so we don't add them as offsets here.
        The hex centers are offset by the vertex extension so vertices reach the edges.
        """
        # X: offset by data margin, bleed, play margin (left), centering, and vertex extension
        # The vertex extension shifts all hex centers right so leftmost vertex aligns with play area left edge
        svg_x = (x - config.min_x) + hex_vertex_extend_h + content_offset_m + play_margin_left_m + center_offset_x_m
        # Y: inverted, offset by data margin, bleed, play margin (top), centering, and vertex extension
        svg_y = (config.max_y - y) + hex_vertex_extend_v + content_offset_m + play_margin_top_m + center_offset_y_m
        return (svg_x, svg_y)

    # Create playable area boundary (union of all hex polygons)
    print("  Creating playable area boundary...")
    from shapely.ops import unary_union
    hex_polys = [grid.hex_polygon(h.q, h.r) for h in hexes]
    playable_area = unary_union(hex_polys)

    # === Create layer groups ===
    # Reference layer (hidden by default - for artist reference)
    # Use expanded clip-path when rotated to fill corners
    layer_reference_tiles = dwg.g(id="Reference_Tiles", visibility="hidden")
    if config.rotation_deg != 0:
        layer_reference_tiles["clip-path"] = "url(#viewbox-clip-rotated)"
    else:
        layer_reference_tiles["clip-path"] = "url(#viewbox-clip)"

    # Base layers
    layer_background = dwg.g(id="Background")
    layer_terrain_open = dwg.g(id="Terrain_Open")

    # Ocean layer (rendered BEFORE terrain so terrain covers island)
    layer_ocean = dwg.g(id="Ocean")

    # Terrain polygons
    layer_terrain_water = dwg.g(id="Terrain_Water")
    layer_terrain_marsh = dwg.g(id="Terrain_Marsh")
    layer_terrain_forest = dwg.g(id="Terrain_Forest")
    layer_terrain_orchard = dwg.g(id="Terrain_Orchard")
    layer_terrain_urban = dwg.g(id="Terrain_Urban")
    layer_farmland = dwg.g(id="Farmland")
    layer_waterways_area = dwg.g(id="Waterways_Area")

    # New terrain polygons
    layer_mangrove = dwg.g(id="Mangrove")
    layer_wetland = dwg.g(id="Wetland")
    layer_heath = dwg.g(id="Heath")
    layer_rocky = dwg.g(id="Rocky_Terrain")
    layer_sand = dwg.g(id="Sand")
    layer_military = dwg.g(id="Military")
    layer_quarries = dwg.g(id="Quarries")
    layer_cemeteries = dwg.g(id="Cemeteries")
    layer_airfields = dwg.g(id="Airfields")
    layer_ports = dwg.g(id="Ports")

    # Point features
    layer_places = dwg.g(id="Places")
    layer_peaks = dwg.g(id="Peaks")
    layer_caves = dwg.g(id="Caves")
    layer_towers = dwg.g(id="Towers")
    layer_fuel = dwg.g(id="Fuel_Infrastructure")

    # Linear features (new)
    layer_dams = dwg.g(id="Dams")

    # Linear features
    layer_streams = dwg.g(id="Streams")
    layer_coastline = dwg.g(id="Coastline")
    layer_contours_regular = dwg.g(id="Contours_Regular")
    layer_contours_index = dwg.g(id="Contours_Index")
    layer_contour_labels = dwg.g(id="Contour_Labels")
    layer_cliffs = dwg.g(id="Cliffs")
    layer_tree_rows = dwg.g(id="Tree_Rows")
    layer_barriers = dwg.g(id="Barriers")
    layer_powerlines = dwg.g(id="Powerlines")
    layer_railways = dwg.g(id="Railways")
    layer_paths = dwg.g(id="Paths")

    # Road layers (separate by type, bottom to top)
    layer_roads_service = dwg.g(id="Roads_Service")
    layer_roads_residential = dwg.g(id="Roads_Residential")
    layer_roads_track = dwg.g(id="Roads_Track")
    layer_roads_minor = dwg.g(id="Roads_Minor")
    layer_roads_major = dwg.g(id="Roads_Major")
    layer_roads_highway = dwg.g(id="Roads_Highway")

    layer_bridges = dwg.g(id="Bridges")

    # Structures
    layer_buildings = dwg.g(id="Buildings")

    # Grid and labels
    layer_hex_grid = dwg.g(id="Hex_Grid")
    layer_hex_markers = dwg.g(id="Hex_Markers")
    layer_hex_labels = dwg.g(id="Hex_Labels")
    layer_mgrs_grid = dwg.g(id="MGRS_Grid")
    layer_mgrs_labels = dwg.g(id="MGRS_Labels")
    layer_map_data = dwg.g(id="Map_Data")
    layer_compass = dwg.g(id="Compass_Rose")
    layer_out_of_play_frame = dwg.g(id="Out_Of_Play_Frame")

    # Map terrain types to their layer groups
    terrain_layers = {
        "water": layer_terrain_water,
        "marsh": layer_terrain_marsh,
        "forest": layer_terrain_forest,
        "orchard": layer_terrain_orchard,
        "urban": layer_terrain_urban,
    }

    # Layer 0: Dark grey background (out of play area)
    # Background fills entire document including bleed area
    layer_background.add(dwg.rect((0, 0), (viewbox_width_with_bleed, viewbox_height_with_bleed), fill="#404040"))

    # === Early coastline/island detection ===
    # Detect islands BEFORE rendering terrain so we can use them as base fill
    detected_island_polygons = []
    coastline_gdf = enhanced_features.get('coastline') if enhanced_features else None
    if coastline_gdf is not None and not coastline_gdf.empty:
        print("  Detecting islands from coastline...")
        from shapely.ops import linemerge
        from shapely.geometry import Polygon as ShapelyPolygon

        # Collect all coastline geometries
        coast_lines = []
        for _, row in coastline_gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            if geom.geom_type == 'LineString':
                coast_lines.append(geom)
            elif geom.geom_type == 'MultiLineString':
                coast_lines.extend(list(geom.geoms))
            elif geom.geom_type == 'Polygon':
                detected_island_polygons.append(geom)
            elif geom.geom_type == 'MultiPolygon':
                detected_island_polygons.extend(list(geom.geoms))

        # Merge coastline segments and check if they form closed rings (islands)
        if coast_lines:
            merged_coast = linemerge(coast_lines)
            merged_lines = []
            if merged_coast.geom_type == 'LineString':
                merged_lines = [merged_coast]
            elif merged_coast.geom_type == 'MultiLineString':
                merged_lines = list(merged_coast.geoms)

            for line in merged_lines:
                if line.is_ring:
                    detected_island_polygons.append(ShapelyPolygon(line.coords))

        if detected_island_polygons:
            print(f"    Found {len(detected_island_polygons)} island polygon(s)")

    # Layer 1: Base terrain fill
    # If we have islands, render them as base fill (ocean is below)
    # Otherwise, render full playable area as base fill
    if detected_island_polygons:
        print("  Rendering island polygons as base terrain...")
        # Clip islands to playable area and render
        for island in detected_island_polygons:
            clipped = island.intersection(playable_area)
            if clipped.is_empty:
                continue
            polys_to_render = []
            if clipped.geom_type == 'Polygon':
                polys_to_render = [clipped]
            elif clipped.geom_type == 'MultiPolygon':
                polys_to_render = list(clipped.geoms)

            for poly in polys_to_render:
                coords = list(poly.exterior.coords)
                svg_points = [to_svg(x, y) for x, y in coords]
                layer_terrain_open.add(dwg.polygon(
                    points=svg_points,
                    fill=TERRAIN_COLORS["open"],
                    stroke="none",
                ))
    else:
        # No islands - use full playable area (mainland map)
        if playable_area.geom_type == "Polygon":
            coords = list(playable_area.exterior.coords)
            svg_points = [to_svg(x, y) for x, y in coords]
            layer_terrain_open.add(dwg.polygon(
                points=svg_points,
                fill=TERRAIN_COLORS["open"],
                stroke="none",
            ))
        elif playable_area.geom_type == "MultiPolygon":
            for poly in playable_area.geoms:
                coords = list(poly.exterior.coords)
                svg_points = [to_svg(x, y) for x, y in coords]
                layer_terrain_open.add(dwg.polygon(
                    points=svg_points,
                    fill=TERRAIN_COLORS["open"],
                    stroke="none",
                ))

    # Layer 2: Continuous terrain polygons (unclipped - frame will mask edges)
    if not landcover.empty:
        print("  Rendering terrain polygons...")

        for _, lc_row in landcover.iterrows():
            geom = lc_row.geometry
            terrain_type = classify_military_terrain(lc_row)

            if terrain_type == "open":
                continue  # Already filled as base

            fill = TERRAIN_COLORS.get(terrain_type, TERRAIN_COLORS["open"])
            target_layer = terrain_layers.get(terrain_type, layer_terrain_open)

            # Handle different geometry types
            if geom.geom_type == "Polygon":
                polys = [geom]
            elif geom.geom_type == "MultiPolygon":
                polys = list(geom.geoms)
            else:
                continue

            for poly in polys:
                coords = list(poly.exterior.coords)
                svg_points = [to_svg(x, y) for x, y in coords]

                # Get outline style if defined
                outline_style = TERRAIN_OUTLINES.get(terrain_type)
                stroke_color = outline_style[0] if outline_style else "none"
                stroke_width = outline_style[1] if outline_style else 0

                # Use solid fill - patterns via url() don't work in Affinity Designer
                fill_val = fill

                target_layer.add(dwg.polygon(
                    points=svg_points,
                    fill=fill_val,
                    stroke=stroke_color,
                    stroke_width=stroke_width,
                ))

    # Render ocean polygon
    # For island maps: ocean is a simple rectangle below the terrain (terrain covers islands)
    # For mainland maps: use complex polygonize approach with coastlines
    if detected_island_polygons:
        # Island map - render simple ocean rectangle (terrain_open will cover land)
        print("  Rendering ocean for island map...")
        buffer = 2000  # 2km buffer beyond data bounds
        ocean_bounds = box(
            config.data_min_x - buffer,
            config.data_min_y - buffer,
            config.data_max_x + buffer,
            config.data_max_y + buffer
        )
        water_fill = TERRAIN_COLORS["water"]
        ext_coords = list(ocean_bounds.exterior.coords)
        ext_svg = [to_svg(x, y) for x, y in ext_coords]
        layer_ocean.add(dwg.polygon(
            points=ext_svg,
            fill=water_fill,
            stroke="none",
        ))
        print(f"    Ocean rendered as simple rectangle (islands detected earlier)")

    elif coastline_gdf is not None and not coastline_gdf.empty and dem is not None:
        # Mainland map - use complex polygonize approach
        print("  Creating ocean polygon from coastline (mainland approach)...")
        try:
            from shapely.ops import polygonize, linemerge, unary_union, snap
            from shapely.geometry import MultiLineString, GeometryCollection

            # Create a large bounding box that extends beyond the map
            buffer = 2000  # 2km buffer beyond data bounds
            ocean_bounds = box(
                config.data_min_x - buffer,
                config.data_min_y - buffer,
                config.data_max_x + buffer,
                config.data_max_y + buffer
            )
            bound_ring = ocean_bounds.exterior

            # Collect coastline geometries (only lines for mainland)
            coast_lines = []
            for _, row in coastline_gdf.iterrows():
                geom = row.geometry
                if geom is None:
                    continue
                if geom.geom_type == 'LineString':
                    coast_lines.append(geom)
                elif geom.geom_type == 'MultiLineString':
                    coast_lines.extend(list(geom.geoms))

            if coast_lines:
                # Merge connected coastline segments
                merged_coast = linemerge(coast_lines)

                # Clip coastline to our bounds
                clipped_coast = merged_coast.intersection(ocean_bounds)

                if not clipped_coast.is_empty:
                    # Get individual coastline segments
                    coast_segments = []
                    if clipped_coast.geom_type == 'LineString':
                        coast_segments.append(clipped_coast)
                    elif clipped_coast.geom_type == 'MultiLineString':
                        coast_segments.extend(list(clipped_coast.geoms))
                    elif clipped_coast.geom_type == 'GeometryCollection':
                        for g in clipped_coast.geoms:
                            if g.geom_type == 'LineString':
                                coast_segments.append(g)
                            elif g.geom_type == 'MultiLineString':
                                coast_segments.extend(list(g.geoms))

                    # For each coastline segment, extend endpoints to the boundary if needed
                    extended_segments = []
                    snap_tolerance = 100  # 100m snap tolerance

                    for seg in coast_segments:
                        if len(seg.coords) < 2:
                            continue
                        coords = list(seg.coords)
                        start_pt = Point(coords[0])
                        end_pt = Point(coords[-1])

                        # Check if endpoints are on the boundary (within tolerance)
                        start_on_boundary = bound_ring.distance(start_pt) < snap_tolerance
                        end_on_boundary = bound_ring.distance(end_pt) < snap_tolerance

                        # If endpoints aren't on boundary, extend them
                        new_coords = list(coords)
                        if not start_on_boundary:
                            # Project start point onto boundary
                            nearest_pt = bound_ring.interpolate(bound_ring.project(start_pt))
                            new_coords.insert(0, (nearest_pt.x, nearest_pt.y))
                        if not end_on_boundary:
                            # Project end point onto boundary
                            nearest_pt = bound_ring.interpolate(bound_ring.project(end_pt))
                            new_coords.append((nearest_pt.x, nearest_pt.y))

                        extended_segments.append(LineString(new_coords))

                    # Snap all lines to the boundary to ensure proper intersection
                    all_lines = [LineString(list(bound_ring.coords))]
                    for seg in extended_segments:
                        snapped = snap(seg, bound_ring, snap_tolerance)
                        all_lines.append(snapped)

                    # Use polygonize to create polygons from the line network
                    # unary_union will automatically node the lines at intersections
                    merged_lines = unary_union(all_lines)
                    polygons = list(polygonize(merged_lines))

                    print(f"    Polygonize created {len(polygons)} polygon(s)")
                    print(f"    Extended segments: {len(extended_segments)}, coast segments: {len(coast_segments)}")

                    # Find ocean polygon(s) by checking elevation at multiple sample points
                    ocean_polys = []
                    transformer_to_dem = Transformer.from_crs(GRID_CRS, dem.crs, always_xy=True)
                    dem_data = dem.read(1)

                    for i, poly in enumerate(polygons):
                        if not poly.is_valid or poly.is_empty:
                            continue

                        # Sample multiple points inside polygon for more robust detection
                        sample_elevations = []

                        # Get representative point and some boundary points
                        test_points = [poly.representative_point()]

                        # Add centroid if it's inside the polygon
                        centroid = poly.centroid
                        if poly.contains(centroid):
                            test_points.append(centroid)

                        # Sample along a grid within the polygon bounds
                        bounds = poly.bounds
                        step = min(bounds[2] - bounds[0], bounds[3] - bounds[1]) / 5
                        if step > 100:  # At least 100m step
                            for x in [bounds[0] + step, (bounds[0] + bounds[2]) / 2, bounds[2] - step]:
                                for y in [bounds[1] + step, (bounds[1] + bounds[3]) / 2, bounds[3] - step]:
                                    pt = Point(x, y)
                                    if poly.contains(pt):
                                        test_points.append(pt)

                        # Check elevation at each sample point
                        for test_point in test_points:
                            dem_x, dem_y = transformer_to_dem.transform(test_point.x, test_point.y)
                            try:
                                row_idx, col_idx = dem.index(dem_x, dem_y)
                                if 0 <= row_idx < dem.height and 0 <= col_idx < dem.width:
                                    elev = dem_data[row_idx, col_idx]
                                    sample_elevations.append(elev)
                            except Exception:
                                pass

                        # Determine if this is ocean based on majority of samples
                        if sample_elevations:
                            ocean_samples = sum(1 for e in sample_elevations if e <= 1)
                            avg_elev = sum(sample_elevations) / len(sample_elevations)
                            is_ocean = ocean_samples > len(sample_elevations) / 2

                            print(f"      Polygon {i}: area={poly.area/1e6:.2f}km², "
                                  f"samples={len(sample_elevations)}, ocean_samples={ocean_samples}, "
                                  f"avg_elev={avg_elev:.1f}m, is_ocean={is_ocean}")

                            if is_ocean:
                                ocean_polys.append(poly)
                        else:
                            # No elevation samples - polygon is outside DEM coverage
                            # If it touches the boundary and is large, assume it's ocean
                            touches_boundary = poly.intersects(bound_ring)
                            area_km2 = poly.area / 1e6
                            # Assume large polygons outside DEM that touch boundary are ocean
                            is_ocean = touches_boundary and area_km2 > 1.0
                            print(f"      Polygon {i}: area={area_km2:.2f}km², "
                                  f"NO DEM SAMPLES, touches_boundary={touches_boundary}, is_ocean={is_ocean}")
                            if is_ocean:
                                ocean_polys.append(poly)

                    if ocean_polys:
                        print(f"    Rendering {len(ocean_polys)} ocean polygon(s)")
                        water_fill = TERRAIN_COLORS["water"]

                        for ocean_poly in ocean_polys:
                            coords = list(ocean_poly.exterior.coords)
                            svg_points = [to_svg(x, y) for x, y in coords]
                            layer_ocean.add(dwg.polygon(
                                points=svg_points,
                                fill=water_fill,
                                stroke="none",
                            ))
                    else:
                        print("    No ocean polygons identified (none at sea level)")

        except Exception as e:
            import traceback
            print(f"    Error creating ocean polygon: {e}")
            traceback.print_exc()

    # === Enhanced Features Rendering ===
    if enhanced_features:
        # Reference tiles (hidden layer for artist reference)
        ref_tiles = enhanced_features.get('reference_tiles')
        if ref_tiles and ref_tiles.get('image_data'):
            print("  Adding reference tile layer (hidden)...")
            # Convert tile bounds to SVG coordinates
            transformer_to_grid = Transformer.from_crs(WGS84, GRID_CRS, always_xy=True)
            west_x, south_y = transformer_to_grid.transform(ref_tiles['bounds']['west'], ref_tiles['bounds']['south'])
            east_x, north_y = transformer_to_grid.transform(ref_tiles['bounds']['east'], ref_tiles['bounds']['north'])

            # SVG coordinates (note Y inversion)
            svg_x, svg_y_top = to_svg(west_x, north_y)
            svg_x2, svg_y_bottom = to_svg(east_x, south_y)
            svg_width = svg_x2 - svg_x
            svg_height = svg_y_bottom - svg_y_top

            layer_reference_tiles.add(dwg.image(
                href=f"data:image/png;base64,{ref_tiles['image_data']}",
                insert=(svg_x, svg_y_top),
                size=(svg_width, svg_height),
            ))

        # Helper function for rendering linestrings
        def render_linestrings(gdf, layer, color, width, dash=None):
            if gdf is None or gdf.empty:
                return 0
            count = 0
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None:
                    continue

                if geom.geom_type == "LineString":
                    lines = [geom]
                elif geom.geom_type == "MultiLineString":
                    lines = list(geom.geoms)
                else:
                    continue

                for line in lines:
                    if len(line.coords) < 2:
                        continue
                    svg_points = [to_svg(x, y) for x, y in line.coords]
                    props = {
                        'points': svg_points,
                        'stroke': color,
                        'stroke_width': width,
                        'fill': 'none',
                    }
                    if dash:
                        props['stroke_dasharray'] = dash
                    layer.add(dwg.polyline(**props))
                    count += 1
            return count

        # Helper function for rendering polygons
        def render_polygons(gdf, layer, fill_color, stroke_color=None, stroke_width=0):
            if gdf is None or gdf.empty:
                return 0
            count = 0
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None:
                    continue

                if geom.geom_type == "Polygon":
                    polys = [geom]
                elif geom.geom_type == "MultiPolygon":
                    polys = list(geom.geoms)
                else:
                    continue

                for poly in polys:
                    coords = list(poly.exterior.coords)
                    svg_points = [to_svg(x, y) for x, y in coords]
                    props = {
                        'points': svg_points,
                        'fill': fill_color,
                    }
                    if stroke_color:
                        props['stroke'] = stroke_color
                        props['stroke_width'] = stroke_width
                    else:
                        props['stroke'] = 'none'
                    layer.add(dwg.polygon(**props))
                    count += 1
            return count

        # Farmland areas
        farmland = enhanced_features.get('farmland')
        if farmland is not None and not farmland.empty:
            print("  Rendering farmland...")
            for _, row in farmland.iterrows():
                geom = row.geometry
                if geom is None:
                    continue
                landuse = row.get('landuse', '')
                fill = PADDY_COLOR if landuse == 'paddy' else FARMLAND_COLOR

                if geom.geom_type == "Polygon":
                    polys = [geom]
                elif geom.geom_type == "MultiPolygon":
                    polys = list(geom.geoms)
                else:
                    continue

                for poly in polys:
                    coords = list(poly.exterior.coords)
                    svg_points = [to_svg(x, y) for x, y in coords]
                    # Solid fill - patterns via url() don't work in Affinity Designer
                    fill_val = PADDY_COLOR if landuse == 'paddy' else FARMLAND_COLOR
                    outline = TERRAIN_OUTLINES.get("farmland")
                    layer_farmland.add(dwg.polygon(
                        points=svg_points,
                        fill=fill_val,
                        stroke=outline[0] if outline else "none",
                        stroke_width=outline[1] if outline else 0
                    ))

        # Water areas
        waterways_area = enhanced_features.get('waterways_area')
        if waterways_area is not None and not waterways_area.empty:
            print("  Rendering water areas...")
            water_outline = TERRAIN_OUTLINES.get("water")
            count = render_polygons(
                waterways_area, layer_waterways_area, TERRAIN_COLORS['water'],
                stroke_color=water_outline[0] if water_outline else None,
                stroke_width=water_outline[1] if water_outline else 0
            )
            print(f"    Rendered {count} water areas")

        # Streams
        streams = enhanced_features.get('streams')
        if streams is not None and not streams.empty:
            print("  Rendering streams...")
            count = render_linestrings(streams, layer_streams, STREAM_COLOR, STREAM_WIDTH_M)
            print(f"    Rendered {count} streams")

        # Coastline
        coastline = enhanced_features.get('coastline')
        if coastline is not None and not coastline.empty:
            print("  Rendering coastline...")
            count = render_linestrings(coastline, layer_coastline, COASTLINE_COLOR, COASTLINE_WIDTH_M)
            print(f"    Rendered {count} coastline segments")

        # Tree rows
        tree_rows = enhanced_features.get('tree_rows')
        if tree_rows is not None and not tree_rows.empty:
            print("  Rendering tree rows...")
            count = render_linestrings(tree_rows, layer_tree_rows, TREE_ROW_COLOR, TREE_ROW_WIDTH_M)
            print(f"    Rendered {count} tree rows")

        # Cliffs
        cliffs = enhanced_features.get('cliffs')
        if cliffs is not None and not cliffs.empty:
            print("  Rendering cliffs...")
            count = render_linestrings(cliffs, layer_cliffs, CLIFF_COLOR, CLIFF_WIDTH_M)
            print(f"    Rendered {count} cliffs")

        # Barriers (fences, walls)
        barriers = enhanced_features.get('barriers')
        if barriers is not None and not barriers.empty:
            print("  Rendering barriers...")
            count = render_linestrings(barriers, layer_barriers, BARRIER_COLOR, BARRIER_WIDTH_M)
            print(f"    Rendered {count} barriers")

        # Power lines
        powerlines = enhanced_features.get('powerlines')
        if powerlines is not None and not powerlines.empty:
            print("  Rendering powerlines...")
            count = render_linestrings(powerlines, layer_powerlines, POWERLINE_COLOR, POWERLINE_WIDTH_M, POWERLINE_DASH)
            print(f"    Rendered {count} powerlines")

        # Railways
        railways = enhanced_features.get('railways')
        if railways is not None and not railways.empty:
            print("  Rendering railways...")
            count = render_linestrings(railways, layer_railways, RAILWAY_COLOR, RAILWAY_WIDTH_M)
            print(f"    Rendered {count} railways")

        # Paths (footways, cycleways)
        paths = enhanced_features.get('paths')
        if paths is not None and not paths.empty:
            print("  Rendering paths...")
            count = render_linestrings(paths, layer_paths, PATH_COLOR, PATH_WIDTH_M, PATH_DASH)
            print(f"    Rendered {count} paths")

        # Bridges
        bridges = enhanced_features.get('bridges')
        if bridges is not None and not bridges.empty:
            print("  Rendering bridges...")
            count = 0
            for _, row in bridges.iterrows():
                geom = row.geometry
                if geom is None:
                    continue

                if geom.geom_type == "LineString":
                    lines = [geom]
                elif geom.geom_type == "MultiLineString":
                    lines = list(geom.geoms)
                elif geom.geom_type == "Polygon":
                    # Render polygon bridges
                    coords = list(geom.exterior.coords)
                    svg_points = [to_svg(x, y) for x, y in coords]
                    layer_bridges.add(dwg.polygon(
                        points=svg_points,
                        fill=BRIDGE_FILL_COLOR,
                        stroke=BRIDGE_OUTLINE_COLOR,
                        stroke_width=2,
                    ))
                    count += 1
                    continue
                else:
                    continue

                for line in lines:
                    if len(line.coords) < 2:
                        continue
                    svg_points = [to_svg(x, y) for x, y in line.coords]
                    # Draw bridge as thick line with outline
                    layer_bridges.add(dwg.polyline(
                        points=svg_points,
                        stroke=BRIDGE_OUTLINE_COLOR,
                        stroke_width=BRIDGE_WIDTH_M + 2,
                        fill='none',
                    ))
                    layer_bridges.add(dwg.polyline(
                        points=svg_points,
                        stroke=BRIDGE_FILL_COLOR,
                        stroke_width=BRIDGE_WIDTH_M - 2,
                        fill='none',
                    ))
                    count += 1
            print(f"    Rendered {count} bridges")

        # Render new tactical features
        # Mangrove areas
        mangrove = enhanced_features.get('mangrove')
        if mangrove is not None and not mangrove.empty:
            print("  Rendering mangrove...")
            count = render_polygons(mangrove, layer_mangrove, MANGROVE_COLOR,
                                   stroke_color=MANGROVE_OUTLINE, stroke_width=1)
            print(f"    Rendered {count} mangrove areas")

        # Wetlands
        wetland = enhanced_features.get('wetland')
        if wetland is not None and not wetland.empty:
            print("  Rendering wetland...")
            count = render_polygons(wetland, layer_wetland, WETLAND_COLOR,
                                   stroke_color=WETLAND_OUTLINE, stroke_width=1)
            print(f"    Rendered {count} wetland areas")

        # Heath/scrubland
        heath = enhanced_features.get('heath')
        if heath is not None and not heath.empty:
            print("  Rendering heath...")
            count = render_polygons(heath, layer_heath, HEATH_COLOR,
                                   stroke_color=HEATH_OUTLINE, stroke_width=1)
            print(f"    Rendered {count} heath areas")

        # Rocky terrain
        rocky = enhanced_features.get('rocky_terrain')
        if rocky is not None and not rocky.empty:
            print("  Rendering rocky terrain...")
            count = render_polygons(rocky, layer_rocky, ROCKY_COLOR,
                                   stroke_color=ROCKY_OUTLINE, stroke_width=1)
            print(f"    Rendered {count} rocky areas")

        # Sand/dunes
        sand = enhanced_features.get('sand')
        if sand is not None and not sand.empty:
            print("  Rendering sand...")
            count = render_polygons(sand, layer_sand, SAND_COLOR,
                                   stroke_color=SAND_OUTLINE, stroke_width=1)
            print(f"    Rendered {count} sand areas")

        # Military areas
        military = enhanced_features.get('military')
        if military is not None and not military.empty:
            print("  Rendering military areas...")
            count = render_polygons(military, layer_military, MILITARY_COLOR,
                                   stroke_color=MILITARY_OUTLINE, stroke_width=MILITARY_WIDTH_M)
            print(f"    Rendered {count} military areas")

        # Quarries
        quarries = enhanced_features.get('quarries')
        if quarries is not None and not quarries.empty:
            print("  Rendering quarries...")
            count = render_polygons(quarries, layer_quarries, QUARRY_COLOR,
                                   stroke_color=QUARRY_OUTLINE, stroke_width=1)
            print(f"    Rendered {count} quarries")

        # Cemeteries
        cemeteries = enhanced_features.get('cemeteries')
        if cemeteries is not None and not cemeteries.empty:
            print("  Rendering cemeteries...")
            count = render_polygons(cemeteries, layer_cemeteries, CEMETERY_COLOR,
                                   stroke_color=CEMETERY_OUTLINE, stroke_width=1)
            print(f"    Rendered {count} cemeteries")

        # Airfields
        airfields = enhanced_features.get('airfields')
        if airfields is not None and not airfields.empty:
            print("  Rendering airfields...")
            poly_count = 0
            line_count = 0
            for _, row in airfields.iterrows():
                geom = row.geometry
                if geom is None:
                    continue
                aeroway_type = row.get('aeroway', '')

                if geom.geom_type in ('Polygon', 'MultiPolygon'):
                    if geom.geom_type == 'Polygon':
                        polys = [geom]
                    else:
                        polys = list(geom.geoms)
                    for poly in polys:
                        coords = list(poly.exterior.coords)
                        svg_points = [to_svg(x, y) for x, y in coords]
                        layer_airfields.add(dwg.polygon(
                            points=svg_points,
                            fill=AIRFIELD_COLOR if aeroway_type != 'runway' else RUNWAY_COLOR,
                            stroke=AIRFIELD_OUTLINE,
                            stroke_width=1,
                        ))
                        poly_count += 1
                elif geom.geom_type == 'LineString':
                    # Runways as lines
                    svg_points = [to_svg(x, y) for x, y in geom.coords]
                    layer_airfields.add(dwg.polyline(
                        points=svg_points,
                        stroke=RUNWAY_COLOR,
                        stroke_width=RUNWAY_WIDTH_M,
                        fill='none',
                    ))
                    line_count += 1
            print(f"    Rendered {poly_count} airfield polygons, {line_count} runways")

        # Ports
        ports = enhanced_features.get('ports')
        if ports is not None and not ports.empty:
            print("  Rendering ports...")
            poly_count = 0
            line_count = 0
            for _, row in ports.iterrows():
                geom = row.geometry
                if geom is None:
                    continue

                if geom.geom_type in ('Polygon', 'MultiPolygon'):
                    if geom.geom_type == 'Polygon':
                        polys = [geom]
                    else:
                        polys = list(geom.geoms)
                    for poly in polys:
                        coords = list(poly.exterior.coords)
                        svg_points = [to_svg(x, y) for x, y in coords]
                        layer_ports.add(dwg.polygon(
                            points=svg_points,
                            fill=PORT_COLOR,
                            stroke=PORT_OUTLINE,
                            stroke_width=1,
                        ))
                        poly_count += 1
                elif geom.geom_type == 'LineString':
                    # Piers as lines
                    svg_points = [to_svg(x, y) for x, y in geom.coords]
                    layer_ports.add(dwg.polyline(
                        points=svg_points,
                        stroke=PIER_COLOR,
                        stroke_width=PIER_WIDTH_M,
                        fill='none',
                    ))
                    line_count += 1
            print(f"    Rendered {poly_count} port polygons, {line_count} piers")

        # Dams
        dams = enhanced_features.get('dams')
        if dams is not None and not dams.empty:
            print("  Rendering dams...")
            count = render_linestrings(dams, layer_dams, DAM_COLOR, DAM_WIDTH_M)
            print(f"    Rendered {count} dams")

        # Places (settlement names)
        places = enhanced_features.get('places')
        if places is not None and not places.empty:
            print("  Rendering place names...")
            count = 0
            for _, row in places.iterrows():
                geom = row.geometry
                if geom is None or geom.geom_type != 'Point':
                    continue
                name = row.get('name', '')
                if not name:
                    continue
                place_type = row.get('place', 'village')
                svg_x, svg_y = to_svg(geom.x, geom.y)

                # Scale font size by place importance
                size_scale = {'city': 1.5, 'town': 1.2, 'village': 1.0, 'hamlet': 0.8}.get(place_type, 0.8)
                font_size = PLACE_LABEL_SIZE_M * size_scale

                layer_places.add(dwg.text(
                    name,
                    insert=(svg_x, svg_y),
                    text_anchor="middle",
                    font_size=font_size,
                    fill=PLACE_LABEL_COLOR,
                    font_family="sans-serif",
                    font_weight="bold" if place_type in ('city', 'town') else "normal",
                ))
                count += 1
            print(f"    Rendered {count} place names")

        # Peaks
        peaks = enhanced_features.get('peaks')
        if peaks is not None and not peaks.empty:
            print("  Rendering peaks...")
            count = 0
            for _, row in peaks.iterrows():
                geom = row.geometry
                if geom is None or geom.geom_type != 'Point':
                    continue
                svg_x, svg_y = to_svg(geom.x, geom.y)
                ele = row.get('ele', '')

                # Triangle marker for peaks
                half = PEAK_MARKER_SIZE_M / 2
                points = [
                    (svg_x, svg_y - half),  # Top
                    (svg_x - half, svg_y + half),  # Bottom left
                    (svg_x + half, svg_y + half),  # Bottom right
                ]
                layer_peaks.add(dwg.polygon(points=points, fill=PEAK_COLOR, stroke="none"))

                # Add elevation label if available
                if ele:
                    try:
                        ele_val = int(float(ele))
                        layer_peaks.add(dwg.text(
                            f"{ele_val}m",
                            insert=(svg_x + PEAK_MARKER_SIZE_M, svg_y),
                            font_size=PEAK_LABEL_SIZE_M,
                            fill=PEAK_COLOR,
                            font_family="sans-serif",
                        ))
                    except:
                        pass
                count += 1
            print(f"    Rendered {count} peaks")

        # Caves
        caves = enhanced_features.get('caves')
        if caves is not None and not caves.empty:
            print("  Rendering caves...")
            count = 0
            for _, row in caves.iterrows():
                geom = row.geometry
                if geom is None or geom.geom_type != 'Point':
                    continue
                svg_x, svg_y = to_svg(geom.x, geom.y)

                # Circle marker for caves
                layer_caves.add(dwg.circle(
                    center=(svg_x, svg_y),
                    r=CAVE_MARKER_SIZE_M / 2,
                    fill="none",
                    stroke=CAVE_COLOR,
                    stroke_width=2,
                ))
                count += 1
            print(f"    Rendered {count} caves")

        # Towers
        towers = enhanced_features.get('towers')
        if towers is not None and not towers.empty:
            print("  Rendering towers...")
            count = 0
            for _, row in towers.iterrows():
                geom = row.geometry
                if geom is None or geom.geom_type != 'Point':
                    continue
                svg_x, svg_y = to_svg(geom.x, geom.y)

                # Square marker for towers
                half = TOWER_MARKER_SIZE_M / 2
                layer_towers.add(dwg.rect(
                    insert=(svg_x - half, svg_y - half),
                    size=(TOWER_MARKER_SIZE_M, TOWER_MARKER_SIZE_M),
                    fill=TOWER_COLOR,
                    stroke="none",
                ))
                count += 1
            print(f"    Rendered {count} towers")

        # Fuel infrastructure
        fuel = enhanced_features.get('fuel_infrastructure')
        if fuel is not None and not fuel.empty:
            print("  Rendering fuel infrastructure...")
            count = 0
            for _, row in fuel.iterrows():
                geom = row.geometry
                if geom is None:
                    continue

                if geom.geom_type == 'Point':
                    svg_x, svg_y = to_svg(geom.x, geom.y)
                    # Diamond marker for fuel
                    half = FUEL_MARKER_SIZE_M / 2
                    points = [
                        (svg_x, svg_y - half),
                        (svg_x + half, svg_y),
                        (svg_x, svg_y + half),
                        (svg_x - half, svg_y),
                    ]
                    layer_fuel.add(dwg.polygon(points=points, fill=FUEL_COLOR, stroke="none"))
                    count += 1
                elif geom.geom_type in ('Polygon', 'MultiPolygon'):
                    if geom.geom_type == 'Polygon':
                        polys = [geom]
                    else:
                        polys = list(geom.geoms)
                    for poly in polys:
                        coords = list(poly.exterior.coords)
                        svg_points = [to_svg(x, y) for x, y in coords]
                        layer_fuel.add(dwg.polygon(
                            points=svg_points,
                            fill=FUEL_COLOR,
                            stroke="#994400",
                            stroke_width=1,
                        ))
                        count += 1
            print(f"    Rendered {count} fuel facilities")

    # Layer 3: Buildings (unclipped - frame will mask edges)
    if not buildings.empty:
        print("  Rendering buildings...")
        for _, bldg_row in buildings.iterrows():
            geom = bldg_row.geometry

            # Handle different geometry types
            if geom.geom_type == "Polygon":
                polys = [geom]
            elif geom.geom_type == "MultiPolygon":
                polys = list(geom.geoms)
            else:
                continue

            for poly in polys:
                coords = list(poly.exterior.coords)
                svg_points = [to_svg(x, y) for x, y in coords]

                layer_buildings.add(dwg.polygon(
                    points=svg_points,
                    fill=BUILDING_COLOR,
                    stroke=BUILDING_OUTLINE_COLOR,
                    stroke_width=BUILDING_OUTLINE_WIDTH_M,
                ))

    # Layer 4: Contour lines (unclipped - frame will mask edges)
    print("  Rendering contours...")

    # Helper function to get position and angle along a line at a given distance
    def get_point_and_angle_at_distance(line_coords, target_distance):
        """Get (x, y, angle_degrees) at target_distance along the line."""
        cumulative = 0
        for i in range(len(line_coords) - 1):
            x1, y1 = line_coords[i]
            x2, y2 = line_coords[i + 1]
            seg_len = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            if cumulative + seg_len >= target_distance:
                # Interpolate position within this segment
                remaining = target_distance - cumulative
                t = remaining / seg_len if seg_len > 0 else 0
                px = x1 + t * (x2 - x1)
                py = y1 + t * (y2 - y1)
                # Calculate angle (in degrees, for SVG rotation)
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                return (px, py, angle)
            cumulative += seg_len

        # Return end point if distance exceeds line length
        x1, y1 = line_coords[-2]
        x2, y2 = line_coords[-1]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        return (line_coords[-1][0], line_coords[-1][1], angle)

    def get_line_length(line_coords):
        """Calculate total length of a line from its coordinates."""
        total = 0
        for i in range(len(line_coords) - 1):
            x1, y1 = line_coords[i]
            x2, y2 = line_coords[i + 1]
            total += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return total

    contour_label_count = 0

    for contour in contours:
        geom = contour["geometry"]
        is_index = contour["is_index"]
        elevation = contour["elevation"]

        # Skip index contours at elevation 0 (sea level)
        if is_index and elevation == 0:
            continue

        # Handle multi-linestrings
        if geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        elif geom.geom_type == "LineString":
            lines = [geom]
        else:
            continue

        # Select target layer based on contour type
        target_layer = layer_contours_index if is_index else layer_contours_regular

        for line in lines:
            if len(line.coords) < 2:
                continue

            svg_points = [to_svg(x, y) for x, y in line.coords]

            target_layer.add(dwg.polyline(
                points=svg_points,
                stroke=INDEX_CONTOUR_COLOR if is_index else CONTOUR_COLOR,
                stroke_width=INDEX_CONTOUR_WIDTH_M if is_index else CONTOUR_WIDTH_M,
                fill="none",
            ))

            # Add elevation labels along index contours every CONTOUR_LABEL_INTERVAL_M
            if is_index:
                # Work in world coordinates for distance calculation
                world_coords = list(line.coords)
                line_length = get_line_length(world_coords)

                # Skip very short lines
                if line_length < CONTOUR_LABEL_INTERVAL_M / 2:
                    continue

                # Calculate label positions
                # Start offset from beginning, then place every CONTOUR_LABEL_INTERVAL_M
                start_offset = CONTOUR_LABEL_INTERVAL_M / 2  # Start halfway through first interval
                distance = start_offset

                while distance < line_length - 100:  # Don't label too close to end
                    # Get position and angle in world coordinates
                    wx, wy, world_angle = get_point_and_angle_at_distance(world_coords, distance)

                    # Convert to SVG coordinates
                    sx, sy = to_svg(wx, wy)

                    # SVG Y is inverted, so flip the angle
                    svg_angle = -world_angle

                    # Ensure text is always readable (not upside down)
                    # If angle would make text upside down, rotate 180°
                    if svg_angle > 90 or svg_angle < -90:
                        svg_angle += 180

                    # Create elevation label text
                    elev_text = f"{int(elevation)}"

                    # Add text with rotation transform
                    text_elem = dwg.text(
                        elev_text,
                        insert=(sx, sy),
                        text_anchor="middle",
                        font_size=CONTOUR_LABEL_SIZE_M,
                        fill=INDEX_CONTOUR_COLOR,
                        font_family="sans-serif",
                        dominant_baseline="middle",
                    )
                    text_elem["transform"] = f"rotate({svg_angle}, {sx}, {sy})"
                    layer_contour_labels.add(text_elem)
                    contour_label_count += 1

                    distance += CONTOUR_LABEL_INTERVAL_M

    print(f"    Added {contour_label_count} contour elevation labels")

    # Layer 5: Roads (separate layers by type, unclipped - frame will mask edges)
    if not roads.empty:
        print("  Rendering roads...")

        # Map road types to their layers
        road_layers = {
            "service": layer_roads_service,
            "residential": layer_roads_residential,
            "track": layer_roads_track,
            "minor_road": layer_roads_minor,
            "major_road": layer_roads_major,
            "highway": layer_roads_highway,
        }

        # Count roads by type
        road_counts = {}

        for _, road_row in roads.iterrows():
            highway = road_row.get("highway", "")
            road_type = classify_road_type(highway)
            geom = road_row.geometry

            road_counts[road_type] = road_counts.get(road_type, 0) + 1

            # Handle multi-linestrings
            if geom.geom_type == "MultiLineString":
                lines = list(geom.geoms)
            elif geom.geom_type == "LineString":
                lines = [geom]
            else:
                continue

            color = ROAD_COLORS.get(road_type, "#000000")
            width = ROAD_WIDTHS_M.get(road_type, 5)
            outline = ROAD_OUTLINES.get(road_type)
            target_layer = road_layers.get(road_type, layer_roads_residential)

            for line in lines:
                if len(line.coords) < 2:
                    continue

                svg_points = [to_svg(x, y) for x, y in line.coords]

                # Draw outline first (if applicable)
                if outline:
                    target_layer.add(dwg.polyline(
                        points=svg_points,
                        stroke=outline,
                        stroke_width=width + (2 * ROAD_OUTLINE_WIDTH_M),
                        fill="none",
                        stroke_linecap="round",
                        stroke_linejoin="round",
                    ))

                # Draw road fill
                line_elem = dwg.polyline(
                    points=svg_points,
                    stroke=color,
                    stroke_width=width,
                    fill="none",
                    stroke_linecap="round",
                    stroke_linejoin="round",
                )

                # Dashed pattern for tracks
                if road_type == "track":
                    line_elem["stroke-dasharray"] = f"{width*2},{width*2}"

                target_layer.add(line_elem)

        print(f"    Road counts: {road_counts}")

    # Layer 6: Out-of-play frame (masks everything outside hex grid)
    # This frame stays UNROTATED - masks partial hexes and artifacts
    print("  Creating out-of-play frame...")

    # Create SVG boundary rectangle covering entire document including bleed
    svg_boundary = box(-10, -10, viewbox_width_with_bleed + 10, viewbox_height_with_bleed + 10)

    # Convert playable area (hex grid union) to SVG coordinates
    if playable_area.geom_type == "Polygon":
        playable_svg_coords = [to_svg(x, y) for x, y in playable_area.exterior.coords]
        playable_svg_poly = Polygon(playable_svg_coords)
    else:
        # MultiPolygon - take union
        playable_svg_poly = unary_union([
            Polygon([to_svg(x, y) for x, y in p.exterior.coords])
            for p in playable_area.geoms
        ])

    # Create frame by subtracting playable hex area from SVG boundary
    frame_poly = svg_boundary.difference(playable_svg_poly)

    # Render the frame
    if frame_poly.geom_type == "Polygon":
        frame_polys = [frame_poly]
    elif frame_poly.geom_type == "MultiPolygon":
        frame_polys = list(frame_poly.geoms)
    else:
        frame_polys = []

    for fp in frame_polys:
        # Exterior ring
        ext_coords = list(fp.exterior.coords)
        # Interior rings (holes)
        int_rings = [list(interior.coords) for interior in fp.interiors]

        # SVG path with holes using even-odd fill rule
        path_data = "M " + " L ".join(f"{x},{y}" for x, y in ext_coords) + " Z"
        for hole in int_rings:
            path_data += " M " + " L ".join(f"{x},{y}" for x, y in hole) + " Z"

        layer_out_of_play_frame.add(dwg.path(
            d=path_data,
            fill="#404040",
            fill_rule="evenodd",
            stroke="none",
        ))

    # Layer 7: Hex grid overlay (darker for contrast)
    print("  Rendering hex grid...")
    for h in hexes:
        poly = grid.hex_polygon(h.q, h.r)
        coords = list(poly.exterior.coords)
        svg_points = [to_svg(x, y) for x, y in coords]

        layer_hex_grid.add(dwg.polygon(
            points=svg_points,
            fill="none",
            stroke=HEX_GRID_COLOR,
            stroke_width=HEX_GRID_WIDTH_M,
            stroke_opacity=HEX_GRID_OPACITY,
        ))

    # Layer 7: Hex markers (center circles) and labels
    print("  Rendering hex markers and labels...")

    for h in hexes:
        cx, cy = grid.axial_to_world(h.q, h.r)

        # Center circle for terrain marker (open, no fill)
        center_svg_x, center_svg_y = to_svg(cx, cy)
        layer_hex_markers.add(dwg.circle(
            center=(center_svg_x, center_svg_y),
            r=HEX_MARKER_RADIUS_M,
            fill="none",
            stroke=HEX_GRID_COLOR,
            stroke_width=HEX_GRID_WIDTH_M,
            stroke_opacity=HEX_GRID_OPACITY,
        ))

        # Position at top of hex (85% up from center to leave margin inside hex)
        top_offset = grid.size * math.sqrt(3) / 2 * 0.80
        svg_x, svg_y = to_svg(cx, cy + top_offset)

        # Adjust y down slightly for text baseline (in meters)
        svg_y += HEX_LABEL_SIZE_M * 0.35

        # Generate label: XX.YY (1-indexed)
        label = f"{h.q + 1:02d}.{h.r + 1:02d}"

        layer_hex_labels.add(dwg.text(
            label,
            insert=(svg_x, svg_y),
            text_anchor="middle",
            font_size=HEX_LABEL_SIZE_M,
            fill=HEX_GRID_COLOR,
            font_family="sans-serif",
            opacity=HEX_GRID_OPACITY,
        ))

    # Layer 8: MGRS Grid (light grey, subtle)
    # Grid lines clipped to playable area, labels at intersection with centermost perpendicular line
    # Labels rotate with map to indicate north direction (top of numbers = north)
    if mgrs_grid:
        print("  Rendering MGRS grid...")
        # Use expanded bounds for initial line generation to ensure full coverage
        mgrs_buffer = MARGIN_M + 100
        map_box = box(
            config.data_min_x - mgrs_buffer,
            config.data_min_y - mgrs_buffer,
            config.data_max_x + mgrs_buffer,
            config.data_max_y + mgrs_buffer
        )

        # Rotation parameters - rotation center is center of trim area
        rot_cx = content_offset_m + trim_width_m / 2
        rot_cy = content_offset_m + trim_height_m / 2
        angle_rad = math.radians(config.rotation_deg)

        def rotate_point(x, y):
            """Rotate a point around the trim area center."""
            dx, dy = x - rot_cx, y - rot_cy
            rx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
            ry = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
            return (rx + rot_cx, ry + rot_cy)

        # Get hex boundary in SVG coords for clipping and intersection testing
        if playable_svg_poly.geom_type == "Polygon":
            hex_boundary_poly = playable_svg_poly
        else:
            hex_boundary_poly = playable_svg_poly.convex_hull

        # Calculate expanded clipping area for rotation
        if config.rotation_deg != 0:
            rotation_buffer = max(viewbox_width, viewbox_height) * 0.5
            clip_poly = box(
                -rotation_buffer,
                -rotation_buffer,
                viewbox_width_with_bleed + rotation_buffer,
                viewbox_height_with_bleed + rotation_buffer
            )
        else:
            clip_poly = hex_boundary_poly

        # Collect line info for rendering and labeling
        easting_lines_info = []  # N-S lines (vertical in unrotated view)
        northing_lines_info = []  # E-W lines (horizontal in unrotated view)

        # Center of hex grid in SVG coordinates (for centermost line selection)
        hex_center_svg_x = content_offset_m + play_margin_left_m + center_offset_x_m + hex_grid_width / 2
        hex_center_svg_y = content_offset_m + play_margin_top_m + center_offset_y_m + hex_grid_height / 2

        # Draw vertical lines (easting/N-S lines) and collect info
        for line_info in mgrs_grid['vertical_lines']:
            line_geom = LineString([line_info['start'], line_info['end']])
            clipped = line_geom.intersection(map_box)

            if clipped.is_empty or clipped.geom_type != 'LineString':
                continue

            # Convert to SVG coordinates
            svg_points = [to_svg(x, y) for x, y in clipped.coords]
            svg_line = LineString(svg_points)
            clipped_svg = svg_line.intersection(clip_poly)

            if clipped_svg.is_empty:
                continue

            # Render the clipped line(s)
            if clipped_svg.geom_type == 'LineString':
                clipped_lines = [clipped_svg]
            elif clipped_svg.geom_type == 'MultiLineString':
                clipped_lines = list(clipped_svg.geoms)
            else:
                clipped_lines = []

            for cline in clipped_lines:
                if len(cline.coords) >= 2:
                    layer_mgrs_grid.add(dwg.polyline(
                        points=list(cline.coords),
                        stroke=MGRS_GRID_COLOR,
                        stroke_width=MGRS_GRID_WIDTH_M,
                        fill="none",
                    ))

            # Store line info for labeling
            easting_km = int(line_info['utm_value'] / 1000) % 100
            label_text = f"{easting_km:02d}"

            easting_lines_info.append({
                'label': label_text,
                'utm_value': line_info['utm_value'],
                'svg_line': svg_line,
                'svg_x': svg_points[0][0],
            })

        # Draw horizontal lines (northing/E-W lines) and collect info
        for line_info in mgrs_grid['horizontal_lines']:
            line_geom = LineString([line_info['start'], line_info['end']])
            clipped = line_geom.intersection(map_box)

            if clipped.is_empty or clipped.geom_type != 'LineString':
                continue

            # Convert to SVG coordinates
            svg_points = [to_svg(x, y) for x, y in clipped.coords]
            svg_line = LineString(svg_points)
            clipped_svg = svg_line.intersection(clip_poly)

            if clipped_svg.is_empty:
                continue

            # Render the clipped line(s)
            if clipped_svg.geom_type == 'LineString':
                clipped_lines = [clipped_svg]
            elif clipped_svg.geom_type == 'MultiLineString':
                clipped_lines = list(clipped_svg.geoms)
            else:
                clipped_lines = []

            for cline in clipped_lines:
                if len(cline.coords) >= 2:
                    layer_mgrs_grid.add(dwg.polyline(
                        points=list(cline.coords),
                        stroke=MGRS_GRID_COLOR,
                        stroke_width=MGRS_GRID_WIDTH_M,
                        fill="none",
                    ))

            # Store line info for labeling
            northing_km = int(line_info['utm_value'] / 1000) % 100
            label_text = f"{northing_km:02d}"

            northing_lines_info.append({
                'label': label_text,
                'utm_value': line_info['utm_value'],
                'svg_line': svg_line,
                'svg_y': svg_points[0][1],
            })

        # Find centermost E-W line (for placing easting labels)
        centermost_ew_line = None
        if northing_lines_info:
            centermost_ew_line = min(northing_lines_info,
                key=lambda x: abs(x['svg_y'] - hex_center_svg_y))

        # Find centermost N-S line (for placing northing labels)
        centermost_ns_line = None
        if easting_lines_info:
            centermost_ns_line = min(easting_lines_info,
                key=lambda x: abs(x['svg_x'] - hex_center_svg_x))

        # Label settings
        label_color = "#808080"  # Medium grey
        label_rotation = config.rotation_deg  # Rotate with map to indicate north
        label_offset_m = 500  # Offset labels from centerline to clarify which line they label

        easting_label_count = 0
        northing_label_count = 0

        # Place easting labels (on N-S lines) at intersection with centermost E-W line
        # Offset 500m north (in SVG pre-rotation coords, north = -Y)
        if centermost_ew_line:
            for line_info in easting_lines_info:
                try:
                    intersection = line_info['svg_line'].intersection(centermost_ew_line['svg_line'])
                    if not intersection.is_empty and intersection.geom_type == 'Point':
                        # Apply offset north (subtract from Y in SVG coords) before rotation
                        offset_x = intersection.x
                        offset_y = intersection.y - label_offset_m
                        # Rotate the offset point
                        rot_x, rot_y = rotate_point(offset_x, offset_y)

                        # Check if within hex boundary after rotation
                        if hex_boundary_poly.contains(Point(rot_x, rot_y)):
                            text_elem = dwg.text(
                                line_info['label'],
                                insert=(rot_x, rot_y + MGRS_LABEL_SIZE_M * 0.35),
                                text_anchor="middle",
                                font_size=MGRS_LABEL_SIZE_M,
                                fill=label_color,
                                font_family="sans-serif",
                            )
                            if label_rotation != 0:
                                text_elem['transform'] = f"rotate({label_rotation}, {rot_x}, {rot_y})"
                            layer_mgrs_labels.add(text_elem)
                            easting_label_count += 1
                except:
                    pass

        # Place northing labels (on E-W lines) at intersection with centermost N-S line
        # Offset 500m east (in SVG pre-rotation coords, east = +X)
        if centermost_ns_line:
            for line_info in northing_lines_info:
                try:
                    intersection = line_info['svg_line'].intersection(centermost_ns_line['svg_line'])
                    if not intersection.is_empty and intersection.geom_type == 'Point':
                        # Apply offset east (add to X in SVG coords) before rotation
                        offset_x = intersection.x + label_offset_m
                        offset_y = intersection.y
                        # Rotate the offset point
                        rot_x, rot_y = rotate_point(offset_x, offset_y)

                        # Check if within hex boundary after rotation
                        if hex_boundary_poly.contains(Point(rot_x, rot_y)):
                            text_elem = dwg.text(
                                line_info['label'],
                                insert=(rot_x, rot_y + MGRS_LABEL_SIZE_M * 0.35),
                                text_anchor="middle",
                                font_size=MGRS_LABEL_SIZE_M,
                                fill=label_color,
                                font_family="sans-serif",
                            )
                            if label_rotation != 0:
                                text_elem['transform'] = f"rotate({label_rotation}, {rot_x}, {rot_y})"
                            layer_mgrs_labels.add(text_elem)
                            northing_label_count += 1
                except:
                    pass

        print(f"    Placed {easting_label_count} easting labels along centermost E-W line")
        print(f"    Placed {northing_label_count} northing labels along centermost N-S line")

    # Layer 9: Map data block (in data margin area - outside bleed, for SVG reference only)
    print("  Rendering map data block...")

    # Transform corners from projected CRS to WGS84
    transformer_to_wgs84 = Transformer.from_crs(GRID_CRS, WGS84, always_xy=True)
    transformer_to_utm = Transformer.from_crs(GRID_CRS, config.utm_crs, always_xy=True)

    # Calculate hex grid visual origin in SVG coordinates
    # This is the top-left corner of the visual hex grid (including vertex extensions)
    hex_grid_svg_x = content_offset_m + play_margin_left_m + center_offset_x_m
    hex_grid_svg_y = content_offset_m + play_margin_top_m + center_offset_y_m

    # Calculate corner coordinates
    # When rotated, we need to find what world coordinates are at the visible corners
    if config.rotation_deg != 0:
        # The visible area corners (in SVG coordinates) - corners of hex grid
        visible_corners_svg = {
            "NW": (hex_grid_svg_x, hex_grid_svg_y),
            "NE": (hex_grid_svg_x + hex_grid_width, hex_grid_svg_y),
            "SW": (hex_grid_svg_x, hex_grid_svg_y + hex_grid_height),
            "SE": (hex_grid_svg_x + hex_grid_width, hex_grid_svg_y + hex_grid_height),
        }

        # Inverse rotation to find world coordinates
        # SVG rotation is around the center of the trim area
        rot_cx = content_offset_m + trim_width_m / 2
        rot_cy = content_offset_m + trim_height_m / 2
        angle_rad = math.radians(-config.rotation_deg)  # Negative for inverse

        def inverse_rotate_svg_to_world(svg_x, svg_y):
            """Convert SVG coord back to world coord, accounting for rotation."""
            # Translate to rotation center
            dx = svg_x - rot_cx
            dy = svg_y - rot_cy
            # Apply inverse rotation
            rx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
            ry = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
            # Translate back
            unrot_svg_x = rx + rot_cx
            unrot_svg_y = ry + rot_cy
            # Convert from SVG to world (inverse of to_svg)
            world_x = (unrot_svg_x - hex_grid_svg_x) + config.min_x
            world_y = config.max_y - (unrot_svg_y - hex_grid_svg_y)
            return (world_x, world_y)

        corners_proj = {}
        for name, (svg_x, svg_y) in visible_corners_svg.items():
            world_x, world_y = inverse_rotate_svg_to_world(svg_x, svg_y)
            corners_proj[name] = (world_x, world_y)
    else:
        # No rotation - corners are the map bounds
        corners_proj = {
            "NW": (config.min_x, config.max_y),
            "NE": (config.max_x, config.max_y),
            "SW": (config.min_x, config.min_y),
            "SE": (config.max_x, config.min_y),
        }

    corners_wgs84 = {}
    for name, (x, y) in corners_proj.items():
        lon, lat = transformer_to_wgs84.transform(x, y)
        corners_wgs84[name] = (lat, lon)

    # Calculate MGRS for center point
    mgrs_converter = mgrs.MGRS()
    center_mgrs = mgrs_converter.toMGRS(config.center_lat, config.center_lon, MGRSPrecision=5)
    # Parse MGRS: e.g., "51RTG6062568790" -> "51R TG 60625 68790"
    mgrs_zone = center_mgrs[:3]  # "51R"
    mgrs_square = center_mgrs[3:5]  # "TG"
    mgrs_easting = center_mgrs[5:10]  # "60625"
    mgrs_northing = center_mgrs[10:15]  # "68790"
    mgrs_formatted = f"{mgrs_zone} {mgrs_square} {mgrs_easting} {mgrs_northing}"

    # Get UTM coordinates for center
    center_utm_e, center_utm_n = transformer_to_utm.transform(config.center_x, config.center_y)

    # Data block positioning - in the data margin area (outside bleed, for SVG reference only)
    # This area will NOT be printed - it's for artist reference in the SVG
    data_block_y = DATA_FONT_SIZE_M + 20  # Start near top of data margin area

    # Left column: MGRS and coordinate data
    left_col_x = DATA_FONT_SIZE_M

    layer_map_data.add(dwg.text(
        "MAP CENTER",
        insert=(left_col_x, data_block_y),
        font_size=DATA_FONT_SIZE_M * 1.1,
        font_weight="bold",
        fill="#cccccc",
        font_family="monospace",
    ))

    layer_map_data.add(dwg.text(
        f"MGRS: {mgrs_formatted}",
        insert=(left_col_x, data_block_y + DATA_LINE_HEIGHT_M),
        font_size=DATA_FONT_SIZE_M,
        fill="#aaaaaa",
        font_family="monospace",
    ))

    layer_map_data.add(dwg.text(
        f"LAT: {config.center_lat:.5f}  LNG: {config.center_lon:.5f}",
        insert=(left_col_x, data_block_y + DATA_LINE_HEIGHT_M * 2),
        font_size=DATA_FONT_SIZE_M,
        fill="#aaaaaa",
        font_family="monospace",
    ))

    layer_map_data.add(dwg.text(
        f"Easting: {center_utm_e:.0f}  Northing: {center_utm_n:.0f}",
        insert=(left_col_x, data_block_y + DATA_LINE_HEIGHT_M * 3),
        font_size=DATA_FONT_SIZE_M,
        fill="#aaaaaa",
        font_family="monospace",
    ))

    # Middle column: Grid info (positioned in data margin area)
    mid_col_x = viewbox_width_with_bleed * 0.35

    layer_map_data.add(dwg.text(
        "GRID INFO",
        insert=(mid_col_x, data_block_y),
        font_size=DATA_FONT_SIZE_M * 1.1,
        font_weight="bold",
        fill="#cccccc",
        font_family="monospace",
    ))

    layer_map_data.add(dwg.text(
        f"Grid: {GRID_WIDTH} x {GRID_HEIGHT} hexes @ {HEX_SIZE_M}m",
        insert=(mid_col_x, data_block_y + DATA_LINE_HEIGHT_M),
        font_size=DATA_FONT_SIZE_M,
        fill="#aaaaaa",
        font_family="monospace",
    ))

    layer_map_data.add(dwg.text(
        f"Coverage: {(config.max_x - config.min_x)/1000:.1f}km x {(config.max_y - config.min_y)/1000:.1f}km",
        insert=(mid_col_x, data_block_y + DATA_LINE_HEIGHT_M * 2),
        font_size=DATA_FONT_SIZE_M,
        fill="#aaaaaa",
        font_family="monospace",
    ))

    if config.rotation_deg != 0:
        layer_map_data.add(dwg.text(
            f"Rotation: {config.rotation_deg}° CW (North is {config.rotation_deg}° left of up)",
            insert=(mid_col_x, data_block_y + DATA_LINE_HEIGHT_M * 3),
            font_size=DATA_FONT_SIZE_M,
            fill="#ffcc00",  # Yellow to highlight rotation
            font_family="monospace",
        ))

    # Right column: Corner coordinates (positioned in data margin area)
    right_col_x = viewbox_width_with_bleed * 0.62

    layer_map_data.add(dwg.text(
        "CORNER COORDINATES (WGS84)",
        insert=(right_col_x, data_block_y),
        font_size=DATA_FONT_SIZE_M * 1.1,
        font_weight="bold",
        fill="#cccccc",
        font_family="monospace",
    ))

    nw_lat, nw_lon = corners_wgs84["NW"]
    ne_lat, ne_lon = corners_wgs84["NE"]
    sw_lat, sw_lon = corners_wgs84["SW"]
    se_lat, se_lon = corners_wgs84["SE"]

    layer_map_data.add(dwg.text(
        f"NW: {nw_lat:.5f}, {nw_lon:.5f}    NE: {ne_lat:.5f}, {ne_lon:.5f}",
        insert=(right_col_x, data_block_y + DATA_LINE_HEIGHT_M),
        font_size=DATA_FONT_SIZE_M,
        fill="#aaaaaa",
        font_family="monospace",
    ))

    layer_map_data.add(dwg.text(
        f"SW: {sw_lat:.5f}, {sw_lon:.5f}    SE: {se_lat:.5f}, {se_lon:.5f}",
        insert=(right_col_x, data_block_y + DATA_LINE_HEIGHT_M * 2),
        font_size=DATA_FONT_SIZE_M,
        fill="#aaaaaa",
        font_family="monospace",
    ))

    # Layer 10: Compass Rose (in data margin area - outside bleed, for SVG reference only)
    # Shows true north direction when map is rotated
    print("  Rendering compass rose...")

    # Position compass in top-right data margin area (outside print area)
    compass_size = HEX_SIZE_M * 0.8  # Slightly smaller than a hex
    compass_cx = viewbox_width_with_bleed - data_margin_m / 2  # Center in right data margin
    compass_cy = data_margin_m / 2 + DATA_LINE_HEIGHT_M * 2  # In top data margin area

    # Arrow dimensions
    arrow_length = compass_size * 0.4
    arrow_width = compass_size * 0.15
    circle_radius = compass_size * 0.45

    # Create compass group with rotation to show grid north
    # Arrow should align with MGRS easting lines (which are rotated with the map)
    compass_group = dwg.g(transform=f"rotate({config.rotation_deg}, {compass_cx}, {compass_cy})")

    # Outer circle
    layer_compass.add(dwg.circle(
        center=(compass_cx, compass_cy),
        r=circle_radius,
        fill="none",
        stroke="#cccccc",
        stroke_width=2,
    ))

    # North arrow (pointing up) - filled red
    north_arrow = [
        (compass_cx, compass_cy - arrow_length),  # Tip
        (compass_cx - arrow_width / 2, compass_cy),  # Bottom left
        (compass_cx + arrow_width / 2, compass_cy),  # Bottom right
    ]
    compass_group.add(dwg.polygon(
        points=north_arrow,
        fill="#cc0000",
        stroke="#990000",
        stroke_width=1,
    ))

    # South arrow (pointing down) - white outline
    south_arrow = [
        (compass_cx, compass_cy + arrow_length),  # Tip
        (compass_cx - arrow_width / 2, compass_cy),  # Top left
        (compass_cx + arrow_width / 2, compass_cy),  # Top right
    ]
    compass_group.add(dwg.polygon(
        points=south_arrow,
        fill="#ffffff",
        stroke="#666666",
        stroke_width=1,
    ))

    # Add rotated arrows to compass layer
    layer_compass.add(compass_group)

    # Add "N" label above the circle (this rotates with the arrow)
    n_label_group = dwg.g(transform=f"rotate({config.rotation_deg}, {compass_cx}, {compass_cy})")
    n_label_group.add(dwg.text(
        "N",
        insert=(compass_cx, compass_cy - circle_radius - 10),
        text_anchor="middle",
        font_size=DATA_FONT_SIZE_M * 1.2,
        font_weight="bold",
        fill="#cc0000",
        font_family="sans-serif",
    ))
    layer_compass.add(n_label_group)

    # Add rotation indicator text (stays horizontal)
    if config.rotation_deg != 0:
        layer_compass.add(dwg.text(
            f"{config.rotation_deg}° CW",
            insert=(compass_cx, compass_cy + circle_radius + DATA_FONT_SIZE_M + 5),
            text_anchor="middle",
            font_size=DATA_FONT_SIZE_M * 0.8,
            fill="#888888",
            font_family="sans-serif",
        ))

    # === Add all layers to drawing in correct z-order ===
    # Layer order: reference -> terrain -> features -> frame -> grid/labels
    print("  Assembling layers...")

    # Add background OUTSIDE the clipped group so it fills entire document
    # This provides the dark gray background for data margin area
    dwg.add(layer_background)

    # Create a master group with clipPath to constrain map content to trim area
    master_group = dwg.g(id="Master_Content")
    master_group["clip-path"] = "url(#viewbox-clip)"

    # If rotation is specified, wrap geographic content in a rotated group
    # Hex grid and base terrain stay axis-aligned (unrotated) with the frame
    # MGRS grid rotates with terrain (represents real-world UTM coordinates)
    if config.rotation_deg != 0:
        print(f"  Applying {config.rotation_deg}° rotation...")
        # Rotation center is the center of the trim area (not the full document)
        rot_cx = content_offset_m + trim_width_m / 2
        rot_cy = content_offset_m + trim_height_m / 2
        # SVG rotation: positive = clockwise

        # Reference tiles in their own rotated group at the very bottom (hidden by default)
        # This is separate from rotated_content so it renders BELOW layer_terrain_open
        rotated_reference = dwg.g(
            id="Rotated_Reference",
            transform=f"rotate({config.rotation_deg}, {rot_cx}, {rot_cy})"
        )
        rotated_reference.add(layer_reference_tiles)

        rotated_content = dwg.g(
            id="Rotated_Content",
            transform=f"rotate({config.rotation_deg}, {rot_cx}, {rot_cy})"
        )
        # Geographic features rotate with the terrain (NOT the base open fill)
        rotated_content.add(layer_terrain_water)     # Water polygons
        rotated_content.add(layer_terrain_marsh)     # Marsh polygons
        rotated_content.add(layer_terrain_forest)    # Forest polygons
        rotated_content.add(layer_terrain_orchard)   # Orchard polygons
        rotated_content.add(layer_terrain_urban)     # Urban polygons
        rotated_content.add(layer_farmland)          # Farmland areas
        # New terrain-like features
        rotated_content.add(layer_mangrove)          # Mangrove areas
        rotated_content.add(layer_wetland)           # Wetland areas
        rotated_content.add(layer_heath)             # Heath/scrubland
        rotated_content.add(layer_rocky)             # Rocky terrain
        rotated_content.add(layer_sand)              # Sandy areas
        rotated_content.add(layer_quarries)          # Quarries
        rotated_content.add(layer_cemeteries)        # Cemeteries
        rotated_content.add(layer_military)          # Military areas
        rotated_content.add(layer_waterways_area)    # Water area polygons
        rotated_content.add(layer_streams)           # Streams/ditches
        rotated_content.add(layer_coastline)         # Coastline
        rotated_content.add(layer_dams)              # Dams
        rotated_content.add(layer_contours_regular)  # Regular contours
        rotated_content.add(layer_contours_index)    # Index contours
        rotated_content.add(layer_contour_labels)    # Contour elevation labels
        rotated_content.add(layer_cliffs)            # Cliffs/embankments
        rotated_content.add(layer_tree_rows)         # Tree rows
        rotated_content.add(layer_barriers)          # Fences/walls
        rotated_content.add(layer_powerlines)        # Power lines
        rotated_content.add(layer_railways)          # Railways
        rotated_content.add(layer_paths)             # Footpaths/cycleways
        # Infrastructure areas
        rotated_content.add(layer_airfields)         # Airfields/runways
        rotated_content.add(layer_ports)             # Ports/docks
        # Road layers (bottom to top by importance)
        rotated_content.add(layer_roads_service)     # Service roads
        rotated_content.add(layer_roads_residential) # Residential roads
        rotated_content.add(layer_roads_track)       # Tracks
        rotated_content.add(layer_roads_minor)       # Minor roads (tertiary)
        rotated_content.add(layer_roads_major)       # Major roads (primary/secondary)
        rotated_content.add(layer_roads_highway)     # Highways
        rotated_content.add(layer_bridges)           # Bridges
        rotated_content.add(layer_buildings)         # Building footprints
        # Point features
        rotated_content.add(layer_towers)            # Towers/masts
        rotated_content.add(layer_fuel)              # Fuel infrastructure
        rotated_content.add(layer_caves)             # Cave entrances
        rotated_content.add(layer_peaks)             # Peaks/summits
        rotated_content.add(layer_places)            # Place labels
        rotated_content.add(layer_mgrs_grid)         # MGRS grid (real-world coords)
        # Reference tiles at very bottom (hidden, for artist reference)
        master_group.add(rotated_reference)
        # Ocean layer below terrain (terrain covers islands)
        master_group.add(layer_ocean)
        # Base open terrain matches hex grid (unrotated)
        master_group.add(layer_terrain_open)
        master_group.add(rotated_content)
        # Out-of-play frame stays UNROTATED - clips the rotated content
        master_group.add(layer_out_of_play_frame)
        # Hex grid stays axis-aligned (unrotated)
        master_group.add(layer_hex_grid)
        master_group.add(layer_hex_markers)
        master_group.add(layer_hex_labels)
        # MGRS labels above frame so they're visible in margins
        master_group.add(layer_mgrs_labels)
    else:
        # Non-rotated layer order
        master_group.add(layer_reference_tiles)   # Reference tiles (hidden)
        master_group.add(layer_ocean)             # Ocean (below terrain, terrain covers islands)
        master_group.add(layer_terrain_open)      # Open terrain fill
        master_group.add(layer_terrain_water)     # Water polygons (ponds, etc.)
        master_group.add(layer_terrain_marsh)     # Marsh polygons
        master_group.add(layer_terrain_forest)    # Forest polygons
        master_group.add(layer_terrain_orchard)   # Orchard polygons
        master_group.add(layer_terrain_urban)     # Urban polygons
        master_group.add(layer_farmland)          # Farmland areas
        # New terrain-like features
        master_group.add(layer_mangrove)          # Mangrove areas
        master_group.add(layer_wetland)           # Wetland areas
        master_group.add(layer_heath)             # Heath/scrubland
        master_group.add(layer_rocky)             # Rocky terrain
        master_group.add(layer_sand)              # Sandy areas
        master_group.add(layer_quarries)          # Quarries
        master_group.add(layer_cemeteries)        # Cemeteries
        master_group.add(layer_military)          # Military areas
        master_group.add(layer_waterways_area)    # Water area polygons
        master_group.add(layer_streams)           # Streams/ditches
        master_group.add(layer_coastline)         # Coastline
        master_group.add(layer_dams)              # Dams
        master_group.add(layer_contours_regular)  # Regular contours
        master_group.add(layer_contours_index)    # Index contours
        master_group.add(layer_contour_labels)    # Contour elevation labels
        master_group.add(layer_cliffs)            # Cliffs/embankments
        master_group.add(layer_tree_rows)         # Tree rows
        master_group.add(layer_barriers)          # Fences/walls
        master_group.add(layer_powerlines)        # Power lines
        master_group.add(layer_railways)          # Railways
        master_group.add(layer_paths)             # Footpaths/cycleways
        # Infrastructure areas
        master_group.add(layer_airfields)         # Airfields/runways
        master_group.add(layer_ports)             # Ports/docks
        # Road layers (bottom to top by importance)
        master_group.add(layer_roads_service)     # Service roads
        master_group.add(layer_roads_residential) # Residential roads
        master_group.add(layer_roads_track)       # Tracks
        master_group.add(layer_roads_minor)       # Minor roads (tertiary)
        master_group.add(layer_roads_major)       # Major roads (primary/secondary)
        master_group.add(layer_roads_highway)     # Highways
        master_group.add(layer_bridges)           # Bridges
        master_group.add(layer_buildings)         # Building footprints
        # Point features
        master_group.add(layer_towers)            # Towers/masts
        master_group.add(layer_fuel)              # Fuel infrastructure
        master_group.add(layer_caves)             # Cave entrances
        master_group.add(layer_peaks)             # Peaks/summits
        master_group.add(layer_places)            # Place labels
        master_group.add(layer_out_of_play_frame) # Frame masks edges
        master_group.add(layer_hex_grid)          # Hex grid lines
        master_group.add(layer_hex_markers)       # Center circles
        master_group.add(layer_hex_labels)        # Hex coordinate labels
        master_group.add(layer_mgrs_grid)         # MGRS grid lines
        master_group.add(layer_mgrs_labels)       # MGRS labels

    # Add master group to drawing (contains clipped map content)
    dwg.add(master_group)

    # Map data and compass are OUTSIDE the clipped group
    # They render in the data margin area (outside trim, not printed)
    dwg.add(layer_map_data)          # Map metadata block
    dwg.add(layer_compass)           # Compass rose

    # Add trim and bleed guide lines (for print setup reference)
    layer_print_guides = dwg.g(id="Print_Guides")

    # Bleed line - outer boundary of bleed area (cyan, dashed)
    # Bleed starts at data_margin from document edge
    bleed_line_x = data_margin_m
    bleed_line_y = data_margin_m
    bleed_line_w = trim_width_m + 2 * bleed_m
    bleed_line_h = trim_height_m + 2 * bleed_m
    layer_print_guides.add(dwg.rect(
        (bleed_line_x, bleed_line_y),
        (bleed_line_w, bleed_line_h),
        fill="none",
        stroke="#00FFFF",  # Cyan
        stroke_width=2,
        stroke_dasharray="20,10",
    ))

    # Trim line - where paper will be cut (magenta, solid)
    # Trim starts at data_margin + bleed from document edge
    trim_line_x = content_offset_m
    trim_line_y = content_offset_m
    layer_print_guides.add(dwg.rect(
        (trim_line_x, trim_line_y),
        (trim_width_m, trim_height_m),
        fill="none",
        stroke="#FF00FF",  # Magenta
        stroke_width=2,
    ))

    # Add labels for the guide lines
    guide_label_size = DATA_FONT_SIZE_M * 0.7
    # Bleed label (top-left corner of bleed line)
    layer_print_guides.add(dwg.text(
        "BLEED",
        insert=(bleed_line_x + 10, bleed_line_y - 5),
        font_size=guide_label_size,
        fill="#00FFFF",
        font_family="sans-serif",
    ))
    # Trim label (top-left corner of trim line)
    layer_print_guides.add(dwg.text(
        "TRIM (34\" x 22\")",
        insert=(trim_line_x + 10, trim_line_y - 5),
        font_size=guide_label_size,
        fill="#FF00FF",
        font_family="sans-serif",
    ))

    dwg.add(layer_print_guides)
    print("  Added print guide lines (trim=magenta, bleed=cyan)")

    # Save
    dwg.save()
    print(f"  Saved to {output_path}")


def load_config_from_file() -> Optional[MapConfig]:
    """Load configuration from map_config.json if it exists."""
    config_path = Path("map_config.json")
    if not config_path.exists():
        return None

    print(f"Loading configuration from {config_path}...")
    with open(config_path) as f:
        data = json.load(f)

    config = MapConfig(
        name=data.get("name", "unnamed"),
        center_lat=data["center_lat"],
        center_lon=data["center_lon"],
        region=data.get("region", ""),
        country=data.get("country", ""),
        rotation_deg=data.get("rotation_deg", 0),
    )

    return config


def get_available_country_pbfs() -> List[Tuple[str, Path]]:
    """Get list of available country PBF files in geofabrik directory.

    Returns list of (region_name, path) tuples, sorted by file size (smallest first).
    """
    if not GEOFABRIK_DIR.exists():
        return []

    pbfs = []
    for pbf_file in GEOFABRIK_DIR.glob("*-latest.osm.pbf"):
        # Extract region name from filename (e.g., "japan-latest.osm.pbf" -> "japan")
        region = pbf_file.stem.replace("-latest.osm", "")
        pbfs.append((region, pbf_file))

    # Sort by file size (try smaller files first for faster failure)
    pbfs.sort(key=lambda x: x[1].stat().st_size)
    return pbfs


def auto_extract_mgrs_data(config: 'MapConfig') -> bool:
    """Try to extract MGRS data from available country PBFs.

    Returns True if extraction succeeded, False otherwise.
    """
    available_pbfs = get_available_country_pbfs()

    if not available_pbfs:
        return False

    # Parse region to get GZD and square
    parts = config.region.split("/")
    if len(parts) != 2:
        print(f"  Invalid region format: {config.region}")
        return False

    gzd, square = parts
    mgrs_square = f"{gzd} {square}"

    print(f"\nAttempting to extract data for {gzd}/{square}...")
    print(f"  Available country PBFs: {', '.join(r for r, _ in available_pbfs)}")

    for region, pbf_path in available_pbfs:
        print(f"\n  Trying {region}...")

        # Call the download script with this region
        # It will skip the download (PBF already exists) and just extract
        cmd = [
            "python3", "download_mgrs_data_osmium.py",
            "--region", region,
            mgrs_square
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout for extraction
            )

            # Check if extraction succeeded by looking for the data directory
            if config.data_path.exists() and (config.data_path / "elevation.tif").exists():
                print(f"\n  Successfully extracted data using {region} PBF")
                return True
            else:
                # Extraction ran but didn't produce data (wrong region)
                if "Error extracting region" in result.stderr or "0 features" in result.stdout:
                    print(f"    No data in {region} for this region")
                    continue

        except subprocess.TimeoutExpired:
            print(f"    Extraction timed out for {region}")
            continue
        except Exception as e:
            print(f"    Error: {e}")
            continue

    return False


def main():
    """Generate tactical map."""

    # Try to load from config file first
    config = load_config_from_file()

    if config is None:
        # Default: Taichung City center coordinates
        print("No map_config.json found, using default configuration...")
        config = MapConfig(
            name="taichung",
            center_lat=24.1477,
            center_lon=120.6736,
            region="taiwan",
            rotation_deg=30,
        )

    print("=" * 60)
    print(f"Tactical Map Generator: {config.name}")
    print("=" * 60)
    print(f"Center: {config.center_lat:.4f}°N, {config.center_lon:.4f}°E")
    print(f"Grid: {GRID_WIDTH} x {GRID_HEIGHT} hexes @ {HEX_SIZE_M}m")
    print(f"Map size: {(config.max_x - config.min_x)/1000:.1f}km x {(config.max_y - config.min_y)/1000:.1f}km")
    if config.rotation_deg != 0:
        print(f"Rotation: {config.rotation_deg}° clockwise")

    # Check if data directory exists and has required files
    dem_path = config.data_path / "elevation.tif"
    data_missing = not config.data_path.exists() or not dem_path.exists()

    if data_missing:
        print(f"\nData not found for {config.region}")

        # Try to auto-extract from available country PBFs
        if auto_extract_mgrs_data(config):
            print("Data extraction complete!")
        else:
            # No country PBF available - show error
            print(f"\n{'='*60}")
            print("ERROR: Could not extract map data!")
            print(f"{'='*60}")
            print(f"Expected: {config.data_path}")
            print(f"\nNo country PBF found that contains this region.")
            print(f"Download the appropriate country from Geofabrik first:")
            print(f"  curl -L -o data/geofabrik/COUNTRY-latest.osm.pbf \\")
            print(f"    https://download.geofabrik.de/asia/COUNTRY-latest.osm.pbf")
            print(f"\nAvailable regions: japan, taiwan, philippines, south-korea,")
            print(f"  indonesia, malaysia-singapore-brunei, vietnam, thailand")
            print(f"{'='*60}")
            return

    # Create output directory
    config.output_path.mkdir(parents=True, exist_ok=True)

    # Create hex grid
    grid = TacticalHexGrid(config)
    hex_coords = grid.generate_grid()
    print(f"Generated {len(hex_coords)} hex coordinates")

    # Load data
    print("\nLoading data...")
    dem = load_dem(config)
    print(f"  DEM: {dem.width}x{dem.height}")

    # Load landcover (uses expanded bounds for rotation)
    lc_path = config.data_path / "landcover.geojson"
    if lc_path.exists():
        print("  Loading landcover...")
        try:
            # Use on_invalid="ignore" to skip malformed geometries
            landcover = gpd.read_file(lc_path, on_invalid="ignore")
            landcover = landcover.to_crs(GRID_CRS)

            # Remove any rows with null/invalid geometries
            landcover = landcover[landcover.geometry.notna() & landcover.geometry.is_valid]

            # Clip to data bounds (expanded for rotation) with buffer
            map_bounds = box(
                config.data_min_x - 1000,
                config.data_min_y - 1000,
                config.data_max_x + 1000,
                config.data_max_y + 1000
            )
            landcover = landcover[landcover.intersects(map_bounds)]
            print(f"  Filtered to {len(landcover)} landcover polygons in bounds")
        except Exception as e:
            print(f"  Warning: Error loading landcover: {e}")
            landcover = gpd.GeoDataFrame()
    else:
        landcover = gpd.GeoDataFrame()

    # Load roads
    roads = load_roads(config)

    # Load buildings
    buildings = load_buildings(config)

    # Load enhanced features (optional - won't fail if files don't exist)
    print("Loading enhanced features...")
    streams = load_optional_features(config, "streams.geojson", "streams")
    paths = load_optional_features(config, "paths.geojson", "paths")
    barriers = load_optional_features(config, "barriers.geojson", "barriers")
    powerlines = load_optional_features(config, "powerlines.geojson", "powerlines")
    bridges = load_optional_features(config, "bridges.geojson", "bridges")
    tree_rows = load_optional_features(config, "tree_rows.geojson", "tree_rows")
    railways = load_optional_features(config, "railways.geojson", "railways")
    farmland = load_optional_features(config, "farmland.geojson", "farmland")
    cliffs = load_optional_features(config, "cliffs.geojson", "cliffs")
    waterways_area = load_optional_features(config, "waterways_area.geojson", "waterways_area")
    coastline = load_optional_features(config, "coastline.geojson", "coastline")

    # Load new features (added for tactical relevance)
    mangrove = load_optional_features(config, "mangrove.geojson", "mangrove")
    wetland = load_optional_features(config, "wetland.geojson", "wetland")
    heath = load_optional_features(config, "heath.geojson", "heath")
    rocky_terrain = load_optional_features(config, "rocky_terrain.geojson", "rocky_terrain")
    sand = load_optional_features(config, "sand.geojson", "sand")
    military = load_optional_features(config, "military.geojson", "military")
    quarries = load_optional_features(config, "quarries.geojson", "quarries")
    cemeteries = load_optional_features(config, "cemeteries.geojson", "cemeteries")
    places = load_optional_features(config, "places.geojson", "places")
    peaks = load_optional_features(config, "peaks.geojson", "peaks")
    caves = load_optional_features(config, "caves.geojson", "caves")
    dams = load_optional_features(config, "dams.geojson", "dams")
    airfields = load_optional_features(config, "airfields.geojson", "airfields")
    ports = load_optional_features(config, "ports.geojson", "ports")
    towers = load_optional_features(config, "towers.geojson", "towers")
    fuel_infrastructure = load_optional_features(config, "fuel_infrastructure.geojson", "fuel_infrastructure")

    # Download/load high-res reference tiles for the map area
    # Falls back to MGRS-level tiles if high-res download fails
    reference_tiles = download_highres_reference_tiles(config)
    if reference_tiles is None:
        reference_tiles = load_reference_tile_info(config)

    # Generate contours
    contours = generate_contours(dem, config)

    # Generate MGRS grid
    mgrs_grid = generate_mgrs_grid(config)

    # Classify terrain
    hexes = classify_terrain(grid, hex_coords, dem, landcover, config)

    # Bundle enhanced features
    enhanced_features = {
        'streams': streams,
        'paths': paths,
        'barriers': barriers,
        'powerlines': powerlines,
        'bridges': bridges,
        'tree_rows': tree_rows,
        'railways': railways,
        'farmland': farmland,
        'cliffs': cliffs,
        'waterways_area': waterways_area,
        'coastline': coastline,
        'reference_tiles': reference_tiles,
        # New tactical features
        'mangrove': mangrove,
        'wetland': wetland,
        'heath': heath,
        'rocky_terrain': rocky_terrain,
        'sand': sand,
        'military': military,
        'quarries': quarries,
        'cemeteries': cemeteries,
        'places': places,
        'peaks': peaks,
        'caves': caves,
        'dams': dams,
        'airfields': airfields,
        'ports': ports,
        'towers': towers,
        'fuel_infrastructure': fuel_infrastructure,
    }

    # Render SVG
    svg_path = config.output_path / f"{config.name}_tactical.svg"
    render_tactical_svg(grid, hexes, contours, roads, buildings, landcover, mgrs_grid, config, svg_path, enhanced_features, dem)

    # Save hex data
    json_path = config.output_path / f"{config.name}_hexdata.json"
    hex_data = {
        "metadata": {
            "name": config.name,
            "center_lat": config.center_lat,
            "center_lon": config.center_lon,
            "hex_size_m": HEX_SIZE_M,
            "grid_width": GRID_WIDTH,
            "grid_height": GRID_HEIGHT,
        },
        "hexes": [
            {
                "coord": [h.q, h.r],
                "terrain": h.terrain,
                "elevation": h.elevation_avg,
            }
            for h in hexes
        ],
    }
    with open(json_path, "w") as f:
        json.dump(hex_data, f, indent=2)
    print(f"  Saved hex data to {json_path}")

    dem.close()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
