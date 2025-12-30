"""
Utility classes for tactical map generation.

This module provides reusable components for coordinate transformation,
bounds management, rotation handling, and SVG layer management.
"""

import math
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Any
from pyproj import Transformer


@dataclass
class Bounds:
    """Represents a rectangular bounds in a coordinate system.

    Attributes:
        min_x: Western/left boundary
        max_x: Eastern/right boundary
        min_y: Southern/bottom boundary
        max_y: Northern/top boundary
    """
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    @property
    def width(self) -> float:
        """Width of the bounds (east-west extent)."""
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        """Height of the bounds (north-south extent)."""
        return self.max_y - self.min_y

    @property
    def center(self) -> Tuple[float, float]:
        """Center point of the bounds as (x, y)."""
        return (
            (self.min_x + self.max_x) / 2,
            (self.min_y + self.max_y) / 2
        )

    def contains(self, x: float, y: float) -> bool:
        """Check if a point is within bounds."""
        return (self.min_x <= x <= self.max_x and
                self.min_y <= y <= self.max_y)

    def expand(self, buffer: float) -> 'Bounds':
        """Return a new Bounds expanded by buffer in all directions."""
        return Bounds(
            min_x=self.min_x - buffer,
            max_x=self.max_x + buffer,
            min_y=self.min_y - buffer,
            max_y=self.max_y + buffer
        )

    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Return bounds as (min_x, min_y, max_x, max_y) tuple."""
        return (self.min_x, self.min_y, self.max_x, self.max_y)


@dataclass
class RotationConfig:
    """Configuration for map rotation.

    Centralizes all rotation-related calculations and provides helper methods
    for rotating points and calculating expanded bounds.

    Attributes:
        angle_deg: Rotation angle in degrees (positive = clockwise in SVG)
        center_x: X coordinate of rotation center
        center_y: Y coordinate of rotation center
    """
    angle_deg: float
    center_x: float
    center_y: float

    @property
    def angle_rad(self) -> float:
        """Rotation angle in radians."""
        return math.radians(self.angle_deg)

    @property
    def is_rotated(self) -> bool:
        """Check if any rotation is applied."""
        return self.angle_deg != 0

    @property
    def cos_angle(self) -> float:
        """Cosine of rotation angle."""
        return math.cos(self.angle_rad)

    @property
    def sin_angle(self) -> float:
        """Sine of rotation angle."""
        return math.sin(self.angle_rad)

    def rotate_point(self, x: float, y: float) -> Tuple[float, float]:
        """Rotate a point around the rotation center.

        Args:
            x: X coordinate of point to rotate
            y: Y coordinate of point to rotate

        Returns:
            Tuple of (rotated_x, rotated_y)
        """
        if not self.is_rotated:
            return (x, y)

        dx = x - self.center_x
        dy = y - self.center_y

        rotated_x = self.center_x + dx * self.cos_angle - dy * self.sin_angle
        rotated_y = self.center_y + dx * self.sin_angle + dy * self.cos_angle

        return (rotated_x, rotated_y)

    def calculate_expanded_bounds(self, width: float, height: float) -> Tuple[float, float]:
        """Calculate how much to expand bounds to cover rotated rectangle.

        When a rectangle is rotated, its axis-aligned bounding box is larger.
        This calculates how much extra space is needed in each direction.

        Args:
            width: Width of the original rectangle
            height: Height of the original rectangle

        Returns:
            Tuple of (expand_x, expand_y) - extra space needed in each direction
        """
        if not self.is_rotated:
            return (0.0, 0.0)

        # For rectangle W x H rotated by θ:
        #   rotated_width = |W * cos(θ)| + |H * sin(θ)|
        #   rotated_height = |W * sin(θ)| + |H * cos(θ)|
        abs_cos = abs(self.cos_angle)
        abs_sin = abs(self.sin_angle)

        rotated_width = width * abs_cos + height * abs_sin
        rotated_height = width * abs_sin + height * abs_cos

        # Expansion needed in each direction, plus safety buffer
        expand_x = (rotated_width - width) / 2 + 500
        expand_y = (rotated_height - height) / 2 + 500

        return (expand_x, expand_y)

    def get_svg_transform(self) -> str:
        """Get SVG transform attribute string for this rotation."""
        if not self.is_rotated:
            return ""
        return f"rotate({self.angle_deg}, {self.center_x}, {self.center_y})"


class CoordinateTransformer:
    """Handles coordinate transformations between different reference systems.

    Supports transformations between:
    - WGS84 (EPSG:4326) - Geographic coordinates (lat/lon)
    - UTM (EPSG:326XX) - Projected meters for a specific zone
    - SVG coordinates - Screen/document coordinates with Y-axis inverted

    Attributes:
        grid_crs: The projected CRS used for calculations (usually UTM)
        wgs84_crs: WGS84 geographic CRS
        map_bounds: The map bounds in grid CRS
        svg_offset_x: X offset for SVG coordinate conversion
        svg_offset_y: Y offset for SVG coordinate conversion (includes Y-flip)
    """

    WGS84 = "EPSG:4326"

    def __init__(
        self,
        grid_crs: str,
        map_bounds: Bounds,
        svg_offset_x: float = 0,
        svg_offset_y: float = 0,
        debug: bool = False
    ):
        """Initialize the coordinate transformer.

        Args:
            grid_crs: Projected CRS string (e.g., "EPSG:32651")
            map_bounds: Map bounds in the grid CRS
            svg_offset_x: Additional X offset for SVG coordinates
            svg_offset_y: Additional Y offset for SVG coordinates
            debug: Enable debug logging of transformations
        """
        self.grid_crs = grid_crs
        self.map_bounds = map_bounds
        self.svg_offset_x = svg_offset_x
        self.svg_offset_y = svg_offset_y
        self.debug = debug

        # Create transformers
        self._to_wgs84 = Transformer.from_crs(grid_crs, self.WGS84, always_xy=True)
        self._from_wgs84 = Transformer.from_crs(self.WGS84, grid_crs, always_xy=True)

        self._debug_log = []

    def utm_to_svg(self, x: float, y: float) -> Tuple[float, float]:
        """Convert UTM/grid coordinates to SVG coordinates.

        SVG Y-axis is inverted (increases downward), so we flip Y.

        Args:
            x: Easting in meters
            y: Northing in meters

        Returns:
            Tuple of (svg_x, svg_y)
        """
        svg_x = (x - self.map_bounds.min_x) + self.svg_offset_x
        svg_y = (self.map_bounds.max_y - y) + self.svg_offset_y

        if self.debug:
            self._debug_log.append({
                'operation': 'utm_to_svg',
                'input': (x, y),
                'output': (svg_x, svg_y)
            })

        return (svg_x, svg_y)

    def svg_to_utm(self, svg_x: float, svg_y: float) -> Tuple[float, float]:
        """Convert SVG coordinates back to UTM/grid coordinates.

        Inverse of utm_to_svg, useful for debugging.

        Args:
            svg_x: SVG X coordinate
            svg_y: SVG Y coordinate

        Returns:
            Tuple of (easting, northing)
        """
        x = (svg_x - self.svg_offset_x) + self.map_bounds.min_x
        y = self.map_bounds.max_y - (svg_y - self.svg_offset_y)
        return (x, y)

    def wgs84_to_utm(self, lon: float, lat: float) -> Tuple[float, float]:
        """Convert WGS84 coordinates to UTM/grid coordinates.

        Args:
            lon: Longitude in degrees
            lat: Latitude in degrees

        Returns:
            Tuple of (easting, northing)
        """
        return self._from_wgs84.transform(lon, lat)

    def utm_to_wgs84(self, x: float, y: float) -> Tuple[float, float]:
        """Convert UTM/grid coordinates to WGS84.

        Args:
            x: Easting in meters
            y: Northing in meters

        Returns:
            Tuple of (longitude, latitude)
        """
        return self._to_wgs84.transform(x, y)

    def wgs84_to_svg(self, lon: float, lat: float) -> Tuple[float, float]:
        """Convert WGS84 coordinates directly to SVG coordinates.

        Convenience method combining wgs84_to_utm and utm_to_svg.

        Args:
            lon: Longitude in degrees
            lat: Latitude in degrees

        Returns:
            Tuple of (svg_x, svg_y)
        """
        x, y = self.wgs84_to_utm(lon, lat)
        return self.utm_to_svg(x, y)

    def get_debug_log(self) -> List[Dict]:
        """Get the debug log of transformations (if debug=True)."""
        return self._debug_log

    def clear_debug_log(self):
        """Clear the debug log."""
        self._debug_log = []


class LayerManager:
    """Manages SVG layer groups and their z-ordering.

    Layers are registered with a z-order value (higher = on top).
    Supports rotation groups where some layers rotate with terrain
    and others stay fixed.

    Attributes:
        layers: Dictionary mapping layer ID to layer info
    """

    def __init__(self, dwg):
        """Initialize the layer manager.

        Args:
            dwg: svgwrite Drawing object
        """
        self.dwg = dwg
        self.layers: Dict[str, Dict[str, Any]] = {}
        self._groups: Dict[str, Any] = {}

    def register_layer(
        self,
        layer_id: str,
        z_order: int,
        rotates: bool = True,
        visible: bool = True,
        clip_path: Optional[str] = None
    ) -> Any:
        """Register and create a new layer group.

        Args:
            layer_id: Unique identifier for the layer
            z_order: Stacking order (higher values render on top)
            rotates: Whether this layer should rotate with the map
            visible: Whether the layer is visible by default
            clip_path: Optional clip-path URL for the layer

        Returns:
            The created SVG group element
        """
        group = self.dwg.g(id=layer_id)

        if not visible:
            group['visibility'] = 'hidden'

        if clip_path:
            group['clip-path'] = clip_path

        self.layers[layer_id] = {
            'group': group,
            'z_order': z_order,
            'rotates': rotates,
            'visible': visible
        }

        self._groups[layer_id] = group
        return group

    def get_layer(self, layer_id: str) -> Any:
        """Get a layer group by ID."""
        return self._groups.get(layer_id)

    def get_layers_by_z_order(self, rotates: Optional[bool] = None) -> List[Any]:
        """Get layers sorted by z-order.

        Args:
            rotates: If specified, filter to only rotating or non-rotating layers

        Returns:
            List of layer groups sorted by z-order (lowest first)
        """
        filtered = self.layers.items()

        if rotates is not None:
            filtered = [(k, v) for k, v in filtered if v['rotates'] == rotates]

        sorted_layers = sorted(filtered, key=lambda x: x[1]['z_order'])
        return [info['group'] for _, info in sorted_layers]

    def get_rotating_layers(self) -> List[Any]:
        """Get all layers that should rotate with the map."""
        return self.get_layers_by_z_order(rotates=True)

    def get_fixed_layers(self) -> List[Any]:
        """Get all layers that should remain fixed (not rotate)."""
        return self.get_layers_by_z_order(rotates=False)

    def assemble_into_group(
        self,
        parent_group: Any,
        layers: List[Any],
        transform: Optional[str] = None
    ):
        """Assemble layers into a parent group.

        Args:
            parent_group: The parent SVG group to add layers to
            layers: List of layer groups to add
            transform: Optional transform to apply to parent group
        """
        if transform:
            parent_group['transform'] = transform

        for layer in layers:
            parent_group.add(layer)


# Z-order constants for standard layers
class LayerZOrder:
    """Standard z-order values for map layers.

    Lower values render first (underneath).
    """
    # Background layers
    BACKGROUND = 0
    REFERENCE_TILES = 10

    # Terrain layers
    OCEAN = 100
    TERRAIN_OPEN = 110
    TERRAIN_WATER = 120
    TERRAIN_MARSH = 130
    TERRAIN_FOREST = 140
    TERRAIN_ORCHARD = 150
    TERRAIN_URBAN = 160
    FARMLAND = 170

    # Natural features
    MANGROVE = 200
    WETLAND = 210
    HEATH = 220
    ROCKY = 230
    SAND = 240

    # Man-made features (terrain-like)
    QUARRIES = 300
    CEMETERIES = 310
    MILITARY = 320

    # Water features
    WATERWAYS_AREA = 400
    STREAMS = 410
    COASTLINE = 420
    DAMS = 430

    # Contours
    CONTOURS_REGULAR = 500
    CONTOURS_INDEX = 510
    CONTOUR_LABELS = 520

    # Terrain features
    CLIFFS = 600
    TREE_ROWS = 610
    BARRIERS = 620

    # Infrastructure
    POWERLINES = 700
    RAILWAYS = 710
    PATHS = 720
    AIRFIELDS = 730
    PORTS = 740

    # Roads (layered by importance)
    ROADS_SERVICE = 800
    ROADS_RESIDENTIAL = 810
    ROADS_TRACK = 820
    ROADS_MINOR = 830
    ROADS_MAJOR = 840
    ROADS_HIGHWAY = 850
    BRIDGES = 860

    # Buildings and structures
    BUILDINGS = 900
    TOWERS = 910
    FUEL = 920
    CAVES = 930

    # Labels and markers
    PEAKS = 1000
    PLACES = 1010
    MGRS_GRID = 1020

    # Fixed layers (don't rotate)
    OUT_OF_PLAY_FRAME = 2000
    HEX_GRID = 2100
    HEX_MARKERS = 2110
    HEX_LABELS = 2120
    MGRS_LABELS = 2130

    # Document elements
    MAP_DATA = 3000
    COMPASS_ROSE = 3010
    PRINT_GUIDES = 3020
