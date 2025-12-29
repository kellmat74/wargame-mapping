"""
hexgrid.py - Core hex grid geometry for wargame maps

Flat-top hexagon orientation with axial coordinates (q, r).
"""

import math
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import svgwrite
from shapely.geometry import Polygon


# === Terrain Colors (from spec) ===
TERRAIN_COLORS = {
    "water": "#4a90d9",
    "urban": "#8b4513",
    "paddy": "#98fb98",
    "forest": "#228b22",
    "rough": "#d2b48c",
    "mountain": "#a0522d",
    "clear": "#f5f5dc",
}

# === SVG Style Constants ===
HEX_STROKE_COLOR = "#333333"
HEX_STROKE_WIDTH = 0.5
COORD_FONT_SIZE = 8
COORD_FONT_COLOR = "#666666"


@dataclass
class HexGrid:
    """
    A hex grid covering a geographic area.
    
    Attributes:
        hex_size_m: Distance from hex center to center (in meters)
        origin_x: X coordinate of grid origin (projected CRS)
        origin_y: Y coordinate of grid origin (projected CRS)
    """
    hex_size_m: float
    origin_x: float
    origin_y: float
    
    @property
    def size(self) -> float:
        """Hex size = radius from center to vertex."""
        # For flat-top: width = sqrt(3) * size, so size = width / sqrt(3)
        return self.hex_size_m / math.sqrt(3)

    @property
    def hex_width(self) -> float:
        """Width of hex (flat edge to flat edge) for flat-top orientation."""
        return self.hex_size_m

    @property
    def hex_height(self) -> float:
        """Height of hex (point to point) for flat-top orientation."""
        return 2 * self.size

    @property
    def col_spacing(self) -> float:
        """Horizontal distance between hex centers (adjacent columns)."""
        # For flat-top hexes: 3/2 * size (where size = center-to-vertex radius)
        return 1.5 * self.size

    @property
    def row_spacing(self) -> float:
        """Vertical distance between hex centers (same column)."""
        # For flat-top hexes: sqrt(3) * size = flat-to-flat distance
        return self.hex_width
    
    def axial_to_world(self, q: int, r: int) -> tuple[float, float]:
        """
        Convert axial coordinates (q, r) to world coordinates (x, y).
        
        Args:
            q: Column coordinate (increases east)
            r: Row coordinate (increases south-southeast)
            
        Returns:
            (x, y) in projected CRS units (meters)
        """
        x = self.origin_x + q * self.col_spacing
        y = self.origin_y - r * self.row_spacing - (q % 2) * self.row_spacing * 0.5
        return (x, y)
    
    def world_to_axial(self, x: float, y: float) -> tuple[int, int]:
        """
        Convert world coordinates to nearest axial hex coordinate.
        
        Args:
            x, y: Coordinates in projected CRS
            
        Returns:
            (q, r) axial coordinates of containing hex
        """
        # Approximate q
        q = round((x - self.origin_x) / self.col_spacing)
        
        # Adjust y for column offset
        adjusted_y = self.origin_y - y
        if q % 2 == 1:
            adjusted_y -= self.row_spacing * 0.5
        
        r = round(adjusted_y / self.row_spacing)
        
        return (int(q), int(r))
    
    def hex_polygon(self, q: int, r: int) -> Polygon:
        """
        Generate a Shapely Polygon for the hex at (q, r).
        
        Returns:
            Shapely Polygon with 6 vertices (flat-top orientation)
        """
        cx, cy = self.axial_to_world(q, r)
        
        # Radius from center to vertex
        radius = self.hex_height / 2
        
        # Generate vertices for flat-top hex (start at rightmost vertex, 0°)
        vertices = []
        for i in range(6):
            angle = i * math.pi / 3  # 0°, 60°, 120°, 180°, 240°, 300°
            vx = cx + radius * math.cos(angle)
            vy = cy + radius * math.sin(angle)
            vertices.append((vx, vy))
        
        return Polygon(vertices)
    
    def hex_edge_midpoints(self, q: int, r: int) -> list[tuple[float, float]]:
        """
        Get midpoints of all 6 hex edges, for marking rivers/borders.
        
        Returns:
            List of 6 (x, y) tuples, edges numbered 0-5 clockwise from top.
        """
        poly = self.hex_polygon(q, r)
        coords = list(poly.exterior.coords)[:-1]  # Remove duplicate closing point
        
        midpoints = []
        for i in range(6):
            x1, y1 = coords[i]
            x2, y2 = coords[(i + 1) % 6]
            midpoints.append(((x1 + x2) / 2, (y1 + y2) / 2))
        
        return midpoints
    
    def generate_grid(
        self, 
        min_x: float, 
        min_y: float, 
        max_x: float, 
        max_y: float
    ) -> Iterator[tuple[int, int]]:
        """
        Generate all hex coordinates covering a bounding box.
        
        Args:
            min_x, min_y, max_x, max_y: Bounding box in world coordinates
            
        Yields:
            (q, r) tuples for each hex in the grid
        """
        # Find range of q values
        q_min = int((min_x - self.origin_x) / self.col_spacing) - 1
        q_max = int((max_x - self.origin_x) / self.col_spacing) + 1
        
        # Find range of r values (approximate)
        r_min = int((self.origin_y - max_y) / self.row_spacing) - 1
        r_max = int((self.origin_y - min_y) / self.row_spacing) + 1
        
        for q in range(q_min, q_max + 1):
            for r in range(r_min, r_max + 1):
                yield (q, r)


def hex_neighbors(q: int, r: int) -> list[tuple[int, int]]:
    """
    Get the 6 neighboring hex coordinates.
    
    Returns neighbors in order matching edge indices (0-5 clockwise from top).
    """
    # Offset depends on whether q is even or odd (flat-top)
    if q % 2 == 0:
        return [
            (q, r - 1),      # Edge 0: top
            (q + 1, r - 1),  # Edge 1: top-right
            (q + 1, r),      # Edge 2: bottom-right
            (q, r + 1),      # Edge 3: bottom
            (q - 1, r),      # Edge 4: bottom-left
            (q - 1, r - 1),  # Edge 5: top-left
        ]
    else:
        return [
            (q, r - 1),      # Edge 0: top
            (q + 1, r),      # Edge 1: top-right
            (q + 1, r + 1),  # Edge 2: bottom-right
            (q, r + 1),      # Edge 3: bottom
            (q - 1, r + 1),  # Edge 4: bottom-left
            (q - 1, r),      # Edge 5: top-left
        ]


@dataclass
class HexData:
    """Data for a single hex cell."""
    q: int
    r: int
    terrain: str = "clear"
    elevation: int = 0
    coastal_edges: list[int] = field(default_factory=list)
    river_edges: list[int] = field(default_factory=list)
    city: dict | None = None
    features: list[str] = field(default_factory=list)


def render_svg(
    grid: HexGrid,
    hexes: list[HexData],
    output_path: str,
    show_coords: bool = True,
    margin_px: int = 20,
    scale: float = 0.01,
    road_connections: list[dict] | None = None,
) -> None:
    """
    Render hex grid to SVG file with all features.

    Args:
        grid: HexGrid instance defining geometry
        hexes: List of HexData objects to render
        output_path: Path to output SVG file
        show_coords: Whether to display (q, r) labels
        margin_px: Margin around the map in pixels
        scale: Scale factor (world units to pixels). 0.01 = 1m -> 0.01px
        road_connections: List of road connections between hexes
    """
    if not hexes:
        print("No hexes to render")
        return

    # Calculate bounding box of all hexes
    all_x = []
    all_y = []
    for h in hexes:
        poly = grid.hex_polygon(h.q, h.r)
        coords = list(poly.exterior.coords)
        all_x.extend(c[0] for c in coords)
        all_y.extend(c[1] for c in coords)

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # SVG dimensions
    width_px = (max_x - min_x) * scale + 2 * margin_px
    height_px = (max_y - min_y) * scale + 2 * margin_px

    # Create SVG
    dwg = svgwrite.Drawing(
        output_path,
        size=(f"{width_px:.0f}px", f"{height_px:.0f}px"),
        viewBox=f"0 0 {width_px:.0f} {height_px:.0f}",
    )

    # Background
    dwg.add(dwg.rect(insert=(0, 0), size=("100%", "100%"), fill="white"))

    # Helper to convert world coords to SVG coords
    def to_svg(x: float, y: float) -> tuple[float, float]:
        sx = (x - min_x) * scale + margin_px
        sy = (max_y - y) * scale + margin_px  # Flip Y axis
        return (sx, sy)

    # === Layer 1: Hex terrain fills ===
    for h in hexes:
        poly = grid.hex_polygon(h.q, h.r)
        coords = list(poly.exterior.coords)
        svg_points = [to_svg(x, y) for x, y in coords]

        fill_color = TERRAIN_COLORS.get(h.terrain, TERRAIN_COLORS["clear"])

        dwg.add(dwg.polygon(
            points=svg_points,
            fill=fill_color,
            stroke=HEX_STROKE_COLOR,
            stroke_width=HEX_STROKE_WIDTH,
        ))

    # === Layer 2: Rivers on hex edges ===
    for h in hexes:
        if not h.river_edges:
            continue

        poly = grid.hex_polygon(h.q, h.r)
        coords = list(poly.exterior.coords)

        for edge_idx in h.river_edges:
            p1 = coords[edge_idx]
            p2 = coords[(edge_idx + 1) % 6]
            svg_p1 = to_svg(*p1)
            svg_p2 = to_svg(*p2)

            dwg.add(dwg.line(
                start=svg_p1,
                end=svg_p2,
                stroke="#0066cc",
                stroke_width=2,
                stroke_linecap="round",
            ))

    # === Layer 3: Roads between hexes ===
    if road_connections:
        road_colors = {
            'highway': '#cc0000',
            'major_road': '#ff6600',
            'minor_road': '#ffaa00',
        }
        road_widths = {
            'highway': 2.5,
            'major_road': 1.8,
            'minor_road': 1.2,
        }

        for conn in road_connections:
            q1, r1 = conn['from']
            q2, r2 = conn['to']
            road_type = conn.get('type', 'minor_road')

            cx1, cy1 = grid.axial_to_world(q1, r1)
            cx2, cy2 = grid.axial_to_world(q2, r2)

            svg_p1 = to_svg(cx1, cy1)
            svg_p2 = to_svg(cx2, cy2)

            dwg.add(dwg.line(
                start=svg_p1,
                end=svg_p2,
                stroke=road_colors.get(road_type, '#ffaa00'),
                stroke_width=road_widths.get(road_type, 1.2),
                stroke_linecap="round",
            ))

    # === Layer 4: City markers and labels ===
    for h in hexes:
        if not h.city:
            continue

        cx, cy = grid.axial_to_world(h.q, h.r)
        sx, sy = to_svg(cx, cy)

        city_size = h.city.get('size', 'small')
        city_name = h.city.get('name', '')

        # Marker sizes based on city size
        marker_radius = {
            'major': 6,
            'large': 4.5,
            'medium': 3,
            'small': 2,
        }.get(city_size, 2)

        # Draw city marker (filled circle with border)
        dwg.add(dwg.circle(
            center=(sx, sy),
            r=marker_radius,
            fill="#000000",
            stroke="#ffffff",
            stroke_width=0.5,
        ))

        # Add label for major and large cities
        if city_size in ('major', 'large') and city_name:
            dwg.add(dwg.text(
                city_name,
                insert=(sx + marker_radius + 2, sy + 1),
                font_size=7 if city_size == 'major' else 5,
                fill="#000000",
                font_family="sans-serif",
                font_weight="bold" if city_size == 'major' else "normal",
            ))

    # === Layer 5: Infrastructure markers ===
    for h in hexes:
        if not h.features:
            continue

        cx, cy = grid.axial_to_world(h.q, h.r)
        sx, sy = to_svg(cx, cy)

        # Offset if hex also has a city
        offset_y = -8 if h.city else 0

        for feature in h.features:
            if feature == 'airfield':
                # Draw airplane symbol (simple triangle)
                dwg.add(dwg.polygon(
                    points=[
                        (sx, sy + offset_y - 4),
                        (sx - 3, sy + offset_y + 2),
                        (sx + 3, sy + offset_y + 2),
                    ],
                    fill="#660099",
                    stroke="#ffffff",
                    stroke_width=0.3,
                ))
            elif feature == 'port':
                # Draw anchor symbol (small square)
                dwg.add(dwg.rect(
                    insert=(sx - 2, sy + offset_y - 2),
                    size=(4, 4),
                    fill="#006699",
                    stroke="#ffffff",
                    stroke_width=0.3,
                ))

    # === Layer 6: Coordinate labels (optional) ===
    if show_coords:
        for h in hexes:
            cx, cy = grid.axial_to_world(h.q, h.r)
            sx, sy = to_svg(cx, cy)
            dwg.add(dwg.text(
                f"{h.q},{h.r}",
                insert=(sx, sy),
                text_anchor="middle",
                dominant_baseline="middle",
                font_size=COORD_FONT_SIZE,
                fill=COORD_FONT_COLOR,
                font_family="monospace",
            ))

    # Save
    dwg.save()
    print(f"Saved SVG to {output_path} ({width_px:.0f}x{height_px:.0f}px)")


def render_svg_outline(
    grid: HexGrid,
    hex_coords: list[tuple[int, int]],
    output_path: str,
    show_coords: bool = True,
    margin_px: int = 20,
    scale: float = 0.01,
) -> None:
    """
    Render hex grid outline only (no terrain data).

    Convenience function for testing grid geometry.
    """
    hexes = [HexData(q=q, r=r) for q, r in hex_coords]
    render_svg(grid, hexes, output_path, show_coords, margin_px, scale)


# === Quick test ===
if __name__ == "__main__":
    import os

    # Taiwan approximate bounds in EPSG:3826 (TWD97)
    # These are rough values - adjust based on actual QGIS export
    TAIWAN_ORIGIN_X = 145000   # Western edge
    TAIWAN_ORIGIN_Y = 2800000  # Northern edge
    HEX_SIZE = 10000  # 10 km

    grid = HexGrid(
        hex_size_m=HEX_SIZE,
        origin_x=TAIWAN_ORIGIN_X,
        origin_y=TAIWAN_ORIGIN_Y
    )

    print(f"Hex dimensions: {grid.hex_width:.0f}m wide, {grid.hex_height:.0f}m tall")
    print(f"Spacing: {grid.col_spacing:.0f}m horizontal, {grid.row_spacing:.0f}m vertical")

    # Test coordinate conversion
    center_q, center_r = 10, 20
    x, y = grid.axial_to_world(center_q, center_r)
    print(f"\nHex ({center_q}, {center_r}) center: ({x:.0f}, {y:.0f})")

    # Round-trip test
    back_q, back_r = grid.world_to_axial(x, y)
    print(f"Round-trip: ({back_q}, {back_r})")

    # Generate a small test grid
    print("\nSample 3x3 grid:")
    for q in range(3):
        for r in range(3):
            cx, cy = grid.axial_to_world(q, r)
            print(f"  ({q}, {r}) -> ({cx:.0f}, {cy:.0f})")

    # === SVG Output Test ===
    print("\n--- SVG Generation Test ---")

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Test 1: Small 5x5 grid with terrain variety
    print("\nGenerating 5x5 test grid...")
    test_hexes = []
    terrains = ["water", "urban", "forest", "mountain", "paddy", "rough", "clear"]
    for q in range(5):
        for r in range(5):
            terrain = terrains[(q + r) % len(terrains)]
            test_hexes.append(HexData(q=q, r=r, terrain=terrain))

    render_svg(grid, test_hexes, "output/test_5x5.svg", show_coords=True, scale=0.005)

    # Test 2: Taiwan-sized grid (empty outline)
    # Approximately 20 wide x 55 tall based on spec
    print("\nGenerating Taiwan-sized grid outline...")
    taiwan_coords = [(q, r) for q in range(20) for r in range(55)]
    render_svg_outline(
        grid,
        taiwan_coords,
        "output/taiwan_grid_outline.svg",
        show_coords=False,  # Too many hexes for readable coords
        scale=0.002,
    )

    print("\nDone! Check output/ directory for SVG files.")
