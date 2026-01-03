"""
game_map_converter.py - Convert detailed tactical maps to game-ready maps

Takes a detailed SVG map + hexdata.json and generates a game map with:
- Elevation-based terrain tinting (darker at higher elevations)
- Maroon border frame
- Simplified hex labels (rows 1, 5, 10, 15, 20, 25 only)
- Multiple output formats (SVG, PNG, PDF)
"""

import json
import math
import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple
from xml.etree import ElementTree as ET

# Register SVG namespace to preserve proper output
ET.register_namespace('', 'http://www.w3.org/2000/svg')
ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')

# === Color Palettes ===

TERRAIN_COLORS_TEMPERATE = {
    "water": "#7BA3C4",      # Muted blue
    "urban": "#A89880",      # Grey-tan
    "forest": "#7A9E70",     # Sage green
    "orchard": "#8DAE7A",    # Lighter sage
    "marsh": "#8EBDB5",      # Blue-green
    "open": "#C8D4A8",       # Light olive/tan
    "sand": "#D4C8A0",       # Sandy tan
}

TERRAIN_COLORS_ARID = {
    "water": "#7BA3C4",      # Muted blue (same)
    "urban": "#A89880",      # Grey-tan
    "forest": "#9AAE80",     # Dusty olive
    "orchard": "#A8B890",    # Light olive
    "marsh": "#8EBDB5",      # Blue-green
    "open": "#D4C8A0",       # Tan/khaki
    "sand": "#E0D4B0",       # Light sand
}

# Frame color (dark maroon from DA PrePRESS reference maps)
FRAME_COLOR = "#6B2D2D"

# Mapping from SVG layer IDs to palette keys
# Note: Urban is excluded - it should stay grey regardless of palette
TERRAIN_LAYER_TO_KEY = {
    'Terrain_Open': 'open',
    'Terrain_Water': 'water',
    'Terrain_Forest': 'forest',
    'Terrain_Orchard': 'orchard',
    'Terrain_Marsh': 'marsh',
    'Sand': 'sand',
}

# Elevation tint darkness per band (0-6)
# Bands are RELATIVE to map's minimum elevation, 100m intervals
# Band 0 = base elevation (no overlay), Band 6 = highest (darkest)
ELEVATION_TINT_OPACITY = {
    0: 0.0,    # Base elevation - no darkening
    1: 0.10,   # +100m - 10% darker
    2: 0.20,   # +200m - 20% darker
    3: 0.30,   # +300m - 30% darker
    4: 0.40,   # +400m - 40% darker
    5: 0.50,   # +500m - 50% darker
    6: 0.60,   # +600m+ - 60% darker (max)
}

# Elevation band settings
ELEVATION_BAND_INTERVAL = 100  # meters per band
ELEVATION_BAND_MAX = 6         # maximum band number


def calculate_relative_elevation_bands(hex_data: dict) -> dict:
    """
    Calculate elevation bands relative to the map's minimum elevation.

    Uses 100m intervals from the lowest point on the map:
    - Band 0: min_elev to min_elev + 100m (no tint)
    - Band 1: +100m to +200m
    - Band 2: +200m to +300m
    - etc., capped at band 6

    Args:
        hex_data: Hex data from JSON with 'elevation' field per hex

    Returns:
        Dict mapping (q, r) to relative elevation band (0-6)
    """
    # Find minimum elevation
    elevations = [h.get('elevation', 0) for h in hex_data['hexes']]
    min_elev = min(elevations)
    max_elev = max(elevations)

    print(f"  Elevation range: {min_elev:.0f}m - {max_elev:.0f}m ({max_elev - min_elev:.0f}m relief)")
    print(f"  Using {ELEVATION_BAND_INTERVAL}m intervals from base ({min_elev:.0f}m)")

    # Calculate relative bands
    bands = {}
    band_counts = {}

    for hex_info in hex_data['hexes']:
        q, r = hex_info['coord']
        elev = hex_info.get('elevation', 0)

        # Calculate band: floor((elev - min) / interval), capped at max
        band = int((elev - min_elev) / ELEVATION_BAND_INTERVAL)
        band = min(band, ELEVATION_BAND_MAX)

        bands[(q, r)] = band
        band_counts[band] = band_counts.get(band, 0) + 1

    # Log band distribution
    for b in sorted(band_counts.keys()):
        low = min_elev + b * ELEVATION_BAND_INTERVAL
        high = min_elev + (b + 1) * ELEVATION_BAND_INTERVAL if b < ELEVATION_BAND_MAX else float('inf')
        high_str = f"{high:.0f}" if high != float('inf') else "+"
        print(f"    Band {b} ({low:.0f}-{high_str}m): {band_counts[b]} hexes")

    return bands

# Hex label rows to show (1-indexed)
LABEL_ROWS = [1, 5, 10, 15, 20, 25]


def get_terrain_palette(lat: float, lon: float, override: Optional[str] = None) -> dict:
    """
    Get terrain color palette based on location or manual override.

    Args:
        lat: Latitude of map center
        lon: Longitude of map center
        override: Optional manual override ('temperate', 'arid', or None for auto)

    Returns:
        Terrain color palette dict
    """
    if override:
        override_lower = override.lower()
        if override_lower == 'arid':
            return TERRAIN_COLORS_ARID
        elif override_lower == 'temperate':
            return TERRAIN_COLORS_TEMPERATE

    # Auto-detect based on latitude
    # Rough heuristic: < 35° latitude in typical arid zones
    # This is simplified - could be enhanced with climate data
    abs_lat = abs(lat)
    if abs_lat < 35:
        # Could be arid (deserts are typically 15-35° latitude)
        # But also need to consider longitude for monsoon regions
        # For now, simple latitude check
        return TERRAIN_COLORS_ARID
    else:
        return TERRAIN_COLORS_TEMPERATE


def darken_color(hex_color: str, amount: float) -> str:
    """
    Darken a hex color by a given amount (0-1).

    Args:
        hex_color: Color in #RRGGBB format
        amount: Darkening amount (0 = no change, 1 = black)

    Returns:
        Darkened color in #RRGGBB format
    """
    # Parse hex color
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # Darken by reducing each channel
    factor = 1 - amount
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)

    return f"#{r:02x}{g:02x}{b:02x}"


def load_hex_data(json_path: Path) -> dict:
    """Load hex data from JSON file."""
    with open(json_path) as f:
        return json.load(f)


def load_svg(svg_path: Path) -> ET.ElementTree:
    """Load SVG file as ElementTree."""
    return ET.parse(svg_path)


def get_svg_dimensions(tree: ET.ElementTree) -> tuple:
    """Get SVG dimensions from root element.

    Returns dimensions from viewBox (in SVG units/meters), not physical size.
    """
    root = tree.getroot()

    # Prefer viewBox as it gives us the coordinate system dimensions
    viewbox = root.get('viewBox')
    if viewbox:
        parts = viewbox.split()
        if len(parts) == 4:
            return float(parts[2]), float(parts[3])

    # Fallback to width/height attributes
    width = root.get('width')
    height = root.get('height')

    if width and height:
        # Remove unit suffixes (px, in, mm, etc.)
        import re
        width_match = re.match(r'([\d.]+)', width)
        height_match = re.match(r'([\d.]+)', height)
        if width_match and height_match:
            return float(width_match.group(1)), float(height_match.group(1))

    raise ValueError("Could not determine SVG dimensions")


def get_rotation_info(tree: ET.ElementTree) -> Optional[Tuple[float, float, float]]:
    """
    Extract rotation info from the SVG's Rotated_Content group.

    Returns:
        Tuple of (angle, center_x, center_y) if rotation exists, None otherwise
    """
    root = tree.getroot()

    # Find Rotated_Content group
    for elem in root.iter():
        if elem.get('id') == 'Rotated_Content':
            transform = elem.get('transform', '')
            # Parse rotate(angle, cx, cy)
            match = re.match(r'rotate\(([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)', transform)
            if match:
                angle = float(match.group(1))
                cx = float(match.group(2))
                cy = float(match.group(3))
                return (angle, cx, cy)

    return None


def reverse_rotate_point(x: float, y: float, angle: float, cx: float, cy: float) -> Tuple[float, float]:
    """
    Reverse-rotate a point around a center.

    If (x, y) is in screen space (after rotation was applied), this returns
    the coordinates in the pre-rotation space.

    Args:
        x, y: Point coordinates in screen space
        angle: Rotation angle in degrees (the original rotation)
        cx, cy: Center of rotation

    Returns:
        (x', y') coordinates in pre-rotation space
    """
    # Convert angle to radians (negative because we're reversing)
    rad = math.radians(-angle)

    # Translate to origin
    tx = x - cx
    ty = y - cy

    # Rotate
    rx = tx * math.cos(rad) - ty * math.sin(rad)
    ry = tx * math.sin(rad) + ty * math.cos(rad)

    # Translate back
    return (rx + cx, ry + cy)


def extract_hex_positions_from_svg(tree: ET.ElementTree, hex_data: dict) -> dict:
    """
    Extract hex center positions from the SVG's Hex_Markers group.

    The hex markers are circles positioned exactly at hex centers.

    Args:
        tree: SVG ElementTree
        hex_data: Hex data from JSON (to get q,r coordinates)

    Returns:
        Dict mapping (q, r) tuple to (cx, cy) center coordinates
    """
    root = tree.getroot()
    positions = {}

    # Find Hex_Markers group - circles at exact hex centers
    hex_markers_group = None
    for elem in root.iter():
        if elem.get('id') == 'Hex_Markers':
            hex_markers_group = elem
            break

    if hex_markers_group is not None:
        # Extract circle centers
        circles = []
        for elem in hex_markers_group.iter():
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag == 'circle':
                cx = elem.get('cx')
                cy = elem.get('cy')
                if cx and cy:
                    circles.append((float(cx), float(cy)))

        # Match circles to hex coordinates by position
        # Circles are in same order as hexes were rendered
        hex_list = hex_data['hexes']
        if len(circles) == len(hex_list):
            for i, hex_info in enumerate(hex_list):
                q, r = hex_info['coord']
                positions[(q, r)] = circles[i]
            print(f"  Extracted {len(positions)} hex centers from Hex_Markers")
            return positions

    # Fallback: try Hex_Labels
    hex_labels_group = None
    for elem in root.iter():
        if elem.get('id') == 'Hex_Labels':
            hex_labels_group = elem
            break

    if hex_labels_group is not None:
        hex_size_m = hex_data['metadata']['hex_size_m']
        size = hex_size_m / math.sqrt(3)
        # Labels are positioned 80% up from center + baseline adjustment
        label_offset_y = size * math.sqrt(3) / 2 * 0.80

        for elem in hex_labels_group.iter():
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag == 'text':
                text_content = elem.text
                if text_content and '.' in text_content:
                    try:
                        parts = text_content.split('.')
                        q = int(parts[0]) - 1
                        r = int(parts[1]) - 1
                        x = float(elem.get('x', 0))
                        y = float(elem.get('y', 0))
                        # Adjust from label position to center
                        cy = y + label_offset_y - hex_size_m * 0.35 * 0.25  # Approximate baseline adjustment
                        positions[(q, r)] = (x, cy)
                    except (ValueError, IndexError):
                        continue
        print(f"  Extracted {len(positions)} hex centers from Hex_Labels (fallback)")

    return positions


def create_hex_polygon_points(cx: float, cy: float, size: float) -> str:
    """
    Create SVG polygon points string for a flat-top hex.

    Args:
        cx, cy: Center coordinates
        size: Distance from center to vertex

    Returns:
        Points string for SVG polygon
    """
    points = []
    for i in range(6):
        angle = math.radians(60 * i)
        x = cx + size * math.cos(angle)
        y = cy + size * math.sin(angle)
        points.append(f"{x},{y}")
    return " ".join(points)


def hex_to_svg_coords(q: int, r: int, hex_size_m: float, origin_x: float, origin_y: float) -> tuple:
    """
    Convert hex axial coordinates to SVG coordinates.

    Args:
        q, r: Axial hex coordinates
        hex_size_m: Hex size in meters (flat edge to flat edge)
        origin_x, origin_y: SVG origin coordinates

    Returns:
        (cx, cy) center coordinates in SVG space
    """
    size = hex_size_m / math.sqrt(3)  # Center to vertex
    col_spacing = 1.5 * size
    row_spacing = hex_size_m

    x = origin_x + q * col_spacing
    y = origin_y + r * row_spacing + (q % 2) * row_spacing * 0.5

    return x, y


def generate_elevation_overlays(
    hex_data: dict,
    hex_centers: dict,
    elevation_bands: dict,
    intensity: float = 1.0
) -> ET.Element:
    """
    Generate semi-transparent overlay polygons for elevation tinting.

    Args:
        hex_data: Hex data from JSON
        hex_centers: Dict mapping (q, r) to (cx, cy) from extract_hex_positions_from_svg
        elevation_bands: Dict mapping (q, r) to relative elevation band (0-6)
        intensity: Multiplier for tint opacity (default 1.0)

    Returns:
        SVG group element containing elevation overlays
    """
    group = ET.Element('g', {'id': 'Game_Elevation_Overlay'})

    hex_size_m = hex_data['metadata']['hex_size_m']
    size = hex_size_m / math.sqrt(3)  # Center to vertex

    overlay_count = 0
    for hex_info in hex_data['hexes']:
        q, r = hex_info['coord']

        # Get relative elevation band
        elevation_band = elevation_bands.get((q, r), 0)

        # Skip if no tinting needed
        opacity = ELEVATION_TINT_OPACITY.get(elevation_band, 0) * intensity
        if opacity <= 0:
            continue

        # Get hex center from extracted positions
        if (q, r) not in hex_centers:
            continue

        cx, cy = hex_centers[(q, r)]

        # Create overlay polygon (dark overlay for elevation)
        # Use exact hex size to align with grid
        points = create_hex_polygon_points(cx, cy, size)

        polygon = ET.SubElement(group, 'polygon', {
            'points': points,
            'fill': '#000000',
            'fill-opacity': str(opacity),
            'stroke': 'none',
        })
        overlay_count += 1

    print(f"  Created {overlay_count} elevation overlay polygons")
    return group


def generate_game_hex_labels(
    hex_data: dict,
    hex_centers: dict,
    label_rows: list = None
) -> ET.Element:
    """
    Generate hex labels only for specified rows.

    Args:
        hex_data: Hex data from JSON
        hex_centers: Dict mapping (q, r) to (cx, cy) from extract_hex_positions_from_svg
        label_rows: List of row numbers to label (1-indexed)

    Returns:
        SVG group element containing hex labels
    """
    if label_rows is None:
        label_rows = LABEL_ROWS

    group = ET.Element('g', {'id': 'Game_Hex_Labels'})

    hex_size_m = hex_data['metadata']['hex_size_m']

    # Label size and positioning (match tactical_map.py)
    label_size_m = 25  # Font size in meters
    size = hex_size_m / math.sqrt(3)  # Center to vertex
    half_height = size * math.sqrt(3) / 2  # Center to flat edge (top/bottom)

    for hex_info in hex_data['hexes']:
        q, r = hex_info['coord']

        # Convert to 1-indexed row number
        row_1indexed = r + 1

        # Only label specified rows
        if row_1indexed not in label_rows:
            continue

        # Get hex center from extracted positions
        if (q, r) not in hex_centers:
            continue

        cx, cy = hex_centers[(q, r)]

        # Position label at top of hex (80% up from center, with baseline adjustment)
        # Match tactical_map.py: top_offset = size * sqrt(3) / 2 * 0.80, then baseline adjust
        top_offset = half_height * 0.80
        label_y = cy - top_offset + label_size_m * 0.35  # Up 80%, then down for baseline

        # Format label as XX.YY (1-indexed)
        label_text = f"{q + 1:02d}.{r + 1:02d}"

        # Halo text (white stroke behind label for visibility on dark backgrounds)
        halo = ET.SubElement(group, 'text', {
            'x': str(cx),
            'y': str(label_y),
            'text-anchor': 'middle',
            'font-family': 'Arial, sans-serif',
            'font-size': str(label_size_m),
            'fill': 'none',
            'stroke': '#e8e8e8',
            'stroke-width': '3',
            'stroke-opacity': '0.4',
        })
        halo.text = label_text

        # Label text on top
        text = ET.SubElement(group, 'text', {
            'x': str(cx),
            'y': str(label_y),
            'text-anchor': 'middle',
            'font-family': 'Arial, sans-serif',
            'font-size': str(label_size_m),
            'fill': '#5e5959',
            'fill-opacity': '0.7',
        })
        text.text = label_text

    return group


def get_hex_neighbors(q: int, r: int) -> list:
    """
    Get the 6 neighboring hex coordinates for flat-top hex grid.

    Returns neighbors by DIRECTION (0-5 clockwise from top).
    This matches hexgrid.py's convention:
    - Dir 0: top neighbor
    - Dir 1: top-right neighbor
    - Dir 2: bottom-right neighbor
    - Dir 3: bottom neighbor
    - Dir 4: bottom-left neighbor
    - Dir 5: top-left neighbor
    """
    if q % 2 == 0:
        return [
            (q, r - 1),      # Dir 0: top
            (q + 1, r - 1),  # Dir 1: top-right
            (q + 1, r),      # Dir 2: bottom-right
            (q, r + 1),      # Dir 3: bottom
            (q - 1, r),      # Dir 4: bottom-left
            (q - 1, r - 1),  # Dir 5: top-left
        ]
    else:
        return [
            (q, r - 1),      # Dir 0: top
            (q + 1, r),      # Dir 1: top-right
            (q + 1, r + 1),  # Dir 2: bottom-right
            (q, r + 1),      # Dir 3: bottom
            (q - 1, r + 1),  # Dir 4: bottom-left
            (q - 1, r),      # Dir 5: top-left
        ]


# Mapping from neighbor direction index to geometric edge index
# Geometric edges use vertices starting at 0° (right), going clockwise:
#   Edge 0: right edge (v0-v1, 0°-60°)
#   Edge 1: bottom edge (v1-v2, 60°-120°)
#   Edge 2: bottom-left edge (v2-v3, 120°-180°)
#   Edge 3: left edge (v3-v4, 180°-240°)
#   Edge 4: top edge (v4-v5, 240°-300°)
#   Edge 5: top-right edge (v5-v0, 300°-360°)
#
# Neighbor directions (from hexgrid.py):
#   Dir 0: top -> Edge 4
#   Dir 1: top-right -> Edge 5
#   Dir 2: bottom-right -> Edge 0
#   Dir 3: bottom -> Edge 1
#   Dir 4: bottom-left -> Edge 2
#   Dir 5: top-left -> Edge 3
NEIGHBOR_DIR_TO_EDGE = [4, 5, 0, 1, 2, 3]


def get_hex_edge_vertices(cx: float, cy: float, size: float, edge_idx: int) -> tuple:
    """
    Get the two vertices of a hex edge.

    Args:
        cx, cy: Hex center coordinates
        size: Distance from center to vertex
        edge_idx: Geometric edge index (0-5, starting from right edge going clockwise)

    Returns:
        ((x1, y1), (x2, y2)) tuple of vertex coordinates
    """
    # Vertices are at angles 0°, 60°, 120°, 180°, 240°, 300° from center
    # Edge i connects vertex i to vertex i+1
    angle1 = math.radians(60 * edge_idx)
    angle2 = math.radians(60 * ((edge_idx + 1) % 6))

    x1 = cx + size * math.cos(angle1)
    y1 = cy + size * math.sin(angle1)
    x2 = cx + size * math.cos(angle2)
    y2 = cy + size * math.sin(angle2)

    return ((x1, y1), (x2, y2))


# Hillside shading colors (darker versions of terrain base colors)
HILLSIDE_COLORS = {
    'temperate': '#6B7A60',  # Darker sage green for temperate terrain
    'arid': '#9A8B70',       # Darker tan/khaki for arid terrain
}


def generate_hillside_shading(
    hex_data: dict,
    hex_centers: dict,
    elevation_bands: dict,
    terrain_style: str = 'temperate',
    intensity: float = 1.0
) -> ET.Element:
    """
    Generate hillside shading bands along edges where elevation drops.

    Creates dark bands on the higher side of elevation transitions to
    give a 3D shadow effect similar to DA PrePRESS NTC maps.

    Args:
        hex_data: Hex data from JSON
        hex_centers: Dict mapping (q, r) to (cx, cy)
        elevation_bands: Dict mapping (q, r) to relative elevation band (0-6)
        terrain_style: 'temperate' or 'arid' for color selection
        intensity: Opacity multiplier (default 1.0)

    Returns:
        SVG group element containing hillside shading polygons
    """
    group = ET.Element('g', {'id': 'Game_Hillside_Shading'})

    hex_size_m = hex_data['metadata']['hex_size_m']
    size = hex_size_m / math.sqrt(3)  # Center to vertex

    # Hillside band width (as fraction of hex size)
    band_width = size * 0.15  # 15% of center-to-vertex distance

    # Select color based on terrain style
    shade_color = HILLSIDE_COLORS.get(terrain_style, HILLSIDE_COLORS['temperate'])
    base_opacity = 1.0 * intensity

    shading_count = 0

    for hex_info in hex_data['hexes']:
        q, r = hex_info['coord']
        if (q, r) not in hex_centers:
            continue

        my_elevation = elevation_bands.get((q, r), 0)
        cx, cy = hex_centers[(q, r)]

        # Check each neighbor
        neighbors = get_hex_neighbors(q, r)

        for dir_idx, (nq, nr) in enumerate(neighbors):
            neighbor_elevation = elevation_bands.get((nq, nr))

            # Skip if neighbor doesn't exist or has same/higher elevation
            if neighbor_elevation is None or neighbor_elevation >= my_elevation:
                continue

            # Calculate elevation difference for opacity scaling
            elev_diff = my_elevation - neighbor_elevation
            opacity = min(base_opacity * elev_diff, 0.6)  # Cap at 60%

            # Convert neighbor direction to geometric edge index
            edge_idx = NEIGHBOR_DIR_TO_EDGE[dir_idx]

            # Get edge vertices
            (x1, y1), (x2, y2) = get_hex_edge_vertices(cx, cy, size, edge_idx)

            # Calculate inward offset for band (toward hex center)
            # Midpoint of edge
            mx = (x1 + x2) / 2
            my_mid = (y1 + y2) / 2

            # Direction from edge midpoint to center
            dx = cx - mx
            dy = cy - my_mid
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 0:
                dx /= dist
                dy /= dist

            # Inner points (offset toward center)
            x1_inner = x1 + dx * band_width
            y1_inner = y1 + dy * band_width
            x2_inner = x2 + dx * band_width
            y2_inner = y2 + dy * band_width

            # Create quad for the hillside band
            points = f"{x1},{y1} {x2},{y2} {x2_inner},{y2_inner} {x1_inner},{y1_inner}"

            polygon = ET.SubElement(group, 'polygon', {
                'points': points,
                'fill': shade_color,
                'fill-opacity': str(opacity),
                'stroke': 'none',
            })
            shading_count += 1

    print(f"  Created {shading_count} hillside shading bands")
    return group


def unhide_game_overlays(tree: ET.ElementTree) -> bool:
    """
    Find and unhide pre-generated Game_Elevation_Overlay and Game_Hillside_Shading layers.

    These layers are now generated during detail map creation (tactical_map.py)
    with visibility="hidden". This function just reveals them for game maps.

    Args:
        tree: SVG ElementTree to search and modify

    Returns:
        True if both layers were found and unhidden, False otherwise
    """
    root = tree.getroot()

    found_elevation = False
    found_hillside = False

    for elem in root.iter():
        elem_id = elem.get('id')
        if elem_id == 'Game_Elevation_Overlay':
            elem.set('visibility', 'visible')
            found_elevation = True
            print(f"  Unhid Game_Elevation_Overlay (pre-generated)")
        elif elem_id == 'Game_Hillside_Shading':
            elem.set('visibility', 'visible')
            found_hillside = True
            print(f"  Unhid Game_Hillside_Shading (pre-generated)")

    if found_elevation and found_hillside:
        print(f"  Using pre-generated overlays from detail map")
        return True

    if found_elevation or found_hillside:
        print(f"  Warning: Only found one overlay layer, generating missing ones")

    return False


def modify_existing_frame_color(tree: ET.ElementTree, color: str = None) -> bool:
    """
    Modify the existing Out_Of_Play_Frame to use the maroon color.

    Args:
        tree: SVG ElementTree to modify in place
        color: Frame color (defaults to FRAME_COLOR)

    Returns:
        True if frame was found and modified, False otherwise
    """
    if color is None:
        color = FRAME_COLOR

    root = tree.getroot()

    # Find Out_Of_Play_Frame group
    for elem in root.iter():
        elem_id = elem.get('id', '')
        if elem_id == 'Out_Of_Play_Frame':
            # Find all child elements and update their fill color
            for child in elem.iter():
                tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                if tag in ('path', 'rect', 'polygon'):
                    child.set('fill', color)
            print(f"  Modified existing frame color to {color}")
            return True

    print("  Warning: Out_Of_Play_Frame not found")
    return False


def hide_out_of_play_frame(tree: ET.ElementTree) -> bool:
    """
    Hide the Out_Of_Play_Frame layer entirely (for multi-map clusters).

    Args:
        tree: SVG ElementTree to modify in place

    Returns:
        True if frame was found and hidden, False otherwise
    """
    root = tree.getroot()

    # Find Out_Of_Play_Frame group
    for elem in root.iter():
        elem_id = elem.get('id', '')
        if elem_id == 'Out_Of_Play_Frame':
            elem.set('visibility', 'hidden')
            print("  Hidden Out_Of_Play_Frame (multi-map cluster)")
            return True

    print("  Warning: Out_Of_Play_Frame not found")
    return False


def hide_original_hex_labels(tree: ET.ElementTree) -> None:
    """
    Hide the original Hex_Labels layer by setting display:none.

    Args:
        tree: SVG ElementTree to modify in place
    """
    root = tree.getroot()

    # Find Hex_Labels group
    for elem in root.iter():
        elem_id = elem.get('id', '')
        if elem_id == 'Hex_Labels':
            elem.set('style', 'display:none')
            break


def hide_detail_layers(tree: ET.ElementTree) -> None:
    """
    Hide detail layers that are too fine for game maps.

    Hides: Paths, Powerlines, Tree_Rows, Barriers, Military

    Args:
        tree: SVG ElementTree to modify in place
    """
    root = tree.getroot()
    layers_to_hide = ['Paths', 'Powerlines', 'Tree_Rows', 'Barriers', 'Military']
    hidden_count = 0

    for elem in root.iter():
        elem_id = elem.get('id', '')
        if elem_id in layers_to_hide:
            elem.set('style', 'display:none')
            hidden_count += 1

    print(f"  Hidden {hidden_count} detail layers: {', '.join(layers_to_hide)}")


def recolor_terrain_layers(tree: ET.ElementTree, palette: dict) -> int:
    """
    Recolor terrain layers based on the selected color palette.

    Args:
        tree: SVG ElementTree to modify in place
        palette: Terrain color palette dict (e.g., TERRAIN_COLORS_TEMPERATE)

    Returns:
        Number of layers recolored
    """
    root = tree.getroot()
    recolored_count = 0

    for elem in root.iter():
        elem_id = elem.get('id', '')

        # Check if this is a terrain layer we can recolor
        palette_key = TERRAIN_LAYER_TO_KEY.get(elem_id)
        if not palette_key:
            continue

        # Get the new color from the palette
        new_color = palette.get(palette_key)
        if not new_color:
            continue

        # Recolor all polygons, paths, and rects within this layer
        for child in elem.iter():
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if tag in ('polygon', 'path', 'rect'):
                old_fill = child.get('fill')
                if old_fill and old_fill != 'none':
                    child.set('fill', new_color)

        recolored_count += 1
        print(f"    Recolored {elem_id} to {new_color}")

    return recolored_count


def add_game_layers(
    tree: ET.ElementTree,
    elevation_overlay: ET.Element,
    hillside_shading: ET.Element,
    hex_labels: ET.Element,
    is_rotated: bool = False
) -> None:
    """
    Add game map layers to SVG at appropriate positions.

    For ROTATED maps:
    - Overlays are placed INSIDE Rotated_Content so they rotate with terrain
    - This ensures overlays visually align with the terrain features they represent
    - Hex_Markers coordinates work directly because they use the same pre-rotation
      coordinate system as the terrain layers

    For NON-ROTATED maps:
    - Overlays are placed as siblings of Hex_Grid (outside any rotation group)

    Hex labels are ALWAYS placed outside rotation to remain readable.

    Z-order for rotated maps (bottom to top):
    - Rotated_Content:
        - Terrain layers
        - Game_Elevation_Overlay (rotates with terrain)
        - Game_Hillside_Shading (rotates with terrain)
    - Hex_Grid (axis-aligned)
    - Hex_Markers
    - Game_Hex_Labels (at end, axis-aligned)

    Args:
        tree: SVG ElementTree to modify
        elevation_overlay: Elevation overlay group
        hillside_shading: Hillside shading group
        hex_labels: Game hex labels group
        is_rotated: Whether the map has rotation applied
    """
    root = tree.getroot()

    # Find element and its parent
    def find_element_and_parent(parent, target_id):
        for idx, child in enumerate(parent):
            if child.get('id') == target_id:
                return parent, idx, child
            result = find_element_and_parent(child, target_id)
            if result:
                return result
        return None

    if is_rotated:
        # For rotated maps, place overlays INSIDE Rotated_Content
        # so they rotate with the terrain
        result = find_element_and_parent(root, 'Rotated_Content')
        if result:
            _, _, rotated_content = result
            # Insert at end of Rotated_Content (above terrain, below Out_Of_Play_Frame)
            rotated_content.append(elevation_overlay)
            rotated_content.append(hillside_shading)
            print(f"  Inserted overlays inside Rotated_Content (will rotate with terrain)")
        else:
            # Fallback if Rotated_Content not found (shouldn't happen for rotated maps)
            print(f"  Warning: Rotated_Content not found, falling back to non-rotated placement")
            is_rotated = False  # Fall through to non-rotated logic

    if not is_rotated:
        # For non-rotated maps, insert overlays BEFORE Hex_Grid
        result = find_element_and_parent(root, 'Hex_Grid')

        if result:
            parent, idx, hex_grid = result
            parent_id = parent.get('id', 'unknown')
            print(f"  Found Hex_Grid at index {idx} in parent '{parent_id}'")

            # Insert overlays BEFORE Hex_Grid (so they're below grid lines but above terrain)
            parent.insert(idx, hillside_shading)
            parent.insert(idx, elevation_overlay)  # This goes before hillside
            print(f"  Inserted overlays before Hex_Grid")
        else:
            # Fallback: find Master_Content and insert at end
            result = find_element_and_parent(root, 'Master_Content')
            if result:
                _, _, master_content = result
                master_content.append(elevation_overlay)
                master_content.append(hillside_shading)
                print(f"  Fallback: Inserted overlays at end of Master_Content")
            else:
                # Last resort: after first group
                for idx, child in enumerate(root):
                    tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                    if tag == 'g':
                        root.insert(idx + 1, elevation_overlay)
                        root.insert(idx + 2, hillside_shading)
                        print(f"  Last resort: Inserted overlays after first group")
                        break

    # Add hex labels at the end (on top) - always outside rotation for readability
    if hex_labels is not None:
        root.append(hex_labels)
        print("  Added game hex labels at top")


def export_svg(tree: ET.ElementTree, output_path: Path) -> None:
    """Export SVG to file."""
    tree.write(output_path, encoding='unicode', xml_declaration=True)
    print(f"  Saved SVG: {output_path}")


def export_png(svg_path: Path, output_path: Path, dpi: int = 150) -> bool:
    """
    Export SVG to PNG using rsvg-convert or cairosvg.

    Args:
        svg_path: Input SVG path
        output_path: Output PNG path
        dpi: Resolution in DPI

    Returns:
        True if successful, False otherwise
    """
    # Try rsvg-convert first (usually better quality)
    try:
        subprocess.run([
            'rsvg-convert',
            '-d', str(dpi),
            '-p', str(dpi),
            '-o', str(output_path),
            str(svg_path)
        ], check=True, capture_output=True)
        print(f"  Saved PNG: {output_path}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Try cairosvg as fallback
    try:
        import cairosvg
        # Calculate dimensions at target DPI
        cairosvg.svg2png(
            url=str(svg_path),
            write_to=str(output_path),
            dpi=dpi
        )
        print(f"  Saved PNG: {output_path}")
        return True
    except ImportError:
        print("  Warning: Could not export PNG (install rsvg-convert or cairosvg)")
        return False
    except Exception as e:
        print(f"  Warning: PNG export failed: {e}")
        return False


def export_pdf(svg_path: Path, output_path: Path) -> bool:
    """
    Export SVG to PDF.

    Args:
        svg_path: Input SVG path
        output_path: Output PDF path

    Returns:
        True if successful, False otherwise
    """
    # Try rsvg-convert first
    try:
        subprocess.run([
            'rsvg-convert',
            '-f', 'pdf',
            '-o', str(output_path),
            str(svg_path)
        ], check=True, capture_output=True)
        print(f"  Saved PDF: {output_path}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Try cairosvg as fallback
    try:
        import cairosvg
        cairosvg.svg2pdf(
            url=str(svg_path),
            write_to=str(output_path)
        )
        print(f"  Saved PDF: {output_path}")
        return True
    except ImportError:
        print("  Warning: Could not export PDF (install rsvg-convert or cairosvg)")
        return False
    except Exception as e:
        print(f"  Warning: PDF export failed: {e}")
        return False


def diagnose_hex_alignment(tree: ET.ElementTree, hex_data: dict, sample_count: int = 3):
    """
    Diagnostic function to compare hex positions and vertices.

    Compares:
    1. Hex_Markers circle positions (what we extract)
    2. Hex_Grid line positions (actual grid vertices)
    3. Generated overlay polygon vertices
    """
    import math
    root = tree.getroot()

    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Hex Alignment Analysis")
    print("=" * 60)

    # 1. Find Hex_Markers and extract first few circle positions
    hex_markers = None
    for elem in root.iter():
        if elem.get('id') == 'Hex_Markers':
            hex_markers = elem
            break

    if hex_markers is None:
        print("ERROR: Hex_Markers not found!")
        return

    circles = []
    for elem in hex_markers.iter():
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        if tag == 'circle':
            cx = float(elem.get('cx', 0))
            cy = float(elem.get('cy', 0))
            circles.append((cx, cy))

    print(f"\n1. HEX_MARKERS: Found {len(circles)} circles")
    print(f"   First {sample_count} circle centers:")
    for i, (cx, cy) in enumerate(circles[:sample_count]):
        print(f"   Circle {i}: ({cx:.2f}, {cy:.2f})")

    # 2. Find Hex_Grid and extract line endpoints to find vertices
    hex_grid = None
    for elem in root.iter():
        if elem.get('id') == 'Hex_Grid':
            hex_grid = elem
            break

    if hex_grid:
        lines = []
        for elem in hex_grid.iter():
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag == 'line':
                x1 = float(elem.get('x1', 0))
                y1 = float(elem.get('y1', 0))
                x2 = float(elem.get('x2', 0))
                y2 = float(elem.get('y2', 0))
                lines.append(((x1, y1), (x2, y2)))

        print(f"\n2. HEX_GRID: Found {len(lines)} lines")
        # Find lines that start near first circle center
        if circles:
            cx, cy = circles[0]
            nearby_lines = []
            for (x1, y1), (x2, y2) in lines:
                dist1 = math.sqrt((x1 - cx)**2 + (y1 - cy)**2)
                dist2 = math.sqrt((x2 - cx)**2 + (y2 - cy)**2)
                if dist1 < 200 or dist2 < 200:  # Within 200 units of center
                    nearby_lines.append(((x1, y1), (x2, y2)))

            print(f"   Lines near first hex center ({cx:.2f}, {cy:.2f}):")
            for i, ((x1, y1), (x2, y2)) in enumerate(nearby_lines[:6]):
                # Calculate angle from center to line start
                angle1 = math.degrees(math.atan2(y1 - cy, x1 - cx))
                print(f"   Line {i}: ({x1:.2f}, {y1:.2f}) -> ({x2:.2f}, {y2:.2f}), angle from center: {angle1:.1f}°")

    # 3. Generate overlay polygon for first hex and show vertices
    hex_size_m = hex_data['metadata']['hex_size_m']
    size = hex_size_m / math.sqrt(3)

    if circles:
        cx, cy = circles[0]
        print(f"\n3. GENERATED OVERLAY for hex at ({cx:.2f}, {cy:.2f}):")
        print(f"   Hex size: {hex_size_m}m, vertex distance: {size:.2f}")
        print(f"   Polygon vertices (using angles 0°, 60°, 120°, ...):")
        for i in range(6):
            angle = math.radians(60 * i)
            vx = cx + size * math.cos(angle)
            vy = cy + size * math.sin(angle)
            print(f"   Vertex {i} (angle {60*i}°): ({vx:.2f}, {vy:.2f})")

    # 4. Check hex_data for first hex
    print(f"\n4. HEX_DATA: First {sample_count} hexes from JSON:")
    for i, hex_info in enumerate(hex_data['hexes'][:sample_count]):
        q, r = hex_info['coord']
        elev = hex_info.get('elevation_band', 0)
        print(f"   Hex {i}: coord=({q}, {r}), elevation_band={elev}")

    # 5. Check where Hex_Markers is in the tree
    print(f"\n5. SVG STRUCTURE:")
    def find_path(elem, target_id, path=""):
        for child in elem:
            child_id = child.get('id', '')
            child_tag = child.tag.split('}')[-1]
            current_path = f"{path}/{child_tag}[{child_id}]" if child_id else f"{path}/{child_tag}"
            if child_id == target_id:
                transform = child.get('transform', 'none')
                return current_path, transform
            result = find_path(child, target_id, current_path)
            if result:
                return result
        return None

    for target in ['Rotated_Content', 'Terrain_Open', 'Hex_Grid', 'Hex_Markers']:
        result = find_path(root, target)
        if result:
            path, transform = result
            print(f"   {target}: {path}")
            if transform != 'none':
                print(f"      transform: {transform}")

    print("\n" + "=" * 60)


def convert_single_sheet(
    svg_path: Path,
    json_path: Path,
    output_dir: Path,
    config: dict,
    is_multi_map: bool = False
) -> Path:
    """
    Convert a single detailed tactical map sheet to a game-ready map.

    Args:
        svg_path: Path to the tactical SVG file
        json_path: Path to the hexdata JSON file
        output_dir: Output directory for game map files
        config: Configuration dict
        is_multi_map: Whether this sheet is part of a multi-map cluster

    Returns:
        Path to output SVG file
    """
    # Get map name from filename
    map_name = svg_path.stem.replace('_tactical', '')

    print(f"\nInput:")
    print(f"  SVG: {svg_path}")
    print(f"  Data: {json_path}")

    # Load data
    print("\nLoading data...")
    hex_data = load_hex_data(json_path)
    tree = load_svg(svg_path)

    # Run diagnostic
    diagnose_hex_alignment(tree, hex_data)

    # Get SVG dimensions
    width, height = get_svg_dimensions(tree)
    print(f"  SVG size: {width:.0f} x {height:.0f}")

    # Get map center for palette selection
    center_lat = hex_data['metadata'].get('center_lat', 0)
    center_lon = hex_data['metadata'].get('center_lon', 0)

    # Select terrain palette and determine actual style
    terrain_style = config.get('terrain_style', 'auto')
    if terrain_style == 'auto':
        palette = get_terrain_palette(center_lat, center_lon)
        actual_style = 'arid' if abs(center_lat) < 35 else 'temperate'
        print(f"  Auto-detected palette: {actual_style}")
    else:
        palette = get_terrain_palette(center_lat, center_lon, terrain_style)
        actual_style = terrain_style
        print(f"  Using palette: {terrain_style}")

    # Recolor terrain layers based on selected palette
    print("\nRecoloring terrain...")
    recolored = recolor_terrain_layers(tree, palette)
    print(f"  Recolored {recolored} terrain layers")

    # Check if map has rotation
    rotation_info = get_rotation_info(tree)
    is_rotated = rotation_info is not None
    if is_rotated:
        angle, cx, cy = rotation_info
        print(f"\n  Map has rotation: {angle}° around ({cx:.1f}, {cy:.1f})")
        print(f"  Note: Overlays will be placed inside Rotated_Content to align with terrain")
    else:
        print("\n  Map has no rotation")

    # Extract hex positions from SVG (more reliable than recalculating)
    # These coordinates are in the same system as terrain layers - works for both rotated and non-rotated maps
    print("\nExtracting hex positions from SVG...")
    hex_size_m = hex_data['metadata']['hex_size_m']
    hex_centers = extract_hex_positions_from_svg(tree, hex_data)
    print(f"  Found {len(hex_centers)} hex positions (screen-space coordinates)")

    if not hex_centers:
        print("  Warning: Could not extract hex positions, using fallback calculation")
        # Fallback: estimate positions from grid
        size = hex_size_m / math.sqrt(3)
        margin = 100
        for hex_info in hex_data['hexes']:
            q, r = hex_info['coord']
            cx, cy = hex_to_svg_coords(q, r, hex_size_m, margin + size, margin + size)
            hex_centers[(q, r)] = (cx, cy)

    print("\nProcessing game map overlays...")

    # Try to unhide pre-generated overlay layers (new approach - layers generated during detail map creation)
    # These are already in the correct coordinate system and position (inside Rotated_Content for rotated maps)
    overlays_unhidden = unhide_game_overlays(tree)

    if not overlays_unhidden:
        # Fallback for older maps without pre-generated overlays: generate them here
        print("  Pre-generated overlays not found, generating (fallback for older maps)...")

        print("\nCalculating relative elevation bands...")
        elevation_bands = calculate_relative_elevation_bands(hex_data)

        # Generate elevation overlays
        elevation_intensity = config.get('elevation_intensity', 1.0)
        elevation_overlay = generate_elevation_overlays(
            hex_data, hex_centers, elevation_bands, elevation_intensity
        )

        # Generate hillside shading for elevation transitions
        hillside_shading = generate_hillside_shading(
            hex_data, hex_centers, elevation_bands, actual_style, elevation_intensity
        )

        # Add generated overlays to SVG (only when not using pre-generated ones)
        add_game_layers(tree, elevation_overlay, hillside_shading, None, is_rotated)

    # Generate game hex labels (always needed, uses screen-space coordinates)
    label_rows = config.get('label_rows', LABEL_ROWS)
    hex_labels = generate_game_hex_labels(hex_data, hex_centers, label_rows)
    print(f"  Created hex labels (rows: {label_rows})")

    # Handle out-of-play frame
    if is_multi_map:
        # Hide frame entirely for multi-map clusters
        hide_out_of_play_frame(tree)
    else:
        # Modify existing frame color to maroon for single maps
        frame_color = config.get('frame_color', FRAME_COLOR)
        modify_existing_frame_color(tree, frame_color)

    # Hide original hex labels
    hide_original_hex_labels(tree)
    print(f"  Hidden original hex labels")

    # Hide detail layers (paths, powerlines, tree rows)
    hide_detail_layers(tree)

    # Add hex labels at the end (on top) - always outside rotation for readability
    tree.getroot().append(hex_labels)
    print("  Added game hex labels at top")

    # Export files
    print("\nExporting...")

    svg_output = output_dir / f"{map_name}_game.svg"
    export_svg(tree, svg_output)

    png_output = output_dir / f"{map_name}_game.png"
    export_png(svg_output, png_output)

    pdf_output = output_dir / f"{map_name}_game.pdf"
    export_pdf(svg_output, pdf_output)

    print(f"  Saved: {svg_output.name}, {png_output.name}, {pdf_output.name}")

    return svg_output


def convert_to_game_map(
    detailed_map_dir: Path,
    config: dict = None
) -> Path:
    """
    Convert detailed tactical map(s) to game-ready map(s).

    Supports both single maps and multi-map clusters. For multi-map clusters,
    all sheets are converted and output to a single game_map folder.

    Args:
        detailed_map_dir: Path to detailed map folder (contains SVG and hexdata.json)
        config: Optional configuration dict with:
            - terrain_style: 'auto', 'temperate', or 'arid'
            - elevation_intensity: float multiplier (default 1.0)
            - frame_color: hex color string
            - label_rows: list of row numbers to label

    Returns:
        Path to output game_map folder
    """
    config = config or {}

    print(f"\n{'=' * 60}")
    print("Game Map Conversion")
    print('=' * 60)

    # Find input files
    detailed_map_dir = Path(detailed_map_dir)
    svg_files = sorted(detailed_map_dir.glob("*_tactical.svg"))
    json_files = sorted(detailed_map_dir.glob("*_hexdata.json"))

    if not svg_files:
        raise FileNotFoundError(f"No *_tactical.svg found in {detailed_map_dir}")
    if not json_files:
        raise FileNotFoundError(f"No *_hexdata.json found in {detailed_map_dir}")

    # Check for multi-map cluster
    is_multi_map = len(svg_files) > 1
    if is_multi_map:
        print(f"\nMulti-map cluster detected: {len(svg_files)} sheets")

    # Create output directory
    output_dir = detailed_map_dir / "game_map"
    output_dir.mkdir(exist_ok=True)

    # Build map of svg -> json by matching names
    svg_to_json = {}
    for svg_path in svg_files:
        base_name = svg_path.stem.replace('_tactical', '')
        # Find matching json
        matching_json = None
        for json_path in json_files:
            if json_path.stem.replace('_hexdata', '') == base_name:
                matching_json = json_path
                break
        if matching_json:
            svg_to_json[svg_path] = matching_json
        else:
            print(f"  Warning: No matching hexdata.json for {svg_path.name}")

    # Convert each sheet
    converted_count = 0
    for svg_path, json_path in svg_to_json.items():
        if is_multi_map:
            sheet_name = svg_path.stem.replace('_tactical', '').split('_')[-1]
            print(f"\n{'=' * 60}")
            print(f"Converting Sheet {sheet_name}")
            print('=' * 60)

        convert_single_sheet(svg_path, json_path, output_dir, config, is_multi_map)
        converted_count += 1

    print(f"\n{'=' * 60}")
    print(f"Game map conversion complete!")
    print(f"  Converted: {converted_count} sheet(s)")
    print(f"  Output: {output_dir}")
    print('=' * 60)

    return output_dir


def main():
    """Command-line interface for game map conversion."""
    import argparse

    parser = argparse.ArgumentParser(description='Convert detailed tactical map to game map')
    parser.add_argument('map_dir', type=Path, help='Path to detailed map folder')
    parser.add_argument('--style', choices=['auto', 'temperate', 'arid'], default='auto',
                       help='Terrain color style')
    parser.add_argument('--intensity', type=float, default=1.0,
                       help='Elevation tint intensity (default: 1.0)')

    args = parser.parse_args()

    config = {
        'terrain_style': args.style,
        'elevation_intensity': args.intensity,
    }

    convert_to_game_map(args.map_dir, config)


if __name__ == "__main__":
    main()
