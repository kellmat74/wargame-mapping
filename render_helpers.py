"""
Rendering helper functions for tactical map SVG generation.

These functions are extracted from render_tactical_svg to improve
code organization and testability. They can be gradually integrated
into the main rendering pipeline.
"""

import math
from typing import List, Dict, Tuple, Callable, Any, Optional
from shapely.geometry import LineString, Polygon, MultiLineString, box as shapely_box


def get_point_and_angle_at_distance(
    line_coords: List[Tuple[float, float]],
    target_distance: float
) -> Tuple[float, float, float]:
    """Get position and angle at a given distance along a line.

    Args:
        line_coords: List of (x, y) coordinate tuples defining the line
        target_distance: Distance along line to find point

    Returns:
        Tuple of (x, y, angle_degrees) at the target distance
    """
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


def get_line_length(line_coords: List[Tuple[float, float]]) -> float:
    """Calculate total length of a line from its coordinates.

    Args:
        line_coords: List of (x, y) coordinate tuples

    Returns:
        Total length of the line
    """
    total = 0
    for i in range(len(line_coords) - 1):
        x1, y1 = line_coords[i]
        x2, y2 = line_coords[i + 1]
        total += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return total


def render_polygons(
    gdf,
    layer,
    dwg,
    fill_color: str,
    to_svg: Callable,
    stroke_color: Optional[str] = None,
    stroke_width: float = 0,
    pattern_url: Optional[str] = None
) -> int:
    """Render polygon features to an SVG layer.

    Args:
        gdf: GeoDataFrame containing polygon geometries
        layer: SVG group to add polygons to
        dwg: svgwrite Drawing object
        fill_color: Fill color (or 'none')
        to_svg: Coordinate transform function (x, y) -> (svg_x, svg_y)
        stroke_color: Optional stroke color
        stroke_width: Stroke width in meters
        pattern_url: Optional pattern URL (e.g., "url(#forest-pattern)")

    Returns:
        Number of polygons rendered
    """
    if gdf is None or gdf.empty:
        return 0

    count = 0
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue

        polygons = []
        if geom.geom_type == "Polygon":
            polygons = [geom]
        elif geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)
        else:
            continue

        for poly in polygons:
            if poly.is_empty:
                continue

            # Get exterior ring coordinates
            coords = list(poly.exterior.coords)
            svg_points = [to_svg(x, y) for x, y in coords]

            fill = pattern_url if pattern_url else fill_color
            props = {
                'points': svg_points,
                'fill': fill,
            }

            if stroke_color:
                props['stroke'] = stroke_color
                props['stroke_width'] = stroke_width

            layer.add(dwg.polygon(**props))
            count += 1

    return count


def render_linestrings(
    gdf,
    layer,
    dwg,
    color: str,
    width: float,
    to_svg: Callable,
    dash: Optional[str] = None
) -> int:
    """Render linestring features to an SVG layer.

    Args:
        gdf: GeoDataFrame containing linestring geometries
        layer: SVG group to add lines to
        dwg: svgwrite Drawing object
        color: Stroke color
        width: Stroke width in meters
        to_svg: Coordinate transform function
        dash: Optional dash pattern (e.g., "10,5")

    Returns:
        Number of linestrings rendered
    """
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


def render_contours(
    contours: List[Dict],
    layer_regular,
    layer_index,
    layer_labels,
    dwg,
    to_svg: Callable,
    clip_bounds: Optional[Tuple[float, float, float, float]] = None,
    contour_color: str = "#8B4513",
    index_color: str = "#654321",
    contour_width: float = 2,
    index_width: float = 5,
    label_interval: float = 1000,
    label_size: float = 25
) -> Tuple[int, int]:
    """Render contour lines and labels.

    Args:
        contours: List of contour dicts with 'geometry', 'elevation', 'is_index'
        layer_regular: SVG group for regular contours
        layer_index: SVG group for index contours
        layer_labels: SVG group for contour labels
        dwg: svgwrite Drawing object
        to_svg: Coordinate transform function
        clip_bounds: Optional (min_x, min_y, max_x, max_y) to clip contours
        contour_color: Color for regular contours
        index_color: Color for index contours
        contour_width: Width for regular contours
        index_width: Width for index contours
        label_interval: Distance between labels on index contours
        label_size: Font size for labels

    Returns:
        Tuple of (contour_count, label_count)
    """
    contour_count = 0
    label_count = 0

    # Create clip box if bounds provided
    clip_box = None
    if clip_bounds:
        clip_box = shapely_box(*clip_bounds)

    for contour in contours:
        geom = contour["geometry"]
        is_index = contour["is_index"]
        elevation = contour["elevation"]

        # Skip index contours at elevation 0 (sea level)
        if is_index and elevation == 0:
            continue

        # Clip contour if clip box provided
        if clip_box:
            clipped_geom = geom.intersection(clip_box)
            if clipped_geom.is_empty:
                continue
        else:
            clipped_geom = geom

        # Handle different geometry types
        if clipped_geom.geom_type == "MultiLineString":
            lines = list(clipped_geom.geoms)
        elif clipped_geom.geom_type == "LineString":
            lines = [clipped_geom]
        elif clipped_geom.geom_type == "GeometryCollection":
            lines = [g for g in clipped_geom.geoms if g.geom_type == "LineString"]
        else:
            continue

        target_layer = layer_index if is_index else layer_regular
        color = index_color if is_index else contour_color
        width = index_width if is_index else contour_width

        for line in lines:
            if len(line.coords) < 2:
                continue

            svg_points = [to_svg(x, y) for x, y in line.coords]

            target_layer.add(dwg.polyline(
                points=svg_points,
                stroke=color,
                stroke_width=width,
                fill="none",
            ))
            contour_count += 1

            # Add labels for index contours
            if is_index:
                world_coords = list(line.coords)
                line_length = get_line_length(world_coords)

                if line_length < label_interval / 2:
                    continue

                start_offset = label_interval / 2
                distance = start_offset

                while distance < line_length - 100:
                    wx, wy, world_angle = get_point_and_angle_at_distance(
                        world_coords, distance
                    )
                    sx, sy = to_svg(wx, wy)

                    # SVG Y is inverted, so flip the angle
                    svg_angle = -world_angle
                    if svg_angle > 90 or svg_angle < -90:
                        svg_angle += 180

                    text_elem = dwg.text(
                        f"{int(elevation)}",
                        insert=(sx, sy),
                        text_anchor="middle",
                        font_size=label_size,
                        fill=index_color,
                        font_family="sans-serif",
                        dominant_baseline="middle",
                    )
                    text_elem["transform"] = f"rotate({svg_angle}, {sx}, {sy})"
                    layer_labels.add(text_elem)
                    label_count += 1

                    distance += label_interval

    return contour_count, label_count


def render_roads(
    roads_gdf,
    road_layers: Dict[str, Any],
    dwg,
    to_svg: Callable,
    classify_road_type: Callable,
    road_colors: Dict[str, str],
    road_widths: Dict[str, float],
    road_outlines: Dict[str, str],
    outline_width: float = 1
) -> Dict[str, int]:
    """Render road features to appropriate layers.

    Args:
        roads_gdf: GeoDataFrame containing road geometries
        road_layers: Dict mapping road type to SVG layer
        dwg: svgwrite Drawing object
        to_svg: Coordinate transform function
        classify_road_type: Function to classify highway tag to road type
        road_colors: Dict mapping road type to fill color
        road_widths: Dict mapping road type to width
        road_outlines: Dict mapping road type to outline color
        outline_width: Width of road outlines

    Returns:
        Dict of road counts by type
    """
    if roads_gdf is None or roads_gdf.empty:
        return {}

    road_counts = {}

    for _, road_row in roads_gdf.iterrows():
        highway = road_row.get("highway", "")
        road_type = classify_road_type(highway)
        geom = road_row.geometry

        road_counts[road_type] = road_counts.get(road_type, 0) + 1

        if geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        elif geom.geom_type == "LineString":
            lines = [geom]
        else:
            continue

        color = road_colors.get(road_type, "#000000")
        width = road_widths.get(road_type, 5)
        outline = road_outlines.get(road_type)
        target_layer = road_layers.get(road_type)

        if target_layer is None:
            continue

        for line in lines:
            if len(line.coords) < 2:
                continue

            svg_points = [to_svg(x, y) for x, y in line.coords]

            # Draw outline first if applicable
            if outline:
                target_layer.add(dwg.polyline(
                    points=svg_points,
                    stroke=outline,
                    stroke_width=width + (2 * outline_width),
                    fill='none',
                    stroke_linecap='round',
                    stroke_linejoin='round',
                ))

            # Draw road fill
            target_layer.add(dwg.polyline(
                points=svg_points,
                stroke=color,
                stroke_width=width,
                fill='none',
                stroke_linecap='round',
                stroke_linejoin='round',
            ))

    return road_counts


def create_hex_polygon_svg(
    center_x: float,
    center_y: float,
    size: float,
    to_svg: Callable
) -> List[Tuple[float, float]]:
    """Create SVG points for a flat-top hexagon.

    Args:
        center_x: Hex center X in world coordinates
        center_y: Hex center Y in world coordinates
        size: Hex size (center to vertex)
        to_svg: Coordinate transform function

    Returns:
        List of (x, y) SVG coordinate tuples for polygon points
    """
    points = []
    for i in range(6):
        angle = math.radians(60 * i)
        wx = center_x + size * math.cos(angle)
        wy = center_y + size * math.sin(angle)
        points.append(to_svg(wx, wy))
    return points
