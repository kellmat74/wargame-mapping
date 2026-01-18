"""
Tests for game_map_converter module.

Run with: pytest tests/test_game_map_converter.py -v

These tests cover:
1. Rotation detection from SVG structure (for logging/diagnostics)
2. Coordinate transformation utilities (reverse_rotate_point)
3. Hex polygon generation

Key principle: The game map converter does NOT generate overlays.
All overlays are pre-generated during detail map creation (tactical_map.py)
and the converter just unhides and optionally customizes them.
"""

import math
import pytest
from xml.etree import ElementTree as ET
from io import StringIO

import sys
sys.path.insert(0, '.')

from game_map_converter import (
    get_rotation_info,
    reverse_rotate_point,
    create_hex_polygon_points,
)


# === Test Fixtures ===

def create_minimal_svg(with_rotation: bool = False, rotation_angle: float = -30) -> ET.ElementTree:
    """
    Create a minimal SVG structure for testing.

    Args:
        with_rotation: If True, include Rotated_Content with rotation transform
        rotation_angle: Rotation angle in degrees (default -30)

    Returns:
        ElementTree with minimal SVG structure matching tactical_map.py output
    """
    svg_content = '''<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10000 6500">
  <g id="Master_Content">
    {rotated_content}
    <g id="Hex_Grid">
      <line x1="100" y1="100" x2="200" y2="100"/>
    </g>
    <g id="Hex_Markers">
      <circle cx="150" cy="150" r="5"/>
    </g>
    <g id="Hex_Labels">
      <text x="150" y="140">01.01</text>
    </g>
  </g>
</svg>'''

    if with_rotation:
        # Rotated_Content contains terrain layers with rotation transform
        rotated_content = f'''<g id="Rotated_Content" transform="rotate({rotation_angle}, 5000, 3250)">
      <g id="Terrain_Open">
        <polygon points="0,0 100,0 100,100 0,100"/>
      </g>
      <g id="Roads">
        <line x1="0" y1="0" x2="100" y2="100"/>
      </g>
    </g>'''
    else:
        # Non-rotated: terrain is direct child of Master_Content
        rotated_content = '''<g id="Terrain_Open">
      <polygon points="0,0 100,0 100,100 0,100"/>
    </g>
    <g id="Roads">
      <line x1="0" y1="0" x2="100" y2="100"/>
    </g>'''

    svg_xml = svg_content.format(rotated_content=rotated_content)
    return ET.ElementTree(ET.fromstring(svg_xml))


# === Tests for get_rotation_info ===

class TestGetRotationInfo:
    """Tests for get_rotation_info function."""

    def test_detects_rotation_in_rotated_svg(self):
        """Test that rotation is detected when Rotated_Content has transform."""
        tree = create_minimal_svg(with_rotation=True, rotation_angle=-30)
        result = get_rotation_info(tree)

        assert result is not None, "Should detect rotation"
        angle, cx, cy = result
        assert angle == pytest.approx(-30)
        assert cx == pytest.approx(5000)
        assert cy == pytest.approx(3250)

    def test_returns_none_for_non_rotated_svg(self):
        """Test that None is returned when no rotation exists."""
        tree = create_minimal_svg(with_rotation=False)
        result = get_rotation_info(tree)

        assert result is None, "Should return None for non-rotated SVG"

    def test_detects_positive_rotation(self):
        """Test detection of positive rotation angle."""
        tree = create_minimal_svg(with_rotation=True, rotation_angle=45)
        result = get_rotation_info(tree)

        assert result is not None
        angle, _, _ = result
        assert angle == pytest.approx(45)

    def test_detects_zero_rotation(self):
        """Test that zero rotation is detected (still returns tuple, not None)."""
        # Create SVG with explicit zero rotation
        svg_xml = '''<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <g id="Rotated_Content" transform="rotate(0, 100, 100)">
    <rect x="0" y="0" width="10" height="10"/>
  </g>
</svg>'''
        tree = ET.ElementTree(ET.fromstring(svg_xml))
        result = get_rotation_info(tree)

        assert result is not None
        angle, _, _ = result
        assert angle == pytest.approx(0)


# === Tests for reverse_rotate_point ===

class TestReverseRotatePoint:
    """Tests for reverse_rotate_point function."""

    def test_zero_rotation_returns_same_point(self):
        """Test that zero rotation returns the original point."""
        x, y = 100, 200
        rx, ry = reverse_rotate_point(x, y, 0, 50, 50)

        assert rx == pytest.approx(x)
        assert ry == pytest.approx(y)

    def test_180_rotation_around_origin(self):
        """Test 180° rotation around origin."""
        # Point (10, 0) rotated 180° around origin should be (-10, 0)
        # Reverse rotation of 180° should also give (-10, 0)
        rx, ry = reverse_rotate_point(10, 0, 180, 0, 0)

        assert rx == pytest.approx(-10)
        assert ry == pytest.approx(0, abs=1e-10)

    def test_90_rotation_around_origin(self):
        """Test 90° rotation around origin."""
        # If original point was (10, 0) and we rotated 90° CW, we'd get (0, 10)
        # Reverse-rotating (0, 10) by 90° should give back (10, 0)
        rx, ry = reverse_rotate_point(0, 10, 90, 0, 0)

        assert rx == pytest.approx(10, abs=1e-10)
        assert ry == pytest.approx(0, abs=1e-10)

    def test_negative_rotation(self):
        """Test negative (counter-clockwise) rotation."""
        # -90° rotation: (10, 0) -> (0, -10)
        # Reverse-rotating (0, -10) by -90° should give (10, 0)
        rx, ry = reverse_rotate_point(0, -10, -90, 0, 0)

        assert rx == pytest.approx(10, abs=1e-10)
        assert ry == pytest.approx(0, abs=1e-10)

    def test_rotation_around_non_origin_center(self):
        """Test rotation around a non-origin center point."""
        # Center at (100, 100), rotate point (200, 100) by 90°
        # After 90° rotation: (100, 200)
        # Reverse should give back (200, 100)
        rx, ry = reverse_rotate_point(100, 200, 90, 100, 100)

        assert rx == pytest.approx(200)
        assert ry == pytest.approx(100, abs=1e-10)

    def test_roundtrip_rotation(self):
        """Test that rotating then reverse-rotating gives original point."""
        original_x, original_y = 150.5, 275.3
        angle = -30
        cx, cy = 5000, 3250

        # Forward rotate
        rad = math.radians(angle)
        tx = original_x - cx
        ty = original_y - cy
        rotated_x = tx * math.cos(rad) - ty * math.sin(rad) + cx
        rotated_y = tx * math.sin(rad) + ty * math.cos(rad) + cy

        # Reverse rotate
        rx, ry = reverse_rotate_point(rotated_x, rotated_y, angle, cx, cy)

        assert rx == pytest.approx(original_x)
        assert ry == pytest.approx(original_y)


# === Tests for create_hex_polygon_points ===

class TestCreateHexPolygonPoints:
    """Tests for hex polygon generation."""

    def test_hex_has_six_vertices(self):
        """Test that hex polygon has 6 vertices."""
        points_str = create_hex_polygon_points(0, 0, 100)
        points = points_str.split()
        assert len(points) == 6

    def test_vertices_at_correct_distance(self):
        """Test that all vertices are at correct distance from center."""
        cx, cy, size = 500, 300, 100
        points_str = create_hex_polygon_points(cx, cy, size)

        for point in points_str.split():
            x, y = map(float, point.split(','))
            distance = math.sqrt((x - cx)**2 + (y - cy)**2)
            assert distance == pytest.approx(size), \
                f"Vertex ({x}, {y}) should be {size} from center ({cx}, {cy})"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
