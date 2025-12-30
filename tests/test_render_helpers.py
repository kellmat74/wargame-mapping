"""
Tests for render_helpers module.

Run with: pytest tests/test_render_helpers.py -v
"""

import math
import pytest
from render_helpers import (
    get_point_and_angle_at_distance,
    get_line_length,
    create_hex_polygon_svg
)


class TestGetLineLength:
    """Tests for get_line_length function."""

    def test_horizontal_line(self):
        """Test length of horizontal line."""
        coords = [(0, 0), (100, 0)]
        assert get_line_length(coords) == 100

    def test_vertical_line(self):
        """Test length of vertical line."""
        coords = [(0, 0), (0, 50)]
        assert get_line_length(coords) == 50

    def test_diagonal_line(self):
        """Test length of diagonal line."""
        coords = [(0, 0), (3, 4)]
        assert get_line_length(coords) == 5  # 3-4-5 triangle

    def test_multi_segment_line(self):
        """Test length of line with multiple segments."""
        coords = [(0, 0), (10, 0), (10, 10), (20, 10)]
        assert get_line_length(coords) == 30  # 10 + 10 + 10

    def test_single_point(self):
        """Test with degenerate single-segment line."""
        coords = [(5, 5), (5, 5)]
        assert get_line_length(coords) == 0


class TestGetPointAndAngleAtDistance:
    """Tests for get_point_and_angle_at_distance function."""

    def test_start_of_line(self):
        """Test getting point at start of line."""
        coords = [(0, 0), (100, 0)]
        x, y, angle = get_point_and_angle_at_distance(coords, 0)
        assert x == pytest.approx(0)
        assert y == pytest.approx(0)
        assert angle == pytest.approx(0)  # Pointing right

    def test_middle_of_line(self):
        """Test getting point in middle of line."""
        coords = [(0, 0), (100, 0)]
        x, y, angle = get_point_and_angle_at_distance(coords, 50)
        assert x == pytest.approx(50)
        assert y == pytest.approx(0)
        assert angle == pytest.approx(0)

    def test_end_of_line(self):
        """Test getting point at end of line."""
        coords = [(0, 0), (100, 0)]
        x, y, angle = get_point_and_angle_at_distance(coords, 100)
        assert x == pytest.approx(100)
        assert y == pytest.approx(0)

    def test_beyond_line(self):
        """Test distance beyond line length returns end point."""
        coords = [(0, 0), (100, 0)]
        x, y, angle = get_point_and_angle_at_distance(coords, 200)
        assert x == pytest.approx(100)
        assert y == pytest.approx(0)

    def test_vertical_line_angle(self):
        """Test angle on vertical line."""
        coords = [(0, 0), (0, 100)]
        x, y, angle = get_point_and_angle_at_distance(coords, 50)
        assert x == pytest.approx(0)
        assert y == pytest.approx(50)
        assert angle == pytest.approx(90)  # Pointing up

    def test_multi_segment_line(self):
        """Test point on multi-segment line."""
        coords = [(0, 0), (10, 0), (10, 10)]
        # At distance 15, we should be 5 units up the second segment
        x, y, angle = get_point_and_angle_at_distance(coords, 15)
        assert x == pytest.approx(10)
        assert y == pytest.approx(5)
        assert angle == pytest.approx(90)


class TestCreateHexPolygonSvg:
    """Tests for create_hex_polygon_svg function."""

    def test_hex_has_six_points(self):
        """Test that hexagon has 6 vertices."""
        def identity_transform(x, y):
            return (x, y)

        points = create_hex_polygon_svg(0, 0, 100, identity_transform)
        assert len(points) == 6

    def test_hex_vertices_correct_distance(self):
        """Test that all vertices are at correct distance from center."""
        def identity_transform(x, y):
            return (x, y)

        size = 100
        points = create_hex_polygon_svg(0, 0, size, identity_transform)

        for x, y in points:
            distance = math.sqrt(x**2 + y**2)
            assert distance == pytest.approx(size)

    def test_hex_with_offset_center(self):
        """Test hexagon centered at non-origin point."""
        def identity_transform(x, y):
            return (x, y)

        cx, cy = 500, 300
        size = 50
        points = create_hex_polygon_svg(cx, cy, size, identity_transform)

        for x, y in points:
            distance = math.sqrt((x - cx)**2 + (y - cy)**2)
            assert distance == pytest.approx(size)

    def test_hex_with_transform(self):
        """Test hexagon with coordinate transform."""
        def offset_transform(x, y):
            return (x + 1000, y + 2000)

        points = create_hex_polygon_svg(0, 0, 100, offset_transform)

        # All points should be offset by (1000, 2000)
        for x, y in points:
            assert x >= 900  # Should be around 1000 +/- 100
            assert x <= 1100
            assert y >= 1900  # Should be around 2000 +/- 100
            assert y <= 2100
