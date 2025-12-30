"""
Tests for map_utils module.

Run with: pytest tests/test_map_utils.py -v
"""

import math
import pytest
from map_utils import Bounds, RotationConfig, CoordinateTransformer, LayerManager, LayerZOrder


class TestBounds:
    """Tests for the Bounds dataclass."""

    def test_bounds_creation(self):
        """Test basic bounds creation."""
        bounds = Bounds(min_x=0, max_x=100, min_y=0, max_y=50)
        assert bounds.min_x == 0
        assert bounds.max_x == 100
        assert bounds.min_y == 0
        assert bounds.max_y == 50

    def test_bounds_width_height(self):
        """Test width and height properties."""
        bounds = Bounds(min_x=10, max_x=110, min_y=20, max_y=70)
        assert bounds.width == 100
        assert bounds.height == 50

    def test_bounds_center(self):
        """Test center property."""
        bounds = Bounds(min_x=0, max_x=100, min_y=0, max_y=100)
        center = bounds.center
        assert center == (50, 50)

    def test_bounds_contains(self):
        """Test point containment check."""
        bounds = Bounds(min_x=0, max_x=100, min_y=0, max_y=100)
        assert bounds.contains(50, 50) is True
        assert bounds.contains(0, 0) is True
        assert bounds.contains(100, 100) is True
        assert bounds.contains(-1, 50) is False
        assert bounds.contains(50, 101) is False

    def test_bounds_expand(self):
        """Test bounds expansion."""
        bounds = Bounds(min_x=10, max_x=90, min_y=20, max_y=80)
        expanded = bounds.expand(10)
        assert expanded.min_x == 0
        assert expanded.max_x == 100
        assert expanded.min_y == 10
        assert expanded.max_y == 90

    def test_bounds_as_tuple(self):
        """Test conversion to tuple."""
        bounds = Bounds(min_x=1, max_x=2, min_y=3, max_y=4)
        assert bounds.as_tuple() == (1, 3, 2, 4)


class TestRotationConfig:
    """Tests for the RotationConfig class."""

    def test_no_rotation(self):
        """Test with zero rotation."""
        rot = RotationConfig(angle_deg=0, center_x=100, center_y=100)
        assert rot.is_rotated is False
        assert rot.angle_rad == 0
        assert rot.cos_angle == 1.0
        assert rot.sin_angle == 0.0

    def test_rotation_properties(self):
        """Test rotation angle properties."""
        rot = RotationConfig(angle_deg=90, center_x=0, center_y=0)
        assert rot.is_rotated is True
        assert rot.angle_rad == pytest.approx(math.pi / 2)
        assert rot.cos_angle == pytest.approx(0, abs=1e-10)
        assert rot.sin_angle == pytest.approx(1)

    def test_rotate_point_no_rotation(self):
        """Test point rotation with zero angle."""
        rot = RotationConfig(angle_deg=0, center_x=50, center_y=50)
        x, y = rot.rotate_point(100, 100)
        assert x == 100
        assert y == 100

    def test_rotate_point_90_degrees(self):
        """Test 90-degree rotation."""
        rot = RotationConfig(angle_deg=90, center_x=0, center_y=0)
        x, y = rot.rotate_point(10, 0)
        assert x == pytest.approx(0, abs=1e-10)
        assert y == pytest.approx(10)

    def test_rotate_point_180_degrees(self):
        """Test 180-degree rotation."""
        rot = RotationConfig(angle_deg=180, center_x=50, center_y=50)
        x, y = rot.rotate_point(100, 50)
        assert x == pytest.approx(0)
        assert y == pytest.approx(50)

    def test_rotate_point_45_degrees(self):
        """Test 45-degree rotation."""
        rot = RotationConfig(angle_deg=45, center_x=0, center_y=0)
        x, y = rot.rotate_point(10, 0)
        expected = 10 / math.sqrt(2)
        assert x == pytest.approx(expected)
        assert y == pytest.approx(expected)

    def test_calculate_expanded_bounds_no_rotation(self):
        """Test bounds expansion with no rotation."""
        rot = RotationConfig(angle_deg=0, center_x=0, center_y=0)
        expand_x, expand_y = rot.calculate_expanded_bounds(100, 50)
        assert expand_x == 0
        assert expand_y == 0

    def test_calculate_expanded_bounds_45_degrees(self):
        """Test bounds expansion with 45-degree rotation."""
        rot = RotationConfig(angle_deg=45, center_x=0, center_y=0)
        expand_x, expand_y = rot.calculate_expanded_bounds(100, 100)
        # Square rotated 45 degrees has bounding box sqrt(2) times larger
        # Expansion = (rotated_size - original_size) / 2 + 500
        sqrt2 = math.sqrt(2)
        expected_expand = (100 * sqrt2 - 100) / 2 + 500
        assert expand_x == pytest.approx(expected_expand)
        assert expand_y == pytest.approx(expected_expand)

    def test_get_svg_transform_no_rotation(self):
        """Test SVG transform string with no rotation."""
        rot = RotationConfig(angle_deg=0, center_x=100, center_y=100)
        assert rot.get_svg_transform() == ""

    def test_get_svg_transform_with_rotation(self):
        """Test SVG transform string with rotation."""
        rot = RotationConfig(angle_deg=45, center_x=100, center_y=200)
        transform = rot.get_svg_transform()
        assert "rotate(45" in transform
        assert "100" in transform
        assert "200" in transform


class TestCoordinateTransformer:
    """Tests for the CoordinateTransformer class."""

    @pytest.fixture
    def transformer(self):
        """Create a transformer for testing."""
        bounds = Bounds(min_x=380000, max_x=390000, min_y=2290000, max_y=2300000)
        return CoordinateTransformer(
            grid_crs="EPSG:32651",  # UTM zone 51N
            map_bounds=bounds,
            svg_offset_x=100,
            svg_offset_y=100
        )

    def test_utm_to_svg_at_origin(self, transformer):
        """Test UTM to SVG at map origin (min_x, max_y)."""
        # Top-left of map should be at SVG offset position
        svg_x, svg_y = transformer.utm_to_svg(380000, 2300000)
        assert svg_x == 100  # svg_offset_x
        assert svg_y == 100  # svg_offset_y

    def test_utm_to_svg_at_corner(self, transformer):
        """Test UTM to SVG at map corner (max_x, min_y)."""
        svg_x, svg_y = transformer.utm_to_svg(390000, 2290000)
        # X: (390000 - 380000) + 100 = 10100
        # Y: (2300000 - 2290000) + 100 = 10100
        assert svg_x == 10100
        assert svg_y == 10100

    def test_svg_to_utm_roundtrip(self, transformer):
        """Test SVG to UTM conversion (inverse)."""
        original_x, original_y = 385000, 2295000
        svg_x, svg_y = transformer.utm_to_svg(original_x, original_y)
        result_x, result_y = transformer.svg_to_utm(svg_x, svg_y)
        assert result_x == pytest.approx(original_x)
        assert result_y == pytest.approx(original_y)

    def test_wgs84_to_utm(self, transformer):
        """Test WGS84 to UTM conversion."""
        # Approximately center of Batanes
        lon, lat = 121.84, 20.76
        x, y = transformer.wgs84_to_utm(lon, lat)
        # Should be within the bounds area
        assert 370000 < x < 400000
        assert 2280000 < y < 2310000

    def test_utm_to_wgs84(self, transformer):
        """Test UTM to WGS84 conversion."""
        x, y = 385000, 2295000
        lon, lat = transformer.utm_to_wgs84(x, y)
        # Should be in Philippines area
        assert 121 < lon < 123
        assert 20 < lat < 22

    def test_wgs84_to_svg(self, transformer):
        """Test direct WGS84 to SVG conversion."""
        # First get UTM coords that are within bounds, then convert to WGS84
        test_utm_x, test_utm_y = 385000, 2295000
        lon, lat = transformer.utm_to_wgs84(test_utm_x, test_utm_y)
        svg_x, svg_y = transformer.wgs84_to_svg(lon, lat)
        # Should produce the same result as direct UTM to SVG
        expected_x, expected_y = transformer.utm_to_svg(test_utm_x, test_utm_y)
        assert svg_x == pytest.approx(expected_x, abs=1)
        assert svg_y == pytest.approx(expected_y, abs=1)


class TestLayerZOrder:
    """Tests for LayerZOrder constants."""

    def test_z_order_ordering(self):
        """Test that z-order constants are properly ordered."""
        # Background should be lowest
        assert LayerZOrder.BACKGROUND < LayerZOrder.OCEAN
        assert LayerZOrder.OCEAN < LayerZOrder.TERRAIN_FOREST
        assert LayerZOrder.TERRAIN_FOREST < LayerZOrder.CONTOURS_REGULAR
        assert LayerZOrder.CONTOURS_REGULAR < LayerZOrder.ROADS_HIGHWAY
        assert LayerZOrder.ROADS_HIGHWAY < LayerZOrder.BUILDINGS
        assert LayerZOrder.BUILDINGS < LayerZOrder.HEX_GRID

    def test_fixed_layers_above_rotating(self):
        """Test that fixed layers have higher z-order than rotating layers."""
        max_rotating = max(
            LayerZOrder.MGRS_GRID,
            LayerZOrder.PEAKS,
            LayerZOrder.PLACES
        )
        min_fixed = min(
            LayerZOrder.OUT_OF_PLAY_FRAME,
            LayerZOrder.HEX_GRID
        )
        assert min_fixed > max_rotating


class TestLayerManager:
    """Tests for the LayerManager class."""

    @pytest.fixture
    def mock_dwg(self):
        """Create a mock drawing object."""
        import svgwrite
        return svgwrite.Drawing()

    @pytest.fixture
    def manager(self, mock_dwg):
        """Create a LayerManager for testing."""
        return LayerManager(mock_dwg)

    def test_register_layer(self, manager):
        """Test layer registration."""
        layer = manager.register_layer("test_layer", z_order=100)
        assert layer is not None
        assert manager.get_layer("test_layer") is layer

    def test_register_multiple_layers(self, manager):
        """Test registering multiple layers."""
        layer1 = manager.register_layer("layer1", z_order=100)
        layer2 = manager.register_layer("layer2", z_order=200)
        layer3 = manager.register_layer("layer3", z_order=50)

        layers = manager.get_layers_by_z_order()
        assert len(layers) == 3
        # Should be ordered by z_order
        assert layers[0] is layer3  # z=50
        assert layers[1] is layer1  # z=100
        assert layers[2] is layer2  # z=200

    def test_filter_rotating_layers(self, manager):
        """Test filtering layers by rotation."""
        rotating = manager.register_layer("rotating", z_order=100, rotates=True)
        fixed = manager.register_layer("fixed", z_order=200, rotates=False)

        rotating_layers = manager.get_rotating_layers()
        fixed_layers = manager.get_fixed_layers()

        assert rotating in rotating_layers
        assert fixed not in rotating_layers
        assert fixed in fixed_layers
        assert rotating not in fixed_layers

    def test_hidden_layer(self, manager):
        """Test creating a hidden layer."""
        layer = manager.register_layer("hidden", z_order=100, visible=False)
        assert layer.attribs.get('visibility') == 'hidden'

    def test_layer_with_clip_path(self, manager):
        """Test creating a layer with clip-path."""
        layer = manager.register_layer(
            "clipped",
            z_order=100,
            clip_path="url(#my-clip)"
        )
        assert layer.attribs.get('clip-path') == "url(#my-clip)"
