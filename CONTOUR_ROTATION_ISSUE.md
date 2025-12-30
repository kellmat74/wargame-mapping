# Contour Rotation Issue - Investigation Notes

## Issue Summary
When map rotation is enabled, contours appear approximately **2.5km too far north** relative to other map features (coastline, reference tiles, roads, etc.). All other features remain properly aligned with each other.

## Key Observations

### What Works
- **Without rotation**: Contours align perfectly with coastline and terrain
- **With rotation**: All features EXCEPT contours align correctly
- **terrain_open**: Now fills correctly for both rotated and non-rotated maps
- **Island detection**: Works correctly with 5000m coastline buffer

### What's Broken
- **Contours with rotation**: Shifted ~2.5km north relative to everything else
- This is approximately 0.022 degrees of latitude
- The offset is consistent regardless of map region

## Technical Analysis

### Coordinate Flow for Contours
1. `config.data_min_x/y` and `config.data_max_x/y` (UTM) define the area to load
2. These are transformed to WGS84 for DEM access (lines 506-507)
3. DEM window is read using `from_bounds()` (lines 518-523)
4. `skimage.measure.find_contours()` generates contours in pixel space
5. Pixel coords converted to WGS84 using `window_transform` (lines 569-570)
6. WGS84 coords transformed to UTM (line 573)
7. `to_svg()` converts UTM to SVG coordinates

### What Changes With Rotation
When `rotation_deg != 0`:
- `config.data_min_x/y` and `config.data_max_x/y` are EXPANDED (lines 347-364)
- Expansion formula accounts for rotated bounding box
- This means a LARGER DEM window is read
- More contours are generated covering a larger area

### Hypothesis: DEM Window Bounds Mismatch

The 2.5km offset (~0.022°) is suspiciously close to the rotation expansion amount.

**Possible cause**: When transforming expanded UTM bounds to WGS84, the resulting rectangle may not align with what `from_bounds()` expects. The `window_transform` returned by rasterio might have an origin that doesn't match our expected geographic coordinates.

### Diagnostic Data Points

From testing with Batanes map (50° rotation):
```
Map bounds (UTM): X(374489-383364), Y(2293295-2299033)
Data bounds (expanded): X(373377-384477), Y(2290421-2301907)

DEM window transform origin: (121.77, 20.82) in WGS84
- This corresponds to the NORTH-WEST corner of the window

Coastline sample point: E380960, N2295032 -> SVG (6470, 4001)
20m contour near coastline: E380985, N2295037 -> SVG (6496, 3996)
Offset: only 26m (NOT 2.5km!)
```

**Critical finding**: The RAW coordinate data shows contours are correctly positioned (26m offset, not 2.5km). This means the offset must be happening during SVG rendering or rotation transform application.

### Where the Bug Likely Is

Since raw coordinates are correct but visual output is wrong WITH rotation:

1. **Rotation center calculation** (lines 3229-3230):
   ```python
   rot_cx = content_offset_m + trim_width_m / 2
   rot_cy = content_offset_m + trim_height_m / 2
   ```

2. **Multiple groups with same rotation** - `rotated_reference` and `rotated_content` both use same center, but are they actually rendering the same?

3. **to_svg() function** uses `config.min_x` and `config.max_y` (MAP bounds), not DATA bounds. But contours are generated from DATA bounds area. Could there be an offset when contours extend beyond map bounds?

4. **SVG clipping** - The viewbox-clip-rotated might be affecting contours differently?

## Recommended Investigation Steps

### Step 1: Verify Rotation Center
Add debug output to confirm rotation center in SVG coordinates matches visual center of map.

### Step 2: Compare SVG Coordinates Directly
Extract actual polyline coordinates from SVG file for:
- A coastline segment
- A contour at the same location
Compare their SVG X,Y values - if different, the bug is in coordinate generation. If same, the bug is in rotation/clipping.

### Step 3: Test Without Rotation Transform
Temporarily disable the rotation transform on `rotated_content` but keep the expanded data loading. See if contours align. This isolates whether it's a data loading issue or a rotation issue.

### Step 4: Test Contours in rotated_reference
Move contour layers to `rotated_reference` group (where reference tiles are) and see if they then align with reference tiles.

---

# Code Cleanup Recommendations

## Current Pain Points

### 1. Monolithic render_tactical_svg Function
The `render_tactical_svg` function is **~2200 lines** (lines 1195-3400+). This makes debugging extremely difficult because:
- State is passed implicitly through many local variables
- Hard to test individual rendering stages
- Changes in one section can have unexpected effects elsewhere

**Recommendation**: Split into smaller functions:
- `create_svg_document()` - document setup, defs, patterns
- `render_terrain_layers()` - terrain polygons, ocean, islands
- `render_infrastructure()` - roads, buildings, railways
- `render_contours()` - contour lines and labels
- `render_overlays()` - hex grid, MGRS, markers
- `assemble_layers()` - final layer ordering and rotation

### 2. Coordinate System Complexity
Multiple coordinate systems are in play:
- WGS84 (EPSG:4326) - DEM, raw GeoJSON
- UTM (EPSG:326XX) - GRID_CRS, internal calculations
- SVG coordinates - final output

The `to_svg()` function is defined INSIDE `render_tactical_svg` (line 1397), making it:
- Hard to test independently
- Hard to verify correctness
- Dependent on closure variables

**Recommendation**:
- Create a `CoordinateTransformer` class that handles all transforms
- Include methods: `utm_to_svg()`, `wgs84_to_utm()`, `svg_to_utm()` (for debugging)
- Include validation/debug mode that logs transforms

### 3. Config Bounds Confusion
`MapConfig` has multiple sets of bounds:
- `min_x, max_x, min_y, max_y` - map bounds
- `data_min_x, data_max_x, data_min_y, data_max_y` - data loading bounds

These are calculated in `__post_init__` and `shift_center`, with rotation expansion logic duplicated.

**Recommendation**:
- Add clear documentation on what each bound represents
- Consider a `Bounds` dataclass with `map_bounds` and `data_bounds` properties
- Single method for calculating expansion, not duplicated

### 4. Layer Management
Layers are created as individual variables (`layer_contours_regular`, `layer_coastline`, etc.) then added to groups in a specific order. This is error-prone.

**Recommendation**:
- Create a `LayerManager` class
- Register layers with z-order and rotation membership
- Automatic assembly based on configuration

### 5. Rotation Logic Scattered
Rotation-related code is scattered throughout:
- Lines 347-364: Data bounds expansion
- Lines 391-403: Same in shift_center
- Lines 1567-1573: terrain_open coverage
- Lines 2737-2746: Rotation helper in rendering
- Lines 3226-3310: Final rotation application

**Recommendation**:
- Centralize rotation logic in a `RotationConfig` class
- Single source of truth for: angle, center, expansion factors
- Helper methods for rotating points, calculating expanded bounds

### 6. No Unit Tests
No test suite exists. This makes refactoring risky and bugs hard to catch.

**Recommendation**:
- Add pytest with test fixtures for sample configs
- Unit tests for coordinate transforms
- Integration tests that verify known-good outputs
- Regression test for this specific contour rotation bug once fixed

## Specific Refactoring for Contour Issue

To make this bug easier to diagnose, create:

```python
class ContourGenerator:
    def __init__(self, dem, config):
        self.dem = dem
        self.config = config
        self.debug_info = {}

    def generate(self, interval=20):
        # Store debug info at each step
        self.debug_info['requested_bounds_utm'] = (...)
        self.debug_info['requested_bounds_wgs84'] = (...)
        self.debug_info['actual_window_bounds'] = (...)
        self.debug_info['sample_contour_coords'] = (...)

        # Generate contours...
        return contours

    def validate_alignment(self, coastline_gdf):
        """Compare contour positions to coastline for debugging."""
        # Return offset statistics
        pass
```

This would allow:
```python
contour_gen = ContourGenerator(dem, config)
contours = contour_gen.generate()
print(contour_gen.debug_info)  # See exactly what happened
alignment = contour_gen.validate_alignment(coastline)  # Verify alignment
```

---

## Files Modified During This Session

1. **tactical_map.py**:
   - Line 91-93: Added `DEM_NORTHING_OFFSET_M` constant (set to 0)
   - Line 575-577: Added offset application in contour generation
   - Line 904: Added `buffer_m` parameter to `load_optional_features`
   - Lines 1563-1616: Modified terrain_open rendering for rotation
   - Line 3672-3673: Increased coastline buffer to 5000m

2. **map_config.json**: Currently set to Batanes with 50° rotation

---

---

## CRITICAL FINDING: Wrong CRS Being Used!

**This is likely THE root cause of the contour offset issue.**

### The Bug
Line 45 of `tactical_map.py`:
```python
GRID_CRS = "EPSG:3826"  # TWD97 / TM2 Taiwan (meters)
```

This CRS is **hard-coded for Taiwan** but is being used for ALL maps, including Batanes (Philippines)!

### Impact
- EPSG:3826 is a Transverse Mercator projection centered on Taiwan (~121°E, **24°N**)
- Batanes is at ~121.8°E, **20.7°N** - about **3.3 degrees south** of Taiwan's projection center
- At this distance from the projection's central meridian/latitude, there will be significant distortion
- **The ~2.5km offset we're seeing could be caused by this CRS mismatch!**

### Evidence
The code correctly calculates `config.utm_crs` (line 287):
```python
self.utm_crs = get_utm_crs(self.center_lon, self.center_lat)
```

But then uses `GRID_CRS` (Taiwan CRS) for nearly all transformations instead of `config.utm_crs`.

### Why It's Worse With Rotation
- Without rotation: All features use the same (wrong) CRS, so they're consistently wrong together
- With rotation: The expanded data bounds transform differently through the wrong CRS, amplifying the error
- DEM window coordinates may not map correctly back to the expected UTM positions

### The Fix
Replace all uses of `GRID_CRS` with `config.utm_crs` (or dynamically computed UTM CRS).

**Lines that need updating** (at minimum):
- Line 328: `calculate_bounds()` - uses GRID_CRS
- Line 415: `shift_center()` - uses GRID_CRS
- Line 505, 547: `generate_contours()` - uses GRID_CRS
- Line 601: terrain classification - uses GRID_CRS
- Line 777, 883, 924, 1782, 1872, 3634: various data loading - uses GRID_CRS

### Quick Test
Change line 45 to use UTM zone 51N (correct for Batanes):
```python
GRID_CRS = "EPSG:32651"  # UTM zone 51N
```

Then regenerate the rotated Batanes map and check if contours align.

---

## Next Steps for Morning

### Priority 1: Test the CRS Fix
1. **Quick test**: Change line 45 from `EPSG:3826` (Taiwan) to `EPSG:32651` (UTM 51N for Philippines)
2. Regenerate the rotated Batanes map
3. Check if contours now align correctly

If this fixes it, the proper solution is to make `GRID_CRS` dynamic based on map location.

### Priority 2: If CRS Fix Doesn't Work
1. Comment out the rotation transform (lines 3237 and 3243) but keep expanded data loading
2. If contours align without rotation, the bug is in the rotation logic
3. If they don't align, the bug is in coordinate generation with expanded bounds

### Priority 3: Add Debug Logging
Print actual SVG coordinates for a sample contour point and a nearby coastline point to compare directly.

3. **Check window_transform**: The `window_transform` returned by rasterio might have an unexpected origin when the requested bounds span a large area with rotation expansion.

4. **Consider alternative**: Instead of using expanded data bounds for DEM, always use the same bounds and just render more contours. The current approach loads different DEM windows for rotated vs non-rotated, which could be the source of inconsistency.

---

## Session 2 Changes (2025-12-30 Morning)

### Changes Made to tactical_map.py

#### 1. CRS Change (line 45)
**Before:** `GRID_CRS = "EPSG:3826"  # TWD97 / TM2 Taiwan (meters)`
**After:** `GRID_CRS = "EPSG:32651"  # UTM zone 51N (meters) - for Philippines/Batanes region`
**Result:** Did NOT fix the contour offset issue

#### 2. Removed DEM buffer (lines 509-514)
**Before:**
```python
# Add buffer for edge contours
buffer = 0.01  # degrees
dem_min_x -= buffer
dem_min_y -= buffer
dem_max_x += buffer
dem_max_y += buffer
```
**After:** Removed the buffer entirely (commented out with explanation)
**Reason:** Buffer caused contours to extend beyond data bounds, creating offset issues with rotation

#### 3. Added contour clipping (lines 2427-2435, 2483-2496)
**Added:** Clip box creation and geometry clipping before rendering contours
```python
contour_clip_box = shapely_box(config.data_min_x, config.data_min_y,
                               config.data_max_x, config.data_max_y)
# ...
clipped_geom = geom.intersection(contour_clip_box)
```
**Reason:** Prevent contours from extending beyond data area

### Key Finding: Negative SVG Y Coordinates

Analysis of the generated SVG revealed:
- **Coastline polylines**: All have positive Y coordinates (within visible area)
- **Contour polylines**: Many have **negative Y coordinates** (e.g., Y range: -2824 to 8797)
- 4000+ contour points have negative Y, meaning they're positioned ABOVE the visible map area

This means contours in the expanded data area (outside map bounds) get negative SVG coordinates.
When the rotation transform is applied around the map center, these points rotate into view
but at unexpected positions.

### Current State
- Both Coastline and Contours_Regular are in the same SVG group (Rotated_Content)
- Both have the same rotation transform applied
- Raw SVG coordinates show contours extend to negative Y values while coastline doesn't
- The offset is still present after CRS fix and clipping changes

### Next Test
Switch to Yonaguni (known working case) with 0° rotation, then 45° rotation to verify baseline.

---

## Session 3 Changes (2025-12-30 Afternoon - Context Recovery)

### Root Cause Identified: DEM-OSM Misalignment

**Key Discovery:** The Batanes SRTM DEM data is systematically misaligned with OSM coastline data.

#### Evidence from DEM Alignment Analysis

Sampling elevation at OSM coastline points (which should be ~0m):

**Batanes:**
- 170 coastline points (23%) show elevation ≥20m
- Mean elevation at coastline: 14.0m
- Max elevation at coastline: 90m (!)
- Average distance to find sea level going East: 110m

**Yonaguni (control):**
- Only 13 coastline points (4.5%) show elevation ≥20m
- Mean elevation at coastline: 6.9m
- Much better alignment

This explains why Yonaguni works with rotation but Batanes doesn't - the Batanes DEM is ~100m west of the OSM data.

#### How Misalignment Causes "North" Offset After Rotation

1. **Pre-rotation:** Contours are ~88m WEST of coastline (dx ≈ -88m)
2. **After 50° rotation:** The west offset rotates to become a north-west offset
3. **Perception:** User sees contours "north" of where they should be

Mathematical verification:
- Pre-rotation offset: dx=-88m, dy=-57m
- Post-50° rotation: dx=-12m, dy=-104m
- The Y component increases after rotation, making the offset appear "north"

### Fix Applied: DEM Easting Offset

Added `DEM_EASTING_OFFSET_M` constant to correct for systematic DEM-OSM misalignment.

**Code changes (tactical_map.py):**

1. Added offset constant (line 96):
```python
DEM_EASTING_OFFSET_M = -150  # Batanes DEM is shifted west of OSM, shift east to correct
```

2. Applied offset in contour generation (line 576):
```python
gx -= DEM_EASTING_OFFSET_M
gy -= DEM_NORTHING_OFFSET_M
```

### Current Status

- The offset correction is being applied correctly (verified: +150m east shift)
- The rotated Batanes map has been regenerated with the correction
- **User needs to visually verify** if the contours now align better with coastline/terrain

### Remaining Questions

1. Is the 150m east correction sufficient, or does it need fine-tuning?
2. Should this offset be configurable per-region instead of global?
3. Is there a better DEM source for Batanes with correct alignment?

### Technical Notes

The offset appears to vary by location within Batanes:
- Y range 500-2000: Nearly aligned (dx≈8m)
- Y range 2000-3500: Contours slightly south of coastline (dx≈8m, dy≈39m)
- Y range 3500-5000: Larger offset (dx≈-92m, dy≈-72m)
- Y range 5000-6500: Medium offset (dx≈-60m, dy≈-21m)

This variation suggests the DEM misalignment is not perfectly uniform, possibly due to:
- Different SRTM tiles with different alignment
- Non-linear distortion in the DEM
- Coastal cliffs causing apparent misalignment (legitimate elevation at cliff edges)

---

## Session 4: Code Refactoring (2025-12-30)

### Summary
Implemented all code cleanup recommendations from this document. The DEM offset fix (-150m easting) did not produce visible improvement in the contour alignment, suggesting the root cause may be more complex than simple DEM misalignment.

### New Files Created

#### 1. `map_utils.py` - Utility Classes
```python
# Key classes:
- Bounds: Dataclass for rectangular bounds (min_x, max_x, min_y, max_y)
  - Properties: width, height, center
  - Methods: contains(), expand(), as_tuple()

- RotationConfig: Centralized rotation handling
  - Properties: angle_rad, is_rotated, cos_angle, sin_angle
  - Methods: rotate_point(), calculate_expanded_bounds(), get_svg_transform()

- CoordinateTransformer: Unified coordinate transformations
  - Methods: utm_to_svg(), svg_to_utm(), wgs84_to_utm(), utm_to_wgs84(), wgs84_to_svg()

- LayerManager: SVG layer registration and z-ordering
  - Methods: register_layer(), get_layer(), get_layers_by_z_order()
  - get_rotating_layers(), get_fixed_layers(), assemble_into_group()

- LayerZOrder: Constants for standard z-order values (BACKGROUND=0 through PRINT_GUIDES=3020)
```

#### 2. `render_helpers.py` - Rendering Functions
```python
# Extracted functions for rendering:
- get_point_and_angle_at_distance(line_coords, target_distance) -> (x, y, angle)
- get_line_length(line_coords) -> float
- render_polygons(gdf, layer, dwg, fill_color, to_svg, ...) -> count
- render_linestrings(gdf, layer, dwg, color, width, to_svg, ...) -> count
- render_contours(contours, layer_regular, layer_index, layer_labels, ...) -> (contour_count, label_count)
- render_roads(roads_gdf, road_layers, dwg, to_svg, ...) -> Dict[str, int]
- create_hex_polygon_svg(center_x, center_y, size, to_svg) -> List[Tuple]
```

#### 3. `tests/` - Pytest Test Suite
- `tests/__init__.py`
- `tests/test_map_utils.py` - 29 tests for utility classes
- `tests/test_render_helpers.py` - 15 tests for render helpers
- **Total: 44 passing tests**

### Changes to `tactical_map.py`
1. Added import for new utility classes:
   ```python
   from map_utils import Bounds, RotationConfig, CoordinateTransformer, LayerManager, LayerZOrder
   ```

2. Added helper properties to MapConfig:
   ```python
   @property
   def map_bounds(self) -> Bounds:
       """Get map bounds as a Bounds object."""

   @property
   def data_bounds(self) -> Bounds:
       """Get expanded data bounds as a Bounds object."""

   def get_rotation_config(self, center_x, center_y) -> RotationConfig:
       """Get a RotationConfig for the given SVG rotation center."""

   def create_coordinate_transformer(self, svg_offset_x, svg_offset_y) -> CoordinateTransformer:
       """Create a CoordinateTransformer for this map configuration."""
   ```

### Git Commits Made
1. `fb4cf92` - Add DEM alignment offset and document contour rotation issue
2. `a452663` - Add utility classes and pytest infrastructure for code cleanup
3. `2770612` - Add render_helpers module with extracted rendering functions

### Remaining Work
1. **Integrate render_helpers into render_tactical_svg** - The helper functions exist but haven't been integrated into the main 2200-line function yet
2. **Root cause of contour offset still unknown** - The -150m DEM easting offset didn't visibly improve alignment
3. **Consider alternative approaches**:
   - Different DEM source for Batanes
   - Per-region offset configuration
   - Manual contour adjustment based on coastline matching

### Debug Scripts Created (not committed)
- `debug_contour_offset.py` - Verify DEM offset is being applied
- `debug_coverage_gap.py` - Find gaps in contour coverage
- `debug_dem_alignment.py` - Check DEM alignment with coastline
- `debug_svg_coords.py` - Compare SVG coordinates
- `debug_svg_rotation.py` - Analyze rotation effects
- `debug_svg_structure.py` - Analyze SVG structure
- `debug_xy_offset.py` - Analyze X/Y offset patterns

### Key Technical Findings
1. **SVG structure is correct** - Both coastline and contours have identical transform chains
2. **DEM-OSM misalignment confirmed** - 23% of Batanes coastline points show elevation ≥20m vs only 4.5% for Yonaguni
3. **Measured offset (~100m) doesn't match reported offset (~1.5km)** - Suggests either measurement methodology issue or additional factors
4. **Offset varies by location** - Not a simple uniform shift, possibly multiple SRTM tiles with different alignment
