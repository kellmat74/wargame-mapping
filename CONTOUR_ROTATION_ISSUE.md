# Contour Rotation Issue - Investigation Notes

## Issue Summary
When map rotation is enabled for Batanes (50°), contours appear approximately **1.5km too far north** relative to other map features (coastline, terrain_open, roads, etc.). All other features remain properly aligned with each other.

**Important**: This issue is specific to Batanes. Yonaguni works correctly with 45° rotation.

## Key Observations

### What Works
- **Without rotation**: Contours align perfectly with coastline and terrain (both Batanes and Yonaguni)
- **With rotation on Yonaguni**: All features including contours align correctly at 45° rotation
- **terrain_open**: Now fills correctly for both rotated and non-rotated maps
- **Island detection**: Works correctly with 5000m coastline buffer
- **SVG structure**: Both coastline and contours have identical rotation transforms

### What's Broken
- **Contours with rotation on Batanes only**: Shifted ~1.5km north relative to coastline
- The issue does NOT occur on Yonaguni with similar rotation

## Root Cause Analysis

### Confirmed: DEM-OSM Misalignment
The Batanes SRTM DEM data is systematically misaligned with OSM coastline data:

**Batanes DEM:**
- 23% of coastline points show elevation ≥20m (should be ~0m)
- Mean elevation at coastline: 14.0m
- Max elevation at coastline: 90m
- The DEM appears shifted ~100m **west** relative to OSM

**Yonaguni DEM (working correctly):**
- Only 4.5% of coastline points show elevation ≥20m
- Mean elevation at coastline: 6.9m
- Much better alignment

### Unresolved Mystery
The measured offset in SVG coordinates is only ~100m, but the visual offset appears to be ~1.5km. This 15x discrepancy was never fully explained. Possible causes:
1. The "nearest contour" measurement methodology may not capture the visual offset accurately
2. The offset may be more pronounced in specific areas that the sampling didn't capture
3. There may be additional factors beyond simple DEM misalignment

## Hypotheses Tested

### ❌ DISPROVEN: CRS Mismatch (EPSG:3826 vs EPSG:32651)
**Original hypothesis**: Using Taiwan's CRS (EPSG:3826) for Philippines maps was causing distortion.

**Test result**: Changed to EPSG:32651 (UTM zone 51N). Did NOT fix the contour offset.

**Conclusion**: The CRS was wrong but wasn't the cause of this specific issue.

### ❌ DISPROVEN: Rotation Transform Logic Error
**Original hypothesis**: Different rotation centers or transforms applied to contours vs coastline.

**Test result**: SVG analysis confirmed both layers have identical transform chains:
- Both are children of `Rotated_Content` group
- Both have same transform: `rotate(50, 5538.4, 3730.0)`
- No clip-path differences

**Conclusion**: SVG rotation is applied correctly and identically to both layers.

### ❌ DISPROVEN: DEM Buffer Causing Extension
**Original hypothesis**: The 0.01° DEM buffer was causing contours to extend incorrectly.

**Test result**: Removed the buffer. Did NOT fix the offset.

### ⚠️ PARTIALLY CONFIRMED: DEM Easting Offset
**Hypothesis**: DEM is shifted west relative to OSM, causing contours to be west of coastline, which appears as "north" after 50° rotation.

**Test result**: Applied -150m easting offset to contour generation. User reported no visible improvement.

**Conclusion**: The offset exists but the correction didn't produce visible results. Either:
- The offset amount is wrong
- There are additional factors
- The visual perception differs from measured coordinates

## Technical Details

### Coordinate Flow for Contours
1. `config.data_min_x/y` and `config.data_max_x/y` (UTM) define the area to load
2. These are transformed to WGS84 for DEM access
3. DEM window is read using `from_bounds()`
4. `skimage.measure.find_contours()` generates contours in pixel space
5. Pixel coords converted to WGS84 using `window_transform`
6. WGS84 coords transformed to UTM with optional offset correction
7. `to_svg()` converts UTM to SVG coordinates

### Offset by Location (SVG Y ranges)
The offset varies by location within the map:
- Y 500-2000: Nearly aligned (dx≈8m, dy≈-9m)
- Y 2000-3500: Contours slightly south (dx≈8m, dy≈39m)
- Y 3500-5000: Larger offset (dx≈-92m, dy≈-72m)
- Y 5000-6500: Medium offset (dx≈-60m, dy≈-21m)

This variation suggests non-uniform DEM alignment, possibly from multiple SRTM tiles.

---

## Code Cleanup Recommendations

**Status: IMPLEMENTED** (Session 4)

### 1. Monolithic render_tactical_svg Function
The `render_tactical_svg` function is **~2200 lines**.

**Implemented**: Created `render_helpers.py` with extracted functions:
- `get_point_and_angle_at_distance()`
- `get_line_length()`
- `render_polygons()`
- `render_linestrings()`
- `render_contours()`
- `render_roads()`
- `create_hex_polygon_svg()`

**Integrated**: Local function definitions in `render_tactical_svg` now delegate to these helpers, reducing ~95 lines of duplicated code.

### 2. Coordinate System Complexity
**Implemented**: Created `CoordinateTransformer` class in `map_utils.py`:
- `utm_to_svg()`, `svg_to_utm()`
- `wgs84_to_utm()`, `utm_to_wgs84()`
- `wgs84_to_svg()`

### 3. Config Bounds Confusion
**Implemented**: Created `Bounds` dataclass and added to MapConfig:
- `map_bounds` property
- `data_bounds` property

### 4. Layer Management
**Implemented**: Created `LayerManager` class with:
- `register_layer()` with z-order and rotation membership
- `get_rotating_layers()`, `get_fixed_layers()`
- `LayerZOrder` constants

### 5. Rotation Logic
**Implemented**: Created `RotationConfig` class with:
- `rotate_point()`
- `calculate_expanded_bounds()`
- `get_svg_transform()`

### 6. Unit Tests
**Implemented**: 44 passing tests in `tests/` directory

---

## Changes Made

### Session 1 (Initial Investigation)
- Identified terrain_open rotation issue (fixed)
- Identified contour offset issue
- Created initial diagnostic notes

### Session 2 (2025-12-30 Morning)
- Changed GRID_CRS from EPSG:3826 to EPSG:32651 - did NOT fix issue
- Removed DEM buffer - did NOT fix issue
- Added contour clipping to data bounds
- Tested Yonaguni - works perfectly with rotation
- Confirmed Batanes 0° rotation works, 50° rotation broken

### Session 3 (2025-12-30 Afternoon)
- Identified DEM-OSM misalignment as likely root cause
- Added DEM_EASTING_OFFSET_M and DEM_NORTHING_OFFSET_M constants
- Applied -150m easting offset - no visible improvement reported

### Session 4 (2025-12-30 - Code Refactoring)
- Created `map_utils.py` with utility classes
- Created `render_helpers.py` with extracted functions
- Added 44 pytest tests
- Updated MapConfig with helper properties

### Session 5 (2025-12-30 - Integration)
- Integrated render_helpers into render_tactical_svg
- Reduced ~95 lines of duplicated code
- All 44 tests still passing

### Git Commits
1. `fb4cf92` - Add DEM alignment offset and document contour rotation issue
2. `a452663` - Add utility classes and pytest infrastructure for code cleanup
3. `2770612` - Add render_helpers module with extracted rendering functions
4. `0a4f442` - Update issue documentation
5. `e10ea13` - Integrate render_helpers into render_tactical_svg

---

## Remaining Work

### Bug Fix (Contour Offset)
1. **Investigate the 100m vs 1.5km discrepancy** - Why does measured offset differ so much from visual?
2. **Try larger DEM offset values** - Maybe -150m isn't enough
3. **Consider per-region offset configuration** - Different regions may need different corrections
4. **Try alternative DEM source** - ASTER GDEM or other sources may have better alignment

### Debug Scripts (Not Committed)
- `debug_contour_offset.py` - Verify DEM offset application
- `debug_coverage_gap.py` - Find gaps in contour coverage
- `debug_dem_alignment.py` - Check DEM-OSM alignment
- `debug_svg_coords.py` - Compare SVG coordinates
- `debug_svg_rotation.py` - Analyze rotation effects
- `debug_svg_structure.py` - Analyze SVG structure
- `debug_xy_offset.py` - Analyze X/Y offset patterns
