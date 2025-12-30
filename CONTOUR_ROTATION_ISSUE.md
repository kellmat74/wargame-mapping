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

### Session 6 (2025-12-30 - Multi-MGRS-Square Support)
- Implemented multi-MGRS-square data loading for maps near square boundaries
- Added `get_mgrs_squares_for_bounds()` to detect all squares covering data bounds
- Added `get_all_data_paths()` and `load_geojson_from_all_squares()` for multi-source loading
- Modified `load_dem()` to merge multiple DEMs using rasterio.merge
- Auto-extracts missing adjacent squares from available PBFs
- **Fixed terrain_open/island detection issue** (see corrections below)

### Git Commits
1. `fb4cf92` - Add DEM alignment offset and document contour rotation issue
2. `a452663` - Add utility classes and pytest infrastructure for code cleanup
3. `2770612` - Add render_helpers module with extracted rendering functions
4. `0a4f442` - Update issue documentation
5. `e10ea13` - Integrate render_helpers into render_tactical_svg
6. `047e6a1` - Fix island detection for multi-MGRS-square coastline data

---

## Corrections to Previous Assumptions

### ✅ CORRECTED: terrain_open "Small Polygons" Issue

**Original assumption**: When terrain_open showed as small scattered polygons after multi-square implementation, the assumption was that coastline data from multiple MGRS squares wasn't forming proper closed rings due to overlapping/duplicate segments.

**Actual root cause**: The detected island polygons were **not sorted by area**. The list had small polygons first (from pre-existing polygon geometries), so when rendering, tiny fragments were processed before the main 28.91 km² island polygon. The main island was in the list but at a later index.

**The fix**:
1. Added `unary_union` before `linemerge` (this does help with overlapping segments)
2. **Critical fix**: Sort `detected_island_polygons` by area descending so largest renders first
3. Added `polygonize` fallback if linemerge doesn't find closed rings

**Verification**: After sorting, the main island polygon (3306 coordinate points, 28.91 km²) renders correctly as the base terrain fill.

### Yonaguni Status: ✅ WORKING

Yonaguni now works correctly with:
- -20° CCW rotation
- 300m map-relative shift (110° bearing = ESE)
- Multi-MGRS-square data from 4 squares: 51R/VG, VH, WG, WH
- 101 contour segments (merged from all squares)
- ~~31 island polygons detected, largest 28.91 km²~~ **Now using OSM land polygons (33 polygons)**

---

### Session 7 (2025-12-30 - OSM Land Polygons)

**Problem**: The old approach for rendering ocean/land was overly complex:
- Built polygons from coastline linestrings using `linemerge` and `polygonize`
- Used elevation sampling to determine which polygons were ocean vs land
- Created "hundreds of ocean objects" on some maps (user complaint)
- Different code paths for island vs mainland maps

**Solution**: Switched to official OSM land polygons approach:
- Download pre-built land polygons from https://osmdata.openstreetmap.de/data/land-polygons.html
- `land-polygons-split-4326.zip` (~800MB, one-time download, cached in `cache/`)
- Simple rendering: ocean rectangle as background, land polygons on top
- No more complex coastline processing or elevation sampling
- Same code path for both island and mainland maps

**Key changes**:
1. Added `download_land_polygons()` - downloads and caches OSM land polygons (~800MB)
2. Added `load_land_polygons_for_bounds()` - loads land polygons for specific bounding box using GeoPandas spatial filtering
3. Simplified `render_tactical_svg()` - removed ~200 lines of complex coastline/ocean polygon code
4. Added `land_polygons` to enhanced_features dict

**Results**:
- Yonaguni: 33 land polygons (clean, simple)
- El Nido: 4 land polygons after clipping (clean, simple)
- Code is much simpler and more maintainable

---

### Batanes Status: ✅ RESOLVED

**The Batanes contour offset issue is now resolved!**

After switching to OSM land polygons (Session 7), Batanes now renders correctly with:
- 0° rotation: Perfect alignment
- 50° rotation: Perfect alignment

The root cause was likely related to the complex coastline-to-polygon conversion code
that was removed when switching to OSM land polygons. The old code used elevation
sampling to determine ocean vs land, which may have introduced coordinate artifacts
that accumulated when combined with rotation transforms.

---

## Debug Scripts (Not Committed)
- `debug_contour_offset.py` - Verify DEM offset application
- `debug_coverage_gap.py` - Find gaps in contour coverage
- `debug_dem_alignment.py` - Check DEM-OSM alignment
- `debug_svg_coords.py` - Compare SVG coordinates
- `debug_svg_rotation.py` - Analyze rotation effects
- `debug_svg_structure.py` - Analyze SVG structure
- `debug_xy_offset.py` - Analyze X/Y offset patterns
