# Taiwan Tactical Map Generator - Lessons Learned

## Project Summary

Built a tactical-scale hex wargame map generator for Taiwan, producing US Army military-style maps suitable for tabletop wargaming. The system generates 250m hex maps from geographic data with configurable center coordinates.

## Accomplishments

### Phase 1: Full Island Map (10km hexes)
- Created `hexgrid.py` for full Taiwan island at 10km hex scale
- Implemented terrain classification from OSM landcover data
- Basic SVG rendering with terrain colors

### Phase 2: Tactical Map Generator (250m hexes)
- Created `tactical_map.py` for detailed tactical maps
- **Grid**: 47 x 26 hexes (fixed size for 34" x 22" print sheets)
- **Scale**: 250m per hex (flat edge to flat edge)
- **Map coverage**: ~10.2km x 6.5km per sheet

### Features Implemented
1. **Configurable center point** - Specify lat/lon to center the map
2. **Contour lines** - 20m interval with 100m index contours (brown)
3. **Terrain rendering** - Continuous polygons (not hex-bound) from OSM landcover
4. **Buildings** - Downloaded from OSM Overpass API, rendered as black fills
5. **Roads** - Classified as highway/major/minor/track with appropriate colors
6. **MGRS grid** - 1km UTM grid lines with labels on all 4 sides
7. **Hex grid overlay** - Dark grey (#444444) with 0.6px stroke
8. **Hex numbering** - XX.YY format (01.01 to 47.26) at top of each hex
9. **Center circles** - Open circles at hex centers for terrain markers
10. **Out-of-play border** - Dark grey (#404040) outside hex grid
11. **SVG layer groups** - Organized for easy editing in Affinity Designer

## Technical Details

### Coordinate Systems
- **WGS84 (EPSG:4326)** - Input lat/lon coordinates
- **TWD97 (EPSG:3826)** - Projected CRS for Taiwan (meters), used for all calculations
- **UTM Zone 51N (EPSG:32651)** - Used for MGRS grid generation

### Hex Geometry (Flat-top)
```
HEX_SIZE_M = 250  # flat edge to flat edge
size = HEX_SIZE_M / sqrt(3)  # center to vertex (~144.34m)
col_spacing = 1.5 * size  # horizontal spacing between columns
row_spacing = HEX_SIZE_M  # vertical spacing between rows
```

### Terrain Colors (Military Style)
```python
TERRAIN_COLORS = {
    "water": "#c6ecff",       # Light blue
    "urban": "#ffc0cb",       # Pink/rose
    "forest": "#c8e6c8",      # Light green
    "orchard": "#d4e6c8",     # Lighter green
    "marsh": "#c6ecff",       # Light blue
    "open": "#fffef0",        # Off-white/cream
}
```

### Contour Colors
```python
CONTOUR_COLOR = "#8b4513"         # Brown (regular)
INDEX_CONTOUR_COLOR = "#654321"   # Darker brown (100m intervals)
```

### Road Colors
```python
ROAD_COLORS = {
    "highway": "#ff0000",     # Red
    "major_road": "#ff6600",  # Orange-red
    "minor_road": "#000000",  # Black
    "track": "#000000",       # Black dashed
}
```

## SVG Layer Structure

Layers are added in this z-order (bottom to top):
1. `Background` - Dark grey base fill
2. `Terrain_Open` - Base cream fill for playable area
3. `Terrain_Water` - Water polygons (unclipped)
4. `Terrain_Marsh` - Marsh/wetland polygons (unclipped)
5. `Terrain_Forest` - Forest polygons (unclipped)
6. `Terrain_Orchard` - Orchard polygons (unclipped)
7. `Terrain_Urban` - Urban area polygons (unclipped)
8. `Buildings` - Building footprints (unclipped)
9. `Contours_Regular` - 20m contour lines (unclipped)
10. `Contours_Index` - 100m index contours (unclipped)
11. `Roads` - All road segments (unclipped)
12. `Out_Of_Play_Frame` - Dark grey frame masking everything outside playable area
13. `Hex_Grid` - Hex outline polygons
14. `Hex_Markers` - Center circles (for terrain type fill in Affinity)
15. `Hex_Labels` - XX.YY coordinate labels
16. `MGRS_Grid` - UTM grid lines
17. `MGRS_Labels` - Grid coordinate labels
18. `Map_Data` - Map metadata in top out-of-play border:
    - MAP CENTER: MGRS (10-digit with zone/square), LAT/LNG, UTM Easting/Northing
    - GRID INFO: Hex dimensions, coverage area
    - CORNER COORDINATES: WGS84 lat/lon for NW, NE, SW, SE corners

**Note:** Map features (terrain, buildings, contours, roads) are rendered unclipped for easier editing in Affinity Designer. The `Out_Of_Play_Frame` layer acts as a mask to hide anything outside the playable hex area.

## Data Sources

### Required Files in `data/taiwan/`
- `coastline.geojson` - Taiwan outline
- `elevation.tif` - DEM raster (SRTM or similar)
- `landcover.geojson` - OSM landuse/natural polygons
- `roads.geojson` - OSM highway network
- `buildings.geojson` - Building footprints (downloaded via Overpass API)
- `places.geojson` - Cities/towns
- `rivers.geojson` - River network
- `infrastructure.geojson` - Ports, airfields

### Downloading Buildings
Buildings were downloaded using OSM Overpass API for the Taichung area:
```python
# Bounding box: [south, west, north, east]
bbox = [23.95, 120.45, 24.35, 120.90]
query = f"""
[out:json][timeout:300];
(way["building"]({bbox}););
out body; >; out skel qt;
"""
```

## Output Files

### Per-map outputs in `output/taiwan/{map_name}/`
- `{name}_tactical.svg` - Vector map (~9 MB)
- `{name}_tactical.pdf` - 150dpi PDF (~1.3 MB)
- `{name}_hexdata.json` - Machine-readable hex data

### PDF Export
```python
import cairosvg
# 34" x 22" at 150dpi
cairosvg.svg2pdf(
    url='input.svg',
    write_to='output.pdf',
    output_width=34 * 150,  # 5100px
    output_height=22 * 150,  # 3300px
)
```

## How to Generate a New Map

1. Edit `tactical_map.py` main() function:
```python
config = MapConfig(
    name="new_location",      # Output folder name
    center_lat=24.1477,       # Latitude of map center
    center_lon=120.6736,      # Longitude of map center
    region="taiwan",          # Data folder to use
)
```

2. Optional: Shift center after creation:
```python
config.shift_center(north_m=1000, east_m=-500)  # Move 1km N, 500m W
```

3. Run generation:
```bash
source venv/bin/activate
python tactical_map.py
```

4. Export PDF:
```python
import cairosvg
cairosvg.svg2pdf(url='output.svg', write_to='output.pdf',
                 output_width=5100, output_height=3300)
```

## Lessons Learned

### 1. Terrain Rendering
- **Don't clip terrain to hex boundaries** - Render actual geographic shapes for authentic military map appearance
- **Use playable area union** - Create union of all hex polygons to clip features cleanly
- **Layer order matters** - Render base terrain first, then overlay features

### 2. Performance
- Building clipping to playable area is slow with 20k+ buildings
- Contour generation from DEM takes time; consider caching
- Use spatial indexes (sindex) for large polygon datasets

### 3. Coordinate Transforms
- Always use `pyproj.Transformer` for coordinate conversions
- TWD97 (EPSG:3826) is best for Taiwan - proper meter units
- UTM Zone 51N needed for proper MGRS grid alignment

### 4. SVG Organization
- Use `<g id="LayerName">` groups for Affinity Designer compatibility
- Add all groups to drawing in correct z-order at the end
- svgwrite groups are created with `dwg.g(id="Name")`

### 5. Hex Numbering
- Use 1-indexed coordinates for human readability (01-47, 01-26)
- Format: `{col:02d}.{row:02d}` for consistent width
- Position at top of hex, ~10% of hex height for font size

### 6. MGRS Grid
- Labels on all 4 sides (N, S, E, W) for usability
- Light grey color so it doesn't compete with hex grid
- Show last 2 digits of km value (e.g., "67" for 267km)

### 7. User Workflow
- SVG output for Affinity Designer post-processing
- Center circles left unfilled for manual terrain marking
- PDF export at 150dpi for print proofing

## Dependencies

```
geopandas>=0.14.0
shapely>=2.0.0
pyproj>=3.6.0
rasterio>=1.3.0
numpy>=1.24.0
svgwrite>=1.4.0
scikit-image  # for contour generation
cairosvg  # for PDF export (requires cairo system library)
```

Install cairo on macOS: `brew install cairo`

## Game Map Converter (December 2025)

### Overview
Created a two-stage pipeline:
1. **Stage 1**: Detailed tactical map generation (existing `tactical_map.py`)
2. **Stage 2**: Game map conversion (`game_map_converter.py`) - simplifies maps for gameplay

### Game Map Features
- Elevation-based terrain tinting (darker at higher elevations)
- Hillside shading bands along elevation transitions
- Maroon border frame (DA PrePRESS style)
- Simplified hex labels (rows 1, 5, 10, 15, 20, 25 only)
- Hidden detail layers (paths, powerlines, tree rows)
- Output: SVG, PNG, PDF

### Critical SVG Structure Discovery
The detailed SVG has a specific structure that must be understood for proper overlay insertion:

```
Master_Content/
├── Rotated_Content/          <- HAS rotation transform
│   ├── Terrain_Open
│   ├── Terrain_Forest
│   ├── Roads
│   └── ... (all terrain/feature layers)
├── Hex_Grid                   <- NO rotation (screen coordinates)
├── Hex_Markers                <- NO rotation (screen coordinates)
└── Hex_Labels
```

**Key insight**: `Hex_Grid` and `Hex_Markers` are OUTSIDE `Rotated_Content`. They use screen coordinates with no rotation transform applied. When extracting hex positions from `Hex_Markers` circles, the coordinates are already in final screen space.

**Overlay insertion**: Game overlays must be inserted as siblings of `Hex_Grid` (inside `Master_Content` but outside `Rotated_Content`) to avoid incorrect rotation being applied.

### Relative Elevation Bands
Changed from absolute elevation bands to relative bands based on each map's terrain:

**Old (absolute)**:
- Band 0: 0-100m, Band 1: 100-500m, etc.
- Problem: Flat maps (e.g., 0-184m) had all hexes in Band 0

**New (relative)**:
- Base = map's minimum elevation (no tint)
- 100m intervals from base
- Up to 6 bands maximum

| Band | Elevation Above Base | Tint Opacity |
|------|---------------------|--------------|
| 0 | 0-100m | 0% |
| 1 | 100-200m | 6% |
| 2 | 200-300m | 12% |
| 3 | 300-400m | 18% |
| 4 | 400-500m | 24% |
| 5 | 500-600m | 30% |
| 6 | 600m+ | 36% |

### Diagnostic Function
Added `diagnose_hex_alignment()` to trace:
1. Hex_Markers circle positions (extracted centers)
2. Hex_Grid line positions (actual grid vertices)
3. Generated overlay polygon vertices
4. SVG structure showing where groups sit in hierarchy

### Known Issues (Pending)
- Elevation overlays affect z-order of roads/buildings (they appear under overlays)
- Road/bridge connections sometimes missing in detail maps

## Rotation-Aware Overlay Alignment (v2.0.0)

### The Problem
On rotated maps, elevation overlays didn't align with the hex grid. The hex grid stays unrotated (flat-top, aligned with print frame), but terrain rotates underneath.

### The Solution: Transparency Sheet Model
Think of it as a transparency sheet rotating over fixed paper:
- **Hex grid** = fixed on the paper (unrotated)
- **Terrain** = transparency that rotates around the map center
- **Elevation values** = sampled based on what terrain "slides into" each hex after rotation

### SVG Structure
```
Master_Content
├── Rotated_Content (transform="rotate(...)")
│   └── Terrain, contours, roads, MGRS grid
├── Out_Of_Play_Frame
├── Game_Elevation_Overlay  ← OUTSIDE rotation (aligns with Hex_Grid)
├── Game_Hillside_Shading   ← OUTSIDE rotation (aligns with Hex_Grid)
├── Hex_Grid                ← stays unrotated
└── Hex_Labels
```

### Rotation-Aware Elevation Sampling
For each hex on the static grid:
1. Get hex center SVG position (hx, hy)
2. Rotate that position by the OPPOSITE angle around the rotation center
3. This gives the "source" terrain position (where terrain was before rotation)
4. Sample elevation at that source position
5. Assign that elevation to the hex

```python
# Inverse rotation to find source terrain position
angle_rad = math.radians(-rotation_deg)  # Opposite direction
dx = svg_cx - rot_center_x
dy = svg_cy - rot_center_y
source_svg_x = rot_center_x + dx * cos(angle_rad) - dy * sin(angle_rad)
source_svg_y = rot_center_y + dx * sin(angle_rad) + dy * cos(angle_rad)
# Sample elevation from DEM at source position
```

### Hillside Shading (Rewritten)
Instead of using a confusing direction-to-edge mapping, hillside shading now uses pure geometry:
1. For each hex, compute all 6 vertices in SVG coordinates
2. For each neighbor in a lower elevation band, compute direction to that neighbor
3. Find which edge faces the neighbor using dot products (edge whose outward normal best aligns with direction)
4. Draw shading band on that edge

This avoids coordinate system confusion from the SVG Y-axis flip.

## Multi-Map Generation (v2.1.0)

### Overview
Implemented support for generating multiple adjacent map sheets that share hex edges for seamless tabletop play.

### Layout Options
| Layout | Sheets | Sheet Positions |
|--------|--------|-----------------|
| Single | 1 | Standard single map |
| 2-wide (short_edge) | 2 | A (west), B (east) |
| 2-tall (long_edge) | 2 | A (north), B (south) |
| 4-grid | 4 | A (NW), B (NE), C (SW), D (SE) |

### Implementation Details

**Sheet Center Calculation:**
```python
# Cluster center is the center of ALL sheets combined
# Individual sheet centers are calculated with offsets

# For 2-wide: offset = map_width - overlap_in_meters
x_offset = width_m - (overlap_hexes * col_spacing)

# Apply rotation to offset vectors for rotated maps
if rotation_deg != 0:
    rotated_dx = dx * cos(angle) - dy * sin(angle)
    rotated_dy = dx * sin(angle) + dy * cos(angle)
```

**Key insight:** The user-specified center point is the cluster center (center of all sheets), not the center of sheet A. Sheet centers are calculated as offsets from this cluster center.

### Output Structure
```
output/{country}/{name}/{timestamp}_{version}/
├── map_config.json              # Original config
├── cluster_metadata.json        # Layout, sheet positions, total coverage
├── {name}_A_tactical.svg
├── {name}_A_hexdata.json
├── {name}_B_tactical.svg
└── ...
```

### Frontend Preview Enhancement
- Multiple boundary polygons drawn for each sheet
- Sheet labels (A, B, C, D) displayed as circular markers at sheet centers
- Orientation arrow hidden for multi-map layouts
- Coverage statistics show total area (e.g., "20.4 x 6.5 km (2 maps)")

### Consistent Elevation Banding Across Sheets (v2.1.1)

**Problem:** When generating multi-map clusters, each sheet calculated its own minimum elevation as "level 0" for elevation banding. This caused visual discontinuity at sheet boundaries - the same absolute elevation could appear with different tinting on adjacent sheets.

**Solution:** Pre-scan all sheets before generating any of them:
1. `scan_sheet_elevation()` - lightweight function to sample DEM for each sheet
2. Find global minimum elevation across all sheets
3. Pass `cluster_min_elevation` to `generate_single_sheet()`
4. All sheets use the same base elevation for band calculation

**Result:** Elevation bands are now consistent across all sheets in a multi-map cluster. The global base elevation is stored in `cluster_metadata.json` as `cluster_base_elevation_m`.

## Future Improvements

- [ ] Add river rendering with proper styling
- [ ] Add city/town labels
- [ ] Add airfield and port markers
- [x] Implement elevation bands for hex data
- [ ] Add legend and scale bar
- [ ] Support for different hex sizes
- [ ] Command-line interface for map generation
- [x] Batch generation of adjacent map sheets (v2.1.0)
- [ ] Fix overlay z-order so roads/buildings render above elevation tinting
- [ ] Fix missing road/bridge connections in detail maps
