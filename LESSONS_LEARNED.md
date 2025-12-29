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

## Future Improvements

- [ ] Add river rendering with proper styling
- [ ] Add city/town labels
- [ ] Add airfield and port markers
- [ ] Implement elevation bands for hex data
- [ ] Add legend and scale bar
- [ ] Support for different hex sizes
- [ ] Command-line interface for map generation
- [ ] Batch generation of adjacent map sheets
