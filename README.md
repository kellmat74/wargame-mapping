# Tactical Map Generator

A tool for generating hex-based tactical wargame maps from OpenStreetMap and elevation data. Produces print-ready SVG files suitable for use in Affinity Designer or other vector graphics software.

## Features

- **Hex Grid Generation**: Creates flat-top hexagonal grids with configurable size (default 250m hex-to-hex)
- **Terrain Classification**: Automatically classifies terrain from OSM data (water, urban, forest, farmland, orchard, etc.)
- **Road Network**: Renders highways, major roads, minor roads, and residential streets
- **Buildings**: Individual building footprints with outlines
- **Elevation Contours**: Topographic contour lines from SRTM elevation data
- **Streams and Rivers**: Water features with configurable widths
- **Reference Tiles**: Downloads OpenTopoMap tiles for artist reference
- **Print-Ready Output**: SVG with proper bleed margins (0.125") for professional printing
- **MGRS Support**: Organizes data by Military Grid Reference System squares

## Output

The generator produces:
- `{name}.svg` - Vector map file for editing in Affinity Designer, Illustrator, etc.
- `{name}_hexdata.json` - Machine-readable hex terrain data
- `reference_tiles/` - OpenTopoMap reference images for the map area

## Quick Start

### 1. Setup

**Mac/Linux:**
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

**Windows:**
```cmd
setup.bat
venv\Scripts\activate.bat
```

### 2. Configure Your Map

Open `map_config.html` in your web browser to configure:
- **Map Name**: Used for output filenames
- **Center Location**: Paste coordinates in any format:
  - Decimal: `24.1477, 120.6736` (Google Maps format)
  - DMS: `24°08'51"N 120°40'25"E`
  - MGRS: `51PTT1234567890` or `51P TT 12345 67890`
- **Rotation**: Rotate the map (0° = North is up)

The tool auto-detects which MGRS data regions are needed.

### 3. Download Geographic Data

Download data for your region (shown in the config tool):
```bash
python download_mgrs_data.py "51P TT"
```

For maps spanning multiple MGRS squares:
```bash
python download_mgrs_data.py "51P TT"
python download_mgrs_data.py "51P TS"
```

### 4. Generate the Map

Save `map_config.json` from the config tool, then run:
```bash
python tactical_map.py
```

### 5. Find Your Output

Generated files are in: `output/{region}/{name}/`

## Map Specifications

| Parameter | Value |
|-----------|-------|
| Hex Size | 250m (center to center) |
| Grid Size | 47 columns × 26 rows |
| Playable Area | ~17km × 6.5km |
| Out-of-Play Border | 2 hexes on all sides |
| Print Bleed | 0.125" (3.175mm) |

## Terrain Types

| Terrain | Color | Description |
|---------|-------|-------------|
| Water | Light blue | Ocean, lakes, rivers |
| Urban | Medium grey | Cities, towns, developed areas |
| Forest | Light green | Forested areas |
| Farmland | Pale yellow | Agricultural land |
| Orchard | Light green | Orchards, vineyards |
| Scrub | Tan | Scrubland, bushes |
| Wetland | Blue-green | Marshes, swamps |
| Beach | Sand | Coastal beaches |
| Clear | Beige | Default/open terrain |

## Graphic Style

The output is styled to match OpenTopoMap aesthetics:
- **Water**: Dark blue 1m outline
- **Urban**: Dark grey 1m outline
- **Farmland**: Brown 1m outline
- **Roads**: All 10m width, dark grey with black 2m outline
- **Buildings**: Dark grey fill with black 1m outline
- **Streams**: 10m width, blue

## File Structure

```
Wargame Mapping/
├── map_config.html      # Web-based configuration tool
├── map_config.json      # Generated config file
├── tactical_map.py      # Main map generator
├── download_mgrs_data.py # Data downloader
├── setup.sh             # Mac/Linux setup script
├── setup.bat            # Windows setup script
├── data/                # Downloaded geographic data (per MGRS square)
│   └── {zone}/{square}/ # e.g., 51P/TT/
├── output/              # Generated maps
│   └── {region}/{name}/ # e.g., 51P_TT/san_carlos/
└── venv/                # Python virtual environment
```

## Dependencies

- Python 3.9+
- numpy
- pandas
- geopandas
- rasterio
- shapely
- pyproj
- svgwrite
- mgrs
- requests
- pillow

All dependencies are installed automatically by the setup scripts.

## Data Sources

- **OpenStreetMap**: Roads, buildings, land use, water features via Overpass API
- **SRTM**: Elevation data (30m resolution) via OpenTopography
- **OpenTopoMap**: Reference tile imagery

## Known Limitations

- SVG pattern fills (for terrain textures) don't render in Affinity Designer - solid fills are used instead
- OpenTopoMap tiles max out at zoom level 15 for some regions
- Large areas may require downloading multiple MGRS squares

## License

This project is for personal/educational use in creating wargame maps.

## Acknowledgments

- OpenStreetMap contributors for geographic data
- OpenTopoMap for reference imagery
- SRTM/NASA for elevation data
