# Taiwan Hex Wargame Map Generator

## Project Overview

Generate a hex-based wargame map of Taiwan from QGIS geographic data. This is a **prototype project** to establish the workflow — specifications will change for future maps.

## Map Parameters

- **Region**: Taiwan (main island)
- **Bounding Box** (approximate, WGS84):
  - Southwest: 120.0°E, 21.8°N
  - Northeast: 122.1°E, 25.4°N
- **Hex Size**: 10 km (center to center, flat-top orientation)
- **Projected CRS**: EPSG:3826 (TWD97 / TM2 Taiwan) — use this for all distance calculations
- **Approximate Output**: ~55 hexes tall, ~20 hexes wide

## Hex Grid Conventions

- **Orientation**: Flat-top hexagons
- **Coordinate System**: Axial coordinates (q, r)
  - q increases to the east
  - r increases to the south-southeast
- **Hex Geometry**: 
  - Width (flat edge to flat edge): 10 km
  - Height (point to point): ~11.55 km
- **Origin**: Place (0, 0) at the northwest corner of the bounding box

## Terrain Classification

Assign each hex ONE primary terrain type based on dominant coverage:

| Terrain    | Description                          | QGIS Source Layer      |
|------------|--------------------------------------|------------------------|
| `water`    | Ocean, sea (hex is >50% water)       | Coastline polygon      |
| `urban`    | Major cities, dense development      | Land cover / urban     |
| `paddy`    | Rice paddies, irrigated agriculture  | Land cover / cropland  |
| `forest`   | Forested areas                       | Land cover / forest    |
| `rough`    | Scrubland, mixed terrain             | Land cover / other     |
| `mountain` | Elevation > 1000m                    | DEM raster             |
| `clear`    | Open lowland, light agriculture      | Default / fallback     |

**Priority Order** (when hex has mixed terrain):
1. water (if >50% of hex)
2. urban
3. mountain (by elevation)
4. forest
5. paddy
6. rough
7. clear

## Elevation Bands

Derived from DEM, used for movement/combat modifiers:

| Level | Elevation Range | Label        |
|-------|-----------------|--------------|
| 0     | 0–100 m         | Lowland      |
| 1     | 100–500 m       | Hills        |
| 2     | 500–1000 m      | Highlands    |
| 3     | 1000–2000 m     | Mountains    |
| 4     | 2000+ m         | High Mountains |

Taiwan's highest point (Yushan) is ~3,952 m.

## Linear Features (Hex-Side Attributes)

These features are marked on hex edges, not hex centers:

### Rivers
- Major rivers (width > 50m in source data) mark hex edges they cross
- Store as: `{hex: [q, r], edges: [0, 1, 2, 3, 4, 5]}` where edges are numbered clockwise from top

### Coastline
- Mark which hex edges border water
- Important for amphibious operations

## Point/Network Features (Hex-Center Attributes)

### Roads
- Classify as: `highway`, `major_road`, `minor_road`
- Store as connections between adjacent hexes: `{from: [q1, r1], to: [q2, r2], type: "highway"}`

### Cities/Towns
- Mark hexes containing significant population centers
- Include name if available: `{hex: [q, r], name: "Taipei", size: "major"}`

### Ports
- Mark coastal hexes with port facilities
- Relevant for naval operations

### Airfields
- Mark hexes containing airports/airfields

## Data Files (Expected in `data/raw/`)

After QGIS export, place files here:

```
data/raw/
├── taiwan_coastline.geojson    # Taiwan outline polygon
├── taiwan_landcover.geojson    # Land cover polygons with 'type' attribute
├── taiwan_elevation.tif        # DEM raster (GeoTIFF)
├── taiwan_rivers.geojson       # River linestrings with 'width' or 'class'
├── taiwan_roads.geojson        # Road network with 'highway' classification
├── taiwan_places.geojson       # Cities/towns with 'name' and 'population'
└── taiwan_infrastructure.geojson  # Ports, airfields, etc.
```

## Output Files (Generated in `output/`)

```
output/
├── taiwan_hexmap.svg           # Visual map with terrain colors
├── taiwan_hexmap.png           # Raster export of SVG
├── taiwan_hexdata.json         # Machine-readable hex data
└── taiwan_connections.json     # Road/river network graph
```

### Hex Data JSON Structure

```json
{
  "metadata": {
    "region": "Taiwan",
    "hex_size_km": 10,
    "crs": "EPSG:3826",
    "generated": "2025-01-15"
  },
  "hexes": [
    {
      "coord": [0, 5],
      "terrain": "water",
      "elevation": 0,
      "coastal_edges": [],
      "river_edges": [],
      "city": null,
      "features": []
    },
    {
      "coord": [3, 8],
      "terrain": "urban",
      "elevation": 0,
      "coastal_edges": [4, 5],
      "river_edges": [2, 3],
      "city": {"name": "Taipei", "size": "major"},
      "features": ["port", "airfield"]
    }
  ]
}
```

## Visual Style (SVG Output)

### Terrain Colors
```
water:    #4a90d9 (blue)
urban:    #8b4513 (brown)
paddy:    #98fb98 (pale green)
forest:   #228b22 (forest green)
rough:    #d2b48c (tan)
mountain: #a0522d (sienna)
clear:    #f5f5dc (beige)
```

### Overlays
- Hex grid: thin black lines (#333333, 0.5px)
- Hex coordinates: small gray text at hex center (optional, toggle-able)
- Rivers on edges: blue lines (#0066cc, 2px)
- Roads: red/orange lines connecting hex centers
- City markers: black dots with labels

## Development Tasks (Suggested Order)

### Phase 1: Core Hex Grid
1. Create hex grid generator with axial coordinates
2. Implement coordinate ↔ geographic transforms (using EPSG:3826)
3. Generate empty hex grid covering Taiwan bounding box
4. Export basic SVG with just hex outlines

### Phase 2: Terrain Classification
5. Load coastline, clip hexes to land/water
6. Load DEM, calculate average elevation per hex
7. Load land cover, calculate dominant terrain per hex
8. Combine into final terrain assignment

### Phase 3: Linear Features
9. Process rivers, detect hex-edge crossings
10. Process coastline edges
11. Process road network as hex-to-hex connections

### Phase 4: Point Features
12. Load cities/towns, assign to hexes
13. Load infrastructure (ports, airfields)

### Phase 5: Rendering
14. Create full SVG renderer with all layers
15. Export JSON data files
16. Add legend and scale bar

## Notes for AI Assistant

- **Start simple**: Get the hex grid math working before loading real data
- **Test with small areas first**: Use a 3x3 hex subset before full Taiwan
- **Validate coordinates**: Taiwan should be roughly 55 hexes N-S at 10km scale
- **Handle partial hexes**: Coastal hexes will be partially water — that's expected
- **Prefer clarity over cleverness**: This code will be modified for other regions later

## QGIS Data Acquisition Tips

For the human operator — suggested data sources:

1. **Coastline/Boundaries**: Natural Earth, GADM, or OpenStreetMap
2. **Land Cover**: ESA WorldCover, Copernicus Global Land Cover
3. **Elevation (DEM)**: SRTM 30m, ASTER GDEM, or Taiwan government open data
4. **Roads**: OpenStreetMap export
5. **Rivers**: OpenStreetMap or HydroSHEDS
6. **Cities**: Natural Earth populated places or OSM

Export all vector data as GeoJSON in EPSG:4326 (WGS84) — the code will reproject.
Export raster as GeoTIFF with embedded CRS.
