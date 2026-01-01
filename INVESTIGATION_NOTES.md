# Road Rendering Investigation - Xitun District vs Taipei

## Problem
Xitun District map is missing many roads compared to Taipei map.

## RESOLVED - 2026-01-01

### Root Cause
The roads.geojson file for 51R/TG was extracted using the old Overpass API method (download_mgrs_data.py) which had incomplete data. The file had roads at the edges of the MGRS square but a large gap in the center where the map was located:

- Lon 120.50-120.55: 1,337 roads
- **Lon 120.55-120.70: 0 roads** ← MAP CENTER AREA
- Lon 120.75-120.80: 378 roads

This was NOT a code bug - the filtering logic in tactical_map.py was correct. The source data file simply didn't contain roads for the map area.

### Evidence
- Buildings.geojson had 25,691 features in the gap area (correctly extracted)
- Landcover.geojson had 2,204 features in the gap area (correctly extracted)
- Roads.geojson had only 5 features in the gap area (extraction problem)
- Direct extraction from taiwan-latest.osm.pbf yielded 38,860 roads for the gap area

### Solution
Re-ran the osmium-based extraction script:
```bash
python3 download_mgrs_data_osmium.py --region taiwan "51R TG" --force
```

### Results
- Before: 33,355 total roads, 12 in map bounds
- After: 118,616 total roads, 22,251 in map bounds

Road distribution by longitude after fix:
- Lon 120.55-120.60: 4,059 roads
- Lon 120.60-120.65: 10,424 roads
- Lon 120.65-120.70: 20,810 roads

### Lesson Learned
Always use `download_mgrs_data_osmium.py` for data extraction instead of the old Overpass-based script. The osmium method is more reliable and extracts complete data.
