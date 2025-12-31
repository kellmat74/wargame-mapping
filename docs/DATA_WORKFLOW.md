# Data Download Workflow

## Current Workflow (Manual Steps Required)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CURRENT WORKFLOW                                 │
└─────────────────────────────────────────────────────────────────────┘

Step 1: Download PBF file manually
┌──────────────────────────────────────────────────────────────────────┐
│ curl -o cache/ukraine-latest.osm.pbf \                               │
│   https://download.geofabrik.de/europe/ukraine-latest.osm.pbf        │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
Step 2: Edit map_server.py - Add bounding box
┌──────────────────────────────────────────────────────────────────────┐
│ GEOFABRIK_REGIONS = {                                                │
│     "philippines": (116.0, 4.5, 127.0, 21.5),                        │
│     "taiwan": (119.0, 21.5, 122.5, 26.0),                            │
│     ...                                                              │
│     "ukraine": (22.0, 44.3, 40.5, 52.4),  ◄── ADD THIS               │
│ }                                                                    │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
Step 3: Edit tactical_map.py - Add country name mapping
┌──────────────────────────────────────────────────────────────────────┐
│ GEOFABRIK_TO_COUNTRY = {                                             │
│     "japan": "Japan",                                                │
│     "taiwan": "Taiwan",                                              │
│     ...                                                              │
│     "ukraine": "Ukraine",  ◄── ADD THIS                              │
│ }                                                                    │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
Step 4: Restart server
┌──────────────────────────────────────────────────────────────────────┐
│ Kill and restart map_server.py                                       │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
Step 5: NOW the config tool works for that region
```

### Problems with Current Workflow

1. **Manual code changes** - Every new country requires editing 2 Python files
2. **Hardcoded bounding boxes** - Must look up coordinates for each country
3. **Server restart required** - Changes don't take effect until restart
4. **No auto-discovery** - System can't see what PBF files are available
5. **Error-prone** - Easy to forget a step or make typos

---

## Proposed Workflow (Auto-Discovery)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PROPOSED WORKFLOW                                │
└─────────────────────────────────────────────────────────────────────┘

Step 1: Download PBF file (same as before)
┌──────────────────────────────────────────────────────────────────────┐
│ curl -o cache/ukraine-latest.osm.pbf \                               │
│   https://download.geofabrik.de/europe/ukraine-latest.osm.pbf        │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
Step 2: DONE - System auto-discovers the file
┌──────────────────────────────────────────────────────────────────────┐
│ Server scans cache/*.osm.pbf on startup                              │
│   → Extracts bounding box from each PBF (osmium fileinfo)            │
│   → Derives country name from filename                               │
│   → No code changes needed!                                          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### 1. Create `cache/regions.json` metadata file

Auto-generated on server startup by scanning PBF files:

```json
{
  "regions": {
    "ukraine": {
      "pbf_file": "ukraine-latest.osm.pbf",
      "bounds": [22.0, 44.3, 40.5, 52.4],
      "display_name": "Ukraine",
      "last_updated": "2025-12-30"
    },
    "taiwan": {
      "pbf_file": "taiwan-latest.osm.pbf",
      "bounds": [119.0, 21.5, 122.5, 26.0],
      "display_name": "Taiwan",
      "last_updated": "2025-12-28"
    }
  }
}
```

### 2. Add startup scan function

```python
def scan_available_regions():
    """Scan cache directory for PBF files and extract metadata."""
    regions = {}
    for pbf in Path("cache").glob("*-latest.osm.pbf"):
        # Extract region name from filename: "ukraine-latest.osm.pbf" -> "ukraine"
        region_name = pbf.stem.replace("-latest.osm", "")

        # Get bounding box using osmium
        bounds = get_pbf_bounds(pbf)  # Uses: osmium fileinfo -e

        # Derive display name
        display_name = region_name.replace("-", " ").title()

        regions[region_name] = {
            "pbf_file": pbf.name,
            "bounds": bounds,
            "display_name": display_name
        }
    return regions
```

### 3. Get bounds from PBF using osmium

```bash
$ osmium fileinfo -e cache/ukraine-latest.osm.pbf | grep "Bounding"
  Bounding boxes:
    (22.137059,44.184639,40.2205439,52.3791473)
```

### 4. Remove hardcoded dictionaries

- Delete `GEOFABRIK_REGIONS` from map_server.py
- Delete `GEOFABRIK_TO_COUNTRY` from tactical_map.py
- Replace with dynamic lookup from `regions.json`

---

## Alternative: Even Simpler Approach

Instead of scanning on startup, add a **one-time registration command**:

```bash
# After downloading, run:
python3 register_region.py cache/ukraine-latest.osm.pbf

# Output:
# Registered region 'ukraine':
#   Bounds: (22.0, 44.3, 40.5, 52.4)
#   Display name: Ukraine
#   Added to cache/regions.json
```

This is simpler than auto-scanning and gives user feedback.

---

## Config Tool Enhancement

Add a "Available Regions" indicator in the web UI:

```
┌─────────────────────────────────────────────────────┐
│  Available Data Regions:                            │
│  ✓ Japan      ✓ Taiwan     ✓ Philippines           │
│  ✓ Ukraine    ○ Poland     ○ Germany               │
│                                                     │
│  [Download New Region...]                           │
└─────────────────────────────────────────────────────┘
```

This would show which regions have cached PBF files and allow downloading new ones directly from the UI.

---

## Summary

| Aspect | Current | Proposed |
|--------|---------|----------|
| Steps to add region | 4 (download, edit 2 files, restart) | 1 (download only) |
| Code changes needed | Yes, every time | No |
| Error potential | High | Low |
| Server restart | Required | Optional (auto-refresh) |
| Discoverability | None | Shows available regions |
