# Future Improvements & Loose Ends

Notes from Sprint 1 (Dec 30, 2025) for future cleanup and enhancement.

## Code Cleanup

### Debug Scripts to Review
Several debug scripts were created during contour/rotation troubleshooting. Consider removing or archiving:
- `debug_contour_offset.py`
- `debug_coverage_gap.py`
- `debug_dem_alignment.py`
- `debug_svg_coords.py`
- `debug_svg_rotation.py`
- `debug_svg_structure.py`
- `debug_xy_offset.py`
- `diagnose_dem.py`
- `test_contour_alignment.py`

### map_config.json in Git
`map_config.json` is modified by the web UI but probably shouldn't be tracked in git (user-specific). Consider adding to `.gitignore`.

## Feature Enhancements

### Hex Spine Style
- Current `HEX_SPINE_LENGTH = 0.18` (18% of edge) - may need tuning after visual review
- Could make spine length configurable via map_defaults.json
- Consider adding option to toggle between spine style and full polygon outlines

### Settings UI
- Settings changes require server restart to take effect for some values (module-level constants)
- Could implement hot-reload or move more settings to runtime config
- Play margins UI exists but may need testing

### Region Registry
- `regions.json` cache file is auto-generated but not in .gitignore
- Could add UI to show available regions and their coverage areas
- Consider adding progress indicator when scanning large PBF files

### Data Block
- Country name inference relies on bounds.json having source.region field
- If user manually downloads/places data, country may not be detected
- Could add manual country override in config

## Known Limitations

### Coordinate Systems
- UTM zone is calculated from map center - maps spanning multiple zones may have edge distortion
- Very large maps (>6° longitude span) would need different handling

### PBF File Detection
- Only detects `*-latest.osm.pbf` naming pattern
- Custom-named PBF files won't be auto-discovered

### Web UI
- No way to cancel a running generation (would need to stop server)
- Progress output doesn't persist after page refresh
- Settings panel could remember collapsed/expanded state

## Performance Considerations

### Large Maps
- Spine rendering iterates all vertices - could be optimized by pre-computing shared vertices once
- For very large grids, consider SVG path optimization (single path with multiple M/L commands vs. many line elements)

### Region Registry Scanning
- `osmium fileinfo` is called for each PBF file on first scan
- Could be slow with many large PBF files
- Cache is invalidated if any PBF file changes - could be smarter about incremental updates

## Documentation

### Incomplete/Missing
- `docs/DATA_WORKFLOW.md` created but could use expansion
- No user-facing README updates for new features (settings UI, auto-discovery)
- API endpoints not documented

## Testing

### No Automated Tests For
- region_registry.py
- Settings API endpoints
- Null coordinate handling in web UI
- Spine rendering output

---

*Last updated: Dec 30, 2025 - End of Sprint 1*
