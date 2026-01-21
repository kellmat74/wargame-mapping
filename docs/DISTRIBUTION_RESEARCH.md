# Distribution Research: Packaging for Mac & Windows

**Status:** Research complete, implementation deferred
**Date:** 2026-01-20
**Version:** v3.0.0

---

## Summary

Research spike for packaging the Wargame Map Generator for distribution to users who don't have Python development environments. Goal: work on both Mac and Windows with minimal technical knowledge.

**Recommendation:** Start with **Option 1 (Enhanced Source Distribution)** for active development, upgrade to **Option 2 (PyInstaller)** later for wider distribution.

---

## Current Application Structure

- **Source code:** ~560KB (14 Python files, 3 HTML files)
- **Dependencies:** 14 packages in requirements.txt, including native extensions (geopandas, rasterio, shapely, pyproj, fiona)
- **Data:** Downloaded on-demand (~50MB-13GB depending on regions)
- **No hardcoded paths** - uses `pathlib.Path` throughout
- **Cross-platform launchers:** .command (Mac) and .bat (Windows)

---

## Option 1: Enhanced Source Distribution (Recommended for Now)

### What It Is
Ship source code + requirements.txt. User installs Python, runs setup script, then runs launcher.

### Package Contents (~2MB + sample data)
```
Wargame-Map-Generator/
├── setup.sh                    # Mac/Linux setup
├── setup.bat                   # Windows setup
├── Start Map Generator.command # Mac launcher
├── Start Map Generator.bat     # Windows launcher
├── requirements.txt
├── *.py (14 files)
├── *.html (3 files)
├── map_defaults.json
├── README.md
├── INSTALL_MAC.md
├── INSTALL_WINDOWS.md
├── check_dependencies.py       # NEW: Verify installation
└── sample_data/                # NEW: Small test region
    └── monaco/                 # Tiny country (~10MB)
```

### User Experience
1. Install Python 3.9+ (one-time)
2. Download and unzip
3. Run setup script (creates venv, installs dependencies)
4. Double-click launcher to start

### Pros
- Small download (~2MB + sample data)
- Easy to update (just replace files)
- Users can inspect/modify code
- Works with user's existing Python
- No build infrastructure needed

### Cons
- User must install Python 3.9+ first
- Setup can fail if pip has issues
- Native dependency compilation can be slow
- More technical knowledge needed

### Development Workflow
```
You: Edit code → Test → git commit → git push → Create GitHub release
User: See announcement → Download new zip → Replace files → Run setup.sh (if deps changed)
```

### Tasks to Implement

1. **Documentation:**
   - [ ] Rewrite README.md with clear install instructions
   - [ ] Create INSTALL_MAC.md with Mac-specific guide
   - [ ] Create INSTALL_WINDOWS.md with Windows-specific guide
   - [ ] Add troubleshooting section for common pip issues

2. **Dependency Checker:**
   - [ ] Create `check_dependencies.py` script
   - [ ] Verify Python version (3.9+)
   - [ ] Verify all packages installed
   - [ ] Check for osmium-tool (optional)
   - [ ] Provide clear error messages

3. **Sample Data:**
   - [ ] Download Monaco PBF file (~1MB)
   - [ ] Process into sample_data/monaco/ directory
   - [ ] Document how to use sample data for testing
   - [ ] Consider: bundle pre-processed data or just PBF?

4. **Testing:**
   - [ ] Test fresh install on Mac (without existing Python/venv)
   - [ ] Test fresh install on Windows
   - [ ] Verify sample data generates map correctly
   - [ ] Document platform-specific quirks

---

## Option 2: PyInstaller Standalone Executable (Future)

### What It Is
Bundle Python interpreter + all dependencies into single .app (Mac) or .exe (Windows).

### Package Contents
- Mac: `Wargame Map Generator.app` (~350-400MB)
- Windows: `Wargame Map Generator.exe` (~350-400MB)
- Plus sample data (~50MB)

### User Experience
1. Download and unzip
2. Double-click to run
3. (Mac: may need to right-click → Open first due to Gatekeeper)

### Pros
- No Python installation required
- Single click to run
- Professional feel
- No dependency issues for users

### Cons
- Large download (~400MB per platform)
- Must build separately for Mac AND Windows
- Mac: arm64 vs x86_64 builds needed (or universal binary)
- Harder to debug issues
- Updates require full re-download
- PyInstaller config can be tricky with native deps

### Development Workflow
```
You: Edit code → Test → Build Mac .app → Build Windows .exe → Upload to GitHub
User: See update notification → Download new .app/.exe → Replace old one
```

### Build Options
1. **Manual:** Build on Mac, build on Windows VM
2. **GitHub Actions:** Auto-build on release tag (recommended)
   - Mac runner builds .app
   - Windows runner builds .exe
   - Artifacts uploaded to release automatically

### Tasks to Implement (When Ready)

1. **PyInstaller Setup:**
   - [ ] Create `wargame_map.spec` PyInstaller spec file
   - [ ] Handle native dependencies (geopandas, rasterio, etc.)
   - [ ] Bundle HTML files and assets
   - [ ] Test on Mac (both arm64 and x86_64)
   - [ ] Test on Windows

2. **GitHub Actions CI:**
   - [ ] Create `.github/workflows/build.yml`
   - [ ] Mac build job
   - [ ] Windows build job
   - [ ] Upload artifacts to releases

3. **Optional Enhancements:**
   - [ ] Add version check on startup
   - [ ] Notify user when update available
   - [ ] Code signing (Mac notarization, Windows signing)

---

## Other Options Considered

### Option 3: Conda/Miniconda Package
- Good for geospatial libraries
- User must install Miniconda (~70MB)
- Less familiar to non-developers
- **Decision:** Skip unless Option 1 has persistent pip issues

### Option 4: Docker Container
- Identical environment everywhere
- Requires Docker Desktop (~2GB)
- Overkill for desktop app
- **Decision:** Consider for server/cloud deployment later

### Option 5: Electron/Web App
- Modern cross-platform approach
- Major rewrite required
- Could become cloud service
- **Decision:** Future consideration only

---

## Technical Notes

### Native Dependencies
These packages require platform-specific compilation:
- geopandas (uses GDAL/GEOS)
- shapely (GEOS bindings)
- pyproj (PROJ library)
- rasterio (GDAL bindings)
- fiona (GDAL bindings)
- numpy (compiled)
- Pillow (compiled)

All have pre-built wheels for Python 3.9+ on Mac/Windows via pip.

### Optional External Tool
- **osmium-tool:** Used for fast OSM data extraction
- Mac: `brew install osmium-tool`
- Windows: Manual download or build
- Falls back to Overpass API if not available

### Path Handling
All code uses `pathlib.Path` with relative paths from `__file__`. No hardcoded absolute paths found.

### Platform-Specific Code
- Minimal platform-specific code
- Browser opening via `webbrowser` module (cross-platform)
- Subprocess calls use `sys.executable` (portable)
- Port checking in .command file uses `lsof` (Mac-only, but not critical)

---

## Sample Data Options

| Region | PBF Size | Processed Size | Notes |
|--------|----------|----------------|-------|
| Monaco | ~1MB | ~10MB | Tiny, good for quick tests |
| Rhode Island | ~20MB | ~50MB | Small US state |
| Delaware | ~15MB | ~40MB | Small US state |

**Recommendation:** Monaco for minimal download, Rhode Island for US relevance.

---

## Comparison: Update Workflow

| Aspect | Option 1 (Source) | Option 2 (PyInstaller) |
|--------|-------------------|------------------------|
| Your release effort | Push code, create release | Build on 2 platforms, upload |
| Update size for users | ~1-2MB (changed files) | ~400MB (full app) |
| User effort to update | Download zip, replace files | Download app, replace |
| Auto-update possible? | No (but could add check) | Could add notification |
| Dependency changes | User re-runs setup.sh | Bundled, no user action |
| Requires build infra? | No | Yes (Mac + Windows) |

---

## Decision Criteria: When to Use Each Option

**Use Option 1 when:**
- Still actively developing
- Target audience is somewhat technical
- Want fast iteration on fixes
- Don't have Windows build environment

**Upgrade to Option 2 when:**
- App is stable, infrequent changes
- Want to reach non-technical users
- Have CI/CD set up for automated builds
- Willing to maintain larger release artifacts
