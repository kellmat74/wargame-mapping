# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for Wargame Map Generator
#
# Build:
#   Mac:     pyinstaller wargame_map.spec
#   Windows: pyinstaller wargame_map.spec   (run on Windows)
#
# Output:
#   dist/Wargame Map Generator.app  (Mac)
#   dist/Wargame Map Generator/     (Windows one-dir)

import sys
from pathlib import Path

ROOT = Path(SPECPATH)

# ---------------------------------------------------------------------------
# All Python source files that need to be importable at runtime
# (PyInstaller finds most via analysis, but explicit is safer for scripts
# that are called as --run subprocesses)
# ---------------------------------------------------------------------------
SOURCE_PY = [str(ROOT / f) for f in [
    'main.py',
    'map_server.py',
    'tactical_map.py',
    'game_map_converter.py',
    'download_mgrs_data_osmium.py',
    'download_mgrs_data.py',
    'region_registry.py',
    'map_utils.py',
    'render_helpers.py',
    'hexgrid.py',
    'fetch_data.py',
    'fetch_elevation.py',
    'fetch_landcover.py',
    'download_region_data.py',
    'convert_to_geopackage.py',
]]

a = Analysis(
    ['main.py'],
    pathex=[str(ROOT)],
    binaries=[
        # Bundled osmium binaries — include both platforms; only the matching
        # one will be used at runtime via get_osmium_path().
        (str(ROOT / 'assets' / 'bin' / 'osmium-darwin'), 'assets/bin'),
        # Windows binary added here when available:
        # (str(ROOT / 'assets' / 'bin' / 'osmium-win.exe'), 'assets/bin'),
    ],
    datas=[
        # HTML/JS UI files
        (str(ROOT / 'map_config.html'),          '.'),
        (str(ROOT / 'game_map_config.html'),     '.'),
        # Config
        (str(ROOT / 'map_defaults.json'),        '.'),
        # App icon
        (str(ROOT / 'assets'),                   'assets'),
    ],
    hiddenimports=[
        # Flask/Werkzeug internals not always auto-detected
        'flask',
        'werkzeug',
        'werkzeug.serving',
        'werkzeug.debug',
        # GIS stack
        'geopandas',
        'fiona',
        'fiona.ogrext',
        'shapely',
        'pyproj',
        'rasterio',
        'rasterio.crs',
        'rasterio._shim',
        # Rendering
        'svgwrite',
        'cairosvg',
        'skimage',
        'sklearn',
        # MGRS
        'mgrs',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib.tests',
        'numpy.tests',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# ---------------------------------------------------------------------------
# Mac .app bundle
# ---------------------------------------------------------------------------
if sys.platform == 'darwin':
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='Wargame Map Generator',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,  # no terminal window
        icon=str(ROOT / 'assets' / 'icon.icns') if (ROOT / 'assets' / 'icon.icns').exists() else None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        name='Wargame Map Generator',
    )
    app = BUNDLE(
        coll,
        name='Wargame Map Generator.app',
        icon=str(ROOT / 'assets' / 'icon.icns') if (ROOT / 'assets' / 'icon.icns').exists() else None,
        bundle_identifier='com.wargametools.mapgenerator',
        info_plist={
            'NSHighResolutionCapable': True,
            'NSRequiresAquaSystemAppearance': False,
            'CFBundleShortVersionString': '3.0.1',
        },
    )

# ---------------------------------------------------------------------------
# Windows .exe (one-dir for easier distribution)
# ---------------------------------------------------------------------------
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='Wargame Map Generator',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        icon=str(ROOT / 'assets' / 'icon.ico') if (ROOT / 'assets' / 'icon.ico').exists() else None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        name='Wargame Map Generator',
    )
