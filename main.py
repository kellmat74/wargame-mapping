"""
Wargame Map Generator — desktop launcher.

Three layers of server lifecycle management:
  1. Single-instance: if port 8080 already bound, open browser and exit.
  2. System tray icon (pystray) with Open / Quit — always visible while running.
  3. In-browser Shut Down button — calls /api/shutdown, shows a "you can close
     this tab" overlay (window.close() is blocked by browsers for regular tabs).

Also serves as the frozen subprocess dispatcher: when PyInstaller bundles the
app, all child-process calls route through this file via --run <module> instead
of trying to exec .py files directly.
"""

import json
import os
import shutil
import socket
import sys
import threading
import webbrowser
from pathlib import Path


# ---------------------------------------------------------------------------
# Frozen subprocess dispatch — must run BEFORE any heavy imports
# ---------------------------------------------------------------------------
if getattr(sys, 'frozen', False) and '--run' in sys.argv:
    _run_idx = sys.argv.index('--run')
    _module = sys.argv[_run_idx + 1]
    sys.argv = sys.argv[:_run_idx] + sys.argv[_run_idx + 2:]

    # Restore workspace as cwd so relative data paths work
    _prefs_file = Path.home() / '.wargame_map_prefs.json'
    if _prefs_file.exists():
        try:
            _workspace = Path(json.loads(_prefs_file.read_text()).get('workspace', ''))
            if _workspace.exists():
                os.chdir(_workspace)
        except Exception:
            pass

    if _module == 'tactical_map':
        from tactical_map import main as _m; _m(); sys.exit(0)
    elif _module == 'download_mgrs_data_osmium':
        from download_mgrs_data_osmium import main as _m; _m(); sys.exit(0)
    elif _module == 'download_mgrs_data':
        from download_mgrs_data import main as _m; _m(); sys.exit(0)
    elif _module == 'game_map_converter':
        from game_map_converter import main as _m; _m(); sys.exit(0)
    else:
        print(f"Unknown --run module: {_module}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PORT = 8080
PREFS_FILE = Path.home() / '.wargame_map_prefs.json'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def bundle_dir() -> Path:
    """Directory where the app's code and assets live (read-only when frozen)."""
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).parent


def load_prefs() -> dict:
    try:
        return json.loads(PREFS_FILE.read_text())
    except Exception:
        return {}


def save_prefs(prefs: dict) -> None:
    PREFS_FILE.write_text(json.dumps(prefs, indent=2))


def default_workspace() -> Path:
    return Path.home() / 'Documents' / 'Wargame Maps'


def resolve_workspace() -> Path:
    """Return the active workspace directory, creating it if needed."""
    prefs = load_prefs()
    workspace = Path(prefs['workspace']) if 'workspace' in prefs else None

    if workspace is None:
        workspace = default_workspace() if getattr(sys, 'frozen', False) else bundle_dir()

    workspace.mkdir(parents=True, exist_ok=True)

    if 'workspace' not in prefs:
        prefs['workspace'] = str(workspace)
        save_prefs(prefs)

    return workspace


def seed_workspace(workspace: Path) -> None:
    """Copy sample data into a fresh workspace so the demo works out of the box."""
    sample_src = bundle_dir() / 'sample_data'
    if not sample_src.exists():
        return

    src_data = sample_src / 'data'
    if src_data.exists():
        dst_data = workspace / 'data'
        for item in src_data.rglob('*'):
            rel = item.relative_to(src_data)
            dst = dst_data / rel
            if item.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
            elif not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dst)

    src_config = sample_src / 'map_config.json'
    dst_config = workspace / 'map_config.json'
    if src_config.exists() and not dst_config.exists():
        shutil.copy2(src_config, dst_config)


def is_port_open(port: int) -> bool:
    """Return True if something is already listening on localhost:port."""
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=0.5):
            return True
    except OSError:
        return False


def start_flask(workspace: Path, ready_event: threading.Event) -> None:
    """Run the Flask server in a daemon thread from the workspace directory."""
    os.chdir(workspace)
    import map_server
    ready_event.set()
    map_server.app.run(host='127.0.0.1', port=PORT, debug=False, use_reloader=False)


def make_tray_image():
    """Return a PIL Image for the system tray icon."""
    from PIL import Image, ImageDraw
    # icon_tray.png is pre-rendered with a white backing for menu bar visibility
    for name in ('icon_tray.png', 'icon.png', 'icon.icns'):
        p = bundle_dir() / 'assets' / name
        if p.exists():
            try:
                return Image.open(p).convert('RGBA').resize((64, 64), Image.LANCZOS)
            except Exception:
                pass
    # Fallback: bright white circle so it's always visible
    img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse([4, 4, 59, 59], fill=(200, 220, 255, 255))
    return img


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    # 1. Single-instance: if the server is already up, just open the browser
    if is_port_open(PORT):
        webbrowser.open(f'http://127.0.0.1:{PORT}')
        return

    workspace = resolve_workspace()
    seed_workspace(workspace)

    # 2. Start Flask in a daemon thread
    ready = threading.Event()
    server_thread = threading.Thread(
        target=start_flask, args=(workspace, ready), daemon=True
    )
    server_thread.start()

    # Wait for Flask to bind (up to 10 s) then open the browser
    ready.wait(timeout=10)
    webbrowser.open(f'http://127.0.0.1:{PORT}')

    # 3. System tray icon — blocks the main thread until Quit is chosen
    import pystray

    icon_image = make_tray_image()

    def on_open(icon, item):
        webbrowser.open(f'http://127.0.0.1:{PORT}')

    def on_quit(icon, item):
        icon.stop()
        os._exit(0)  # daemon thread (Flask) dies with the process

    tray = pystray.Icon(
        'Wargame Map Generator',
        icon_image,
        'Wargame Map Generator',
        menu=pystray.Menu(
            pystray.MenuItem('Open', on_open),
            pystray.MenuItem('Quit', on_quit),
        ),
    )
    tray.run()


if __name__ == '__main__':
    main()
