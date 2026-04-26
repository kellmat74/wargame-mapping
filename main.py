"""
Wargame Map Generator — native window entry point.

Starts the Flask server in a background thread and displays it inside a
PySide6 QWebEngineView so users get a double-click app with no terminal.

Also serves as the frozen subprocess dispatcher: when PyInstaller bundles
the app, all child-process calls route through this file via --run <module>
instead of trying to exec .py files directly.
"""

import json
import os
import shutil
import sys
import threading
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
# Normal (windowed) startup
# ---------------------------------------------------------------------------
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtCore import QUrl, QTimer
from PySide6.QtGui import QIcon


PORT = 8080
PREFS_FILE = Path.home() / '.wargame_map_prefs.json'


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
    """Default workspace location for a packaged install."""
    return Path.home() / 'Documents' / 'Wargame Maps'


def resolve_workspace() -> Path:
    """Return the active workspace directory, creating it if needed."""
    prefs = load_prefs()
    workspace = Path(prefs['workspace']) if 'workspace' in prefs else None

    if workspace is None:
        if getattr(sys, 'frozen', False):
            workspace = default_workspace()
        else:
            workspace = bundle_dir()  # dev mode: use project root

    workspace.mkdir(parents=True, exist_ok=True)

    # Persist if this is the first time we're setting it
    if 'workspace' not in prefs:
        prefs['workspace'] = str(workspace)
        save_prefs(prefs)

    return workspace


def seed_workspace(workspace: Path) -> None:
    """Copy sample data into a fresh workspace so the demo works out of the box."""
    sample_src = bundle_dir() / 'sample_data'
    if not sample_src.exists():
        return

    # Copy data/ tree
    src_data = sample_src / 'data'
    if src_data.exists():
        dst_data = workspace / 'data'
        for item in src_data.rglob('*'):
            rel = item.relative_to(src_data)
            dst = dst_data / rel
            if item.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
            elif not dst.exists():  # never overwrite user data
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dst)

    # Drop in a starter map_config.json only if one doesn't exist
    src_config = sample_src / 'map_config.json'
    dst_config = workspace / 'map_config.json'
    if src_config.exists() and not dst_config.exists():
        shutil.copy2(src_config, dst_config)


def start_flask(workspace: Path, ready_event: threading.Event) -> None:
    """Run the Flask server in a daemon thread from the workspace directory."""
    os.chdir(workspace)
    import map_server
    ready_event.set()
    map_server.app.run(host='127.0.0.1', port=PORT, debug=False, use_reloader=False)


class AppWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Wargame Map Generator")
        self.resize(1440, 900)

        icon_path = bundle_dir() / 'assets' / 'icon.icns'
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self._browser = QWebEngineView()
        self._browser.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True
        )
        self.setCentralWidget(self._browser)
        self._browser.setHtml(
            "<body style='background:#1a1f2e;color:#8b9dc3;font-family:sans-serif;"
            "display:flex;align-items:center;justify-content:center;height:100vh;margin:0'>"
            "<p style='font-size:18px'>Starting Wargame Map Generator…</p></body>"
        )

    def load_app(self) -> None:
        self._browser.load(QUrl(f"http://127.0.0.1:{PORT}"))

    def closeEvent(self, event) -> None:
        event.accept()


def main() -> None:
    workspace = resolve_workspace()
    seed_workspace(workspace)

    app = QApplication(sys.argv)
    app.setApplicationName("Wargame Map Generator")
    app.setOrganizationName("Wargame Tools")

    window = AppWindow()
    window.show()

    ready = threading.Event()
    server_thread = threading.Thread(
        target=start_flask, args=(workspace, ready), daemon=True
    )
    server_thread.start()

    def _poll() -> None:
        if ready.is_set():
            QTimer.singleShot(200, window.load_app)
        else:
            QTimer.singleShot(100, _poll)

    QTimer.singleShot(100, _poll)
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
