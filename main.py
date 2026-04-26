"""
Wargame Map Generator — native window entry point.

Starts the Flask server in a background thread and displays it inside a
PySide6 QWebEngineView so users get a double-click app with no terminal.

Also serves as the frozen subprocess dispatcher: when PyInstaller bundles
the app, all child-process calls route through this file via --run <module>
instead of trying to exec .py files directly.
"""

import os
import sys
import threading
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Frozen subprocess dispatch — must run BEFORE any heavy imports
# ---------------------------------------------------------------------------
# When frozen, map_server spawns: [sys.executable, '--run', 'tactical_map', ...]
# This block intercepts that and calls the right module's main().
if getattr(sys, 'frozen', False) and '--run' in sys.argv:
    _run_idx = sys.argv.index('--run')
    _module = sys.argv[_run_idx + 1]
    # Strip --run <module> so the target sees clean argv
    sys.argv = sys.argv[:_run_idx] + sys.argv[_run_idx + 2:]

    # Change to bundle directory so relative data paths work
    os.chdir(Path(sys._MEIPASS))

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


def base_dir() -> Path:
    """Root directory for bundled assets and Python files."""
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).parent


def start_flask(ready_event: threading.Event) -> None:
    """Run the Flask server in a daemon thread."""
    os.chdir(base_dir())
    import map_server
    # Signal the window to load once Flask has bound its port
    ready_event.set()
    map_server.app.run(host='127.0.0.1', port=PORT, debug=False, use_reloader=False)


class AppWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Wargame Map Generator")
        self.resize(1440, 900)

        icon_path = base_dir() / 'assets' / 'icon.icns'
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
        # Daemon thread dies automatically; just accept.
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Wargame Map Generator")
    app.setOrganizationName("Wargame Tools")

    window = AppWindow()
    window.show()

    ready = threading.Event()
    server_thread = threading.Thread(target=start_flask, args=(ready,), daemon=True)
    server_thread.start()

    def _on_ready() -> None:
        # Give Flask one extra tick to finish binding before we load
        QTimer.singleShot(200, window.load_app)

    # Poll until the ready event is set, then hand off to Qt timer
    def _poll() -> None:
        if ready.is_set():
            _on_ready()
        else:
            QTimer.singleShot(100, _poll)

    QTimer.singleShot(100, _poll)

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
