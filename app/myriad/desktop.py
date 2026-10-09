"""Myriad desktop app (`myriad-desktop`): the node service (node, gateway, local UI) runs in a
background thread; the main thread shows the UI in a native window (pywebview: Edge WebView2 on
Windows, WKWebView on macOS, Qt/GTK on Linux) and a tray / menu-bar icon (pystray) with open, pause,
resume and quit. Closing the window keeps the node running in the tray. One instance per data
directory; a second launch asks the first one to show its window. Without pywebview, the UI opens in
the default browser."""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

from . import __version__
from .config import APP_NAME, Config, home_dir

log = logging.getLogger("myriad.desktop")

TRAY_TEXT = {
    "fr": {"open": "Ouvrir Myriad", "pause": "Mettre en pause le partage", "resume": "Reprendre le partage",
           "quit": "Quitter", "paused": "en pause", "sharing": "partage actif", "starting": "démarrage"},
    "en": {"open": "Open Myriad", "pause": "Pause sharing", "resume": "Resume sharing", "quit": "Quit",
           "paused": "paused", "sharing": "sharing", "starting": "starting"},
}


# ---------------------------------------------------------------------------------------------------
# Single instance

class InstanceLock:
    """An OS-level lock on `<home>/myriad.lock`, released automatically if the process dies."""

    def __init__(self, home: Path):
        self.path = Path(home) / "myriad.lock"
        self._f = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        f = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.close()
            return False
        self._f = f
        return True

    def release(self) -> None:
        if self._f is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                self._f.seek(0)
                msvcrt.locking(self._f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._f.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        self._f.close()
        self._f = None


def show_request_path(home: Path) -> Path:
    return Path(home) / "myriad.show"


def stop_request_path(home: Path) -> Path:
    return Path(home) / "myriad.stop"


def url_path(home: Path) -> Path:
    return Path(home) / "myriad.url"


# ---------------------------------------------------------------------------------------------------
# Service thread

class ServiceThread:
    """Runs service.App on its own asyncio loop."""

    def __init__(self, home: Path, tracker: str | None):
        self.home, self.tracker = home, tracker
        self.loop: asyncio.AbstractEventLoop | None = None
        self.app = None
        self.url: str | None = None
        self.error: BaseException | None = None
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._main, name="myriad-service", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _main(self) -> None:
        try:
            asyncio.run(self._run())
        except BaseException as e:  # noqa: BLE001 - reported to the main thread
            self.error = e
            log.exception("service stopped with an error")
        finally:
            self.ready.set()

    async def _run(self) -> None:
        from .service import App

        self.loop = asyncio.get_running_loop()
        self.app = App(self.home, tracker_override=self.tracker)

        def on_ready(url: str) -> None:
            self.url = url
            try:
                url_path(self.home).write_text(url, encoding="utf-8")
            except OSError:
                pass
            self.ready.set()

        await self.app.run(on_ready=on_ready)

    def call(self, coro_fn, timeout: float = 30.0):
        """Run a coroutine on the service loop from another thread."""
        if self.loop is None or self.loop.is_closed():
            return None
        fut = asyncio.run_coroutine_threadsafe(coro_fn(), self.loop)
        return fut.result(timeout)

    def stop(self, timeout: float = 40.0) -> None:
        if self.loop is not None and self.app is not None and not self.loop.is_closed():
            try:
                self.loop.call_soon_threadsafe(self.app.request_stop)
            except RuntimeError:
                pass
        self.thread.join(timeout)


# ---------------------------------------------------------------------------------------------------
# Tray icon

def tray_image():
    from PIL import Image

    from .icon import render_icon
    try:
        return render_icon(64)
    except Exception:  # never fail the app for an icon
        return Image.new("RGBA", (64, 64), (91, 108, 255, 255))


def lang_of(home: Path) -> str:
    try:
        lang = Config.load(home).extra.get("lang")
    except Exception:
        lang = None
    return lang if lang in TRAY_TEXT else "fr"


def make_tray(svc: ServiceThread, on_open, on_quit, home: Path):
    try:
        import pystray
    except Exception as e:  # missing package, or no tray support on this desktop
        log.warning("no tray icon: %s", e)
        return None

    def t(key: str) -> str:
        return TRAY_TEXT[lang_of(home)][key]

    def accepting() -> bool:
        rt = svc.app.runtime if svc.app is not None else None
        return bool(rt and rt.accepting)

    def toggle(icon, item) -> None:
        rt = svc.app.runtime if svc.app is not None else None
        if rt is None:
            return
        want = not rt.accepting
        try:
            svc.call(lambda: rt.set_accepting(want), timeout=10)
        except Exception as e:
            log.warning("pause/resume failed: %s", e)
        icon.update_menu()

    menu = pystray.Menu(
        pystray.MenuItem(lambda item: t("open"), lambda icon, item: on_open(), default=True),
        pystray.MenuItem(lambda item: t("pause") if accepting() else t("resume"), toggle),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(lambda item: t("quit"), lambda icon, item: on_quit()),
    )
    kwargs = {}
    if sys.platform == "darwin":
        try:
            from AppKit import NSApplication
            kwargs["darwin_nsapplication"] = NSApplication.sharedApplication()
        except Exception:
            pass
    try:
        icon = pystray.Icon("myriad", tray_image(), APP_NAME, menu, **kwargs)
    except Exception as e:
        log.warning("no tray icon: %s", e)
        return None
    if not getattr(icon, "HAS_MENU", True):
        # e.g. pystray's plain X11 backend: no menu, so no way to quit once the window is hidden.
        log.warning("tray backend without a menu (%s): closing the window quits", type(icon).__module__)
        return None
    return icon


# ---------------------------------------------------------------------------------------------------
# Main

def setup_logging(home: Path) -> None:
    logs = Path(home) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    if sys.stdout is None or sys.stderr is None:  # windowed build: no console
        stream = open(logs / "myriad-console.log", "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or stream
        sys.stderr = sys.stderr or stream
    handlers = [logging.FileHandler(logs / "myriad.log", encoding="utf-8"), logging.StreamHandler(sys.stderr)]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=handlers, force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="myriad-desktop", description=f"{APP_NAME} : application de bureau")
    ap.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    ap.add_argument("--home", help="dossier de données (défaut : dossier utilisateur, ou MYRIAD_HOME)")
    ap.add_argument("--tracker", help="URL du traqueur (remplace la configuration)")
    ap.add_argument("--browser", action="store_true", help="ouvrir l'interface dans le navigateur, sans fenêtre")
    ap.add_argument("--hidden", action="store_true", help="démarrer dans la zone de notification, sans fenêtre")
    ap.add_argument("--no-tray", action="store_true", help="pas d'icône de notification")
    ap.add_argument("--headless", action="store_true",
                    help="ni fenêtre, ni navigateur, ni icône : sert l'interface et attend (serveurs, tests)")
    ap.add_argument("--stop", action="store_true", help="arrêter proprement l'instance en cours, puis quitter")
    args = ap.parse_args(argv)
    home = Path(args.home).expanduser() if args.home else home_dir()
    home.mkdir(parents=True, exist_ok=True)
    setup_logging(home)

    lock = InstanceLock(home)
    if args.stop:
        if lock.acquire():
            lock.release()
            print(f"{APP_NAME} ne tourne pas.")
            return 0
        stop_request_path(home).write_text(str(time.time()), encoding="utf-8")
        for _ in range(120):  # the running instance stops llama-server, then releases the lock
            time.sleep(0.5)
            if lock.acquire():
                lock.release()
                print(f"{APP_NAME} est arrêté.")
                return 0
        print(f"{APP_NAME} ne s'est pas arrêté à temps.", file=sys.stderr)
        return 1
    if not lock.acquire():
        # Another instance runs: ask it to show its window, or open its UI in the browser.
        try:
            show_request_path(home).write_text(str(time.time()), encoding="utf-8")
        except OSError:
            pass
        time.sleep(1.5)
        if show_request_path(home).exists() and url_path(home).exists():
            webbrowser.open(url_path(home).read_text(encoding="utf-8").strip())
        print(f"{APP_NAME} est déjà lancé.")
        return 0

    try:
        show_request_path(home).unlink(missing_ok=True)
        stop_request_path(home).unlink(missing_ok=True)
        return _run(home, args)
    finally:
        lock.release()
        try:
            url_path(home).unlink(missing_ok=True)
        except OSError:
            pass


def _run(home: Path, args) -> int:
    svc = ServiceThread(home, args.tracker)
    svc.start()
    if not svc.ready.wait(120) or svc.url is None:
        log.error("the service did not start: %s", svc.error)
        print(f"{APP_NAME} n'a pas pu démarrer : {svc.error}", file=sys.stderr)
        svc.stop(5)
        return 1
    url = svc.url
    log.info("UI at %s", url)
    if args.headless:
        print(f"{APP_NAME} : {url}", flush=True)
        try:
            while svc.thread.is_alive() and not stop_request_path(home).exists():  # `--stop` stops it
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        svc.stop()
        return 0

    webview = None
    if not args.browser:
        try:
            import webview  # noqa: F811
        except Exception as e:
            log.warning("pywebview unavailable (%s): using the browser", e)
            webview = None

    quitting = threading.Event()
    window = None
    icon = None

    def open_ui() -> None:
        if window is not None:
            try:
                window.show()
                window.restore()
                return
            except Exception:
                pass
        webbrowser.open(url)

    def quit_app() -> None:
        if quitting.is_set():
            return
        quitting.set()
        log.info("quitting")
        svc.stop()  # stops the node and terminates llama-server
        if icon is not None:
            try:
                icon.stop()
            except Exception:
                pass
        if window is not None:
            try:
                window.destroy()
            except Exception:
                pass

    if not args.no_tray:
        icon = make_tray(svc, open_ui, lambda: threading.Thread(target=quit_app, daemon=True).start(), home)

    def watch_show_requests() -> None:
        p = show_request_path(home)
        while not quitting.is_set():
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
                open_ui()
            if stop_request_path(home).exists():
                quit_app()
                return
            if svc.error is not None or not svc.thread.is_alive():
                quit_app()
                return
            time.sleep(0.7)

    threading.Thread(target=watch_show_requests, name="myriad-watch", daemon=True).start()

    if webview is not None:
        try:
            window = webview.create_window(APP_NAME, url, width=1320, height=860, min_size=(960, 640),
                                          hidden=args.hidden, text_select=True, background_color="#0b0d14")

            def on_closing():
                if icon is not None and not quitting.is_set():
                    window.hide()  # keep serving the network from the tray
                    return False
                threading.Thread(target=quit_app, daemon=True).start()
                return True

            window.events.closing += on_closing
            if icon is not None:
                if sys.platform == "darwin":
                    icon.run_detached()  # integrates with the AppKit loop started by webview.start
                else:
                    threading.Thread(target=icon.run, name="myriad-tray", daemon=True).start()
            webview.start(private_mode=False, storage_path=str(home / "webview"))
            quit_app()
            return 0
        except Exception as e:
            log.warning("native window failed (%s): using the browser", e)
            window = None

    # Browser fallback.
    if not args.hidden:
        webbrowser.open(url)
    print(f"{APP_NAME} : {url}  (Ctrl+C pour quitter)", flush=True)
    try:
        if icon is not None:
            icon.run()  # blocks until "Quit"
        else:
            while svc.thread.is_alive() and not quitting.is_set():
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    quit_app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
