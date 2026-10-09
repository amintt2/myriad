"""Myriad desktop app (`myriad-desktop`): the node service (node, gateway, local UI) runs in a
background thread; the main thread shows the UI in a native window (pywebview: Edge WebView2 on
Windows, WKWebView on macOS, Qt/GTK on Linux) and a tray / menu-bar icon (pystray) with open, pause,
resume, update and quit. Closing the window keeps the node running in the tray. One instance per data
directory; a second launch asks the first one to show its window. Without pywebview, the UI opens in
the default browser. An update the user asked for (or `install_on_quit`) is installed once the service
has stopped (updater.py); the relaunched app waits for this one to release its lock (--after-update)."""
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
           "quit": "Quitter", "paused": "en pause", "sharing": "partage actif", "starting": "démarrage",
           "update_check": "Rechercher une mise à jour", "update_apply": "Mettre à jour et redémarrer ({v})",
           "up_to_date": "Myriad est à jour ({v}).", "update_found": "Myriad {v} est disponible.",
           "update_failed": "Vérification impossible : {e}"},
    "en": {"open": "Open Myriad", "pause": "Pause sharing", "resume": "Resume sharing", "quit": "Quit",
           "paused": "paused", "sharing": "sharing", "starting": "starting",
           "update_check": "Check for updates", "update_apply": "Update and restart ({v})",
           "up_to_date": "Myriad is up to date ({v}).", "update_found": "Myriad {v} is available.",
           "update_failed": "Could not check: {e}"},
}
AFTER_UPDATE_WAIT_S = 90.0  # a relaunched app waits this long for the old one to exit


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

    def updater():
        return getattr(svc.app, "updater", None) if svc.app is not None else None

    def update_ready() -> bool:
        u = updater()
        return bool(u is not None and u.status()["state"] == "ready" and u.kind.can_apply)

    def check_update(icon, item) -> None:
        u = updater()
        if u is None:
            return

        def work() -> None:
            try:
                st = svc.call(lambda: u.check(force=True), timeout=60) or u.status()
            except Exception as e:
                _notify(icon, t("update_failed").format(e=e))
                return
            if st.get("available"):
                _notify(icon, t("update_found").format(v=st["latest"]))
                on_open()  # the window shows the update banner
            elif st.get("state") == "error":
                _notify(icon, t("update_failed").format(e=st.get("error")))
            else:
                _notify(icon, t("up_to_date").format(v=st["current"]))
            try:
                icon.update_menu()
            except Exception:
                pass
        threading.Thread(target=work, name="myriad-update-check", daemon=True).start()

    def apply_update(icon, item) -> None:
        u = updater()
        if u is None:
            return
        try:
            svc.call(lambda: _async(u.request_install), timeout=10)
        except Exception as e:
            log.warning("update request failed: %s", e)

    menu = pystray.Menu(
        pystray.MenuItem(lambda item: t("open"), lambda icon, item: on_open(), default=True),
        pystray.MenuItem(lambda item: t("pause") if accepting() else t("resume"), toggle),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(lambda item: t("update_check"), check_update),
        pystray.MenuItem(lambda item: t("update_apply").format(v=(updater().latest if updater() else "") or ""),
                         apply_update, visible=lambda item: update_ready()),
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


async def _async(fn):
    return fn()


def _notify(icon, message: str) -> None:
    """A tray notification where the backend has them (best effort)."""
    try:
        if getattr(icon, "HAS_NOTIFICATION", False):
            icon.notify(message, APP_NAME)
            return
    except Exception:
        pass
    log.info("%s", message)


def _quietly(fn) -> None:
    try:
        fn()
    except Exception:  # noqa: BLE001 - quitting: best effort
        pass


class Quitter:
    """Quit once: stop the service (node, llama-server, gateway), then the tray icon and the window.
    Callable from any thread, any number of times; a call made while another one is stopping the
    service WAITS for it, so that the caller (e.g. the main thread once the window is closed) never
    returns, and main() never releases the instance lock, before the service has really stopped."""

    def __init__(self, svc, icon=lambda: None, window=lambda: None):
        self.svc, self._icon, self._window = svc, icon, window
        self.quitting = threading.Event()  # asked
        self.done = threading.Event()  # finished
        self._lock = threading.Lock()

    def __call__(self) -> None:
        self.quitting.set()
        with self._lock:  # a second caller blocks here until the first one has finished
            if self.done.is_set():
                return
            try:
                log.info("quitting")
                self.svc.stop()  # stops the node and terminates llama-server
                for get, how in ((self._icon, "stop"), (self._window, "destroy")):
                    obj = get()
                    if obj is not None:  # bounded: a GUI loop already gone must not block the exit
                        t = threading.Thread(target=_quietly, args=(getattr(obj, how),), daemon=True)
                        t.start()
                        t.join(5)
            finally:
                self.done.set()


def finish_update(svc) -> bool:
    """After the service has stopped (node, llama-server, gateway): install a pending update."""
    upd = getattr(getattr(svc, "app", None), "updater", None)
    if upd is None or svc.thread.is_alive():
        return False
    try:
        return upd.apply_pending()
    except Exception:  # noqa: BLE001 - the app is quitting: log, never raise
        log.exception("installing the update failed")
        return False


def acquire_after_update(lock: "InstanceLock", wait_s: float = AFTER_UPDATE_WAIT_S, step: float = 0.5) -> bool:
    """--after-update: the previous instance may still be exiting; wait for its lock."""
    end = time.monotonic() + wait_s
    while True:
        if lock.acquire():
            return True
        if time.monotonic() >= end:
            return False
        time.sleep(step)


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
    ap.add_argument("--after-update", action="store_true", help=argparse.SUPPRESS)  # relaunch after an update
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
    if not (acquire_after_update(lock) if args.after_update else lock.acquire()):
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

    holder: dict = {}
    try:
        show_request_path(home).unlink(missing_ok=True)
        stop_request_path(home).unlink(missing_ok=True)
        return _run(home, args, holder)
    finally:
        try:
            url_path(home).unlink(missing_ok=True)
        except OSError:
            pass
        try:
            # Everything is stopped: install a pending update, if any, STILL holding the lock, so that
            # no other instance starts (and cleans the files being swapped) meanwhile. The relaunched
            # app (--after-update) waits for the lock.
            if holder.get("svc") is not None:
                finish_update(holder["svc"])
        finally:
            lock.release()


def _run(home: Path, args, holder: dict | None = None) -> int:
    svc = ServiceThread(home, args.tracker)
    if holder is not None:
        holder["svc"] = svc
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

    window = None
    icon = None
    gui_done = False  # webview.start() returned: no window left to destroy
    quit_app = Quitter(svc, icon=lambda: icon, window=lambda: None if gui_done else window)
    quitting = quit_app.quitting

    def open_ui() -> None:
        if window is not None:
            try:
                window.show()
                window.restore()
                return
            except Exception:
                pass
        webbrowser.open(url)

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
                                          hidden=args.hidden, text_select=True, background_color="#101113")

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
            gui_done = True
            quit_app()  # waits for a stop already under way (closing the window started one)
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
