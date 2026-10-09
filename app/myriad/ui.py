"""Local web interface (127.0.0.1:8401): setup wizard, dashboard (network, counters, node card, limits),
chat playground with a live view of the peers, about panel.

Mutating calls need a per-process token embedded in the page: another web site open in the same
browser cannot read the page (same-origin policy), so it cannot obtain the token."""
from __future__ import annotations

import asyncio
import contextvars
import hmac
import json
import time
from importlib import resources
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from . import __version__
from .config import APP_NAME, Config, parse_active_hours
from .fusion import detect_task_hint, extract_answer
from .gateway import SWARM_MODEL, GatewayError, check_local, completion_body, normalize_messages, read_json
from .protocol import Assigned, JobError, JobFrame, ResultFrame
from .runtime import NodeRuntime

TEXT = "text/plain; charset=utf-8"
STATIC = {
    "app.js": "text/javascript; charset=utf-8",
    "i18n.js": "text/javascript; charset=utf-8",
    "netviz.js": "text/javascript; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
    "icon.svg": "image/svg+xml",
    "fonts/inter-latin-wght-normal.woff2": "font/woff2",
    "fonts/inter-latin-ext-wght-normal.woff2": "font/woff2",
    "fonts/jetbrains-mono-latin-wght-normal.woff2": "font/woff2",
}
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; font-src 'self'; connect-src 'self'; "
       "img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
RESEARCH_URL = "https://github.com/amintt2/myriad/tree/main/docs"
RATE_WINDOW_S = 60.0
_CHAT_STREAM = contextvars.ContextVar("myriad_chat_stream", default=None)


def _asset(name: str) -> bytes:
    return resources.files("myriad").joinpath("web", *name.split("/")).read_bytes()


def my_tokens_per_s(recent: list[dict], now: float | None = None, window_s: float = RATE_WINDOW_S) -> float:
    now = time.time() if now is None else now
    toks = sum(int(e.get("tokens") or 0) for e in recent if e.get("done_ts", e.get("ts", 0)) >= now - window_s)
    return round(toks / window_s, 2)


def make_ui_app(node=None, gateway=None, config: Config | None = None, home: Path | None = None, token: str = "",
                runtime: NodeRuntime | None = None, wizard=None) -> FastAPI:
    if not token:
        raise ValueError("token required")
    rt = runtime or NodeRuntime.attached(node, gateway, config or Config(), home)
    app = FastAPI(title="myriad ui", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    stats_cache: dict = {}

    def not_ready() -> JSONResponse:
        msg = rt.error or ("installation à terminer" if not rt.configured() else "le nœud démarre")
        return JSONResponse({"error": msg, "runtime": rt.status()}, status_code=503)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        bad = check_local(request)
        if bad:
            return JSONResponse({"error": bad}, status_code=403)
        if request.method != "GET":
            given = request.headers.get("x-myriad-token", "")
            if not hmac.compare_digest(given.encode(), token.encode()):
                return JSONResponse({"error": "jeton d'interface invalide"}, status_code=403)
        resp = await call_next(request)
        resp.headers["Content-Security-Policy"] = CSP
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        if not request.url.path.startswith("/static/fonts/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.get("/")
    async def index():
        html = _asset("index.html").decode("utf-8").replace("__TOKEN__", token).replace("__VERSION__", __version__)
        return HTMLResponse(html)

    @app.get("/static/{name:path}")
    async def static(name: str):
        if name not in STATIC:
            return JSONResponse({"error": "introuvable"}, status_code=404)
        resp = Response(_asset(name), media_type=STATIC[name])
        if name.startswith("fonts/"):
            resp.headers["Cache-Control"] = "public, max-age=86400"
        return resp

    @app.get("/api/status")
    async def status():
        n, gw = rt.node, rt.gateway
        base = {"app": APP_NAME, "version": __version__, "runtime": rt.status(), "research": RESEARCH_URL,
                "setup_job": wizard.progress() if wizard is not None else None}
        if n is None:
            return {**base, "node": None, "balance": None, "history": [], "gateway": None, "default_k": 4}
        st = n.status()
        st["tokens_per_s"] = my_tokens_per_s(st.get("recent", []))
        return {**base, "node": st, "balance": await gw.balance(), "history": gw.history[:20],
                "gateway": f"http://127.0.0.1:{rt.config.gateway_port}/v1", "default_k": gw.default_k}

    @app.get("/api/hardware")
    async def hardware_info():
        if "hw" not in stats_cache:
            from .hardware import detect
            stats_cache["hw"] = (await asyncio.to_thread(detect)).to_dict()
        return stats_cache["hw"]

    @app.get("/api/peers")
    async def peers():
        if rt.gateway is None:
            return not_ready()
        try:
            p, rel = await rt.gateway.directory()
        except GatewayError as e:
            return JSONResponse({"error": e.message}, status_code=e.status)
        return {"peers": p, "reliability": rel}

    @app.get("/api/network")
    async def network():
        """Peers, reliability and the tracker's network statistics (GET /v1/stats, when available)."""
        gw, n = rt.gateway, rt.node
        if gw is None:
            return not_ready()
        try:
            p, rel = await gw.directory()
        except GatewayError as e:
            return JSONResponse({"error": e.message}, status_code=e.status)
        now = time.monotonic()
        hit = stats_cache.get("s")
        if hit is None or now - hit[0] > 2.5:
            stats = None
            try:
                r = await gw.http.get("/v1/stats", params={"node_id": n.node_id}, timeout=5)
                if r.status_code == 200:
                    stats = r.json()
            except (httpx.HTTPError, ValueError):
                stats = None
            hit = (now, stats)
            stats_cache["s"] = hit
        stats = hit[1]
        if stats is None:  # an older tracker: what the directory tells
            fams: dict[str, int] = {}
            for x in p:
                f = x.get("family") or x.get("model")
                fams[f] = fams.get(f, 0) + 1
            stats = {"nodes_serving": len(p), "families": fams, "tokens_per_s": None, "jobs_per_min": None,
                     "busy_slots": sum(int(x.get("busy", 0)) for x in p), "partial": True}
        return {"peers": p, "reliability": rel, "stats": stats, "me": n.node_id}

    @app.post("/api/limits")
    async def limits(request: Request):
        n = rt.node
        try:
            body = await read_json(request)
            mp = body.get("max_parallel")
            mp = None if mp is None else int(mp)
            if mp is not None and not 0 <= mp <= 64:
                raise ValueError("jobs simultanés entre 0 et 64")
            hours = body.get("active_hours", ...)
            if hours is not ...:
                hours = (str(hours).strip() or None) if hours is not None else None
                parse_active_hours(hours)
            acc = body.get("accepting")
            acc = None if acc is None else bool(acc)
        except GatewayError as e:
            return JSONResponse({"error": e.message}, status_code=e.status)
        except (TypeError, ValueError) as e:
            return JSONResponse({"error": f"valeur invalide : {e}"}, status_code=400)
        if n is None:
            return not_ready()
        try:
            await n.set_limits(max_parallel=mp, accepting=acc, active_hours=hours)
        except ConnectionError:
            pass  # saved locally, sent to the tracker at the next connection
        cfg = rt.config
        cfg.max_parallel, cfg.accepting, cfg.active_hours = n.max_parallel, n.accepting, n.active_hours
        if rt.home is not None:
            saved = Config.load(rt.home) if (rt.home / "config.json").exists() else cfg
            saved.max_parallel, saved.accepting, saved.active_hours = n.max_parallel, n.accepting, n.active_hours
            saved.save(rt.home)
        return {"ok": True, "node": n.status()}

    async def _chat_args(request: Request) -> tuple[list[dict], dict]:
        body = await read_json(request)
        msg = str(body.get("message", "")).strip()
        if not msg:
            raise GatewayError(400, "message vide")
        k = body.get("k")
        hint = body.get("task_hint") or None
        return (normalize_messages([{"role": "user", "content": msg}]),
                {"max_tokens": int(body.get("max_tokens") or 512), "k": int(k) if k else None, "task_hint": hint})

    @app.post("/api/chat")
    async def chat(request: Request):
        try:
            messages, kw = await _chat_args(request)
            if rt.gateway is None:
                return not_ready()
            ans = await rt.gateway.ask(messages, **kw)
        except GatewayError as e:
            return JSONResponse({"error": e.message}, status_code=e.status)
        except (TypeError, ValueError) as e:
            return JSONResponse({"error": f"valeur invalide : {e}"}, status_code=400)
        return completion_body(ans, SWARM_MODEL)

    @app.post("/api/chat/stream")
    async def chat_stream(request: Request):
        """Server-sent events: `asked` (a job leaves for a peer), `answered` / `failed` (its result
        arrives, with the extracted answer), then `final` (the fused answer and the vote) or `error`."""
        try:
            messages, kw = await _chat_args(request)
        except GatewayError as e:
            return JSONResponse({"error": e.message}, status_code=e.status)
        except (TypeError, ValueError) as e:
            return JSONResponse({"error": f"valeur invalide : {e}"}, status_code=400)
        n, gw = rt.node, rt.gateway
        if gw is None:
            return not_ready()
        hint = kw["task_hint"] or detect_task_hint(messages)
        try:
            by_id = {p["node_id"]: p for p in (await gw.directory())[0]}
        except GatewayError:
            by_id = {}
        q: asyncio.Queue = asyncio.Queue()
        mine: dict[str, float] = {}
        tag = object()
        t0 = time.perf_counter()

        def ms() -> float:
            return round((time.perf_counter() - t0) * 1000, 1)

        def observe(direction: str, frame) -> None:
            if direction == "out" and isinstance(frame, JobFrame) and _CHAT_STREAM.get() is tag:
                jid = frame.job.job_id
                mine[jid] = time.perf_counter()
                target = getattr(frame, "target", None)
                p = by_id.get(target or "", {})
                q.put_nowait({"type": "asked", "job_id": jid, "node_id": target, "model": p.get("model"),
                              "family": p.get("family"), "ms": ms()})
            elif direction == "in":
                res = getattr(frame, "result", None)
                jid = getattr(res, "job_id", None) or getattr(frame, "job_id", None)
                if jid not in mine:
                    return
                if isinstance(frame, ResultFrame):
                    q.put_nowait({"type": "answered", "job_id": jid, "node_id": res.node_id, "model": res.model,
                                  "answer": extract_answer(res.text, hint), "text": res.text[:6000],
                                  "tokens": res.completion_tokens, "compute_ms": res.compute_ms, "ms": ms()})
                elif isinstance(frame, JobError):
                    q.put_nowait({"type": "failed", "job_id": jid, "error": frame.error, "ms": ms()})
                elif isinstance(frame, Assigned):  # essaim/1.1: the tracker names the peer it chose
                    card = frame.peer
                    q.put_nowait({"type": "assigned", "job_id": jid, "node_id": card.node_id, "model": card.model,
                                  "family": card.family, "ms": ms()})

        async def run() -> None:
            _CHAT_STREAM.set(tag)
            try:
                ans = await gw.ask(messages, **kw)
                q.put_nowait({"type": "final", "body": completion_body(ans, SWARM_MODEL), "ms": ms()})
            except GatewayError as e:
                q.put_nowait({"type": "error", "message": e.message, "status": e.status})
            except Exception as e:  # never leave the stream hanging
                q.put_nowait({"type": "error", "message": f"{type(e).__name__}: {e}", "status": 500})
            finally:
                q.put_nowait(None)

        n.add_observer(observe)
        q.put_nowait({"type": "start", "task_hint": hint, "k": kw["k"] or gw.default_k})
        task = asyncio.create_task(run())

        async def events():
            try:
                while True:
                    ev = await q.get()
                    if ev is None:
                        break
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            finally:
                n.remove_observer(observe)
                if not task.done():
                    task.cancel()

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.post("/api/prefs")
    async def prefs(request: Request):
        """Interface preferences kept in the configuration (the tray menu follows the language)."""
        try:
            body = await read_json(request)
        except GatewayError as e:
            return JSONResponse({"error": e.message}, status_code=e.status)
        lang = body.get("lang")
        if lang not in ("fr", "en"):
            return JSONResponse({"error": "langue inconnue"}, status_code=400)
        rt.config.extra["lang"] = lang
        if rt.home is not None and (rt.home / "config.json").exists():
            saved = Config.load(rt.home)
            saved.extra["lang"] = lang
            saved.save(rt.home)
        return {"ok": True}

    @app.post("/api/tracker")
    async def set_tracker(request: Request):
        """Change the tracker (e.g. the default one is unreachable): saved, then the node reconnects."""
        from .wizard import SetupError, check_tracker_url
        try:
            body = await read_json(request)
            url = check_tracker_url(str(body.get("url") or ""))
        except GatewayError as e:
            return JSONResponse({"error": e.message}, status_code=e.status)
        except SetupError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        if rt.home is None or not (rt.home / "config.json").exists():
            return JSONResponse({"error": "aucune configuration à modifier"}, status_code=409)
        saved = Config.load(rt.home)
        saved.tracker_url = url
        saved.save(rt.home)
        rt.config.tracker_url = url
        if getattr(rt, "tracker_override", None):
            rt.tracker_override = None  # the user's choice wins over the command line
        try:
            await rt.restart()
        except Exception as e:  # noqa: BLE001 - reported to the UI
            return JSONResponse({"error": f"redémarrage du nœud : {e}"}, status_code=500)
        return {"ok": True, "tracker_url": url}

    # ---------- setup wizard ----------
    if wizard is not None:
        from .wizard import SetupError

        @app.get("/api/setup")
        async def setup_state(refresh: bool = False):
            return await wizard.state(refresh=refresh)

        @app.get("/api/setup/progress")
        async def setup_progress():
            return {"job": wizard.progress(), "runtime": rt.status()}

        @app.post("/api/setup/install")
        async def setup_install(request: Request):
            try:
                body = await read_json(request)
                job = await wizard.start(body)
            except GatewayError as e:
                return JSONResponse({"error": e.message}, status_code=e.status)
            except SetupError as e:
                return JSONResponse({"error": str(e)}, status_code=400)
            return {"job": job}

        @app.post("/api/setup/cancel")
        async def setup_cancel():
            await wizard.cancel()
            return {"job": wizard.progress()}

    return app
