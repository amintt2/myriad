"""First-run setup (also used later to change the model): detect the hardware, recommend a model,
then install in the background (llama.cpp binary, GGUF model, node key, configuration) with progress
reporting, cancellation and resumable downloads, and finally start the node."""
from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from . import catalog, hardware, llamacpp
from .config import DEFAULT_TRACKER, Config, key_path, parse_active_hours
from .crypto import Identity
from .downloader import DownloadError, Progress, download
from .node import http_url

log = logging.getLogger("myriad.wizard")


class SetupError(ValueError):
    pass


def default_tracker() -> str:
    return DEFAULT_TRACKER


def check_tracker_url(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    u = urlsplit(url)
    if u.scheme not in ("http", "https", "ws", "wss") or not u.hostname or len(url) > 300:
        raise SetupError("URL de traqueur invalide (http://hôte:port ou https://…)")
    return url


def models_dir(home: Path) -> Path:
    return Path(home) / "models"


def model_path(home: Path, m: catalog.CatalogModel, quant: str) -> Path:
    return models_dir(home) / m.repo.replace("/", "__") / m.file_for(quant).file


class InstallJob:
    def __init__(self, choice: dict):
        self.choice = choice
        self.steps: list[tuple[str, Progress]] = []
        self.state = "en cours"  # en cours, terminé, erreur, annulé
        self.error: str | None = None
        self.cancel = asyncio.Event()
        self.task: asyncio.Task | None = None

    def to_dict(self) -> dict:
        return {"state": self.state, "error": self.error, "choice": self.choice,
                "steps": [{"id": sid, **p.to_dict()} for sid, p in self.steps]}


class SetupWizard:
    def __init__(self, home: Path, runtime, detect=hardware.detect, http: httpx.AsyncClient | None = None):
        self.home = Path(home)
        self.runtime = runtime
        self._detect = detect
        self._http = http  # injected in tests
        self.hw: hardware.Hardware | None = None
        self.job: InstallJob | None = None

    async def hardware(self, refresh: bool = False) -> hardware.Hardware:
        if self.hw is None or refresh:
            self.hw = await asyncio.to_thread(self._detect)
        return self.hw

    async def network_families(self, tracker_url: str) -> dict[str, int]:
        """Families served on the network now (to favour the missing ones); {} if unreachable."""
        own = self._http is None
        client = self._http or httpx.AsyncClient(timeout=3)
        try:
            r = await client.get(http_url(tracker_url) + "/v1/peers", timeout=3)
            fams: dict[str, int] = {}
            for p in r.json().get("peers", []):
                f = p.get("family") or p.get("model")
                if f:
                    fams[f] = fams.get(f, 0) + 1
            return fams
        except (httpx.HTTPError, ValueError, AttributeError):
            return {}
        finally:
            if own:
                await client.aclose()

    async def state(self, refresh: bool = False) -> dict:
        hw = await self.hardware(refresh)
        cfg = Config.load(self.home)
        tracker = cfg.tracker_url if (self.home / "config.json").exists() else default_tracker()
        fams = await self.network_families(tracker)
        accel = hardware.accelerator(hw)
        existing = llamacpp.find_existing(self.home, cfg.llama_server)
        backends = []
        for b in llamacpp.variants(hw.os, hw.arch):
            assets = llamacpp.assets_for(hw.os, hw.arch, b)
            backends.append({"id": b, "size": sum(a.size for a in assets),
                             "installed": llamacpp.find_in(llamacpp.install_dir(self.home, b)) is not None})
        return {
            "configured": self.runtime.configured(),
            "hardware": hw.to_dict(),
            "catalog": catalog.evaluate(hw, fams),
            "recommendation": catalog.recommend(hw, fams),
            "network_families": fams,
            "engine": {"build": llamacpp.BUILD, "existing": existing, "backends": backends,
                       "default_backend": llamacpp.default_backend(hw.os, hw.arch, accel)},
            "defaults": {"tracker_url": tracker, "max_parallel": cfg.max_parallel or 1,
                         "accepting": cfg.accepting, "active_hours": cfg.active_hours,
                         "model": cfg.extra.get("catalog_id"), "quant": cfg.extra.get("quant")},
            "free_disk": shutil.disk_usage(self.home).free,
            "job": self.job.to_dict() if self.job else None,
        }

    def progress(self) -> dict | None:
        return self.job.to_dict() if self.job else None

    def validate(self, body: dict) -> dict:
        model_id = body.get("model")
        quant = body.get("quant") or catalog.DEFAULT_QUANT
        if model_id is not None:
            try:
                m = catalog.get(str(model_id))
                m.file_for(quant)
            except KeyError as e:
                raise SetupError(str(e.args[0])) from None
        try:
            mp = int(body.get("max_parallel", 1))
        except (TypeError, ValueError):
            raise SetupError("jobs simultanés : nombre entier attendu") from None
        if not 1 <= mp <= 16:
            raise SetupError("jobs simultanés entre 1 et 16")
        hours = body.get("active_hours")
        hours = (str(hours).strip() or None) if hours is not None else None
        try:
            parse_active_hours(hours)
        except (ValueError, TypeError):
            raise SetupError("plages horaires invalides (exemple : 8-23)") from None
        backend = body.get("backend") or None
        return {"model": model_id, "quant": quant, "max_parallel": mp, "accepting": bool(body.get("accepting", True)),
                "active_hours": hours, "tracker_url": check_tracker_url(body.get("tracker_url") or default_tracker()),
                "backend": backend}

    async def start(self, body: dict) -> dict:
        if self.job is not None and self.job.state == "en cours":
            raise SetupError("une installation est déjà en cours")
        choice = self.validate(body)
        hw = await self.hardware()
        if choice["model"] is not None:
            avail = llamacpp.variants(hw.os, hw.arch)
            if choice["backend"] is not None and choice["backend"] not in avail:
                raise SetupError(f"moteur {choice['backend']} indisponible ici ({', '.join(avail) or 'aucun'})")
        job = InstallJob(choice)
        self.job = job
        job.task = asyncio.create_task(self._install(job, hw))
        return job.to_dict()

    async def cancel(self) -> None:
        job = self.job
        if job is None or job.task is None or job.task.done():
            return
        job.cancel.set()
        job.task.cancel()
        try:
            await job.task
        except (asyncio.CancelledError, Exception):
            pass
        job.state = "annulé"

    async def _install(self, job: InstallJob, hw: hardware.Hardware) -> None:
        c = job.choice
        cfg = Config.load(self.home)
        try:
            plan: list[tuple[str, str, Path, str, int]] = []  # (step id, url, dest, sha256, size)
            binary = None
            backend = None
            model = catalog.get(c["model"]) if c["model"] else None
            if model is not None:
                existing = llamacpp.find_existing(self.home, cfg.llama_server)
                backend = c["backend"]
                if backend is None and existing:
                    binary = existing
                else:
                    backend = backend or llamacpp.default_backend(hw.os, hw.arch, hardware.accelerator(hw))
                    binary = llamacpp.find_in(llamacpp.install_dir(self.home, backend))
                    if binary is None:
                        for a in llamacpp.assets_for(hw.os, hw.arch, backend):
                            plan.append((f"engine:{a.name}", a.url, self.home / "downloads" / a.name, a.sha256,
                                         a.size))
                mf = model.file_for(c["quant"])
                plan.append(("model", model.url(c["quant"]), model_path(self.home, model, c["quant"]), mf.sha256,
                             mf.size))
            need = sum(size for _, _, dest, _, size in plan if not dest.exists())
            free = shutil.disk_usage(self.home).free
            if need and free < need * 1.1 + (1 << 30):
                raise SetupError(f"espace disque insuffisant : {need / 1e9:.1f} Go nécessaires, "
                                 f"{free / 1e9:.1f} Go libres")
            for sid, url, dest, _, size in plan:
                job.steps.append((sid, Progress(label=dest.name, total=size)))
            own = self._http is None
            client = self._http or httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(30.0, read=120.0),
                                                     headers={"User-Agent": "Myriad"})
            try:
                for (sid, url, dest, sha, size), (_, prog) in zip(plan, job.steps):
                    await download(url, dest, sha256=sha, size=size, client=client, progress=prog, cancel=job.cancel)
            finally:
                if own:
                    await client.aclose()
            if binary is None and backend is not None:
                target = llamacpp.install_dir(self.home, backend)
                tmp = target.with_name(target.name + ".tmp")
                if tmp.exists():
                    shutil.rmtree(tmp)
                for sid, _, dest, _, _ in plan:
                    if sid.startswith("engine:"):
                        await asyncio.to_thread(llamacpp.extract, dest, tmp)
                if target.exists():
                    shutil.rmtree(target)
                tmp.rename(target)
                found = llamacpp.find_in(target)
                if found is None:
                    raise SetupError("llama-server introuvable dans l'archive téléchargée")
                binary = str(found)
                for sid, _, dest, _, _ in plan:  # the archives are no longer needed
                    if sid.startswith("engine:"):
                        dest.unlink(missing_ok=True)
            Identity.load_or_create(key_path(self.home))
            cfg = Config.load(self.home)
            cfg.tracker_url = c["tracker_url"]
            cfg.max_parallel, cfg.accepting, cfg.active_hours = c["max_parallel"], c["accepting"], c["active_hours"]
            if model is None:
                cfg.model = cfg.gguf_path = cfg.family = cfg.params_b = None
            else:
                cfg.model = model.spec(c["quant"])
                cfg.gguf_path = str(model_path(self.home, model, c["quant"]))
                cfg.family, cfg.params_b = model.family, model.params_b
                cfg.llama_server = str(binary) if binary else cfg.llama_server
                cfg.extra.update(catalog_id=model.id, quant=c["quant"], backend=backend or cfg.extra.get("backend"))
            cfg.save(self.home)
            if self.runtime is not None:
                await self.runtime.restart()
            job.state = "terminé"
        except asyncio.CancelledError:
            job.state = "annulé"
            raise
        except (SetupError, DownloadError, OSError, KeyError, ValueError, RuntimeError) as e:
            job.state, job.error = "erreur", str(e)
            log.error("installation : %s", e)
