"""`myriad node` / Myriad desktop: the node, its llama-server, the gateway (127.0.0.1:8400) and the
interface (127.0.0.1:8401) in one process. Without a configuration (first run of the desktop app),
the interface serves the setup wizard and the node starts when the wizard is done."""
from __future__ import annotations

import asyncio
import logging
import secrets
import socket
from pathlib import Path
from typing import Callable

import uvicorn

from .config import Config, key_path
from .crypto import Identity
from .engine import LlamaServerEngine
from .gateway import Gateway
from .node import NodeClient
from .priors import family_of, params_of
from .routing import node_tags
from .runtime import NodeRuntime
from .ui import make_ui_app

log = logging.getLogger("myriad")


def build(cfg: Config, home: Path, serve: bool = True) -> tuple[NodeClient, Gateway, object]:
    kp = key_path(home)
    if not kp.exists():
        raise SystemExit(f"Aucune clé dans {home}. Lancez d'abord : myriad init")
    ident = Identity.load(kp)
    engine = None
    if serve and cfg.gguf_path:
        engine = LlamaServerEngine(cfg.gguf_path, cfg.repo_id, binary=cfg.llama_server, ctx=cfg.ctx,
                                   parallel=max(1, cfg.max_parallel), n_gpu_layers=cfg.n_gpu_layers,
                                   log_dir=home / "logs")
    repo = cfg.repo_id if engine else None
    node = NodeClient(ident, cfg.tracker_url, engine=engine, model=repo,
                      family=(cfg.family or family_of(repo)) if repo else None,
                      gguf=Path(cfg.gguf_path).name if engine else None,
                      params_b=(cfg.params_b or params_of(repo)) if repo else None,
                      ctx=cfg.ctx if engine else 0, max_parallel=cfg.max_parallel, accepting=cfg.accepting,
                      active_hours=cfg.active_hours, max_job_tokens=cfg.max_job_tokens,
                      tags=node_tags(repo, cfg.tags))
    gateway = Gateway(node, default_k=cfg.default_k, timeout_s=cfg.request_timeout_s)
    gateway.verify_commands = dict(cfg.verify_commands or {})  # sub-agent verification allow-list
    return node, gateway, engine


def port_free(port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


class App:
    """The UI server (always), the wizard, and the node runtime (once configured)."""

    def __init__(self, home: Path, serve: bool = True, tracker_override: str | None = None,
                 require_config: bool = False, ui_port: int | None = None):
        from .wizard import SetupWizard

        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True)
        self.runtime = NodeRuntime(self.home, serve=serve, tracker_override=tracker_override)
        self.require_config = require_config
        self.token = secrets.token_urlsafe(24)
        self.wizard = SetupWizard(self.home, self.runtime)
        cfg = self.runtime.config
        port = ui_port or cfg.ui_port
        if ui_port is None and not port_free(port):
            with socket.socket() as s:  # the default port is taken (another program): use a free one
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
        self.ui_port = port
        self.ui = make_ui_app(config=cfg, home=self.home, token=self.token, runtime=self.runtime, wizard=self.wizard)
        self.server = uvicorn.Server(uvicorn.Config(self.ui, host="127.0.0.1", port=port, log_level="warning"))
        self.stopped = asyncio.Event()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.ui_port}/"

    async def run(self, on_ready: Callable[[str], None] | None = None) -> None:
        if self.runtime.configured():
            try:
                await self.runtime.start()
            except RuntimeError as e:
                if self.require_config:
                    raise SystemExit(str(e)) from None
                log.error("le nœud n'a pas démarré : %s", e)  # the UI shows it; the wizard can repair
            n, cfg = self.runtime.node, self.runtime.config
            if n is not None:
                print(f"Nœud {n.node_id}")
                print(f"  modèle servi : {n.model or 'aucun (client seulement)'}")
                print(f"  traqueur     : {cfg.tracker_url}")
                print(f"  API OpenAI   : http://127.0.0.1:{cfg.gateway_port}/v1  (modèle « myriad »)")
        elif self.require_config:
            raise SystemExit(f"Aucune configuration complète dans {self.home}. Lancez d'abord : myriad init "
                             "(ou l'application Myriad, qui a un assistant d'installation)")
        else:
            print("Première utilisation : terminez l'installation dans l'interface.")
        print(f"  interface    : {self.url}", flush=True)
        serve = asyncio.create_task(self.server.serve())
        try:
            while not self.server.started:
                if serve.done():
                    serve.result()  # raises the bind error
                    raise SystemExit(f"l'interface n'a pas démarré sur le port {self.ui_port}")
                await asyncio.sleep(0.05)
            if on_ready:
                on_ready(self.url)
            stop = asyncio.create_task(self.stopped.wait())
            await asyncio.wait({serve, stop}, return_when=asyncio.FIRST_COMPLETED)
            if serve.done() and serve.exception():
                raise serve.exception()
        finally:
            self.server.should_exit = True
            await self.wizard.cancel()
            await self.runtime.stop()
            try:
                await asyncio.wait_for(serve, 5)
            except (asyncio.TimeoutError, Exception):
                serve.cancel()

    def request_stop(self) -> None:
        self.stopped.set()


async def run_node(cfg: Config, home: Path, serve: bool = True) -> None:
    """`myriad node`: needs `myriad init` first (or the desktop wizard)."""
    override = cfg.tracker_url if cfg.tracker_url != Config.load(home).tracker_url else None
    app = App(home, serve=serve, tracker_override=override, require_config=True, ui_port=cfg.ui_port)
    await app.run()
