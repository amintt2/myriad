"""Lifecycle of the node inside a long-running app: build it from the configuration, start it (node,
llama-server, OpenAI gateway on 127.0.0.1), pause/resume it, restart it after the wizard changed the
model, and stop it cleanly (llama-server is always terminated)."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import uvicorn

from .config import Config, key_path

log = logging.getLogger("myriad.runtime")


def is_configured(home: Path, serve: bool = True) -> bool:
    """A key, a configuration, and either no model (client only, or not serving) or a downloaded one."""
    cfg_file = Path(home) / "config.json"
    if not key_path(home).exists() or not cfg_file.exists():
        return False
    cfg = Config.load(home)
    if cfg.model is None or not serve:
        return True
    return bool(cfg.gguf_path) and Path(cfg.gguf_path).exists()


class NodeRuntime:
    def __init__(self, home: Path | None, serve: bool = True, tracker_override: str | None = None,
                 gateway_server: bool = True):
        self.home = Path(home) if home is not None else None
        self.serve, self.tracker_override, self.gateway_server = serve, tracker_override, gateway_server
        self.config = Config.load(self.home) if self.home is not None else Config()
        self.node = self.gateway = self.engine = None
        self.state = "arrêté"  # arrêté, démarrage, en marche, erreur
        self.error: str | None = None
        self.gateway_error: str | None = None
        self._tasks: list[asyncio.Task] = []
        self._server: uvicorn.Server | None = None
        self._lock = asyncio.Lock()
        self._attached = False

    @classmethod
    def attached(cls, node, gateway, config: Config, home: Path | None) -> "NodeRuntime":
        """A runtime around an already running node and gateway (tests, embedding)."""
        rt = cls(None, gateway_server=False)
        rt.home, rt.config, rt.node, rt.gateway = (Path(home) if home else None), config, node, gateway
        rt.engine = getattr(node, "engine", None)
        rt.state = "en marche"
        rt._attached = True
        return rt

    @property
    def running(self) -> bool:
        return self.node is not None

    def configured(self) -> bool:
        return self._attached or (self.home is not None and is_configured(self.home, self.serve))

    def status(self) -> dict:
        return {"state": self.state, "error": self.error, "gateway_error": self.gateway_error,
                "configured": self.configured()}

    async def start(self) -> None:
        from .service import build  # late import: service imports this module

        async with self._lock:
            if self.node is not None:
                return
            self.state, self.error = "démarrage", None
            try:
                cfg = Config.load(self.home)
                if self.tracker_override:
                    cfg.tracker_url = self.tracker_override
                node, gateway, engine = build(cfg, self.home, self.serve)
            except (SystemExit, Exception) as e:
                self.state, self.error = "erreur", str(e)
                raise RuntimeError(self.error) from None
            self.config, self.node, self.gateway, self.engine = cfg, node, gateway, engine
            self._tasks = [asyncio.create_task(node.run(), name="node"),
                           asyncio.create_task(self._start_engine(), name="engine")]
            if self.gateway_server:
                self._server = uvicorn.Server(uvicorn.Config(gateway.app, host="127.0.0.1", port=cfg.gateway_port,
                                                             log_level="warning"))
                self._tasks.append(asyncio.create_task(self._serve_gateway(), name="gateway"))
            self.state = "en marche"

    async def _serve_gateway(self) -> None:
        try:
            await self._server.serve()
        except (SystemExit, OSError) as e:  # port taken: the UI still works, only the local API is missing
            self.gateway_error = f"API locale indisponible sur le port {self.config.gateway_port} : {e}"
            log.error(self.gateway_error)
        if self._server is not None and not self._server.started and self.gateway_error is None:
            self.gateway_error = f"API locale indisponible sur le port {self.config.gateway_port}"

    async def _start_engine(self) -> None:
        engine, node = self.engine, self.node
        if engine is None:
            return
        try:
            await engine.start()
            log.info("llama-server prêt (%s)", getattr(engine, "gguf", "?"))
            await node.send_status()
        except Exception as e:
            engine.state = "erreur"
            node.last_error = f"moteur : {e}"
            log.error("llama-server n'a pas démarré : %s", e)

    async def stop(self) -> None:
        async with self._lock:
            node, gateway, engine, tasks = self.node, self.gateway, self.engine, self._tasks
            if node is None:
                return
            if self._server is not None:
                self._server.should_exit = True
            try:
                await node.stop()
            except Exception:
                pass
            if self._server is not None and tasks and tasks[-1].get_name() == "gateway":
                try:
                    await asyncio.wait_for(asyncio.shield(tasks[-1]), 5)
                except (asyncio.TimeoutError, Exception):
                    pass
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await gateway.close()
            finally:
                if engine is not None:
                    await engine.close()  # terminates llama-server
            self.node = self.gateway = self.engine = self._server = None
            self._tasks = []
            self.state, self.gateway_error = "arrêté", None

    async def restart(self) -> None:
        await self.stop()
        await self.start()

    async def set_accepting(self, accepting: bool) -> None:
        """Pause or resume serving the network; saved in the configuration."""
        node = self.node
        if node is not None:
            try:
                await node.set_limits(accepting=accepting)
            except ConnectionError:
                pass
        self.config.accepting = bool(accepting)
        if self.home is not None:
            cfg = Config.load(self.home)
            cfg.accepting = bool(accepting)
            cfg.save(self.home)

    @property
    def accepting(self) -> bool:
        return bool(self.node.accepting if self.node is not None else self.config.accepting)
