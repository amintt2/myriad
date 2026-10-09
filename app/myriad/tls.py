"""TLS trust roots for every HTTPS / WSS client of the app (httpx, websockets, huggingface_hub, urllib).

A frozen app (PyInstaller) ships its own Python and OpenSSL, which on macOS (and on some Linux distributions)
do not find the system CA certificates: every connection then fails with CERTIFICATE_VERIFY_FAILED. Fix:
use the operating system trust store through `truststore` (it patches `ssl.SSLContext` globally, so all the
clients created afterwards use it), and fall back on certifi's bundle if truststore is missing or fails.
Verification is never disabled. Call `setup_tls()` at the very start of every entry point."""
from __future__ import annotations

import logging
import os
import ssl
import sys
from typing import MutableMapping

log = logging.getLogger("myriad")

DEFAULT_SELFTEST_URL = "https://myriad.french-web.com/v1/health"
_CA_VARS = ("SSL_CERT_FILE", "SSL_CERT_DIR")
_mode: str | None = None


def _use_certifi(env: MutableMapping[str, str]) -> str:
    """Point OpenSSL (and httpx) at certifi's bundle, without overriding what the user set."""
    try:
        import certifi
        path = certifi.where()
    except Exception as e:  # noqa: BLE001 - nothing to fall back on: keep the interpreter defaults
        log.warning("no CA bundle available (certifi): %s", e)
        return "none"
    if not os.path.isfile(path):
        log.warning("certifi bundle missing: %s", path)
        return "none"
    if not any(env.get(v) for v in _CA_VARS):
        env["SSL_CERT_FILE"] = path
    return "certifi"


def _default_paths_usable() -> bool:
    p = ssl.get_default_verify_paths()
    return bool((p.cafile and os.path.isfile(p.cafile)) or (p.capath and os.path.isdir(p.capath)))


def setup_tls(env: MutableMapping[str, str] | None = None, *, force: bool = False) -> str:
    """Configure the trust roots; returns "env" (user-set SSL_CERT_FILE/DIR respected, left alone),
    "truststore", "certifi" or "none". Idempotent (the result is cached unless `force`)."""
    global _mode
    if _mode is not None and not force:
        return _mode
    env = os.environ if env is None else env
    if any(env.get(v) for v in _CA_VARS):
        mode = "env"  # the user chose the roots: OpenSSL and httpx both honor these variables
    else:
        try:
            import truststore
            truststore.inject_into_ssl()
            mode = "truststore"
            # On Linux truststore reads OpenSSL's default paths: if the distribution has none, add certifi.
            if sys.platform not in ("win32", "darwin") and not _default_paths_usable():
                _use_certifi(env)
        except Exception as e:  # noqa: BLE001 - ImportError, or a platform API failing
            log.warning("truststore unavailable (%s: %s): falling back on certifi", type(e).__name__, e)
            mode = _use_certifi(env)
    _mode = mode
    log.info("TLS trust roots: %s", mode)
    return mode


def selftest(url: str = DEFAULT_SELFTEST_URL, timeout: float = 20.0) -> int:
    """A real HTTPS GET with the app's HTTP client; 0 if the certificate chain verifies and the server answers."""
    import httpx
    mode = setup_tls()
    try:
        r = httpx.get(url, timeout=timeout, follow_redirects=True)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001 - any failure is a failed self-test
        _say(f"selftest-tls FAILED ({mode}): {url}: {type(e).__name__}: {e}", err=True)
        return 1
    _say(f"selftest-tls OK ({mode}): {url} -> HTTP {r.status_code}")
    return 0


def _say(msg: str, err: bool = False) -> None:
    stream = sys.stderr if err else sys.stdout
    if stream is not None:  # windowed PyInstaller apps have no stdio
        print(msg, file=stream, flush=True)
    log.log(logging.ERROR if err else logging.INFO, msg)
