"""Smoke test of a packaged Myriad build: start it headless on a fresh data directory, check that the
interface, its assets, the status and the setup wizard answer, then stop it with `--stop`.

    python packaging/smoke_test.py dist/Myriad/Myriad.exe            (Windows)
    python packaging/smoke_test.py dist/Myriad.app/Contents/MacOS/Myriad
    python packaging/smoke_test.py dist/Myriad/Myriad                 (Linux)

Standard library only (it runs outside the app's environment in CI)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


def get(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, r.read(), r.headers


def main() -> int:
    exe = Path(sys.argv[1]).resolve()
    home = Path(tempfile.mkdtemp(prefix="myriad-smoke-"))
    env = {**os.environ, "MYRIAD_HOME": str(home)}
    # A tracker that refuses connections at once: the app must still start and serve its interface.
    args = [str(exe), "--home", str(home), "--headless", "--tracker", "http://127.0.0.1:9"]
    proc = subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    url_file = home / "myriad.url"
    try:
        for _ in range(240):
            if url_file.exists() and url_file.read_text().strip():
                break
            if proc.poll() is not None:
                print(proc.stdout.read().decode(errors="replace"))
                raise SystemExit(f"the app exited early with code {proc.returncode}")
            time.sleep(0.5)
        else:
            raise SystemExit("no UI URL after 120 s")
        url = url_file.read_text().strip()
        status, body, headers = get(url)
        assert status == 200 and b"Myriad" in body and b"myriad-token" in body, "index page"
        assert "frame-ancestors 'none'" in headers.get("Content-Security-Policy", ""), "CSP header"
        for asset in ("static/app.js", "static/style.css", "static/i18n.js", "static/netviz.js", "static/icon.svg",
                      "static/fonts/inter-latin-wght-normal.woff2"):
            st, data, _ = get(url + asset)
            assert st == 200 and len(data) > 100, asset
        st = json.loads(get(url + "api/status")[1])
        assert st["app"] == "Myriad" and st["node"] is None and st["runtime"]["configured"] is False, st
        setup = json.loads(get(url + "api/setup")[1])
        assert setup["catalog"] and setup["recommendation"]["model"] and setup["hardware"]["os"], "wizard state"
        print(f"OK: {url} serves the UI; hardware: {setup['hardware']['cpu']} / "
              f"{(setup['hardware']['best_gpu'] or {}).get('name', 'no GPU')}; "
              f"recommended: {setup['recommendation']['model']} {setup['recommendation']['quant']}")
    finally:
        stop = subprocess.run([str(exe), "--home", str(home), "--stop"], env=env, capture_output=True, timeout=90)
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise SystemExit("the app did not stop after --stop")
    print(f"OK: stopped cleanly (exit {proc.returncode}, --stop exit {stop.returncode})")
    return 0 if proc.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
