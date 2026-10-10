"""Bound official CLI calls and sanitize diagnostics; forward interruption to every CLI descendant."""
import contextlib
import io
import json
import os
import re
import signal
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_timeout import run as bounded_run


def redact(text):
    # Mask all URL queries (including unfamiliar signed parameters) and common non-URL credentials.
    text = re.sub(r"(https?://[^\s?'\"<>]+)\?[^\s'\"<>]*", r"\1?[REDACTED]", text, flags=re.I)
    text = re.sub(r"((?:colab-runtime-proxy-token|access_token|token|api_key)\s*['\"]?\s*[:=]\s*['\"]?)"
                  r"[^\s&,'\"<>]+", r"\1[REDACTED]", text, flags=re.I)
    return re.sub(r"(bearer\s+)[^\s,'\"<>]+", r"\1[REDACTED]", text, flags=re.I)


def emit(source, dest):
    source.seek(0)
    while True:
        line = source.readline(65537)
        if not line:
            break
        if len(line) > 65536:
            # Do not leak a credential split across bounded reads of an oversized line.
            while line and not line.endswith(b"\n"):
                line = source.readline(65537)
            dest.write("[oversized CLI diagnostic omitted]\n")
        else:
            dest.write(redact(line.decode("utf-8", errors="replace")))
    dest.flush()


def run(args, seconds=None):
    if os.environ.get("DLLM_REQUIRE_OWNER") == "1" and args[0] in ("upload", "download", "exec", "stop"):
        from session_json import parse, owned_receipt
        try:
            saved = owned_receipt()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = run(["sessions"], seconds=seconds)
            if code:
                return code
            if parse(output.getvalue())["phase0"] != saved["phase0"]:
                raise PermissionError("owned session absent or replaced")
            if owned_receipt() != saved:
                raise PermissionError("receipt changed during identity read")
        except PermissionError as exc:
            print(f"cloud operation refused: {exc}; persistent ownership retained", file=sys.stderr)
            return 73
        except (ValueError, OSError, KeyError) as exc:
            print(f"cloud operation refused: {exc}; persistent ownership retained", file=sys.stderr)
            return 65
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        if seconds is None:
            seconds = float(os.environ.get("DLLM_CLI_SECONDS", "120"))
            if "DLLM_CLI_SECONDS" not in os.environ and args[0] == "exec" and "--timeout" in args:
                seconds = max(seconds, float(args[args.index("--timeout") + 1]) + 30)
        code = bounded_run(["colab", *args], seconds=seconds,
                           grace=float(os.environ.get("DLLM_CLI_GRACE", "5")), interrupt_term=False,
                           stdout=out, stderr=err)
        emit(out, sys.stdout)
        emit(err, sys.stderr)
    if code:
        print(f"colab {args[0]}: exit {code if code >= 0 else 128 - code}", file=sys.stderr)
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    sys.exit(run(sys.argv[1:]))
