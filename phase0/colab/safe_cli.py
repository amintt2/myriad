"""Capture official CLI diagnostics before emitting sanitized output; inherit the supervisor's process group."""
import re
import signal
import subprocess
import sys
import tempfile


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


def run(args):
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        p = subprocess.Popen(["colab", *args], stdout=out, stderr=err)
        try:
            code = p.wait()
        except BaseException:
            p.kill()
            p.wait()
            raise
        emit(out, sys.stdout)
        emit(err, sys.stderr)
    if code:
        print(f"colab {args[0]}: exit {code if code >= 0 else 128 - code}", file=sys.stderr)
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    sys.exit(run(sys.argv[1:]))
