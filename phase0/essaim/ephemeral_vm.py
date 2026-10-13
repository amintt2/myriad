"""Explicit Colab-only shell backend for non-adversarial public academic grading."""
from __future__ import annotations

import ctypes
import platform
import sys
from pathlib import Path

# The launcher runs under -I from a pinned source directory, not the candidate working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from essaim import local_linux

VERSION = "ephemeral-vm-v1"


def colab_markers() -> dict:
    """Require two filesystem markers supplied by the Colab image; environment variables are ignored."""
    return {"hostname_file": Path("/var/colab/hostname").is_file(),
            "drive_binary": Path("/opt/google/drive/drive").is_file()}


def require_colab() -> dict:
    markers = colab_markers()
    if (not sys.platform.startswith("linux") or platform.machine() != "x86_64"
            or "microsoft" in platform.release().lower() or not all(markers.values())):
        raise local_linux.IsolationError("ephemeral-vm requires a detected Colab VM and explicit selection")
    return markers


def select_backend(name: str):
    if name == "strict":
        return local_linux.LocalLinuxEnv
    if name == "ephemeral-vm":
        require_colab()
        return EphemeralVMEnv
    raise ValueError("unknown sandbox backend")


def without_landlock(work: Path, runtime: Path):
    # Preserve the dedicated-runtime restriction even though filesystem access is not confined by the kernel.
    python = (runtime / "bin/python").resolve()
    if python.parent != Path("/usr/bin"):
        prefix = python.parent.parent
        if prefix.parent.parts[-4:] != (".local", "share", "uv", "python") or not prefix.name.startswith("cpython-"):
            raise local_linux.IsolationError("runtime Python must be system Python or a dedicated uv installation")
    if ctypes.CDLL(None, use_errno=True).prctl(38, 1, 0, 0, 0):
        raise local_linux.IsolationError("no_new_privs failed")
    return None


def candidate_guard(work: Path) -> str:
    """Reuse the E11 Python policy without changing E11 or installing a hook in the controller/shell."""
    from essaim.sandbox import CHILD
    policy = CHILD[CHILD.index("_TMP ="):CHILD.index("_landlock_status =")]
    return ("import os, sys\n" + f"cfg = {{'tmp': {str(work)!r}, 'deny': [], 'read_extra': []}}\n"
            + policy + "\nsys.addaudithook(_hook)\n")


class EphemeralVMEnv(local_linux.LocalLinuxEnv):
    launcher = Path(__file__).resolve()
    version = VERSION

    def __init__(self, work: Path, runtime: Path):
        self.colab = require_colab()
        super().__init__(work, runtime)

    def run(self, cmd: str, timeout_s: float) -> tuple[int, str]:
        code, output = super().run(cmd, timeout_s)
        expected = {"isolation_ready": VERSION, "landlock_abi": None, "seccomp": "on",
                    "namespaces": ["user", "mount", "pid", "net", "ipc", "uts"]}
        if self.isolation != expected:
            raise local_linux.IsolationError("incomplete ephemeral-vm receipt")
        return code, output


if __name__ == "__main__" and sys.argv[1:] == ["--launch"]:
    require_colab()
    local_linux.launch(without_landlock, VERSION)
elif __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--enter":
    require_colab()
    local_linux.enter(int(sys.argv[2]), str(Path(__file__).resolve()))
