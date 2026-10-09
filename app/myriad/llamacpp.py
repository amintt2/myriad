"""The llama.cpp server binary: find one already installed, or download the pinned official release
for this platform and accelerator, check its SHA-256 against the pinned table, and unpack it in the
app data directory.

Digests come from the GitHub release b11505 of ggml-org/llama.cpp (asset `digest` fields)."""
from __future__ import annotations

import os
import shutil
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

BUILD = "b11505"
RELEASE_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{BUILD}/"
EXE = "llama-server.exe" if os.name == "nt" else "llama-server"


@dataclass(frozen=True)
class Asset:
    name: str
    size: int
    sha256: str

    @property
    def url(self) -> str:
        return RELEASE_URL + self.name


A = Asset
ASSETS: dict[tuple[str, str, str], tuple[Asset, ...]] = {
    ("windows", "x64", "cpu"): (A("llama-b11505-bin-win-cpu-x64.zip", 19483521,
                                  "1b050214d5a0a0e40dc9ff1ca1ec50608ad989a4afaa87e2597e6b3841fc8674"),),
    ("windows", "x64", "vulkan"): (A("llama-b11505-bin-win-vulkan-x64.zip", 33442611,
                                     "ace19f118b9e32382ec9b9b123aa19e8fdf54981fb82e82977b5298efaf01d49"),),
    ("windows", "x64", "cuda"): (A("llama-b11505-bin-win-cuda-12.4-x64.zip", 265086800,
                                   "9e71430ee2d51d1746318a276575efd64125a0ff33a9e569c689e770cc468a6c"),
                                 A("cudart-llama-bin-win-cuda-12.4-x64.zip", 391443627,
                                   "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6")),
    ("windows", "arm64", "cpu"): (A("llama-b11505-bin-win-cpu-arm64.zip", 12313134,
                                    "406079f6edc408368a94e40a431361553436cc4fb943894e8f23df4a24ad3055"),),
    ("windows", "arm64", "vulkan"): (A("llama-b11505-bin-win-vulkan-arm64.zip", 25983599,
                                       "55e0ead98adc45009df815ee77bd8ab1d52ec0e62ab99a3e22e6c4cad7228131"),),
    ("macos", "arm64", "metal"): (A("llama-b11505-bin-macos-arm64.tar.gz", 12041656,
                                    "633dab37eb84bef55227e69c82a0f807bb54790f8df84166479ed3b5689e405d"),),
    ("macos", "x64", "cpu"): (A("llama-b11505-bin-macos-x64.tar.gz", 11568006,
                                "5178f1ceeb267e1303cc62226595319ac8c233f7fcd701fdf54a90fc0b93f70d"),),
    ("linux", "x64", "cpu"): (A("llama-b11505-bin-ubuntu-x64.tar.gz", 17788404,
                                "9e23bb8c48e0abbd03a558e8c341ec768fa32d56146a09e5fd13db5b0461e60c"),),
    ("linux", "x64", "vulkan"): (A("llama-b11505-bin-ubuntu-vulkan-x64.tar.gz", 31741729,
                                   "0eb809f4c17e3c484f064fb807a42a402f9c4a560d532a7b43e6e167a21228ab"),),
    ("linux", "x64", "cuda"): (A("llama-b11505-bin-ubuntu-cuda-12.8-x64.tar.gz", 172008023,
                                 "df35bc577d6c48e19062b49bbdb7697498051e3caa710f9073a0e0094d774b58"),
                               A("cudart-llama-b11505-bin-ubuntu-cuda-12.8-x64.tar.gz", 594377392,
                                 "4761f7566bb7d6c14df80e9ee6082911d2278f0da8becf5a187812089ef6c8d5")),
    ("linux", "arm64", "cpu"): (A("llama-b11505-bin-ubuntu-arm64.tar.gz", 13786065,
                                  "adf1064ca125f42fd2f5a038346a2c6c0b58fe68bd3b21a55a825c9373f9a34a"),),
    ("linux", "arm64", "vulkan"): (A("llama-b11505-bin-ubuntu-vulkan-arm64.tar.gz", 24950781,
                                     "edbb717e24ab9dfdaa5f5a750e59161820bdf695797f8f721baf6d5462fb37ba"),),
}
del A


def variants(os_name: str, arch: str) -> list[str]:
    return [b for (o, a, b) in ASSETS if o == os_name and a == arch]


def default_backend(os_name: str, arch: str, accelerator: str) -> str:
    """Metal on Apple Silicon; Vulkan for any other usable GPU (one small download that runs on
    NVIDIA, AMD and Intel alike; CUDA stays available as an option); the CPU build otherwise."""
    avail = variants(os_name, arch)
    if accelerator == "metal" and "metal" in avail:
        return "metal"
    if accelerator in ("cuda", "vulkan") and "vulkan" in avail:
        return "vulkan"
    return "cpu" if "cpu" in avail else (avail[0] if avail else "cpu")


def assets_for(os_name: str, arch: str, backend: str) -> tuple:
    try:
        return ASSETS[(os_name, arch, backend)]
    except KeyError:
        raise KeyError(f"pas de llama.cpp {BUILD} pour {os_name}/{arch}/{backend}") from None


def install_dir(home: Path, backend: str) -> Path:
    return Path(home) / "llama.cpp" / f"{BUILD}-{backend}"


def find_in(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    direct = directory / EXE
    if direct.is_file():
        return direct
    for p in sorted(directory.rglob(EXE)):
        if p.is_file():
            return p
    return None


def bundled_dirs() -> list[Path]:
    """Where a packaged app may ship llama.cpp: next to the executable, or in the PyInstaller bundle."""
    out = []
    if getattr(sys, "frozen", False):
        out.append(Path(sys.executable).resolve().parent / "llama.cpp")
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            out.append(Path(meipass) / "llama.cpp")
    return out


def find_existing(home: Path, configured: str | None = None) -> str | None:
    """Configured path, LLAMA_SERVER, a bundled copy, an earlier download, then the PATH."""
    for cand in (configured, os.environ.get("LLAMA_SERVER")):
        if cand and Path(cand).is_file():
            return str(Path(cand))
    for d in bundled_dirs():
        p = find_in(d)
        if p:
            return str(p)
    root = Path(home) / "llama.cpp"
    if root.is_dir():
        for d in sorted(root.iterdir(), reverse=True):
            if d.name.startswith(BUILD):
                p = find_in(d)
                if p:
                    return str(p)
    found = shutil.which(EXE)
    return found


def _safe_target(root: Path, name: str) -> Path:
    target = (root / name).resolve()
    if root.resolve() not in target.parents and target != root.resolve():
        raise ValueError(f"chemin d'archive refusé : {name}")
    return target


def extract(archive: Path, dest: Path) -> None:
    """Unpack a release archive; refuses absolute paths, '..' and links that leave `dest`."""
    dest.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                _safe_target(dest, info.filename)
            z.extractall(dest)
    else:
        with tarfile.open(archive, "r:*") as t:
            for m in t.getmembers():
                _safe_target(dest, m.name)
            t.extractall(dest, filter="data")
    if os.name != "nt":
        for p in dest.rglob("*"):
            if p.is_file() and not p.is_symlink() and (p.name.startswith("llama-") or p.suffix in (".so", ".dylib")):
                p.chmod(p.stat().st_mode | 0o755)
