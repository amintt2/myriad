"""Hardware detection for the first-run wizard: GPU name and memory, RAM, CPU, and an estimate of the
memory bandwidth that bounds token generation speed.

Every probe is best effort and bounded in time; parsing is done by pure functions (tested on sample
outputs). Nothing here needs administrator rights."""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROBE_TIMEOUT_S = 8.0
MIN_VRAM_GB = 2.0


@dataclass
class Gpu:
    name: str
    vendor: str  # nvidia, amd, intel, apple, other
    vram_gb: float | None = None  # dedicated memory; for Apple Silicon: the unified memory
    unified: bool = False
    driver: str | None = None
    source: str = ""  # nvidia-smi, vulkaninfo, registry, sysfs, sysctl


@dataclass
class Hardware:
    os: str  # windows, macos, linux
    arch: str  # x64, arm64
    cpu: str
    cores: int
    ram_gb: float | None
    gpus: list[Gpu] = field(default_factory=list)

    @property
    def best_gpu(self) -> Gpu | None:
        """The GPU llama.cpp should use: unified memory (Apple Silicon), else the one with the most
        dedicated memory, at least MIN_VRAM_GB (an integrated GPU with shared memory is not used)."""
        usable = [g for g in self.gpus if g.unified or (g.vram_gb or 0.0) >= MIN_VRAM_GB]
        if not usable:
            return None
        return max(usable, key=lambda g: (g.unified, g.vram_gb or 0.0))

    def to_dict(self) -> dict:
        d = asdict(self)
        best = self.best_gpu
        d["best_gpu"] = asdict(best) if best else None
        d["accelerator"] = accelerator(self)
        d["bandwidth_gbs"] = round(bandwidth_gbs(self), 1)
        d["model_budget_gb"] = round(model_budget_gb(self), 2)
        return d


# ---------------------------------------------------------------------------------------------------
# Parsers (pure functions)

def parse_nvidia_smi(out: str) -> list[Gpu]:
    """`nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits`."""
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0]:
            continue
        try:
            mib = float(parts[1])
        except ValueError:
            continue
        gpus.append(Gpu(name=parts[0], vendor="nvidia", vram_gb=round(mib / 1024, 2),
                        driver=parts[2] if len(parts) > 2 and parts[2] else None, source="nvidia-smi"))
    return gpus


_VK_NAME = re.compile(r"^\s*deviceName\s*=\s*(.+?)\s*$", re.M)
_VK_TYPE = re.compile(r"^\s*deviceType\s*=\s*(\S+)", re.M)
_VK_VENDOR = re.compile(r"^\s*vendorID\s*=\s*(0x[0-9a-fA-F]+)", re.M)
VENDOR_IDS = {"0x10de": "nvidia", "0x1002": "amd", "0x1022": "amd", "0x8086": "intel", "0x106b": "apple"}


def vendor_of(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("nvidia", "geforce", "quadro", "tesla", "rtx", "gtx")):
        return "nvidia"
    if any(k in n for k in ("amd", "radeon", "ati ")):
        return "amd"
    if any(k in n for k in ("intel", "arc ", "iris", "uhd graphics")):
        return "intel"
    if n.startswith("apple"):
        return "apple"
    return "other"


def parse_vulkaninfo_summary(out: str) -> list[Gpu]:
    """`vulkaninfo --summary`: one block per device; CPU (llvmpipe) and virtual devices are skipped.
    Memory is not in the summary: it is filled from another source when available."""
    gpus = []
    blocks = re.split(r"\n\s*GPU\d+:\s*\n", "\n" + out)
    for b in blocks[1:]:
        name = _VK_NAME.search(b)
        if not name:
            continue
        typ = _VK_TYPE.search(b)
        typ = typ.group(1) if typ else ""
        if "CPU" in typ or "VIRTUAL" in typ:
            continue
        vid = _VK_VENDOR.search(b)
        vendor = VENDOR_IDS.get(vid.group(1).lower(), None) if vid else None
        gpus.append(Gpu(name=name.group(1), vendor=vendor or vendor_of(name.group(1)), source="vulkaninfo"))
    return gpus


def parse_meminfo(text: str) -> float | None:
    m = re.search(r"^MemTotal:\s+(\d+)\s+kB", text, re.M)
    return round(int(m.group(1)) / 1024**2, 2) if m else None


def parse_cpuinfo(text: str) -> str | None:
    m = re.search(r"^model name\s*:\s*(.+)$", text, re.M)
    return m.group(1).strip() if m else None


def apple_chip(brand: str) -> str | None:
    """'Apple M2 Pro' -> 'm2 pro'; None for an Intel Mac."""
    m = re.match(r"\s*Apple\s+(M\d+)(?:\s+(Pro|Max|Ultra))?", brand, re.I)
    if not m:
        return None
    return (m.group(1) + (" " + m.group(2) if m.group(2) else "")).lower()


# ---------------------------------------------------------------------------------------------------
# Bandwidth table (GB/s, desktop parts; laptops are slower). Token generation reads every weight once
# per token, so tokens/s is bounded by bandwidth / model size.

GPU_BANDWIDTH = (
    (r"rtx\s*5090", 1792), (r"rtx\s*5080", 960), (r"rtx\s*5070\s*ti", 896), (r"rtx\s*5070", 672),
    (r"rtx\s*5060", 448), (r"rtx\s*4090", 1008), (r"rtx\s*4080", 716), (r"rtx\s*4070\s*ti", 504),
    (r"rtx\s*4070", 504), (r"rtx\s*4060\s*ti", 288), (r"rtx\s*4060", 272), (r"rtx\s*3090", 936),
    (r"rtx\s*3080", 760), (r"rtx\s*3070", 448), (r"rtx\s*3060\s*ti", 448), (r"rtx\s*3060", 360),
    (r"rtx\s*3050", 224), (r"rtx\s*2080\s*ti", 616), (r"rtx\s*20[6-8]0", 400), (r"gtx\s*16[56]0", 192),
    (r"gtx\s*1080\s*ti", 484), (r"gtx\s*10[78]0", 288), (r"gtx\s*1060", 192),
    (r"rx\s*9070", 640), (r"rx\s*7900\s*xtx", 960), (r"rx\s*7900", 800), (r"rx\s*7800", 624),
    (r"rx\s*7700", 432), (r"rx\s*7600", 288), (r"rx\s*6[89]\d0", 512), (r"rx\s*67\d0", 384),
    (r"rx\s*6650", 280), (r"rx\s*6600\s*xt", 256), (r"rx\s*66\d0", 224), (r"arc\s*a770", 560), (r"arc\s*a750", 512), (r"arc\s*b580", 456),
)
APPLE_BANDWIDTH = {
    "m1": 68, "m1 pro": 200, "m1 max": 400, "m1 ultra": 800, "m2": 100, "m2 pro": 200, "m2 max": 400,
    "m2 ultra": 800, "m3": 100, "m3 pro": 150, "m3 max": 400, "m3 ultra": 819, "m4": 120, "m4 pro": 273,
    "m4 max": 546, "m5": 153,
}
CPU_BANDWIDTH = 40.0  # dual-channel DDR4/DDR5 desktop, roughly


def gpu_bandwidth(g: Gpu) -> float:
    if g.vendor == "apple":
        chip = apple_chip(g.name) or ""
        return float(APPLE_BANDWIDTH.get(chip, APPLE_BANDWIDTH.get(chip.split(" ")[0], 100)))
    n = g.name.lower()
    for pat, bw in GPU_BANDWIDTH:
        if re.search(pat, n):
            return float(bw)
    if g.vram_gb:  # unknown discrete GPU: more memory usually comes with a wider bus
        return float(max(120.0, min(g.vram_gb * 28.0, 700.0)))
    return 60.0 if g.vendor == "intel" else 150.0


def accelerator(hw: Hardware) -> str:
    """What llama.cpp will run on: metal, cuda, vulkan or cpu."""
    g = hw.best_gpu
    if g is None:
        return "cpu"
    if g.vendor == "apple":
        return "metal"
    if g.vendor == "nvidia":
        return "cuda"
    return "vulkan"


def model_budget_gb(hw: Hardware) -> float:
    """Memory available for the weights (plus a little context) without hurting the user's machine."""
    g = hw.best_gpu
    ram = hw.ram_gb or 8.0
    if g is not None and g.unified:
        return max(1.0, (g.vram_gb or ram) * 0.6)  # macOS lets the GPU wire ~2/3 of the memory
    if g is not None and g.vram_gb:
        return max(0.5, g.vram_gb - 1.2)  # context, compute buffers, the desktop itself
    return max(1.0, min(ram * 0.4, ram - 6.0))  # CPU: leave room for the OS and the user's apps


def bandwidth_gbs(hw: Hardware) -> float:
    g = hw.best_gpu
    if g is not None:
        return gpu_bandwidth(g)
    return CPU_BANDWIDTH


def estimate_tps(hw: Hardware, size_gb: float) -> float:
    """Expected generation speed (tokens/s) for a model file of `size_gb`: bandwidth/size times an
    efficiency factor (llama.cpp reaches ~60 % of the peak on a GPU, ~50 % on a CPU)."""
    if size_gb <= 0:
        return 0.0
    eff = 0.6 if accelerator(hw) != "cpu" else 0.5
    return eff * bandwidth_gbs(hw) / size_gb


# ---------------------------------------------------------------------------------------------------
# Probes

def _run(cmd: list[str]) -> str | None:
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=PROBE_TIMEOUT_S, creationflags=flags,
                           errors="replace", stdin=subprocess.DEVNULL)
        return r.stdout if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _nvidia_smi() -> str | None:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    if os.name == "nt":
        cand = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        if cand.exists():
            return str(cand)
    return None


def _ram_windows() -> float | None:
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return None
    return round(st.ullTotalPhys / 1024**3, 2)


def _windows_registry_gpus() -> list[Gpu]:
    """Display adapters from the registry: the 64-bit memory size (WMI's AdapterRAM stops at 4 GB)."""
    try:
        import winreg
    except ImportError:
        return []
    key = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    gpus = []
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key)
    except OSError:
        return []
    with root:
        for i in range(64):
            try:
                sub = winreg.EnumKey(root, i)
            except OSError:
                break
            if not sub.isdigit():
                continue
            try:
                with winreg.OpenKey(root, sub) as k:
                    name = str(winreg.QueryValueEx(k, "DriverDesc")[0])
                    try:
                        mem = winreg.QueryValueEx(k, "HardwareInformation.qwMemorySize")[0]
                        mem = int.from_bytes(mem, "little") if isinstance(mem, bytes) else int(mem)
                    except OSError:
                        mem = None
            except OSError:
                continue
            if "basic display" in name.lower() or "remote" in name.lower() or "virtual" in name.lower():
                continue
            gpus.append(Gpu(name=name, vendor=vendor_of(name), vram_gb=round(mem / 1024**3, 2) if mem else None,
                            source="registry"))
    return gpus


def _linux_sysfs_vram() -> list[float]:
    out = []
    for p in sorted(Path("/sys/class/drm").glob("card*/device/mem_info_vram_total")):
        try:
            out.append(round(int(p.read_text().strip()) / 1024**3, 2))
        except (OSError, ValueError):
            pass
    return out


def merge_gpus(primary: list[Gpu], extra: list[Gpu]) -> list[Gpu]:
    """Add the GPUs of `extra` not already in `primary` (same vendor); fill missing memory sizes."""
    out = list(primary)
    for g in extra:
        same = [p for p in out if p.vendor == g.vendor and (p.name.lower() in g.name.lower()
                                                            or g.name.lower() in p.name.lower())]
        if same:
            for p in same:
                if p.vram_gb is None and g.vram_gb is not None:
                    p.vram_gb = g.vram_gb
            continue
        if any(p.vendor == g.vendor for p in out) and g.vendor == "nvidia":
            continue  # nvidia-smi already listed every NVIDIA GPU
        out.append(g)
    return out


def total_ram_gb() -> float | None:
    """Physical memory only (cheap, no GPU probing)."""
    try:
        if os.name == "nt":
            return _ram_windows()
        if sys.platform == "darwin":
            mem = (_run(["sysctl", "-n", "hw.memsize"]) or "").strip()
            return round(int(mem) / 1024**3, 2) if mem.isdigit() else None
        return parse_meminfo(Path("/proc/meminfo").read_text())
    except (OSError, ValueError, AttributeError):
        return None


def detect() -> Hardware:
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
    cores = os.cpu_count() or 1
    if sys.platform == "darwin":
        brand = (_run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "").strip()
        mem = (_run(["sysctl", "-n", "hw.memsize"]) or "").strip()
        ram = round(int(mem) / 1024**3, 2) if mem.isdigit() else None
        hw = Hardware(os="macos", arch=arch, cpu=brand or platform.processor() or "?", cores=cores, ram_gb=ram)
        if apple_chip(brand):
            hw.gpus.append(Gpu(name=brand, vendor="apple", vram_gb=ram, unified=True, source="sysctl"))
        return hw
    if os.name == "nt":
        cpu = None
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                cpu = str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
        hw = Hardware(os="windows", arch=arch, cpu=cpu or platform.processor() or "?", cores=cores,
                      ram_gb=_ram_windows())
    else:
        try:
            cpu = parse_cpuinfo(Path("/proc/cpuinfo").read_text(errors="replace"))
        except OSError:
            cpu = None
        try:
            ram = parse_meminfo(Path("/proc/meminfo").read_text())
        except OSError:
            ram = None
        hw = Hardware(os="linux", arch=arch, cpu=cpu or platform.processor() or "?", cores=cores, ram_gb=ram)
    gpus: list[Gpu] = []
    smi = _nvidia_smi()
    if smi:
        gpus = parse_nvidia_smi(_run([smi, "--query-gpu=name,memory.total,driver_version",
                                      "--format=csv,noheader,nounits"]) or "")
    if os.name == "nt":
        gpus = merge_gpus(gpus, _windows_registry_gpus())
    else:
        vk = shutil.which("vulkaninfo")
        others = parse_vulkaninfo_summary(_run([vk, "--summary"]) or "") if vk else []
        vram = _linux_sysfs_vram()
        amd = [g for g in others if g.vendor == "amd"]
        for g, v in zip(amd, vram):
            g.vram_gb = v
        gpus = merge_gpus(gpus, others)
    hw.gpus = gpus
    return hw
