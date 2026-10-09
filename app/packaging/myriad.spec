# PyInstaller spec of the Myriad desktop app (one folder; a .app bundle on macOS).
#   cd app && uv run --extra desktop --group build pyinstaller packaging/myriad.spec --noconfirm
# Output: dist/Myriad/ (Windows, Linux) or dist/Myriad.app (macOS).
import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

HERE = Path(SPECPATH)
APP = HERE.parent
VERSION = re.search(r'__version__ = "([^"]+)"', (APP / "myriad" / "__init__.py").read_text()).group(1)
ICONS = HERE / "icons"

datas = collect_data_files("myriad", includes=["web/**/*", "priors.json"])
datas += [(str(APP.parent / name), ".") for name in ("LICENSE", "NOTICE") if (APP.parent / name).exists()]
hiddenimports = collect_submodules("myriad") + [
    "uvicorn.logging", "uvicorn.loops.auto", "uvicorn.loops.asyncio", "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl", "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl", "uvicorn.lifespan.on",
]
if sys.platform == "win32":
    hiddenimports += ["pystray._win32"]
elif sys.platform == "darwin":
    hiddenimports += ["pystray._darwin"]
else:
    hiddenimports += ["pystray._appindicator", "pystray._gtk", "pystray._xorg"]

a = Analysis(
    [str(HERE / "myriad_launcher.py")],
    pathex=[str(APP)],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "pytest", "IPython", "matplotlib", "numpy", "PyInstaller"],
    noarchive=False,
)
pyz = PYZ(a.pure)

version_file = None
if sys.platform == "win32":  # file properties of Myriad.exe (Explorer, SmartScreen)
    nums = tuple(int(x) for x in re.findall(r"\d+", VERSION)[:3]) + (0,)
    nums = (nums + (0, 0, 0, 0))[:4]
    version_file = Path(workpath) / "version_info.txt"
    version_file.parent.mkdir(parents=True, exist_ok=True)
    version_file.write_text(f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers={nums}, prodvers={nums}, mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1,
                    subtype=0x0, date=(0, 0)),
  kids=[StringFileInfo([StringTable('040904B0', [
      StringStruct('CompanyName', 'Myriad contributors'),
      StringStruct('FileDescription', 'Myriad - a myriad of small models, one answer'),
      StringStruct('FileVersion', '{VERSION}'),
      StringStruct('InternalName', 'Myriad'),
      StringStruct('LegalCopyright', 'Apache-2.0'),
      StringStruct('OriginalFilename', 'Myriad.exe'),
      StringStruct('ProductName', 'Myriad'),
      StringStruct('ProductVersion', '{VERSION}')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])]
)
""", encoding="utf-8")

icon = str(ICONS / ("myriad.ico" if sys.platform == "win32" else "myriad.icns" if sys.platform == "darwin"
                    else "myriad-256.png"))
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="Myriad",
    console=False,  # a windowed app: logs go to <data dir>/logs/
    icon=icon,
    version=str(version_file) if version_file else None,
    codesign_identity=None,  # macOS signing is done after the build (see .github/workflows/release.yml)
    entitlements_file=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="Myriad")

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Myriad.app",
        icon=icon,
        bundle_identifier="com.amintt2.myriad",
        version=VERSION,
        info_plist={
            "CFBundleName": "Myriad",
            "CFBundleDisplayName": "Myriad",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
            "NSHumanReadableCopyright": "Apache-2.0",
        },
    )
