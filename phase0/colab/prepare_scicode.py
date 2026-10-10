"""VM-only SciCode preparation. No model is launched until every pinned input is validated."""
import hashlib
import importlib.metadata
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

GDOWN_VERSION = "5.2.0"
DRIVE_ID = "17G_k65N_6yFFZ2O-jQH00Lh6iaw3z-AW"
DOWNLOAD_SECONDS = 900


def digest(path, algorithm="sha256", blob=False):
    hashed = hashlib.new(algorithm)
    if blob:
        hashed.update(b"blob %d\0" % path.stat().st_size)
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            hashed.update(block)
    return hashed.hexdigest()


def publish(target, expected, download, algorithm="sha256", blob=False):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(target, algorithm, blob) != expected:
            raise ValueError(f"corrupt SciCode input: {target.name}")
        return target
    with tempfile.TemporaryDirectory(prefix=".scicode-", dir=target.parent) as directory:
        temporary = Path(directory) / target.name
        download(temporary)
        if digest(temporary, algorithm, blob) != expected:
            raise ValueError(f"SciCode digest mismatch: {target.name}")
        with temporary.open("r+b") as content:
            os.fsync(content.fileno())
        os.replace(temporary, target)
    return target


def prepare():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from essaim import data, scicode
    from huggingface_hub import hf_hub_download
    import huggingface_hub
    if importlib.metadata.version("gdown") != GDOWN_VERSION:
        raise ValueError("unexpected gdown version")

    def drive(output):
        # The subprocess bounds connection setup and all gdown requests; never expose remote error text.
        subprocess.run([sys.executable, "-m", "gdown", DRIVE_ID, "-O", str(output), "--quiet", "--no-cookies"],
                       check=True, timeout=DOWNLOAD_SECONDS, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    publish(data.DATA / scicode.H5_NAME, scicode.H5_SHA256, drive)
    paths = {}
    for name in scicode.FILES.values():
        def download(output, name=name):
            path = hf_hub_download(data.REPOS["scicode"], name, repo_type="dataset",
                                   revision=data.REVISIONS["scicode"], etag_timeout=30)
            shutil.copyfile(path, output)
        paths[name] = publish(data.DATA / "scicode_inputs" / name, scicode.BLOB_SHA1[name], download, "sha1", True)

    # Reuse the scientific parser and cache format, but only from the two validated local blobs.
    original_data, original_download = data.DATA, huggingface_hub.hf_hub_download
    with tempfile.TemporaryDirectory(prefix=".scicode-cache-", dir=original_data) as directory:
        try:
            data.DATA = Path(directory)
            huggingface_hub.hf_hub_download = lambda repo, name, **kw: str(paths[name])
            for split in scicode.FILES:
                scicode.problems(split)
            for source in sorted(data.DATA.iterdir()):
                expected = digest(source)
                publish(original_data / source.name, expected, lambda output, source=source: shutil.copyfile(source, output))
        finally:
            data.DATA, huggingface_hub.hf_hub_download = original_data, original_download


if __name__ == "__main__":
    try:
        prepare()
        print("SciCode inputs and cache verified", flush=True)
    except BaseException as exc:
        print(f"SciCode preparation failed: {type(exc).__name__}", flush=True)
        sys.exit(1)
