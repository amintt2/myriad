"""Rename essaim -> myriad: the legacy data directory is copied (never moved or deleted) on first start."""
from __future__ import annotations

import json
import os

import pytest

from myriad import config
from myriad.config import Config, home_dir, migrate_legacy_home


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def legacy(tmp_path):
    """A data directory as the essaim versions left it: key, config with absolute paths, a model, logs."""
    old = tmp_path / "essaim"
    (old / "models" / "Qwen__Qwen3-1.7B-GGUF").mkdir(parents=True)
    (old / "logs").mkdir()
    (old / "node_key.pem").write_bytes(b"not a real key, only bytes to copy\n")
    model = old / "models" / "Qwen__Qwen3-1.7B-GGUF" / "Qwen3-1.7B-Q8_0.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 2048)
    cfg = Config(model="Qwen/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q8_0.gguf", gguf_path=str(model),
                 llama_server="/usr/local/bin/llama-server", max_parallel=3)
    cfg.save(old)
    (old / "logs" / "myriad.log").write_text("old log\n", encoding="utf-8")
    (old / "myriad.url").write_text("http://127.0.0.1:8401/", encoding="utf-8")  # per-instance: not copied
    return old


@pytest.fixture
def platform(tmp_path, monkeypatch):
    monkeypatch.delenv("MYRIAD_HOME", raising=False)
    monkeypatch.delenv("ESSAIM_HOME", raising=False)
    monkeypatch.setattr(config, "platform_dir", lambda name: tmp_path / name)
    return tmp_path


def test_first_start_copies_the_legacy_directory(platform, legacy):
    before = snapshot(legacy)
    home = home_dir()
    assert home == platform / "myriad" and home.is_dir()
    # The legacy directory is untouched.
    assert snapshot(legacy) == before
    # Everything is there, except the per-instance files of the desktop app.
    after = snapshot(home)
    assert set(after) == set(before) - {"myriad.url"}
    assert after["node_key.pem"] == before["node_key.pem"]
    assert after["models/Qwen__Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf"] == before[
        "models/Qwen__Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf"]
    # Paths into the legacy directory now point into the new one; others are kept.
    cfg = Config.load(home)
    assert cfg.gguf_path == str(home / "models" / "Qwen__Qwen3-1.7B-GGUF" / "Qwen3-1.7B-Q8_0.gguf")
    assert cfg.llama_server == "/usr/local/bin/llama-server" and cfg.max_parallel == 3
    assert not list(platform.glob("myriad.migrating-*"))


def test_migration_runs_once(platform, legacy):
    home = home_dir()
    Config(model="other/repo:x.gguf").save(home)
    (legacy / "node_key.pem").write_bytes(b"changed later in the legacy directory")
    assert home_dir() == home
    assert Config.load(home).model == "other/repo:x.gguf"  # not copied again
    assert (home / "node_key.pem").read_bytes() != b"changed later in the legacy directory"


def test_no_legacy_directory(platform):
    home = home_dir()
    assert home == platform / "myriad" and not home.exists()  # created later by whoever needs it


def test_existing_new_directory_wins(platform, legacy):
    new = platform / "myriad"
    new.mkdir()
    assert home_dir() == new and not (new / "node_key.pem").exists()


def test_environment_overrides(platform, legacy, monkeypatch, tmp_path):
    monkeypatch.setenv("ESSAIM_HOME", str(tmp_path / "legacy-env"))
    assert home_dir() == tmp_path / "legacy-env"  # the former variable is still honoured
    monkeypatch.setenv("MYRIAD_HOME", str(tmp_path / "env"))
    assert home_dir() == tmp_path / "env"  # and the new one wins
    assert not (platform / "myriad").exists()  # no migration when the directory is given


def test_failed_copy_keeps_using_the_legacy_directory(platform, legacy, monkeypatch):
    before = snapshot(legacy)

    def boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(config.shutil, "copytree", boom)
    assert home_dir() == legacy
    assert snapshot(legacy) == before
    assert not (platform / "myriad").exists() and not list(platform.glob("myriad.migrating-*"))


def test_unreadable_config_keeps_using_the_legacy_directory(platform, legacy):
    (legacy / "config.json").write_text("{not json", encoding="utf-8")
    assert home_dir() == legacy
    assert not (platform / "myriad").exists() and not list(platform.glob("myriad.migrating-*"))


def test_running_legacy_instance_postpones_the_migration(platform, legacy):
    from myriad.desktop import InstanceLock

    lock = InstanceLock(legacy)  # an old desktop instance holds its data directory
    assert lock.acquire()
    try:
        assert home_dir() == legacy
        assert not (platform / "myriad").exists()
    finally:
        lock.release()
    assert home_dir() == platform / "myriad"  # migrated at the next start
    assert not (platform / "myriad" / "myriad.lock").exists()


def test_only_complete_large_models_are_hard_linked(platform, legacy, monkeypatch):
    monkeypatch.setattr(config, "LINK_MIN_BYTES", 1024)
    models = legacy / "models" / "Qwen__Qwen3-1.7B-GGUF"
    (models / "Qwen3-4B-Q8_0.gguf.part").write_bytes(b"partial" * 1024)  # resumed in place: copied
    (legacy / "llama.cpp" / "b1-cpu").mkdir(parents=True)
    (legacy / "llama.cpp" / "b1-cpu" / "llama-server.exe").write_bytes(b"MZ" * 2048)  # re-extracted: copied
    probe = platform / "probe"
    probe.write_bytes(b"x")
    try:
        os.link(probe, platform / "probe-link")
        can_link = True
    except OSError:
        can_link = False
    home = migrate_legacy_home(legacy, platform / "myriad")
    rel = os.path.join("models", "Qwen__Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q8_0.gguf")
    assert (home / rel).read_bytes() == (legacy / rel).read_bytes()
    assert os.path.samefile(home / rel, legacy / rel) == can_link
    for other in (os.path.join("models", "Qwen__Qwen3-1.7B-GGUF", "Qwen3-4B-Q8_0.gguf.part"),
                  os.path.join("llama.cpp", "b1-cpu", "llama-server.exe"), "node_key.pem", "config.json"):
        assert (home / other).read_bytes() and not os.path.samefile(home / other, legacy / other), other
    assert (home / "llama.cpp" / "b1-cpu" / "llama-server.exe").read_bytes() == b"MZ" * 2048
    assert json.loads((home / "config.json").read_text(encoding="utf-8"))["gguf_path"].startswith(str(home))
