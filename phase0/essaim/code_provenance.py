"""Explicit, nonsecret model/server declarations, required before any pilot inference."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

MODEL_KEYS = {"declared_at", "endpoint_kind", "model", "alias", "licence", "repository", "revision", "file",
              "quantization", "bytes", "sha256", "server", "hardware", "energy"}
SERVER_KEYS = {"version", "commit", "artifact", "sha256", "context", "parallel", "gpu_layers", "reasoning_budget", "jinja"}
HARDWARE_KEYS = {"cpu", "host_memory_gib", "execution", "gpu", "cpu_threads"}
ENERGY_KEYS = {"watts", "eur_kwh", "measured", "scope"}


def fields(value, keys):
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("provenance requires the exact documented nonsecret fields; credentials/extra fields refused")


def text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError("provenance strings must be nonempty and bounded")


def hex_value(value, length):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{" + length + "}", value):
        raise ValueError("invalid provenance hash/revision")


def declarations(path: Path, requested: list[str], watts: float, eur_kwh: float) -> dict:
    """Accept the parent's single-model file, or {models: {request_model: same declaration}} for comparison."""
    if path.stat().st_size > 65536:
        raise ValueError("provenance file too large")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict) and set(data) == {"models"}:
        models = data["models"]
    else:
        fields(data, MODEL_KEYS)
        models = {data["alias"]: data}
    if not isinstance(models, dict) or set(models) != set(requested):
        raise ValueError("provenance must cover exactly every requested peer/reference")
    for declaration in models.values():
        fields(declaration, MODEL_KEYS)
        for key in MODEL_KEYS - {"server", "hardware", "energy", "bytes"}:
            text(declaration[key])
        hex_value(declaration["sha256"], "64")
        hex_value(declaration["revision"], "40")
        if type(declaration["bytes"]) is not int or declaration["bytes"] <= 0:
            raise ValueError("positive model file size required")
        if any(c in declaration["file"] for c in "/\\"):
            raise ValueError("declare the model filename, not a private local path")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", declaration["repository"]):
            raise ValueError("declare a public repository identifier without credentials")
        server, hardware, energy = (declaration[k] for k in ("server", "hardware", "energy"))
        fields(server, SERVER_KEYS)
        for key in ("version", "commit", "artifact", "sha256"):
            text(server[key])
        hex_value(server["sha256"], "64")
        hex_value(server["commit"], "7,40")
        for key in ("context", "parallel", "gpu_layers", "reasoning_budget"):
            if type(server[key]) is not int or server[key] < (1 if key in ("context", "parallel") else 0):
                raise ValueError("server parameters must be explicit nonnegative integers")
        if type(server["jinja"]) is not bool:
            raise ValueError("explicit server template policy required")
        fields(hardware, HARDWARE_KEYS)
        for key in HARDWARE_KEYS - {"host_memory_gib"}:
            text(hardware[key])
        memory = hardware["host_memory_gib"]
        if type(memory) not in (int, float) or not math.isfinite(memory) or memory <= 0:
            raise ValueError("positive host memory declaration required")
        fields(energy, ENERGY_KEYS)
        text(energy["scope"])
        if any(type(energy[k]) not in (int, float) or not math.isfinite(energy[k]) for k in ("watts", "eur_kwh")):
            raise ValueError("finite numeric energy assumptions required")
        if energy["measured"] is not False or energy["watts"] != watts or energy["eur_kwh"] != eur_kwh:
            raise ValueError("energy declaration must match the CLI assumptions and remain explicitly estimated")
    return models
