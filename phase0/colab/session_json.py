"""Strict adapter for the installed CLI's sessions output; never infer absence from an empty read."""
import hashlib
import json
import re
import sys


def parse(text):
    lines = text.strip().splitlines()
    if lines == ["[colab] No active sessions found on server."]:
        return {"schema": 1, "phase0": None}
    found = None
    if not lines:
        raise ValueError("empty sessions response")
    for line in lines:
        m = re.fullmatch(r"\[([^\]]+)\] (\S+) \| Hardware: (\S+) \| Shape: (.+) \| Variant: (\S+)", line)
        if not m:
            raise ValueError("unrecognized sessions response")
        if m[1] == "phase0":
            if found is not None:
                raise ValueError("duplicate phase0 session")
            found = {"fingerprint": hashlib.sha256(m[2].encode()).hexdigest(), "hardware": m[3]}
        if m[1] == "?":
            raise ValueError("untracked allocation: manual review required")
    return {"schema": 1, "phase0": found}


if __name__ == "__main__":
    print(json.dumps(parse(sys.stdin.read())))
