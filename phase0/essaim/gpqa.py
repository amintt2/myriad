"""E12: GPQA Diamond (198 graduate-level multiple-choice questions, 4 options).

GPQA is gated on Hugging Face (Idavidrein/gpqa, CC BY 4.0): the terms ask not to reveal examples "in plain text
or images online" (to keep the questions out of training corpora). Consequences here:
  * nothing in the code or the repository holds a question: the CSV is read from the owner's own copy
    (phase0/data/gpqa_diamond.csv, or $GPQA_DIAMOND_CSV) or downloaded with the owner's own Hugging Face login
    (huggingface_hub reads it), and checked against the file pinned below (size and git blob id, both read
    from the public metadata of the pinned commit; the SHA-256 of the canonical questions is recorded in
    every manifest by data._cached);
  * the cache (phase0/data/, ignored by git) and the raw results (aa_*_gpqa_*.jsonl, whose model outputs may
    quote a question) are left out of the public export (tools/export_public.py);
  * result files do not hold the question text, only the record id, the model output and the letters.

Options are shown in a fixed order per question: the correct answer goes to position rank mod 4 (rank in
the order of the Record IDs), the three wrong answers fill the others in a seeded shuffle. Gold letters are
therefore balanced exactly (49 or 50 of each), whatever a model's position bias.
"""
from __future__ import annotations

import csv
import hashlib
import io
import os
import random
from pathlib import Path

from . import data

FILE = "gpqa_diamond.csv"
PINNED = {"size": 1373492, "git_blob_sha1": "7589e3e467d69a1dceb126a60c4108d6d4f1d166"}  # at data.REVISIONS["gpqa"]
N_QUESTIONS = 198
COLUMNS = ("Record ID", "Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2",
           "Incorrect Answer 3")
GATED_HELP = (
    "GPQA Diamond est protégé (accès sur demande). Étape manuelle pour le propriétaire : se connecter à "
    "Hugging Face, ouvrir https://huggingface.co/datasets/Idavidrein/gpqa, accepter les conditions (ne pas "
    "publier les questions), puis soit télécharger gpqa_diamond.csv dans phase0/data/ (ou pointer GPQA_DIAMOND_CSV "
    "dessus), soit lancer `huggingface-cli login` avant ce programme. Le fichier est ensuite envoyé tel quel sur "
    "la VM par colab/colab_phase0.sh (jamais commité, jamais publié).")


def git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()


def check_file(raw: bytes) -> None:
    """Refuse any file but the pinned one (a changed copy would silently change the benchmark)."""
    if len(raw) != PINNED["size"] or git_blob_sha1(raw) != PINNED["git_blob_sha1"]:
        raise SystemExit(f"{FILE} ne correspond pas à la version fixée (commit {data.REVISIONS['gpqa'][:8]}, "
                         f"{PINNED['size']} octets, blob {PINNED['git_blob_sha1'][:8]}) : {len(raw)} octets, "
                         f"blob {git_blob_sha1(raw)[:8]}")


def local_path(default: Path | None = None) -> Path | None:
    """The owner's local CSV, override first; no download or validation."""
    for cand in (os.environ.get("GPQA_DIAMOND_CSV"), default if default is not None else data.DATA / FILE):
        if cand and Path(cand).is_file():
            return Path(cand)
    return None


def read_csv_bytes() -> bytes:
    path = local_path()
    if path is not None:
        raw = path.read_bytes()
        check_file(raw)
        return raw
    try:  # the owner's own Hugging Face login (cached token); nothing is ever passed on the command line
        path = data.hf_hub_download(data.REPOS["gpqa"], FILE, repo_type="dataset", revision=data.REVISIONS["gpqa"])
    except Exception as e:  # gated, not logged in, offline: all need the manual step
        raise SystemExit(f"{GATED_HELP}\n(détail : {type(e).__name__})")
    raw = Path(path).read_bytes()
    check_file(raw)
    return raw


def balanced_items(rows: list[dict], seed: int = 0) -> list[dict]:
    """Items {id, question, options, answer, domain, subdomain} from the CSV rows (see the module docstring)."""
    rows = sorted(rows, key=lambda r: r["Record ID"])
    out = []
    for rank, r in enumerate(rows):
        wrong = [r[f"Incorrect Answer {k}"].strip() for k in (1, 2, 3)]
        random.Random(f"gpqa-{seed}-{r['Record ID']}").shuffle(wrong)
        pos = rank % 4
        options = wrong[:pos] + [r["Correct Answer"].strip()] + wrong[pos:]
        out.append({"id": r["Record ID"], "question": r["Question"].strip(), "options": options, "answer": pos,
                    "domain": r.get("High-level domain", ""), "subdomain": r.get("Subdomain", "")})
    return out


def parse(raw: bytes, seed: int = 0) -> list[dict]:
    rd = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    missing = [c for c in COLUMNS if c not in (rd.fieldnames or [])]
    if missing:
        raise SystemExit(f"{FILE} : colonnes absentes {missing}")
    return balanced_items(list(rd), seed)


def diamond(seed: int = 0) -> list[dict]:
    """The 198 questions (one split: they are all used as test; there is no dev, see analyze_e12.py)."""
    def build():
        items = parse(read_csv_bytes(), seed)
        if len(items) != N_QUESTIONS or len({x["id"] for x in items}) != N_QUESTIONS:
            raise SystemExit(f"GPQA Diamond : {len(items)} questions lues, {N_QUESTIONS} attendues")
        return items
    return data._cached(f"gpqa_diamond_s{seed}", build, "gpqa")
