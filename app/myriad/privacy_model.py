"""Local PII detector: should this prompt leave the machine?

`scan(text)` returns

    {"p_sensitive": float, "decision": "send" | "mask" | "local" | "ask",
     "spans": [{"start", "end", "type", "score", "source"}], "model": str | None, "guarantee": bool, ...}

Two layers run on the user's machine and nothing is sent anywhere:

1. the deterministic rules of `privacy_rules` (regex + checksums), always available;
2. an optional token-classification model (ONNX, CPU) from `PII_CATALOG`, downloaded through the
   verified downloader (pinned revision + SHA-256) and run with onnxruntime + tokenizers, without
   PyTorch and without any code from the model repository. These two packages come with the `pii`
   extra; without them, or without the model files, the scan degrades to the rules alone
   (`"model": None, "guarantee": False`).

The decision uses thresholds on the model's token scores, calibrated offline (bench/pii,
docs/09_detection_pii.md). A rule hit counts as score 1.

* The low threshold comes from split-conformal calibration on documents that contain personal data:
  the expected rate of such documents that get through is at most `alpha` for a new document drawn
  like the calibration ones.
  - Policy "block" (default, the owner's « if in doubt, do not send ») uses `lambda_low`: the prompt is
    held when a rule fires or when its best token score reaches the threshold.
  - Policy "mask" masks the rules' spans (reversible placeholders, as the `security` branch does) and
    uses `lambda_mask`, calibrated on the model's tokens *outside* those spans: a prompt whose personal
    data is not all covered by rules is held when the model scores its remainder at least that high.
    The model's own spans are never masked-and-sent automatically: per-span coverage could not be
    calibrated to a useful level (docs/09_detection_pii.md, section 5).
* `lambda_high` is chosen on the train split (false-positive rate). A model span scoring between the
  low threshold and `lambda_high` is the grey zone: the text is treated as sensitive and the user is
  asked ("ask"); above it the prompt stays local ("local"). See `decide`. Both hold the prompt: the
  bound is about automatic sending, and a user who then allows the sending is outside it.
* A low threshold of 0 means no useful threshold exists: every non-empty text is then held.

`p_sensitive` is the best token score (1 for a rule hit): a score to compare with the thresholds, not a
calibrated probability that the document contains personal data.

The integrated guard uses `model_decision`, based on model evidence alone, and ignores the rule spans
already handled with its fine types and preferences. The catalogue calibration is currently invalidated
by the rule/policy integration: low thresholds are zero and `guarantee` is False. A running model
therefore requires confirmation for every non-empty prompt. Displayed model spans use 0.5 in this mode;
they are descriptive, not a masking guarantee. The offline policies below remain available to the bench.

The bound holds only under exchangeability between the calibration documents and real prompts, which
the benchmarks do not ensure: see the documentation. Every failure of the model (error at inference)
is fail-closed: the decision becomes "ask" when the rules found nothing."""
from __future__ import annotations

import json
import threading
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import privacy_rules

TYPES = ("PERSON", "EMAIL", "PHONE", "ADDRESS", "ID", "FINANCIAL", "SECRET", "IP", "USERNAME", "DATE_OF_BIRTH")
DECISIONS = ("send", "mask", "local", "ask")
POLICIES = ("mask", "block")
SHARED_RULES = True  # the guard already scans the canonical rules; it can request model evidence only

# model label (without B-/I-, upper case) -> canonical type; labels absent from the map (CITY, DATE,
# COMPANY_NAME, AGE...) are not personal data on their own and do not count.
LABEL_MAP = {
    **{k: "PERSON" for k in ("GIVEN_NAME", "SURNAME", "FIRST_NAME", "LAST_NAME", "NAME", "PERSON", "GIVENNAME",
                             "MIDDLE_NAME", "FULL_NAME", "NAME_STUDENT", "NAME_MEDICAL_PROFESSIONAL")},
    **{k: "EMAIL" for k in ("EMAIL", "EMAIL_ADDRESS")},
    **{k: "PHONE" for k in ("PHONE", "PHONE_NUMBER", "TELEPHONENUM", "FAX_NUMBER")},
    **{k: "ADDRESS" for k in ("STREET_ADDRESS", "STREET_NAME", "BUILDING_NUMBER", "SECONDARY_ADDRESS", "STREET",
                              "BUILDINGNUM", "COORDINATE", "LOCATION_ADDRESS", "LOCATION_STREET")},
    **{k: "ID" for k in ("SSN", "US_SSN", "GOVERNMENT_ID", "NATIONAL_ID", "TAX_ID", "US_ITIN", "PASSPORT",
                         "US_PASSPORT", "DRIVERS_LICENSE", "US_DRIVER_LICENSE", "MEDICAL_RECORD_NUMBER",
                         "HEALTH_PLAN_BENEFICIARY_NUMBER", "CERTIFICATE_LICENSE_NUMBER", "CUSTOMER_ID",
                         "EMPLOYEE_ID", "LICENSE_PLATE", "US_LICENSE_PLATE", "VEHICLE_IDENTIFIER",
                         "DEVICE_IDENTIFIER", "BIOMETRIC_IDENTIFIER", "UNIQUE_ID", "MAC_ADDRESS", "IMEI",
                         "IDCARDNUM", "TAXNUM", "SOCIALNUM", "PASSPORTNUM", "DRIVERLICENSENUM")},
    **{k: "FINANCIAL" for k in ("CREDIT_CARD", "CREDIT_DEBIT_CARD", "IBAN", "IBAN_CODE", "ACCOUNT_NUMBER",
                                "BANK_ACCOUNT", "US_BANK_NUMBER", "CVV", "CREDITCARDNUMBER", "ACCOUNTNUM")},
    **{k: "SECRET" for k in ("PASSWORD", "API_KEY", "PIN", "HTTP_COOKIE", "SECRET", "TOKEN")},
    **{k: "IP" for k in ("IP_ADDRESS", "IPV4", "IPV6")},
    **{k: "USERNAME" for k in ("USERNAME", "USER_NAME")},
    "DATE_OF_BIRTH": "DATE_OF_BIRTH",
}


@dataclass(frozen=True)
class PiiFile:
    file: str
    size: int
    sha256: str


@dataclass(frozen=True)
class Calibration:
    """Thresholds on the token score. `alpha` is the target expected miss rate on documents that
    contain personal data; `n_cal` the number of such calibration documents. lambda_low: policy
    "block" (best token of the document); lambda_mask: policy "mask" (best token outside the rules'
    spans); lambda_high: end of the grey zone."""
    lambda_low: float
    lambda_mask: float
    lambda_high: float
    alpha: float
    n_cal: int
    note: str = ""
    validated: bool = True

    def low(self, policy: str) -> float:
        return self.lambda_mask if policy == "mask" else self.lambda_low


@dataclass(frozen=True)
class PiiModel:
    id: str
    name: str
    licence: str
    repo: str
    revision: str
    onnx: PiiFile
    tokenizer: PiiFile
    config: PiiFile
    calibration: Calibration
    langs: str = ""
    normalize: str | None = None  # "lower_strip_accents" for uncased models trained on folded text
    max_len: int = 512

    def files(self) -> list[PiiFile]:
        return [self.onnx, self.tokenizer, self.config]

    def url(self, f: PiiFile) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{f.file}"


# Calibrated on bench/pii (see docs/09_detection_pii.md, numbers in bench/results/pii_report.json).
PII_CATALOG: dict[str, PiiModel] = {
    m.id: m for m in (
        PiiModel(
            id="nym-pii-small-edge", name="nym-pii-multilingual-small (edge int8)", licence="mit",
            repo="Wismut/nym-pii-multilingual-small", revision="4348999cd3c2e20c49615e9af7c6bbb45b64cd85",
            onnx=PiiFile("edge-int8/model_int8.onnx", 108233527,
                         "e9a3a8c8cd55b3bcf329de5a9307cfae5053ce93c00330c33facd022e60daa17"),
            tokenizer=PiiFile("edge-int8/tokenizer.json", 12389891,
                              "c299144e68dfec1dc536204a7ae3712710c5c6cade9269a83f5042250d47d8de"),
            config=PiiFile("edge-int8/config.json", 5688,
                           "3f07065571e22bb73eba28ddb1fae4509c0cf703762e3bfa7a6ab2111cf7cb88"),
            # Rule coverage and guard preferences changed since the source benchmark. Until the
            # integrated policy is recalibrated, retain every non-empty prompt when the model runs.
            calibration=Calibration(lambda_low=0.0, lambda_mask=0.0, lambda_high=0.9883, alpha=0.01,
                                    n_cal=0, note="calibration de l'intégration non validée", validated=False),
            langs="22 langues (dont FR, EN, DE, ES, IT, PT, NL, PL)"),
    )
}
DEFAULT_PII_MODEL = "nym-pii-small-edge"
assert all(m.licence in ("mit", "apache-2.0", "bsd-3-clause") for m in PII_CATALOG.values())


def model_dir(model: PiiModel, home: Path | None = None) -> Path:
    from .config import home_dir
    return (home or home_dir()) / "privacy" / model.id / model.revision[:12]


def is_installed(model: PiiModel, home: Path | None = None) -> bool:
    d = model_dir(model, home)
    return all((d / f.file).is_file() and (d / f.file).stat().st_size == f.size for f in model.files())


async def ensure_model(model: PiiModel | None = None, home: Path | None = None, client=None,
                       on_progress: Callable | None = None) -> Path:
    """Download the model files (resumable, size and SHA-256 checked) and return the model directory."""
    from .downloader import download
    model = model or PII_CATALOG[DEFAULT_PII_MODEL]
    d = model_dir(model, home)
    for f in model.files():
        await download(model.url(f), d / f.file, sha256=f.sha256, size=f.size, client=client,
                       on_progress=on_progress)
    return d


def runtime_available() -> bool:
    try:
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
    except ModuleNotFoundError as e:
        if e.name in ("numpy", "onnxruntime", "tokenizers"):
            return False
        raise
    return True


def fold(text: str) -> tuple[str, list[int]]:
    """Lower-case, NFKD and strip combining marks, character by character, keeping a map from each
    folded character to its index in `text` (offsets stay exact even when a character expands)."""
    out, idx = [], []
    for i, ch in enumerate(text):
        f = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c)).lower()
        out.append(f)
        idx.extend([i] * len(f))
    return "".join(out), idx


class OnnxTokenClassifier:
    """BIO token classifier exported to ONNX. Long texts are cut into overlapping windows of `max_len`
    tokens (never truncated); a token seen by two windows keeps its higher score (fail-closed).

    The score of a token is the probability mass of the in-scope labels, 1 - P(O) - P(out-of-scope
    labels): a recall-oriented score, not tied to the argmax label."""

    def __init__(self, onnx_path: Path, tokenizer_path: Path, config_path: Path, normalize: str | None = None,
                 max_len: int = 512, overlap: int = 64, threads: int | None = None, batch: int = 1,
                 low_memory: bool = True, session=None):
        import numpy as np
        from tokenizers import Tokenizer
        self.np = np
        # Memory (measured on nym-pii-multilingual-small, docs/09_detection_pii.md): parsing a large
        # tokenizer.json has a transient peak of ~250 MB, so it is loaded before the session rather than
        # on top of it; without the CPU arena and one window per run, the resident size after
        # inference stays ~230 MB instead of ~520 MB (long texts are ~1.5x slower).
        # Correctness: keep batch=1 for ModernBERT exports. Running several windows in one call changes
        # their logits (by up to 3.4, even for unpadded rows) in the nym and ettin ONNX graphs; BERT
        # exports (Rampart, bert-small) give identical results either way.
        self.tok = Tokenizer.from_file(str(tokenizer_path))
        self.tok.no_truncation()
        self.tok.no_padding()
        if session is None:  # tests inject a fake session
            import onnxruntime as ort
            so = ort.SessionOptions()
            if threads:
                so.intra_op_num_threads = threads
            if low_memory:
                so.enable_cpu_mem_arena = False
                so.enable_mem_pattern = False
            so.log_severity_level = 3
            session = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        self.session = session
        self.inputs = {i.name for i in self.session.get_inputs()}
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        id2label = {int(k): v for k, v in cfg["id2label"].items()}
        n = max(id2label) + 1
        self.types: list[str | None] = [None] * n
        for i, lab in id2label.items():
            base = lab[2:] if lab[:2] in ("B-", "I-", "S-", "E-", "L-", "U-") else lab
            self.types[i] = LABEL_MAP.get(base.upper())
        self.in_scope = np.array([t is not None for t in self.types])
        self.type_names = sorted({t for t in self.types if t})
        self.type_index = np.array([self.type_names.index(t) if t else -1 for t in self.types])
        special = self.tok.encode("", add_special_tokens=True).ids
        self.prefix, self.suffix = special[:1], special[1:2]
        self.pad_id = self.tok.token_to_id("[PAD]") or self.tok.token_to_id("<pad>") or 0
        self.normalize = normalize
        self.max_len, self.overlap, self.batch = max_len, overlap, batch

    def windows(self, n: int) -> list[tuple[int, int]]:
        w = self.max_len - len(self.prefix) - len(self.suffix)
        if n <= w:
            return [(0, n)]
        step = max(1, w - min(self.overlap, w // 2))
        out, s = [], 0
        while True:
            out.append((s, min(s + w, n)))
            if s + w >= n:
                return out
            s += step

    def token_scores(self, text: str) -> list[tuple[int, int, float, str]]:
        """(start, end, score, type) for every token of `text`, with offsets in the original text."""
        np = self.np
        src, idx = (fold(text) if self.normalize else (text, None))
        enc = self.tok.encode(src, add_special_tokens=False)
        ids, offs = enc.ids, enc.offsets
        if not ids:
            return []
        best = np.zeros(len(ids), dtype=np.float32)
        best_type = np.full(len(ids), -1)
        wins = self.windows(len(ids))
        for b in range(0, len(wins), self.batch):
            chunk = wins[b:b + self.batch]
            rows = [self.prefix + ids[s:e] + self.suffix for s, e in chunk]
            width = max(len(r) for r in rows)
            inp = np.full((len(rows), width), self.pad_id, dtype=np.int64)
            att = np.zeros((len(rows), width), dtype=np.int64)
            for i, r in enumerate(rows):
                inp[i, :len(r)] = r
                att[i, :len(r)] = 1
            feed = {"input_ids": inp, "attention_mask": att}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(inp)
            logits = self.session.run(None, feed)[0].astype(np.float32)
            logits -= logits.max(-1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(-1, keepdims=True)
            score = (p * self.in_scope).sum(-1)
            pin = np.where(self.in_scope, p, -1.0)
            arg = pin.argmax(-1)
            for i, (s, e) in enumerate(chunk):
                k = len(self.prefix)
                sc, ty = score[i, k:k + e - s], self.type_index[arg[i, k:k + e - s]]
                upd = sc > best[s:e]
                best[s:e] = np.where(upd, sc, best[s:e])
                best_type[s:e] = np.where(upd, ty, best_type[s:e])
        out = []
        for j, (a, z) in enumerate(offs):
            if z <= a:
                continue
            if idx is not None:
                a, z = idx[a], idx[z - 1] + 1
            t = self.type_names[best_type[j]] if best_type[j] >= 0 else "PERSON"
            out.append((a, z, float(best[j]), t))
        return out


def merge_tokens(tokens: list[tuple[int, int, float, str]], threshold: float, text: str = "",
                 gap: int = 1) -> list[dict]:
    """Contiguous tokens scoring at least `threshold` become one span (max score, type of the best
    token); tokens separated by at most `gap` characters of whitespace or punctuation are joined."""
    spans: list[dict] = []
    for a, z, s, t in sorted(tokens):
        if s < threshold:
            continue
        if spans:
            last = spans[-1]
            between = text[last["end"]:a] if text else ""
            if a <= last["end"] or (a - last["end"] <= gap and (not text or not between.strip(" -._'’"))):
                last["end"] = max(last["end"], z)
                if s > last["score"]:
                    last["score"], last["type"] = s, t
                continue
        spans.append({"start": a, "end": z, "type": t, "score": s})
    return spans


@dataclass
class PrivacyScanner:
    """Lazy, thread-safe scanner. `classifier` can be injected (tests); otherwise the catalogue model is
    loaded from `home` on the first scan when its files and the runtime are present."""
    model: PiiModel = field(default_factory=lambda: PII_CATALOG[DEFAULT_PII_MODEL])
    home: Path | None = None
    policy: str = "block"
    classifier: object | None = None
    calibration: Calibration | None = None
    threads: int | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _tried: bool = field(default=False, repr=False)
    _files_state: tuple | None = field(default=None, repr=False)
    _verified_state: tuple | None = field(default=None, repr=False)
    load_error: str | None = None
    broken: bool = False

    def __post_init__(self):
        if self.policy not in POLICIES:
            raise ValueError(f"politique inconnue : {self.policy}")
        self.calibration = self.calibration or self.model.calibration

    def _load(self):
        """The classifier, or None. A missing runtime or model is the normal degraded mode (checked
        again on the next scan, so a model downloaded meanwhile is picked up); a model that is present
        but fails its SHA-256 check or does not load is a failure (`broken`), handled fail-closed.
        File metadata is checked on each scan; a change retries size/hash verification and loading.
        Unchanged files reuse their verification or failure, without hashing the weights again."""
        with self._lock:
            if self.classifier is not None and self._files_state is None:  # injected classifier
                return self.classifier
            collected = False
            try:
                d = model_dir(self.model, self.home)
                state = []
                for f in self.model.files():
                    try:
                        st = (d / f.file).stat()
                        state.append((st.st_dev, st.st_ino, st.st_mode, st.st_size, st.st_mtime_ns, st.st_ctime_ns))
                    except FileNotFoundError:
                        state.append(None)
                state = tuple(state)
                collected = True
                if state == self._files_state and (self.classifier is not None or self._tried):
                    return self.classifier
                if state != self._files_state:
                    self.classifier, self._tried, self.broken = None, False, False
                    self._files_state = state
                present = any(s is not None for s in state)
                if not is_installed(self.model, self.home):
                    if present:
                        raise ValueError("fichiers du modèle incomplets ou modifiés")
                    self.load_error = "modèle non téléchargé"
                    return None
                # Verify present files even without the optional runtime: corruption is never absence.
                from .downloader import sha256_file
                if state != self._verified_state:
                    for f in self.model.files():
                        if sha256_file(d / f.file) != f.sha256:
                            raise ValueError(f"empreinte SHA-256 incorrecte pour {f.file}")
                    self._verified_state = state
                if not runtime_available():
                    self.load_error = "runtime absent (installer l'extra « pii »)"
                    return None
                self._tried = True
                self.load_error, self.broken = None, False
                self.classifier = OnnxTokenClassifier(d / self.model.onnx.file, d / self.model.tokenizer.file,
                                                      d / self.model.config.file, normalize=self.model.normalize,
                                                      max_len=self.model.max_len, threads=self.threads)
            except Exception as e:  # corrupted or tampered file, incompatible runtime...
                if not collected:  # incomplete metadata must not cache a failure against the old state
                    self._files_state = None
                self.load_error = f"chargement impossible : {e}"
                self._tried, self.broken = True, True
                self.classifier = None
            return self.classifier

    def available(self) -> bool:
        return self._load() is not None

    def scan(self, text: str, *, include_rules: bool = True) -> dict:
        rules = [dict(s, source="rules") for s in privacy_rules.find(text)] if include_rules else []
        cal = self.calibration
        clf = self._load()
        res = {"model": None, "guarantee": False, "policy": self.policy, "error": self.load_error}
        tokens = []
        failed = self.broken
        if clf is not None:
            try:
                tokens = clf.token_scores(text)
                res.update(model=self.model.id, guarantee=cal.validated, error=None)
            except Exception as e:  # fail closed below
                res.update(error=f"inférence impossible : {e}")
                clf, failed = None, True
        # The guard consumes this model-only decision. Rule hits keep their configured modes there.
        model_low = cal.lambda_low if cal.validated else 0.0
        model_only = merge_tokens(tokens, model_low, text)
        model_decision = "send"
        if failed or (clf is not None and model_low <= 0 and text.strip()):
            model_decision = "ask"
        elif clf is not None:
            model_decision = decide([], model_only, cal.lambda_high, "block")
        res["model_decision"] = model_decision
        low = cal.low(self.policy) if cal.validated else 0.0
        if self.policy == "mask":
            # the rules' spans get masked anyway: the model only judges what they leave
            tokens = residual_tokens(tokens, rules)
        model_spans = [dict(s, source="model") for s in merge_tokens(tokens, low if cal.validated else 0.5, text)]
        spans = sorted(rules + model_spans, key=lambda s: (s["start"], s["end"]))
        # a rule hit is certain for the decision (score 1); model scores need a validated calibration
        p_model = max((t[2] for t in tokens), default=0.0)
        p = 1.0 if rules else p_model
        res.update(p_sensitive=round(p, 4), spans=spans,
                   thresholds={"low": low, "high": cal.lambda_high, "alpha": cal.alpha})
        if clf is None:
            # degraded mode: rules only, no statistical guarantee. A model that is absent (not
            # installed) lets clean texts through; a model that is there but broken is fail-closed.
            res["decision"] = "ask" if failed else (("mask" if self.policy == "mask" else "local") if rules else "send")
            return res
        if low <= 0 and text.strip():
            # no useful calibrated threshold: everything is sensitive (a span-free residue could leak)
            res["decision"] = "local" if rules and self.policy == "block" else "ask"
            return res
        res["decision"] = decide(rules, model_spans, cal.lambda_high, self.policy)
        return res


def residual_tokens(tokens: list, rules: list) -> list:
    """Model tokens that no rule span overlaps."""
    return [t for t in tokens if not any(t[0] < r["end"] and r["start"] < t[1] for r in rules)]


def decide(rules: list, model_spans: list, lambda_high: float, policy: str) -> str:
    """The scanner's decision. `model_spans` are the model's spans above the low threshold (for policy
    "mask": outside the rules' spans).

    block: a rule hit or a model span >= lambda_high keeps the prompt local; a model span in the grey
           zone only asks; nothing: send.
    mask:  the rules' spans are masked; a model span left over means the model saw personal data the
           rules did not mask: grey zone -> ask, above lambda_high -> local (the model's spans are not
           calibrated for masking, so they are never masked and sent automatically); else rules -> mask,
           nothing -> send."""
    grey = any(s["score"] < lambda_high for s in model_spans)
    if policy == "mask":
        if model_spans:
            return "ask" if grey else "local"
        return "mask" if rules else "send"
    if rules:
        return "local"
    if model_spans:
        return "ask" if grey else "local"
    return "send"


_default: PrivacyScanner | None = None
_default_lock = threading.Lock()


def default_scanner() -> PrivacyScanner:
    global _default
    with _default_lock:
        if _default is None:
            _default = PrivacyScanner()
        return _default


def scan(text: str, *, include_rules: bool = True) -> dict:
    """Scan `text` with the default scanner (lazy model load; rules only when the model is absent)."""
    return default_scanner().scan(text, include_rules=include_rules)
