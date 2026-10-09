"""Privacy layer: deterministic rules, the optional model scanner (fake model, no network), conformal maths."""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from bench.pii import conformal
from myriad import privacy_model as pm
from myriad import privacy_rules as pr
from myriad import privacy

# ---------------------------------------------------------------- rules


def kinds(text: str) -> set[str]:
    return {s["type"] for s in pr.find(text)}


def test_checksums():
    assert pr.luhn_ok("4111111111111111") and not pr.luhn_ok("4111111111111112")
    assert pr.iban_ok("FR14 2004 1010 0505 0001 3M02 606") and not pr.iban_ok("FR15 2004 1010 0505 0001 3M02 606")
    body = "1850578006084"
    key = 97 - int(body) % 97
    assert pr.nir_ok(f"{body}{key:02d}") and not pr.nir_ok(f"{body}{(key % 97) + 1:02d}")
    assert pr.nir_ok("2 69 05 2A 123 456 " + f"{97 - int('2690519123456') % 97:02d}")


@pytest.mark.parametrize("text,kind", [
    ("écris à jean.dupont@example.org stp", "EMAIL"),
    ("rappelle-moi au 06 12 34 56 78", "PHONE"),
    ("mon numéro : +33 6 12 34 56 78", "PHONE"),
    ("IBAN FR14 2004 1010 0505 0001 3M02 606", "FINANCIAL"),
    ("carte 4111 1111 1111 1111 refusée", "FINANCIAL"),
    ("aws_access_key_id = AKIAZ7Q2K4LMN8P3RT5V", "SECRET"),
    ("token ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8", "SECRET"),
    ('client = OpenAI(api_key="sk-proj-abcdefghijklmnopqrstuvwxyz0123456789")', "SECRET"),
    ("-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----", "SECRET"),
    ("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36P", "SECRET"),
    ("password: Soleil2024!", "SECRET"),
    ("mon mot de passe Wi-Fi est Bordeaux4495*", "SECRET"),
    (r'File "C:\Users\mdupont\projets\main.py", line 3', "USERNAME"),
    ("open '/home/camille/.env'", "USERNAME"),
    ("ssh karim@203.0.113.7 -p 22", "USERNAME"),
    ("j'habite au 12 rue des Lilas, 75011 Paris", "ADDRESS"),
    ("46 Maple Drive, London SW1A 1AA", "ADDRESS"),
    ("Je m'appelle Camille Lefèvre", "PERSON"),
    ("née le 12/03/1985 à Lyon", "DATE_OF_BIRTH"),
    ("server 203.0.113.7 is down", "IP"),
])
def test_rules_find(text, kind):
    assert kind in kinds(text), pr.find(text)


@pytest.mark.parametrize("text", [
    "Quelle est la capitale de l'Australie ?",
    "commit 3f2a9c1e0b7d4a6f8e9d0c1b2a3f4e5d6c7b8a9f broke the build",
    "Version 1.2.3.4 broke the build",
    "numéro de commande 4970 1012 3456 7891",  # fails Luhn
    "Listen on 127.0.0.1:8104",
])
def test_rules_hard_negatives(text):
    assert pr.find(text) == [], pr.find(text)


@pytest.mark.parametrize("text,kind", [
    ("pwd=654321", "api_key"), ("pwd = abcdef", "api_key"), ("password=123456", "api_key"),
    ("token=abc.def", "api_key"), ("token=abc(def)", "api_key"), ("password=bonjour", "api_key"),
    ("mot de passe: Soleil2024!", "api_key"), ("sk-ant-" + "a" * 24, "api_key"),
    ("sk-live-" + "a" * 24, "api_key"), ("ghp_" + "a" * 30, "api_key"),
    ("The docs use AKIAIOSFODNN7EXAMPLE as a sample key", "api_key"),
    (r"C:\Users\Public\Documents\rapport.docx", "path"), ("/home/runner/work", "path"),
    (r"C:\Users\admin\file.txt", "path"),
    ('client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])', "api_key"),
    ('api_key = "YOUR_API_KEY"', "api_key"), ("password: ${DB_PASSWORD}", "api_key"),
    ('api_key="abcdef', "api_key"), ("auth_token=abcdef", "api_key"), ("access-token:123456", "api_key"),
    ("Bearer " + "a" * 24, "api_key"), ("github_pat_" + "a" * 42, "api_key"), ("xoxp-" + "a" * 24, "api_key"),
    ("+1 (415) 555 0132", "phone"), ("+49 30 1234 5678", "phone"), ("+61 2 1234 5678", "phone"),
    ("0033 6 12 34 56 78", "phone"), (r"C:\Documents and Settings\John\file.txt", "path"),
    ("/Users/$user/a", "path"),
    ("Dear John Smith", "name"), ("Bonjour Marie Curie", "name"),
])
def test_legacy_guard_formats_are_not_lost(text, kind):
    assert any(s.type == kind for s in privacy.detect(text))


def test_merge_keeps_best_type():
    out = pr.merge([{"start": 0, "end": 5, "type": "PHONE", "score": 0.7},
                    {"start": 3, "end": 9, "type": "FINANCIAL", "score": 1.0}])
    assert out == [{"start": 0, "end": 9, "type": "FINANCIAL", "score": 1.0}]


# ---------------------------------------------------------------- scanner with a fake classifier

CAL = pm.Calibration(lambda_low=0.3, lambda_mask=0.2, lambda_high=0.8, alpha=0.01, n_cal=1000)


class FakeClassifier:
    """Scores every occurrence of the given words."""

    def __init__(self, scores: dict[str, float], fail: bool = False):
        self.scores, self.fail = scores, fail

    def token_scores(self, text):
        if self.fail:
            raise RuntimeError("boom")
        out = []
        for w, s in self.scores.items():
            i = text.find(w)
            if i >= 0:
                out.append((i, i + len(w), s, "PERSON"))
        return out


def scanner(tmp_path, **kw):
    return pm.PrivacyScanner(home=tmp_path, calibration=CAL, **kw)


def test_regex_only_when_model_absent(tmp_path):
    s = scanner(tmp_path)  # default policy: block
    r = s.scan("écris à jean.dupont@example.org")
    assert r["model"] is None and r["guarantee"] is False and r["decision"] == "local" and r["policy"] == "block"
    assert r["spans"][0]["type"] == "EMAIL" and r["p_sensitive"] == 1.0
    assert s.scan("Quelle heure est-il ?")["decision"] == "send"
    assert scanner(tmp_path, policy="mask").scan("jean.dupont@example.org")["decision"] == "mask"
    assert s.load_error


def test_decisions_block(tmp_path):
    s = scanner(tmp_path, classifier=FakeClassifier({"Camille": 0.95, "Hugo": 0.5, "rien": 0.05}))
    assert s.scan("Il ne se passe rien.")["decision"] == "send"
    r = s.scan("Voici : Camille")
    assert r["decision"] == "local" and r["guarantee"] and r["spans"][0]["source"] == "model"
    assert r["spans"][0]["start"] == 8 and r["p_sensitive"] == 0.95
    assert s.scan("Victor Hugo")["decision"] == "ask"  # grey zone
    assert s.scan("Camille a lu Hugo")["decision"] == "ask"  # one span in the grey zone is enough to ask
    assert s.scan("Hugo : jean@example.org")["decision"] == "local"  # a rule hit is certain


def test_decisions_mask(tmp_path):
    clf = FakeClassifier({"Camille": 0.95, "Hugo": 0.5, "jean@example.org": 0.99})
    s = scanner(tmp_path, policy="mask", classifier=clf)
    # the model's score on the e-mail is ignored: the rule masks it
    r = s.scan("écris à jean@example.org")
    assert r["decision"] == "mask" and [x["source"] for x in r["spans"]] == ["rules"]
    assert s.scan("Camille : jean@example.org")["decision"] == "local"  # the model saw more than the rules
    assert s.scan("Hugo : jean@example.org")["decision"] == "ask"
    assert s.scan("rien à signaler")["decision"] == "send"


def test_policy_selects_threshold(tmp_path):
    clf = FakeClassifier({"Inès": 0.25})
    # 0.25 is above lambda_mask (0.2) but below lambda_low (0.3)
    assert scanner(tmp_path, policy="mask", classifier=clf).scan("voir Inès")["decision"] == "ask"
    r = scanner(tmp_path, policy="block", classifier=clf).scan("voir Inès")
    assert r["decision"] == "send" and r["thresholds"]["low"] == 0.3 and r["p_sensitive"] == 0.25


def test_zero_threshold_holds_everything(tmp_path):
    zero = pm.Calibration(lambda_low=0.0, lambda_mask=0.0, lambda_high=0.8, alpha=0.01, n_cal=50)
    for policy in ("block", "mask"):
        s = pm.PrivacyScanner(home=tmp_path, calibration=zero, policy=policy, classifier=FakeClassifier({}))
        assert s.scan("Quelle heure est-il ?")["decision"] == "ask"
        assert s.scan("   ")["decision"] == "send"
    s = pm.PrivacyScanner(home=tmp_path, calibration=zero, classifier=FakeClassifier({}))
    assert s.scan("jean@example.org")["decision"] == "local"


def test_decide_table():
    hi = 0.9

    def span(s):
        return {"score": s}
    assert pm.decide([], [], hi, "block") == "send"
    assert pm.decide([1], [], hi, "block") == "local"
    assert pm.decide([], [span(0.5)], hi, "block") == "ask"
    assert pm.decide([], [span(0.95)], hi, "block") == "local"
    assert pm.decide([1], [], hi, "mask") == "mask"
    assert pm.decide([1], [span(0.5)], hi, "mask") == "ask"
    assert pm.decide([], [span(0.95)], hi, "mask") == "local"
    toks = [(0, 4, 0.9, "PERSON"), (10, 12, 0.8, "EMAIL")]
    assert pm.residual_tokens(toks, [{"start": 9, "end": 20}]) == [(0, 4, 0.9, "PERSON")]


def test_model_failure_is_fail_closed(tmp_path):
    s = scanner(tmp_path, classifier=FakeClassifier({}, fail=True))
    r = s.scan("texte anodin")
    assert r["decision"] == "ask" and r["guarantee"] is False and "inférence" in r["error"]
    assert s.scan("jean@example.org")["decision"] == "ask"
    m = scanner(tmp_path, policy="mask", classifier=FakeClassifier({}, fail=True))
    assert m.scan("jean@example.org")["decision"] == "ask"


def _tiny_model(tmp_path, onnx_bytes=b"onnx"):
    blobs = {"m.onnx": onnx_bytes, "tokenizer.json": b"{}", "config.json": b'{"id2label": {}}'}

    def f(name, b):
        return pm.PiiFile(name, len(b), hashlib.sha256(b).hexdigest())
    model = replace(pm.PII_CATALOG[pm.DEFAULT_PII_MODEL], onnx=f("m.onnx", b"onnx"),
                    tokenizer=f("tokenizer.json", blobs["tokenizer.json"]), config=f("config.json", blobs["config.json"]))
    return model, blobs


def _install(model, blobs, home):
    d = pm.model_dir(model, home)
    d.mkdir(parents=True, exist_ok=True)
    for name, b in blobs.items():
        (d / name).write_bytes(b)


def test_model_downloaded_later_is_picked_up(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "runtime_available", lambda: True)
    monkeypatch.setattr(pm, "OnnxTokenClassifier", lambda *a, **k: FakeClassifier({"Camille": 0.99}))
    model, blobs = _tiny_model(tmp_path)
    s = pm.PrivacyScanner(model=model, home=tmp_path, calibration=CAL)
    assert s.scan("Voici : Camille")["decision"] == "send" and s.load_error == "modèle non téléchargé"
    _install(model, blobs, tmp_path)
    r = s.scan("Voici : Camille")
    assert r["decision"] == "local" and r["guarantee"] and r["model"] == model.id


def test_tampered_model_is_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "runtime_available", lambda: True)
    loaded = []
    monkeypatch.setattr(pm, "OnnxTokenClassifier", lambda *a, **k: loaded.append(1) or FakeClassifier({}))
    model, blobs = _tiny_model(tmp_path)
    _install(model, dict(blobs, **{"m.onnx": b"ONNX"}), tmp_path)  # same size, other bytes
    s = pm.PrivacyScanner(model=model, home=tmp_path, calibration=CAL)
    r = s.scan("texte anodin")
    assert r["decision"] == "ask" and not loaded and "SHA-256" in r["error"] and s.broken
    assert s.scan("jean@example.org")["decision"] == "ask"


def test_truncated_model_is_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "runtime_available", lambda: True)
    model, blobs = _tiny_model(tmp_path)
    _install(model, dict(blobs, **{"m.onnx": b"on"}), tmp_path)  # wrong size
    s = pm.PrivacyScanner(model=model, home=tmp_path, calibration=CAL)
    assert s.scan("texte anodin")["decision"] == "ask" and s.broken


@pytest.mark.parametrize("damage", ["incomplete", "size", "hash"])
def test_changed_model_files_recover_without_restart(tmp_path, monkeypatch, damage):
    from myriad import downloader
    model, blobs = _tiny_model(tmp_path)
    bad = dict(blobs)
    if damage == "incomplete":
        del bad["config.json"]
    else:
        bad["m.onnx"] = b"on" if damage == "size" else b"ONNX"
    _install(model, bad, tmp_path)
    hashed, loaded = [], []
    original = downloader.sha256_file
    def hash_file(path):
        hashed.append(path)
        return original(path)
    monkeypatch.setattr(downloader, "sha256_file", hash_file)
    monkeypatch.setattr(pm, "runtime_available", lambda: True)
    monkeypatch.setattr(pm, "OnnxTokenClassifier", lambda *a, **k: loaded.append(1) or FakeClassifier({"Camille": 0.99}))
    s = pm.PrivacyScanner(model=model, home=tmp_path, calibration=CAL)
    assert s.scan("Voici : Camille")["model_decision"] == "ask" and s.broken
    n = len(hashed)
    for _ in range(3):
        assert s.scan("texte anodin")["model_decision"] == "ask" and s.broken
    assert len(hashed) == n and not loaded  # unchanged corruption is retained without another hash
    _install(model, blobs, tmp_path)
    r = s.scan("Voici : Camille")
    assert r["model"] == model.id and r["model_decision"] == "local" and not s.broken and s.load_error is None
    assert len(hashed) == n + len(blobs) and loaded == [1]
    n = len(hashed)
    assert s.scan("texte anodin")["model_decision"] == "send"
    assert len(hashed) == n and loaded == [1]
    (pm.model_dir(model, tmp_path) / model.onnx.file).write_bytes(b"ONNX")
    assert s.scan("texte anodin")["model_decision"] == "ask" and s.broken


@pytest.mark.parametrize("error", [PermissionError, OSError])
@pytest.mark.parametrize("failures", [1, 3])
def test_metadata_access_recovers_with_unchanged_files(tmp_path, monkeypatch, error, failures):
    from myriad import downloader
    model, blobs = _tiny_model(tmp_path)
    _install(model, blobs, tmp_path)
    target = pm.model_dir(model, tmp_path) / model.tokenizer.file
    hashed, loaded = [], []
    original_hash, original_stat = downloader.sha256_file, Path.stat
    incident = False
    def hash_file(path):
        hashed.append(path)
        return original_hash(path)
    def stat(path, *args, **kwargs):
        if incident and path == target:
            raise error("accès temporairement impossible")
        return original_stat(path, *args, **kwargs)
    def classifier(*args, **kwargs):
        loaded.append(FakeClassifier({"Camille": 0.99}))
        return loaded[-1]
    monkeypatch.setattr(downloader, "sha256_file", hash_file)
    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(pm, "runtime_available", lambda: True)
    monkeypatch.setattr(pm, "OnnxTokenClassifier", classifier)
    s = pm.PrivacyScanner(model=model, home=tmp_path, calibration=CAL)
    assert s.scan("Voici : Camille")["model_decision"] == "local" and s.classifier is loaded[0]
    state = s._files_state
    assert len(hashed) == len(blobs) and len(loaded) == 1
    incident = True
    for _ in range(failures):
        r = s.scan("texte anodin")
        assert r["model_decision"] == "ask" and r["model"] is None and r["error"]
        assert s.broken and s.classifier is None and s._files_state is None
    assert len(hashed) == len(blobs) and len(loaded) == 1
    incident = False
    r = s.scan("Voici : Camille")
    assert r["model_decision"] == "local" and r["model"] == model.id and r["error"] is None
    assert not s.broken and s._files_state == state and s.classifier is loaded[1]
    assert loaded[1] is not loaded[0] and len(hashed) == len(blobs) and len(loaded) == 2
    for _ in range(3):
        assert s.scan("texte anodin")["model_decision"] == "send"
    assert len(hashed) == len(blobs) and len(loaded) == 2  # unchanged verified files are not rehashed
    (pm.model_dir(model, tmp_path) / model.onnx.file).write_bytes(b"ONNX")
    assert s.scan("texte anodin")["model_decision"] == "ask" and s.broken
    n = len(hashed)
    assert n == len(blobs) + 1
    for _ in range(3):
        assert s.scan("texte anodin")["model_decision"] == "ask" and s.broken
    assert len(hashed) == n and len(loaded) == 2  # stable corruption still caches its failure


@pytest.mark.parametrize("prefix", ["", "GB83 WEST 1234 5698 7654 32 "])
def test_adjacent_ibans_exact_coverage_and_restoration(tmp_path, monkeypatch, prefix):
    monkeypatch.setattr(pm, "_default", scanner(tmp_path))
    values = ["GB82 WEST 1234 5698 7654 32", "FR14 2004 1010 0505 0001 3M02 606"]
    text = prefix + " ".join(values) + " merci"
    spans = [s for s in privacy.detect(text) if s.type == "iban"]
    assert [(s.start, s.end) for s in spans] == [(text.index(v), text.index(v) + len(v)) for v in values]
    assert [(s["start"], s["end"]) for s in pr.find(text) if s["type"] == "FINANCIAL"] == \
        [(s.start, s.end) for s in spans]
    a = privacy.analyze([{"content": text}])
    assert a.decision == "mask" and list(a.mapping.values()) == values
    assert a.messages[0]["content"] == prefix + "IBAN_1 IBAN_2 merci"
    assert privacy.restore(a.messages[0]["content"], a.mapping) == text


@pytest.mark.parametrize("value", ["alice%dept@example.com", "alice.test+dept_name-test@example.org"])
def test_complete_email_local_part(tmp_path, monkeypatch, value):
    monkeypatch.setattr(pm, "_default", scanner(tmp_path))
    text = f"Voici : {value} merci"
    spans = [s for s in privacy.detect(text) if s.type == "email"]
    assert [(s.start, s.end) for s in spans] == [(8, 8 + len(value))]
    st = privacy.PrivacySettings.from_dict({"modes": {"email": "mask"}})
    a = privacy.analyze([{"content": text}], st)
    assert a.messages[0]["content"] == "Voici : EMAIL_1 merci" and a.mapping == {"EMAIL_1": value}
    assert privacy.restore(a.messages[0]["content"], a.mapping) == text


@pytest.mark.parametrize("key", ["pwd", "mot de passe", "password", "a-" * 3000 + "token", "foo.bar.api-key",
                                  "-" * 10000 + "pwd", "key.secret." + "a" * 10000])
def test_assignment_whole_keys_keep_legacy_values(key):
    text = f'{key}="abc.def(654321)"'
    spans = [s for s in pr.matches(text) if s["type"] == "SECRET"]
    assert any((s["start"], s["end"]) == (len(key) + 2, len(text) - 1) for s in spans)
    assert any(s.type == "api_key" for s in privacy.detect("ordinary=" + text))


def test_long_negative_assignments_at_prompt_limit_finish(tmp_path):
    # A generous subprocess timeout catches the former quadratic scan, without comparing short timings.
    code = """
import sys
from pathlib import Path
from myriad import privacy, privacy_model as pm
from myriad.protocol import MAX_PROMPT_CHARS
pm._default = pm.PrivacyScanner(home=Path(sys.argv[1]))
for text in ('-' * MAX_PROMPT_CHARS, 'a-' * (MAX_PROMPT_CHARS // 2), 'a=' * (MAX_PROMPT_CHARS // 2)):
    a = privacy.analyze([{'content': text}])
    assert a.decision == 'send' and not a.findings
"""
    r = subprocess.run([sys.executable, "-c", code, str(tmp_path)], cwd=Path(__file__).resolve().parents[1],
                       capture_output=True, timeout=15)
    assert r.returncode == 0, r.stderr.decode(errors="replace")


def test_guard_scans_shared_rules_once(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_default", scanner(tmp_path))
    calls, original = [], pr.matches
    def matches(text):
        calls.append(text)
        return original(text)
    monkeypatch.setattr(pr, "matches", matches)
    text = "pwd=654321"
    assert privacy.analyze([{"content": text}]).decision == "mask" and calls == [text]
    assert pm.scan(text)["decision"] == "local"  # public scan(text) keeps its standalone contract


@pytest.mark.parametrize("value,kind", [
    ("12 rue des Lilas, 75011 Paris", "other"),
    ("12 rue de la Paix, 75002 Paris", "other"),
    ("12 rue Jean Moulin", "other"),
    ("FR14 2004 1010 0505 0001 3M02 606", "iban"),
    ("fr14 2004 1010 0505 0001 3m02 606", "iban"),
    ("DE89370400440532013000", "iban"),
])
def test_exact_bounds_and_restoration(value, kind, tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_default", scanner(tmp_path))
    text = f"Voici : {value} merci."
    spans = [s for s in privacy.detect(text) if s.type == kind]
    assert [(s.start, s.end) for s in spans] == [(8, 8 + len(value))]
    a = privacy.analyze([{"role": "user", "content": text}])
    assert a.decision == "mask" and list(a.mapping.values()) == [value]
    assert a.messages[0]["content"].endswith(" merci.")
    assert privacy.restore(a.messages[0]["content"], a.mapping) == text


@pytest.mark.parametrize("mode,expected", [("off", "send"), ("warn", "ask"), ("mask", "mask"), ("local", "local")])
def test_guard_keeps_fine_preferences_with_optional_scanner(tmp_path, monkeypatch, mode, expected):
    monkeypatch.setattr(pm, "_default", scanner(tmp_path))
    text = "FR14 2004 1010 0505 0001 3M02 606 merci"
    st = privacy.PrivacySettings.from_dict({"modes": {"iban": mode}})
    a = privacy.analyze([{"role": "user", "content": text}], st)
    assert a.decision == expected and a.plugin_decision == "send"
    if mode == "warn":
        assert a.confirm_types == ["iban"]
        assert privacy.analyze([{"content": text}], st, confirm=True).decision == "send"
        st.confirmed = ["iban"]
        assert privacy.analyze([{"content": text}], st).decision == "send"
    assert all(f.type == "iban" for f in a.findings)


@pytest.mark.parametrize("damage", ["size", "hash", "missing", "runtime", "inference"])
def test_guard_failure_never_sends_masked_rules_without_confirmation(tmp_path, monkeypatch, damage):
    model, blobs = _tiny_model(tmp_path)
    if damage == "size":
        blobs["m.onnx"] = b"on"
    elif damage == "hash":
        blobs["m.onnx"] = b"ONNX"
    elif damage == "missing":
        del blobs["config.json"]
    _install(model, blobs, tmp_path)
    monkeypatch.setattr(pm, "runtime_available", lambda: True)
    if damage == "runtime":
        def broken_runtime():
            raise ImportError("runtime incompatible")
        monkeypatch.setattr(pm, "runtime_available", broken_runtime)
    monkeypatch.setattr(pm, "OnnxTokenClassifier", lambda *a, **k: FakeClassifier({}, fail=damage == "inference"))
    monkeypatch.setattr(pm, "_default", pm.PrivacyScanner(model=model, home=tmp_path, calibration=CAL))
    a = privacy.analyze([{"content": "carte 4111 1111 1111 1111"}])
    assert a.decision == "ask" and a.confirm_types == ["other"] and "CARTE_1" in a.messages[0]["content"]


def test_missing_runtime_is_optional_but_corruption_is_not(tmp_path, monkeypatch):
    model, blobs = _tiny_model(tmp_path)
    _install(model, blobs, tmp_path)
    monkeypatch.setattr(pm, "runtime_available", lambda: False)
    s = pm.PrivacyScanner(model=model, home=tmp_path)
    assert s.scan("texte anodin")["model_decision"] == "send" and not s.broken
    (pm.model_dir(model, tmp_path) / model.onnx.file).write_bytes(b"on")
    assert s.scan("texte anodin")["model_decision"] == "ask" and s.broken


def test_unvalidated_integration_holds_nonempty_text(tmp_path, monkeypatch):
    s = pm.PrivacyScanner(home=tmp_path, classifier=FakeClassifier({}))
    monkeypatch.setattr(pm, "_default", s)
    r = s.scan("texte anodin")
    assert not r["guarantee"] and r["model_decision"] == "ask" and r["thresholds"]["low"] == 0
    assert privacy.analyze([{"content": "texte anodin"}]).decision == "ask"
    assert privacy.analyze([{"content": "texte anodin"}], confirm=True).decision == "send"


def test_model_only_decision_does_not_repeat_rule_policy(tmp_path, monkeypatch):
    s = scanner(tmp_path, classifier=FakeClassifier({}))
    monkeypatch.setattr(pm, "_default", s)
    text = "carte 4111 1111 1111 1111"
    assert s.scan(text)["decision"] == "local"
    assert privacy.analyze([{"content": text}]).decision == "mask"


def test_bad_policy(tmp_path):
    with pytest.raises(ValueError):
        scanner(tmp_path, policy="yolo")


def test_merge_tokens_and_fold():
    text = "Jean-Pierre  Dupont habite ici"
    toks = [(0, 4, 0.9, "PERSON"), (4, 5, 0.85, "PERSON"), (5, 11, 0.95, "PERSON"), (13, 19, 0.6, "PERSON"),
            (20, 26, 0.1, "ADDRESS")]
    spans = pm.merge_tokens(toks, 0.5, text)
    assert [(s["start"], s["end"]) for s in spans] == [(0, 11), (13, 19)] and spans[0]["score"] == 0.95
    folded, idx = pm.fold("Éloïse ﬁt")
    assert folded == "eloise fit" and idx[-2] == idx[-3] == 7 and len(idx) == len(folded)


# ---------------------------------------------------------------- the ONNX token classifier with a fake session

class _Input:
    def __init__(self, name):
        self.name = name


class FakeSession:
    """Logits favour label 1 (B-GIVEN_NAME) on token id `hot`, label 0 (O) elsewhere; records batches."""

    def __init__(self, hot: int, n_labels: int = 3):
        self.hot, self.n, self.calls = hot, n_labels, []

    def get_inputs(self):
        return [_Input("input_ids"), _Input("attention_mask")]

    def run(self, _, feed):
        import numpy as np
        ids = feed["input_ids"]
        self.calls.append(ids.shape)
        out = np.zeros(ids.shape + (self.n,), dtype=np.float32)
        out[..., 0] = 4.0
        out[ids == self.hot, 1] = 8.0
        return [out]


def test_onnx_token_classifier_windows(tmp_path):
    pytest.importorskip("numpy")
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "bonjour": 4, "camille": 5, "et": 6, "salut": 7}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tok.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "config.json").write_text(json.dumps({"id2label": {"0": "O", "1": "B-GIVEN_NAME", "2": "B-CITY"}}))
    sess = FakeSession(hot=5)
    clf = pm.OnnxTokenClassifier(tmp_path / "x.onnx", tmp_path / "tokenizer.json", tmp_path / "config.json",
                                 max_len=8, overlap=2, batch=2, session=sess)
    text = " ".join(["bonjour"] * 9 + ["camille"] + ["et", "salut"] * 5)
    toks = clf.token_scores(text)
    assert len(toks) == 20 and len(sess.calls) >= 2  # several windows, batched by 2
    hot = [t for t in toks if t[2] > 0.5]
    assert len(hot) == 1 and text[hot[0][0]:hot[0][1]] == "camille" and hot[0][3] == "PERSON"
    assert all(w[1] - w[0] <= 6 for w in clf.windows(20)) and clf.windows(20)[-1][1] == 20
    # out-of-scope labels (CITY) do not count
    assert max(t[2] for t in toks if t != hot[0]) < 0.05


# ---------------------------------------------------------------- download through the verified downloader

async def test_ensure_model_downloads_and_verifies(tmp_path):
    blobs = {"m.onnx": b"onnx-bytes", "tokenizer.json": b"{}", "config.json": b'{"id2label": {}}'}

    def f(name):
        b = blobs[name]
        return pm.PiiFile(name, len(b), hashlib.sha256(b).hexdigest())
    model = replace(pm.PII_CATALOG[pm.DEFAULT_PII_MODEL], onnx=f("m.onnx"), tokenizer=f("tokenizer.json"),
                    config=f("config.json"))
    seen = []

    def handler(req: httpx.Request):
        seen.append(str(req.url))
        return httpx.Response(200, content=blobs[req.url.path.rsplit("/", 1)[1]])
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        d = await pm.ensure_model(model, home=tmp_path, client=client)
    assert pm.is_installed(model, tmp_path) and (d / "m.onnx").read_bytes() == b"onnx-bytes"
    assert all(model.revision in u for u in seen)
    bad = replace(model, onnx=pm.PiiFile("m.onnx", len(blobs["m.onnx"]), "0" * 64))
    (d / "m.onnx").unlink()
    from myriad.downloader import DownloadError
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DownloadError):
            await pm.ensure_model(bad, home=tmp_path, client=client)
    assert not pm.is_installed(bad, tmp_path)


def test_catalog_is_pinned_and_permissive():
    for m in pm.PII_CATALOG.values():
        assert len(m.revision) == 40 and m.licence in ("mit", "apache-2.0", "bsd-3-clause")
        for f in m.files():
            assert len(f.sha256) == 64 and f.size > 0 and m.revision in m.url(f)
        assert 0 <= m.calibration.lambda_low <= m.calibration.lambda_high <= 1


# ---------------------------------------------------------------- conformal maths

def test_conformal_k_and_thresholds():
    assert conformal.marginal_k(99, 0.01) == 1 and conformal.marginal_k(98, 0.01) == 0
    assert conformal.min_n(0.01) == 99
    assert conformal.marginal_threshold([0.5, 0.1, 0.9] * 40, 0.01) == 0.1  # k = 1 -> smallest
    assert conformal.marginal_threshold([0.3] * 50, 0.01) == 0.0  # too few documents: flag everything
    n = conformal.min_n(0.01, 0.05)
    assert n == 299  # (1 - 0.01)^n <= 0.05
    assert conformal.pac_k(n, 0.01, 0.05) == 1 and conformal.pac_k(n - 1, 0.01, 0.05) == 0


def test_clopper_pearson_known_values():
    lo, hi = conformal.clopper_pearson(0, 100)
    assert lo == 0.0 and abs(hi - 0.03621669) < 1e-6
    lo, hi = conformal.clopper_pearson(5, 100)
    assert abs(lo - 0.01643) < 1e-4 and abs(hi - 0.11283) < 1e-4
    assert abs(conformal.binom_cdf(2, 10, 0.3) - 0.3827827864) < 1e-9


def test_conformal_marginal_guarantee_by_simulation():
    rng = random.Random(0)
    alpha, n, misses, trials = 0.05, 199, 0, 4000
    for _ in range(trials):
        cal = [rng.random() for _ in range(n)]
        lam = conformal.marginal_threshold(cal, alpha)
        misses += rng.random() < lam
    rate = misses / trials
    assert rate <= alpha + 3 * (alpha * (1 - alpha) / trials) ** 0.5  # k/(n+1) = 0.05 exactly here
