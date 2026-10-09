"""Unified documents for the PII benchmark: {id, source, lang, text, spans[{start, end, type}]}.

Gold labels of every dataset are mapped to Myriad's canonical types (privacy_model.TYPES). Labels that
are not personal data on their own (city, country, company, date, time, age, gender, occupation, URL,
bank codes...) are dropped: a document is *sensitive* when it holds at least one in-scope span.

Splits are fixed by a hash of the document group: train 30 % (probe training, choices), cal 30 %
(conformal calibration), test 40 % (reported numbers). dev = train + cal. The group is the Nemotron uid
(several renderings of one record) or the hand-made template, the document id otherwise.

gretel's spans were produced automatically and are noisy outside English (inspection: « Damen »,
« Lizenznehmer », « Dieser Hausfrachtbrief », « [E-Mail] » labelled as names or e-mails; documents
whose only « personal data » is such a word). A 1 % miss-rate target is dominated by such label errors,
so gretel is kept out of the calibration and of the guaranteed test numbers (`guarantee` = False) and
reported apart as a robustness check; its negatives still count in the false-positive rates."""
from __future__ import annotations

import ast
import hashlib
import json
import random

from .handmade import build as build_handmade
from .registry import DATASET_PATH, DATASETS

TYPES = ("PERSON", "EMAIL", "PHONE", "ADDRESS", "ID", "FINANCIAL", "SECRET", "IP", "USERNAME", "DATE_OF_BIRTH")

MAP = {
    # Nemotron-PII / gretel (snake case)
    "first_name": "PERSON", "last_name": "PERSON", "name": "PERSON",
    "email": "EMAIL", "phone_number": "PHONE", "fax_number": "PHONE",
    "street_address": "ADDRESS", "coordinate": "ADDRESS", "local_latlng": "ADDRESS",
    "ssn": "ID", "national_id": "ID", "tax_id": "ID", "passport_number": "ID", "driver_license_number": "ID",
    "certificate_license_number": "ID", "medical_record_number": "ID", "health_plan_beneficiary_number": "ID",
    "customer_id": "ID", "employee_id": "ID", "license_plate": "ID", "vehicle_identifier": "ID",
    "device_identifier": "ID", "biometric_identifier": "ID", "unique_id": "ID", "mac_address": "ID",
    "account_number": "FINANCIAL", "credit_debit_card": "FINANCIAL", "credit_card_number": "FINANCIAL",
    "iban": "FINANCIAL", "bban": "FINANCIAL", "cvv": "FINANCIAL", "credit_card_security_code": "FINANCIAL",
    "pin": "SECRET", "account_pin": "SECRET", "password": "SECRET", "api_key": "SECRET", "http_cookie": "SECRET",
    "ipv4": "IP", "ipv6": "IP", "user_name": "USERNAME", "date_of_birth": "DATE_OF_BIRTH",
    # ai4privacy openpii (upper case)
    "GIVENNAME": "PERSON", "SURNAME": "PERSON", "EMAIL": "EMAIL", "TELEPHONENUM": "PHONE",
    "STREET": "ADDRESS", "BUILDINGNUM": "ADDRESS", "IDCARDNUM": "ID", "TAXNUM": "ID", "DRIVERLICENSENUM": "ID",
    "SOCIALNUM": "ID", "PASSPORTNUM": "ID", "CREDITCARDNUMBER": "FINANCIAL", "ACCOUNTNUM": "FINANCIAL",
    "USERNAME": "USERNAME", "PASSWORD": "SECRET",
}

GRETEL_LANG = {"English": "en", "France": "fr", "German": "de", "Dutch": "nl", "Spanish": "es", "Italian": "it",
               "Swedish": "sv"}
OPENPII_QUOTA = {"en": 300, "fr": 300, "de": 120, "es": 120, "it": 120, "nl": 120, "pt": 120}
OPENPII_OTHER = 30
NEMOTRON_N = 1200
GRETEL_PER_LANG = 300
OASST_QUOTA = {"en": 500, "fr": 300}
OASST_OTHER = 40
SEED = 20261009
NOISY = {"gretel"}


def split_of(doc_id: str) -> str:
    h = int(hashlib.sha256(doc_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 30 else "cal" if h < 60 else "test"


def _canon(spans, label_key="label") -> list[dict]:
    out = []
    for s in spans:
        t = MAP.get(s[label_key])
        if t:
            out.append({"start": int(s["start"]), "end": int(s["end"]), "type": t})
    return out


def _nemotron(rng):
    import pyarrow.parquet as pq
    d = DATASETS["nemotron"]
    t = pq.read_table(DATASET_PATH("nemotron") / d["file"], columns=["uid", "text", "spans", "document_format"])
    rows = t.to_pylist()
    rng.shuffle(rows)
    for k, r in enumerate(rows[:NEMOTRON_N]):
        # a uid can appear several times (renderings of the same record): the draw index disambiguates
        yield {"id": f"nem-{r['uid']}-{k}", "source": "nemotron", "lang": "en", "text": r["text"],
               "spans": _canon(ast.literal_eval(r["spans"])), "fmt": r["document_format"], "group": f"nem-{r['uid']}"}


def _gretel(rng):
    import pyarrow.parquet as pq
    d = DATASETS["gretel"]
    rows = pq.read_table(DATASET_PATH("gretel") / d["file"]).to_pylist()
    rng.shuffle(rows)
    seen: dict[str, int] = {}
    for r in rows:
        lang = GRETEL_LANG.get(r["language"])
        if not lang or seen.get(lang, 0) >= GRETEL_PER_LANG:
            continue
        seen[lang] = seen.get(lang, 0) + 1
        yield {"id": f"gre-{r['index']}", "source": "gretel", "lang": lang, "text": r["generated_text"],
               "spans": _canon(json.loads(r["pii_spans"]))}


def _openpii(rng):
    rows = [json.loads(line) for line in open(DATASET_PATH("openpii") / "validation.sample.jsonl", encoding="utf-8")]
    rng.shuffle(rows)
    seen: dict[str, int] = {}
    for r in rows:
        lang = r["language"]
        if seen.get(lang, 0) >= OPENPII_QUOTA.get(lang, OPENPII_OTHER):
            continue
        seen[lang] = seen.get(lang, 0) + 1
        yield {"id": f"opi-{r['uid']}", "source": "openpii", "lang": lang, "text": r["source_text"],
               "spans": _canon(r["privacy_mask"])}


def _oasst(rng):
    import pyarrow.parquet as pq
    d = DATASETS["oasst"]
    rows = [r for r in pq.read_table(DATASET_PATH("oasst") / d["file"]).to_pylist()
            if r["role"] == "prompter" and r["parent_id"] is None and not r.get("deleted")]
    rng.shuffle(rows)
    seen: dict[str, int] = {}
    for r in rows:
        lang = r["lang"]
        if seen.get(lang, 0) >= OASST_QUOTA.get(lang, OASST_OTHER):
            continue
        seen[lang] = seen.get(lang, 0) + 1
        yield {"id": f"oas-{r['message_id']}", "source": "oasst", "lang": lang, "text": r["text"], "spans": [],
               "presumed": True}


def _humaneval(rng):
    import pyarrow.parquet as pq
    d = DATASETS["humaneval"]
    for r in pq.read_table(DATASET_PATH("humaneval") / d["file"]).to_pylist():
        yield {"id": f"hev-{r['task_id']}", "source": "humaneval", "lang": "code", "text": r["prompt"], "spans": []}


def load(sources: tuple[str, ...] = ("handmade", "nemotron", "gretel", "openpii", "oasst", "humaneval")) -> list[dict]:
    docs = []
    for src in sources:
        rng = random.Random(f"{SEED}-{src}")
        it = build_handmade() if src == "handmade" else globals()[f"_{src}"](rng)
        for d in it:
            d["split"] = split_of(d.get("group", d["id"]))  # renderings of one record stay together
            d["label"] = int(bool(d["spans"]))
            d["guarantee"] = src not in NOISY
            docs.append(d)
    assert len({d["id"] for d in docs}) == len(docs), "duplicate document ids"
    return docs
