"""Detectors of the benchmark, behind one interface.

`run(text)` returns a dict with any of:
  tokens: [(start, end, score, type)]  token scores of span models (offsets in the original text)
  doc:    float                         document-level probability (bunker-laya)
  chunks: [(start, end, vector)]        sentence embeddings of the chunks (probes, trained later)

Token-classification models use the app runtime (myriad.privacy_model.OnnxTokenClassifier), so the
benchmark measures exactly what Myriad would run."""
from __future__ import annotations

import json
import re

import numpy as np

from myriad import privacy_rules
from myriad.privacy_model import OnnxTokenClassifier

from .registry import DETECTOR_PATH, MODELS


class Rules:
    name = "regex"

    def run(self, text: str) -> dict:
        return {"tokens": [(s["start"], s["end"], 1.0, s["type"]) for s in privacy_rules.find(text)]}


class TokCls:
    def __init__(self, name: str, threads: int | None = None):
        m, d = MODELS[name], DETECTOR_PATH(name)
        self.name = name
        self.clf = OnnxTokenClassifier(d / m["onnx"], d / m["tokenizer"], d / m["config"],
                                       normalize="lower_strip_accents" if m.get("normalize") else None,
                                       max_len=m["max_len"], threads=threads)

    def run(self, text: str) -> dict:
        return {"tokens": [t for t in self.clf.token_scores(text) if t[2] >= 0.005]}


def _session(path, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def _tokenizer(path):
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(str(path))
    t.no_truncation()
    t.no_padding()
    return t


# GLiNER zero-shot prompts -> canonical types
GLINER_LABELS = {"name": "PERSON", "email address": "EMAIL", "phone number": "PHONE", "location address": "ADDRESS",
                 "ssn": "ID", "passport number": "ID", "driver license number": "ID", "id number": "ID",
                 "credit card number": "FINANCIAL", "iban": "FINANCIAL", "account number": "FINANCIAL",
                 "password": "SECRET", "api key": "SECRET", "ip address": "IP", "username": "USERNAME",
                 "dob": "DATE_OF_BIRTH"}
WORD = re.compile(r"\w+(?:[-_]\w+)*|\S")


class Gliner:
    """GLiNER token-level ONNX graph (knowledgator/gliner-pii-small-v1.0), re-implemented from the
    published input contract: [CLS] (<<ENT>> label)* <<SEP>> words [SEP]; words_mask gives the
    1-based index of the word each first sub-token starts; logits are (batch, words, classes, 3) for
    start / end / inside. A word's score is max over classes of its inside probability (recall-first)."""
    name = "gliner-pii-small"

    def __init__(self, threads=None):
        m, d = MODELS[self.name], DETECTOR_PATH(self.name)
        self.sess = _session(d / m["onnx"], threads)
        self.tok = _tokenizer(d / m["tokenizer"])
        cfg = json.loads((d / m["config"]).read_text(encoding="utf-8"))
        self.ent = self.tok.token_to_id(cfg["ent_token"])
        self.sep = self.tok.token_to_id(cfg["sep_token"])
        special = self.tok.encode("", add_special_tokens=True).ids
        self.cls, self.end = special[:1], special[1:2]
        self.labels = list(GLINER_LABELS)
        self.prompt = []
        for lab in self.labels:
            self.prompt += [self.ent] + self.tok.encode(lab, add_special_tokens=False).ids
        self.prompt.append(self.sep)
        self.max_words = 200

    def run(self, text: str) -> dict:
        words = [(m.start(), m.end()) for m in WORD.finditer(text)]
        tokens = []
        for w0 in range(0, len(words), self.max_words - 20):
            ws = words[w0:w0 + self.max_words]
            ids, wmask = list(self.cls) + self.prompt, [0] * (len(self.cls) + len(self.prompt))
            for k, (a, z) in enumerate(ws):
                sub = self.tok.encode(text[a:z], add_special_tokens=False).ids or [self.tok.token_to_id("[UNK]") or 0]
                ids += sub
                wmask += [k + 1] + [0] * (len(sub) - 1)
            ids, wmask = ids[:1000] + self.end, wmask[:1000] + [0]
            feed = {"input_ids": np.array([ids], dtype=np.int64), "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
                    "words_mask": np.array([wmask], dtype=np.int64), "text_lengths": np.array([[len(ws)]], dtype=np.int64)}
            out = self.sess.run(None, feed)[0]
            p = 1 / (1 + np.exp(-out.astype(np.float32)))
            p = p.reshape(-1, p.shape[-3] if p.ndim == 4 else 1, len(self.labels), 3)[0] if p.shape[0] == 1 else p[:, 0]
            # p: (words, classes, 3)
            score = np.maximum(p[..., 2], np.minimum(p[..., 0], p[..., 1]))  # inside, or single-word span
            for k in range(min(len(ws), score.shape[0])):
                c = int(score[k].argmax())
                s = float(score[k, c])
                if s >= 0.005:
                    a, z = ws[k]
                    tokens.append((a, z, s, GLINER_LABELS[self.labels[c]]))
            if w0 + self.max_words >= len(words):
                break
        return {"tokens": tokens}


class BunkerLaya:
    """impacte/bunker-laya decision head (ModernBERT-large), `pii_present` question, rebuilt from the
    opencode-bunker onnx-local contract: [CLS] "noul question: <instructions>" [SEP]
    [MASK] "false: ..." [MASK] "true: ..." [SEP] <state> [SEP]; P(true) = softmax(logits / T[2])[1].
    Long states are cut into windows (max over windows)."""
    name = "bunker-laya"
    INSTR = "Does the text contain personally identifiable information (PII)?"

    def __init__(self, threads=None):
        m, d = MODELS[self.name], DETECTOR_PATH(self.name)
        self.sess = _session(d / m["onnx"], threads)
        self.tok = _tokenizer(d / m["tokenizer"])
        cfg = json.loads((d / m["config"]).read_text(encoding="utf-8"))
        self.T = cfg["temperature"][2]
        self.max_len = cfg["max_len"]
        cls = self.tok.token_to_id("[CLS]")
        sep = self.tok.token_to_id("[SEP]")
        mask = self.tok.token_to_id("[MASK]")
        enc = lambda s: self.tok.encode(s, add_special_tokens=False).ids  # noqa: E731
        head = [cls] + enc(f"noul question: {self.INSTR}") + [sep]
        self.markers = []
        for opt in ("false: no, the statement does not hold", "true: yes, the statement holds"):
            self.markers.append(len(head))
            head += [mask] + enc(opt)
        head.append(sep)
        self.head, self.sep = head, sep

    def run(self, text: str) -> dict:
        ids = self.tok.encode(text, add_special_tokens=False).ids
        room = self.max_len - len(self.head) - 1
        best = 0.0
        for s in range(0, max(len(ids), 1), room - 64 if len(ids) > room else room):
            seq = self.head + ids[s:s + room] + [self.sep]
            feed = {"input_ids": np.array([seq], dtype=np.int64), "attention_mask": np.ones((1, len(seq)), dtype=np.int64),
                    "marker_pos": np.array([self.markers], dtype=np.int64), "marker_mask": np.ones((1, 2), dtype=bool),
                    "qtype": np.array([2], dtype=np.int64)}
            logits = self.sess.run(["logits"], feed)[0][0].astype(np.float64) / self.T
            e = np.exp(logits - logits.max())
            best = max(best, float(e[1] / e.sum()))
            if s + room >= len(ids):
                break
        return {"doc": best}


class Embed:
    """Sentence embedding of each chunk (mean pooling, or the model's own pooled output)."""

    def __init__(self, name: str, threads=None):
        m, d = MODELS[name], DETECTOR_PATH(name)
        self.name, self.m = name, m
        self.sess = _session(d / m["onnx"], threads)
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.tok = _tokenizer(d / m["tokenizer"])
        special = self.tok.encode("", add_special_tokens=True).ids
        self.prefix_ids = self.tok.encode(m["prefix"], add_special_tokens=False).ids if m["prefix"] else []
        if name.startswith("embeddinggemma"):
            self.cls, self.end = special[:1], special[1:2] if len(special) > 1 else []
        else:
            self.cls, self.end = special[:1], special[1:2]
        self.room = 256 - len(self.cls) - len(self.end) - len(self.prefix_ids)

    def run(self, text: str) -> dict:
        enc = self.tok.encode(text, add_special_tokens=False)
        ids, offs = enc.ids, enc.offsets
        chunks = []
        step = self.room - 32
        for s in range(0, max(len(ids), 1), step):
            part = ids[s:s + self.room]
            seq = self.cls + self.prefix_ids + part + self.end
            feed = {"input_ids": np.array([seq], dtype=np.int64), "attention_mask": np.ones((1, len(seq)), dtype=np.int64)}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros((1, len(seq)), dtype=np.int64)
            for k in ("image_features", "video_features", "audio_features"):
                if k in self.inputs:
                    feed[k] = np.zeros((0, 512), dtype=np.float32)
            if self.m["pooling"] == "model":
                vec = self.sess.run(["sentence_embedding"], feed)[0][0]
            else:
                h = self.sess.run(None, feed)[0][0]
                vec = h.mean(0)
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            a = offs[s][0] if part else 0
            z = offs[min(s + self.room, len(ids)) - 1][1] if part else len(text)
            chunks.append((a, z, vec.astype(np.float32)))
            if s + self.room >= len(ids):
                break
        return {"chunks": chunks}


def make(name: str, threads: int | None = None):
    if name == "regex":
        return Rules()
    kind = MODELS[name]["kind"]
    if kind == "tokcls":
        return TokCls(name, threads)
    if kind == "gliner":
        return Gliner(threads)
    if kind == "laya":
        return BunkerLaya(threads)
    return Embed(name, threads)
