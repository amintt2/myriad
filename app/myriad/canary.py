"""Canary jobs of the tracker (essaim/1.3): audits the peers cannot tell from real jobs.

With end-to-end encryption the tracker can no longer read user jobs, so it cannot duplicate them for a
spot check any more. It sends its OWN jobs instead: a short greedy math question with a checkable final
answer, encrypted like any job (same frame, a one-job pseudonym as requester, a token limit, a deadline
and a padded size copied from a real job just relayed to the same peer, sent after a random delay). The
same question goes to two peers serving the same model file; the tracker decrypts both answers and
compares their extracted answers exactly as the former spot checks did (same reputation logic).

Honeytokens: some canaries carry fake, unique, per-peer traceable details (an e-mail address, a URL
on the tracker's own domain, an API-key-looking string). The tracker records which peer received which
token; if one ever shows up elsewhere (POST /v1/honeytoken, or a request to the URL /h/<token>), the
peer that leaked it is identified. This detects a leak after the fact; it does not prevent it.

Indistinguishability is what makes the audit rate effective (docs/08_securite.md): at the frame level
it holds by construction (tests compare the frames); at the content level it is only heuristic, since
these questions come from templates. Honeytokens are kept to details that real (masked) questions also
contain: e-mails and URLs are not masked by default, while a raw API key would stand out next to real
questions whose keys are masked, so key-like tokens are rare.
"""
from __future__ import annotations

import random
import secrets
from urllib.parse import urlsplit

_FR = [
    ("Léa a {a} billes et en gagne {b} à la récréation. Combien de billes a-t-elle maintenant ?", lambda a, b: a + b),
    ("Un libraire reçoit {a} cartons de {b} livres. Combien de livres reçoit-il en tout ?", lambda a, b: a * b),
    ("Un train transporte {a} passagers ; {b} descendent à la première gare. Combien en reste-t-il ?",
     lambda a, b: a - b),
    ("Une recette demande {b} œufs par gâteau. Combien d'œufs faut-il pour {a} gâteaux ?", lambda a, b: a * b),
    ("Paul économise {b} euros par semaine pendant {a} semaines. Combien a-t-il économisé ?", lambda a, b: a * b),
    ("Dans une salle de {a} places, {b} sont occupées. Combien de places sont libres ?", lambda a, b: a - b),
]
_EN = [
    ("Tom has {a} apples and buys {b} more. How many apples does he have?", lambda a, b: a + b),
    ("A shop sells {a} boxes of {b} pens. How many pens is that in total?", lambda a, b: a * b),
    ("A bus carries {a} people and {b} get off at the first stop. How many are left on the bus?", lambda a, b: a - b),
    ("Each crate holds {b} bottles. How many bottles are in {a} crates?", lambda a, b: a * b),
    ("Mia reads {b} pages a day for {a} days. How many pages does she read?", lambda a, b: a * b),
    ("A theater has {a} seats and {b} are taken. How many seats are free?", lambda a, b: a - b),
]
_CONTEXT_FR = ["", "C'est pour un exercice de ma fille. ", "Petite question rapide : ",
               "Je prépare une fiche pour ma classe. ", "Je vérifie un calcul : "]
_CONTEXT_EN = ["", "Quick question: ", "Helping my son with homework. ", "Checking a calculation: ",
               "For a worksheet I am preparing: "]
_HONEY_FR = ["Si besoin, mon adresse est {email}. ", "Le sujet est aussi sur {url} . ",
             "(Contact : {email}, fiche : {url}) "]
_HONEY_EN = ["My email is {email} if needed. ", "The worksheet is also at {url} . ",
             "(Contact: {email}, sheet: {url}) "]
_NAMES = ["camille", "lucas", "emma", "noah", "lea", "hugo", "chloe", "louis", "alice", "jules", "sam", "nina"]


def draw(rng: random.Random) -> dict:
    """Parameters of one canary question (shared by the two copies of a canary pair)."""
    lang = rng.choice(("fr", "en"))
    tmpl = rng.randrange(len(_FR if lang == "fr" else _EN))
    a, b = rng.randint(12, 480), rng.randint(2, 11)
    return {"lang": lang, "tmpl": tmpl, "a": a, "b": b, "ctx": rng.randrange(5),
            "honey": rng.random() < 0.5, "honey_tmpl": rng.randrange(3), "name": rng.choice(_NAMES),
            "key": rng.random() < 0.05}


def honeytokens(public_url: str | None, name: str = "alex") -> dict:
    """Fresh unique tokens for one copy of a canary (one peer). Without a public URL, only a key."""
    tok = secrets.token_hex(12)
    out = {"key": f"myr_live_{secrets.token_hex(16)}"}
    if public_url:
        u = urlsplit(public_url)
        host = u.hostname or ""
        if host and "." in host:
            out["url"] = f"{public_url.rstrip('/')}/h/{tok}"
            out["email"] = f"{name}.{tok[:10]}@{host}"
    return out


def question(p: dict, tokens: dict | None = None) -> tuple[str, str]:
    """(question text, expected final answer). `tokens`: honeytokens of this copy (or None)."""
    fr = p["lang"] == "fr"
    text, fn = (_FR if fr else _EN)[p["tmpl"]]
    body = (_CONTEXT_FR if fr else _CONTEXT_EN)[p["ctx"]] + text.format(a=p["a"], b=p["b"])
    if tokens and p["honey"] and "url" in tokens:
        body += " " + (_HONEY_FR if fr else _HONEY_EN)[p["honey_tmpl"]].format(email=tokens["email"],
                                                                               url=tokens["url"]).rstrip()
    if tokens and p["key"]:
        body += (" (clé de test : " if fr else " (test key: ") + tokens["key"] + ")"
    return body, str(fn(p["a"], p["b"]))


def used_tokens(p: dict, tokens: dict) -> dict:
    """The honeytokens actually placed in the question (the ones worth recording)."""
    out = {}
    if p["honey"] and "url" in tokens:
        out["url"], out["email"] = tokens["url"], tokens["email"]
    if p["key"]:
        out["key"] = tokens["key"]
    return out
