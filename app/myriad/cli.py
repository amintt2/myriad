"""Command line: myriad init | node | tracker | chat | agents run | status | update (`essaim` is a deprecated
alias)."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import httpx

from . import __version__
from .config import DEFAULT_MODEL, Config, home_dir, key_path
from .crypto import Identity
from .priors import family_of, params_of


def _home(args) -> Path:
    return Path(args.home).expanduser() if getattr(args, "home", None) else home_dir()


def parse_model(spec: str) -> tuple[str, str]:
    """'owner/repo:file.gguf' -> ('owner/repo', 'file.gguf')."""
    repo, sep, fname = spec.partition(":")
    if not sep or "/" not in repo or not fname.lower().endswith(".gguf") or ".." in fname or fname.startswith("/"):
        raise SystemExit(f"Modèle invalide : {spec!r}. Format attendu : dépôt/nom:fichier.gguf "
                         f"(exemple : {DEFAULT_MODEL})")
    return repo, fname


def download_model(repo: str, fname: str) -> str:
    from huggingface_hub import hf_hub_download

    print(f"Téléchargement de {fname} depuis {repo} (Hugging Face)…")
    return hf_hub_download(repo_id=repo, filename=fname)


def cmd_init(args) -> int:
    home = _home(args)
    home.mkdir(parents=True, exist_ok=True)
    kp = key_path(home)
    existed = kp.exists()
    ident = Identity.load_or_create(kp)
    cfg = Config.load(home)
    if args.tracker:
        cfg.tracker_url = args.tracker
    if args.llama_server:
        cfg.llama_server = args.llama_server
    if args.max_parallel is not None:
        cfg.max_parallel = args.max_parallel
    if args.no_model:
        cfg.model = cfg.gguf_path = cfg.family = cfg.params_b = None
    elif args.model or not cfg.model:
        spec = args.model or DEFAULT_MODEL
        repo, fname = parse_model(spec)
        cfg.model = f"{repo}:{fname}"
        cfg.family, cfg.params_b = family_of(repo), params_of(repo)
        cfg.gguf_path = None if args.no_download else download_model(repo, fname)
    elif cfg.model and not cfg.gguf_path and not args.no_download:
        cfg.gguf_path = download_model(*parse_model(cfg.model))
    path = cfg.save(home)
    print(f"Dossier Myriad : {home}")
    print(f"Clé du nœud    : {kp} ({'existante' if existed else 'créée'})")
    print(f"Identifiant    : {ident.node_id}")
    print(f"Configuration  : {path}")
    print(f"Traqueur       : {cfg.tracker_url}")
    print(f"Modèle         : {cfg.model or 'aucun (client seulement)'}" + (f" -> {cfg.gguf_path}" if cfg.gguf_path else ""))
    if cfg.model and not cfg.gguf_path:
        print("  (pas encore téléchargé : relancez myriad init sans --no-download)")
    try:
        from .engine import resolve_binary
        print(f"llama-server   : {resolve_binary(cfg.llama_server)}")
    except Exception as e:
        if cfg.model:
            print(f"Attention : {e}")
    print("Ensuite : myriad node")
    return 0


def cmd_node(args) -> int:
    from .service import run_node

    home = _home(args)
    cfg = Config.load(home)
    if args.tracker:
        cfg.tracker_url = args.tracker
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        asyncio.run(run_node(cfg, home, serve=not args.no_serve))
    except KeyboardInterrupt:
        pass
    return 0


def cmd_bans(args) -> int:
    """`myriad tracker-bans`: the bans a tracker recorded (honeytoken sightings, admin), and lift one."""
    from .ledger import Ledger

    led = Ledger(args.db)
    try:
        if args.remove:
            print("levé" if led.remove_ban(args.remove) else "aucun bannissement pour ce nœud")
        for nid, reason in sorted(led.bans().items()):
            print(f"{nid}  {reason}")
    finally:
        led.close()
    return 0


def cmd_tracker(args) -> int:
    from .tracker import run

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from .netem import wan_from_rtt

    kw = {}
    if args.release_repo is not None:  # else MYRIAD_RELEASE_REPO, default amintt2/myriad ("off": none)
        repo = args.release_repo.strip()
        kw["release_repo"] = None if repo.lower() in ("", "off", "none", "0") else repo
    run(host=args.host, port=args.port, db=args.db, starter_credit=args.starter_credit, spot_rate=args.spot_rate,
        receipt_grace_s=args.receipt_grace, wan=wan_from_rtt(args.wan_rtt_ms, args.wan_sigma),
        landing=not args.no_landing, public_url=args.public_url, ban_file=args.ban_file, **kw)
    return 0


def cmd_chat(args) -> int:
    cfg = Config.load(_home(args))
    base = args.gateway or f"http://127.0.0.1:{cfg.gateway_port}"
    # Understood by gateways from before the rename too: they know neither the model id "myriad" nor the
    # field "myriad", so the swarm is asked by leaving the model out, and the options go under both names.
    options = {"k": args.k, "task_hint": args.hint}
    body = {"messages": [{"role": "user", "content": args.question}], "max_tokens": args.max_tokens,
            "myriad": options, "essaim": options}
    if args.model not in ("myriad", "essaim"):
        body["model"] = args.model
    try:
        r = httpx.post(f"{base}/v1/chat/completions", json=body, timeout=args.timeout)
    except httpx.HTTPError as e:
        print(f"Passerelle injoignable ({base}) : {e}. Le nœud tourne-t-il (myriad node) ?", file=sys.stderr)
        return 2
    j = r.json()
    if r.status_code != 200:
        print(f"Erreur {r.status_code} : {j.get('error', {}).get('message', j)}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(j, ensure_ascii=False, indent=2))
        return 0
    print(j["choices"][0]["message"]["content"])
    m = j.get("myriad") or j.get("essaim") or {}  # "essaim": gateways before the rename
    rule = {"vote": "vote pondéré", "medoid": "médoïde", "single": "un seul pair"}.get(m.get("decision"), m.get("decision"))
    print(f"\n[{rule}" + (f", réponse {m['answer']}" if m.get("answer") else "")
          + f", {m.get('peers_answered')}/{m.get('peers_asked')} pairs, {m.get('latency_ms')} ms"
          + (", arrêt anticipé" if m.get("early_stop") else "") + "]")
    if args.details:
        for p in m.get("peers", []):
            mark = "*" if p.get("chosen") else " "
            print(f" {mark} {p['model']:<45} {p['status']:<12} réponse={p.get('answer')!s:<8} poids={p['weight']:<7}"
                  f" {p.get('latency_ms') or '-'} ms" + (f"  ({p['error']})" if p.get("error") else ""))
    return 0


def cmd_agents_run(args) -> int:
    """`myriad agents run plan.json`: sub-tasks in parallel on the network, progress on stderr."""
    import shlex
    import time as _time

    from .agents import AgentsError, run_remote

    cfg = Config.load(_home(args))
    base = args.gateway or f"http://127.0.0.1:{cfg.gateway_port}"
    body: dict = {}
    if args.plan:
        try:
            data = json.loads(Path(args.plan).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as e:
            print(f"Plan illisible ({args.plan}) : {e}", file=sys.stderr)
            return 2
        body = {"plan": data} if isinstance(data, list) else dict(data)
    if args.task:
        body["task"] = args.task
    if args.auto or "plan" not in body:
        body["plan"] = "auto"
    allow = dict(body.get("allow_commands") or {})
    for spec in args.allow:
        name, sep, cmd = spec.partition("=")
        if not sep or not name or not cmd.strip():
            print(f"--allow attend NOM=COMMANDE, reçu : {spec!r}", file=sys.stderr)
            return 2
        if os.name == "nt":  # keep backslashes; "C:\Program Files\..." quoted, quotes removed
            allow[name] = [a[1:-1] if len(a) >= 2 and a[0] == a[-1] and a[0] in "\"'" else a
                           for a in shlex.split(cmd, posix=False)]
        else:
            allow[name] = shlex.split(cmd)
    if allow:
        body["allow_commands"] = allow
    t0 = _time.monotonic()

    def show(ev: dict) -> None:
        if args.quiet:
            return
        k, sec = ev.get("type"), f"[{_time.monotonic() - t0:6.1f} s]"
        st = ev.get("subtask") if isinstance(ev.get("subtask"), dict) else None
        if k == "plan":
            print(f"{sec} plan ({ev['source']}) : {len(ev['subtasks'])} sous-tâches", file=sys.stderr)
        elif k == "job":
            print(f"{sec} {ev['subtask']} -> {ev.get('family') or '?'} ({(ev.get('model') or '?').split('/')[-1]})",
                  file=sys.stderr)
        elif k == "verified":
            print(f"{sec} {ev['subtask']} vérification {'OK' if ev['passed'] else 'ÉCHEC'} "
                  f"({ev.get('model') or '?'})", file=sys.stderr)
        elif k == "escalate":
            print(f"{sec} {ev['subtask']} escalade : {ev.get('reason')}", file=sys.stderr)
        elif k == "subtask" and st and st["status"] in ("ok", "failed", "skipped"):
            print(f"{sec} {st['id']} {st['status']}" + (f" ({st['error']})" if st.get("error") else "")
                  + (f", {st['tokens']} jetons" if st.get("tokens") else ""), file=sys.stderr)

    try:
        res = run_remote(body, base, show)
    except AgentsError as e:
        print(f"Erreur : {e.message}", file=sys.stderr)
        return 1
    except httpx.HTTPError as e:
        print(f"Passerelle injoignable ({base}) : {e}. Le nœud tourne-t-il (myriad node) ?", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print(res["result"])
        tm = res["timing"]
        print(f"\n[{res['status']} : {res['usage']['ok']}/{res['usage']['subtasks']} sous-tâches, "
              f"{res['usage']['completion_tokens']} jetons, {tm['wall_ms'] / 1000:.1f} s "
              f"(somme des sous-tâches {tm['subtasks_sum_ms'] / 1000:.1f} s)]", file=sys.stderr)
    return 0 if res["status"] == "ok" else 1


def cmd_status(args) -> int:
    home = _home(args)
    cfg = Config.load(home)
    try:
        r = httpx.get(f"http://127.0.0.1:{cfg.gateway_port}/v1/myriad/status", timeout=5)
        if r.status_code == 404:  # a gateway from before the rename
            r = httpx.get(f"http://127.0.0.1:{cfg.gateway_port}/v1/essaim/status", timeout=5)
        s = r.json()
        n = s["node"]
        print(f"Nœud {n['node_id']} : {n['state']} ({n['tracker']})")
        print(f"  modèle : {n['model'] or 'aucun'} ; moteur : {(n['engine'] or {}).get('state', '-')}")
        print(f"  accepte des jobs : {'oui' if n['accepting_now'] else 'non'} ; jobs en cours : {n['running']}/{n['max_parallel']}")
        st = n["stats"]
        print(f"  servis : {st['served']} ; échecs : {st['failed']} ; annulés : {st['cancelled']} ; jetons : {st['tokens']}")
        print(f"  crédits : {s['balance']}")
        if n.get("last_error"):
            print(f"  dernière erreur : {n['last_error']}")
        return 0
    except (httpx.HTTPError, KeyError, ValueError):
        pass
    print("Le nœud ne tourne pas (myriad node pour le lancer).")
    kp = key_path(home)
    if not kp.exists():
        print(f"Aucune clé dans {home} : lancez myriad init.")
        return 1
    ident = Identity.load(kp)
    print(f"Identifiant : {ident.node_id}\nModèle : {cfg.model or 'aucun'}\nTraqueur : {cfg.tracker_url}")
    from .node import http_url
    try:
        from .crypto import account_params
        r = httpx.get(f"{http_url(cfg.tracker_url)}/v1/balance/{ident.node_id}", params=account_params(ident), timeout=5)
        if r.status_code == 200:
            print(f"Crédits : {r.json()['balance']}")
    except httpx.HTTPError:
        print("Traqueur injoignable.")
    return 0


def cmd_update(args) -> int:
    """`myriad update [--check]`: is there a newer Myriad? The tracker answers (GitHub if it cannot);
    an installed desktop build downloads and verifies the update, then installs it on the next restart
    of the app; other installs get the link and the command to run."""
    from .updater import Updater

    home = _home(args)
    cfg = Config.load(home)
    tracker = args.tracker or cfg.tracker_url

    async def go() -> tuple[dict, Updater]:
        up = Updater(home, tracker_url=tracker, auto_update=False)
        try:
            st = await up.check(force=True)
            if not args.check and st["available"] and up.kind.asset(st["latest"]) and up.kind.can_apply:
                await up.download_now()
                await up.wait_download()
                st = up.status()
            return st, up
        finally:
            await up.close()

    st, up = asyncio.run(go())
    print(f"Version installée : {st['current']} ({st['kind']})")
    if st["state"] == "error" and not st["available"]:
        print(f"Vérification impossible : {st['error']}", file=sys.stderr)
        return 2
    if not st["available"]:
        print("Myriad est à jour.")
        return 0
    print(f"Nouvelle version  : {st['latest']} (source : {st['source']})")
    print(f"Notes de version  : {st['notes_url']}")
    if args.check:
        return 0
    if st["state"] == "ready":
        print(f"Téléchargée et vérifiée (SHA-256) : {up.file}")
        print("Elle s'installe depuis l'application : « Mettre à jour et redémarrer ».")
        return 0
    if st["state"] == "error":
        print(f"Téléchargement impossible : {st['error']}", file=sys.stderr)
        return 1
    hints = {"source": "Depuis les sources : git pull puis uv sync (ou pip install -U myriad).",
             "linux-deb": f"Paquet .deb : télécharger myriad_{st['latest']}_amd64.deb depuis la page ci-dessus, "
                          "puis sudo apt install ./myriad_….deb",
             "linux-archive": "Archive : télécharger la nouvelle archive depuis la page ci-dessus.",
             "windows-portable": "Version portable : télécharger le nouveau zip depuis la page ci-dessus."}
    print(hints.get(st["kind"]) or (st.get("reason") or "Télécharger la nouvelle version depuis la page ci-dessus."))
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="myriad", description="Myriad : LLM décentralisé (un nœud par PC).")
    ap.add_argument("--version", action="version", version=f"myriad {__version__}")
    ap.add_argument("--home", help="dossier de configuration (défaut : dossier utilisateur, ou MYRIAD_HOME)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="créer la clé et la configuration, télécharger le modèle")
    p.add_argument("--model", help=f"modèle GGUF dépôt:fichier (défaut : {DEFAULT_MODEL})")
    p.add_argument("--no-model", action="store_true", help="client seulement : ne sert aucun modèle")
    p.add_argument("--tracker", help="URL du traqueur (http://hôte:8500 ou https://…)")
    p.add_argument("--llama-server", help="chemin de llama-server (sinon LLAMA_SERVER ou le PATH)")
    p.add_argument("--max-parallel", type=int, help="jobs simultanés au plus")
    p.add_argument("--no-download", action="store_true", help="ne pas télécharger le modèle maintenant")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("node", help="lancer le nœud, la passerelle (8400) et l'interface (8401)")
    p.add_argument("--tracker", help="URL du traqueur (remplace la configuration)")
    p.add_argument("--no-serve", action="store_true", help="ne sert pas de modèle (client seulement)")
    p.set_defaults(func=cmd_node)

    p = sub.add_parser("tracker", help="lancer le traqueur (rendez-vous, relais, crédits)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8500)
    p.add_argument("--db", default="tracker.sqlite")
    p.add_argument("--starter-credit", type=float, default=1000.0)
    p.add_argument("--spot-rate", type=float, default=0.05,
                   help="taux d'audit : part des jobs suivis d'un job canari (chiffrés) ou dupliqués (en clair)")
    p.add_argument("--public-url", default=None,
                   help="URL publique du traqueur (adresses et liens des jetons-pièges des canaris)")
    p.add_argument("--ban-file", default=None,
                   help="fichier JSON des nœuds bannis par l'administrateur (relu quand il change)")
    p.add_argument("--receipt-grace", type=float, default=60.0, help="délai avant règlement sans reçu (s)")
    p.add_argument("--wan-rtt-ms", type=float, default=0.0,
                   help="EXPÉRIENCES SEULEMENT : retard réseau simulé, aller-retour médian client-traqueur (ms) ; 0 = désactivé")
    p.add_argument("--wan-sigma", type=float, default=0.25, help="dispersion lognormale du retard simulé")
    p.add_argument("--no-landing", action="store_true", help="ne pas servir la page d'accueil publique sur /")
    p.add_argument("--release-repo", default=None,
                   help="dépôt GitHub dont les versions sont annoncées aux nœuds (défaut : MYRIAD_RELEASE_REPO, "
                        "sinon amintt2/myriad ; « off » : aucun)")
    p.set_defaults(func=cmd_tracker)

    p = sub.add_parser("tracker-bans", help="bannissements enregistrés par un traqueur (jetons-pièges) ; en lever un")
    p.add_argument("--db", default="tracker.sqlite")
    p.add_argument("--remove", metavar="NODE_ID", help="lever le bannissement de ce nœud")
    p.set_defaults(func=cmd_bans)

    p = sub.add_parser("chat", help="poser une question à l'essaim par la passerelle locale")
    p.add_argument("question")
    p.add_argument("--k", type=int, default=None, help="nombre de pairs (familles différentes)")
    p.add_argument("--hint", choices=["math", "mc", "free"], default=None, help="type de tâche (défaut : automatique)")
    p.add_argument("--model", default="myriad", help="« myriad » (le réseau) ou un modèle précis du réseau")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--gateway", help="URL de la passerelle (défaut : http://127.0.0.1:8400)")
    p.add_argument("--details", action="store_true", help="afficher la réponse de chaque pair")
    p.add_argument("--json", action="store_true", help="réponse brute (JSON)")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("agents", help="sous-agents : des sous-tâches en parallèle sur le réseau")
    asub = p.add_subparsers(dest="agents_cmd", required=True)
    r = asub.add_parser("run", help="exécuter un plan (fichier JSON) ou un plan automatique (--task)")
    r.add_argument("plan", nargs="?", help="fichier JSON : requête complète, ou liste de sous-tâches")
    r.add_argument("--task", help="la tâche (obligatoire pour un plan automatique)")
    r.add_argument("--auto", action="store_true", help="plan écrit par un pair orchestrateur")
    r.add_argument("--allow", action="append", default=[], metavar="NOM=COMMANDE",
                   help="autoriser une commande de vérification locale pour cette requête (répétable)")
    r.add_argument("--gateway", help="URL de la passerelle (défaut : http://127.0.0.1:8400)")
    r.add_argument("--json", action="store_true", help="résultat complet (JSON)")
    r.add_argument("--quiet", action="store_true", help="sans la progression")
    r.set_defaults(func=cmd_agents_run)

    p = sub.add_parser("status", help="état du nœud")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("update", help="rechercher une nouvelle version de Myriad")
    p.add_argument("--check", action="store_true", help="vérifier seulement, sans rien télécharger")
    p.add_argument("--tracker", help="URL du traqueur à interroger (défaut : celui de la configuration)")
    p.set_defaults(func=cmd_update)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
