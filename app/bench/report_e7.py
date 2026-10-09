"""Markdown report of E7 (results/e7_report.md), generated from the JSON files: no number is typed by hand."""
from __future__ import annotations

import json
from pathlib import Path


def _load(d: Path, name: str) -> dict | None:
    p = d / f"e7_{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f(x, nd: int = 2, pct: bool = False) -> str:
    if x is None:
        return "–"
    if pct:
        return f"{100 * x:.1f} %"
    return f"{x:.{nd}f}"


def ms(x) -> str:
    return "–" if x is None else f"{1000 * x:.0f}"


def gain(slow, fast) -> float | None:
    """Relative latency reduction 1 - fast / slow; None when a value is missing."""
    if slow is None or fast is None or slow <= 0:
        return None
    return 1 - fast / slow


def busy_share(ph: dict) -> float | None:
    st = ph["peer_status"]
    total = sum(st.values())
    return ph["peer_errors"].get("peer_busy", 0) / total if total else None


def sustained(phases: list[dict]) -> dict | None:
    """Highest offered step served without saturation (same criterion as the run)."""
    from bench.bench_scale import saturated

    best = None
    for ph in phases:
        if saturated(ph, phases[0]):
            break
        best = ph
    return best


def write_report(d: Path) -> Path:
    d = Path(d)
    lat, sca, tpu, fai, dco = (_load(d, n) for n in ("latency", "scale", "throughput", "failure", "dircost"))
    any_meta = next((x for x in (lat, sca, tpu, fai, dco) if x), None)
    # Cost of one /v1/peers answer per N, from the dedicated measurement (plan dircost), used for the
    # other plans only when it was made on the same machine (otherwise their own probe is shown).
    dir_ms = {s["scenario"]["nodes"]: s["directory_cost"] for s in (dco or {}).get("scenarios", [])}

    def same_machine(plan: dict | None) -> bool:
        return bool(dco and plan and dco["machine"] == plan["machine"])

    def dir_cost(n: int, fallback: dict, plan: dict | None) -> float | None:
        src = dir_ms.get(n) if same_machine(plan) else None
        return (src or fallback or {}).get("peers_ms")
    L: list[str] = ["# E7 : passage à l'échelle du protocole et du traqueur", ""]
    if any_meta is None:
        L.append("Aucun résultat.")
    else:
        m = any_meta["machine"]
        st = any_meta["settings"]
        L += [
            "> Généré par `uv run python -m bench.bench_scale report` depuis les fichiers `e7_*.json` de ce dossier.",
            "> Aucun chiffre n'est recopié à la main.", "",
            "## Montage", "",
            f"- Machine : {m['platform']}, {m['cpus']} fils, Python {m['python']}. Tout tourne sur ce seul PC, sans GPU.",
            "- **Traqueur réel** (code de production, réglages par défaut : contrôles aléatoires 5 %, balayage 0,25 s, "
            "jobs gardés 600 s, registre SQLite sur disque), dans **son propre processus** : son temps CPU est mesuré seul "
            "(`time.process_time`). Exception : crédit de départ 10⁹ pour que les demandeurs ne tombent jamais à court.",
            "- **Nœuds simulés** : vrais `NodeClient` (WebSocket, signatures ed25519) avec `FakeEngine`, répartis sur "
            "1 à 4 processus ; un seul job à la fois par nœud (`max_parallel = 1`, un GPU grand public).",
            "- **Demandeurs** : vraies `Gateway` (sélection des pairs, vote pondéré, certificat d'arrêt, reçus), dans 1 à "
            f"3 processus, arrivées de Poisson (charge ouverte). Délai par requête {st['gateway_timeout_s']:.0f} s, "
            f"annuaire des pairs gardé {st.get('gateway_peers_ttl_s', 2.0):g} s (valeur de production) sauf mention.",
            f"- **Modèle de réponses** : le pair i a raison avec la probabilité p_i de sa famille ; s'il se trompe, il "
            f"donne la mauvaise réponse « populaire » de la question avec la probabilité √c, sinon une réponse à lui. "
            f"Deux pairs qui se trompent coïncident donc avec la probabilité c = {st['answer_model']['collision']} "
            "(valeur mesurée sur GSM8K). Les familles mesurées en phase 0 : "
            + ", ".join(f"{x['family']} p = {x['accuracy']}" for x in st["families"] if x["measured"])
            + f". Pour k > 4, familles synthétiques plus faibles (p = {st['families'][-1]['accuracy']}).",
            f"- **Temps de calcul** : lognormal, médiane mesurée par famille ("
            + ", ".join(f"{k} {v} s" for k, v in st["compute_model"]["per_family_medians"].items())
            + f"), σ = {st['compute_model']['sigma']} (phase 0, GSM8K test, un GPU). **Facteur d'échelle** indiqué "
            "pour chaque série : 1 = temps mesurés ; 0,1 et 0,025 = temps divisés par 10 et 40 pour charger le traqueur "
            "en un temps raisonnable (les RTT, eux, ne sont pas réduits).",
            f"- **WAN émulé** dans le relais du traqueur (`myriad/netem.py`) : chaque trame reçue et chaque trame envoyée "
            f"attend un retard aller simple lognormal (σ = {st['wan_sigma']}), ordre conservé ; « RTT » = aller-retour "
            "médian nominal client–traqueur (2 × la médiane aller simple). Une requête traverse 4 sauts retardés "
            "(demandeur → traqueur → nœud → traqueur → demandeur), soit 2 RTT, plus l'annuaire HTTP (retardé aussi).",
            "- Exactitude : réponse fusionnée comparée à la vraie réponse (oracle). « Attendue » : exactitude Monte-Carlo "
            "du vote complet pondéré avec les poids a priori, quand tous les pairs répondent.", ""]

    if lat:
        L += ["## A. Latence de décision : certificat d'arrêt contre attente de tous les pairs", "",
              "Temps de calcul mesurés (échelle 1), 256 nœuds, 2 requêtes/s (charge légère), mêmes questions dans "
              "les deux modes.", "",
              "| scénario | RTT (ms) | k | mode | requêtes | p50 (s) | p95 (s) | p99 (s) | pairs attendus | arrêt anticipé "
              "| exactitude | attendue | pairs refusés (occupés) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for s in lat["scenarios"]:
            sc = s["scenario"]
            for ph in s["phases"]:
                mode = "certificat" if ph["phase"]["early_stop"] else "attendre tous"
                lt = ph["latency_s"]
                L.append(f"| {sc['name']} | {sc['rtt_ms']} | {sc['k']} | {mode} | {ph['requests']} | {f(lt['p50'])} | "
                         f"{f(lt['p95'])} | {f(lt['p99'])} | {f(ph['peers_answered_mean'])} / {f(ph['peers_asked_mean'])} "
                         f"| {f(ph['early_stop_frac'], pct=True)} | {f(ph['accuracy_all'], pct=True)} | "
                         f"{f(s['expected_full_vote_accuracy'], pct=True)} | {f(busy_share(ph), pct=True)} |")
        L += ["", "Les deux modes reçoivent les mêmes questions (numérotées depuis 0) ; seules les dates d'arrivée "
              "(Poisson) diffèrent, d'où des nombres de requêtes un peu différents.", "",
              "Gain du certificat (médiane et p95, en secondes) :", "",
              "| scénario | p50 attendre tous | p50 certificat | gain p50 | p95 attendre tous | p95 certificat | gain p95 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        for s in lat["scenarios"]:
            on = next((p for p in s["phases"] if p["phase"]["early_stop"]), None)
            off = next((p for p in s["phases"] if not p["phase"]["early_stop"]), None)
            if on and off:
                a, b = off["latency_s"], on["latency_s"]
                L.append(f"| {s['scenario']['name']} | {f(a['p50'])} | {f(b['p50'])} | "
                         f"{f(gain(a['p50'], b['p50']), pct=True)} | {f(a['p95'])} | {f(b['p95'])} | "
                         f"{f(gain(a['p95'], b['p95']), pct=True)} |")
        L.append("")

    if sca:
        L += ["## B. Nombre de nœuds × RTT", "",
              "Temps de calcul / 10 (échelle 0,1), k = 4, charge fixée à 0,15 requête/s par nœud, plafonnée à "
              "12 requêtes/s (sous la saturation du traqueur à 1024 nœuds, voir C).", "",
              "| N | RTT (ms) | req/s offertes | servies | p50 (s) | p95 (s) | p99 (s) | pairs attendus | exactitude "
              "| CPU traqueur | ms CPU / requête | trames / requête | annuaire (ms / appel) | appels annuaire / s |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for s in sca["scenarios"]:
            sc, ph = s["scenario"], s["phases"][0]
            t, lt = ph["tracker"], ph["latency_s"]
            L.append(f"| {sc['nodes']} | {sc['rtt_ms']} | {f(ph['offered_rps'], 1)} | {f(ph['goodput_rps'], 1)} | "
                     f"{f(lt['p50'])} | {f(lt['p95'])} | {f(lt['p99'])} | {f(ph['peers_answered_mean'])} | "
                     f"{f(ph['accuracy_all'], pct=True)} | {f(t['cpu_util'], pct=True)} | {f(t['cpu_ms_per_request'], 1)} "
                     f"| {f(t['frames_per_request'], 1)} | {f(dir_cost(sc['nodes'], s['directory_cost'], sca), 1)} | "
                     f"{f(t['http_peers_per_s'], 1)} |")
        L += ["", "« ms CPU / requête » = tout le CPU du traqueur divisé par le nombre de requêtes : il inclut le travail "
              "de fond (battements de cœur des nœuds toutes les 15 s, balayage, sonde de latence), qui domine aux "
              "faibles débits (N = 4, 16). Annuaire : section E.", ""]

    if tpu:
        L += ["## C. Débit : montée en charge jusqu'à saturation", "",
              "Temps de calcul / 40 (échelle 0,025, médiane ≈ 0,1 s) pour que les nœuds ne soient pas le goulot ; "
              "24 passerelles dans 3 processus ; paliers de 15 s ; arrêt au premier palier saturé (débit servi < 90 % "
              "de l'offert, p95 > 3 × celui du premier palier + 1 s, ou plus de 5 % d'échecs).", ""]
        summary_rows = []
        for s in tpu["scenarios"]:
            sc = s["scenario"]
            L += [f"### {sc['name']} (N = {sc['nodes']}, RTT {sc['rtt_ms']} ms, annuaire gardé "
                  f"{sc.get('peers_ttl_s', 2.0):g} s)", "",
                  "| offert (req/s) | servi (req/s) | échecs | p50 (s) | p95 (s) | exactitude | pairs occupés | CPU traqueur "
                  "| ms CPU / requête | µs CPU / trame | appels annuaire / s | retard boucle p99 (ms) | CPU demandeurs (max) |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            for ph in s["phases"]:
                t, lt = ph["tracker"], ph["latency_s"]
                L.append(f"| {f(ph['offered_rps'], 1)} | {f(ph['goodput_rps'], 1)} | {ph['failed']} | {f(lt['p50'])} | "
                         f"{f(lt['p95'])} | {f(ph['accuracy_all'], pct=True)} | {f(busy_share(ph), pct=True)} | "
                         f"{f(t['cpu_util'], pct=True)} | {f(t['cpu_ms_per_request'], 1)} | {f(t['cpu_us_per_frame'], 0)} | "
                         f"{f(t['http_peers_per_s'], 1)} | {f(t['loop_lag_ms']['p99'], 0)} | "
                         f"{f(ph['requester_workers_cpu_util'], pct=True)} |")
            best = sustained(s["phases"])
            summary_rows.append((sc, best, s))
            L.append("")
        L += ["### Débit soutenu", "",
              "| scénario | N | RTT (ms) | annuaire gardé (s) | débit soutenu (req/s) | CPU traqueur à ce débit | "
              "ms CPU / requête | coût d'un appel à l'annuaire (ms) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for sc, best, s in summary_rows:
            L.append(f"| {sc['name']} | {sc['nodes']} | {sc['rtt_ms']} | {sc.get('peers_ttl_s', 2.0):g} | "
                     f"{f(best['goodput_rps'], 1) if best else '–'} | "
                     f"{f(best['tracker']['cpu_util'], pct=True) if best else '–'} | "
                     f"{f(best['tracker']['cpu_ms_per_request'], 1) if best else '–'} | "
                     f"{f(dir_cost(sc['nodes'], s['directory_cost'], tpu), 1)} |")
        L.append("")

    if fai:
        L += ["## D. Pannes : 25 % des nœuds tombent pendant la mesure", "",
              "Temps de calcul mesurés (échelle 1), 256 nœuds, RTT 50 ms, 2,5 requêtes/s pendant 120 s, panne à 40 s. "
              "« crash » : connexion coupée net (TCP réinitialisé) ; « hang » : le nœud reste connecté mais ne répond "
              "plus. Délai par requête 30 s.", "",
              "| mode | fenêtre | requêtes | servies | exactitude | p50 (s) | p95 (s) | max (s) | pairs attendus | "
              "erreurs des pairs | échecs |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for s in fai["scenarios"]:
            ph = s["phases"][0]
            mode = ph["phase"]["kill"]["mode"]
            for wname, w in ph.get("windows", {}).items():
                L.append(f"| {mode} | {wname} | {w['requests']} | {w['ok']} | {f(w['accuracy_all'], pct=True)} | "
                         f"{f(w['p50'])} | {f(w['p95'])} | {f(w['max'])} | {f(w['peers_answered_mean'])} | "
                         f"{', '.join(f'{k} {v}' for k, v in w['peer_errors'].items()) or '–'} | "
                         f"{', '.join(f'{k} {v}' for k, v in w['fail_codes'].items()) or '–'} |")
        L.append("")
    if dco:
        L += ["## E. Coût de l'annuaire des pairs selon N", "",
              f"Mesuré sur {dco['machine']['platform']} ({dco['machine']['cpus']} fils), machine sans autre charge. "
              "Temps de construction d'une réponse à `GET /v1/peers` (tous les pairs, comme FastAPI la produit : vue de "
              "chaque pair avec deux requêtes SQLite, puis `jsonable_encoder` et JSON), moyenne sur 20 appels : temps "
              "écoulé (la boucle du traqueur est bloquée pendant ce temps) et temps CPU du processus sur le lot (le "
              "compteur CPU de Windows avance par 15,6 ms : valeur fiable seulement aux grands N). Puis le temps vu par "
              "un client HTTP local (médiane de 20 appels, RTT 0). Les sections B et C reprennent le temps écoulé quand "
              "leurs mesures viennent de la même machine.", "",
              "| N | vue des pairs (ms) | encodage (ms) | total écoulé (ms) | total CPU (ms) | HTTP, vu du client (ms) "
              "| taille (octets) | /v1/reliability (ms) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for n in sorted(dir_ms):
            c = dir_ms[n]
            L.append(f"| {n} | {f(c.get('peers_view_ms'))} | {f(c.get('peers_encode_ms'))} | {f(c.get('peers_ms'))} | "
                     f"{f(c.get('peers_cpu_ms'))} | {f(c.get('http_peers_ms_p50'))} | {c.get('http_peers_bytes', '–')} | "
                     f"{f(c.get('reliability_ms'))} |")
        L.append("")
    L += v11_section(d, {"scale": sca, "throughput": tpu, "failure": fai, "dircost": dco})
    out = d / "e7_report.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


# ============================================================ essaim/1.1
REFUSALS = ("peer_busy", "busy", "peer_paused", "paused")


def refused_share(ph: dict) -> float | None:
    """Share of the peers asked that refused the job (busy or paused, tracker- or node-side)."""
    total = sum(ph["peer_status"].values())
    return sum(ph["peer_errors"].get(e, 0) for e in REFUSALS) / total if total else None


def _by_name(plan: dict | None) -> dict[str, dict]:
    return {s["scenario"]["name"]: s for s in (plan or {}).get("scenarios", [])}


def _sustained_cell(s: dict | None) -> tuple[str, str, str, str]:
    """(sustained req/s, p50, p95, note) of a load ladder; the note says when the ladder never saturated."""
    if not s:
        return "–", "–", "–", ""
    best = sustained(s["phases"])
    if best is None:
        return "–", "–", "–", ""
    from bench.bench_scale import saturated
    never = not any(saturated(ph, s["phases"][0]) for ph in s["phases"])
    note = f"non saturé au dernier palier ({f(s['phases'][-1]['offered_rps'], 0)} req/s offertes)" if never else ""
    return (("≥ " if never else "") + f(best["goodput_rps"], 1), f(best["latency_s"]["p50"]),
            f(best["latency_s"]["p95"]), note)


def v11_section(d: Path, before: dict) -> list[str]:
    after = {n: _load(d, f"v11_{n}") for n in ("throughput", "scale", "failure", "busy", "dircost")}
    if not any(after.values()):
        return []
    meta = next(x for x in after.values() if x)
    b_meta = next((x for x in before.values() if x), None)
    L = ["## v1.1 : sélection par le traqueur, nœuds figés détectés, pairs refusés remplacés", "",
         f"Protocole {meta.get('protocol_version', '?')} (compatible avec essaim/1). « Avant » : les fichiers "
         "`e7_*.json` d'E7 (essaim/1, mesurés le "
         f"{b_meta['date'][:10] if b_meta else '?'}) ; « après » : les fichiers `e7_v11_*.json` (mesurés le "
         f"{meta['date'][:10]}, {meta['machine']['platform']}, {meta['machine']['cpus']} fils). Même montage, mêmes "
         "modèles de réponses et de temps de calcul, mêmes réglages de production (annuaire gardé 2 s par les "
         "passerelles essaim/1, délai 60 s par requête sauf mention).", "",
         "Ce qui change :", "",
         "- **Sélection côté traqueur** : la passerelle envoie ses k jobs sans cible ; le traqueur choisit chaque pair "
         "en O(k) dans des index par modèle (nœuds connectés, acceptant des jobs, non suspendus, avec un créneau libre), "
         "une famille différente par job, le modèle le plus fiable d'abord, le meilleur de deux tirages au hasard "
         "(moins chargé, moins de manquements). Une trame `assigned` nomme le pair avant tout autre message sur ce job. "
         "La passerelle ne télécharge plus l'annuaire ; `GET /v1/peers` est servi depuis un instantané reconstruit par "
         "morceaux en arrière-plan.",
         "- **Nœuds figés** : ping applicatif toutes les ~10 s (pong attendu 5 s, après une sonde du moteur) ; un nœud "
         "muet, au moteur en panne, ou qui dépasse deux fois de suite le délai d'un job (délai ≥ 10 s) est suspendu "
         "10 s × 2^niveau (300 s au plus), puis réadmis après un pong sain.",
         "- **Pairs refusés** : un pair qui refuse (occupé, en pause) ou échoue vite est remplacé une fois, dans une "
         "famille pas encore utilisée, sinon la même, tant qu'il reste un quart du délai. Le certificat d'arrêt compte "
         "tous les jobs en attente, remplaçants compris, et n'est évalué qu'une fois leurs poids connus : la décision "
         "reste exactement celle du vote complet.",
         "- Le balayage du traqueur n'examine plus tous les jobs gardés (600 s) à chaque passage : tas d'échéances et "
         "files.", ""]

    tb, ta = _by_name(before.get("throughput")), _by_name(after["throughput"])
    if ta:
        L += ["### V1. Débit soutenu (montée en charge, temps de calcul / 40)", "",
              "Même échelle de charge qu'en C (paliers de 15 s, 24 passerelles dans 3 processus, arrêt au premier "
              "palier saturé). Débit soutenu = débit servi au dernier palier non saturé ; p50 et p95 à ce palier.", "",
              "| N | RTT (ms) | avant : req/s | avant : p50 (s) | avant : p95 (s) | après : req/s | après : p50 (s) | "
              "après : p95 (s) | CPU traqueur (après) | ms CPU / requête (après) | remarque |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for name, s in sorted(ta.items(), key=lambda kv: (kv[1]["scenario"]["rtt_ms"] != 100, kv[1]["scenario"]["nodes"],
                                                          kv[1]["scenario"]["rtt_ms"])):
            sc = s["scenario"]
            b = tb.get(name)
            bs, b50, b95, _ = _sustained_cell(b)
            as_, a50, a95, note = _sustained_cell(s)
            best = sustained(s["phases"])
            L.append(f"| {sc['nodes']} | {sc['rtt_ms']} | {bs} | {b50} | {b95} | {as_} | {a50} | {a95} | "
                     f"{f(best['tracker']['cpu_util'], pct=True) if best else '–'} | "
                     f"{f(best['tracker']['cpu_ms_per_request'], 1) if best else '–'} | "
                     f"{note or ('pas de mesure avant' if b is None else '')} |")
        L += ["", "« – » avant : configuration non mesurée en E7 (E7 n'a mesuré le débit à RTT 100 ms qu'à 1024 nœuds, "
              "et n'est pas allé au-delà de 1024 nœuds).", ""]
        L += ["### V2. Détail des paliers (après)", ""]
        for name, s in sorted(ta.items(), key=lambda kv: (kv[1]["scenario"]["rtt_ms"] != 100, kv[1]["scenario"]["nodes"],
                                                          kv[1]["scenario"]["rtt_ms"])):
            sc = s["scenario"]
            L += [f"#### {name} (N = {sc['nodes']}, RTT {sc['rtt_ms']} ms)", "",
                  "| offert (req/s) | servi (req/s) | échecs | p50 (s) | p95 (s) | exactitude | pairs interrogés | "
                  "pairs refusés | remplacements / requête | CPU traqueur | ms CPU / requête | retard boucle p99 (ms) "
                  "| CPU demandeurs (max) |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            for ph in s["phases"]:
                t, lt = ph["tracker"], ph["latency_s"]
                L.append(f"| {f(ph['offered_rps'], 1)} | {f(ph['goodput_rps'], 1)} | {ph['failed']} | {f(lt['p50'])} | "
                         f"{f(lt['p95'])} | {f(ph['accuracy_all'], pct=True)} | {f(ph['peers_asked_mean'])} | "
                         f"{f(refused_share(ph), pct=True)} | {f(ph.get('replacements_per_request'))} | "
                         f"{f(t['cpu_util'], pct=True)} | {f(t['cpu_ms_per_request'], 1)} | "
                         f"{f(t['loop_lag_ms']['p99'], 0)} | {f(ph['requester_workers_cpu_util'], pct=True)} |")
            L.append("")

    sb, sa = _by_name(before.get("scale")), _by_name(after["scale"])
    if sa:
        L += ["### V3. Charge fixe, RTT 100 ms (temps de calcul / 10)", "",
              "0,15 requête/s par nœud, plafonnée à 12 requêtes/s, k = 4 (comme en B).", "",
              "| N | version | req/s offertes | servies | p50 (s) | p95 (s) | p99 (s) | pairs attendus | exactitude | "
              "CPU traqueur | ms CPU / requête | appels annuaire / s |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for name, s in sorted(sa.items(), key=lambda kv: kv[1]["scenario"]["nodes"]):
            for label, x in (("avant", sb.get(name)), ("après", s)):
                if x is None:
                    L.append(f"| {s['scenario']['nodes']} | {label} | – | – | – | – | – | – | – | – | – | – |")
                    continue
                ph = x["phases"][0]
                t, lt = ph["tracker"], ph["latency_s"]
                L.append(f"| {x['scenario']['nodes']} | {label} | {f(ph['offered_rps'], 1)} | {f(ph['goodput_rps'], 1)} | "
                         f"{f(lt['p50'])} | {f(lt['p95'])} | {f(lt['p99'])} | {f(ph['peers_answered_mean'])} | "
                         f"{f(ph['accuracy_all'], pct=True)} | {f(t['cpu_util'], pct=True)} | "
                         f"{f(t['cpu_ms_per_request'], 1)} | {f(t['http_peers_per_s'], 1)} |")
        L.append("")

    fb, fa = _by_name(before.get("failure")), _by_name(after["failure"])
    if fa:
        L += ["### V4. Pannes de 25 % des nœuds (256 nœuds, RTT 50 ms, temps réels, délai 30 s)", "",
              "« hang » : le moteur se fige (ses générations et sa sonde de santé ne répondent plus ; le nœud reste "
              "connecté). « hang_engine » (nouveau) : seules les générations se figent, la sonde répond normalement : "
              "seuls les dépassements de délai trahissent le nœud. Fenêtres par date de soumission : avant la panne, "
              "10 s après, puis le reste.", "",
              "| mode | fenêtre | version | requêtes | servies | exactitude | p50 (s) | p95 (s) | max (s) | pairs attendus "
              "| erreurs des pairs |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for name, s in fa.items():
            mode = s["phases"][0]["phase"]["kill"]["mode"]
            for wname in ("before", "during_10s", "after"):
                for label, x in (("avant", fb.get(name)), ("après", s)):
                    w = x["phases"][0].get("windows", {}).get(wname) if x else None
                    if w is None:
                        if label == "avant":
                            L.append(f"| {mode} | {wname} | avant | – | – | – | – | – | – | – | non mesuré en E7 |")
                        continue
                    L.append(f"| {mode} | {wname} | {label} | {w['requests']} | {w['ok']} | "
                             f"{f(w['accuracy_all'], pct=True)} | {f(w['p50'])} | {f(w['p95'])} | {f(w['max'])} | "
                             f"{f(w['peers_answered_mean'])} | "
                             f"{', '.join(f'{k} {v}' for k, v in w['peer_errors'].items()) or '–'} |")
        L += ["", "Santé vue par le traqueur pendant les mesures « après » (suspensions par motif, réadmissions ; un "
              "nœud toujours figé à la fin de sa suspension est suspendu de nouveau, plus longtemps), et remplacements "
              "faits par les passerelles sur toute la mesure :", "",
              "| mode | événements | nœuds suspendus à la fin | remplacements / requête |", "| --- | --- | --- | --- |"]
        for name, s in fa.items():
            ph = s["phases"][0]
            ev = ", ".join(f"{k} {v}" for k, v in sorted(ph.get("health", {}).items())) or "–"
            L.append(f"| {ph['phase']['kill']['mode']} | {ev} | {ph.get('suspended_end', '–')} | "
                     f"{f(ph.get('replacements_per_request'), 3)} |")
        L += ["", "Un pair figé ne renvoie pas d'erreur rapide : il n'est pas remplacé, il est écarté des requêtes "
              "suivantes par la suspension. Quand la sonde du moteur ne voit rien (hang_engine), seuls deux "
              "dépassements de délai consécutifs le trahissent, et la plupart de ses jobs sont annulés par le certificat "
              "avant leur délai : la détection reste lente, d'où un p95 encore au délai de la requête.", ""]

    bu = after["busy"]
    if bu or tb:
        L += ["### V5. Pairs occupés", ""]
        if tb.get("tput_n64_rtt0") and ta.get("tput_n64_rtt0"):
            b_ph = {round(p["phase"]["rate"]): p for p in tb["tput_n64_rtt0"]["phases"]}
            a_ph = {round(p["phase"]["rate"]): p for p in ta["tput_n64_rtt0"]["phases"]}
            L += ["64 nœuds, temps / 40, RTT 0 (échelle de charge de V1) : part des pairs interrogés qui refusent le job, "
                  "et pairs qui ont répondu, à chaque palier mesuré dans les deux versions.", "",
                  "| offert (req/s) | avant : refusés | avant : pairs interrogés | avant : exactitude | après : refusés | "
                  "après : pairs interrogés | après : remplacements / requête | après : exactitude |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
            for rate in sorted(set(b_ph) & set(a_ph)):
                b, a = b_ph[rate], a_ph[rate]
                L.append(f"| {rate} | {f(refused_share(b), pct=True)} | {f(b['peers_asked_mean'])} | "
                         f"{f(b['accuracy_all'], pct=True)} | {f(refused_share(a), pct=True)} | "
                         f"{f(a['peers_asked_mean'])} | {f(a.get('replacements_per_request'))} | "
                         f"{f(a['accuracy_all'], pct=True)} |")
            L += ["", "Le traqueur ne choisit que des nœuds qui ont un créneau libre : plus aucun pair n'est refusé. Quand "
                  "une famille n'a plus de nœud libre, la requête part avec moins de pairs (colonne « pairs "
                  "interrogés ») au lieu de perdre des pairs refusés, et il n'y a rien à remplacer. Le remplacement sert "
                  "aux refus et pannes rapides vus par le nœud lui-même (coupure, moteur en erreur, refus local).", ""]
        if bu:
            L += ["Charge proche de la saturation des nœuds (64 nœuds, temps / 10, médiane ≈ 0,4 s, un job à la fois "
                  "par nœud, RTT 0), trois variantes mesurées dans la même session : v1.1 complète ; sans remplacement ; "
                  "et passerelles en mode annuaire sans remplacement (le comportement d'essaim/1).", "",
                  "| variante | offert (req/s) | servi (req/s) | p50 (s) | p95 (s) | pairs interrogés | pairs ayant répondu "
                  "| refusés | remplacements / requête | exactitude |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            labels = {"busy_n64": "v1.1", "busy_n64_noreplace": "v1.1 sans remplacement",
                      "busy_n64_directory": "annuaire (essaim/1)"}
            for s in bu["scenarios"]:
                for ph in s["phases"]:
                    lt = ph["latency_s"]
                    L.append(f"| {labels.get(s['scenario']['name'], s['scenario']['name'])} | {f(ph['offered_rps'], 1)} | "
                             f"{f(ph['goodput_rps'], 1)} | {f(lt['p50'])} | {f(lt['p95'])} | {f(ph['peers_asked_mean'])} "
                             f"| {f(ph['peers_answered_mean'])} | {f(refused_share(ph), pct=True)} | "
                             f"{f(ph.get('replacements_per_request'))} | {f(ph['accuracy_all'], pct=True)} |")
            L.append("")

    da = after["dircost"]
    if da:
        db = {s["scenario"]["nodes"]: s["directory_cost"] for s in (before.get("dircost") or {}).get("scenarios", [])}
        L += ["### V6. Coût de l'annuaire et de la sélection", "",
              "Avant : chaque `GET /v1/peers` construisait la liste complète (boucle bloquée pendant ce temps). Après : "
              "la réponse vient d'un instantané ; sa reconstruction (au plus toutes les 3 s, seulement si quelqu'un lit "
              "l'annuaire) rend la main à la boucle tous les 256 pairs ; la sélection de 4 pairs ne dépend pas de N. "
              "Reconstruction : moyenne sur 20 ; plus long blocage : maximum sur ces 20 reconstructions (il inclut "
              "l'assemblage final de la réponse et les pauses du ramasse-miettes). Sélection : moyenne sur 2000 "
              "appels.", "",
              "| N | avant : construction (ms, boucle bloquée) | avant : HTTP vu du client (ms) | après : HTTP vu du client "
              "(ms) | après : reconstruction (ms) | après : plus long blocage (ms) | après : sélection de 4 pairs (µs) |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        for s in sorted(da["scenarios"], key=lambda x: x["scenario"]["nodes"]):
            n, c = s["scenario"]["nodes"], s["directory_cost"]
            b = db.get(n, {})
            L.append(f"| {n} | {f(b.get('peers_ms'))} | {f(b.get('http_peers_ms_p50'))} | {f(c.get('http_peers_ms_p50'))} | "
                     f"{f(c.get('snapshot_work_ms'))} | {f(c.get('snapshot_block_ms'))} | {f(c.get('select4_us'), 1)} |")
        L.append("")
    return L
