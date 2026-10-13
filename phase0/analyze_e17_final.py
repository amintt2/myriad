"""Descriptive frozen E17 analysis only after full recovery and confirmed owned-campaign closure."""
import argparse
from pathlib import Path

from e17_campaign import canonical
from e17_final_freeze import read_json, validate_retrieved


def analyze(directory, report, output):
    if (report.get("verdict") != "test-complete" or report.get("closed") is not True
            or report.get("absence_confirmed") is not True or report.get("retrieved_complete") is not True
            or report.get("error") is not None or report.get("recovery_error") is not None):
        raise ValueError("partial or unclosed campaign cannot produce a final score")
    inventory = read_json((directory / "retrieval.json").read_bytes())
    if inventory != report.get("inventory") or inventory["request"]["plan"] != report.get("plan"):
        raise ValueError("foreign recovery/closure report")
    validate_retrieved(directory, inventory, inventory["request"])
    rows = read_json((directory / "verdicts.json").read_bytes())
    attempts = read_json((directory / "registry.json").read_bytes())["attempts"]
    def metrics(selected):
        known = [row["completion_tokens"] for row in selected]
        seconds = sum(row["seconds"] for row in selected)
        return {"episodes": len(selected), "passed": sum(row["passed"] for row in selected),
                "completion_tokens": sum(known) if None not in known else None,
                "known_completion_tokens": sum(row["known_completion_tokens"] for row in selected),
                "charged_request_ceilings": sum(row["charged_request_ceilings"] for row in selected),
                "unknown_usage_requests": sum(row["unknown_usage_requests"] for row in selected),
                "requests": sum(row["requests"] for row in selected), "episode_seconds": seconds,
                "request_seconds": sum(row["request_seconds"] for row in selected),
                "model_loading_seconds": sum(a["loading_seconds"] for a in attempts
                                             if any(a["index"] == r["cell_index"] for r in selected)),
                "server_request_seconds": sum(a["request_seconds"] for a in attempts
                                              if any(a["index"] == r["cell_index"] for r in selected)),
                "estimated_wh": 100 * seconds / 3600, "estimated_eur": 100 * seconds / 3600 / 1000 * .25}
    modes = report["plan"]["mode_order"]
    summary = {"schema": 1, "freeze_sha256": report["plan"]["freeze_sha256"],
               "scope": "four reserved tasks; paired descriptive pilot, no official population score",
               "trust_scope": "public-benchmark-nonadversarial", "adversarial_attestation": False,
               "simulation": "centralized-on-one-vm", "energy": {"watts": 100, "eur_kwh": .25, "measured": False},
               "mode_order": modes,
               "modes": {mode: metrics([r for r in rows if r["mode"] == mode]) for mode in modes},
               "by_benchmark": {bench: {mode: metrics([r for r in rows if r["mode"] == mode and r["benchmark"] == bench])
                                        for mode in modes} for bench in ("aider", "bigcode")},
               "usage_observations": report["usage"], "account_is_not_a_scientific_invoice": True,
               "partial_scores_allowed": False, "proposal_feedback_allowed": False}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "summary.json").open("xb") as out:
        out.write(canonical(summary))
    lines = ["# E17 finale figée", "", "Quatre tâches réservées, seize cellules complètes, clôture confirmée.",
             "Comparaison appariée descriptive ; aucun score officiel ni preuve de généralisation.", "",
             "| mode | réussites | jetons connus | plafonds réservés | secondes épisodes |",
             "|---|---:|---:|---:|---:|"]
    for mode, row in summary["modes"].items():
        lines.append(f"| {mode} | {row['passed']}/{row['episodes']} | {row['known_completion_tokens']} | "
                     f"{row['charged_request_ceilings']} | {row['episode_seconds']:.6f} |")
    lines.extend(["", "Jetons connus distincts du total inconnu et des plafonds réservés. Temps d'appels, "
                  "épisodes, chargements et clôture sont distincts ; variation du compte sans facture scientifique.",
                  "Énergie/coût hypothétiques :100W et0,25EUR/kWh, non mesurés. Simulation centralisée sur VM.",
                  "Notation publique non adversariale, aucune attestation cryptographique de Python arbitraire."])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--artifacts", type=Path, required=True)
    cli.add_argument("--campaign-report", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    analyze(args.artifacts, read_json(args.campaign_report.read_bytes()), args.output)


if __name__ == "__main__":
    main()
