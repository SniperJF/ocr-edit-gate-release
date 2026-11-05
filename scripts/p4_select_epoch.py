#!/usr/bin/env python3
"""P4-c: pick the Qwen3.8-27B LoRA epoch on the HELD-OUT part of train (never dev); same rule as scripts/p2_select_epoch.py
(lowest held-out weighted cMER, tie: higher pref, lower epoch) but written to results/p4_epoch_selection.json so that the
P2/P3 file is left untouched.

    python scripts/p4_select_epoch.py qwen38-lora
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
runs = [json.loads(l) for l in (ROOT / "results" / "runs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
latest = {}
for r in runs:
    if r.get("system") == "zeroshot" and r.get("tag") and not r.get("limit"):
        latest[r["tag"]] = r
out, lines = {}, ["| model | epoch | held-out train cMER / pref (official cells of the held-out subset) | dev cMER / pref (report only) | chosen |", "|---|---|---|---|---|"]
for tag in sys.argv[1:]:
    cands = []
    for ep in (1, 2, 3):
        ho = latest.get(f"{tag}-ep{ep}-ho")
        if not ho:
            continue
        dv = latest.get(f"{tag}-ep{ep}")
        cands.append((ho["cmer_micro_weighted"], -ho["pref_score_cmer_macro_weighted"], ep, ho, dv))
    if not cands:
        continue
    best = min(cands)[2]
    out[tag] = {"chosen_epoch": best, "criterion": "lowest held-out-train weighted cMER (tie: higher pref, lower epoch)",
                "candidates": {c[2]: {"heldout_run": c[3]["run_id"], "heldout_cmer": c[3]["cmer_micro_weighted"], "heldout_pref": c[3]["pref_score_cmer_macro_weighted"],
                                      "dev_run": c[4]["run_id"] if c[4] else None, "dev_cmer": c[4]["cmer_micro_weighted"] if c[4] else None,
                                      "dev_pref": c[4]["pref_score_cmer_macro_weighted"] if c[4] else None} for c in cands}}
    for c in sorted(cands, key=lambda c: c[2]):
        dv = f"{c[4]['cmer_micro_weighted']:.4f} / {c[4]['pref_score_cmer_macro_weighted']:.3f}" if c[4] else "-"
        lines.append(f"| {tag} | {c[2]} | {c[3]['cmer_micro_weighted']:.4f} / {c[3]['pref_score_cmer_macro_weighted']:.3f} | {dv} | {'**yes**' if c[2] == best else ''} |")
(ROOT / "results" / "p4_epoch_selection.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("\n".join(lines))
