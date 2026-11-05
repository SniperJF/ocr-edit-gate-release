#!/usr/bin/env python3
"""Build results/stats_pairs_p3.json (pairs for scripts/p2_stats.sh) from results/runs.jsonl by run tags.

    python scripts/p3_stats_pairs.py BEST_ENSEMBLE_TAG [SYNTH_TAG]
BEST_ENSEMBLE_TAG = tag of the P3-4 frozen dev ensemble run (system ensemble); SYNTH_TAG = chosen-epoch synthetic-mix LoRA-72B dev tag
(e.g. lora72b-syn-ep2) if the P3-3 model exists.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
runs = [json.loads(l) for l in (ROOT / "results" / "runs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
latest = {}
for r in runs:
    if r.get("split") == "dev" and not r.get("limit") and not r.get("subset") and "dryrun" not in (r.get("tag") or "") and "smoke" not in (r.get("tag") or ""):
        latest[(r["system"], r.get("tag") or "")] = r["run_id"]
best, synth = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
pairs, missing = [], []


def g(system, tag):
    return latest.get((system, tag), f"MISSING:{system}:{tag}")


def add(label, base, test):
    if base in latest.values() and test in latest.values():
        pairs.append({"label": label, "base": base, "test": test})
    else:
        missing.append(label)


P2BEST = g("ensemble", "dev-3pool-agree"); L32G = g("zeroshot_gate", "lora32b-ep2_p2_pooled_r-global"); L72 = g("zeroshot_nogate", "lora72b-ep2"); L72G = g("zeroshot_gate", "lora72b-ep2_p2_pooled_r-global")
# P3-1
add("LoRA-72B ungated vs LoRA-32B ungated", g("zeroshot_nogate", "lora32b-ep2"), L72)
add("LoRA-72B: P2 full gate (global) vs no gate", L72, L72G)
add("LoRA-72B: P2 text-only gate (global) vs no gate", L72, g("zeroshot_gate", "lora72b-ep2_p2_pooled_nolp_r-global"))
add("LoRA-72B: label oracle (upper bound) vs no gate", L72, g("zeroshot_oraclelabel", "lora72b-ep2"))
add("LoRA-72B + P2 gate vs LoRA-32B + P2 gate", L32G, L72G)
add("LoRA-72B + P2 gate vs P2 best ensemble (3-pool agreement)", P2BEST, L72G)
# P3-2 / P3-4 ensembles vs P2 best
for t, lab in [("dev-p3-3pool-L3", "P2 pool, 3 levels"), ("dev-p3-4greedy-L2", "4 greedy (+LoRA-72B), 2 levels"), ("dev-p3-4greedy-L3", "4 greedy (+LoRA-72B), 3 levels"),
               ("dev-p3-s72-L3", "4 greedy + 4 LoRA-72B samples, 3 levels"), ("dev-p3-s32-L3", "4 greedy + 4 LoRA-32B samples, 3 levels"),
               ("dev-p3-s72s32-L3", "4 greedy + 8 samples, 3 levels (families)"), ("dev-p3-s72s32-L3-votes", "4 greedy + 8 samples, 3 levels (votes)")]:
    add(f"ensemble {lab} vs P2 best ensemble", P2BEST, g("ensemble", t))
add("ensemble 4 greedy 3 levels vs 4 greedy 2 levels", g("ensemble", "dev-p3-4greedy-L2"), g("ensemble", "dev-p3-4greedy-L3"))
add("ensemble 4 greedy 3 levels vs LoRA-72B + P2 gate", L72G, g("ensemble", "dev-p3-4greedy-L3"))
# P3-3
if synth:
    S = g("zeroshot_nogate", synth); SG = g("zeroshot_gate", f"{synth}_p2_pooled_r-global")
    add("LoRA-72B + synthetic ungated vs LoRA-72B ungated", L72, S)
    add("LoRA-72B + synthetic, P2 gate vs LoRA-72B, P2 gate", L72G, SG)
    add("LoRA-72B + synthetic: P2 full gate vs no gate", S, SG)
# P3-4 frozen
B = g("ensemble", best)
add(f"P3 frozen ensemble ({best}) vs P2 best ensemble (3-pool agreement, 0.0089)", P2BEST, B)
add(f"P3 frozen ensemble ({best}) vs LoRA-72B + P2 gate", L72G, B)
add(f"P3 frozen ensemble ({best}) vs best Phase 1 system (Mistral-24B zs + P1 full gate)", g("zeroshot_gate", "mistral24b_qwen06_pooled"), B)
add(f"P3 frozen ensemble ({best}) vs no correction", g("nocorrect", ""), B)
(ROOT / "results" / "stats_pairs_p3.json").write_text(json.dumps(pairs, indent=1), encoding="utf-8")
print(f"{len(pairs)} pairs; missing: {missing}")
