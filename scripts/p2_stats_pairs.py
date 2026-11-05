#!/usr/bin/env python3
"""Build results/stats_pairs_p2.json (pairs for scripts/p2_stats.sh) from results/runs.jsonl by run tags.

    python scripts/p2_stats_pairs.py LORA24_TAG LORA32_TAG [BEST_SINGLE_TAG]
BEST_SINGLE_TAG = tag of the single-rewriter gated run the ensembles are compared with (e.g. lora32b-ep2_p2_pooled_r-global).
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
lora24, lora32 = sys.argv[1], sys.argv[2]
best_single = sys.argv[3] if len(sys.argv) > 3 else f"{lora32}_p2_pooled_r-global"
RW = [("Qwen3-8B zs", "8b-lp"), ("Qwen3-32B zs", "32b-lp"), ("Mistral-24B zs", "mistral24b"), ("Qwen2.5-72B zs", "q72b"),
      ("LoRA-8B (P1-b)", "lora8b"), ("LoRA-24B (P2-1)", lora24), ("LoRA-32B (P2-1)", lora32)]
pairs, missing = [], []


def add(label, base, test):
    if base in latest.values() and test in latest.values():
        pairs.append({"label": label, "base": base, "test": test})
    else:
        missing.append(label)


def g(system, tag):
    return latest.get((system, tag), f"MISSING:{system}:{tag}")


for name, tag in RW:
    ng = g("zeroshot_nogate", tag)
    add(f"{name}: P2 full gate (global) vs no gate", ng, g("zeroshot_gate", f"{tag}_p2_pooled_r-global"))
    add(f"{name}: P2 text-only gate (global) vs no gate", ng, g("zeroshot_gate", f"{tag}_p2_pooled_nolp_r-global"))
    add(f"{name}: P1 frozen full gate vs no gate", ng, g("zeroshot_gate", f"{tag}_qwen06_pooled"))
    add(f"{name}: P2 full gate (global) vs P1 frozen full gate", g("zeroshot_gate", f"{tag}_qwen06_pooled"), g("zeroshot_gate", f"{tag}_p2_pooled_r-global"))
    for rule in ["editrate", "sysconf", "lang", "lang_editrate"]:
        add(f"{name}: P2 full gate rule {rule} vs rule global", g("zeroshot_gate", f"{tag}_p2_pooled_r-global"), g("zeroshot_gate", f"{tag}_p2_pooled_r-{rule}"))
    add(f"{name}: label oracle (upper bound) vs no gate", ng, g("zeroshot_oraclelabel", tag))
# rewriter comparisons (all with the P2 full gate, global rule)
add("LoRA-32B + P2 gate vs LoRA-24B + P2 gate", g("zeroshot_gate", f"{lora24}_p2_pooled_r-global"), g("zeroshot_gate", f"{lora32}_p2_pooled_r-global"))
add("LoRA-32B + P2 gate vs best Phase 1 system (Mistral-24B zs + P1 full gate)", g("zeroshot_gate", "mistral24b_qwen06_pooled"), g("zeroshot_gate", f"{lora32}_p2_pooled_r-global"))
add("LoRA-24B + P2 gate vs best Phase 1 system (Mistral-24B zs + P1 full gate)", g("zeroshot_gate", "mistral24b_qwen06_pooled"), g("zeroshot_gate", f"{lora24}_p2_pooled_r-global"))
add("LoRA-32B ungated vs best Phase 1 system (Mistral-24B zs + P1 full gate)", g("zeroshot_gate", "mistral24b_qwen06_pooled"), g("zeroshot_nogate", lora32))
# ensembles
bs = g("zeroshot_gate", best_single)
for t, lab in [("dev-2pool-lora24-lora32", "2-pool LoRA-24B+LoRA-32B (global)"), ("dev-2pool-lora24-q72b", "2-pool LoRA-24B+72B (global)"),
               ("dev-3pool-lora24-lora32-q72b", "3-pool LoRA-24B+LoRA-32B+72B (global)"), ("dev-3pool-agree", "3-pool agreement thresholds"),
               ("dev-2pool-agree", "2-pool LoRA-24B+LoRA-32B agreement thresholds")]:
    add(f"ensemble {lab} vs best single ({best_single})", bs, g("ensemble", t))
add("ensemble 3-pool agreement vs 3-pool global", g("ensemble", "dev-3pool-lora24-lora32-q72b"), g("ensemble", "dev-3pool-agree"))
add("ensemble 3-pool agreement vs LoRA-32B no gate", g("zeroshot_nogate", lora32), g("ensemble", "dev-3pool-agree"))
add("ensemble 3-pool agreement vs best Phase 1 system (Mistral-24B zs + P1 full gate)", g("zeroshot_gate", "mistral24b_qwen06_pooled"), g("ensemble", "dev-3pool-agree"))
(ROOT / "results" / "stats_pairs_p2.json").write_text(json.dumps(pairs, indent=1), encoding="utf-8")
print(f"{len(pairs)} pairs; missing: {missing}")
