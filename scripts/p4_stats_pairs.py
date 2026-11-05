#!/usr/bin/env python3
"""Build results/stats_pairs_p4.json (dev pairs for scripts/p2_stats.sh) from results/runs.jsonl by run tags.

    python scripts/p4_stats_pairs.py qwen38-lora-ep<N> [dev-p4-best]
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
lora, best = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
pairs, missing = [], []


def g(system, tag):
    return latest.get((system, tag), f"MISSING:{system}:{tag}")


def add(label, base, test):
    if base in latest.values() and test in latest.values():
        pairs.append({"label": label, "base": base, "test": test})
    else:
        missing.append(label)


ZS = g("zeroshot_nogate", "qwen38-zs"); ZSG = g("zeroshot_gate", "qwen38-zs_p2_pooled_r-global"); ZST = g("zeroshot_gate", "qwen38-zs_p2_pooled_nolp_r-global"); ZSO = g("zeroshot_oraclelabel", "qwen38-zs")
LO = g("zeroshot_nogate", lora); LOG = g("zeroshot_gate", f"{lora}_p2_pooled_r-global"); LOT = g("zeroshot_gate", f"{lora}_p2_pooled_nolp_r-global"); LOO = g("zeroshot_oraclelabel", lora)
L72G = g("zeroshot_gate", "lora72b-ep2_p2_pooled_r-global"); L32G = g("zeroshot_gate", "lora32b-ep2_p2_pooled_r-global"); Q72G = g("zeroshot_gate", "q72b_p2_pooled_r-global")
P3 = g("ensemble", "dev-p3-frozen"); P2BEST = g("ensemble", "dev-3pool-agree")
add("Qwen3.8 zero-shot: P2 full gate (global) vs no gate", ZS, ZSG)
add("Qwen3.8 zero-shot: P2 text-only gate (global) vs no gate", ZS, ZST)
add("Qwen3.8 zero-shot: label oracle (upper bound) vs no gate", ZS, ZSO)
add("Qwen3.8 LoRA: P2 full gate (global) vs no gate", LO, LOG)
add("Qwen3.8 LoRA: P2 text-only gate (global) vs no gate", LO, LOT)
add("Qwen3.8 LoRA: label oracle (upper bound) vs no gate", LO, LOO)
add("Qwen3.8 LoRA ungated vs Qwen3.8 zero-shot ungated", ZS, LO)
add("Qwen3.8 LoRA + P2 full gate vs Qwen3.8 zero-shot + P2 full gate", ZSG, LOG)
add("Qwen3.8 zero-shot + P2 full gate vs Qwen2.5-72B zero-shot + P2 full gate", Q72G, ZSG)
add("Qwen3.8 zero-shot ungated vs Qwen2.5-72B zero-shot ungated", g("zeroshot_nogate", "q72b"), ZS)
add("Qwen3.8 LoRA + P2 full gate vs LoRA-32B + P2 full gate", L32G, LOG)
add("Qwen3.8 LoRA + P2 full gate vs LoRA-72B + P2 full gate (best P3 single)", L72G, LOG)
add("Qwen3.8 LoRA ungated vs LoRA-72B ungated", g("zeroshot_nogate", "lora72b-ep2"), LO)
add("Qwen3.8 LoRA + P2 full gate vs P3 frozen ensemble", P3, LOG)
if best:
    B = g("ensemble", best)
    add(f"best P4 ensemble ({best}) vs P3 frozen ensemble", P3, B)
    add(f"best P4 ensemble ({best}) vs P2 best ensemble (3-pool agreement)", P2BEST, B)
    add(f"best P4 ensemble ({best}) vs Qwen3.8 LoRA + P2 full gate", LOG, B)
    add(f"best P4 ensemble ({best}) vs LoRA-72B + P2 full gate", L72G, B)
(ROOT / "results" / "stats_pairs_p4.json").write_text(json.dumps(pairs, indent=1), encoding="utf-8")
print(f"{len(pairs)} pairs; missing: {missing}")
