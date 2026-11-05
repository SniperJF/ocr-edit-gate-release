#!/usr/bin/env python3
"""Third (P4) TEST look, stats step: paired bootstrap (10,000 resamples, seed 1234; ocrgate.stats, method of test run 1) for the
pre-declared comparisons of the P4 secondaries: Qwen3.8 zero-shot + P2 full gate, Qwen3.8-LoRA + P2 full gate (and the P4
ensemble if it was run) each vs the P3 primary ensemble (test look 2), vs Qwen2.5-72B + P1 gate (test run 1), vs bnf-mistral
run1 / run3 (ungated participant files, official scorer JSONs of test run 1); the gate-vs-no-gate pairs of both rewriters; and
declared side-by-sides. Everything here is SECONDARY / post-hoc / third look. Writes results/stats_test_p4.jsonl and
results/test_stats_p4.md.

    CONFIRM_TEST_RUN=1 $VLLM_PYTHON scripts/test_stats_p4.py
"""
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
PY = str(Path.home() / "vllm-env/bin/python")
if os.environ.get("CONFIRM_TEST_RUN") != "1":
    sys.exit("set CONFIRM_TEST_RUN=1 only for the approved final test run")


def one_run(pattern: str) -> str:
    g = sorted(glob.glob(f"runs/*_{pattern}"))
    if len(g) != 1:
        sys.exit(f"expected exactly one run for {pattern}, found {g}")
    return g[0]


def run_stats(args: list[str]) -> dict:
    out = subprocess.run([PY, "-m", "ocrgate.stats", "--config", "configs/phase2.yaml", *args, "--n", "10000", "--seed", "1234", "--allow-test"],
                         capture_output=True, text=True, env={**os.environ, "CONFIRM_TEST_RUN": "1"})
    if out.returncode != 0:
        sys.exit(f"stats failed for {args}:\n{out.stderr[-2000:]}")
    return json.loads(out.stdout)


P3 = one_run("ensemble_test_test-p3-4fam-L3-p3frozen")          # second look, primary
NOCORR = "runs/20261029T084853Z_nocorrect_test"
Q72_T1 = "runs/20261030T070207Z_zeroshot_gate_test_q72b-frozen_qwen06_pooled"      # test run 1: Qwen2.5-72B zs + P1 full gate
L72G = one_run("zeroshot_gate_test_lora72b-p3frozen_p2_pooled_r-global")           # second look
L32G = one_run("zeroshot_gate_test_lora32b-p3frozen_p2_pooled_r-global")
Q72G = one_run("zeroshot_gate_test_q72b-p2frozen_p2_pooled_r-global")
ens = json.load(open("results/p4_frozen_ensemble.json"))
P4ENS = one_run("ensemble_test_test-p4-ens-p4frozen") if ens["beats_p3"] else None
REWRITERS = [("qwen38-zs-p4frozen", "Qwen3.8-27B zero-shot"), ("qwen38-lora-p4frozen", "Qwen3.8-27B LoRA")]
GATES = [("P2 full gate (global)", "zeroshot_gate_test_{t}_p2_pooled_r-global"),
         ("P2 text-only gate (global)", "zeroshot_gate_test_{t}_p2_pooled_nolp_r-global"),
         ("label oracle (upper bound)", "zeroshot_oraclelabel_test_{t}")]
rows = []


def add(label, kind, base, test, **extra):
    print(f"[stats] {label}", file=sys.stderr, flush=True)
    r = run_stats(["--base", base, "--test", test]); r["label"] = label; r["kind"] = kind; r.update(extra); rows.append(r)


SYS = __import__("os").path.join(__import__("os").environ.get("HIPE_EVAL_DIR", "HIPE-OCRepair-2026-eval"), "data/systems"); OUT = "runs/external_gated_test"
systems = {}
for f in sorted(glob.glob(f"{SYS}/*_run[0-9].jsonl")):
    stem = Path(f).stem; team, rest = stem.split("_", 1)
    label = rest[len("hipe-ocrepair-bench_v0.9_"):].rsplit("_run", 1)[0].replace("_masked-test_", "_test_")
    systems.setdefault((team, stem.rsplit("_run", 1)[1]), {})[label] = (f, stem)


def add_participant(label, team, run_n, test):
    cells = systems[(team, run_n)]
    base = {lab: {"file": f, "score": f"{OUT}/scores/{stem}.ungated.json"} for lab, (f, stem) in cells.items()}
    for v in base.values():
        for p in (v["file"], v["score"]):
            if not Path(p).is_file():
                sys.exit(f"missing {p}")
    print(f"[stats] {label}", file=sys.stderr, flush=True)
    r = run_stats(["--external-base", json.dumps(base), "--test", test, "--split", "test"])
    r["label"] = label; r["kind"] = "participant"; r["team"] = team; r["run"] = run_n; r["n_cells"] = len(cells); rows.append(r)


MAIN = [("Qwen3.8 zero-shot + P2 full gate", one_run("zeroshot_gate_test_qwen38-zs-p4frozen_p2_pooled_r-global")),
        ("Qwen3.8 LoRA + P2 full gate", one_run("zeroshot_gate_test_qwen38-lora-p4frozen_p2_pooled_r-global"))]
if P4ENS:
    MAIN.append(("P4 ensemble", P4ENS))
# A. each P4 system vs the pre-declared references
for name, run in MAIN:
    add(f"{name} vs P3 primary ensemble (test look 2)", "main", P3, run)
    add(f"{name} vs Qwen2.5-72B + P1 full gate (test run 1)", "main", Q72_T1, run)
    add(f"{name} vs no-correction (OCR)", "main", NOCORR, run)
    add_participant(f"{name} vs bnf-mistral run1 (ungated participant file)", "bnf-mistral", "1", run)
    add_participant(f"{name} vs bnf-mistral run3 (ungated participant file)", "bnf-mistral", "3", run)
# B. gates vs no gate
for t, name in REWRITERS:
    base = one_run(f"zeroshot_nogate_test_{t}")
    for glabel, pat in GATES:
        add(f"{name}: {glabel} vs no gate", "rewriter", base, one_run(pat.format(t=t)))
# C. declared side-by-sides
ZS = one_run("zeroshot_nogate_test_qwen38-zs-p4frozen"); LO = one_run("zeroshot_nogate_test_qwen38-lora-p4frozen")
ZSG, LOG = MAIN[0][1], MAIN[1][1]
add("Qwen3.8 LoRA ungated vs Qwen3.8 zero-shot ungated", "side", ZS, LO)
add("Qwen3.8 LoRA + P2 full gate vs Qwen3.8 zero-shot + P2 full gate", "side", ZSG, LOG)
add("Qwen3.8 zero-shot + P2 full gate vs Qwen2.5-72B zero-shot + P2 full gate (test look 2)", "side", Q72G, ZSG)
add("Qwen3.8 zero-shot ungated vs Qwen2.5-72B zero-shot ungated (test look 2)", "side", one_run("zeroshot_nogate_test_q72b-p2frozen"), ZS)
add("Qwen3.8 LoRA + P2 full gate vs LoRA-32B + P2 full gate (test look 2)", "side", L32G, LOG)
add("Qwen3.8 LoRA + P2 full gate vs LoRA-72B + P2 full gate (test look 2)", "side", L72G, LOG)
add("Qwen3.8 LoRA ungated vs LoRA-72B ungated (test look 2)", "side", one_run("zeroshot_nogate_test_lora72b-p3frozen"), LO)
if P4ENS:
    add("P4 ensemble vs Qwen3.8 LoRA + P2 full gate", "side", LOG, P4ENS)

Path("results/stats_test_p4.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def fmt_d(d, nd=4):
    lo, hi = d["ci95"]; p = d["p_two_sided"]
    ps = f"{p:.4f}" if p >= 2.1e-4 else "<0.0002"
    return f"{d['point']:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}] p={ps}"


md = ["# Third (P4) TEST look: paired bootstrap (test split; one pass; every system here is SECONDARY / post-hoc)", "",
      "Tool: `ocrgate/stats.py` via `scripts/test_stats_p4.py`; raw output `results/stats_test_p4.jsonl`. Method identical to",
      "`results/test_stats.md` / `test_stats_p3.md`: per-unit counts recomputed with the official scorer's functions and checked against every",
      "official scorer JSON; paired bootstrap over units within cells, 10,000 resamples, `numpy.random.default_rng(1234)`; weighted value =",
      "official design weights over the 8 official cells (DTA 1/3 each; 590 scored units of 740). Δ = test − base; negative Δ cMER and",
      "positive Δ pref are improvements; 95% CI = 2.5/97.5 percentiles; two-sided p with the +1 correction (floor ≈ 0.0002). The P3 primary",
      "ensemble, the LoRA-72B/32B and Qwen2.5-72B secondaries are the runs of test look 2; Qwen2.5-72B + P1 gate is test run 1; participant",
      "files are the submitted (ungated) outputs scored in test run 1. The P4 system is always 'test' in a pair.", ""]
for kind, title in (("main", "P4 systems vs the pre-declared references"), ("participant", "P4 systems vs bnf-mistral run1 / run3 (ungated participant files)"),
                    ("rewriter", "Gates vs no gate (P2 gate unchanged)"), ("side", "Declared side-by-sides")):
    md += [f"## {title}", "", "| comparison | cells | cMER base -> test | Δ cMER [95% CI] p | pref base -> test | Δ pref [95% CI] p |", "|---|---|---|---|---|---|"]
    for r in rows:
        if r["kind"] != kind:
            continue
        w = r["cells"]["weighted"]
        md.append(f"| {r['label']} | {w['n_official_cells']}/8 | {w['cmer_base']:.4f} -> {w['cmer_test']:.4f} | {fmt_d(w['d_cmer'])} | {w['pref_base']:.4f} -> {w['pref_test']:.4f} | {fmt_d(w['d_pref'], 3)} |")
    md.append("")
md += ["## Per cell", ""]
for r in rows:
    b = r["base"] if isinstance(r["base"], str) else "participant submission files"
    md += [f"### {r['label']}", "", f"base `{b}`, test `{r['test']}`", "",
           "| cell | units scored | weight | cMER base -> test | Δ cMER [95% CI] p | pref base -> test | Δ pref [95% CI] p |", "|---|---|---|---|---|---|---|"]
    for c, v in r["cells"].items():
        if c == "weighted":
            continue
        md.append(f"| {c} | {v['n_units']} | {v['weight']:.3f} | {v['cmer_base']:.4f} -> {v['cmer_test']:.4f} | {fmt_d(v['d_cmer'])} | {v['pref_base']:.4f} -> {v['pref_test']:.4f} | {fmt_d(v['d_pref'], 3)} |")
    md.append("")
Path("results/test_stats_p4.md").write_text("\n".join(md), encoding="utf-8")
print(f"wrote results/stats_test_p4.jsonl ({len(rows)} pairs) and results/test_stats_p4.md")
