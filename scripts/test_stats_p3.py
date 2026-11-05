#!/usr/bin/env python3
"""Second (P3) TEST look, stats step: paired bootstrap (10,000 resamples, seed 1234; ocrgate.stats, method of test run 1)
for the pre-declared comparisons: P3 primary ensemble vs no-correction, vs Qwen2.5-72B + P1 gate of test run 1, vs the P2 frozen
3-pool ensemble, vs the participant files (ungated, official scorer JSONs of test run 1), plus the single-rewriter secondaries
(full / text-only gate / label oracle vs no gate) and a few declared side-by-sides. Writes results/stats_test_p3.jsonl (raw)
and results/test_stats_p3.md (tables).

    CONFIRM_TEST_RUN=1 $VLLM_PYTHON scripts/test_stats_p3.py
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


P3 = one_run("ensemble_test_test-p3-4fam-L3-p3frozen")
P2 = one_run("ensemble_test_test-p2-3pool-agree-p3frozen")
NOCORR = "runs/20261029T084853Z_nocorrect_test"
Q72_T1 = "runs/20261030T070207Z_zeroshot_gate_test_q72b-frozen_qwen06_pooled"      # test run 1: Qwen2.5-72B zs + P1 full gate
Q72_T1_NOGATE = "runs/20261030T070141Z_zeroshot_nogate_test_q72b-frozen"
REWRITERS = [("lora24b-p3frozen", "LoRA-24B"), ("lora32b-p3frozen", "LoRA-32B"), ("lora72b-p3frozen", "LoRA-72B"),
             ("lora72b-syn-p3frozen", "LoRA-72B-syn"), ("q72b-p2frozen", "Qwen2.5-72B zero-shot (P2 script, secondary)")]
GATES = [("P2 full gate (global)", "zeroshot_gate_test_{t}_p2_pooled_r-global"),
         ("P2 text-only gate (global)", "zeroshot_gate_test_{t}_p2_pooled_nolp_r-global"),
         ("label oracle (upper bound)", "zeroshot_oraclelabel_test_{t}")]
rows = []


def add(label, kind, base, test, **extra):
    print(f"[stats] {label}", file=sys.stderr, flush=True)
    r = run_stats(["--base", base, "--test", test]); r["label"] = label; r["kind"] = kind; r.update(extra); rows.append(r)


# A. primary vs the pre-declared references
add("P3 primary ensemble vs no-correction (OCR)", "primary", NOCORR, P3)
add("P3 primary ensemble vs Qwen2.5-72B + P1 full gate (test run 1, best system there)", "primary", Q72_T1, P3)
add("P3 primary ensemble vs P2 frozen 3-pool ensemble (secondary, this run)", "primary", P2, P3)
add("P3 primary ensemble vs LoRA-72B + P2 full gate (pre-declared single-rewriter reference = best single on dev; this run)", "primary", one_run("zeroshot_gate_test_lora72b-p3frozen_p2_pooled_r-global"), P3)
add("P2 frozen 3-pool ensemble vs no-correction (OCR)", "secondary", NOCORR, P2)
add("P2 frozen 3-pool ensemble vs Qwen2.5-72B + P1 full gate (test run 1)", "secondary", Q72_T1, P2)
# B. participants (ungated, the official scorer JSONs recomputed in test run 1) vs the P3 primary
SYS = __import__("os").path.join(__import__("os").environ.get("HIPE_EVAL_DIR", "HIPE-OCRepair-2026-eval"), "data/systems"); OUT = "runs/external_gated_test"
systems = {}
for f in sorted(glob.glob(f"{SYS}/*_run[0-9].jsonl")):
    stem = Path(f).stem; team, rest = stem.split("_", 1)
    label = rest[len("hipe-ocrepair-bench_v0.9_"):].rsplit("_run", 1)[0].replace("_masked-test_", "_test_")
    systems.setdefault((team, stem.rsplit("_run", 1)[1]), {})[label] = (f, stem)
for team, run_n in [("blocr", "1"), ("bnf-mistral", "1"), ("bnf-mistral", "2"), ("bnf-mistral", "3"), ("l3i", "1"), ("l3i", "2")]:
    cells = systems[(team, run_n)]
    base = {lab: {"file": f, "score": f"{OUT}/scores/{stem}.ungated.json"} for lab, (f, stem) in cells.items()}
    for v in base.values():
        for p in (v["file"], v["score"]):
            if not Path(p).is_file():
                sys.exit(f"missing {p}")
    label = f"P3 primary ensemble vs {team} run{run_n} (ungated participant file)"
    print(f"[stats] {label}", file=sys.stderr, flush=True)
    r = run_stats(["--external-base", json.dumps(base), "--test", P3, "--split", "test"])
    r["label"] = label; r["kind"] = "participant"; r["team"] = team; r["run"] = run_n; r["n_cells"] = len(cells); rows.append(r)
# C. single-rewriter secondaries: gates vs no gate
for t, name in REWRITERS:
    base = one_run(f"zeroshot_nogate_test_{t}")
    for glabel, pat in GATES:
        add(f"{name}: {glabel} vs no gate", "rewriter", base, one_run(pat.format(t=t)))
# D. declared side-by-sides among the single systems
add("LoRA-72B + P2 full gate vs LoRA-32B + P2 full gate", "rewriter", one_run("zeroshot_gate_test_lora32b-p3frozen_p2_pooled_r-global"), one_run("zeroshot_gate_test_lora72b-p3frozen_p2_pooled_r-global"))
add("LoRA-72B-syn + P2 full gate vs LoRA-72B + P2 full gate", "rewriter", one_run("zeroshot_gate_test_lora72b-p3frozen_p2_pooled_r-global"), one_run("zeroshot_gate_test_lora72b-syn-p3frozen_p2_pooled_r-global"))
add("LoRA-72B-syn ungated vs LoRA-72B ungated", "rewriter", one_run("zeroshot_nogate_test_lora72b-p3frozen"), one_run("zeroshot_nogate_test_lora72b-syn-p3frozen"))
add("LoRA-72B + P2 full gate vs Qwen2.5-72B + P1 full gate (test run 1)", "rewriter", Q72_T1, one_run("zeroshot_gate_test_lora72b-p3frozen_p2_pooled_r-global"))
add("Qwen2.5-72B zero-shot ungated: this run (P2 script) vs test run 1 (same model, prompt, decoding)", "rewriter", Q72_T1_NOGATE, one_run("zeroshot_nogate_test_q72b-p2frozen"))

Path("results/stats_test_p3.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def fmt_d(d, nd=4):
    lo, hi = d["ci95"]; p = d["p_two_sided"]
    ps = f"{p:.4f}" if p >= 2.1e-4 else "<0.0002"
    return f"{d['point']:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}] p={ps}"


md = ["# Second (P3) TEST look: paired bootstrap (test split; one pass, nothing re-selected)", "",
      "Tool: `ocrgate/stats.py` via `scripts/test_stats_p3.py`; raw output `results/stats_test_p3.jsonl`. Method identical to",
      "`results/test_stats.md` (test run 1): per-unit counts recomputed with the official scorer's functions and checked against every",
      "official scorer JSON (4-decimal agreement enforced before any resampling); paired bootstrap over units within cells,",
      "10,000 resamples, `numpy.random.default_rng(1234)`; weighted value = official design weights over the 8 official cells",
      "(DTA cells 1/3 each, others 1; 590 scored units of 740). Δ = test − base; negative Δ cMER and positive Δ pref are improvements;",
      "95% CI = 2.5/97.5 percentiles; two-sided p with the +1 correction (floor ≈ 0.0002). Participant files are the submitted (ungated)",
      "outputs, scored by the official scorer in test run 1 (`runs/external_gated_test/scores/*.ungated.json`); the P3 ensemble is 'test'.", ""]
for kind, title in (("primary", "Primary (P3 4-family 3-level ensemble) vs the pre-declared references"),
                    ("participant", "P3 primary ensemble vs participant submissions (ungated), weighted over the cells each system covers"),
                    ("secondary", "P2 frozen 3-pool ensemble (secondary)"),
                    ("rewriter", "Single rewriters (secondaries): gates vs no gate, and declared side-by-sides")):
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
Path("results/test_stats_p3.md").write_text("\n".join(md), encoding="utf-8")
print(f"wrote results/stats_test_p3.jsonl ({len(rows)} pairs) and results/test_stats_p3.md")
