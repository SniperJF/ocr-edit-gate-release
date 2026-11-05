#!/usr/bin/env python3
"""Single TEST run, step 3: paired bootstrap (10,000 resamples, seed 1234; method of results/p1_stats.md / ocrgate.stats)
for every rewriter (full gate, text-only gate and label oracle vs no gate) and every participant submission
(text-only gate vs ungated). Writes results/stats_test.jsonl (raw) and results/test_stats.md (tables).

    CONFIRM_TEST_RUN=1 $VLLM_PYTHON scripts/test_stats.py
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

REWRITERS = [("8b", "Qwen3-8B zero-shot"), ("32b", "Qwen3-32B zero-shot"), ("lora8b", "Qwen3-8B + LoRA (P1-b)"),
             ("mistral24b", "Mistral-Small-24B-Instruct zero-shot"), ("q72b", "Qwen2.5-72B-Instruct zero-shot")]
GATES = [("full gate (qwen06_pooled)", "zeroshot_gate_test_{tag}-frozen_qwen06_pooled"),
         ("text-only gate (qwen06_pooled_nolp)", "zeroshot_gate_test_{tag}-frozen_qwen06_pooled_nolp"),
         ("label oracle (upper bound)", "zeroshot_oraclelabel_test_{tag}-frozen")]


def one_run(pattern: str) -> str:
    g = sorted(glob.glob(f"runs/*_{pattern}"))
    if len(g) != 1:
        sys.exit(f"expected exactly one run for {pattern}, found {g}")
    return Path(g[0]).name


def run_stats(args: list[str]) -> dict:
    out = subprocess.run([PY, "-m", "ocrgate.stats", *args, "--n", "10000", "--seed", "1234", "--allow-test"],
                         capture_output=True, text=True, env={**os.environ, "CONFIRM_TEST_RUN": "1"})
    if out.returncode != 0:
        sys.exit(f"stats failed for {args}:\n{out.stderr[-2000:]}")
    return json.loads(out.stdout)


rows = []
# A. our rewriters
for tag, name in REWRITERS:
    base = one_run(f"zeroshot_nogate_test_{tag}-frozen")
    for glabel, pat in GATES:
        test = one_run(pat.format(tag=tag))
        print(f"[stats] {name}: {glabel}", file=sys.stderr, flush=True)
        r = run_stats(["--base", f"runs/{base}", "--test", f"runs/{test}"])
        r["label"] = f"{name}: {glabel} vs no gate"; r["kind"] = "rewriter"; rows.append(r)

# B. participants: ungated submission vs the same file gated with the frozen text-only gate
SYS = __import__("os").path.join(__import__("os").environ.get("HIPE_EVAL_DIR", "HIPE-OCRepair-2026-eval"), "data/systems"); OUT = "runs/external_gated_test"
subs = sorted(glob.glob(f"{SYS}/*_run[0-9].jsonl"))
systems = {}
for f in subs:
    stem = Path(f).stem; team, rest = stem.split("_", 1)
    label = rest[len("hipe-ocrepair-bench_v0.9_"):].rsplit("_run", 1)[0].replace("_masked-test_", "_test_")
    run_n = stem.rsplit("_run", 1)[1]
    systems.setdefault((team, run_n), {})[label] = (f, stem, rest)
for (team, run_n), cells in sorted(systems.items(), key=lambda kv: (kv[0][0].lower(), kv[0][1])):
    base, test = {}, {}
    for label, (f, stem, rest) in cells.items():
        gated = f"{OUT}/{team}-tgate_{rest}.jsonl"
        for p in (f, gated, f"{OUT}/scores/{stem}.ungated.json", f"{OUT}/scores/{team}-tgate_{rest}.json"):
            if not Path(p).is_file():
                sys.exit(f"missing {p}")
        base[label] = {"file": f, "score": f"{OUT}/scores/{stem}.ungated.json"}
        test[label] = {"file": gated, "score": f"{OUT}/scores/{team}-tgate_{rest}.json"}
    print(f"[stats] participant {team} run{run_n} ({len(cells)} cells)", file=sys.stderr, flush=True)
    r = run_stats(["--external-base", json.dumps(base), "--external-test", json.dumps(test), "--split", "test"])
    r["label"] = f"{team} run{run_n}: text-only gate vs ungated submission"; r["kind"] = "participant"
    r["team"] = team; r["run"] = run_n; r["n_cells"] = len(cells); rows.append(r)

Path("results/stats_test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def fmt_d(d, nd=4):
    lo, hi = d["ci95"]; p = d["p_two_sided"]
    ps = f"{p:.4f}" if p >= 1e-4 else f"<{1e-4:.4f}"
    return f"{d['point']:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}] p={ps}"


md = ["# TEST run, step 3: paired bootstrap, gate vs no gate (test split, single approved run)", "",
      "Tool: `ocrgate/stats.py` via `scripts/test_stats.py`; raw output `results/stats_test.jsonl`. Method identical to",
      "`results/p1_stats.md`: per-unit counts recomputed with the official scorer's functions and checked against every",
      "official scorer JSON (4-decimal agreement enforced before any resampling); paired bootstrap over units within cells,",
      "10,000 resamples, `numpy.random.default_rng(1234)`; weighted value = official design weights over the official cells",
      "(DTA cells 1/3 each, others 1; 740 reference units of which 590 are scored: the scorer drops the 50 flagged units of each",
      "DTA test cell). Δ = test − base; negative Δ cMER and positive Δ pref are improvements. 95% CI = 2.5/97.5 percentiles;",
      "two-sided p with the +1 correction (floor ≈ 0.0002).", ""]
for kind, title in (("rewriter", "Our rewriters (gate vs no gate), weighted over the 8 official test cells"),
                    ("participant", "Participant submissions (frozen text-only gate vs the submitted file), weighted over the cells each system covers")):
    md += [f"## {title}", "", "| comparison | cells | cMER base -> test | Δ cMER [95% CI] p | pref base -> test | Δ pref [95% CI] p |", "|---|---|---|---|---|---|"]
    for r in rows:
        if r["kind"] != kind:
            continue
        w = r["cells"]["weighted"]
        md.append(f"| {r['label']} | {w['n_official_cells']}/8 | {w['cmer_base']:.4f} -> {w['cmer_test']:.4f} | {fmt_d(w['d_cmer'])} | {w['pref_base']:.4f} -> {w['pref_test']:.4f} | {fmt_d(w['d_pref'], 3)} |")
    md.append("")
md += ["## Per cell", ""]
for r in rows:
    b = r["base"] if isinstance(r["base"], str) else "submission files"; t = r["test"] if isinstance(r["test"], str) else "gated files in runs/external_gated_test"
    md += [f"### {r['label']}", "", f"base `{b}`, test `{t}`", "",
           "| cell | units scored | weight | cMER base -> test | Δ cMER [95% CI] p | pref base -> test | Δ pref [95% CI] p |", "|---|---|---|---|---|---|---|"]
    for c, v in r["cells"].items():
        if c == "weighted":
            continue
        md.append(f"| {c} | {v['n_units']} | {v['weight']:.3f} | {v['cmer_base']:.4f} -> {v['cmer_test']:.4f} | {fmt_d(v['d_cmer'])} | {v['pref_base']:.4f} -> {v['pref_test']:.4f} | {fmt_d(v['d_pref'], 3)} |")
    md.append("")
Path("results/test_stats.md").write_text("\n".join(md), encoding="utf-8")
print(f"wrote results/stats_test.jsonl ({len(rows)} pairs) and results/test_stats.md")
