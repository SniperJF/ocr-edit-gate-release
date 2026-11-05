"""Render results/phase0.md from results/runs.jsonl (generated file; do not edit by hand)."""

import json
import sys
from pathlib import Path

HEADER = """# Phase 0 kill-test results

Generated from `results/runs.jsonl` by `python -m ocrgate.report`. Every number below is read from
the official scorer output (`HIPE-OCRepair-2026-eval/lib/score_one.py`, hipe-ocrepair-scorer 0.9.9).

- **cMER** = weighted mean over cells of per-cell `cmer_micro` (lower is better).
- **pref** = weighted mean over cells of per-cell `pref_score_cmer_macro` (higher is better).
- Weights follow the official competition design (DTA levels 1/3 each, other cells 1); only
  `official` cells enter the mean. Dev has no icdar2017-fr file, so **dev means cover 7 of 8 cells**;
  `overproof` is scored but not in the mean (not an official competition cell).
- `oracle_gated` uses the ground truth to choose which spans may change: it is an upper-bound
  diagnostic, not a valid system.
- Runs with `--limit` (debug subsets) and Phase 0b probe-subset runs (`subset`) are omitted here; the
  latter are tabulated in `results/FINAL_REPORT_0b.md`.

Phase 0 GO criterion: oracle_gated on dev with cMER < 0.0106 and pref > 0.70 (BLOCR's official
*test* numbers, quoted from the competition overview paper, not produced here).
"""


def _fmt(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def write_report(runs_path: Path, out_path: Path) -> None:
    runs = []
    if runs_path.is_file():
        runs = [json.loads(l) for l in runs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    # Phase 1 runs (train-split rewriters, gates) are tabulated in results/p1a_dev.md / FINAL_REPORT_P1.md.
    runs = [r for r in runs if not r.get("limit") and not r.get("subset") and not r.get("phase")]

    lines = [HEADER, "## Summary (all runs)\n"]
    lines.append("| system | split | model | cMER (weighted) | pref (weighted) | cells | units | wall time (s) | run_id | commit |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        lines.append(
            f"| {r['system']} | {r['split']} | {r['model'] or '-'} | {_fmt(r['cmer_micro_weighted'])} | "
            f"{_fmt(r['pref_score_cmer_macro_weighted'])} | {r['n_official_cells']} | {r['n_units']} | "
            f"{r['wall_s']} | `{r['run_id']}` | `{r['git_commit'][:8]}{'+dirty' if r['git_dirty'] else ''}` |"
        )

    latest = {}
    for r in runs:
        latest[(r["split"], r["system"])] = r
    for split in sorted({s for s, _ in latest}):
        sel = [latest[k] for k in sorted(latest) if k[0] == split]
        lines.append(f"\n## Per-cell scores, split `{split}` (latest run per system)\n")
        lines.append("cMER = `cmer_micro` [95% CI]; pref = `pref_score_cmer_macro`.\n")
        lines.append("| cell | weight | " + " | ".join(f"{r['system']} cMER | {r['system']} pref" for r in sel) + " |")
        lines.append("|---|---|" + "---|---|" * len(sel))
        cells = [c["cell"] for c in sel[0]["cells"]]
        for i, cell in enumerate(cells):
            c0 = sel[0]["cells"][i]
            w = f"{c0['weight']:.3f}" + ("" if c0["official"] else " (not in mean)")
            vals = []
            for r in sel:
                c = next((c for c in r["cells"] if c["cell"] == cell), None)
                if c is None:
                    vals += ["-", "-"]
                    continue
                lo, hi = c["cmer_micro_ci"]
                vals += [f"{_fmt(c['cmer_micro'])} [{_fmt(lo, 3)}, {_fmt(hi, 3)}]", _fmt(c["pref_score_cmer_macro"])]
            lines.append(f"| {cell} | {w} | " + " | ".join(vals) + " |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    write_report(root / "results" / "runs.jsonl", root / "results" / "phase0.md")
    sys.exit(0)
