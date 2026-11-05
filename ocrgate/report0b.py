"""Phase 0b comparison tables from results/runs.jsonl (every number is official-scorer output)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_runs(path: Path = ROOT / "results" / "runs.jsonl") -> dict[str, dict]:
    runs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            runs[r["run_id"]] = r
    return runs


def _f(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def cell_table(columns: list[tuple[str, dict, tuple[str, ...]]]) -> list[str]:
    """columns: (header label, run summary, metrics to show: 'cmer' and/or 'pref').

    One row per cell (all cells of the first run, official and not), then the weighted mean over the
    official cells, read from the run summaries (`cmer_micro_weighted`, `pref_score_cmer_macro_weighted`).
    """
    heads = ["cell", "units", "weight"]
    for label, _, metrics in columns:
        for m in metrics:
            heads.append(f"{label} {'cMER' if m == 'cmer' else 'pref'}")
    lines = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    first = columns[0][1]
    n_units = _units_per_cell(first)
    for c0 in first["cells"]:
        w = f"{c0['weight']:.3f}" + ("" if c0["official"] else " (not in mean)")
        row = [c0["cell"], str(n_units.get(c0["cell"], "")), w]
        for _, run, metrics in columns:
            c = next((c for c in run["cells"] if c["cell"] == c0["cell"]), None)
            for m in metrics:
                if c is None:
                    row.append("-")
                elif m == "cmer":
                    row.append(_f(c["cmer_micro"]))
                else:
                    row.append(_f(c["pref_score_cmer_macro"]))
        lines.append("| " + " | ".join(row) + " |")
    row = ["**weighted mean (official cells)**", str(first["n_units"]), ""]
    for _, run, metrics in columns:
        for m in metrics:
            row.append("**" + _f(run["cmer_micro_weighted"] if m == "cmer" else run["pref_score_cmer_macro_weighted"]) + "**")
    lines.append("| " + " | ".join(row) + " |")
    return lines


def _units_per_cell(run: dict) -> dict[str, int]:
    """Unit counts per cell from the run's scorer JSONs (fall back to the log)."""
    out = {}
    log = ROOT / "runs" / run["run_id"] / "log.jsonl"
    if log.is_file():
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["cell"]] = out.get(r["cell"], 0) + 1
    return out


def summary_table(runs: list[tuple[str, dict]]) -> list[str]:
    lines = ["| system | model | thinking | units | cMER (weighted) | pref (weighted) | gen. wall (s) | wall (s) | run_id |",
             "|---|---|---|---|---|---|---|---|---|"]
    for label, r in runs:
        th = "-" if r.get("thinking") is None else ("on" if r["thinking"] else "off")
        lines.append(
            f"| {label} | {r.get('model') or '-'} | {th} | {r['n_units']} | {_f(r['cmer_micro_weighted'])} | "
            f"{_f(r['pref_score_cmer_macro_weighted'])} | {r.get('generation_wall_s', '-')} | {r['wall_s']} | `{r['run_id']}` |"
        )
    return lines


def compare_cells(a: dict, b: dict) -> dict:
    """Per official cell: does run `a` beat run `b` on cMER (lower) and pref (higher)? Ties count as not beating."""
    out = {}
    for ca in a["cells"]:
        if not ca["official"]:
            continue
        cb = next(c for c in b["cells"] if c["cell"] == ca["cell"])
        out[ca["cell"]] = {
            "cmer": ca["cmer_micro"] < cb["cmer_micro"],
            "pref": ca["pref_score_cmer_macro"] > cb["pref_score_cmer_macro"],
        }
    return out
