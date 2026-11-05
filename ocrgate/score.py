"""Score submission files with the OFFICIAL scorer script (HIPE-OCRepair-2026-eval/lib/score_one.py).

score_one.py scores one hypothesis file against the reference whose stem it derives from the
hypothesis filename, and writes a JSON with per-dataset `fold_scores`. We call it once per cell,
read `cmer_micro` and `pref_score_cmer_macro` for that cell, and then combine cells with the
official competition weights (weighted mean, as in lib/build_rankings.py).

Command line (one hypothesis file = one cell; the reference is found from the file name exactly as the
official scorer does, `<team>_<reference-stem>_run<N>.jsonl`):

    python -m ocrgate.score --hypothesis <team>_hipe-ocrepair-bench_v0.9_icdar2017_v1.1_test_en_run1.jsonl \
        [--reference-dir DIR] [--scorer PATH/score_one.py] [--out-dir DIR] [--json]

Defaults: --reference-dir = $HIPE_EVAL_DIR/data/reference (the official test references; for dev/train cells pass the
directory of the data repo that holds the reference file, e.g. $HIPE_DATA_DIR/data/v0.9/icdar2017/en), --scorer =
$HIPE_EVAL_DIR/lib/score_one.py, --out-dir = a temporary directory. The official scorer runs in a subprocess with
`python -I` (hipe-ocrepair-scorer must be installed in the current interpreter).
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from .data import ROOT, Cell, eval_dir, read_jsonl, resolve, write_jsonl


def select_refs(cell: Cell, limit: int = 0, subset: dict | None = None) -> list[dict]:
    """Reference records of a cell, optionally cut to the first `limit` units or to the document ids
    listed in `subset[cell.label]` (Phase 0b probe subset; order of the reference file is kept)."""
    refs = read_jsonl(cell.path)
    if subset is not None:
        ids = set(subset.get(cell.label, []))
        refs = [r for r in refs if r["document_metadata"]["document_id"] in ids]
    if limit:
        refs = refs[:limit]
    return refs


def _reference_dir(cells: list[Cell], cache: Path, limit: int = 0, subset: dict | None = None) -> Path:
    """score_one.py expects a flat reference directory; build one from symlinks.

    With a debug `limit` or a `subset`, write reduced reference copies so that only the predicted
    units are scored.
    """
    cache.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        link = cache / cell.path.name
        if link.is_symlink() or link.exists():
            link.unlink()
        if limit or subset is not None:
            write_jsonl(link, select_refs(cell, limit, subset))
        else:
            link.symlink_to(cell.path)
    return cache


def scorer_script(cfg: dict | None = None) -> Path:
    """Path of the official lib/score_one.py (config key data.scorer_script, resolved through HIPE_EVAL_DIR)."""
    rel = (cfg or {}).get("data", {}).get("scorer_script", "HIPE-OCRepair-2026-eval/lib/score_one.py")
    return resolve(rel)


def run_official_scorer(hyp: Path, ref_dir: Path, out_json: Path, log_file: Path, scorer: Path) -> dict:
    """Run score_one.py once and return the scorer JSON (one dataset fold)."""
    if not scorer.is_file():
        raise SystemExit(f"official scorer not found at {scorer} (run scripts/fetch_data.sh or set HIPE_EVAL_DIR)")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-I", str(scorer), "--hypothesis", str(hyp), "--reference-dir", str(ref_dir),
           "--output", str(out_json), "--log-file", str(log_file)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_json.is_file():
        raise RuntimeError(f"Official scorer failed for {hyp.name} (exit {proc.returncode}):\n{proc.stderr[-2000:]}")
    res = json.loads(out_json.read_text(encoding="utf-8"))
    folds = res["fold_scores"]
    if len(folds) != 1:
        raise RuntimeError(f"Expected one dataset fold in {out_json}, got {list(folds)}")
    return next(iter(folds.values()))


def score_cell(cell: Cell, hyp: Path, ref_dir: Path, out_dir: Path, cfg: dict) -> dict:
    out_json = out_dir / f"{hyp.stem}.json"
    log_file = out_dir / f"{hyp.stem}.log"
    fold = run_official_scorer(hyp, ref_dir, out_json, log_file, scorer_script(cfg))
    try:
        rel = str(out_json.relative_to(ROOT))
    except ValueError:
        rel = str(out_json)
    return {
        "cell": cell.label,
        "weight": cell.weight,
        "official": cell.official,
        "cmer_micro": fold["cmer_micro"][0],
        "cmer_micro_ci": fold["cmer_micro"][1:],
        "pref_score_cmer_macro": fold["pref_score_cmer_macro"][0],
        "pref_score_cmer_macro_ci": fold["pref_score_cmer_macro"][1:],
        "cmer_macro": fold["cmer_macro"][0],
        "wmer_micro": fold["wmer_micro"][0],
        "scorer_json": rel,
    }


def weighted(rows: list[dict], key: str) -> float | None:
    pairs = [(r[key], r["weight"]) for r in rows if r["official"] and r[key] is not None]
    total = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total if total else None


def score_run(
    cells: list[Cell], pred_dir: Path, names: dict[str, str], out_dir: Path, cfg: dict, limit: int = 0,
    subset: dict | None = None,
) -> dict:
    ref_dir = _reference_dir(cells, out_dir / "refs", limit, subset)
    rows = [score_cell(c, pred_dir / names[c.stem], ref_dir, out_dir, cfg) for c in cells]
    return {
        "cmer_micro_weighted": weighted(rows, "cmer_micro"),
        "pref_score_cmer_macro_weighted": weighted(rows, "pref_score_cmer_macro"),
        "n_official_cells": sum(r["official"] for r in rows),
        "cells": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hypothesis", required=True, help="one submission file, <team>_<reference-stem>_run<N>.jsonl")
    ap.add_argument("--reference-dir", default=None, help="directory holding the reference jsonl (default $HIPE_EVAL_DIR/data/reference)")
    ap.add_argument("--scorer", default=None, help="path of lib/score_one.py (default $HIPE_EVAL_DIR/lib/score_one.py)")
    ap.add_argument("--out-dir", default=None, help="where the scorer JSON and log are written (default: a temporary directory)")
    ap.add_argument("--json", action="store_true", help="print the full scorer fold as JSON instead of the two headline numbers")
    args = ap.parse_args()
    hyp = Path(args.hypothesis).expanduser().resolve()
    ref_dir = Path(args.reference_dir).expanduser() if args.reference_dir else eval_dir() / "data" / "reference"
    scorer = Path(args.scorer).expanduser() if args.scorer else scorer_script()
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else Path(tempfile.mkdtemp(prefix="ocrgate_score_"))
    fold = run_official_scorer(hyp, ref_dir, out_dir / f"{hyp.stem}.json", out_dir / f"{hyp.stem}.log", scorer)
    if args.json:
        print(json.dumps(fold, indent=1))
    else:
        print(f"{hyp.name}: cmer_micro={fold['cmer_micro'][0]} [{fold['cmer_micro'][1]}, {fold['cmer_micro'][2]}] "
              f"pref_score_cmer_macro={fold['pref_score_cmer_macro'][0]} [{fold['pref_score_cmer_macro'][1]}, {fold['pref_score_cmer_macro'][2]}]"
              f"  (scorer JSON: {out_dir / (hyp.stem + '.json')})")


if __name__ == "__main__":
    main()
