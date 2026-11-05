"""Phase 1: paired bootstrap over units of one run vs a baseline run (gate vs no gate), cMER and pref.

    python -m ocrgate.stats --config configs/phase1.yaml --base runs/<nogate_run> --test runs/<gate_run> [--n 10000] [--seed 1234]

Per-unit statistics are recomputed with the official scorer's functions (`norm`, `process_characters`,
`mer_from_counts`): per-cell cMER = micro MER = sum(S+D+I)/sum(H+S+D+I) over the units of the cell, and pref =
mean over units of sign(cMER_ocr - cMER_sys). Both per-cell values are checked against the official scorer JSON of
the run (they must agree within the scorer's 4-decimal rounding, |diff| <= 5e-5) before any bootstrap is done. Resampling: units are drawn with replacement
within each cell (paired: the same unit indices for both runs); the weighted mean uses the official cell weights
over the official cells. 95% CI = 2.5/97.5 percentiles of the bootstrap distribution of the difference
(test - base); two-sided p = 2 * min(P(diff <= 0), P(diff >= 0)) (with the +1 correction), floor 1/n.
Overproof is reported per cell but is not in the weighted mean. Output: JSON on stdout (no text).

Test split (single approved run only; needs --allow-test and CONFIRM_TEST_RUN=1): units flagged
`ground_truth.exclude_from_icdar_evaluation = True` are dropped exactly as the official scorer does (DTA test cells:
30 of 80 units scored). External submission files (participants' public test outputs, ungated vs gated with the
text-only gate) are compared with --external-base / --external-test: JSON {cell label: {"file": submission jsonl,
"score": official scorer JSON}}; outputs are aligned to the reference by document_id with the scorer's own rules
(last record of a duplicated id wins; missing / None / "None" / "" output = OCR input, as score_one.py imputes).
Cells absent from a submission are left out and the weighted mean covers the official cells present (reported).
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from hipe_ocrepair_scorer.ocrepair_eval import mer_from_counts, norm, process_characters

from .data import ROOT, load_config, read_jsonl, resolve, split_cells
from .score import select_refs


def unit_counts(gold_norm: str, hyp: str):
    o = process_characters(gold_norm, norm(hyp))
    return o.hits, o.substitutions, o.deletions, o.insertions


def load_outputs(run_dir: Path) -> dict[str, list[str]]:
    out: dict[str, list] = {}
    for line in (run_dir / "log.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out.setdefault(r["cell"], []).append(r["output"])
    return out


def _excluded(ref: dict) -> bool:
    gt = ref.get("ground_truth")
    return isinstance(gt, dict) and gt.get("exclude_from_icdar_evaluation") is True


def per_unit(cfg, split, outputs):
    """Per cell: arrays err (S+D+I), tot (H+S+D+I), pref (+1/0/-1 vs OCR) for every scored unit
    (units the official scorer excludes via ground_truth.exclude_from_icdar_evaluation are dropped)."""
    res = {}
    for cell in split_cells(cfg, split):
        refs = select_refs(cell)
        if cell.label not in outputs:
            continue
        hyps = outputs[cell.label]
        assert len(refs) == len(hyps), cell.label
        res[cell.label] = _cell_arrays(cell, [(r, h) for r, h in zip(refs, hyps) if not _excluded(r)])
    return res


def load_external(cfg, split, files: dict[str, str]) -> dict[str, list[str]]:
    """Outputs of an external submission file per cell, aligned to the reference like the official scorer."""
    outputs = {}
    for cell in split_cells(cfg, split):
        path = files.get(cell.label)
        if not path:
            continue
        by_id = {}
        for rec in read_jsonl(resolve(path)):
            by_id[rec["document_metadata"]["document_id"]] = rec  # duplicate id: the last record wins (scorer dict)
        hyps = []
        for ref in select_refs(cell):
            src = ref["ocr_hypothesis"]["transcription_unit"]
            rec = by_id.get(ref["document_metadata"]["document_id"])
            if rec is None:
                hyps.append("")  # absent document: the scorer scores an empty output (none occur in the 2026 files)
                continue
            post = rec.get("ocr_postcorrection_output")
            out = post.get("transcription_unit") if isinstance(post, dict) else None
            hyps.append(src if (out is None or out == "None" or out == "") else out)  # score_one.py imputation
        outputs[cell.label] = hyps
    return outputs


def _cell_arrays(cell, pairs):
        err, tot, pref = [], [], []
        for ref, hyp in pairs:
            gold = norm(ref["ground_truth"]["transcription_unit"])
            h, s, d, i = unit_counts(gold, hyp)
            h0, s0, d0, i0 = unit_counts(gold, ref["ocr_hypothesis"]["transcription_unit"])
            c_sys, c_ocr = mer_from_counts(h, s, d, i), mer_from_counts(h0, s0, d0, i0)
            err.append(s + d + i); tot.append(h + s + d + i)
            pref.append(1 if c_sys < c_ocr else (-1 if c_sys > c_ocr else 0))
        return (np.array(err, float), np.array(tot, float), np.array(pref, float), cell.weight, cell.official)


def check_against_scorer(res, summary):
    for c in summary["cells"]:
        err, tot, pref, _, _ = res[c["cell"]]
        cm, pf = err.sum() / tot.sum(), pref.mean()
        # the official scorer JSON reports values rounded to 4 decimals
        if abs(cm - c["cmer_micro"]) > 5.0e-5 + 1e-9 or abs(pf - c["pref_score_cmer_macro"]) > 5.0e-5 + 1e-9:
            raise SystemExit(f"{c['cell']}: recomputed {cm:.6f}/{pf:.4f} != scorer {c['cmer_micro']:.6f}/{c['pref_score_cmer_macro']:.4f}")


def check_against_scorer_json(res, score_paths: dict[str, str]):
    """Same check as check_against_scorer, against the official scorer JSON written by score_one.py."""
    for label, (err, tot, pref, _, _) in res.items():
        p = score_paths[label]
        j = json.loads(resolve(p).read_text())["averaged_scores"]
        cm_off, pf_off = j["cmer_micro"][0], j["pref_score_cmer_macro"][0]
        cm, pf = err.sum() / tot.sum(), pref.mean()
        if abs(cm - cm_off) > 5.0e-5 + 1e-9 or abs(pf - pf_off) > 5.0e-5 + 1e-9:
            raise SystemExit(f"{label}: recomputed {cm:.6f}/{pf:.4f} != scorer {cm_off:.6f}/{pf_off:.4f} ({p})")


def bootstrap(res_b, res_t, n, seed):
    rng = np.random.default_rng(seed)
    cells = [c for c in res_b if c in res_t]
    out = {}
    w_cm_b, w_cm_t, w_pf_b, w_pf_t, wsum = 0, 0, 0, 0, 0.0
    for c in cells:
        eb, tb, pb, w, off = res_b[c]; et, tt, pt, _, _ = res_t[c]
        m = len(eb)
        idx = rng.integers(0, m, size=(n, m))
        cm_b = eb[idx].sum(1) / tb[idx].sum(1); cm_t = et[idx].sum(1) / tt[idx].sum(1)
        pf_b = pb[idx].mean(1); pf_t = pt[idx].mean(1)
        out[c] = {"n_units": m, "weight": w, "official": off,
                  "cmer_base": float(eb.sum() / tb.sum()), "cmer_test": float(et.sum() / tt.sum()),
                  "pref_base": float(pb.mean()), "pref_test": float(pt.mean()),
                  "d_cmer": summarize(cm_t - cm_b, float(et.sum() / tt.sum() - eb.sum() / tb.sum())),
                  "d_pref": summarize(pf_t - pf_b, float(pt.mean() - pb.mean()))}
        if off:
            w_cm_b = w_cm_b + w * cm_b; w_cm_t = w_cm_t + w * cm_t; w_pf_b = w_pf_b + w * pf_b; w_pf_t = w_pf_t + w * pf_t; wsum += w
    point = {k: sum(res[c][0].sum() / res[c][1].sum() * res[c][3] for c in cells if res[c][4]) / wsum for k, res in (("b", res_b), ("t", res_t))}
    pointp = {k: sum(res[c][2].mean() * res[c][3] for c in cells if res[c][4]) / wsum for k, res in (("b", res_b), ("t", res_t))}
    out["weighted"] = {"n_official_cells": sum(1 for c in cells if res_b[c][4]), "weight_sum": wsum,
                       "cmer_base": point["b"], "cmer_test": point["t"], "pref_base": pointp["b"], "pref_test": pointp["t"],
                       "d_cmer": summarize((w_cm_t - w_cm_b) / wsum, point["t"] - point["b"]),
                       "d_pref": summarize((w_pf_t - w_pf_b) / wsum, pointp["t"] - pointp["b"])}
    return out


def summarize(d: np.ndarray, point: float) -> dict:
    n = len(d)
    p_le = (np.sum(d <= 0) + 1) / (n + 1); p_ge = (np.sum(d >= 0) + 1) / (n + 1)
    return {"point": point, "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_two_sided": float(min(1.0, 2 * min(p_le, p_ge)))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--base", default=None, help="runs/<run> (our rewriter, e.g. no gate)")
    ap.add_argument("--test", default=None, help="runs/<run> (e.g. gated)")
    ap.add_argument("--external-base", default=None, help='JSON {cell label: {"file": jsonl, "score": official json}} or a path to it')
    ap.add_argument("--external-test", default=None)
    ap.add_argument("--split", default=None, help="required with --external-*; otherwise taken from the run summaries")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--allow-test", action="store_true", help="only for the single, approved test run (also needs CONFIRM_TEST_RUN=1)")
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)

    def side(run, ext):
        if run:
            sm = json.loads((ROOT / run / "summary.json").read_text())
            return sm["split"], sm["run_id"], per_unit(cfg, sm["split"], load_outputs(ROOT / run)), sm, None
        spec = json.loads(Path(ext).read_text()) if Path(ext).is_file() else json.loads(ext)
        files = {c: v["file"] for c, v in spec.items()}; scores = {c: v["score"] for c, v in spec.items()}
        return args.split, spec, per_unit(cfg, args.split, load_external(cfg, args.split, files)), None, scores

    split_b, id_b, rb, sm_b, sc_b = side(args.base, args.external_base)
    split_t, id_t, rt, sm_t, sc_t = side(args.test, args.external_test)
    if split_b != split_t:
        raise SystemExit(f"split mismatch: {split_b} vs {split_t}")
    if split_b == "test" and not (args.allow_test and os.environ.get("CONFIRM_TEST_RUN") == "1"):
        raise SystemExit("the test split is locked (pass --allow-test and set CONFIRM_TEST_RUN=1 only for the approved final run)")
    for res, sm, sc in ((rb, sm_b, sc_b), (rt, sm_t, sc_t)):
        check_against_scorer(res, sm) if sm is not None else check_against_scorer_json(res, sc)
    res = bootstrap(rb, rt, args.n, args.seed)
    print(json.dumps({"base": id_b, "test": id_t, "split": split_b, "n_resamples": args.n, "seed": args.seed, "cells": res}))


if __name__ == "__main__":
    main()
