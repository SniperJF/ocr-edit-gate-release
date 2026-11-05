"""Phase 1: apply an edit-veto gate to a rewrite run's edit blocks and score with the official scorer (dev).

    python -m ocrgate.gate_apply --run runs/<dev_run> --mode none|oracle_span|oracle_label|gate [--gate runs/gates/NAME]
        [--threshold T] [--tag TAG]

Modes:
  none          keep every block (= the rewriter without a gate; sanity check against the run's own score)
  oracle_span   keep the blocks that overlap an ORACLE error span (Phase 0b block filter; uses ground truth)
  oracle_label  keep exactly the blocks with gain > 0 (label oracle, the tightest block-level upper bound; GT)
  gate          keep blocks whose gate score >= threshold; the gate is `lp_threshold` or `learned`
                (runs/gates/NAME/gate.json). The default threshold is the one chosen on held-out TRAIN.

Block-level diagnostics on the run (ROC-AUC, PR-AUC over helpful vs harmful blocks, kept fractions) are
written next to the scores. ORACLE modes are upper-bound diagnostics, not valid systems.

Phase 2: `--rule NAME` applies a decision rule from runs/gates/NAME/rules.json (`ocrgate.rules`: global, editrate,
sysconf, lang, lang_editrate; thresholds chosen on held-out TRAIN) instead of the single stored threshold; the per-block
gate scores are cached in runs/<run>/gate_scores_<gate>.json (reused by later calls and by `ocrgate.ensemble`).
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .data import ROOT, load_config, split_cells, submission_name, submission_record, write_jsonl
from .edits import apply_blocks
from .gate_train import metrics
from .run import finalize_run, git_state, load_subset, new_run_dir
from .score import score_run, select_refs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--run", required=True, help="runs/<run_id> with blocks.jsonl (see ocrgate.blocks)")
    ap.add_argument("--mode", required=True, choices=["none", "oracle_span", "oracle_label", "gate"])
    ap.add_argument("--gate", default=None, help="runs/gates/NAME (mode gate)")
    ap.add_argument("--threshold", type=float, default=None, help="override the gate's stored threshold")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-test", action="store_true", help="only for the single, approved test run (also needs CONFIRM_TEST_RUN=1)")
    ap.add_argument("--rule", default=None, help="Phase 2: decision rule name in <gate>/rules.json")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    run_dir = ROOT / args.run
    src_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    split = src_summary["split"]
    if split == "test" and not (args.allow_test and os.environ.get("CONFIRM_TEST_RUN") == "1"):
        raise SystemExit("the test split is locked (pass --allow-test and set CONFIRM_TEST_RUN=1 only for the approved final run)")
    subset = load_subset(src_summary.get("subset"))
    limit = int(src_summary.get("limit") or 0)
    blocks = [json.loads(l) for l in open(run_dir / "blocks.jsonl", encoding="utf-8")]
    outputs: dict[str, list] = {}  # per cell, in reference order (document ids are not unique)
    for line in (run_dir / "log.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        outputs.setdefault(r["cell"], []).append(r)

    t0 = time.perf_counter()
    gate_meta, threshold = None, None
    if args.mode == "none":
        scores = np.ones(len(blocks))
        threshold = 0.5
    elif args.mode == "oracle_span":
        scores = np.array([1.0 if b["in_oracle_span"] else 0.0 for b in blocks]); threshold = 0.5
    elif args.mode == "oracle_label":
        scores = np.array([1.0 if b["gain"] > 0 else 0.0 for b in blocks]); threshold = 0.5
    else:
        gate_dir = ROOT / args.gate
        gate_meta = json.loads((gate_dir / "gate.json").read_text(encoding="utf-8"))
        threshold = args.threshold if args.threshold is not None else gate_meta["threshold"]
        if gate_meta["type"] == "lp_threshold":
            st = gate_meta["lp_stat"]
            scores = np.array([b[st] if b.get("lp_ok") and b.get(st) is not None else -99.0 for b in blocks])
        else:
            cache = run_dir / f"gate_scores_{Path(args.gate).name}.json"
            if cache.is_file():
                scores = np.array(json.loads(cache.read_text(encoding="utf-8")))
                assert len(scores) == len(blocks), "cached scores do not match blocks.jsonl"
            else:
                from .gate import load_gate, load_tokenizer, predict

                model, std, meta = load_gate(gate_dir, args.device)
                tok = load_tokenizer(meta["encoder"])
                scores = np.array(predict(model, tok, std, blocks, meta["max_len"], meta["use_lp"], device=args.device))
                cache.write_text(json.dumps([float(x) for x in scores]), encoding="utf-8")
    rule = None
    if args.rule:
        from .rules import thresholds_for
        rules = json.loads((ROOT / args.gate / "rules.json").read_text(encoding="utf-8"))
        rule = rules["rules"][args.rule]
        threshold = thresholds_for(rule, blocks, scores)  # per-block thresholds
    keep = scores >= threshold

    # block-level diagnostics (labels from the run's blocks.jsonl; neutral blocks excluded)
    lab = np.array([b["label"] != "neutral" for b in blocks])
    y = np.array([1.0 if b["label"] == "helpful" else 0.0 for b in blocks])
    g = np.array([b["gain"] for b in blocks])
    diag = metrics(y[lab], scores[lab], g[lab], threshold[lab] if isinstance(threshold, np.ndarray) else threshold) if lab.any() else {}
    if isinstance(threshold, np.ndarray):
        diag["threshold"] = {"rule": args.rule, "min": float(threshold.min()), "max": float(threshold.max()), "mean": float(threshold.mean())}
    diag["n_blocks_total"] = len(blocks); diag["n_kept_total"] = int(keep.sum())

    # per-unit outputs
    per_unit: dict[tuple, list] = {}
    for b, k in zip(blocks, keep):
        per_unit.setdefault((b["cell"], b["unit_idx"]), []).append((b["s"], b["e"], b["ds"], b["de"], bool(k)))
    gate_name = Path(args.gate).name if args.gate else None
    system = {"none": f"{src_summary['system']}_nogate", "oracle_span": f"{src_summary['system']}_oraclespan",
              "oracle_label": f"{src_summary['system']}_oraclelabel", "gate": f"{src_summary['system']}_gate"}[args.mode]
    tag = args.tag or (src_summary.get("tag") or "")
    if args.mode == "gate":
        tag = f"{tag}_{gate_name}" if tag else gate_name
        if args.rule:
            tag = f"{tag}_r-{args.rule}"
    cells = split_cells(cfg, split)
    run_id, out_dir = new_run_dir(system, split, tag or None)
    pred_dir = out_dir / "predictions"
    names, n_docs, n_changed = {}, 0, 0
    with open(out_dir / "log.jsonl", "w", encoding="utf-8") as log_f:
        for cell in cells:
            refs = select_refs(cell, limit, subset)
            records = []
            for ui, ref in enumerate(refs):
                did = ref["document_metadata"]["document_id"]
                src = ref["ocr_hypothesis"]["transcription_unit"]
                r = outputs[cell.label][ui]
                if r["document_id"] != did or r["ocr"] != src:
                    raise SystemExit(f"OCR text mismatch for {did}")
                hyp = r["output"]
                items = per_unit.get((cell.label, ui), [])
                bl = [(s, e, ds, de) for s, e, ds, de, _ in items]
                out = apply_blocks(src, hyp, bl, [k for *_, k in items])
                records.append(submission_record(ref, out))
                n_changed += out != src
                log_f.write(json.dumps({"run_id": run_id, "cell": cell.label, "document_id": did, "ocr": src, "output": out,
                                        "changed": out != src, "n_blocks": len(bl), "n_kept": sum(k for *_, k in items),
                                        "source_output": hyp}, ensure_ascii=False) + "\n")
            names[cell.stem] = submission_name(system, cell)
            write_jsonl(pred_dir / names[cell.stem], records)
            n_docs += len(refs)
    gen_wall = time.perf_counter() - t0
    scores_ = score_run(cells, pred_dir, names, out_dir / "scores", cfg, limit=limit, subset=subset)
    summary = {
        "run_id": run_id, "phase": 1, "system": system, "tag": tag or None, "split": split, "limit": limit,
        "subset": src_summary.get("subset"), "model": src_summary.get("model"), "thinking": src_summary.get("thinking"),
        "source_run": src_summary["run_id"], "gate_mode": args.mode, "gate": args.gate, "gate_type": gate_meta["type"] if gate_meta else None,
        "gate_encoder": gate_meta.get("encoder") if gate_meta else None, "rule": args.rule,
        "threshold": (diag["threshold"] if isinstance(threshold, np.ndarray) else (float(threshold) if np.isfinite(threshold) else "inf")),
        "block_diag": diag, "served_models": src_summary.get("served_models"), "base_url": None, **git_state(),
        "config_path": str(args.config), "seed": cfg["seed"], "generation": src_summary.get("generation"),
        "system_config": {"block_rule": "edits.edit_blocks", "context": None}, "started_utc": run_id.split("_", 1)[0],
        "n_units": n_docs, "generation_wall_s": round(gen_wall, 1), "wall_s": round(time.perf_counter() - t0, 1),
        "counters": {"n_blocks": len(blocks), "n_blocks_kept": int(keep.sum()), "n_changed_units": n_changed}, **scores_,
    }
    finalize_run(out_dir, summary)
    print(f"[{system}/{split}] weighted cMER={summary['cmer_micro_weighted']:.4f} pref={summary['pref_score_cmer_macro_weighted']:.4f} "
          f"AUC={diag.get('roc_auc')} kept={diag.get('kept_frac')} -> runs/{run_id}", flush=True)


if __name__ == "__main__":
    main()
