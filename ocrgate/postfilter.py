"""Phase 0b-1: offline oracle filtering of an existing run's outputs (no GPU).

    python -m ocrgate.postfilter --source-run runs/<run_id> --mode oracle --granularity block
    python -m ocrgate.postfilter --source-run runs/<run_id> --mode reverse --granularity block
    python -m ocrgate.postfilter --source-run runs/<run_id> --mode none --subset configs/subset.json

For every unit of the source run (its `log.jsonl` has `ocr` and `output`), the output is character-
aligned to the OCR input (Levenshtein). `--mode oracle` keeps only the edits that touch an ORACLE
span (alignment OCR vs ground truth, same gate as `oracle_gated`) and reverts the rest to the OCR;
`--mode reverse` reverts the edits touching an oracle span and keeps the rest (sanity check);
`--mode none` keeps the output unchanged (used to rescore a run on a probe subset). The result is
scored with the official scorer and recorded like any other run. ORACLE modes read the ground truth
and are upper-bound diagnostics, not valid systems.
"""

import argparse
import json
import time

from .data import ROOT, load_config, split_cells, submission_name, submission_record, write_jsonl
from .edits import filter_edits
from .run import COUNTER_KEYS, finalize_run, git_state, load_subset, new_run_dir
from .score import score_run, select_refs
from .systems import gated_spans

SYSTEM_NAMES = {"oracle": "orfilt", "reverse": "revfilt", "none": "rescore"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase0.yaml")
    ap.add_argument("--source-run", required=True, help="runs/<run_id> directory of the run to filter")
    ap.add_argument("--mode", choices=sorted(SYSTEM_NAMES), required=True)
    ap.add_argument("--granularity", choices=["block", "char"], default="block")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--subset", default=None, help="JSON {cell label: [document ids]}: score only these units")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--system-name", default=None, help="override the derived system name")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    src_dir = ROOT / args.source_run
    src_summary = json.loads((src_dir / "summary.json").read_text(encoding="utf-8"))
    if src_summary["split"] != args.split:
        raise SystemExit(f"source run is on split {src_summary['split']!r}, not {args.split!r}")
    if src_summary.get("limit"):
        raise SystemExit("source run is a --limit debug run")
    outputs = {}
    for line in (src_dir / "log.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        outputs[(r["cell"], r["document_id"])] = (r["ocr"], r["output"])

    base = src_summary["system"]
    system = args.system_name or (base if args.mode == "none" else f"{base}_{SYSTEM_NAMES[args.mode]}_{args.granularity}")
    subset = load_subset(args.subset)
    gate_cfg = cfg["systems"]["oracle_gated"]
    cells = split_cells(cfg, args.split)
    run_id, run_dir = new_run_dir(system, args.split, args.tag)
    pred_dir = run_dir / "predictions"

    t0 = time.perf_counter()
    names, n_docs, counters = {}, 0, {}
    with open(run_dir / "log.jsonl", "w", encoding="utf-8") as log_f:
        for cell in cells:
            refs = select_refs(cell, 0, subset)
            records = []
            for ref in refs:
                did = ref["document_metadata"]["document_id"]
                src, tgt = ref["ocr_hypothesis"]["transcription_unit"], ref["ground_truth"]["transcription_unit"]
                ocr_logged, hyp = outputs[(cell.label, did)]
                if ocr_logged != src:
                    raise SystemExit(f"OCR text mismatch for {did}")
                if args.mode == "none":
                    out, doc_log = hyp, {}
                else:
                    spans = gated_spans(src, tgt, gate_cfg)
                    out, st = filter_edits(src, hyp, spans, keep_inside=(args.mode == "oracle"), granularity=args.granularity)
                    doc_log = {
                        "n_spans": len(spans),
                        "n_blocks": st.n_blocks,
                        "n_blocks_kept": st.n_blocks_kept,
                        "n_ops": st.n_ops,
                        "n_ops_kept": st.n_ops_kept,
                        "source_output": hyp,
                    }
                records.append(submission_record(ref, out))
                for k in COUNTER_KEYS:
                    if k in doc_log:
                        counters[k] = counters.get(k, 0) + int(doc_log[k])
                counters["n_changed_units"] = counters.get("n_changed_units", 0) + int(out != src)
                log_f.write(json.dumps({
                    "run_id": run_id, "cell": cell.label, "document_id": did, "ocr": src, "output": out,
                    "changed": out != src, **doc_log,
                }, ensure_ascii=False) + "\n")
            names[cell.stem] = submission_name(system, cell)
            write_jsonl(pred_dir / names[cell.stem], records)
            n_docs += len(refs)
            print(f"[{system}/{args.split}] {cell.label}: {len(refs)} units", flush=True)
    gen_wall = time.perf_counter() - t0
    scores = score_run(cells, pred_dir, names, run_dir / "scores", cfg, subset=subset)
    wall = time.perf_counter() - t0
    summary = {
        "run_id": run_id,
        "system": system,
        "tag": args.tag,
        "split": args.split,
        "limit": 0,
        "subset": args.subset,
        "model": src_summary.get("model"),
        "thinking": src_summary.get("thinking"),
        "source_run": src_summary["run_id"],
        "filter": {"mode": args.mode, "granularity": args.granularity} if args.mode != "none" else None,
        "served_models": src_summary.get("served_models"),
        "base_url": None,
        **git_state(),
        "config_path": str(args.config),
        "seed": cfg["seed"],
        "generation": src_summary.get("generation"),
        "system_config": gate_cfg if args.mode != "none" else src_summary.get("system_config"),
        "started_utc": run_id.split("_", 1)[0],
        "n_units": n_docs,
        "generation_wall_s": round(gen_wall, 1),
        "wall_s": round(wall, 1),
        "counters": counters,
        **scores,
    }
    finalize_run(run_dir, summary)
    print(
        f"[{system}/{args.split}] weighted cMER={summary['cmer_micro_weighted']:.4f} "
        f"pref={summary['pref_score_cmer_macro_weighted']:.4f} wall={summary['wall_s']}s -> runs/{run_id}",
        flush=True,
    )


if __name__ == "__main__":
    main()
