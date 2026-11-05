"""Phase 3: rescore an existing run (dev or held-out train) on a unit subset with the official scorer (no GPU, no model).

    python -m ocrgate.p3_rescore --run runs/<run_id> --subset configs/p3_gate_heldout_subset.json --tag TAG

Units are matched by (cell, document id, OCR text) between the subset's reference records and the run's log.jsonl; the run's
output is copied unchanged. Used for the P3-4 selection criterion: official weighted cMER / pref on the `gate_heldout` units only.
"""
import argparse
import json
import time

from .data import ROOT, load_config, split_cells, submission_name, submission_record, write_jsonl
from .run import finalize_run, git_state, load_subset, new_run_dir
from .score import score_run, select_refs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase2.yaml")
    ap.add_argument("--run", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    src = json.loads((ROOT / args.run / "summary.json").read_text(encoding="utf-8"))
    split = src["split"]
    if split == "test":
        raise SystemExit("the test split is locked")
    outputs = {}
    for line in (ROOT / args.run / "log.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        outputs.setdefault((r["cell"], r["document_id"], r["ocr"]), r["output"])
    subset = load_subset(args.subset)
    cells = split_cells(cfg, split)
    system = "rescore"
    run_id, out_dir = new_run_dir(system, split, args.tag)
    t0 = time.perf_counter(); names = {}; n = 0; miss = 0
    for cell in cells:
        refs = select_refs(cell, 0, subset)
        if not refs:
            continue
        recs = []
        for ref in refs:
            key = (cell.label, ref["document_metadata"]["document_id"], ref["ocr_hypothesis"]["transcription_unit"])
            out = outputs.get(key)
            if out is None:
                miss += 1; out = ref["ocr_hypothesis"]["transcription_unit"]
            recs.append(submission_record(ref, out))
        names[cell.stem] = submission_name(system, cell)
        write_jsonl(out_dir / "predictions" / names[cell.stem], recs); n += len(refs)
    cells = [c for c in cells if c.stem in names]
    scores = score_run(cells, out_dir / "predictions", names, out_dir / "scores", cfg, limit=0, subset=subset)
    summary = {"run_id": run_id, "phase": 3, "system": system, "tag": args.tag, "split": split, "limit": 0, "subset": args.subset,
               "source_run": src["run_id"], "source_system": src["system"], "source_tag": src.get("tag"), "model": src.get("model"),
               **git_state(), "config_path": args.config, "seed": cfg["seed"], "started_utc": run_id.split("_", 1)[0], "n_units": n,
               "n_missing_units": miss, "wall_s": round(time.perf_counter() - t0, 1), **scores}
    finalize_run(out_dir, summary)
    print(f"[rescore/{split}] {src['run_id']} on {args.subset}: weighted cMER={summary['cmer_micro_weighted']:.4f} pref={summary['pref_score_cmer_macro_weighted']:.4f} units={n} missing={miss} -> runs/{run_id}")


if __name__ == "__main__":
    main()
