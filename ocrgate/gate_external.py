"""Phase 1 freeze: apply a TEXT-ONLY gate to an external system's submission file (no logprobs, no labels).

    python -m ocrgate.gate_external --gate runs/gates/<name> --submission <team>_<ref-stem>_run<N>.jsonl
        --reference <reference jsonl> --out-dir <dir> [--team-suffix gated] [--threshold T] [--device cuda]

Reads the OCR input and the system output of every record of the submission (official format), cuts the
output into edit blocks against the OCR (`edits.edit_blocks`), builds the text-only block features (the gate
must have been trained with --no-lp; logprob features are zero), keeps the blocks whose gate score is >= the
gate's stored threshold, and writes a new submission file `<team>-<suffix>_<ref-stem>_run<N>.jsonl` with the
gated outputs in --out-dir. The reference file is used ONLY for the record order / OCR text check and for
`document_metadata`; ground truth is never read (the field is not accessed). This tool never scores anything:
scoring is a separate, explicit step with the official scorer. It refuses nothing by split: the caller is
responsible for the test lock (Phase 1 dry-runs use dev files only).
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .data import ROOT, read_jsonl, write_jsonl
from .edits import apply_blocks, edit_blocks
from .gate import load_gate, load_tokenizer, predict


def blocks_for_unit(cell_label: str, meta: dict, src: str, hyp: str, unit_idx: int, W: int = 60) -> list[dict]:
    bl = edit_blocks(src, hyp)
    changed = sum(max(e - s, de - ds) for s, e, ds, de in bl)
    rows = []
    for bi, (s, e, ds, de) in enumerate(bl):
        rows.append({
            "cell": cell_label, "document_id": meta.get("document_id"), "unit_idx": unit_idx, "block_idx": bi,
            "lang": meta.get("language"), "dataset": meta.get("primary_dataset_name"),
            "s": s, "e": e, "ds": ds, "de": de, "ocr_text": src[s:e], "new_text": hyp[ds:de],
            "left": src[max(0, s - W):s], "right": src[e:e + W], "len_ocr": e - s, "len_new": de - ds,
            "is_insert": e == s, "is_delete": de == ds, "n_blocks_unit": len(bl), "len_unit": len(src),
            "frac_changed_unit": changed / max(1, len(src)),
            "lp_ok": False, "lp_mean": None, "lp_min": None, "lp_sum": None, "lp_n": None, "lp_first": None, "lp_unit_mean": None,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--team-suffix", default="gated")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    t0 = time.perf_counter()
    gate_dir = Path(args.gate) if args.gate.startswith("/") else ROOT / args.gate
    model, std, meta = load_gate(gate_dir, args.device)
    if meta.get("use_lp", True):
        raise SystemExit("gate_external needs a text-only gate (trained with --no-lp); this gate uses logprob features")
    threshold = args.threshold if args.threshold is not None else meta["threshold"]
    tok = load_tokenizer(meta["encoder"])

    sub_path = Path(args.submission)
    refs = read_jsonl(Path(args.reference))
    subs = read_jsonl(sub_path)
    by_id: dict[str, list] = {}
    for r in subs:
        by_id.setdefault(r["document_metadata"]["document_id"], []).append(r)
    cell_label = sub_path.stem.split("hipe-ocrepair-bench_v0.9_", 1)[-1].rsplit("_run", 1)[0]
    units, all_blocks = [], []
    n_missing = n_ocr_mismatch = 0
    for ui, ref in enumerate(refs):
        did = ref["document_metadata"]["document_id"]
        src = ref["ocr_hypothesis"]["transcription_unit"]
        cands = by_id.get(did, [])
        rec = cands.pop(0) if cands else None
        hyp = None
        if rec is not None:
            hyp = rec.get("ocr_postcorrection_output", {}).get("transcription_unit")
            if rec["ocr_hypothesis"]["transcription_unit"] != src:
                # The masked-test file given to a team can differ from the reference OCR for a few documents (in the
                # 2026 test data: '-' vs '—' etc., all in units the scorer excludes). The system's edits are relative
                # to the OCR it saw, so cut the blocks against the submission's own OCR text instead of aborting.
                n_ocr_mismatch += 1
                print(f"[gate_external] OCR text differs from the reference for {did}: using the submission's OCR as source", flush=True)
                src = rec["ocr_hypothesis"]["transcription_unit"]
        if hyp is None or hyp == "None" or not isinstance(hyp, str):
            n_missing += 1
            hyp = src  # the scorer would substitute the OCR input; do the same
        bl = blocks_for_unit(cell_label, ref["document_metadata"], src, hyp, ui)
        units.append((ref, src, hyp, bl))
        all_blocks.extend(bl)
    scores = np.array(predict(model, tok, std, all_blocks, meta["max_len"], False, device=args.device)) if all_blocks else np.array([])
    keep = scores >= threshold
    pos, records, n_changed = 0, [], 0
    for ref, src, hyp, bl in units:
        k = keep[pos:pos + len(bl)]; pos += len(bl)
        out = apply_blocks(src, hyp, [(b["s"], b["e"], b["ds"], b["de"]) for b in bl], [bool(x) for x in k])
        n_changed += out != src
        records.append({
            "document_metadata": {k_: ref["document_metadata"][k_] for k_ in ("document_id", "primary_dataset_name", "language")},
            "ocr_hypothesis": {"transcription_unit": src},
            "ocr_postcorrection_output": {"transcription_unit": out},
        })
    team, rest = sub_path.name.split("_", 1)
    out_path = Path(args.out_dir) / f"{team}-{args.team_suffix}_{rest}"
    write_jsonl(out_path, records)
    sha = hashlib.sha256((gate_dir / "model.pt").read_bytes()).hexdigest()
    info = {"gate": str(gate_dir), "gate_model_sha256": sha, "threshold": threshold, "submission": str(sub_path),
            "out": str(out_path), "n_units": len(units), "n_missing_outputs": n_missing, "n_ocr_mismatch": n_ocr_mismatch, "n_blocks": len(all_blocks),
            "n_blocks_kept": int(keep.sum()), "n_changed_units": n_changed, "wall_s": round(time.perf_counter() - t0, 1)}
    (out_path.with_suffix(".gate.json")).write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
