"""Phase 1: turn a full-unit rewrite run into labelled edit blocks (no GPU).

    python -m ocrgate.blocks --config configs/phase1.yaml --run runs/<run_id> [--workers 32]

For every unit of the run (`log.jsonl`: `ocr`, `output`, and the Phase 1 `tokens`/`logprobs` of the raw
completion) the rewrite is character-aligned to the OCR (Levenshtein `editops`) and cut into edit blocks
(`edits.edit_blocks`: maximal runs of adjacent character edits, the Phase 0b block rule). Each block is
labelled against the ground truth with the official scorer's own functions:

    gain = cMER(norm(gold), norm(ocr)) - cMER(norm(gold), norm(ocr with ONLY this block applied))

where cMER = (S+D+I)/(H+S+D+I) at character level (`hipe_ocrepair_scorer.ocrepair_eval.process_characters`
+ `mer_from_counts`) and `norm` is the scorer's normalisation (lower-case, punctuation -> space, ...).
label = "helpful" if gain > 0, "harmful" if gain < 0, "neutral" if gain == 0 (e.g. case/punctuation-only
blocks, which the scorer ignores). Blocks are labelled one at a time, holding all other blocks at the OCR
text; interactions between neighbouring blocks are ignored by the label (documented limitation).

Features written per block (runs/<run_id>/blocks.jsonl, git-ignored because it holds dataset text):
  - OCR context: `left`/`right` (up to --context chars of OCR on each side), `ocr_text`, `new_text`
  - rewriter logprobs over the characters of `new_text`: lp_mean, lp_min, lp_sum, lp_n (tokens covering the
    block), lp_first (token at the block start; for a pure deletion the token that follows the deleted text),
    lp_unit_mean (mean token logprob of the whole completion), lp_ok (False if the completion text could not
    be mapped to the tokens, then the lp_* fields are null)
  - shape: len_ocr, len_new, n_blocks_unit, len_unit, frac_changed_unit, is_insert, is_delete
  - oracle: in_oracle_span (block overlaps an oracle span of Phase 0/0b; diagnostic), gain, label
  - ids: cell, document_id, block_idx, lang, dataset, gt_sha1 (for dedupe / dev-overlap exclusion, no text)
"""

import argparse
import hashlib
import json
import time
from multiprocessing import Pool

from hipe_ocrepair_scorer.ocrepair_eval import mer_from_counts, norm, process_characters

from .data import ROOT, load_config, split_cells
from .edits import _touches, edit_blocks
from .llm import THINK_RE
from .score import select_refs
from .systems import gated_spans

_CFG = {}


def cmer(gold_norm: str, hyp: str) -> float:
    """Character MER exactly as the official scorer computes it per unit (on normalised text)."""
    o = process_characters(gold_norm, norm(hyp))
    return mer_from_counts(o.hits, o.substitutions, o.deletions, o.insertions)


def char_logprobs(out: str, tokens: list[str] | None, logprobs: list[float] | None):
    """Per-character logprob of `out` (the stripped, think-free output) from the raw completion tokens.
    Returns (lp_char, tok_id_char, unit_mean) or None if the output cannot be located in the tokens."""
    if not tokens:
        return None
    raw = "".join(tokens)  # = raw completion text + the end-of-turn token (e.g. <|im_end|>)
    text = THINK_RE.sub("", raw)
    # `out` is the think-free, stripped output; locate it inside the raw token text (search after any
    # removed <think> block first)
    start = raw.find(out, len(raw) - len(text)) if out else -1
    if start < 0:
        start = raw.find(out) if out else -1
    if start < 0 or raw[start:start + len(out)] != out:
        return None
    lp_char, tok_char, pos = [], [], 0
    for ti, (t, lp) in enumerate(zip(tokens, logprobs)):
        lp_char.extend([lp] * len(t))
        tok_char.extend([ti] * len(t))
        pos += len(t)
    unit_mean = sum(logprobs) / len(logprobs) if logprobs else None
    return lp_char[start:start + len(out)], tok_char[start:start + len(out)], unit_mean


def unit_blocks(args):
    cell_label, unit_idx, meta, src, tgt, hyp, tokens, logprobs = args
    cfg = _CFG
    W = int(cfg["context"])
    blocks = edit_blocks(src, hyp)
    if not blocks:
        return []
    gold_norm = norm(tgt)
    base = cmer(gold_norm, src)
    spans = gated_spans(src, tgt, cfg["gate_cfg"])
    lp = char_logprobs(hyp, tokens, logprobs)
    changed = sum(max(e - s, de - ds) for s, e, ds, de in blocks)
    rows = []
    for bi, (s, e, ds, de) in enumerate(blocks):
        cand = src[:s] + hyp[ds:de] + src[e:]
        gain = base - cmer(gold_norm, cand)
        row = {
            "cell": cell_label,
            "document_id": meta["document_id"],
            "unit_idx": unit_idx,  # position of the unit in its cell (document ids are not unique)
            "block_idx": bi,
            "lang": meta.get("language"),
            "dataset": meta.get("primary_dataset_name"),
            "gt_sha1": hashlib.sha1(tgt.encode("utf-8")).hexdigest(),
            "s": s, "e": e, "ds": ds, "de": de,
            "ocr_text": src[s:e],
            "new_text": hyp[ds:de],
            "left": src[max(0, s - W):s],
            "right": src[e:e + W],
            "len_ocr": e - s,
            "len_new": de - ds,
            "is_insert": e == s,
            "is_delete": de == ds,
            "n_blocks_unit": len(blocks),
            "len_unit": len(src),
            "frac_changed_unit": changed / max(1, len(src)),
            "cmer_ocr_unit": base,
            "in_oracle_span": _touches(spans, s, e),
            "gain": gain,
            "label": "helpful" if gain > 0 else ("harmful" if gain < 0 else "neutral"),
            "lp_ok": lp is not None,
            "lp_mean": None, "lp_min": None, "lp_sum": None, "lp_n": None, "lp_first": None, "lp_unit_mean": None,
        }
        if lp is not None:
            lp_char, tok_char, unit_mean = lp
            if de > ds:
                toks = sorted(set(tok_char[ds:de]))
                vals = [lp_char[j] for j in range(ds, de)]
                # mean over the distinct tokens covering the block (not over characters)
                tok_lp = {tok_char[j]: lp_char[j] for j in range(ds, de)}
                tvals = list(tok_lp.values())
                row.update(lp_mean=sum(tvals) / len(tvals), lp_min=min(tvals), lp_sum=sum(tvals), lp_n=len(toks),
                           lp_first=vals[0], lp_unit_mean=unit_mean)
            else:  # pure deletion: the token right after the deletion point carries the decision
                j = min(ds, len(lp_char) - 1)
                v = lp_char[j] if lp_char else None
                row.update(lp_mean=v, lp_min=v, lp_sum=v, lp_n=0, lp_first=v, lp_unit_mean=unit_mean)
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--run", required=True, help="runs/<run_id> of a zeroshot run (log.jsonl with tokens/logprobs)")
    ap.add_argument("--context", type=int, default=60, help="OCR context chars on each side of a block")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    run_dir = ROOT / args.run
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    split = summary["split"]
    cells = split_cells(cfg, split)
    # log lines are written in reference-file order per cell; match by position (document ids are not
    # unique: overproof train has one id three times)
    outputs: dict[str, list] = {}
    for line in (run_dir / "log.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        outputs.setdefault(r["cell"], []).append(r)
    global _CFG
    _CFG = {"context": args.context, "gate_cfg": cfg["systems"]["oracle_gated"]}
    jobs = []
    from .run import load_subset

    subset = load_subset(summary.get("subset"))
    for cell in cells:
        refs = select_refs(cell, int(summary.get("limit") or 0), subset)
        if len(refs) != len(outputs.get(cell.label, [])):
            raise SystemExit(f"{cell.label}: {len(refs)} references but {len(outputs.get(cell.label, []))} log lines")
        for ui, (ref, r) in enumerate(zip(refs, outputs[cell.label])):
            did = ref["document_metadata"]["document_id"]
            src, tgt = ref["ocr_hypothesis"]["transcription_unit"], ref["ground_truth"]["transcription_unit"]
            if r["document_id"] != did or r["ocr"] != src:
                raise SystemExit(f"OCR text mismatch for {did}")
            jobs.append((cell.label, ui, ref["document_metadata"], src, tgt, r["output"], r.get("tokens"), r.get("logprobs")))
    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        results = pool.map(unit_blocks, jobs, chunksize=4)
    rows = [row for rs in results for row in rs]
    out_path = run_dir / "blocks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n = len(rows)
    stats = {
        "run_id": summary["run_id"], "split": split, "n_units": len(jobs), "n_units_with_blocks": sum(1 for rs in results if rs),
        "n_blocks": n,
        "n_helpful": sum(r["label"] == "helpful" for r in rows),
        "n_harmful": sum(r["label"] == "harmful" for r in rows),
        "n_neutral": sum(r["label"] == "neutral" for r in rows),
        "n_lp_missing": sum(not r["lp_ok"] for r in rows),
        "n_in_oracle_span": sum(r["in_oracle_span"] for r in rows),
        "helpful_in_oracle_span": sum(r["label"] == "helpful" and r["in_oracle_span"] for r in rows),
        "harmful_in_oracle_span": sum(r["label"] == "harmful" and r["in_oracle_span"] for r in rows),
        "sum_gain_helpful": sum(r["gain"] for r in rows if r["gain"] > 0),
        "sum_gain_harmful": sum(r["gain"] for r in rows if r["gain"] < 0),
        "context": args.context, "wall_s": round(time.perf_counter() - t0, 1),
    }
    (run_dir / "blocks_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
