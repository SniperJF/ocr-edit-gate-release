"""Phase 1: train (or calibrate) an edit-veto gate on TRAIN-split blocks; select on a held-out part of TRAIN.

    python -m ocrgate.gate_train --blocks runs/<train_run>/blocks.jsonl [runs/<other_train_run>/blocks.jsonl]
        --encoder Qwen/Qwen3-0.6B|Qwen/Qwen3-1.7B|google/byt5-base|lp --name NAME [--no-lp] [--epochs 3] ...

Data: blocks labelled helpful (1) / harmful (0); neutral blocks (gain == 0, invisible to the scorer) are
excluded from fitting and from AUCs. Units whose ground truth also occurs in the dev split are dropped,
and units with identical ground truth inside train (DTA matched/unmatched overlap) are kept once.
Units are grouped by ground truth (sha1 of the gold unit; DTA matched pages share one gold text across the
3 noise-level cells and 17 pages also occur in the unmatched files) and, per cell, 15% of the groups present in
that cell are drawn as held-out with seed 1234 (`split_units`); every unit of a held-out group, in every cell
and every rewriter run, is held out. Because shared groups are drawn independently in each cell, the held-out
share of UNITS is larger than 15% (about 20%). Hyper-parameter selection (epoch, decision threshold) uses the
held-out part only; dev is never read here. The decision threshold is ONE global value for all cells and languages.

Decision threshold: the single global value that maximises the total labelled gain of the kept blocks on
the held-out part (sum of `gain` over blocks with score >= t), i.e. the per-unit cMER reduction the scorer
would see if blocks were independent. No per-language or per-cell thresholds are used.

encoder = lp: training-free baseline, score = `--lp-stat` (lp_mean or lp_min); only the threshold is chosen
(on the fit part; the held-out part reports AUC).

Outputs: runs/gates/<name>/ (model.pt, gate.json; git-ignored) and one line in results/gates_p1.jsonl
(metrics only, no text).

Phase 2 (`--split configs/p2_split.json`): the held-out part is the fixed `gate_heldout` role of the Phase 2 train split
(units are looked up by (cell, document_id); LoRA-rewriter blocks exist only for held-out-from-LoRA units, so no block
of a LoRA rewriter comes from a unit that rewriter was trained on). Every other unit is fit data; `dev_overlap` units
are dropped. The held-out scores of the best epoch are written to runs/gates/<name>/heldout_scores.jsonl (no text:
run, cell, document id, block index, language, dataset, score, gain, label, unit edit-rate features) so that decision
rules (`ocrgate.rules`) can be chosen on held-out TRAIN afterwards without re-running the encoder. `--results` selects
the metrics file (default results/gates_p1.jsonl; Phase 2 uses results/gates_p2.jsonl).
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np

from .data import ROOT, load_config, split_cells
from .score import select_refs


def load_blocks(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        src = Path(p).parent.name
        for line in open(ROOT / p, encoding="utf-8"):
            r = json.loads(line)
            r["src_run"] = src
            rows.append(r)
    return rows


def dev_gt_hashes(cfg: dict) -> set[str]:
    hs = set()
    for cell in split_cells(cfg, "dev"):
        for ref in select_refs(cell):
            hs.add(hashlib.sha1(ref["ground_truth"]["transcription_unit"].encode("utf-8")).hexdigest())
    return hs


def unit_key(b: dict) -> tuple:
    return (b["src_run"], b["cell"], b["document_id"])


def filter_units(rows: list[dict], dev_hashes: set[str]) -> tuple[list[dict], dict]:
    """Drop units whose ground truth also occurs in dev. Units that share a ground truth inside train (the
    DTA matched files hold the same page at 3 noise levels; 17 pages also occur in the unmatched files) are
    all kept, since their OCR inputs differ, but `split_units` puts them on the same side of the split."""
    keep_units, n_dev = set(), set()
    for b in rows:
        u = unit_key(b)
        if b["gt_sha1"] in dev_hashes:
            n_dev.add(u)
        else:
            keep_units.add(u)
    out = [b for b in rows if unit_key(b) in keep_units]
    groups = {b["gt_sha1"] for b in out}
    return out, {"units_dropped_gt_in_dev": len(n_dev), "units_kept": len(keep_units), "gt_groups": len(groups)}


def split_units(rows: list[dict], seed: int, heldout_frac: float) -> set[tuple]:
    """Held-out unit keys: ~heldout_frac of the GT groups of every cell (a GT group = all units sharing a
    ground truth, across runs and cells); grouping keeps shared pages on one side of the split."""
    group_of = {unit_key(b): b["gt_sha1"] for b in rows}
    cell_groups: dict[str, set] = {}
    for u, g in group_of.items():
        cell_groups.setdefault(u[1], set()).add(g)
    rng = random.Random(seed)
    held_groups: set[str] = set()
    for cell, groups in sorted(cell_groups.items()):
        gs = sorted(groups)
        rng.shuffle(gs)
        n = max(1, round(len(gs) * heldout_frac))
        held_groups.update(gs[:n])
    return {u for u, g in group_of.items() if g in held_groups}


def best_threshold(scores: np.ndarray, gains: np.ndarray) -> tuple[float, float]:
    """Threshold t maximising sum(gain[score >= t]); returns (t, total_gain). Also considers 'keep all'."""
    order = np.argsort(-scores)
    cum = np.cumsum(gains[order])
    i = int(np.argmax(cum))
    if cum[i] <= 0:
        return float("inf"), 0.0  # keep nothing
    s_sorted = scores[order]
    # threshold halfway between the last kept score and the next one (or the kept score itself)
    t = float(s_sorted[i]) if i + 1 >= len(s_sorted) else float((s_sorted[i] + s_sorted[i + 1]) / 2)
    return t, float(cum[i])


def metrics(y: np.ndarray, s: np.ndarray, gains: np.ndarray, t: float) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    keep = s >= t
    return {
        "n": int(len(y)), "n_helpful": int(y.sum()), "n_harmful": int((1 - y).sum()),
        "roc_auc": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None,
        "pr_auc_helpful": float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else None,
        "pr_auc_harmful": float(average_precision_score(1 - y, -s)) if 0 < y.sum() < len(y) else None,
        "threshold": t,
        "kept_frac": float(keep.mean()),
        "helpful_kept_frac": float(keep[y == 1].mean()) if y.sum() else None,
        "harmful_rejected_frac": float((~keep)[y == 0].mean()) if (1 - y).sum() else None,
        "gain_kept": float(gains[keep].sum()), "gain_all": float(gains.sum()), "gain_oracle": float(gains[gains > 0].sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--blocks", nargs="+", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--no-lp", action="store_true", help="feature-reduced gate: zero out all logprob features")
    ap.add_argument("--lp-stat", default="lp_mean", choices=["lp_mean", "lp_min"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--heldout-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-fit", type=int, default=0, help="debug: subsample the fit blocks (smoke tests only)")
    ap.add_argument("--max-heldout", type=int, default=0, help="debug: subsample the held-out blocks (smoke tests only)")
    ap.add_argument("--split", default=None, help="Phase 2: configs/p2_split.json; held-out = role gate_heldout")
    ap.add_argument("--results", default="results/gates_p1.jsonl", help="metrics jsonl to append to")
    args = ap.parse_args()

    import torch

    from .gate import GateModel, Standardizer, block_text, load_tokenizer, raw_features, save_gate

    cfg = load_config(ROOT / args.config)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    t0 = time.perf_counter()
    rows = load_blocks(args.blocks)
    n_raw = len(rows)
    rows, filt = filter_units(rows, dev_gt_hashes(cfg))
    split_info = {}
    if args.split:
        sp = json.loads((ROOT / args.split).read_text(encoding="utf-8"))
        role_of = {(u["cell"], u["document_id"]): u["role"] for u in sp["units"]}
        n_before = len(rows)
        rows = [b for b in rows if role_of[(b["cell"], b["document_id"])] != "dev_overlap"]
        held = {unit_key(b) for b in rows if role_of[(b["cell"], b["document_id"])] == "gate_heldout"}
        split_info = {"split": args.split, "n_blocks_dropped_dev_overlap_role": n_before - len(rows),
                      "n_blocks_per_run": {r: sum(1 for b in rows if b["src_run"] == r) for r in sorted({b["src_run"] for b in rows})},
                      "n_heldout_blocks_per_run": {r: sum(1 for b in rows if b["src_run"] == r and unit_key(b) in held) for r in sorted({b["src_run"] for b in rows})}}
    labelled = [b for b in rows if b["label"] != "neutral"]
    if not args.split:
        held = split_units(rows, args.seed, args.heldout_frac)
    fit = [b for b in labelled if unit_key(b) not in held]
    ho = [b for b in labelled if unit_key(b) in held]
    if args.max_fit:
        fit = random.Random(args.seed).sample(fit, min(args.max_fit, len(fit)))
    if args.max_heldout:
        ho = random.Random(args.seed).sample(ho, min(args.max_heldout, len(ho)))
    info = {
        "name": args.name, "encoder": args.encoder, "use_lp": not args.no_lp, "blocks": args.blocks,
        "n_blocks_raw": n_raw, **filt, "n_labelled": len(labelled), "n_fit": len(fit), "n_heldout": len(ho),
        "n_units_heldout": len(held), "seed": args.seed, "heldout_frac": args.heldout_frac, **split_info,
    }
    print(json.dumps(info), flush=True)
    y_fit = np.array([1.0 if b["label"] == "helpful" else 0.0 for b in fit]); g_fit = np.array([b["gain"] for b in fit])
    y_ho = np.array([1.0 if b["label"] == "helpful" else 0.0 for b in ho]); g_ho = np.array([b["gain"] for b in ho])
    gate_dir = ROOT / "runs" / "gates" / args.name
    gate_dir.mkdir(parents=True, exist_ok=True)

    if args.encoder == "lp":
        def score(bs):
            return np.array([b[args.lp_stat] if b.get("lp_ok") and b.get(args.lp_stat) is not None else -99.0 for b in bs])
        s_fit, s_ho = score(fit), score(ho)
        t, gain_fit = best_threshold(s_fit, g_fit)
        meta = {"type": "lp_threshold", "lp_stat": args.lp_stat, "threshold": t, **info}
        (gate_dir / "gate.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        result = {**meta, "fit": metrics(y_fit, s_fit, g_fit, t), "heldout": metrics(y_ho, s_ho, g_ho, t),
                  "wall_s": round(time.perf_counter() - t0, 1)}
    else:
        tok = load_tokenizer(args.encoder)
        model = GateModel(args.encoder, len(raw_features(fit[0]))).to(args.device)
        X_fit = torch.tensor([raw_features(b, not args.no_lp) for b in fit])
        std = Standardizer().fit(X_fit)
        X_fit = std(X_fit)
        X_ho = std(torch.tensor([raw_features(b, not args.no_lp) for b in ho]))
        enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
        head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
        opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr}, {"params": head_params, "lr": args.head_lr}], weight_decay=0.01)
        steps_per_epoch = (len(fit) + args.batch_size - 1) // args.batch_size
        total = steps_per_epoch * args.epochs
        warm = max(1, int(0.05 * total))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)) if s >= warm else (s + 1) / warm)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        texts_fit = [block_text(b) for b in fit]
        texts_ho = [block_text(b) for b in ho]

        def predict_texts(texts, X):
            model.eval()
            out = []
            with torch.no_grad():
                for i in range(0, len(texts), 128):
                    enc = tok(texts[i:i + 128], padding=True, truncation=True, max_length=args.max_len, return_tensors="pt")
                    with torch.autocast(args.device if args.device != "cpu" else "cpu", dtype=torch.bfloat16):
                        lg = model(enc["input_ids"].to(args.device), enc["attention_mask"].to(args.device), X[i:i + 128].to(args.device))
                    out.extend(torch.sigmoid(lg.float()).cpu().tolist())
            return np.array(out)

        best, history, step = None, [], 0
        idx = np.arange(len(fit))
        for epoch in range(1, args.epochs + 1):
            model.train()
            np.random.shuffle(idx)
            run_loss, n_seen = 0.0, 0
            for i in range(0, len(idx), args.batch_size):
                bi = idx[i:i + args.batch_size]
                enc = tok([texts_fit[j] for j in bi], padding=True, truncation=True, max_length=args.max_len, return_tensors="pt")
                yb = torch.tensor(y_fit[bi], dtype=torch.float32, device=args.device)
                with torch.autocast(args.device if args.device != "cpu" else "cpu", dtype=torch.bfloat16):
                    lg = model(enc["input_ids"].to(args.device), enc["attention_mask"].to(args.device), X_fit[bi].to(args.device))
                loss = loss_fn(lg.float(), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); step += 1
                run_loss += loss.item() * len(bi); n_seen += len(bi)
                if step % 200 == 0:
                    print(f"epoch {epoch} step {step}/{total} loss {run_loss / n_seen:.4f}", flush=True)
            s_ho = predict_texts(texts_ho, X_ho)
            t, _ = best_threshold(s_ho, g_ho)
            m = metrics(y_ho, s_ho, g_ho, t)
            m.update(epoch=epoch, train_loss=run_loss / max(1, n_seen))
            history.append(m)
            print(json.dumps(m), flush=True)
            if best is None or (m["roc_auc"] or 0) > (best["roc_auc"] or 0):
                best = m
                save_gate(gate_dir, model, std, {
                    "type": "learned", "encoder": args.encoder, "use_lp": not args.no_lp, "max_len": args.max_len,
                    "threshold": t, "epoch": epoch, "hidden": 256, "dropout": 0.1, **info,
                })
                with open(gate_dir / "heldout_scores.jsonl", "w", encoding="utf-8") as f:
                    for b, sc in zip(ho, s_ho):
                        f.write(json.dumps({"src_run": b["src_run"], "cell": b["cell"], "document_id": b["document_id"],
                                            "block_idx": b["block_idx"], "lang": b.get("lang"), "dataset": b.get("dataset"),
                                            "score": float(sc), "gain": b["gain"], "label": b["label"],
                                            "frac_changed_unit": b["frac_changed_unit"], "n_blocks_unit": b["n_blocks_unit"],
                                            "len_unit": b["len_unit"], "len_ocr": b["len_ocr"], "len_new": b["len_new"]}) + "\n")
        s_fit = predict_texts(texts_fit, X_fit) if len(fit) <= 60000 else None
        result = {**info, "type": "learned", "max_len": args.max_len, "epochs": args.epochs, "lr": args.lr, "head_lr": args.head_lr,
                  "batch_size": args.batch_size, "best_epoch": best["epoch"], "threshold": best["threshold"], "heldout": best,
                  "history": history, "wall_s": round(time.perf_counter() - t0, 1)}
        if s_fit is not None:
            result["fit"] = metrics(y_fit, s_fit, g_fit, best["threshold"])
    result["debug"] = bool(args.max_fit or args.max_heldout)
    (gate_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["debug"]:
        with open(ROOT / args.results, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k in ("name", "encoder", "threshold", "heldout", "wall_s")}, indent=1))


if __name__ == "__main__":
    main()
