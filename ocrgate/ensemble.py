"""Phase 2 (P2-3): edit-pool ensemble of several rewriters under one gate.

    python -m ocrgate.ensemble --runs runs/<A> runs/<B> [runs/<C>] --gate runs/gates/NAME [--rule NAME | --threshold T]
        [--agree-thresholds T1 T2] [--dump-union PATH] --tag TAG [--device cuda]

All runs must be on the same split; the units are those of the FIRST run (its subset, if any) and are matched in the
other runs by (cell, document id, OCR text). For every unit the edit blocks of all runs are pooled: identical proposals
(same OCR span and replacement) are merged (n_agree = number of rewriters proposing it, score = max of their gate scores)
and every block is a candidate iff its gate score >= its threshold (the rule of `ocrgate.rules`, the gate's stored
threshold, or --threshold; with --agree-thresholds T1 T2 a candidate needs score >= T1 if n_agree == 1 and >= T2 if
n_agree >= 2 instead). Candidates that overlap on the OCR (strict interval overlap; an insertion inside a block; two
insertions at the same point) are resolved in favour of the higher score (then higher agreement); the accepted blocks
are applied to the OCR text. Gate scores come from the per-run cache runs/<run>/gate_scores_<gate>.json written by
gate_apply (computed here if missing). Output = a run directory scored with the official scorer (like gate_apply) plus
block-level diagnostics on the pooled blocks (labels = the source run's labelled gain, identical for identical blocks).
--dump-union writes one line per pooled candidate (score, n_agree, gain, label, lang; no text) so that T1/T2 can be
chosen on held-out TRAIN with `--select` of this module (see `choose_agree_thresholds`).
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .data import ROOT, load_config, split_cells, submission_name, submission_record, write_jsonl
from .gate_train import best_threshold, metrics
from .run import finalize_run, git_state, load_subset, new_run_dir
from .score import score_run, select_refs


def conflicts(a, b) -> bool:
    sa, ea, sb, eb = a[0], a[1], b[0], b[1]
    if ea > sa and eb > sb:
        return sa < eb and sb < ea
    if ea == sa and eb > sb:
        return sb < sa < eb
    if eb == sb and ea > sa:
        return sa < sb < ea
    return sa == sb


def load_run(run: str):
    run_dir = ROOT / run
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    blocks = [json.loads(l) for l in open(run_dir / "blocks.jsonl", encoding="utf-8")]
    logs = {}
    for line in (run_dir / "log.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        logs[(r["cell"], r["document_id"], r["ocr"])] = r
    return run_dir, summary, blocks, logs


def gate_scores(run_dir: Path, gate: str, blocks: list[dict], device: str) -> np.ndarray:
    cache = run_dir / f"gate_scores_{Path(gate).name}.json"
    if cache.is_file():
        s = np.array(json.loads(cache.read_text(encoding="utf-8")))
        assert len(s) == len(blocks)
        return s
    from .gate import load_gate, load_tokenizer, predict  # torch / transformers: GPU environment only

    model, std, meta = load_gate(ROOT / gate, device)
    tok = load_tokenizer(meta["encoder"])
    s = np.array(predict(model, tok, std, blocks, meta["max_len"], meta["use_lp"], device=device))
    cache.write_text(json.dumps([float(x) for x in s]), encoding="utf-8")
    return s


def level_of(n_agree: int, levels: int) -> int:
    """Agreement level index of a block proposed by n_agree members: 0 for 1 member, ..., levels-1 for >= levels members."""
    return min(int(n_agree), levels) - 1


def choose_agree_thresholds(rows: list[dict], levels: int = 2) -> dict:
    """On dumped held-out union candidates: one threshold per agreement level (n_agree == 1, == 2, ..., >= levels)
    maximising total gain kept, chosen independently per level (the levels are disjoint, so the optimum separates).
    levels = 2 is the Phase 2 rule (t_single, t_agree)."""
    s = np.array([r["score"] for r in rows]); g = np.array([r["gain"] for r in rows]); n = np.array([r["n_agree"] for r in rows])
    lv = np.array([level_of(x, levels) for x in n])
    ts, gains, counts, helpful = [], [], [], []
    for k in range(levels):
        m = lv == k
        t, gk = best_threshold(s[m], g[m]) if m.any() else (float("inf"), 0.0)
        ts.append(t); gains.append(gk); counts.append(int(m.sum())); helpful.append(float((g[m] > 0).mean()) if m.any() else None)
    tg, gg = best_threshold(s, g)
    return {"t_single": ts[0], "t_agree": ts[1] if levels > 1 else None, "thresholds": ts, "levels": levels, "gain_kept_per_level": gains,
            "gain_kept_agree_rule": float(sum(gains)), "t_global": tg, "gain_kept_global": gg,
            "n_single": counts[0], "n_agree": int((n >= 2).sum()), "n_per_level": counts, "gain_oracle": float(g[g > 0].sum()),
            "helpful_frac_single": helpful[0], "helpful_frac_agree": float((g[n >= 2] > 0).mean()) if (n >= 2).any() else None,
            "helpful_frac_per_level": helpful}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase2.yaml")
    ap.add_argument("--runs", nargs="+")
    ap.add_argument("--gate")
    ap.add_argument("--rule", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--agree-thresholds", type=float, nargs="+", default=None, metavar="T",
                    help="T1 T2 [T3 ...]: threshold for blocks proposed by 1, 2, ..., >= len members (Phase 2: two values)")
    ap.add_argument("--families", nargs="+", default=None,
                    help="Phase 3: one family name per run; n_agree counts distinct families (sampled candidates of one rewriter count once)")
    ap.add_argument("--levels", type=int, default=2, help="with --select: number of agreement levels (Phase 2: 2)")
    ap.add_argument("--dump-union", default=None, help="write pooled candidates (no text) to this jsonl (selection on held-out)")
    ap.add_argument("--select", default=None, help="choose T1/T2 from a dumped held-out union file and print them")
    ap.add_argument("--select-split", default=None, help="with --select: configs/p2_split.json, keep only --select-roles units")
    ap.add_argument("--select-roles", default="gate_heldout")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args()
    if args.select:
        rows = [json.loads(l) for l in open(args.select, encoding="utf-8")]
        if args.select_split:
            sp = json.loads((ROOT / args.select_split).read_text(encoding="utf-8"))
            roles = set(args.select_roles.split(","))
            ok = {(u["cell"], u["document_id"]) for u in sp["units"] if u["role"] in roles}
            rows = [r for r in rows if (r["cell"], r["document_id"]) in ok]
        res = choose_agree_thresholds(rows, args.levels); res["n_rows"] = len(rows); res["roles"] = args.select_roles if args.select_split else "all"
        print(json.dumps(res, indent=1)); return

    cfg = load_config(ROOT / args.config)
    t0 = time.perf_counter()
    runs = [load_run(r) for r in args.runs]
    families = args.families or [str(i) for i in range(len(runs))]
    if len(families) != len(runs):
        raise SystemExit("--families needs one name per run")
    split = runs[0][1]["split"]
    if any(r[1]["split"] != split for r in runs):
        raise SystemExit("all runs must be on the same split")
    if split == "test" and not (args.allow_test and os.environ.get("CONFIRM_TEST_RUN") == "1"):
        raise SystemExit("the test split is locked")
    subset = load_subset(runs[0][1].get("subset"))
    limit = int(runs[0][1].get("limit") or 0)
    rules = json.loads((ROOT / args.gate / "rules.json").read_text(encoding="utf-8")) if args.rule else None
    gate_meta = json.loads((ROOT / args.gate / "gate.json").read_text(encoding="utf-8"))
    from .rules import thresholds_for

    # per run: scores, thresholds; pooled candidates per unit key
    cand: dict[tuple, list] = {}
    for ri, (run_dir, summary, blocks, logs) in enumerate(runs):
        sc = gate_scores(run_dir, args.gate, blocks, args.device)
        if args.agree_thresholds:
            thr = np.full(len(blocks), args.agree_thresholds[0])
        elif args.rule:
            thr = thresholds_for(rules["rules"][args.rule], blocks, sc)
        else:
            thr = np.full(len(blocks), args.threshold if args.threshold is not None else gate_meta["threshold"])
        # unit lookup: blocks carry (cell, unit_idx) relative to the run's own subset -> map to (cell, document_id, ocr)
        by_pos = {}
        for key, r in logs.items():
            by_pos.setdefault(key[0], []).append(key)
        # log lines are in reference order per cell; blocks' unit_idx indexes that order
        order = {}
        for line in (run_dir / "log.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            order.setdefault(r["cell"], []).append((r["cell"], r["document_id"], r["ocr"]))
        for b, s_, t_ in zip(blocks, sc, thr):
            key = order[b["cell"]][b["unit_idx"]]
            hyp = logs[key]["output"]
            cand.setdefault(key, []).append({"s": b["s"], "e": b["e"], "new": hyp[b["ds"]:b["de"]], "score": float(s_), "thr": float(t_),
                                             "gain": b["gain"], "label": b["label"], "lang": b.get("lang"), "run": ri, "family": families[ri],
                                             "frac_changed_unit": b["frac_changed_unit"]})
    # merge identical proposals, decide, resolve conflicts
    cells = split_cells(cfg, split)
    system = "ensemble"
    run_id, out_dir = new_run_dir(system, split, args.tag)
    pred_dir = out_dir / "predictions"
    names, n_docs, n_changed, n_conf = {}, 0, 0, 0
    pooled, kept_flags = [], []
    dump_f = open(args.dump_union, "w", encoding="utf-8") if args.dump_union else None
    with open(out_dir / "log.jsonl", "w", encoding="utf-8") as log_f:
        for cell in cells:
            refs = select_refs(cell, limit, subset)
            records = []
            for ref in refs:
                did = ref["document_metadata"]["document_id"]; src = ref["ocr_hypothesis"]["transcription_unit"]
                key = (cell.label, did, src)
                for ri, (_, _, _, logs) in enumerate(runs):
                    if key not in logs:
                        raise SystemExit(f"unit {did} of {cell.label} missing in run {args.runs[ri]}")
                merged: dict[tuple, dict] = {}
                for c in cand.get(key, []):
                    k = (c["s"], c["e"], c["new"])
                    m = merged.get(k)
                    if m is None:
                        merged[k] = {**c, "n_votes": 1, "runs": [c["run"]], "fams": {c["family"]}, "cand": c["score"] >= c["thr"]}
                    else:
                        m["n_votes"] += 1; m["runs"].append(c["run"]); m["fams"].add(c["family"])
                        m["score"] = max(m["score"], c["score"]); m["thr"] = min(m["thr"], c["thr"])
                        m["cand"] = m["cand"] or c["score"] >= c["thr"]
                items = list(merged.values())
                for m in items:
                    m["n_agree"] = len(m.pop("fams"))  # distinct families (= runs unless --families)
                if args.agree_thresholds:
                    for m in items:
                        m["cand"] = m["score"] >= args.agree_thresholds[level_of(m["n_agree"], len(args.agree_thresholds))]
                accepted = []
                for m in sorted(items, key=lambda m: (-m["score"], -m["n_agree"], m["s"])):
                    m["keep"] = False
                    if not m["cand"]:
                        continue
                    if any(conflicts((m["s"], m["e"]), (a["s"], a["e"])) for a in accepted):
                        n_conf += 1; m["conflict"] = True
                        continue
                    m["keep"] = True; accepted.append(m)
                accepted.sort(key=lambda m: (m["s"], m["e"]))
                out, pos = [], 0
                for m in accepted:
                    out.append(src[pos:m["s"]]); out.append(m["new"]); pos = m["e"]
                out.append(src[pos:])
                out = "".join(out)
                for m in items:
                    pooled.append(m); kept_flags.append(m["keep"])
                    if dump_f:
                        dump_f.write(json.dumps({"cell": cell.label, "document_id": did, "s": m["s"], "e": m["e"], "score": m["score"],
                                                 "n_agree": m["n_agree"], "n_votes": m["n_votes"], "gain": m["gain"], "label": m["label"], "lang": m["lang"],
                                                 "frac_changed_unit": m["frac_changed_unit"], "runs": m["runs"], "keep": m["keep"]}) + "\n")
                records.append(submission_record(ref, out))
                n_changed += out != src
                log_f.write(json.dumps({"run_id": run_id, "cell": cell.label, "document_id": did, "ocr": src, "output": out,
                                        "changed": out != src, "n_pooled": len(items), "n_kept": len(accepted),
                                        "n_agree2": sum(m["n_agree"] >= 2 for m in items)}, ensure_ascii=False) + "\n")
            names[cell.stem] = submission_name(system, cell)
            write_jsonl(pred_dir / names[cell.stem], records)
            n_docs += len(refs)
    if dump_f:
        dump_f.close()
    gen_wall = time.perf_counter() - t0
    scores_ = score_run(cells, pred_dir, names, out_dir / "scores", cfg, limit=limit, subset=subset)
    lab = np.array([m["label"] != "neutral" for m in pooled]); y = np.array([1.0 if m["label"] == "helpful" else 0.0 for m in pooled])
    g = np.array([m["gain"] for m in pooled]); s = np.array([m["score"] for m in pooled]); keep = np.array(kept_flags)
    diag = {}
    if lab.any():
        # metrics() thresholds on scores; emulate the decided keep set with a +-inf threshold vector
        thrv = np.where(keep, -np.inf, np.inf)
        diag = metrics(y[lab], s[lab], g[lab], thrv[lab])
        diag["threshold"] = "per block (rule / agreement / conflict resolution)"
    n_ag = np.array([m["n_agree"] for m in pooled])
    diag.update(n_pooled=len(pooled), n_kept=int(keep.sum()), n_conflicts_dropped=n_conf,
                n_agree2=int((n_ag >= 2).sum()), helpful_frac_agree2=float(y[(n_ag >= 2) & lab].mean()) if ((n_ag >= 2) & lab).any() else None,
                helpful_frac_single=float(y[(n_ag == 1) & lab].mean()) if ((n_ag == 1) & lab).any() else None)
    summary = {
        "run_id": run_id, "phase": cfg.get("phase", 2), "system": system, "tag": args.tag, "split": split, "limit": limit,
        "subset": runs[0][1].get("subset"), "model": "+".join(str(r[1].get("model")) for r in runs), "source_runs": [r[1]["run_id"] for r in runs],
        "gate": args.gate, "rule": args.rule, "threshold": args.threshold, "agree_thresholds": args.agree_thresholds, "families": families,
        "block_diag": diag, **git_state(), "config_path": str(args.config), "seed": cfg["seed"], "started_utc": run_id.split("_", 1)[0],
        "n_units": n_docs, "generation_wall_s": round(gen_wall, 1), "wall_s": round(time.perf_counter() - t0, 1),
        "counters": {"n_blocks": len(pooled), "n_blocks_kept": int(keep.sum()), "n_changed_units": n_changed, "n_conflicts": n_conf}, **scores_,
    }
    finalize_run(out_dir, summary)
    print(f"[ensemble/{split}] weighted cMER={summary['cmer_micro_weighted']:.4f} pref={summary['pref_score_cmer_macro_weighted']:.4f} "
          f"pooled={len(pooled)} kept={int(keep.sum())} conflicts={n_conf} -> runs/{run_id}", flush=True)


if __name__ == "__main__":
    main()
