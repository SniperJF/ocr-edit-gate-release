"""Phase 2 (P2-2): decision rules on top of a gate score, chosen on held-out TRAIN blocks only.

    python -m ocrgate.rules --gate runs/gates/<name> [--bins 4] [--out runs/gates/<name>/rules.json]

Reads runs/gates/<name>/heldout_scores.jsonl (written by gate_train at the best epoch: one line per labelled held-out
block with its gate score, labelled gain, language and unit edit-rate features; no text) and chooses, for every rule,
the thresholds that maximise the total labelled gain of the kept blocks on the held-out part (the Phase 1 criterion,
`gate_train.best_threshold`). Nothing from dev is read.

Rules (a block is kept iff score >= t(block)):
  global      (i)   one threshold for all blocks (= the Phase 1 rule)
  editrate    (ii)  t depends on the unit's edit rate (frac_changed_unit = fraction of the OCR unit's characters the
                    rewriter changed): quantile bins of the held-out blocks, one threshold per bin. A unit the rewriter
                    barely touched is treated as "confident" and gets its own (usually lower) threshold.
  sysconf     (ii') t depends on the rewriter's confidence as seen by the gate: proxy = mean gate score over ALL blocks
                    of the system on the data being gated (label-free, so it is available for any system, including an
                    unseen one). On held-out train every rewriter run gives one (proxy, best threshold) point; t(proxy) is
                    a least-squares line through those points, clipped to [t_min, t_max] of the points.
  lang        (iii) one threshold per language (en / fr / de; other -> global)
  lang_editrate     (iii) x (ii): per language, per edit-rate bin (same bin edges as editrate)

Every rule is evaluated on the same held-out blocks (total gain kept, helpful kept / harmful rejected, and per-rewriter
gain kept vs the rewriter's keep-all gain, i.e. whether the rule hurts a strong system on held-out). Because the rules
are chosen on these very blocks, the more flexible rules are favoured in-sample; `--cv K` therefore also reports a
cross-fitted held-out gain: the held-out units are split into K folds by (cell, document id) with seed 1234, each rule
is chosen on K-1 folds and scored on the remaining fold, and the K fold gains are summed (`cv_eval`). The frozen rule is
the one with the highest cross-fitted gain. All of this is written to rules.json next to the gate; `gate_apply --rule
NAME` applies one rule to a run.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import ROOT
from .gate_train import best_threshold

LANGS = ("en", "fr", "de")


def _gain_kept(scores, gains, thr):
    keep = scores >= thr
    return float(gains[keep].sum()), int(keep.sum())


def _bin_index(x: np.ndarray, edges: list[float]) -> np.ndarray:
    """edges = interior cut points (len K-1) -> bin 0..K-1."""
    return np.searchsorted(np.array(edges), x, side="right")


def choose_rules(rows: list[dict], n_bins: int = 4) -> dict:
    s = np.array([r["score"] for r in rows]); g = np.array([r["gain"] for r in rows])
    y = np.array([1.0 if r["label"] == "helpful" else 0.0 for r in rows])
    lang = np.array([r.get("lang") or "other" for r in rows]); run = np.array([r["src_run"] for r in rows])
    er = np.array([r["frac_changed_unit"] for r in rows])
    rules = {}
    # (i) global
    t, _ = best_threshold(s, g)
    rules["global"] = {"type": "global", "threshold": t}
    # (ii) unit edit-rate bins (quantile edges over held-out blocks)
    qs = np.quantile(er, [i / n_bins for i in range(1, n_bins)]).tolist()
    bi = _bin_index(er, qs)
    thr_bins = []
    for k in range(n_bins):
        m = bi == k
        tk, _ = best_threshold(s[m], g[m]) if m.any() else (t, 0.0)
        thr_bins.append(tk if np.isfinite(tk) else 1.0 + 1e-9)
    rules["editrate"] = {"type": "editrate", "edges": qs, "thresholds": thr_bins,
                         "bin_counts": [int((bi == k).sum()) for k in range(n_bins)]}
    # (ii') system-confidence proxy -> threshold (one point per held-out rewriter run)
    pts = []
    for r in sorted(set(run)):
        m = run == r
        tr, _ = best_threshold(s[m], g[m])
        if np.isfinite(tr):
            pts.append({"run": r, "proxy": float(s[m].mean()), "threshold": float(tr), "n_blocks": int(m.sum()),
                        "mean_editrate": float(er[m].mean()), "helpful_frac": float(y[m].mean())})
    px = np.array([p["proxy"] for p in pts]); pt = np.array([p["threshold"] for p in pts])
    if len(pts) >= 2 and px.std() > 1e-9:
        b, a = np.polyfit(px, pt, 1)
    else:
        b, a = 0.0, float(pt.mean()) if len(pts) else t
    rules["sysconf"] = {"type": "sysconf", "slope": float(b), "intercept": float(a), "clip": [float(pt.min()), float(pt.max())] if len(pts) else [t, t],
                        "points": pts, "note": "proxy = mean gate score over all blocks of the system being gated"}
    # (iii) per language
    thr_lang = {}
    for L in LANGS:
        m = lang == L
        tl, _ = best_threshold(s[m], g[m]) if m.any() else (t, 0.0)
        thr_lang[L] = tl if np.isfinite(tl) else 1.0 + 1e-9
    rules["lang"] = {"type": "lang", "thresholds": thr_lang, "default": t,
                     "counts": {L: int((lang == L).sum()) for L in LANGS}}
    # (iii) x (ii)
    thr_le = {}
    for L in LANGS:
        thr_le[L] = []
        for k in range(n_bins):
            m = (lang == L) & (bi == k)
            tk, _ = best_threshold(s[m], g[m]) if m.sum() >= 50 else (thr_lang[L], 0.0)
            thr_le[L].append(tk if np.isfinite(tk) else 1.0 + 1e-9)
    rules["lang_editrate"] = {"type": "lang_editrate", "edges": qs, "thresholds": thr_le, "default": thr_bins}
    # evaluation of every rule on the held-out blocks
    evals = {}
    for name, rule in rules.items():
        thr = thresholds_for(rule, rows, s)
        keep = s >= thr
        per_run = {}
        for r in sorted(set(run)):
            m = run == r
            per_run[r] = {"gain_kept": float(g[m & keep].sum()), "gain_keep_all": float(g[m].sum()), "gain_oracle": float(g[m][g[m] > 0].sum()),
                          "kept_frac": float(keep[m].mean()), "helpful_kept": float(keep[m & (y == 1)].mean()) if (m & (y == 1)).any() else None,
                          "harmful_rejected": float((~keep)[m & (y == 0)].mean()) if (m & (y == 0)).any() else None}
        per_lang = {L: {"gain_kept": float(g[(lang == L) & keep].sum()), "gain_keep_all": float(g[lang == L].sum())} for L in LANGS}
        evals[name] = {"gain_kept": float(g[keep].sum()), "gain_keep_all": float(g.sum()), "gain_oracle": float(g[g > 0].sum()),
                       "kept_frac": float(keep.mean()), "helpful_kept": float(keep[y == 1].mean()), "harmful_rejected": float((~keep)[y == 0].mean()),
                       "per_run": per_run, "per_lang": per_lang}
    return {"n_heldout_blocks": len(rows), "n_bins": n_bins, "rules": rules, "heldout_eval": evals}


def cross_fit(rows: list[dict], n_bins: int, k: int, seed: int = 1234) -> dict:
    """Cross-fitted held-out gain per rule (rules chosen on K-1 folds of the held-out UNITS, scored on the K-th)."""
    import random
    units = sorted({(r["cell"], r["document_id"]) for r in rows})
    rng = random.Random(seed); rng.shuffle(units)
    fold_of = {u: i % k for i, u in enumerate(units)}
    out = {}
    for f in range(k):
        fit = [r for r in rows if fold_of[(r["cell"], r["document_id"])] != f]
        ev = [r for r in rows if fold_of[(r["cell"], r["document_id"])] == f]
        res = choose_rules(fit, n_bins)
        s_ev = np.array([r["score"] for r in ev]); g_ev = np.array([r["gain"] for r in ev])
        run_ev = np.array([r["src_run"] for r in ev])
        for name, rule in res["rules"].items():
            # sysconf uses the per-system proxy: evaluate per run of the evaluation fold
            thr = np.empty(len(ev))
            for r_ in sorted(set(run_ev)):
                m = run_ev == r_
                thr[m] = thresholds_for(rule, [ev[i] for i in np.flatnonzero(m)], s_ev[m])
            keep = s_ev >= thr
            d = out.setdefault(name, {"gain_kept": 0.0, "gain_keep_all": 0.0, "gain_oracle": 0.0, "per_run_gain_kept": {}, "per_run_gain_keep_all": {}})
            d["gain_kept"] += float(g_ev[keep].sum()); d["gain_keep_all"] += float(g_ev.sum()); d["gain_oracle"] += float(g_ev[g_ev > 0].sum())
            for r_ in sorted(set(run_ev)):
                m = run_ev == r_
                d["per_run_gain_kept"][r_] = d["per_run_gain_kept"].get(r_, 0.0) + float(g_ev[m & keep].sum())
                d["per_run_gain_keep_all"][r_] = d["per_run_gain_keep_all"].get(r_, 0.0) + float(g_ev[m].sum())
    return {"k": k, "seed": seed, "n_units": len(units), "eval": out, "best": max(out, key=lambda n: out[n]["gain_kept"])}


def thresholds_for(rule: dict, blocks: list[dict], scores: np.ndarray) -> np.ndarray:
    """Per-block threshold under `rule` for the blocks of ONE system/run (scores = its gate scores)."""
    n = len(blocks)
    if rule["type"] == "global":
        return np.full(n, rule["threshold"])
    if rule["type"] == "editrate":
        bi = _bin_index(np.array([b["frac_changed_unit"] for b in blocks]), rule["edges"])
        return np.array(rule["thresholds"])[bi]
    if rule["type"] == "sysconf":
        proxy = float(scores.mean()) if n else 0.0
        t = rule["intercept"] + rule["slope"] * proxy
        lo, hi = rule["clip"]
        return np.full(n, float(min(max(t, lo), hi)))
    if rule["type"] == "lang":
        return np.array([rule["thresholds"].get(b.get("lang") or "", rule["default"]) for b in blocks])
    if rule["type"] == "lang_editrate":
        bi = _bin_index(np.array([b["frac_changed_unit"] for b in blocks]), rule["edges"])
        return np.array([rule["thresholds"].get(b.get("lang") or "", rule["default"])[k] for b, k in zip(blocks, bi)])
    raise ValueError(rule["type"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--bins", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cv", type=int, default=0, help="K-fold cross-fitted evaluation of the rules on the held-out units")
    args = ap.parse_args()
    gate_dir = ROOT / args.gate
    rows = [json.loads(l) for l in open(gate_dir / "heldout_scores.jsonl", encoding="utf-8")]
    res = choose_rules(rows, args.bins)
    res["gate"] = str(args.gate)
    if args.cv:
        res["cv_eval"] = cross_fit(rows, args.bins, args.cv)
        print(f"cross-fitted ({args.cv}-fold) held-out gain per rule: " + ", ".join(f"{n} {v['gain_kept']:.3f}" for n, v in res["cv_eval"]["eval"].items()) + f" -> best {res['cv_eval']['best']}")
    out = Path(args.out) if args.out else gate_dir / "rules.json"
    tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(res, indent=1), encoding="utf-8"); tmp.replace(out)
    for name, ev in res["heldout_eval"].items():
        print(f"{name:14s} gain kept {ev['gain_kept']:.3f} / keep-all {ev['gain_keep_all']:.3f} / oracle {ev['gain_oracle']:.3f}  kept {ev['kept_frac']:.3f}  helpful kept {ev['helpful_kept']:.3f}  harmful rejected {ev['harmful_rejected']:.3f}")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "points"} for k, v in res["rules"].items()}, indent=1))


if __name__ == "__main__":
    main()
