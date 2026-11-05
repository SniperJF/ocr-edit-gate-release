"""Phase 3 (P3-4): choose the ensemble pool, the agreement rule and its thresholds on the gate_heldout units.

    python -m ocrgate.p3_pool --members NAME=runs/<heldout run>[:FAMILY] ... --gate runs/gates/p2_pooled
        [--groups GROUP=NAME,NAME ...] [--levels 2 3] [--split configs/p2_split.json] [--out results/p3_pool_search.json]

Every member is a zeroshot run on the held-out train subset (configs/p2_heldout_subset.json) with blocks.jsonl; its blocks are
scored with the gate (cached per run, computed on the GPU if missing). Only units of role `gate_heldout` are used. A
member's FAMILY (default: its NAME) decides how agreement is counted: a block proposed by k members counts n_agree =
number of distinct families (so the sampled candidates of one rewriter add votes to their own family only) and, as a
second variant, n_votes = number of members. --groups ties members together (e.g. the 4 samples of a rewriter are
added or dropped as one option). For every subset of options (every group in or out, singletons in or out) and every
agreement rule (families / votes) x number of levels, the thresholds are chosen per level by `best_threshold` on the
labelled gains (as in Phase 2, conflicts ignored) and the pool is ranked by total held-out gain kept. The P2 pool
(LoRA-24B + LoRA-32B + 72B zero-shot, 2 levels, families) is always evaluated as the reference. Output: json with every
configuration's gain and thresholds; the winner is the one with the highest held-out gain (tie -> fewer members).
"""

import argparse
import itertools
import json
from collections import defaultdict

import numpy as np

from .data import ROOT
from .ensemble import gate_scores, level_of, load_run
from .gate_train import best_threshold


def load_member(spec: str, gate: str, device: str, ok_units: set):
    name, rest = spec.split("=", 1)
    run, fam = (rest.split(":", 1) + [name])[:2] if ":" in rest else (rest, name)
    run_dir, summary, blocks, logs = load_run(run)
    sc = gate_scores(run_dir, gate, blocks, device)
    order = {}
    for line in (run_dir / "log.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        order.setdefault(r["cell"], []).append((r["cell"], r["document_id"], r["ocr"]))
    cands = defaultdict(list)  # unit key -> list of (s, e, new, score, gain)
    for b, s_ in zip(blocks, sc):
        key = order[b["cell"]][b["unit_idx"]]
        if (key[0], key[1]) not in ok_units:
            continue
        hyp = logs[key]["output"]
        cands[key].append((b["s"], b["e"], hyp[b["ds"]:b["de"]], float(s_), float(b["gain"])))
    return name, fam, run, cands


def evaluate(members: list, levels: int, by: str) -> dict:
    """Merge identical proposals over the given members; choose per-level thresholds; return gain kept etc."""
    merged = defaultdict(dict)  # unit -> (s,e,new) -> [score, gain, fams, votes]
    for name, fam, _, cands in members:
        for key, lst in cands.items():
            d = merged[key]
            for s, e, new, score, gain in lst:
                k = (s, e, new)
                m = d.get(k)
                if m is None:
                    d[k] = [score, gain, {fam}, 1]
                else:
                    m[0] = max(m[0], score); m[2].add(fam); m[3] += 1
    scores, gains, nag = [], [], []
    for d in merged.values():
        for score, gain, fams, votes in d.values():
            scores.append(score); gains.append(gain); nag.append(len(fams) if by == "families" else votes)
    s = np.array(scores); g = np.array(gains); n = np.array(nag)
    lv = np.array([level_of(x, levels) for x in n]) if len(n) else np.array([], dtype=int)
    ts, per = [], []
    for k in range(levels):
        m = lv == k
        t, gk = best_threshold(s[m], g[m]) if m.any() else (float("inf"), 0.0)
        ts.append(t); per.append(gk)
    tg, gg = best_threshold(s, g) if len(s) else (float("inf"), 0.0)
    return {"gain": float(sum(per)), "thresholds": ts, "gain_per_level": per, "n_per_level": [int((lv == k).sum()) for k in range(levels)],
            "gain_global_threshold": gg, "t_global": tg, "n_candidates": int(len(s)), "gain_oracle": float(g[g > 0].sum()) if len(g) else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--gate", default="runs/gates/p2_pooled")
    ap.add_argument("--groups", nargs="*", default=[])
    ap.add_argument("--levels", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--split", default="configs/p2_split.json")
    ap.add_argument("--roles", default="gate_heldout")
    ap.add_argument("--reference", default="lora24b,lora32b,q72b", help="member names of the P2 pool (always evaluated, 2 levels, families)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/p3_pool_search.json")
    args = ap.parse_args()
    sp = json.loads((ROOT / args.split).read_text(encoding="utf-8"))
    roles = set(args.roles.split(","))
    ok = {(u["cell"], u["document_id"]) for u in sp["units"] if u["role"] in roles}
    members = {m[0]: m for m in (load_member(spec, args.gate, args.device, ok) for spec in args.members)}
    print("members:", {k: (v[1], v[2], sum(len(x) for x in v[3].values())) for k, v in members.items()})
    groups = {}
    for g in args.groups:
        gname, names = g.split("=", 1)
        groups[gname] = names.split(",")
    grouped = {n for ns in groups.values() for n in ns}
    options = [(k, v) for k, v in groups.items()] + [(n, [n]) for n in members if n not in grouped]
    results = []
    ref = args.reference.split(",")
    if all(r in members for r in ref):
        res = evaluate([members[r] for r in ref], 2, "families"); res.update(members=ref, levels=2, by="families", reference=True); results.append(res)
        print(f"reference P2 pool {ref}: gain {res['gain']:.3f} thresholds {res['thresholds']}")
    for r in range(1, len(options) + 1):
        for combo in itertools.combinations(options, r):
            names = [n for _, ns in combo for n in ns]
            if len(names) < 2:
                continue
            for levels in args.levels:
                for by in ("families", "votes"):
                    if by == "votes" and len({members[n][1] for n in names}) == len(names):
                        continue  # families == votes when every member is its own family
                    res = evaluate([members[n] for n in names], levels, by)
                    res.update(members=names, options=[o for o, _ in combo], levels=levels, by=by, reference=False)
                    results.append(res)
    results.sort(key=lambda r: (-r["gain"], len(r["members"]), r["levels"]))
    for res in results[:15]:
        print(f"gain {res['gain']:.3f} (global-thr {res['gain_global_threshold']:.3f}, oracle {res['gain_oracle']:.3f}) levels {res['levels']} by {res['by']:8s} "
              f"thr {[round(t, 3) for t in res['thresholds']]} n {res['n_per_level']} members {res['members']}")
    best = results[0]
    (ROOT / args.out).write_text(json.dumps({"roles": args.roles, "gate": args.gate, "n_units": len(ok), "best": best, "all": results,
                                             "member_runs": {k: v[2] for k, v in members.items()}, "member_families": {k: v[1] for k, v in members.items()}}, indent=1), encoding="utf-8")
    print("best:", json.dumps({k: best[k] for k in ("members", "levels", "by", "thresholds", "gain")}))


if __name__ == "__main__":
    main()
