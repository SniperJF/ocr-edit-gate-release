"""Phase 2: fixed split of the TRAIN split by ground-truth groups (no GPU, no dev/test text read beyond GT hashes).

    python -m ocrgate.p2_split --config configs/phase1.yaml --out configs/p2_split.json --subset-out configs/p2_heldout_subset.json

Units of all train cells are grouped by ground truth (sha1 of the gold unit; DTA matched pages share one gold text
across the three noise-level cells and 17 pages also occur in the unmatched files) and, additionally, units that share
a document id inside one cell are forced into one group (overproof train has one id three times; the run.py --subset
mechanism selects by document id). Units whose ground truth also occurs in DEV are flagged `dev_overlap` and excluded
from every role (never used for LoRA training nor for the gate; as in Phase 1 gate training).

Groups are stratified by the set of cells they occur in; inside each stratum, with seed 1234, ~30% of the groups are
drawn as LORA-HELD-OUT (role `heldout`: never seen by any Phase 2 LoRA rewriter), the rest is `fit` (LoRA training
data). Inside the held-out part, half of the groups (again per stratum, same rng) are marked `gate_heldout`: these
units are the ONLY ones on which Phase 2 gate hyper-parameters, thresholds and decision rules are selected; the other
half of the held-out part (`heldout_fit`) supplies LoRA-rewriter blocks for gate FITTING. Zero-shot rewriter blocks
from `fit` and `heldout_fit` units are gate-fit data, zero-shot blocks from `gate_heldout` units are gate-selection data.

Outputs: --out JSON with one record per unit {cell, unit_idx, document_id, gt_sha1, group, role, dev_overlap} plus the
counts; --subset-out {cell label: [document ids]} of all held-out units (role heldout_fit + gate_heldout), the format
of `ocrgate.run --subset`, so the LoRA rewriters can be run on exactly the held-out units.
"""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict

from .data import ROOT, load_config, split_cells
from .score import select_refs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--out", default="configs/p2_split.json")
    ap.add_argument("--subset-out", default="configs/p2_heldout_subset.json")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--heldout-frac", type=float, default=0.30)
    ap.add_argument("--gate-frac-of-heldout", type=float, default=0.50)
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    dev_hashes = set()
    for cell in split_cells(cfg, "dev"):
        for ref in select_refs(cell):
            dev_hashes.add(hashlib.sha1(ref["ground_truth"]["transcription_unit"].encode("utf-8")).hexdigest())
    units = []
    for cell in split_cells(cfg, "train"):
        for ui, ref in enumerate(select_refs(cell)):
            h = hashlib.sha1(ref["ground_truth"]["transcription_unit"].encode("utf-8")).hexdigest()
            units.append({"cell": cell.label, "unit_idx": ui, "document_id": ref["document_metadata"]["document_id"],
                          "gt_sha1": h, "dev_overlap": h in dev_hashes})
    # union-find over gt_sha1 and (cell, document_id)
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for u in units:
        union(("gt", u["gt_sha1"]), ("doc", u["cell"], u["document_id"]))
    for u in units:
        u["group"] = find(("gt", u["gt_sha1"]))
    # name groups deterministically by the smallest gt_sha1 they contain
    g_min = {}
    for u in units:
        g_min[u["group"]] = min(g_min.get(u["group"], u["gt_sha1"]), u["gt_sha1"])
    for u in units:
        u["group"] = g_min[u["group"]]
    # strata = sorted tuple of cells the group occurs in (dev-overlap groups are excluded from the draw)
    cells_of = defaultdict(set)
    overlap_groups = {u["group"] for u in units if u["dev_overlap"]}
    for u in units:
        if u["group"] not in overlap_groups:
            cells_of[u["group"]].add(u["cell"])
    strata = defaultdict(list)
    for g, cs in cells_of.items():
        strata[tuple(sorted(cs))].append(g)
    rng = random.Random(args.seed)
    role_of = {}
    for st in sorted(strata):
        gs = sorted(strata[st]); rng.shuffle(gs)
        n_h = round(len(gs) * args.heldout_frac)
        held = gs[:n_h]
        n_g = round(len(held) * args.gate_frac_of_heldout)
        for g in held[:n_g]:
            role_of[g] = "gate_heldout"
        for g in held[n_g:]:
            role_of[g] = "heldout_fit"
        for g in gs[n_h:]:
            role_of[g] = "fit"
    for u in units:
        u["role"] = "dev_overlap" if u["group"] in overlap_groups else role_of[u["group"]]
    counts = Counter(u["role"] for u in units)
    per_cell = {c: dict(Counter(u["role"] for u in units if u["cell"] == c)) for c in sorted({u["cell"] for u in units})}
    n_groups = {r: len({u["group"] for u in units if u["role"] == r}) for r in counts}
    out = {"seed": args.seed, "heldout_frac": args.heldout_frac, "gate_frac_of_heldout": args.gate_frac_of_heldout,
           "n_units": len(units), "n_groups": len({u["group"] for u in units}), "counts_units": dict(counts),
           "counts_groups": n_groups, "strata": {" + ".join(k): len(v) for k, v in sorted(strata.items())},
           "per_cell": per_cell, "units": units}
    (ROOT / args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    subset = defaultdict(list)
    for u in units:
        if u["role"] in ("heldout_fit", "gate_heldout") and u["document_id"] not in subset[u["cell"]]:
            subset[u["cell"]].append(u["document_id"])
    (ROOT / args.subset_out).write_text(json.dumps(dict(subset), indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "units"}, indent=1))


if __name__ == "__main__":
    main()
