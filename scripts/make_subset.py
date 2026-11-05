#!/usr/bin/env python3
"""Phase 0b-3 probe subset: ~100 dev units, fixed seed, stratified across cells (document ids only).

    python scripts/make_subset.py  -> configs/subset_0b3.json
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ocrgate.data import ROOT, load_config, read_jsonl, split_cells  # noqa: E402

SEED = 1234
# fixed allocation per cell (small cells get a floor so that per-cell numbers are not meaningless)
ALLOC = {
    "icdar2017_v1.1_dev_en": 30,
    "impresso-snippets_v1.0_dev_en": 5,
    "impresso-snippets_v1.0_dev_fr": 5,
    "impresso-snippets_v1.0_dev_de": 5,
    "dta19-l0_v0.1_dev-unmatched_de": 15,
    "dta19-l1_v0.1_dev-unmatched_de": 15,
    "dta19-l2_v0.1_dev-unmatched_de": 15,
    "overproof-combined_v1.0_dev_en": 10,
}
cfg = load_config(ROOT / "configs/phase0b.yaml")
rng = random.Random(SEED)
subset = {}
for cell in split_cells(cfg, "dev"):
    ids = [r["document_metadata"]["document_id"] for r in read_jsonl(cell.path)]
    k = ALLOC[cell.label]
    chosen = set(rng.sample(ids, k))
    subset[cell.label] = [i for i in ids if i in chosen]  # keep reference-file order
out = ROOT / "configs" / "subset_0b3.json"
out.write_text(json.dumps({"seed": SEED, "note": "Phase 0b-3 probe subset (document ids only)", **subset}, indent=1) + "\n")
print(out, {k: len(v) for k, v in subset.items()}, "total", sum(len(v) for v in subset.values()))
