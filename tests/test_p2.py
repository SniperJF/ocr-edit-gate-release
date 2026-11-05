"""Phase 2 offline tests: rule thresholds, ensemble conflict resolution, split invariants."""
import json
from pathlib import Path

import numpy as np

from ocrgate.ensemble import choose_agree_thresholds, conflicts
from ocrgate.rules import choose_rules, thresholds_for

ROOT = Path(__file__).resolve().parent.parent


def test_conflicts():
    assert conflicts((0, 3), (2, 5))
    assert not conflicts((0, 3), (3, 5))          # adjacent blocks may both apply
    assert conflicts((2, 2), (0, 3))              # insertion strictly inside a block
    assert not conflicts((3, 3), (0, 3))          # insertion at the edge is allowed
    assert conflicts((4, 4), (4, 4))              # two insertions at one point
    assert not conflicts((4, 4), (5, 5))


def test_rules_roundtrip():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(2000):
        lang = ["en", "fr", "de"][i % 3]
        run = "r%d" % (i % 4)
        er = float(rng.uniform(0, 0.3))
        score = float(rng.uniform(0, 1))
        helpful = rng.uniform() < score * (1.2 if lang == "fr" else 0.9)
        rows.append({"src_run": run, "cell": "c", "document_id": str(i), "block_idx": 0, "lang": lang, "dataset": "d",
                     "score": score, "gain": (0.01 if helpful else -0.01), "label": "helpful" if helpful else "harmful",
                     "frac_changed_unit": er, "n_blocks_unit": 3, "len_unit": 500, "len_ocr": 4, "len_new": 4})
    res = choose_rules(rows, n_bins=3)
    assert set(res["rules"]) == {"global", "editrate", "sysconf", "lang", "lang_editrate"}
    s = np.array([r["score"] for r in rows])
    for name, rule in res["rules"].items():
        thr = thresholds_for(rule, rows, s)
        assert thr.shape == (len(rows),)
        ev = res["heldout_eval"][name]
        assert ev["gain_kept"] >= ev["gain_keep_all"] - 1e-9  # a rule never does worse than keep-all on its own selection set
    # lang rule must be at least as good as global on held-out (it contains global as a special case)
    assert res["heldout_eval"]["lang"]["gain_kept"] >= res["heldout_eval"]["global"]["gain_kept"] - 1e-9
    # sysconf threshold is constant within a run and inside the clip range
    thr = thresholds_for(res["rules"]["sysconf"], rows, s)
    lo, hi = res["rules"]["sysconf"]["clip"]
    assert lo - 1e-9 <= thr[0] <= hi + 1e-9


def test_agree_thresholds():
    rows = [{"score": 0.9, "n_agree": 2, "gain": 0.1}, {"score": 0.2, "n_agree": 2, "gain": 0.1},
            {"score": 0.6, "n_agree": 1, "gain": -0.1}, {"score": 0.8, "n_agree": 1, "gain": 0.1}]
    r = choose_agree_thresholds(rows)
    assert r["t_agree"] <= 0.2 and 0.6 < r["t_single"] <= 0.8


def test_split_invariants():
    sp = json.loads((ROOT / "configs" / "p2_split.json").read_text())
    roles = {}
    for u in sp["units"]:
        roles.setdefault(u["group"], set()).add(u["role"])
    assert all(len(r) == 1 for r in roles.values())  # one role per ground-truth group
    sub = json.loads((ROOT / "configs" / "p2_heldout_subset.json").read_text())
    held = {(u["cell"], u["document_id"]) for u in sp["units"] if u["role"] in ("heldout_fit", "gate_heldout")}
    assert held == {(c, d) for c, ds in sub.items() for d in ds}
    n = sp["counts_units"]
    assert abs((n["heldout_fit"] + n["gate_heldout"]) / sp["n_units"] - 0.30) < 0.01
