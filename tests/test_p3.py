"""Phase 3 offline tests: multi-level agreement thresholds and family-aware agreement."""
import numpy as np

from ocrgate.ensemble import choose_agree_thresholds, level_of
from ocrgate.p3_pool import evaluate


def test_level_of():
    assert [level_of(n, 2) for n in (1, 2, 3, 5)] == [0, 1, 1, 1]
    assert [level_of(n, 3) for n in (1, 2, 3, 5)] == [0, 1, 2, 2]


def test_three_levels_separate():
    rows = [{"score": 0.9, "n_agree": 3, "gain": 0.1}, {"score": 0.1, "n_agree": 3, "gain": 0.1},
            {"score": 0.5, "n_agree": 2, "gain": 0.1}, {"score": 0.3, "n_agree": 2, "gain": -0.1},
            {"score": 0.8, "n_agree": 1, "gain": 0.1}, {"score": 0.6, "n_agree": 1, "gain": -0.1}]
    r = choose_agree_thresholds(rows, 3)
    assert r["levels"] == 3 and len(r["thresholds"]) == 3
    assert r["thresholds"][2] <= 0.1 and 0.3 < r["thresholds"][1] <= 0.5 and 0.6 < r["thresholds"][0] <= 0.8
    assert abs(r["gain_kept_agree_rule"] - 0.4) < 1e-9
    r2 = choose_agree_thresholds(rows, 2)
    assert r2["t_single"] == r["thresholds"][0]


def test_families_vs_votes():
    # two members of one family ("a") and one of another ("b") propose the same block
    u = ("cell", "doc", "ocr text")
    blk = (0, 3, "new", 0.5, 0.2)
    members = [("a1", "a", "run1", {u: [blk]}), ("a2", "a", "run2", {u: [blk]}), ("b", "b", "run3", {u: [blk]})]
    fam = evaluate(members, 3, "families")
    votes = evaluate(members, 3, "votes")
    assert fam["n_per_level"] == [0, 1, 0]      # 2 distinct families -> level 1
    assert votes["n_per_level"] == [0, 0, 1]    # 3 votes -> level 2
    assert fam["n_candidates"] == 1 and abs(fam["gain"] - 0.2) < 1e-9
