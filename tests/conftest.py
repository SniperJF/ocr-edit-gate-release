"""Tests that read HIPE-OCRepair reference files are skipped when the data repo is not available
(run scripts/fetch_data.sh or set HIPE_DATA_DIR / HIPE_EVAL_DIR)."""
import pytest

from ocrgate.data import data_dir

NEEDS_DATA = {
    "test_gold_hypothesis_round_trip",
    "test_gt_splice_reconstructs_ground_truth",
    "test_systems_with_fake_llm",
    "test_window_gated_fake",
}


def pytest_collection_modifyitems(config, items):
    if (data_dir() / "data").is_dir():
        return
    skip = pytest.mark.skip(reason="HIPE-OCRepair-2026-data not found (run scripts/fetch_data.sh or set HIPE_DATA_DIR)")
    for item in items:
        if item.name in NEEDS_DATA:
            item.add_marker(skip)
