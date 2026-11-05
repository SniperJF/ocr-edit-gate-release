"""model.safetensors and model.pt of a gate give identical predictions (CPU).

Needs torch, transformers and safetensors (the GPU environment) and a gate directory holding gate.json, model.pt and
model.safetensors: set OCRGATE_TEST_GATE_DIRS to one or more such directories (os.pathsep-separated). Skipped otherwise.
The base encoder (Qwen/Qwen3-0.6B) must be available locally or downloadable; all weights are then overwritten by the
gate checkpoint."""
import os
from pathlib import Path

import pytest

DIRS = [Path(p) for p in os.environ.get("OCRGATE_TEST_GATE_DIRS", "").split(os.pathsep) if p]

pytestmark = pytest.mark.skipif(not DIRS, reason="set OCRGATE_TEST_GATE_DIRS to gate dirs with model.pt + model.safetensors")


def _blocks():
    base = dict(lang="en", dataset="synthetic", lp_ok=True, lp_mean=-0.3, lp_min=-1.2, lp_sum=-0.9, lp_n=3, lp_first=-0.1,
                lp_unit_mean=-0.05, n_blocks_unit=4, len_unit=200, frac_changed_unit=0.03, is_insert=False, is_delete=False)
    pairs = [("The qnick brown fox ", "jurnped", "jumped", " over the lazy dog."),
             ("Tbe ", "Tbe", "The", " house was large."),
             ("in the year ", "l848", "1848", ", the city grew"),
             ("Die Stadt ", "wnrde", "wurde", " gegründet."),
             ("une ville ", "tres", "très", " ancienne")]
    out = []
    for i, (left, ocr, new, right) in enumerate(pairs):
        b = dict(base, left=left, ocr_text=ocr, new_text=new, right=right, len_ocr=len(ocr), len_new=len(new))
        if i == 4:
            b["lp_ok"] = False
        out.append(b)
    return out


@pytest.mark.parametrize("gate_dir", DIRS, ids=[d.name for d in DIRS])
def test_safetensors_matches_pt(gate_dir):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    from ocrgate.gate import load_gate, load_state_dict_file, load_tokenizer, predict

    sd_pt = load_state_dict_file(gate_dir, "pt")
    sd_st = load_state_dict_file(gate_dir, "safetensors")
    assert set(sd_pt) == set(sd_st)
    for k in sd_pt:
        assert sd_pt[k].dtype == sd_st[k].dtype and sd_pt[k].shape == sd_st[k].shape and torch.equal(sd_pt[k], sd_st[k]), k
    del sd_pt, sd_st

    m_pt, std, meta = load_gate(gate_dir, device="cpu", fmt="pt")
    m_st, _, _ = load_gate(gate_dir, device="cpu", fmt="safetensors")
    m_auto, _, _ = load_gate(gate_dir, device="cpu")  # auto prefers model.safetensors
    tok = load_tokenizer(meta["encoder"])
    blocks = _blocks()
    kw = dict(max_len=meta.get("max_len", 256), use_lp=meta.get("use_lp", True), batch_size=8, device="cpu")
    p_pt = predict(m_pt, tok, std, blocks, **kw)
    p_st = predict(m_st, tok, std, blocks, **kw)
    p_auto = predict(m_auto, tok, std, blocks, **kw)
    assert p_pt == p_st == p_auto
    assert all(0.0 <= p <= 1.0 for p in p_pt)
