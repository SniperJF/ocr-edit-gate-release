"""Offline tests for the Phase 1 block machinery (no GPU, no LLM)."""

from ocrgate.edits import apply_blocks, edit_blocks, filter_edits
from ocrgate.systems import gated_spans
from ocrgate.blocks import cmer, char_logprobs
from hipe_ocrepair_scorer.ocrepair_eval import norm

GATE_CFG = {"expand_to_tokens": True, "merge_adjacent_tokens": True, "drop_scoring_irrelevant": True}


def test_apply_blocks_endpoints():
    src = "The prioe one penny, is 'pub luhed every Monday."
    hyp = "The price one penny, is published every Monday!"
    b = edit_blocks(src, hyp)
    assert apply_blocks(src, hyp, b, [True] * len(b)) == hyp
    assert apply_blocks(src, hyp, b, [False] * len(b)) == src
    # keeping only the blocks that overlap oracle spans equals the Phase 0b block filter
    tgt = "The price one penny, is published every Monday."
    spans = gated_spans(src, tgt, GATE_CFG)
    from ocrgate.edits import _touches
    keep = [_touches(spans, s, e) for s, e, _, _ in b]
    assert apply_blocks(src, hyp, b, keep) == filter_edits(src, hyp, spans, True, "block")[0]


def test_label_rule_is_scorer_normalised():
    gold = norm("The price one penny")
    base = cmer(gold, "The prioe one penny")
    assert cmer(gold, "The price one penny") < base  # helpful
    assert cmer(gold, "The PRIOE one penny") == base  # case-only: neutral for the scorer
    assert cmer(gold, "The prioe one penny.") == base  # punctuation-only: neutral
    assert cmer(gold, "The prior one penny") >= base  # not closer


def test_char_logprobs_maps_output_to_tokens():
    tokens = ["<think>", "\n\n", "</think>", "\n\n", "Hel", "lo", " wor", "ld", "<|im_end|>"]
    lps = [0.0, 0.0, 0.0, 0.0, -0.1, -0.2, -0.3, -0.4, 0.0]
    out = "Hello world"
    lp_char, tok_char, unit_mean = char_logprobs(out, tokens, lps)
    assert len(lp_char) == len(out)
    assert lp_char[0] == -0.1 and lp_char[3] == -0.2 and lp_char[5] == -0.3 and lp_char[-1] == -0.4
    assert char_logprobs("something else", tokens, lps) is None
