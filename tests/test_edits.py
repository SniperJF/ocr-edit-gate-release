"""Offline checks of the edit filter (Phase 0b)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hipe_ocrepair_scorer.ocrepair_eval import norm  # noqa: E402

from ocrgate.align import Span, oracle_spans, splice  # noqa: E402
from ocrgate.data import load_config, read_jsonl, split_cells, ROOT  # noqa: E402
from ocrgate.edits import filter_edits, span_replacements  # noqa: E402
from ocrgate.systems import _scoring_relevant  # noqa: E402

CFG = load_config(ROOT / "configs/phase0.yaml")


def _dev_records():
    for cell in split_cells(CFG, "dev"):
        yield from read_jsonl(cell.path)


def test_toy():
    src = "The prioe one penny, is 'pub luhed every Monday"
    hyp = "The price one penny, is published every Mon day"  # two good fixes + one bad split outside
    spans = [Span(4, 9, 0, 0), Span(24, 34, 0, 0)]
    assert src[4:9] == "prioe" and src[24:34] == "'pub luhed"
    for g in ("block", "char"):
        out, st = filter_edits(src, hyp, spans, True, g)
        assert out == "The price one penny, is published every Monday", (g, out)
        assert st.n_blocks >= 3 and st.n_blocks_kept == st.n_blocks - 1
        out, st = filter_edits(src, hyp, spans, False, g)
        assert out == "The prioe one penny, is 'pub luhed every Mon day", (g, out)
        assert st.n_blocks_kept == 1
    # insertion at the right edge of a span is kept; one just outside the span (after the space) is not
    assert filter_edits("ab cd", "abX cd", [Span(0, 2, 0, 0)], True, "char")[0] == "abX cd"
    assert filter_edits("ab cd", "ab Xcd", [Span(0, 2, 0, 0)], True, "char")[0] == "ab cd"
    # char granularity never changes text outside the spans; block granularity keeps the whole block
    src, hyp = "foo bar baz", "foo barbaz"
    assert filter_edits(src, hyp, [Span(4, 7, 0, 0)], True, "char")[0] == "foo bar baz"
    assert filter_edits(src, hyp, [Span(4, 8, 0, 0)], True, "char")[0] == "foo barbaz"
    assert filter_edits(src, hyp, [Span(4, 7, 0, 0)], True, "block")[0] == "foo bar baz"  # deleted space at 7 is outside [4,7)
    assert filter_edits(src, hyp, [Span(4, 8, 0, 0)], True, "block")[0] == "foo barbaz"
    # a block that straddles the span edge: block keeps it whole, char keeps only the in-span part
    assert filter_edits("ab cd", "aXYd", [Span(0, 2, 0, 0)], True, "block")[0] == "aXYd"
    assert filter_edits("ab cd", "aXYd", [Span(0, 2, 0, 0)], True, "char")[0] in ("aX cd", "aXY cd", "aXYcd")


def test_gold_hypothesis_round_trip():
    """With hyp = ground truth every edit lies inside the oracle spans: oracle-filtered == GT and
    reverse-filtered == OCR. With the scoring-relevance filter the normalised text is still perfect."""
    n = 0
    for rec in _dev_records():
        src, tgt = rec["ocr_hypothesis"]["transcription_unit"], rec["ground_truth"]["transcription_unit"]
        spans = oracle_spans(src, tgt)
        for g in ("block", "char"):
            assert filter_edits(src, tgt, spans, True, g)[0] == tgt
            assert filter_edits(src, tgt, spans, False, g)[0] == src
        spans = oracle_spans(src, tgt, relevance_fn=_scoring_relevant)
        for g in ("block", "char"):
            assert norm(filter_edits(src, tgt, spans, True, g)[0]) == norm(tgt)
        n += 1
    assert n > 500


def test_char_filter_is_a_splice_on_real_outputs():
    """On the Phase 0 zeroshot outputs (if present): the char-level oracle filter equals a splice of
    per-span replacements into the OCR (copy constraint outside the spans, by construction), and the
    reverse char filter plus the oracle char filter together account for every edit op."""
    log = ROOT / "runs" / "20261029T091500Z_zeroshot_dev" / "log.jsonl"
    if not log.is_file():
        return
    gt = {r["document_metadata"]["document_id"]: r["ground_truth"]["transcription_unit"] for r in _dev_records()}
    n = 0
    for line in log.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        src, hyp, tgt = r["ocr"], r["output"], gt[r["document_id"]]
        spans = oracle_spans(src, tgt, relevance_fn=_scoring_relevant)
        out, st = filter_edits(src, hyp, spans, True, "char")
        reps, _ = span_replacements(src, hyp, spans)
        assert out == splice(src, spans, reps)
        _, st_rev = filter_edits(src, hyp, spans, False, "char")
        assert st.n_ops_kept + st_rev.n_ops_kept == st.n_ops
        _, sb = filter_edits(src, hyp, spans, True, "block")
        _, sb_rev = filter_edits(src, hyp, spans, False, "block")
        assert sb.n_blocks_kept + sb_rev.n_blocks_kept == sb.n_blocks
        n += 1
    assert n == 548


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
