"""Offline checks of the oracle gate and the LLM code paths (no GPU / endpoint needed).

    python -m pytest tests/        or        python tests/test_oracle.py
"""

import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hipe_ocrepair_scorer.ocrepair_eval import norm  # noqa: E402

from ocrgate.align import oracle_spans, splice  # noqa: E402
from ocrgate.data import load_config, read_jsonl, split_cells, ROOT  # noqa: E402
from ocrgate.llm import Completion  # noqa: E402
from ocrgate.prompts import CLOSE, OPEN  # noqa: E402
from ocrgate.systems import _scoring_relevant, oracle_gated, zeroshot  # noqa: E402

CFG = load_config(ROOT / "configs/phase0.yaml")


def _dev_records():
    for cell in split_cells(CFG, "dev"):
        yield from read_jsonl(cell.path)


def test_toy_alignment():
    src, tgt = "The prioe one penny, is 'pub luhed every day", "The price one penny, is published every day"
    spans = oracle_spans(src, tgt)
    assert [src[s.start:s.end] for s in spans] == ["prioe", "'pub luhed"]
    assert splice(src, spans, [tgt[s.gt_start:s.gt_end] for s in spans]) == tgt


def test_gt_splice_reconstructs_ground_truth():
    """Copy-outside-gate + gold replacements must give back the ground truth exactly."""
    for rec in _dev_records():
        src, tgt = rec["ocr_hypothesis"]["transcription_unit"], rec["ground_truth"]["transcription_unit"]
        spans = oracle_spans(src, tgt)
        assert splice(src, spans, [tgt[s.gt_start:s.gt_end] for s in spans]) == tgt
        # with the scoring-relevance filter, the normalised text is still perfect
        spans = oracle_spans(src, tgt, relevance_fn=_scoring_relevant)
        assert norm(splice(src, spans, [tgt[s.gt_start:s.gt_end] for s in spans])) == norm(tgt)


class FakeLLM:
    """Stands in for ocrgate.llm.LLM. mode='echo' returns the input unchanged; mode='gold' answers spans with GT."""

    def __init__(self, mode, gold=None):
        self.mode, self.gold, self.calls = mode, gold, 0

    async def chat(self, messages, max_tokens):
        self.calls += 1
        user = messages[-1]["content"]
        m = re.search(re.escape(OPEN) + r"(.*?)" + re.escape(CLOSE), user.split("Excerpt:\n", 1)[-1], flags=re.DOTALL)
        if "Excerpt:\n" in user:  # span prompt
            text = self.gold.pop(0) if self.mode == "gold" else m.group(1)
            return Completion(f"<fix>{text}</fix>", "stop", 1, 1, 0.0)
        return Completion(user.split("Text:\n", 1)[1], "stop", 1, 1, 0.0)


def test_systems_with_fake_llm():
    recs = list(_dev_records())[:40]
    ocfg = CFG["systems"]["oracle_gated"]
    for rec in recs:
        src, tgt = rec["ocr_hypothesis"]["transcription_unit"], rec["ground_truth"]["transcription_unit"]
        out, _ = asyncio.run(zeroshot(rec, FakeLLM("echo"), CFG["systems"]["zeroshot"]))
        assert out == src.strip()
        out, log = asyncio.run(oracle_gated(rec, FakeLLM("echo"), ocfg))
        assert out == src and log["n_fallback_spans"] == 0
        spans = oracle_spans(src, tgt, relevance_fn=_scoring_relevant)
        gold = [tgt[s.gt_start:s.gt_end].strip() for s in spans]
        out, _ = asyncio.run(oracle_gated(rec, FakeLLM("gold", gold), ocfg))
        assert norm(out) == norm(tgt)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
