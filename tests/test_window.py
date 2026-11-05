"""Offline checks of the Phase 0b window_gated system with a fake LLM (no GPU / endpoint needed)."""

import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hipe_ocrepair_scorer.ocrepair_eval import norm  # noqa: E402

from ocrgate.data import load_config, read_jsonl, split_cells, ROOT  # noqa: E402
from ocrgate.llm import Completion  # noqa: E402
from ocrgate.prompts import CLOSE, OPEN  # noqa: E402
from ocrgate.systems import gated_spans, window_gated  # noqa: E402

CFG = load_config(ROOT / "configs/phase0b.yaml")
WCFG = CFG["systems"]["window_gated"]
MARK = re.compile(re.escape(OPEN) + r"(.*?)" + re.escape(CLOSE), flags=re.DOTALL)


class FakeWindowLLM:
    """mode='echo': returns the window unchanged (markers removed). mode='gold': replaces the marked span by
    the gold text. mode='leak': gold span plus a corruption of the context outside the span (must be reverted).
    mode='think': like gold but wrapped in a <think> block, as with Qwen3 thinking on."""

    def __init__(self, mode, gold=None):
        self.mode, self.gold, self.calls = mode, gold, 0

    async def chat(self, messages, max_tokens):
        self.calls += 1
        excerpt = messages[-1]["content"].split("Excerpt:\n", 1)[1]
        m = MARK.search(excerpt)
        left, span, right = excerpt[: m.start()], m.group(1), excerpt[m.end():]
        if self.mode == "echo":
            text = left + span + right
        else:
            g = self.gold.pop(0)
            if self.mode == "leak":  # one wrong character on each side of the span (outside it)
                l2 = ("Q" + left[1:]) if left else left
                r2 = (right[:-1] + "Q") if right else right
                text = l2 + g + r2
            else:
                text = left + g + right
        if self.mode == "think":
            return Completion(f"<think>\nhmm\n</think>\n\n<fix>{text}</fix>", "stop", 1, 1, 0.0)
        return Completion(f"<fix>{text}</fix>", "stop", 1, 1, 0.0)


def test_window_gated_fake():
    n = 0
    for cell in split_cells(CFG, "dev"):
        for rec in read_jsonl(cell.path)[:12]:
            src, tgt = rec["ocr_hypothesis"]["transcription_unit"], rec["ground_truth"]["transcription_unit"]
            spans = gated_spans(src, tgt, WCFG)
            out, log = asyncio.run(window_gated(rec, FakeWindowLLM("echo"), WCFG))
            assert out == src and log["n_fallback_spans"] == 0 and log["n_echo"] == len(spans)
            gold = [tgt[s.gt_start:s.gt_end] for s in spans]
            out, log = asyncio.run(window_gated(rec, FakeWindowLLM("gold", list(gold)), WCFG))
            assert norm(out) == norm(tgt), (rec["document_metadata"]["document_id"])
            # the fake only touches the marked span, so every edit op must be kept
            assert log["n_ops"] == log["n_ops_kept"]
            out, log = asyncio.run(window_gated(rec, FakeWindowLLM("think", list(gold)), WCFG))
            assert norm(out) == norm(tgt)
            # context corruption outside the span is reverted by the filter
            out, log = asyncio.run(window_gated(rec, FakeWindowLLM("leak", list(gold)), WCFG))
            assert norm(out) == norm(tgt), (rec["document_metadata"]["document_id"], out)
            n += 1
    assert n > 80


if __name__ == "__main__":
    test_window_gated_fake()
    print("ok")
