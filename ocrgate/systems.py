"""Phase 0 systems. Each takes a reference record and returns (corrected_text, per-document log dict)."""

import asyncio
import re

from hipe_ocrepair_scorer.ocrepair_eval import norm

from .align import Span, oracle_spans, splice
from .edits import span_replacements
from .llm import LLM
from .prompts import CLOSE, OPEN, span_messages, window_messages, zeroshot_messages

FIX_RE = re.compile(r"<fix>(.*?)</fix>", flags=re.DOTALL)


async def nocorrect(rec: dict, llm: LLM | None, cfg: dict) -> tuple[str, dict]:
    return rec["ocr_hypothesis"]["transcription_unit"], {}


async def zeroshot(rec: dict, llm: LLM, cfg: dict) -> tuple[str, dict]:
    ocr = rec["ocr_hypothesis"]["transcription_unit"]
    max_tokens = max(int(cfg["max_tokens_min"]), int(cfg["max_tokens_per_char"] * len(ocr)))
    c = await llm.chat(zeroshot_messages(rec["document_metadata"], ocr), max_tokens=max_tokens)
    out = c.text.strip()
    failure = None
    if c.error:
        failure = "request_error"
    elif c.finish_reason == "length":
        failure = "truncated"
    elif not out:
        failure = "empty"
    fallback = failure is not None and cfg.get("fallback_to_input_on_failure", True)
    log = {
        "raw_output": c.text,
        "finish_reason": c.finish_reason,
        "prompt_tokens": c.prompt_tokens,
        "completion_tokens": c.completion_tokens,
        "latency_s": round(c.latency_s, 3),
        "error": c.error,
        "failure": failure,
        "fallback_to_input": fallback,
    }
    if c.tokens is not None:  # Phase 1: per-token logprobs of the raw completion (before strip/think removal)
        log["tokens"] = c.tokens
        log["logprobs"] = c.logprobs
    return (ocr if fallback else out), log


def _context(src: str, start: int, end: int, n: int) -> tuple[str, str]:
    """Up to n chars of context on each side, cut at whitespace so no partial words are shown."""
    left = src[max(0, start - n):start]
    if start - n > 0:
        m = re.search(r"\s", left)
        left = left[m.end():] if m else ""
    right = src[end:end + n]
    if end + n < len(src):
        m = re.search(r"\s(?=\S*$)", right)
        right = right[:m.start()] if m else ""
    return left, right


def _scoring_relevant(src: str, s: int, e: int, replacement: str) -> bool:
    """True iff replacing src[s:e] by the ground-truth text changes the official normalised text."""
    return norm(src[:s] + replacement + src[e:]) != norm(src)


def gated_spans(src: str, tgt: str, cfg: dict):
    """ORACLE gate: spans of the OCR that differ from the ground truth (see `align.oracle_spans`)."""
    return oracle_spans(
        src,
        tgt,
        expand_to_tokens=cfg["expand_to_tokens"],
        merge_adjacent_tokens=cfg["merge_adjacent_tokens"],
        relevance_fn=_scoring_relevant if cfg["drop_scoring_irrelevant"] else None,
    )


async def oracle_gated(rec: dict, llm: LLM, cfg: dict) -> tuple[str, dict]:
    """ORACLE-gated + copy-constrained correction (upper-bound diagnostic; uses ground truth).

    1. Levenshtein-align OCR vs ground truth -> spans of the OCR that contain errors (the oracle gate).
    2. For each gated span, the model sees the span marked inside its OCR context and proposes a
       replacement. The ground-truth text of the span is never shown to the model.
    3. Output = OCR copied verbatim except gated spans, which are replaced by the proposals.
    """
    src = rec["ocr_hypothesis"]["transcription_unit"]
    tgt = rec["ground_truth"]["transcription_unit"]
    spans = gated_spans(src, tgt, cfg)

    async def propose(span):
        original = src[span.start:span.end]
        left, right = _context(src, span.start, span.end, int(cfg["context_chars"]))
        c = await llm.chat(span_messages(rec["document_metadata"], left, original, right), int(cfg["max_tokens"]))
        m = FIX_RE.search(c.text)
        limit = max(cfg["max_len_ratio"] * len(original), len(original) + cfg["max_len_slack"])
        # The model sometimes copies our span markers into the fix; they are prompt notation, not text.
        fix = m.group(1).replace(OPEN, "").replace(CLOSE, "").strip() if m else ""
        if c.error:
            status, rep = "request_error", original
        elif m is None:
            status, rep = "unparsable", original
        elif len(fix) > limit:
            status, rep = "too_long", original
        else:
            status, rep = "ok", fix
        return rep, {
            "start": span.start,
            "end": span.end,
            "ocr": original,
            "gt": tgt[span.gt_start:span.gt_end],  # diagnostics only, never sent to the model
            "proposal": rep,
            "raw_output": c.text,
            "status": status,
            "finish_reason": c.finish_reason,
            "prompt_tokens": c.prompt_tokens,
            "completion_tokens": c.completion_tokens,
            "latency_s": round(c.latency_s, 3),
            "error": c.error,
        }

    results = await asyncio.gather(*(propose(s) for s in spans))
    replacements = [r for r, _ in results]
    out = splice(src, spans, replacements)
    span_logs = [log for _, log in results]
    statuses = [log["status"] for log in span_logs]
    log = {
        "n_spans": len(spans),
        "gated_chars": sum(s.end - s.start for s in spans),
        "n_ok": statuses.count("ok"),
        "n_echo": sum(1 for log in span_logs if log["status"] == "ok" and log["proposal"] == log["ocr"]),
        "n_fallback_spans": len(statuses) - statuses.count("ok"),
        "prompt_tokens": sum(max(0, s["prompt_tokens"]) for s in span_logs),
        "completion_tokens": sum(max(0, s["completion_tokens"]) for s in span_logs),
        "spans": span_logs,
    }
    return out, log


async def window_gated(rec: dict, llm: LLM, cfg: dict) -> tuple[str, dict]:
    """Phase 0b-2: ORACLE-gated window rewrite + edit filter (upper-bound diagnostic; uses ground truth).

    1. Oracle gate as in `oracle_gated` (Levenshtein OCR vs ground truth).
    2. For each gated span the model rewrites a window of about `window_chars` characters of OCR
       context on each side with the span marked; it returns the whole corrected window.
    3. The rewrite is character-aligned to the window (`edits.span_replacements`, char granularity):
       only edits inside the marked span (insertions also at its edges) are kept; the rest is reverted.
    4. Output = OCR copied verbatim except the gated spans, which are replaced by the filtered text.
    The gold text of a span is never shown to the model (it is logged for diagnosis only).
    """
    src = rec["ocr_hypothesis"]["transcription_unit"]
    tgt = rec["ground_truth"]["transcription_unit"]
    spans = gated_spans(src, tgt, cfg)
    n_ctx = int(cfg["window_chars"])

    async def propose(span):
        original = src[span.start:span.end]
        left, right = _context(src, span.start, span.end, n_ctx)
        window = left + original + right
        wspan = [Span(len(left), len(left) + len(original), 0, 0)]
        c = await llm.chat(window_messages(rec["document_metadata"], left, original, right), int(cfg["max_tokens"]))
        m = FIX_RE.search(c.text)
        fix = m.group(1).replace(OPEN, "").replace(CLOSE, "").strip() if m else ""
        limit = cfg["max_len_ratio"] * len(window) + cfg["max_len_slack"]
        st = None
        if c.error:
            status, rep = "request_error", original
        elif m is None:
            status, rep = ("truncated" if c.finish_reason == "length" else "unparsable"), original
        elif len(fix) > limit:
            status, rep = "too_long", original
        else:
            reps, st = span_replacements(window, fix, wspan)
            status, rep = "ok", reps[0]
        return rep, {
            "start": span.start,
            "end": span.end,
            "ocr": original,
            "gt": tgt[span.gt_start:span.gt_end],  # diagnostics only, never sent to the model
            "window": window,
            "window_out": fix if m else None,
            "proposal": rep,
            "raw_output": c.text,
            "status": status,
            "n_ops": st.n_ops if st else None,
            "n_ops_kept": st.n_ops_kept if st else None,
            "finish_reason": c.finish_reason,
            "prompt_tokens": c.prompt_tokens,
            "completion_tokens": c.completion_tokens,
            "latency_s": round(c.latency_s, 3),
            "error": c.error,
        }

    results = await asyncio.gather(*(propose(s) for s in spans))
    replacements = [r for r, _ in results]
    out = splice(src, spans, replacements)
    span_logs = [log for _, log in results]
    statuses = [log["status"] for log in span_logs]
    log = {
        "n_spans": len(spans),
        "gated_chars": sum(s.end - s.start for s in spans),
        "n_ok": statuses.count("ok"),
        "n_echo": sum(1 for log in span_logs if log["status"] == "ok" and log["proposal"] == log["ocr"]),
        "n_changed_spans": sum(1 for log in span_logs if log["proposal"] != log["ocr"]),
        "n_fallback_spans": len(statuses) - statuses.count("ok"),
        "n_truncated": statuses.count("truncated"),
        "n_ops": sum(log["n_ops"] or 0 for log in span_logs),
        "n_ops_kept": sum(log["n_ops_kept"] or 0 for log in span_logs),
        "prompt_tokens": sum(max(0, s["prompt_tokens"]) for s in span_logs),
        "completion_tokens": sum(max(0, s["completion_tokens"]) for s in span_logs),
        "spans": span_logs,
    }
    return out, log


SYSTEMS = {"nocorrect": nocorrect, "zeroshot": zeroshot, "oracle_gated": oracle_gated, "window_gated": window_gated}
NEEDS_LLM = {"zeroshot", "oracle_gated", "window_gated"}
