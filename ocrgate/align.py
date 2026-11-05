"""Character-level Levenshtein alignment and ORACLE edit-gate spans.

Given an OCR string `src` and its ground truth `tgt`, `oracle_spans` returns the regions of `src`
that contain OCR errors according to one optimal Levenshtein alignment. Everything outside these
spans is identical between OCR and ground truth, so copying it verbatim is always correct.

This uses the ground truth and is therefore an ORACLE (upper-bound diagnostic), not a system.
"""

from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein


@dataclass
class Span:
    start: int  # [start, end) in the OCR string
    end: int
    gt_start: int  # aligned [gt_start, gt_end) in the ground truth (for diagnostics only)
    gt_end: int


def _boundary_maps(ops, n_src: int) -> tuple[list, list]:
    """For every src boundary p in 0..n_src, the min/max aligned target boundary.

    Boundaries strictly inside a replace/delete op have no unique image and stay None.
    """
    lo: list = [None] * (n_src + 1)
    hi: list = [None] * (n_src + 1)

    def put(p, d):
        lo[p] = d if lo[p] is None else min(lo[p], d)
        hi[p] = d if hi[p] is None else max(hi[p], d)

    for op in ops:
        s, e, ds, de = op.src_start, op.src_end, op.dest_start, op.dest_end
        if op.tag == "equal":
            for p in range(s, e + 1):
                put(p, ds + (p - s))
        else:  # replace / delete / insert: only the op edges are well-defined boundaries
            put(s, ds)
            put(e, de)
    return lo, hi


def _expand_to_tokens(src: str, s: int, e: int) -> tuple[int, int]:
    """Grow [s, e) to whitespace-delimited token boundaries; never return an empty span."""
    n = len(src)
    while s > 0 and not src[s - 1].isspace():
        s -= 1
    while e < n and not src[e].isspace():
        e += 1
    if s == e:  # zero-width insertion inside a whitespace run: attach to the previous (else next) token
        if s > 0:
            while s > 0 and src[s - 1].isspace():
                s -= 1
            e = s
            while s > 0 and not src[s - 1].isspace():
                s -= 1
        else:
            while e < n and src[e].isspace():
                e += 1
            while e < n and not src[e].isspace():
                e += 1
    return s, e


def oracle_spans(
    src: str,
    tgt: str,
    expand_to_tokens: bool = True,
    merge_adjacent_tokens: bool = True,
    relevance_fn=None,
) -> list[Span]:
    """Gated spans of `src` (OCR) that differ from `tgt` (ground truth).

    relevance_fn(src, span_start, span_end, replacement) -> bool, if given, drops spans whose
    ground-truth replacement does not change the scored (normalised) text.
    """
    ops = Levenshtein.opcodes(src, tgt)
    lo, hi = _boundary_maps(ops, len(src))

    raw = []
    for op in ops:
        if op.tag == "equal":
            continue
        s, e = op.src_start, op.src_end
        if expand_to_tokens:
            s, e = _expand_to_tokens(src, s, e)
        elif s == e:  # pure insertion: gate one neighbouring character so the span is editable
            s, e = (s - 1, e) if s > 0 else (s, e + 1)
        raw.append([max(0, s), min(len(src), e)])
    if not raw:
        return []

    raw.sort()
    merged = [raw[0]]
    for s, e in raw[1:]:
        last = merged[-1]
        gap = src[last[1]:s]
        if s <= last[1] or (merge_adjacent_tokens and gap.strip() == ""):
            last[1] = max(last[1], e)
        else:
            merged.append([s, e])

    spans = []
    for s, e in merged:
        if lo[s] is None or hi[e] is None:  # cannot happen for spans grown from op edges; be explicit
            raise AssertionError(f"span boundary inside an edit op: {(s, e)}")
        span = Span(s, e, lo[s], hi[e])
        if relevance_fn is not None and not relevance_fn(src, s, e, tgt[span.gt_start:span.gt_end]):
            continue
        spans.append(span)
    return spans


def splice(src: str, spans: list[Span], replacements: list[str]) -> str:
    """Copy `src` verbatim except for the gated spans, which are replaced."""
    out, pos = [], 0
    for span, rep in zip(spans, replacements, strict=True):
        out.append(src[pos:span.start])
        out.append(rep)
        pos = span.end
    out.append(src[pos:])
    return "".join(out)
