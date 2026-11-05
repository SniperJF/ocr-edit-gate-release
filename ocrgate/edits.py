"""Character-level edit filtering (Phase 0b).

Given an OCR string `src`, a model rewrite `hyp` and a list of gated spans of `src`, keep only the
edits of `hyp` that touch a gated span and revert every other edit to the OCR text (or the reverse,
for the sanity check). Edits come from one optimal Levenshtein character alignment of `src` and
`hyp` (`rapidfuzz.distance.Levenshtein.editops`).

Two granularities:
- "block": an edit is a maximal run of adjacent non-equal alignment ops (one non-`equal` opcode). A
  block is kept as a whole if it overlaps a span (an insertion-only block if it touches a span).
  A kept block may extend beyond the span it overlaps.
- "char": each single-character op is kept iff its OCR position lies inside a span (an insertion
  if it lies inside or at either edge of a span). Kept edits then never change text outside the
  spans, which is the copy constraint of the edit-gated design (`span_replacements` + `splice`).
"""

from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

from .align import Span, splice


@dataclass
class FilterStats:
    n_blocks: int = 0  # edit blocks (non-equal opcodes) in hyp vs src
    n_blocks_kept: int = 0  # blocks with at least one kept op
    n_ops: int = 0  # single-character edit ops
    n_ops_kept: int = 0


def _touches(spans: list[Span], s: int, e: int) -> bool:
    """Does the src interval [s, e) overlap a span? A zero-width interval (insertion) touches a span if
    it lies inside it or at one of its edges."""
    for sp in spans:
        if e > s:
            if s < sp.end and e > sp.start:
                return True
        elif sp.start <= s <= sp.end:
            return True
    return False


def _blocks(ops) -> list[tuple[str, int, int, int, int]]:
    """Opcodes with adjacent non-equal opcodes merged into one 'edit' block (tag, s, e, ds, de).

    rapidfuzz's `as_opcodes` keeps e.g. a replace and the delete right after it as two opcodes; an
    edit block here is the maximal run of adjacent single-character edits.
    """
    blocks: list[list] = []
    for op in ops.as_opcodes():
        tag = "equal" if op.tag == "equal" else "edit"
        if blocks and tag == "edit" and blocks[-1][0] == "edit" and blocks[-1][2] == op.src_start:
            blocks[-1][2], blocks[-1][4] = op.src_end, op.dest_end
        else:
            blocks.append([tag, op.src_start, op.src_end, op.dest_start, op.dest_end])
    return [tuple(b) for b in blocks]


def _char_ops(src: str, hyp: str, spans: list[Span], keep_inside: bool):
    """Kept single-character ops: inserts[src_pos] = [(dest_pos, char)], subst[src_pos] = char | None."""
    ops = Levenshtein.editops(src, hyp)
    stats = FilterStats(n_ops=len(ops))
    inserts: dict[int, list[tuple[int, str]]] = {}
    subst: dict[int, str | None] = {}
    for op in ops:
        i, j = op.src_pos, op.dest_pos
        touch = _touches(spans, i, i) if op.tag == "insert" else _touches(spans, i, i + 1)
        if touch != keep_inside:
            continue
        stats.n_ops_kept += 1
        if op.tag == "insert":
            inserts.setdefault(i, []).append((j, hyp[j]))
        elif op.tag == "replace":
            subst[i] = hyp[j]
        else:
            subst[i] = None
    for tag, s, e, _, _ in _blocks(ops):
        if tag == "edit":
            stats.n_blocks += 1
            if any(k in subst for k in range(s, e)) or any(k in inserts for k in range(s, e + 1)):
                stats.n_blocks_kept += 1
    return inserts, subst, stats


def _apply(src: str, inserts: dict, subst: dict, lo: int, hi: int) -> str:
    """Apply the kept ops to src[lo:hi] (inserts at positions lo..hi inclusive; rapidfuzz convention:
    an insert at src_pos i goes before src[i])."""
    out = []
    for i in range(lo, hi + 1):
        for _, ch in sorted(inserts.get(i, [])):
            out.append(ch)
        if i < hi:
            if i in subst:
                if subst[i] is not None:
                    out.append(subst[i])
            else:
                out.append(src[i])
    return "".join(out)


def span_replacements(src: str, hyp: str, spans: list[Span]) -> tuple[list[str], FilterStats]:
    """Char-granularity oracle filter as per-span replacement strings (for `align.splice`).

    Spans are non-overlapping and separated by at least one character, so an insertion at a span
    edge belongs to exactly one span.
    """
    inserts, subst, stats = _char_ops(src, hyp, spans, True)
    return [_apply(src, inserts, subst, sp.start, sp.end) for sp in spans], stats


def filter_edits(
    src: str, hyp: str, spans: list[Span], keep_inside: bool = True, granularity: str = "block"
) -> tuple[str, FilterStats]:
    """Return `src` with only the selected edits of `hyp` applied.

    keep_inside=True keeps the edits that touch a span and reverts the rest (oracle-filtered);
    keep_inside=False reverts the edits that touch a span and keeps the rest (reverse filter).
    """
    if granularity == "char":
        if keep_inside:
            reps, stats = span_replacements(src, hyp, spans)
            return splice(src, spans, reps), stats
        inserts, subst, stats = _char_ops(src, hyp, spans, False)
        return _apply(src, inserts, subst, 0, len(src)), stats
    if granularity != "block":
        raise ValueError(granularity)
    ops = Levenshtein.editops(src, hyp)
    stats = FilterStats(n_ops=len(ops))
    out = []
    for tag, s, e, ds, de in _blocks(ops):
        if tag == "equal":
            out.append(src[s:e])
            continue
        stats.n_blocks += 1
        if _touches(spans, s, e) == keep_inside:
            stats.n_blocks_kept += 1
            stats.n_ops_kept += max(e - s, de - ds)
            out.append(hyp[ds:de])
        else:
            out.append(src[s:e])
    return "".join(out), stats


# ---------------------------------------------------------------------------------------------
# Phase 1: edit blocks as units of an edit-veto gate


def edit_blocks(src: str, hyp: str) -> list[tuple[int, int, int, int]]:
    """Maximal runs of adjacent single-character edits between `src` (OCR) and `hyp` (rewrite):
    (s, e, ds, de) with src[s:e] replaced by hyp[ds:de]. Same block rule as `filter_edits(granularity="block")`."""
    return [(s, e, ds, de) for tag, s, e, ds, de in _blocks(Levenshtein.editops(src, hyp)) if tag == "edit"]


def apply_blocks(src: str, hyp: str, blocks: list[tuple[int, int, int, int]], keep: list[bool]) -> str:
    """`src` with the edit blocks whose `keep` flag is true applied (taken from `hyp`); all other text is
    copied from `src`. keep=all True reproduces `hyp`, keep=all False reproduces `src`."""
    out, pos = [], 0
    for (s, e, ds, de), k in zip(blocks, keep, strict=True):
        out.append(src[pos:s])
        out.append(hyp[ds:de] if k else src[s:e])
        pos = e
    out.append(src[pos:])
    return "".join(out)
