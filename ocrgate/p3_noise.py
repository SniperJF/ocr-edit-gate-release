"""Phase 3 (P3-3): noise-matched synthetic OCR pairs.

    python -m ocrgate.p3_noise stats  --config configs/phase2.yaml --split configs/p2_split.json --roles fit --out data_synth/noise_stats.json
    python -m ocrgate.p3_noise apply  --stats data_synth/noise_stats.json --clean data_synth/clean.jsonl --out data_synth/synth.jsonl
        [--per-dataset dta19-l0=200,...] [--seed 1234]

`stats`: per dataset family (icdar2017-en, icdar2017-fr, impresso-snippets-{en,fr,de}, dta19-l0/l1/l2, overproof) a
character-level confusion model estimated from the TRAIN units of the given roles (default: the LoRA fit units only, so
no held-out or dev text is read): OCR <-> ground truth are aligned with Levenshtein editops (rapidfuzz); for every gold
character c we count P(kept | c), P(substituted by x | c), P(deleted | c), and P(an insertion of x follows c | c)
(+ insertions before the first character). Unit lengths and per-unit CERs are summarised to size the synthetic units.
No dataset text is written: only character-pair counts and length/CER summaries.

`apply`: reads clean text units ({"meta": document_metadata-like dict with language/document_type/date, "dataset": family,
"gt": text}) and writes {"meta", "dataset", "ocr", "gt"} pairs where `ocr` is the gold text corrupted by sampling the
family's confusion model character by character (gold -> kept / substituted / deleted, then inserted characters).
A per-unit noise multiplier (log-normal around 1, matching the spread of the family's unit CERs) reproduces the
clean-vs-noisy unit mix of the real data. `apply` never reads benchmark files.
"""

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict

from rapidfuzz.distance import Levenshtein

from .data import ROOT, load_config, split_cells
from .score import select_refs

FAMILY_OF = {"icdar2017_v1.1_train_en": "icdar2017-en", "icdar2017_v1.1_train_fr": "icdar2017-fr",
             "impresso-snippets_v1.0_train_en": "impresso-en", "impresso-snippets_v1.0_train_fr": "impresso-fr",
             "impresso-snippets_v1.0_train_de": "impresso-de", "dta19-l0_v0.1_train-unmatched_de": "dta19-l0",
             "dta19-l1_v0.1_train-unmatched_de": "dta19-l1", "dta19-l2_v0.1_train-unmatched_de": "dta19-l2",
             "dta19-l0_v0.1_train_de": "dta19-l0", "dta19-l1_v0.1_train_de": "dta19-l1", "dta19-l2_v0.1_train_de": "dta19-l2",
             "overproof-combined_v1.0_train_en": "overproof"}


def unit_cer(gt: str, ocr: str) -> float:
    return Levenshtein.distance(gt, ocr) / max(1, len(gt))


def count_unit(gt: str, ocr: str, st: dict) -> None:
    """Accumulate confusion counts of one unit into st (keys: keep, sub, del, ins)."""
    ops = Levenshtein.editops(gt, ocr)  # src = gt, dest = ocr
    sub = {}; dele = set(); ins = defaultdict(str)  # ins[i] = chars inserted BEFORE gt position i
    for op in ops:
        if op.tag == "replace":
            sub[op.src_pos] = ocr[op.dest_pos]
        elif op.tag == "delete":
            dele.add(op.src_pos)
        else:
            ins[op.src_pos] += ocr[op.dest_pos]
    for i, c in enumerate(gt):
        st["n"][c] += 1
        if i in sub:
            st["sub"][c][sub[i]] += 1
        elif i in dele:
            st["del"][c] += 1
        else:
            st["keep"][c] += 1
    # insertions: attribute to the preceding gold char (position i -> after gt[i-1]; i == 0 -> "^")
    for i, s in ins.items():
        prev = gt[i - 1] if i > 0 else "^"
        st["ins"][prev][s] += 1
        st["ins_n"][prev] += 1


def cmd_stats(args) -> None:
    cfg = load_config(ROOT / args.config)
    allowed = None
    if args.split:
        sp = json.loads((ROOT / args.split).read_text(encoding="utf-8"))
        roles = set(args.roles.split(","))
        allowed = {(u["cell"], u["unit_idx"]) for u in sp["units"] if u["role"] in roles}
    fam_st = {}
    for cell in split_cells(cfg, "train"):
        fam = FAMILY_OF.get(cell.label)
        if fam is None:
            continue
        st = fam_st.setdefault(fam, {"n": Counter(), "keep": Counter(), "sub": defaultdict(Counter), "del": Counter(),
                                     "ins": defaultdict(Counter), "ins_n": Counter(), "lens": [], "cers": [], "lines": [], "units": 0})
        for ui, ref in enumerate(select_refs(cell)):
            if allowed is not None and (cell.label, ui) not in allowed:
                continue
            gt = ref["ground_truth"]["transcription_unit"]; ocr = ref["ocr_hypothesis"]["transcription_unit"]
            if not gt.strip():
                continue
            count_unit(gt, ocr, st)
            st["lens"].append(len(gt)); st["cers"].append(unit_cer(gt, ocr)); st["units"] += 1
            st["lines"].append(gt.count("\n") + 1)
    out = {"source": "TRAIN units of roles %s of %s" % (args.roles, args.split), "families": {}}
    for fam, st in fam_st.items():
        n_tot = sum(st["n"].values())
        n_sub = sum(sum(v.values()) for v in st["sub"].values()); n_del = sum(st["del"].values())
        n_ins = sum(len(k) * v for d in st["ins"].values() for k, v in d.items())
        cers = st["cers"]
        out["families"][fam] = {
            "units": st["units"], "gold_chars": n_tot, "cer_micro": (n_sub + n_del + n_ins) / max(1, n_tot),
            "p_sub": n_sub / max(1, n_tot), "p_del": n_del / max(1, n_tot), "p_ins_chars": n_ins / max(1, n_tot),
            "unit_cer_mean": statistics.mean(cers), "unit_cer_median": statistics.median(cers),
            "unit_cer_q10": sorted(cers)[int(0.1 * len(cers))], "unit_cer_q90": sorted(cers)[int(0.9 * len(cers))],
            "unit_cer_zero_frac": sum(c == 0 for c in cers) / len(cers),
            "unit_len_mean": statistics.mean(st["lens"]), "unit_len_median": statistics.median(st["lens"]),
            "unit_len_q10": sorted(st["lens"])[int(0.1 * len(st["lens"]))], "unit_len_q90": sorted(st["lens"])[int(0.9 * len(st["lens"]))],
            "chars_per_line_median": statistics.median([l / max(1, n) for l, n in zip(st["lens"], st["lines"])]),
            "n": dict(st["n"]), "keep": dict(st["keep"]), "del": dict(st["del"]),
            "sub": {c: dict(v) for c, v in st["sub"].items()}, "ins": {c: dict(v) for c, v in st["ins"].items()}, "ins_n": dict(st["ins_n"]),
        }
        f = out["families"][fam]
        print(f"{fam:14s} units {f['units']:5d} gold chars {n_tot:8d} CER {f['cer_micro']:.4f} (sub {f['p_sub']:.4f} del {f['p_del']:.4f} ins {f['p_ins_chars']:.4f}) "
              f"unit CER med {f['unit_cer_median']:.4f} q90 {f['unit_cer_q90']:.4f} zero {f['unit_cer_zero_frac']:.2f} | len med {f['unit_len_median']:.0f} q10-q90 {f['unit_len_q10']}-{f['unit_len_q90']} chars/line {f['chars_per_line_median']:.0f}")
    (ROOT / args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


class Noiser:
    """Samples OCR-like corruptions of a gold string from a family's confusion counts.

    For unseen characters a back-off model (pooled over all characters of the family) is used. `mult` scales the error
    probabilities of one unit (keep probability is renormalised); a smoothing count `alpha` is added to "keep"."""

    def __init__(self, f: dict, alpha: float = 0.5):
        self.f = f
        self.n = f["n"]; self.keep = f["keep"]; self.dele = f["del"]; self.sub = f["sub"]; self.ins = f["ins"]; self.ins_n = f["ins_n"]
        self.alpha = alpha
        # pooled back-off
        n_tot = sum(self.n.values())
        self.bo_del = sum(self.dele.values()) / n_tot
        self.bo_sub = sum(sum(v.values()) for v in self.sub.values()) / n_tot
        self.bo_ins = sum(self.ins_n.values()) / n_tot
        pooled_ins = Counter()
        for d in self.ins.values():
            pooled_ins.update(d)
        self.bo_ins_choices = list(pooled_ins.items())

    def corrupt(self, gt: str, rng: random.Random, mult: float) -> str:
        out = []
        # insertions before the first char
        out.append(self._insert("^", rng, mult))
        for c in gt:
            n = self.n.get(c, 0)
            if n > 0:
                p_del = self.dele.get(c, 0) / (n + self.alpha); subs = self.sub.get(c, {}); p_sub = sum(subs.values()) / (n + self.alpha)
            else:
                p_del, p_sub, subs = self.bo_del, self.bo_sub, {}
            p_del *= mult; p_sub *= mult
            tot = p_del + p_sub
            if tot > 0.95:
                p_del *= 0.95 / tot; p_sub *= 0.95 / tot
            r = rng.random()
            if r < p_del:
                pass
            elif r < p_del + p_sub and subs:
                ks, ws = zip(*subs.items()); out.append(rng.choices(ks, ws)[0])
            else:
                out.append(c)
            out.append(self._insert(c, rng, mult))
        return "".join(out)

    def _insert(self, prev: str, rng: random.Random, mult: float) -> str:
        n = self.n.get(prev, 0) if prev != "^" else None
        if prev == "^":
            p = self.ins_n.get("^", 0) / max(1, self.f["units"]) * mult; d = self.ins.get("^", {})
        elif n:
            p = self.ins_n.get(prev, 0) / (n + self.alpha) * mult; d = self.ins.get(prev, {})
        else:
            p, d = self.bo_ins * mult, {}
        if rng.random() >= min(p, 0.95):
            return ""
        if not d:
            if not self.bo_ins_choices:
                return ""
            ks, ws = zip(*self.bo_ins_choices)
        else:
            ks, ws = zip(*d.items())
        return rng.choices(ks, ws)[0]


def cmd_apply(args) -> None:
    stats = json.loads((ROOT / args.stats).read_text(encoding="utf-8"))["families"]
    rng = random.Random(args.seed)
    noisers = {fam: Noiser(f) for fam, f in stats.items()}
    rows = [json.loads(l) for l in open(ROOT / args.clean, encoding="utf-8") if l.strip()]
    n_out = 0; cers = defaultdict(list)
    with open(ROOT / args.out, "w", encoding="utf-8") as f:
        for ex in rows:
            fam = ex["dataset"]; st = stats[fam]
            # per-unit multiplier: log-normal with sigma matching the family's unit CER spread (clean units kept clean with the
            # family's zero-CER share)
            if rng.random() < st["unit_cer_zero_frac"] * args.zero_scale:
                mult = 0.0
            else:
                sigma = max(0.3, min(1.2, math.log(max(1.5, st["unit_cer_q90"] / max(1e-4, st["unit_cer_median"]))) / 1.28))
                mult = math.exp(rng.gauss(0.0, sigma))
            ocr = noisers[fam].corrupt(ex["gt"], rng, mult)
            cers[fam].append(unit_cer(ex["gt"], ocr))
            f.write(json.dumps({"meta": ex["meta"], "dataset": fam, "source": ex.get("source"), "ocr": ocr, "gt": ex["gt"]}, ensure_ascii=False) + "\n")
            n_out += 1
    for fam, c in sorted(cers.items()):
        print(f"{fam:14s} synth units {len(c):5d} unit CER mean {statistics.mean(c):.4f} median {statistics.median(c):.4f} zero {sum(x == 0 for x in c) / len(c):.2f} "
              f"| real: mean {stats[fam]['unit_cer_mean']:.4f} median {stats[fam]['unit_cer_median']:.4f} zero {stats[fam]['unit_cer_zero_frac']:.2f}")
    print(f"wrote {n_out} synthetic pairs -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("stats"); a.add_argument("--config", default="configs/phase2.yaml"); a.add_argument("--split", default="configs/p2_split.json")
    a.add_argument("--roles", default="fit"); a.add_argument("--out", default="data_synth/noise_stats.json")
    b = sub.add_parser("apply"); b.add_argument("--stats", default="data_synth/noise_stats.json"); b.add_argument("--clean", required=True)
    b.add_argument("--out", required=True); b.add_argument("--seed", type=int, default=1234); b.add_argument("--zero-scale", type=float, default=1.0)
    args = ap.parse_args()
    cmd_stats(args) if args.cmd == "stats" else cmd_apply(args)


if __name__ == "__main__":
    main()
