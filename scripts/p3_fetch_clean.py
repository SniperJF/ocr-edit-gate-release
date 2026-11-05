#!/usr/bin/env python3
"""P3-3: fetch clean historical texts with permissive licences and cut them into benchmark-like units.

    python scripts/p3_fetch_clean.py fetch     # downloads to data_synth/raw/ (cached; polite 1 req/s)
    python scripts/p3_fetch_clean.py units     # data_synth/clean.jsonl: {"meta", "dataset", "source", "gt"} per unit

Sources (all documented with licence in results/p3_3_synth.md):
  en: Project Gutenberg 19th-century periodicals (The Mirror of Literature 1827-29, Notes and Queries 1849-50, Chambers's
      Edinburgh Journal, Punch 1841-1890s) - public domain in the US; PG header/footer stripped.
  fr: Project Gutenberg: Le Tour du Monde (1860), Journal des Goncourt, Hugo Actes et Paroles, Guizot Memoires - public domain.
  de: Wikisource "Die Gartenlaube" (1853-1899 family periodical; transcriptions CC BY-SA 3.0 / texts public domain) via the
      MediaWiki API; Gutenberg "Anzeiger fuer Kunde der deutschen Vorzeit" (1866-1880); and the ground truth of the
      impresso-nzz TRAIN units of role `fit` (CC BY-NC 4.0, NZZ black-letter ground truth, already part of the train split
      but excluded from LoRA training by the Phase 1 recipe) - no held-out/dev/test text is used.
Units: per target family the gold length is sampled between the family's q10 and q90 unit lengths (results/p3_3_noise_stats.json);
icdar2017 units are single-line (as in the benchmark); the others are re-wrapped to the family's median line width with
end-of-line hyphenation (U+00AC in impresso/overproof GT) at the observed rate; dta19 clean units are shared by l0/l1/l2
(matched design, as in dta19 train).
"""
import csv
import html
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data_synth" / "raw"; RAW.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "ocr-edit-gate-research/1.0 (academic OCR post-correction research)"}

GUTENBERG = {  # id: (lang, source label, year)
    # en periodicals
    **{i: ("en", "Mirror of Literature", 1827) for i in [9882, 9884, 9918, 9919, 9922, 10026, 10074]},
    **{i: ("en", "Mirror of Literature", 1828) for i in [8640]},
    **{i: ("en", "Mirror of Literature", 1829) for i in [3246]},
    **{i: ("en", "Notes and Queries", 1849) for i in [8603]},
}
GUT_TITLE_PATTERNS = [  # (lang, regex on title, max issues, label, year regex)
    ("en", r"^The Mirror of Literature, Amusement, and Instruction", 40, "Mirror of Literature"),
    ("en", r"^Notes and Queries, Number", 25, "Notes and Queries"),
    ("en", r"^Chambers's Edinburgh Journal", 25, "Chambers's Edinburgh Journal"),
    ("en", r"^Punch, or the London Charivari, Volume (?:[1-9]|[1-9][0-9]|10[0-9]|11[0-7]),", 25, "Punch"),
    ("fr", r"^Le Tour du Monde;", 20, "Le Tour du Monde"),
    ("fr", r"^Journal des Goncourt", 9, "Journal des Goncourt"),
    ("fr", r"^Actes et Paroles", 4, "Hugo, Actes et Paroles"),
    ("fr", r"^Mémoires pour servir à l'Histoire de mon temps", 8, "Guizot, Memoires"),
    ("fr", r"^Portraits littéraires", 3, "Sainte-Beuve, Portraits litteraires"),
    ("de", r"^Anzeiger für Kunde der deutschen Vorzeit", 4, "Anzeiger fuer Kunde der deutschen Vorzeit"),
]
GARTENLAUBE = [(y, h) for y in (1855, 1860, 1865, 1870, 1875, 1880, 1885, 1890) for h in range(1, 9)]


def get(url: str, dest: Path, sleep: float = 1.0) -> str | None:
    if dest.is_file():
        return dest.read_text(encoding="utf-8", errors="replace")
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        txt = data.decode("utf-8", errors="replace")
        dest.write_text(txt, encoding="utf-8")
        time.sleep(sleep)
        return txt
    except Exception as e:
        print(f"FAILED {url}: {e}", file=sys.stderr)
        time.sleep(sleep)
        return None


def catalog_ids() -> list[tuple[int, str, str, str]]:
    rows = list(csv.DictReader(open("/tmp/pg_catalog.csv", encoding="utf-8")))
    out = []
    for lang, pat, maxn, label in GUT_TITLE_PATTERNS:
        hits = [r for r in rows if r["Language"] == lang and r["Type"] == "Text" and re.search(pat, r["Title"])]
        for r in hits[:maxn]:
            m = re.search(r"\b(1[6-9]\d\d)\b", r["Title"])
            out.append((int(r["Text#"]), lang, label, m.group(1) if m else "n/a"))
    return out


def strip_pg(txt: str) -> str:
    s = re.search(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", txt, flags=re.I)
    e = re.search(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", txt, flags=re.I)
    body = txt[s.end() if s else 0: e.start() if e else len(txt)]
    return body.replace("\r\n", "\n")


def html_to_text(h: str) -> str:
    h = re.sub(r"(?s)<(script|style|table|sup|span class=\"pagenum\"[^>]*).*?</\1>", " ", h)
    h = re.sub(r"(?s)<div class=\"(?:noprint|printfooter|navbox|toc|mw-references-wrap)[^>]*>.*?</div>", " ", h)
    h = re.sub(r"<br\s*/?>", "\n", h); h = re.sub(r"</(p|div|h\d|li|tr)>", "\n", h)
    h = re.sub(r"(?s)<[^>]+>", "", h)
    return html.unescape(h)


def fetch() -> None:
    for gid, lang, label, year in catalog_ids():
        dest = RAW / f"pg{gid}.txt"
        txt = get(f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt", dest)
        meta = RAW / f"pg{gid}.json"
        if txt is not None and not meta.is_file():
            meta.write_text(json.dumps({"id": gid, "lang": lang, "source": f"Project Gutenberg #{gid}: {label}", "year": year,
                                        "licence": "public domain (Project Gutenberg License: PG header/footer removed)"}), encoding="utf-8")
    for y, h in GARTENLAUBE:
        page = f"Die Gartenlaube ({y})/Heft {h}"
        dest = RAW / f"ws_gartenlaube_{y}_{h}.html"
        url = "https://de.wikisource.org/w/api.php?" + urllib.parse.urlencode({"action": "parse", "page": page, "prop": "text", "format": "json", "disabletoc": 1})
        js = get(url, dest)
        if js is None:
            continue
        meta = RAW / f"ws_gartenlaube_{y}_{h}.json"
        if not meta.is_file():
            meta.write_text(json.dumps({"page": page, "lang": "de", "source": f"de.wikisource.org: {page}", "year": str(y),
                                        "licence": "text public domain; Wikisource transcription CC BY-SA 3.0 (attribution: Wikisource contributors)"}), encoding="utf-8")
    print("fetch done")


def clean_text(txt: str, lang: str) -> list[str]:
    """Paragraphs of running text (>= 200 chars, mostly letters), with in-paragraph line breaks joined."""
    txt = txt.replace("\u00a0", " ").replace("\u00ac\n", "").replace("\u00ac ", "")  # undo source line-end hyphenation (nzz GT)
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", txt)]
    out = []
    for p in paras:
        if len(p) < 200:
            continue
        letters = sum(c.isalpha() for c in p)
        if letters / len(p) < 0.7:
            continue
        if re.search(r"(?i)project gutenberg|wikisource|transcriber|illustration|\[pg|\{\{|\}\}|https?://|^\s*[\d\.\s]+$|Seite:|Textdaten|BEARBEITUNGSSTAND", p):
            continue
        if sum(c.isupper() for c in p) / max(1, letters) > 0.4:
            continue
        out.append(p)
    return out


def wrap(text: str, width: int, rng: random.Random, hyph_rate: float, hyph_char: str) -> str:
    """Greedy re-wrap at whitespace to about `width` chars per line; sometimes split a word at the line end with hyph_char."""
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        if not cur:
            cur = w; continue
        if len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            room = width - len(cur) - 1
            if hyph_rate > 0 and room >= 3 and len(w) >= 7 and rng.random() < hyph_rate:
                k = rng.randint(2, min(len(w) - 3, room - 1))
                lines.append(cur + " " + w[:k] + hyph_char); cur = w[k:]
            else:
                lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def units() -> None:
    stats = json.loads((ROOT / "results" / "p3_3_noise_stats.json").read_text(encoding="utf-8"))["families"]
    rng = random.Random(1234)
    pools = {"en": [], "fr": [], "de": [], "de-news": []}  # (paragraph text, source dict); de-news = nzz GT (impresso-de only)
    for meta in sorted(RAW.glob("*.json")):
        m = json.loads(meta.read_text(encoding="utf-8"))
        if meta.name.startswith("pg"):
            txt = strip_pg((RAW / (meta.stem + ".txt")).read_text(encoding="utf-8", errors="replace"))
        else:
            js = json.loads((RAW / (meta.stem + ".html")).read_text(encoding="utf-8"))
            txt = html_to_text(js.get("parse", {}).get("text", {}).get("*", ""))
        for p in clean_text(txt, m["lang"]):
            pools[m["lang"]].append((p, m))
    # nzz fit ground truth (German newspaper, CC BY-NC 4.0)
    sp = json.loads((ROOT / "configs" / "p2_split.json").read_text(encoding="utf-8"))
    fit = {u["unit_idx"] for u in sp["units"] if u["role"] == "fit" and u["cell"] == "impresso-nzz_v1.1_train_de"}
    sys.path.insert(0, str(ROOT)); from ocrgate.data import resolve
    nzz = resolve("HIPE-OCRepair-2026-data/data/v0.9/impresso-nzz/de/hipe-ocrepair-bench_v0.9_impresso-nzz_v1.1_train_de.jsonl")
    n_nzz = 0
    for i, line in enumerate(open(nzz, encoding="utf-8")):
        if i not in fit:
            continue
        r = json.loads(line)
        for p in clean_text(r["ground_truth"]["transcription_unit"], "de"):
            pools["de-news"].append((p, {"source": "impresso-nzz TRAIN ground truth (fit role only), NZZ black-letter GT v1.1", "year": r["document_metadata"]["date"][:4],
                                    "licence": "CC BY-NC 4.0 (https://doi.org/10.5281/zenodo.3333627)"})); n_nzz += 1
    for k, v in pools.items():
        print(k, "paragraphs", len(v), "chars", sum(len(p) for p, _ in v), "(nzz paras", n_nzz if k == "de" else "-", ")")
    # targets: (family, lang, n_units, doc_type, hyph_char)
    targets = [("icdar2017-en", "en", 300, "newspaper"), ("impresso-en", "en", 60, "newspaper"), ("overproof", "en", 100, "newspaper"),
               ("icdar2017-fr", "fr", 250, "newspaper"), ("impresso-fr", "fr", 60, "newspaper"),
               ("dta19", "de", 230, "book"), ("impresso-de", "de-news", 60, "newspaper")]
    pools["de-news"] += [x for x in pools["de"] if "wikisource" in x[1]["source"]]  # impresso-de: nzz GT + Gartenlaube
    hyph = {"impresso-en": ("¬", 0.0), "impresso-fr": ("¬", 0.0), "impresso-de": ("¬", 0.0), "overproof": ("¬", 0.0)}
    for fam in ("impresso-de", "overproof", "impresso-en", "impresso-fr"):
        f = stats[fam]; n_lines = f["gold_chars"] / f["chars_per_line_median"]
        hyph[fam] = ("¬", min(0.5, f["n"].get("¬", 0) / n_lines))
    out = open(ROOT / "data_synth" / "clean.jsonl", "w", encoding="utf-8")
    n_out = 0
    for fam, lang, n, doc_type in targets:
        pool = pools[lang]; rng.shuffle(pool)
        sf = stats["dta19-l1"] if fam == "dta19" else stats[fam]
        lo, hi = int(sf["unit_len_q10"]), int(sf["unit_len_q90"])
        width = int(sf["chars_per_line_median"])
        # build units by concatenating consecutive sentences of a paragraph (or whole paragraphs) up to the target length
        pi = 0; made = 0
        while made < n and pi < len(pool):
            target = rng.randint(lo, hi)
            p, m = pool[pi]; pi += 1
            text = p
            while len(text) < target and pi < len(pool) and pool[pi][1]["source"] == m["source"]:
                text += " " + pool[pi][0]; pi += 1
            if len(text) > target:  # cut at a sentence end near the target
                cut = text.rfind(". ", 0, target + 1)
                text = text[:cut + 1] if cut > lo // 2 else text[:target]
            if len(text) < max(150, lo // 2):
                continue
            if fam.startswith("icdar2017"):
                gt = text  # single line, as in the benchmark
            else:
                hc, hr = hyph.get(fam, ("-", 0.0))
                gt = wrap(text, width, rng, hr, hc)
            year = m.get("year", "n/a")
            meta = {"document_id": f"synth-{fam}-{made:04d}", "primary_dataset_name": "synthetic", "language": lang[:2], "document_type": doc_type,
                    "date": year if fam != "icdar2017-en" and fam != "icdar2017-fr" else "n/a"}
            fams = ["dta19-l0", "dta19-l1", "dta19-l2"] if fam == "dta19" else [fam]
            for f_ in fams:
                out.write(json.dumps({"meta": {**meta, "document_id": meta["document_id"].replace(fam, f_)}, "dataset": f_, "source": m["source"],
                                      "licence": m["licence"], "gt": gt}, ensure_ascii=False) + "\n"); n_out += 1
            made += 1
        print(f"{fam:14s} units {made:4d} (pool paragraphs used {pi}/{len(pool)})")
    out.close()
    print("wrote", n_out, "clean units -> data_synth/clean.jsonl")


if __name__ == "__main__":
    fetch() if sys.argv[1] == "fetch" else units()
