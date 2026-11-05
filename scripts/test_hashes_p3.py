#!/usr/bin/env python3
"""Second (P3) TEST look, step 1: recompute sha256 + size of every frozen artefact against frozen/p3_frozen_hashes.tsv
(P3 table) and frozen/p2_frozen_hashes.tsv (text-only gate + P2 adapter configs used by the secondaries and the P2
comparison), check the stored thresholds against FROZEN_CONFIG_P3.md / P2, record git HEAD. Writes results/test_hashes_p3.txt.
Exit 1 on any mismatch (the caller writes P3_TEST_HASH_MISMATCH and stops).

    python3 scripts/test_hashes_p3.py
"""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()

refs = {}  # path -> (bytes, sha, source)
for src in ("frozen/p3_frozen_hashes.tsv", "frozen/p2_frozen_hashes.tsv"):
    for line in Path(src).read_text().splitlines():
        f = line.rstrip("\n").split("\t")
        if not f or f[0] in ("role", "") :
            continue
        path, size, sha = (f[1], f[2], f[3]) if len(f) == 4 else (f[0], f[1], f[2])
        if path in refs and refs[path][:2] != (int(size), sha):
            sys.exit(f"reference tables disagree on {path}")
        refs.setdefault(path, (int(size), sha, src))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


rows, ok = [], True
for path, (size, sha, src) in refs.items():
    p = Path(path)
    if not p.is_file():
        rows.append((path, "-", "MISSING", sha, src, "MISMATCH")); ok = False; continue
    n, d = p.stat().st_size, sha256(p)
    m = "OK" if (n == size and d == sha) else "MISMATCH"
    ok &= m == "OK"
    rows.append((path, f"{n:,}", d, sha, src, m))
    print(f"[{m}] {path}", file=sys.stderr, flush=True)

# thresholds and ensemble configuration (values quoted in FROZEN_CONFIG_P3.md / FROZEN_CONFIG_P2.md)
checks = []
ens = json.load(open("frozen/p3_frozen_ensemble.json"))
checks.append(("p3_frozen_ensemble.json thresholds", ens["thresholds"], [0.7786493301391602, 0.3098590523004532, 0.011687257327139378]))
checks.append(("p3_frozen_ensemble.json members/families", (ens["members"], ens["families"], ens["levels"]),
               (["lora24b", "lora32b", "lora72b", "lora72bsyn"], ["lora24b", "lora32b", "lora72b", "lora72bsyn"], 3)))
gf = json.load(open("runs/gates/p2_pooled/gate.json")); gt = json.load(open("runs/gates/p2_pooled_nolp/gate.json"))
checks.append(("full gate stored threshold", gf["threshold"], 0.37570643424987793))
checks.append(("text-only gate stored threshold", gt["threshold"], 0.29665425419807434))
for g, exp in (("runs/gates/p2_pooled", 0.37570643424987793), ("runs/gates/p2_pooled_nolp", 0.29665425419807434)):
    r = json.load(open(f"{g}/rules.json"))
    rule = r["rules"]["global"]
    thr = rule["threshold"] if isinstance(rule, dict) and "threshold" in rule else rule
    checks.append((f"{g}/rules.json rule global threshold", thr, exp))
a3 = json.load(open("frozen/agree_thresholds_3pool.json"))
checks.append(("P2 3-pool agreement thresholds (t_single, t_agree)", (a3["t_single"], a3["t_agree"]), (0.7563575506210327, 0.07864424213767052)))
for name, got, exp in checks:
    m = "OK" if got == exp else "MISMATCH"
    ok &= m == "OK"

out = [f"# sha256 check of every frozen Phase 3 artefact before the SECOND (P3) TEST look ({now})", "",
       f"git HEAD: {head}", f"git working tree: {'clean' if not dirty else 'DIRTY: ' + dirty.replace(chr(10), ' ; ')}", "",
       "| file | size (bytes) | sha256 recomputed | reference | table | match |", "|---|---|---|---|---|---|"]
for path, n, d, sha, src, m in rows:
    out.append(f"| `{path}` | {n} | `{d}` | `{sha}` | {Path(src).name} | {m} |")
out += ["", "| check | stored | expected (FROZEN_CONFIG) | match |", "|---|---|---|---|"]
for name, got, exp in checks:
    out.append(f"| {name} | `{got}` | `{exp}` | {'OK' if got == exp else 'MISMATCH'} |")
out += ["", f"Result: {'ALL FROZEN HASHES AND THRESHOLDS MATCH' if ok else 'MISMATCH (see above)'}", ""]
Path("results/test_hashes_p3.txt").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
sys.exit(0 if ok else 1)
