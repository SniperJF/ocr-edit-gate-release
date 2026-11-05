#!/usr/bin/env python3
"""Renumber results/server_segments_p3.tsv by start time (concurrent segments opened before an earlier one closed share a number
otherwise) and render results/gpu_hours_p3.txt via scripts/gpu_hours.py _p3."""
import csv
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
p = ROOT / "results" / "server_segments_p3.tsv"
rows = list(csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"))
rows.sort(key=lambda r: r["start_utc"])
with open(p, "w", encoding="utf-8") as f:
    f.write("segment\tstart_utc\tend_utc\tgpus\tnote\n")
    for i, r in enumerate(rows, 1):
        f.write(f"{i}\t{r['start_utc']}\t{r['end_utc']}\t{r['gpus']}\t{r['note']}\n")
print(subprocess.run(["python3", str(ROOT / "scripts" / "gpu_hours.py"), "_p3"], capture_output=True, text=True).stdout)
