#!/usr/bin/env python3
"""Compute GPU-hours from results/server_segments.tsv and write results/gpu_hours.txt.

    python scripts/gpu_hours.py [SUFFIX]   e.g. SUFFIX=_0b -> server_segments_0b.tsv / gpu_hours_0b.txt

Each row of the TSV is one vLLM server lifetime: start_utc, end_utc (ISO 8601 Z), number of GPUs
used, note. GPU-hours = sum over segments of wall-clock hours x GPUs. Times are recorded by the
operator when the server process is started and when it exits (see FINAL_REPORT.md).
"""
import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else ""
seg_path = ROOT / "results" / f"server_segments{SUFFIX}.tsv"
out_path = ROOT / "results" / f"gpu_hours{SUFFIX}.txt"


def parse(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")


rows = list(csv.DictReader(open(seg_path, encoding="utf-8"), delimiter="\t"))
PHASE = {"_0b": "Phase 0b", "_p1": "Phase 1", "_test": "single approved TEST run", "_p2": "Phase 2", "_p3": "Phase 3", "_test_p3": "second (P3) TEST look", "_p4": "Phase 4 (dev)", "_test_p4": "third (P4) TEST look"}.get(SUFFIX, "Phase 0 kill test")
lines = [f"GPU wall-clock for the {PHASE} (vLLM server lifetimes, 2x H200 NVL)", ""]
lines.append(f"{'seg':<4}{'start_utc':<22}{'end_utc':<22}{'hours':>7}{'gpus':>5}{'gpu_h':>8}  note")
total_h = total_gpu_h = 0.0
for r in rows:
    if not r["end_utc"]:
        sys.exit(f"segment {r['segment']} has no end time yet")
    h = (parse(r["end_utc"]) - parse(r["start_utc"])).total_seconds() / 3600
    g = int(r["gpus"])
    total_h += h
    total_gpu_h += h * g
    lines.append(f"{r['segment']:<4}{r['start_utc']:<22}{r['end_utc']:<22}{h:>7.3f}{g:>5}{h*g:>8.3f}  {r['note']}")
lines += ["", f"total server wall-clock hours: {total_h:.3f}", f"total GPU-hours (hours x GPUs): {total_gpu_h:.3f}"]
out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
