#!/usr/bin/env python3
"""Print the held-out-chosen thresholds of one pool configuration from a p3_pool search json.

    python scripts/p3_ens_cfg.py results/p3_pool_search.json LEVELS BY MEMBER[,MEMBER...]   -> "T1 T2 [T3]"
"""
import json
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
levels, by, members = int(sys.argv[2]), sys.argv[3], set(sys.argv[4].split(","))
for r in d["all"]:
    if r["levels"] == levels and r["by"] == by and set(r["members"]) == members:
        print(" ".join(str(t) for t in r["thresholds"]))
        sys.exit(0)
sys.exit(f"no configuration {levels} {by} {sorted(members)} in {sys.argv[1]}")
