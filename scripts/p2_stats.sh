#!/usr/bin/env bash
# Phase 2: paired bootstrap (10,000 resamples, seed 1234) for the pairs in results/stats_pairs_p2.json -> results/stats_p2.jsonl
# (dev runs only; ocrgate.stats refuses the test split without the approved flags). Usage: scripts/p2_stats.sh [pairs.json] [out.jsonl]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${VLLM_PYTHON:-${VLLM_ENV:-$HOME/vllm-env}/bin/python}
PAIRS=${1:-results/stats_pairs_p2.json}; OUT=${2:-results/stats_p2.jsonl}
: > "$OUT"
$PY - "$PAIRS" <<'PYEOF' | while IFS=$'\t' read -r label base test; do
import json, sys
for p in json.load(open(sys.argv[1])):
    print(f"{p['label']}\t{p['base']}\t{p['test']}")
PYEOF
  echo "[stats] $label" >&2
  $PY -m ocrgate.stats --config configs/phase2.yaml --base "runs/$base" --test "runs/$test" --n 10000 --seed 1234 2>/dev/null | LABEL="$label" $PY -c "import json,sys,os; d=json.load(sys.stdin); d['label']=os.environ['LABEL']; print(json.dumps(d))" >> "$OUT"
done
echo "STATS DONE $(date -u +%FT%TZ)"
