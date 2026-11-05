#!/usr/bin/env bash
# Phase 0 kill test: run nocorrect, zeroshot and oracle_gated on dev, score each with the official
# scorer, and regenerate results/phase0.md.
#
#   ./run_all.sh [ENDPOINT]        default ENDPOINT=http://127.0.0.1:8000/v1
#
# Env: VLLM_API_KEY (else read from $VLLM_API_KEY_FILE, default $HOME/.config/vllm/api_key), SPLITS (default "dev"),
#      SYSTEMS (default "nocorrect zeroshot oracle_gated"), CONFIG, PYTHON, WAIT_S.
set -euo pipefail
cd "$(dirname "$0")"

ENDPOINT=${1:-http://127.0.0.1:8000/v1}
PYTHON=${PYTHON:-.venv/bin/python}
CONFIG=${CONFIG:-configs/phase0.yaml}
SPLITS=${SPLITS:-dev}
SYSTEMS=${SYSTEMS:-nocorrect zeroshot oracle_gated}
WAIT_S=${WAIT_S:-900}

if [ -z "${VLLM_API_KEY:-}" ]; then
  KEY_FILE=${VLLM_API_KEY_FILE:-$HOME/.config/vllm/api_key}
  [ -r "$KEY_FILE" ] || { echo "VLLM_API_KEY unset and $KEY_FILE unreadable" >&2; exit 1; }
  VLLM_API_KEY="$(tr -d '[:space:]' < "$KEY_FILE")"
  export VLLM_API_KEY
fi

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "WARNING: uncommitted changes; runs will be logged as git_dirty=true" >&2
fi

# Wait for the endpoint (model loading can take a few minutes). Header passed via stdin, not argv.
echo "Waiting for $ENDPOINT/models (up to ${WAIT_S}s) ..."
deadline=$((SECONDS + WAIT_S))
until printf 'header = "Authorization: Bearer %s"\n' "$VLLM_API_KEY" \
      | curl -fsS -o /dev/null --max-time 5 -K - "$ENDPOINT/models"; do
  if [ $SECONDS -ge $deadline ]; then echo "Endpoint not ready after ${WAIT_S}s" >&2; exit 1; fi
  sleep 10
done

mkdir -p logs
LOG=logs/run_all_$(date -u +%Y%m%dT%H%M%SZ).log
for split in $SPLITS; do
  for system in $SYSTEMS; do
    "$PYTHON" -m ocrgate.run --config "$CONFIG" --system "$system" --split "$split" --base-url "$ENDPOINT" 2>&1 | tee -a "$LOG"
  done
done
echo "Done. Table: results/phase0.md   Per-run logs: runs/<run_id>/log.jsonl   Console log: $LOG"
