#!/usr/bin/env bash
# FROZEN Phase 1 configuration: the exact commands of the single, later TEST run. NOT executed in Phase 1.
# Refuses to run unless CONFIRM_TEST_RUN=1 is set (the test split is locked until the project lead gives the OK).
# Dry-run on dev: `SPLIT=dev ONLY=B DEVICE=cpu CONFIRM_TEST_RUN=1 scripts/test_run_frozen.sh` (reads no test file;
# ONLY=A|B restricts to part A (our rewriters) or part B (participant outputs); DEVICE is the gate device).
#
# Frozen gates (see frozen/FROZEN_CONFIG_P1.md for sha256 and thresholds):
#   full gate      runs/gates/qwen06_pooled       (Qwen3-0.6B + logprob features; our own rewriters)
#   text-only gate runs/gates/qwen06_pooled_nolp  (Qwen3-0.6B, no logprob features; any system's output)
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "${CONFIRM_TEST_RUN:-}" != "1" ]; then echo "refusing: set CONFIRM_TEST_RUN=1 only for the approved final test run" >&2; exit 2; fi
SPLIT=${SPLIT:-test}
PY=.venv/bin/python; TPY=${VLLM_PYTHON:-${VLLM_ENV:-$HOME/vllm-env}/bin/python}
: "${VLLM_API_KEY:?set VLLM_API_KEY to the bearer token of your local vLLM server}"; export VLLM_API_KEY TOKENIZERS_PARALLELISM=false
FULL=runs/gates/qwen06_pooled; TEXT=runs/gates/qwen06_pooled_nolp
ALLOW=(); [ "$SPLIT" = test ] && ALLOW=(--allow-test)
ONLY=${ONLY:-AB}; DEVICE=${DEVICE:-cuda}

serve() { # serve MODEL DP TP  (env vars inside the tmux command string; waits for the endpoint)
  tmux new -d -s vllm_test "cd $PWD && VLLM_USE_FLASHINFER_SAMPLER=0 MODEL=$1 DP=$2 TP=$3 ./serve_model.sh"
  until [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/v1/models)" = 401 ]; do sleep 5; done
}
stop() { tmux kill-session -t vllm_test; until ! pgrep -f "^${VLLM_ENV:-$HOME/vllm-env}/bin/python.*vllm serve" >/dev/null; do sleep 2; done; }

# A. our rewriters, with and without the gate (config phase1.yaml: logprobs on; Qwen3 thinking off; greedy; seed 1234)
[[ "$ONLY" == *A* ]] && for spec in "Qwen/Qwen3-8B 2 1 8b" "Qwen/Qwen3-32B 2 1 32b" "mistralai/Mistral-Small-24B-Instruct-2501 2 1 mistral24b" "Qwen/Qwen2.5-72B-Instruct 1 2 q72b"; do
  set -- $spec; MODEL=$1; DP=$2; TP=$3; TAG=$4
  serve "$MODEL" "$DP" "$TP"
  $PY -m ocrgate.run --config configs/phase1.yaml --system zeroshot --split "$SPLIT" --model "$MODEL" --tag "${TAG}-frozen"
  stop
  RUN=$(ls -td runs/*_zeroshot_${SPLIT}_${TAG}-frozen | head -1)
  $PY -m ocrgate.blocks --config configs/phase1.yaml --run "$RUN"            # labels are computed but only used for diagnostics
  $TPY -m ocrgate.gate_apply --run "$RUN" --mode none --device "$DEVICE" "${ALLOW[@]}"            # = rewriter without gate (official scorer)
  $TPY -m ocrgate.gate_apply --run "$RUN" --mode gate --gate $FULL --device "$DEVICE" "${ALLOW[@]}"
  $TPY -m ocrgate.gate_apply --run "$RUN" --mode gate --gate $TEXT --device "$DEVICE" "${ALLOW[@]}"
  $TPY -m ocrgate.gate_apply --run "$RUN" --mode oracle_label --device "$DEVICE" "${ALLOW[@]}"    # upper bound, reported separately
done
[[ "$ONLY" == *B* ]] || { echo "FROZEN RUN DONE (part A only, $SPLIT)"; exit 0; }

# B. the frozen TEXT-ONLY gate on every participant's public test output (HIPE-OCRepair-2026-eval/data/systems/),
#    scored with the official scorer against data/reference/. (dev dry-run: our own dev predictions instead.)
OUT=runs/external_gated_${SPLIT}; mkdir -p "$OUT" "$OUT/scores"
if [ "$SPLIT" = test ]; then
  SUBS=(HIPE-OCRepair-2026-eval/data/systems/*_run[0-9].jsonl); REFDIR=HIPE-OCRepair-2026-eval/data/reference
else
  SUBS=(runs/20261030T011131Z_zeroshot_nogate_dev_8b-lp/predictions/*.jsonl); REFDIR=runs/20261030T011131Z_zeroshot_nogate_dev_8b-lp/scores/refs
fi
for SUB in "${SUBS[@]}"; do
  STEM=$(basename "$SUB" .jsonl); REFSTEM=$(echo "$STEM" | sed -E 's/^[^_]+_//; s/_run[0-9]+$//; s/_masked-test_/_test_/')
  $TPY -m ocrgate.gate_external --gate $TEXT --submission "$SUB" --reference "$REFDIR/$REFSTEM.jsonl" --out-dir "$OUT" --team-suffix tgate --device "$DEVICE"
  GATED="$OUT/${STEM%%_*}-tgate_${STEM#*_}.jsonl"   # exact name written by gate_external (<team>-tgate_<rest>); a glob without the team matched other teams' files
  # score_one.py derives the reference from the hypothesis file name (masked-test -> test is handled by the scorer)
  $PY -I HIPE-OCRepair-2026-eval/lib/score_one.py --hypothesis "$GATED" --reference-dir "$REFDIR" --output "$OUT/scores/$(basename "$GATED" .jsonl).json" --log-file "$OUT/scores/$(basename "$GATED" .jsonl).log"
  $PY -I HIPE-OCRepair-2026-eval/lib/score_one.py --hypothesis "$SUB" --reference-dir "$REFDIR" --output "$OUT/scores/$STEM.ungated.json" --log-file "$OUT/scores/$STEM.ungated.log"
done
echo "FROZEN RUN DONE ($SPLIT) $(date -u +%FT%TZ)"
