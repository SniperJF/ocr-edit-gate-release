#!/usr/bin/env bash
# TEST run, step 2: part A of scripts/test_run_frozen.sh applied unchanged to the LoRA-8B rewriter
# (Qwen/Qwen3-8B + adapter runs/lora/qwen3-8b-p1 served as `p1lora`, DP=1, --enable-lora; FROZEN_CONFIG.md, dev LoRA rows).
# --send-thinking-kwarg on reproduces the dev LoRA run (enable_thinking=false was sent to the Qwen3 template there);
# the later 'auto' rule keys on 'qwen3' in the model name and would have silently left thinking on for `p1lora`.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "${CONFIRM_TEST_RUN:-}" != "1" ]; then echo "refusing: set CONFIRM_TEST_RUN=1 only for the approved final test run" >&2; exit 2; fi
PY=.venv/bin/python; TPY=${VLLM_PYTHON:-${VLLM_ENV:-$HOME/vllm-env}/bin/python}
: "${VLLM_API_KEY:?set VLLM_API_KEY to the bearer token of your local vLLM server}"; export VLLM_API_KEY TOKENIZERS_PARALLELISM=false
FULL=runs/gates/qwen06_pooled; TEXT=runs/gates/qwen06_pooled_nolp; DEVICE=${DEVICE:-cuda}
tmux new -d -s vllm_test "cd $PWD && VLLM_USE_FLASHINFER_SAMPLER=0 MODEL=Qwen/Qwen3-8B DP=1 TP=1 LORA_MODULES=p1lora=$PWD/runs/lora/qwen3-8b-p1 ./serve_model.sh"
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/v1/models)" = 401 ]; do sleep 5; done
$PY -m ocrgate.run --config configs/phase1.yaml --system zeroshot --split test --model p1lora --tag lora8b-frozen --send-thinking-kwarg on
tmux kill-session -t vllm_test; until ! pgrep -f "^${VLLM_ENV:-$HOME/vllm-env}/bin/python.*vllm serve" >/dev/null; do sleep 2; done
RUN=$(ls -td runs/*_zeroshot_test_lora8b-frozen | head -1)
$PY -m ocrgate.blocks --config configs/phase1.yaml --run "$RUN"
$TPY -m ocrgate.gate_apply --run "$RUN" --mode none --device "$DEVICE" --allow-test
$TPY -m ocrgate.gate_apply --run "$RUN" --mode gate --gate $FULL --device "$DEVICE" --allow-test
$TPY -m ocrgate.gate_apply --run "$RUN" --mode gate --gate $TEXT --device "$DEVICE" --allow-test
$TPY -m ocrgate.gate_apply --run "$RUN" --mode oracle_label --device "$DEVICE" --allow-test
echo "LORA TEST DONE $(date -u +%FT%TZ)"
