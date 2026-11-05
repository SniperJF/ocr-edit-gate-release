#!/usr/bin/env bash
# FROZEN Phase 4 configuration: the exact commands of the single, pre-declared THIRD TEST look (secondary / post-hoc systems only;
# the paper's primary stays the P3 ensemble). Refuses to run unless CONFIRM_TEST_RUN=1 is set. Dry-run on dev (reads no test file):
#   SPLIT=dev CONFIRM_TEST_RUN=1 SUF=dryrun scripts/test_run_frozen_p4.sh      (ONLY=A|B restricts the parts; for a part-B-only
#   dry-run on existing dev runs set ZS_RUN / LORA_RUN to those run dirs)
# Runs ONLY: Qwen3.8-27B zero-shot {no gate, full gate, text-only gate, label oracle}, Qwen3.8-LoRA (frozen epoch) {same four},
# and the P4 ensemble ONLY IF frozen/p4_frozen_ensemble.json says beats_p3 = true (gate_heldout criterion of scripts/p4_select.sh).
# Frozen artefacts (sha256 in frozen/p4_frozen_hashes.tsv, frozen/FROZEN_CONFIG_P4.md): gate runs/gates/p2_pooled (rule global,
# unchanged since Phase 2), runs/gates/p2_pooled_nolp, adapter runs/lora/qwen38-27b-p4/ep<N>, frozen/p4_frozen_ensemble.json.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "${CONFIRM_TEST_RUN:-}" != "1" ]; then echo "refusing: set CONFIRM_TEST_RUN=1 only for the approved final test run" >&2; exit 2; fi
SPLIT=${SPLIT:-test}; ONLY=${ONLY:-AB}; DEVICE=${DEVICE:-cuda}; CFG=configs/phase2.yaml
PY=.venv/bin/python; TPY=${VLLM_PYTHON:-${VLLM_ENV:-$HOME/vllm-env}/bin/python}
: "${VLLM_API_KEY:?set VLLM_API_KEY to the bearer token of your local vLLM server}"; export VLLM_API_KEY TOKENIZERS_PARALLELISM=false
FULL=runs/gates/p2_pooled; TEXT=runs/gates/p2_pooled_nolp
ALLOW=(); [ "$SPLIT" = test ] && ALLOW=(--allow-test)
SUF=${SUF:-p4frozen}
EP=$($TPY -c "import json;print(json.load(open('frozen/p4_epoch_selection.json'))['qwen38-lora']['chosen_epoch'])")
ENS=frozen/p4_frozen_ensemble.json
BEATS=$($TPY -c "import json;print(str(json.load(open('$ENS'))['beats_p3']).lower())")
THR=$($TPY -c "import json;print(' '.join(str(t) for t in json.load(open('$ENS'))['thresholds']))")
FAMS=$($TPY -c "import json;print(' '.join(json.load(open('$ENS'))['families']))")
serve() { # MODEL DP TP "LORA_MODULES"
  tmux new -d -s vllm_test "cd $PWD && VLLM_USE_FLASHINFER_SAMPLER=0 MODEL=$1 DP=$2 TP=$3 LORA_MODULES='$4' EXTRA_ARGS='--language-model-only' ./serve_model.sh"
  until [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/v1/models)" = 401 ]; do sleep 5; done
}
stop() { tmux kill-session -t vllm_test; until ! pgrep -f "^${VLLM_ENV:-$HOME/vllm-env}/bin/python.*vllm serve" >/dev/null; do sleep 2; done; }
gate_all() { # RUN
  $PY -m ocrgate.blocks --config $CFG --run "$1"                                   # labels: diagnostics only
  $TPY -m ocrgate.gate_apply --config $CFG --run "$1" --mode none --device "$DEVICE" "${ALLOW[@]}"
  $TPY -m ocrgate.gate_apply --config $CFG --run "$1" --mode gate --gate $FULL --rule global --device "$DEVICE" "${ALLOW[@]}"
  $TPY -m ocrgate.gate_apply --config $CFG --run "$1" --mode gate --gate $TEXT --rule global --device "$DEVICE" "${ALLOW[@]}"
  $TPY -m ocrgate.gate_apply --config $CFG --run "$1" --mode oracle_label --device "$DEVICE" "${ALLOW[@]}"   # upper bound only
}
# A. one server: Qwen3.8-27B text-only (TP=1) + the frozen LoRA adapter; zero-shot pass (base) and LoRA pass; the four gate variants each
if [[ "$ONLY" == *A* ]]; then
  serve Qwen/Qwen3.8-27B 1 1 "qwen38-lora-ep$EP=$PWD/runs/lora/qwen38-27b-p4/ep$EP"
  $PY -m ocrgate.run --config $CFG --system zeroshot --split "$SPLIT" --model Qwen/Qwen3.8-27B --tag qwen38-zs-$SUF
  $PY -m ocrgate.run --config $CFG --system zeroshot --split "$SPLIT" --model qwen38-lora-ep$EP --tag qwen38-lora-$SUF --send-thinking-kwarg on
  stop; gate_all "$(ls -td runs/*_zeroshot_${SPLIT}_qwen38-zs-$SUF | head -1)"; gate_all "$(ls -td runs/*_zeroshot_${SPLIT}_qwen38-lora-$SUF | head -1)"
fi
# B. the P4 ensemble, ONLY if it beat the P3 ensemble on gate_heldout (members = frozen families; the P3 member runs of this split
#    are the frozen P3 test runs when SPLIT=test, the P3 dev runs when SPLIT=dev)
if [[ "$ONLY" == *B* ]]; then
  if [ "$BEATS" != true ]; then echo "P4 ensemble NOT run: beats_p3=$BEATS in $ENS (pre-declared rule)"; else
    ZS=${ZS_RUN:-$(ls -td runs/*_zeroshot_${SPLIT}_qwen38-zs-$SUF | head -1)}; LORA=${LORA_RUN:-$(ls -td runs/*_zeroshot_${SPLIT}_qwen38-lora-$SUF | head -1)}
    if [ "$SPLIT" = test ]; then P3S=p3frozen; L24=$(ls -d runs/*_zeroshot_test_lora24b-$P3S); L32=$(ls -d runs/*_zeroshot_test_lora32b-$P3S); L72=$(ls -d runs/*_zeroshot_test_lora72b-$P3S); SYN=$(ls -d runs/*_zeroshot_test_lora72b-syn-$P3S); Q72=$(ls -d runs/*_zeroshot_test_q72b-p2frozen 2>/dev/null || true)
    else L24=${L24_RUN:-runs/20261030T095711Z_zeroshot_dev_lora24b-ep2}; L32=${L32_RUN:-runs/20261030T103005Z_zeroshot_dev_lora32b-ep2}; L72=${L72_RUN:-runs/20261031T020135Z_zeroshot_dev_lora72b-ep2}; SYN=${SYN_RUN:-runs/20261031T073628Z_zeroshot_dev_lora72b-syn-ep1}; Q72=${Q72_RUN:-runs/20261030T055235Z_zeroshot_dev_q72b}; fi  # dev run ids of the original runs; override with *_RUN
    RUNS=(); for f in $FAMS; do case $f in lora24b) RUNS+=($L24);; lora32b) RUNS+=($L32);; lora72b) RUNS+=($L72);; lora72bsyn) RUNS+=($SYN);; q72b) RUNS+=($Q72);; qwen38zs) RUNS+=($ZS);; qwen38lora) RUNS+=($LORA);; *) echo "unknown family $f" >&2; exit 1;; esac; done
    $TPY -m ocrgate.ensemble --config $CFG --runs "${RUNS[@]}" --gate $FULL --agree-thresholds $THR --families $FAMS --tag ${SPLIT}-p4-ens-$SUF --device "$DEVICE" "${ALLOW[@]}"
  fi
fi
echo "FROZEN P4 RUN DONE ($SPLIT, $ONLY) $(date -u +%FT%TZ)"
