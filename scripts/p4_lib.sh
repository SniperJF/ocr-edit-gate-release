# Phase 4 shared (copy of p3_lib.sh with the P4 ledger, status file and tmux session; serve() passes EXTRA_ARGS to serve_model.sh) shell helpers (source me). GPU ledger: results/server_segments_p4.tsv (segment, start, end, gpus, note).
PY=.venv/bin/python; TPY=${VLLM_PYTHON:-${VLLM_ENV:-$HOME/vllm-env}/bin/python}
: "${VLLM_API_KEY:?set VLLM_API_KEY to the bearer token of your local vLLM server}"; export VLLM_API_KEY TOKENIZERS_PARALLELISM=false
SEG=results/server_segments_p4.tsv; CFG=configs/phase2.yaml; HO=configs/p2_heldout_subset.json
mkdir -p logs results; STATUS=${STATUS_FILE:-logs/p4_status.txt}
now() { date -u +%FT%TZ; }
mark() { echo "$1 $(now) ${2:-}" >> "$STATUS"; }
nseg() { awk -F'\t' 'NR>1 && $1 ~ /^[0-9]+$/ {n=$1} END {print n+1}' "$SEG"; }
seg_open() { # GPUS NOTE -> sets SEG_N SEG_START SEG_GPUS SEG_NOTE
  SEG_N=$(nseg); SEG_START=$(now); SEG_GPUS=$1; SEG_NOTE="$2"; echo "[seg $SEG_N open] $SEG_NOTE ($SEG_START)"; }
seg_close() { printf '%s\t%s\t%s\t%s\t%s\n' "$SEG_N" "$SEG_START" "$(now)" "$SEG_GPUS" "$SEG_NOTE" >> "$SEG"; echo "[seg $SEG_N closed] $(now)"; }
serve() { # MODEL DP TP "LORA_MODULES" NOTE  (vLLM in tmux session vllm_p4; env inside the tmux command string)
  seg_open $(( $2 * $3 )) "$5"
  local vis=""; [ -n "${VISIBLE:-}" ] && vis="CUDA_VISIBLE_DEVICES=$VISIBLE "
  tmux new -d -s vllm_p4 "cd $PWD && ${vis}VLLM_USE_FLASHINFER_SAMPLER=0 MODEL=$1 DP=$2 TP=$3 LORA_MODULES='$4' EXTRA_ARGS='${EXTRA_ARGS:-}' ./serve_model.sh"
  until [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/v1/models)" = 401 ]; do
    sleep 5
    if ! tmux has-session -t vllm_p4 2>/dev/null; then echo "vLLM session died" >&2; seg_close; return 1; fi
  done
  echo "[serve] $1 DP=$2 TP=$3 up at $(now)"
}
stop() {
  tmux kill-session -t vllm_p4 || true
  until ! pgrep -f "^${VLLM_ENV:-$HOME/vllm-env}/bin/python.*vllm serve" >/dev/null; do sleep 2; done
  sleep 3; seg_close
}
blocks() { # RUN_DIR: label the edit blocks (CPU)
  $PY -m ocrgate.blocks --config $CFG --run "$1" | head -c 400; echo
}
latest_run() { ls -td runs/*_"$1" | head -1; }
