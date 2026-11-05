#!/usr/bin/env bash
# Phase 1: train the learned gates on the TRAIN-split blocks of one rewriter run (one GPU).
#   scripts/p1_train_gates.sh <suffix> <train_blocks.jsonl> [more blocks...]
# e.g. CUDA_VISIBLE_DEVICES=0 scripts/p1_train_gates.sh 8b runs/<train_8b>/blocks.jsonl
# Encoders: Qwen3-0.6B (lr 3e-5), byT5-base (lr 1e-4, byte inputs -> max_len 384), Qwen3-1.7B (lr 2e-5).
# Hyper-parameters are fixed a priori; epoch and threshold are selected on the held-out part of TRAIN.
set -euo pipefail
cd "$(dirname "$0")/.."
SUF=$1; shift
PY=${VLLM_PYTHON:-${VLLM_ENV:-$HOME/vllm-env}/bin/python}
export TOKENIZERS_PARALLELISM=false
$PY -m ocrgate.gate_train --blocks "$@" --encoder Qwen/Qwen3-0.6B --name "qwen06_$SUF" --epochs 3 --lr 3e-5 --max-len 256
$PY -m ocrgate.gate_train --blocks "$@" --encoder google/byt5-base --name "byt5_$SUF" --epochs 3 --lr 1e-4 --max-len 384
$PY -m ocrgate.gate_train --blocks "$@" --encoder Qwen/Qwen3-1.7B --name "qwen17_$SUF" --epochs 3 --lr 2e-5 --max-len 256
echo "GATES DONE $SUF $(date -u +%FT%TZ)"
