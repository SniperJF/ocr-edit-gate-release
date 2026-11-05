#!/usr/bin/env bash
# Phase 3 GPU chain after P3-1: P3-2 sampled candidates (LoRA-72B, then LoRA-32B), P3-3 synthetic-mix retraining of the best
# rewriter (LoRA-72B) and its held-out/dev inference. Sequential (every step needs both H200s).
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=Qwen/Qwen2.5-72B-Instruct SERVED=lora72b-ep2 TAG=lora72b-ep2 LORA_MODULES="lora72b-ep2=$PWD/runs/lora/qwen25-72b-p3/ep2" N=4 T=0.7 scripts/p3_samples.sh
MODEL=Qwen/Qwen3-32B SERVED=lora32b-ep2 TAG=lora32b-ep2 LORA_MODULES="lora32b-ep2=$PWD/runs/lora/qwen3-32b-p2/ep2" N=4 T=0.7 THINK=on DP=2 TP=1 scripts/p3_samples.sh
echo "P3_2_SAMPLES_ALL_DONE $(date -u +%FT%TZ)"
MODEL=Qwen/Qwen2.5-72B-Instruct OUT=runs/lora/qwen25-72b-syn-p3 EXTRA=data_synth/synth.jsonl DEVMAP=auto NOTE="GPU0+1: bf16 LoRA-tune Qwen2.5-72B-Instruct (sharded) on P2 fit (1572) + 1520 noise-matched synthetic pairs, 2 epochs, adapters ep1/ep2" scripts/p3_lora_synth_train.sh
LORA=$PWD/runs/lora/qwen25-72b-syn-p3 TAG=lora72b-syn scripts/p3_lora72b_infer.sh
echo "P3_CHAIN_DONE $(date -u +%FT%TZ)"
