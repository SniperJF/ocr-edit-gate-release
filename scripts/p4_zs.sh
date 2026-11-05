#!/usr/bin/env bash
# P4-a/b: serve Qwen3.8-27B text-only on GPU 1 (TP=1, --enable-lora with the P4-a smoke adapter attached for the vLLM LoRA
# loader check), (a) 8-unit smoke through the smoke adapter, (b) Phase 0 zero-shot prompt on the held-out train subset + dev
# (model = the base), blocks labelled; markers P4_A_DONE / P4_B_DONE.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/p4_lib.sh
VISIBLE=1 EXTRA_ARGS='--language-model-only' serve Qwen/Qwen3.8-27B 1 1 "qwen38-smoke=$PWD/runs/lora/qwen38-27b-p4-smoke" \
  "GPU1: vLLM Qwen3.8-27B text-only TP=1 + smoke adapter: (a) LoRA-loader smoke, (b) zero-shot held-out train subset + dev"
$PY -m ocrgate.run --config $CFG --system zeroshot --split dev --model qwen38-smoke --limit 1 --tag qwen38-lora-smoke --send-thinking-kwarg on 2>&1 | grep -v -i warn | tail -2
mark "P4_A_DONE" "weights verified (18 shards, 55,563,006,776 bytes, 1,199 tensors); vLLM 0.32 serves Qwen3_5ForConditionalGeneration text-only (--language-model-only) with thinking off (chat_template_kwargs enable_thinking=false; no <think> in outputs); LoRA targets attn q/k/v/o + GDN in_proj_qkv/z/b/a + out_proj + MLP (12 modules, 116.7M params, r16); smoke adapter loads in vLLM: $(ls -td runs/*_qwen38-lora-smoke | head -1); results/p4_feasibility.md"
$PY -m ocrgate.run --config $CFG --system zeroshot --split train --subset $HO --model Qwen/Qwen3.8-27B --tag qwen38-zs-ho 2>&1 | grep -v -i warn | tail -1
$PY -m ocrgate.run --config $CFG --system zeroshot --split dev --model Qwen/Qwen3.8-27B --tag qwen38-zs 2>&1 | grep -v -i warn | tail -1
stop
blocks "$(latest_run qwen38-zs-ho)"; blocks "$(latest_run qwen38-zs)"
mark "P4_B_DONE" "Qwen3.8-27B zero-shot (Phase 0 prompt, greedy, thinking off): held-out $(latest_run qwen38-zs-ho) dev $(latest_run qwen38-zs)"
echo "P4 ZS DONE $(now)"
