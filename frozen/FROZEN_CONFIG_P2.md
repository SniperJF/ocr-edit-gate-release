# Phase 2 FROZEN configuration for the single, later TEST run (NOT executed; test split untouched)

Frozen on 2026-10-30 after the dev-only Phase 2 (`results/FINAL_REPORT_P2.md`). Every choice below was made on the
held-out part of TRAIN (role `gate_heldout` of `configs/p2_split.json`, 360 units; LoRA epochs on the full 721
held-out units); dev was used for reporting only. Original (private) development repository; this release reproduces its code. The test split (`HIPE-OCRepair-2026-eval/data/reference/`) and the participant files (`data/systems/`) were
not opened in Phase 2.

## Frozen artefacts (git-ignored weights; sha256 of the files on disk, `frozen/p2_frozen_hashes.tsv`)

| role | path | size (bytes) | sha256 |
|---|---|---|---|
| **full gate** (uses the rewriter's token logprobs) | `runs/gates/p2_pooled/model.pt` | 1,193,273,291 | `a8d11d8782040baf8e1485559461a8e5709d6c896e14f382f850d58911bc3762` |
| full gate metadata (standardisation, stored threshold) | `runs/gates/p2_pooled/gate.json` | 2,467 | `364d5318ec549c8b28a477d9435ed9cbf5df193003ad2fbd3a5418adf95b4634` |
| full gate decision rules | `runs/gates/p2_pooled/rules.json` | 20,462 | `ee24d0543cee178b43179184cd7d798d09f95bfdf1466f45e8444296c370e3be` |
| **text-only gate** (any system's output; logprob features zero) | `runs/gates/p2_pooled_nolp/model.pt` | 1,193,273,291 | `a3a543d305957b3a53e4585783ebc83ca504a14b944c1a97be87055149fc3a9a` |
| text-only gate metadata | `runs/gates/p2_pooled_nolp/gate.json` | 2,244 | `07f75b2b4dcdeae3a801d9ab34bba2476d147ad3a663ca7681226cb76def6209` |
| text-only gate decision rules | `runs/gates/p2_pooled_nolp/rules.json` | 20,479 | `50efc7cb601377acd62b6f9abeb8eedc3e980cd3dabe017ad8f65c6d14c2b5ee` |
| **LoRA-24B adapter** (Mistral-Small-24B-Instruct-2501, epoch 2) | `runs/lora/mistral24b-p2/ep2/adapter_model.safetensors` | 369,698,576 | `aee8ca243dd2990d367b2192426f24dd14b76fc6a2cfd211151b493a1853107a` |
| | `runs/lora/mistral24b-p2/ep2/adapter_config.json` | 1,197 | `5f93bb13eb78fb78c69b7bf9a878da85b97eadf141ead51c145bc3421c1c659f` |
| **LoRA-32B adapter** (Qwen3-32B, epoch 2) | `runs/lora/qwen3-32b-p2/ep2/adapter_model.safetensors` | 536,991,984 | `4ca100cc0b8db683e263c28f28b5d9b18de5a3bf7e18351c52cf9a48e97e3472` |
| | `runs/lora/qwen3-32b-p2/ep2/adapter_config.json` | 1,170 | `066491a5943eb2669654f5ad3534c7252bb2b8ea907d5b81bab9648feeb16adb` |
| ensemble agreement thresholds | `frozen/agree_thresholds_3pool.json` (tracked) | | t_single = 0.7563575506210327, t_agree = 0.07864424213767052 |

## Gates

Same architecture, features, block rule and text format as Phase 1 (`frozen/FROZEN_CONFIG_P1.md`): Qwen3-0.6B encoder
(HF snapshot `c1899de289a04d12100db370d81485cdf75e47ca`), last-token pooling, bf16, head Linear(1024+15 -> 256) + GELU +
Dropout(0.1) + Linear(256 -> 1), `max_len` 256, 3 epochs (batch 32, lr 3e-5 / 1e-3, AdamW wd 0.01, 5% warm-up, seed
1234), **checkpoint = epoch 1** (held-out-best for both). Training data = pooled blocks of zero-shot Qwen3-8B, Qwen3-32B,
Mistral-Small-24B and Qwen2.5-72B on the full TRAIN split and of LoRA-24B / LoRA-32B (epoch 2) on the 721 LoRA-held-out
train units; fit = every unit except role `gate_heldout` (370,049 labelled blocks), held-out = `gate_heldout`
(91,340 labelled blocks). Commands (`scripts/p2_train_gates.sh`):

```
$VLLM_PYTHON -m ocrgate.gate_train --config configs/phase2.yaml --blocks runs/20261030T010915Z_zeroshot_train_8b-lp/blocks.jsonl runs/20261030T012216Z_zeroshot_train_32b-lp/blocks.jsonl runs/20261030T084539Z_zeroshot_train_mistral24b-p2/blocks.jsonl runs/20261030T090807Z_zeroshot_train_q72b-p2/blocks.jsonl runs/20261030T094959Z_zeroshot_train_lora24b-ep2-ho/blocks.jsonl runs/20261030T101923Z_zeroshot_train_lora32b-ep2-ho/blocks.jsonl --encoder Qwen/Qwen3-0.6B --name p2_pooled --split configs/p2_split.json --results results/gates_p2.jsonl --epochs 3 --lr 3e-5 --max-len 256
(same with --name p2_pooled_nolp --no-lp)
$VLLM_PYTHON -m ocrgate.rules --gate runs/gates/p2_pooled --cv 5      (and p2_pooled_nolp)
```

## Decision rule: `global` (one threshold for all blocks, languages and systems)

| gate | threshold on P(helpful) | kept block if |
|---|---|---|
| full `p2_pooled` | **0.37570643424987793** | score >= threshold |
| text-only `p2_pooled_nolp` | **0.29665425419807434** | score >= threshold |

Selection: five rules (global, editrate, sysconf, lang, lang_editrate; `ocrgate/rules.py`) were chosen on the held-out
blocks and compared by 5-fold cross-fitted held-out gain; all five lie within 0.15% of each other (full gate 33.05-33.10
of an oracle 38.28; text-only 32.63-32.77), so the simplest rule is frozen (tie-break: a rule needs > 0.1 gain units,
0.3%, over `global` to be preferred; none has it). The thresholds are those of `rules.json["rules"]["global"]` and equal
the stored `gate.json` thresholds. The other rules stay in `rules.json` for reference only. `gate_apply --rule global`
and `gate_apply` without `--rule` are identical.

## Rewriters

| rewriter | serving | thinking | prompt / decoding |
|---|---|---|---|
| **LoRA-24B** = `mistralai/Mistral-Small-24B-Instruct-2501` + adapter `runs/lora/mistral24b-p2/ep2`, served as `lora24b-ep2` | TP=2, `--enable-lora --max-lora-rank 16`, vLLM mistral tokenizer mode (auto) | n/a | Phase 0 zero-shot prompt (`prompts.zeroshot_messages`), greedy, temperature 0, top_p 1, seed 1234, max_tokens = max(512, 0.6 x input chars), fallback to OCR on empty/truncated/failed; logprobs on |
| **LoRA-32B** = `Qwen/Qwen3-32B` + adapter `runs/lora/qwen3-32b-p2/ep2`, served as `lora32b-ep2` | TP=2, `--enable-lora --max-lora-rank 16` | off (`--send-thinking-kwarg on` so that `enable_thinking=false` is sent to the Qwen3 template under the adapter name) | same |
| Qwen2.5-72B-Instruct zero-shot | TP=2 bf16 | n/a | same (as in Phase 1) |

vLLM 0.32.0 on 127.0.0.1:8000 with the bearer token, `--generation-config vllm --seed 1234`, `VLLM_USE_FLASHINFER_SAMPLER=0`.
LoRA recipe (`ocrgate/lora_train.py`, `results/p2_1_lora.md`): r=16, alpha=32, dropout 0.05, all attention+MLP projections,
lr 1e-4, grad-accum 16, 2 epochs on the 1,572 `fit` units (impresso-nzz excluded), epoch 2 chosen on held-out train.

## Ensemble (P2-3; `ocrgate/ensemble.py`)

Pool of the edit blocks of LoRA-24B, LoRA-32B and Qwen2.5-72B; identical proposals merged (n_agree); a block is a
candidate if its full-gate score >= **t_single = 0.7563575506210327** when one rewriter proposes it and
>= **t_agree = 0.07864424213767052** when two or three do (both chosen on the `gate_heldout` units,
`frozen/agree_thresholds_3pool.json`); overlapping candidates resolved by higher score (then higher
agreement); the accepted blocks are applied to the OCR. Secondary systems: the same 3-pool with the global rule, and the
2-pool LoRA-24B + LoRA-32B with the agreement thresholds.

## Systems to report on test (`scripts/test_run_frozen_p2.sh`, refuses without `CONFIRM_TEST_RUN=1`)

Primary: **3-pool ensemble with agreement thresholds** (dev 0.0089 / 0.971). Secondary: LoRA-32B + full gate (dev 0.0110 /
0.949), LoRA-24B + full gate (0.0112 / 0.930), 3-pool global (0.0100 / 0.963), 2-pool agreement (0.0099 / 0.971); each
rewriter also without gate, with the text-only gate and with the label oracle (upper bound, not a system). Exact commands:

```
# A. LoRA rewriters (per model: serve with the adapter, run test, blocks, no gate / full gate / text-only gate / oracle)
tmux new -d -s vllm_test "cd $REPO && VLLM_USE_FLASHINFER_SAMPLER=0 MODEL=mistralai/Mistral-Small-24B-Instruct-2501 DP=1 TP=2 LORA_MODULES='lora24b-ep2=.../runs/lora/mistral24b-p2/ep2' ./serve_model.sh"
.venv/bin/python -m ocrgate.run --config configs/phase2.yaml --system zeroshot --split test --model lora24b-ep2 --tag lora24b-p2frozen
tmux new -d -s vllm_test "... MODEL=Qwen/Qwen3-32B DP=1 TP=2 LORA_MODULES='lora32b-ep2=.../runs/lora/qwen3-32b-p2/ep2' ./serve_model.sh"
.venv/bin/python -m ocrgate.run --config configs/phase2.yaml --system zeroshot --split test --model lora32b-ep2 --tag lora32b-p2frozen --send-thinking-kwarg on
# B. Qwen2.5-72B zero-shot: MODEL=Qwen/Qwen2.5-72B-Instruct DP=1 TP=2; --tag q72b-p2frozen
# per run:
.venv/bin/python -m ocrgate.blocks --config configs/phase2.yaml --run runs/<RUN>
CONFIRM_TEST_RUN=1 $VLLM_PYTHON -m ocrgate.gate_apply --config configs/phase2.yaml --run runs/<RUN> --mode none --allow-test
CONFIRM_TEST_RUN=1 $VLLM_PYTHON -m ocrgate.gate_apply --config configs/phase2.yaml --run runs/<RUN> --mode gate --gate runs/gates/p2_pooled --rule global --allow-test
CONFIRM_TEST_RUN=1 $VLLM_PYTHON -m ocrgate.gate_apply --config configs/phase2.yaml --run runs/<RUN> --mode gate --gate runs/gates/p2_pooled_nolp --rule global --allow-test
CONFIRM_TEST_RUN=1 $VLLM_PYTHON -m ocrgate.gate_apply --config configs/phase2.yaml --run runs/<RUN> --mode oracle_label --allow-test
# C. ensembles
CONFIRM_TEST_RUN=1 $VLLM_PYTHON -m ocrgate.ensemble --config configs/phase2.yaml --runs <L24> <L32> <Q72> --gate runs/gates/p2_pooled --agree-thresholds 0.7563575506210327 0.07864424213767052 --tag test-3pool-agree-p2frozen --allow-test
CONFIRM_TEST_RUN=1 $VLLM_PYTHON -m ocrgate.ensemble --config configs/phase2.yaml --runs <L24> <L32> <Q72> --gate runs/gates/p2_pooled --rule global --tag test-3pool-global-p2frozen --allow-test
CONFIRM_TEST_RUN=1 $VLLM_PYTHON -m ocrgate.ensemble --config configs/phase2.yaml --runs <L24> <L32> --gate runs/gates/p2_pooled --agree-thresholds 0.7563575506210327 0.07864424213767052 --tag test-2pool-agree-p2frozen --allow-test
```

The gate must run on the GPU (bf16), as every dev number did (Phase 1 finding: CPU fp32 flips threshold-adjacent blocks).
Estimated test cost: ~1.0 GPU-h (three TP=2 servers for 740 units each + gate inference).

## Dry-run evidence (dev only; no test file opened)

`SPLIT=dev CONFIRM_TEST_RUN=1 ONLY=C SUF=dryrun L24_RUN=... L32_RUN=... Q72_RUN=... scripts/test_run_frozen_p2.sh`
(`logs/p2_frozen_dryrun_dev.log`) re-ran part C on the existing dev runs and reproduced the dev ensembles exactly:
3-pool agreement 0.0089 / 0.9713 (`20261030T113453Z_ensemble_dev_dev-3pool-agree-dryrun` vs
`20261030T113401Z_ensemble_dev_dev-3pool-agree`), 3-pool global 0.0100 / 0.9633, 2-pool agreement 0.0099 / 0.9707.
Parts A and B are the commands that produced the dev LoRA / 72B runs (`scripts/p2_generate.sh` steps L, K, Q with
`--split dev`; the `gate_all` sequence is the one of `scripts/p2_apply_gates.sh`).

## What the test run must NOT change

The adapters, gate checkpoints, thresholds (0.3757 / 0.2967 / t_single / t_agree), prompt, decoding parameters, block
rule, conflict rule and feature set above. Any change requires a new dev cycle. **Not to be done on test:** rule
comparison, epoch choice, participant-file gating (the Phase 1 test run already covered the participants once).
