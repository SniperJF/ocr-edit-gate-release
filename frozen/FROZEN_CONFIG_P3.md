# Phase 3 FROZEN configuration for the single, later TEST run (NOT executed; test split untouched)

Frozen on 2026-10-31 after the dev-only Phase 3 (`results/FINAL_REPORT_P3.md`). Every choice below was made on the held-out part of
TRAIN (`configs/p2_split.json`: LoRA epochs on the 721 held-out units, ensemble pool / agreement levels / thresholds on the 360
`gate_heldout` units); dev was used for reporting only. Original (private) development repository; this release reproduces its code. The test split (`HIPE-OCRepair-2026-eval/data/reference/`) and the participant files (`data/systems/`) were not opened in Phase 3.
The Phase 2 frozen configuration (`frozen/FROZEN_CONFIG_P2.md`) remains valid as the fallback; Phase 3 replaces its ensemble.

## Frozen artefacts (git-ignored weights; sha256 of the files on disk, `frozen/p3_frozen_hashes.tsv`)

| role | path | size (bytes) | sha256 |
|---|---|---|---|
| **LoRA-72B adapter** (Qwen2.5-72B-Instruct, P3-1, epoch 2) | `runs/lora/qwen25-72b-p3/ep2/adapter_model.safetensors` | 842,289,128 | `5362684672139aea9f61ae8728839e426f4839684c127fd7e9b26e9c6ea8eb60` |
|  | `runs/lora/qwen25-72b-p3/ep2/adapter_config.json` | 1,181 | `79152304f54a7d2b4eb5bcc77faa887c7181adc2ed33354219cc6a99a64cc359` |
| **LoRA-72B-syn adapter** (Qwen2.5-72B-Instruct, P3-3, fit + synthetic, epoch 1) | `runs/lora/qwen25-72b-syn-p3/ep1/adapter_model.safetensors` | 842,289,128 | `f28609ec9d16c280a14d39bdd8a09c0cdf4109161fbdade61f41cfc8886cbb18` |
|  | `runs/lora/qwen25-72b-syn-p3/ep1/adapter_config.json` | 1,181 | `2a04363c41f8b5e68a48a1c7739bfe732138f2bb6347bad1d0082510600ca6d7` |
| LoRA-24B adapter (Phase 2, unchanged) | `runs/lora/mistral24b-p2/ep2/adapter_model.safetensors` | 369,698,576 | `aee8ca243dd2990d367b2192426f24dd14b76fc6a2cfd211151b493a1853107a` |
| LoRA-32B adapter (Phase 2, unchanged) | `runs/lora/qwen3-32b-p2/ep2/adapter_model.safetensors` | 536,991,984 | `4ca100cc0b8db683e263c28f28b5d9b18de5a3bf7e18351c52cf9a48e97e3472` |
| **full gate** (Phase 2, unchanged) | `runs/gates/p2_pooled/model.pt` | 1,193,273,291 | `a8d11d8782040baf8e1485559461a8e5709d6c896e14f382f850d58911bc3762` |
| full gate metadata | `runs/gates/p2_pooled/gate.json` | 2,467 | `364d5318ec549c8b28a477d9435ed9cbf5df193003ad2fbd3a5418adf95b4634` |
| full gate rules (rule `global`) | `runs/gates/p2_pooled/rules.json` | 20,462 | `ee24d0543cee178b43179184cd7d798d09f95bfdf1466f45e8444296c370e3be` |
| **ensemble configuration** (tracked) | `frozen/p3_frozen_ensemble.json` | 609 | `6a2d5841ca48e1ec12ad6eae89bf9d435da0d9bf3a4191cb4ad1a3e35a423f86` |

The LoRA-24B / LoRA-32B adapters and the gate files have the same sha256 as in `frozen/FROZEN_CONFIG_P2.md`.

## Gate (unchanged from Phase 2)

`runs/gates/p2_pooled` (Qwen3-0.6B encoder + 15 numeric features, epoch 1), rule `global`, threshold on P(helpful)
**0.37570643424987793** for the single-system secondaries; inside the ensemble the per-level thresholds below replace it. The gate
was not retrained in Phase 3 (no LoRA-72B block in its training data); it must run on the GPU in bf16, as every dev number did.

## Rewriters (all served by vLLM 0.32 on 127.0.0.1:8000 with the bearer token, `--generation-config vllm --seed 1234`, `VLLM_USE_FLASHINFER_SAMPLER=0`; Phase 0 zero-shot prompt, greedy, temperature 0, top_p 1, seed 1234, max_tokens = max(512, 0.6 x input chars), fallback to OCR on empty/truncated/failed; logprobs on)

| family | rewriter | serving | thinking kwarg |
|---|---|---|---|
| `lora24b` | `mistralai/Mistral-Small-24B-Instruct-2501` + `runs/lora/mistral24b-p2/ep2`, served as `lora24b-ep2` | TP=2, `--enable-lora --max-lora-rank 16`, mistral tokenizer mode | none |
| `lora32b` | `Qwen/Qwen3-32B` + `runs/lora/qwen3-32b-p2/ep2`, served as `lora32b-ep2` | TP=2, LoRA | `--send-thinking-kwarg on` (thinking off) |
| `lora72b` | `Qwen/Qwen2.5-72B-Instruct` + `runs/lora/qwen25-72b-p3/ep2`, served as `lora72b-ep2` | TP=2, LoRA (both 72B adapters in one server, `--max-loras 2`) | none (Qwen2.5 has no thinking switch; auto = off) |
| `lora72bsyn` | `Qwen/Qwen2.5-72B-Instruct` + `runs/lora/qwen25-72b-syn-p3/ep1`, served as `lora72b-syn-ep1` | same server | none |

LoRA recipe (`ocrgate/lora_train.py`): r=16, alpha=32, dropout 0.05, all attention+MLP projections, lr 1e-4, grad-accum 16, 2 epochs
on the 1,572 `fit` units (impresso-nzz excluded), seed 1234; the 72B models in bf16 sharded over both H200s (`--device-map auto`);
`lora72bsyn` additionally trained on the 1,520 synthetic pairs of `data_synth/synth.jsonl` (`--extra-jsonl`; regenerable with
`scripts/p3_fetch_clean.py` + `ocrgate.p3_noise`, seed 1234; `results/p3_3_synth.md`). Epochs chosen on the 721 held-out train
units: `lora72b` epoch 2 (0.0108 / 0.865), `lora72bsyn` epoch 1 (0.0106 / 0.890). The Qwen2.5-72B zero-shot rewriter of the P2 pool is
**not** part of the P3 ensemble (every pool containing it lost on gate_heldout, `results/p3_4_candidates.tsv`).

## Ensemble (PRIMARY system; `ocrgate/ensemble.py`, `frozen/p3_frozen_ensemble.json`)

Pool of the edit blocks of the four rewriters above; identical proposals merged; agreement n = number of distinct **families**
proposing a block (`--families lora24b lora32b lora72b lora72bsyn`); a block is a candidate iff its full-gate score >= the threshold of
its agreement level: **n = 1: 0.7786493301391602, n = 2: 0.3098590523004532, n >= 3: 0.011687257327139378** (3 levels); overlapping
candidates resolved by higher score (then higher agreement); accepted blocks applied to the OCR. Selection (`scripts/p3_4_select.sh`):
18 candidate configurations (9 pools x 2 / 3 levels), per-level thresholds by labelled gain on gate_heldout (as in Phase 2), ranked
by the official weighted cMER of the actual ensemble on the 360 gate_heldout units (tie: pref) - this pool: **0.0076 / 0.883**
(P2 frozen ensemble on the same units: 0.0081 / 0.853). Dev (reporting): **0.0084 / 0.971** (run `20261031T082249Z_ensemble_dev_dev-p3-frozen`).

## Systems to report on test (`scripts/test_run_frozen_p3.sh`, refuses without `CONFIRM_TEST_RUN=1`)

Primary: **4-family ensemble, 3 agreement levels** (dev 0.0084 / 0.971). Secondaries (single rewriters): each of the four rewriters
without gate, with the full gate (global 0.3757), with the text-only gate (`runs/gates/p2_pooled_nolp`, 0.2967) and with the label
oracle (upper bound, not a system); dev references: LoRA-72B + full gate 0.0099 / 0.969, LoRA-72B-syn + full gate 0.0099 / 0.964.
Optional comparison: the P2 frozen 3-pool ensemble if a Qwen2.5-72B zero-shot test run is made (part B of `test_run_frozen_p2.sh`;
+0.6 GPU-h); it is not required by Phase 3. Exact commands:

```
# A. Phase 2 LoRA rewriters (per model: serve with the adapter TP=2, run test, blocks, none / full gate / text-only gate / oracle)
tmux new -d -s vllm_test "cd $REPO && VLLM_USE_FLASHINFER_SAMPLER=0 MODEL=mistralai/Mistral-Small-24B-Instruct-2501 DP=1 TP=2 LORA_MODULES='lora24b-ep2=.../runs/lora/mistral24b-p2/ep2' ./serve_model.sh"
.venv/bin/python -m ocrgate.run --config configs/phase2.yaml --system zeroshot --split test --model lora24b-ep2 --tag lora24b-p3frozen
tmux new -d -s vllm_test "... MODEL=Qwen/Qwen3-32B DP=1 TP=2 LORA_MODULES='lora32b-ep2=.../runs/lora/qwen3-32b-p2/ep2' ./serve_model.sh"
.venv/bin/python -m ocrgate.run --config configs/phase2.yaml --system zeroshot --split test --model lora32b-ep2 --tag lora32b-p3frozen --send-thinking-kwarg on
# B. Phase 3 Qwen2.5-72B LoRA rewriters (one server, two adapters)
tmux new -d -s vllm_test "... MODEL=Qwen/Qwen2.5-72B-Instruct DP=1 TP=2 LORA_MODULES='lora72b-ep2=.../runs/lora/qwen25-72b-p3/ep2 lora72b-syn-ep1=.../runs/lora/qwen25-72b-syn-p3/ep1' ./serve_model.sh"
.venv/bin/python -m ocrgate.run --config configs/phase2.yaml --system zeroshot --split test --model lora72b-ep2 --tag lora72b-p3frozen
.venv/bin/python -m ocrgate.run --config configs/phase2.yaml --system zeroshot --split test --model lora72b-syn-ep1 --tag lora72b-syn-p3frozen
# per run:
.venv/bin/python -m ocrgate.blocks --config configs/phase2.yaml --run runs/<RUN>
CONFIRM_TEST_RUN=1 $VLLM_PYTHON -m ocrgate.gate_apply --config configs/phase2.yaml --run runs/<RUN> --mode none --allow-test
CONFIRM_TEST_RUN=1 $VLLM_PYTHON -m ocrgate.gate_apply --config configs/phase2.yaml --run runs/<RUN> --mode gate --gate runs/gates/p2_pooled --rule global --allow-test
CONFIRM_TEST_RUN=1 $VLLM_PYTHON -m ocrgate.gate_apply --config configs/phase2.yaml --run runs/<RUN> --mode gate --gate runs/gates/p2_pooled_nolp --rule global --allow-test
CONFIRM_TEST_RUN=1 $VLLM_PYTHON -m ocrgate.gate_apply --config configs/phase2.yaml --run runs/<RUN> --mode oracle_label --allow-test
# C. PRIMARY ensemble
CONFIRM_TEST_RUN=1 $VLLM_PYTHON -m ocrgate.ensemble --config configs/phase2.yaml --runs <L24> <L32> <L72> <SYN> --gate runs/gates/p2_pooled --agree-thresholds 0.7786493301391602 0.3098590523004532 0.011687257327139378 --families lora24b lora32b lora72b lora72bsyn --tag test-p3-4fam-L3-p3frozen --allow-test
```

Estimated test cost: ~1.6 GPU-h (three TP=2 servers for 740 units each, the 72B server running two adapters, + gate inference).

## Dry-run evidence (dev only; no test file opened)

`SPLIT=dev CONFIRM_TEST_RUN=1 ONLY=C SUF=dryrun L24_RUN=... L32_RUN=... L72_RUN=... SYN_RUN=... Q72_RUN=... scripts/test_run_frozen_p3.sh`
(`logs/p3_frozen_dryrun_dev.log`) re-ran part C on the existing dev runs and reproduced the dev ensembles exactly:
P3 primary 0.0084 / 0.9708 (`20261031T082350Z_ensemble_dev_dev-p3-4fam-L3-dryrun` vs `20261031T082249Z_ensemble_dev_dev-p3-frozen`),
P2 3-pool 0.0089 / 0.9713 (vs `20261030T113401Z_ensemble_dev_dev-3pool-agree`). Without `CONFIRM_TEST_RUN=1` the script exits 2.
Parts A and B are the commands that produced the dev LoRA runs (`scripts/p2_generate.sh` L/K, `scripts/p3_lora72b_infer.sh`).

## What the test run must NOT change

The four adapters, the gate checkpoint, the three agreement thresholds, the family assignment, the prompt, decoding parameters,
block rule, conflict rule and feature set. Any change requires a new dev cycle. **Not to be done on test:** pool or level
re-selection, epoch choice, sampled candidates, participant-file gating.
