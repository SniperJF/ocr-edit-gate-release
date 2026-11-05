# Phase 1 FROZEN configuration for the single, later TEST run (NOT executed; test split untouched)

Frozen on 2026-10-30 after the dev experiments in `results/p1a_dev.md`, `results/p1_transfer.md`, `results/p1_stats.md`.
Selection rule for the gates: highest held-out-TRAIN ROC-AUC among the Qwen3-0.6B gates (no dev number was used to
choose); the pooled gates have the highest held-out AUC of their kind (full 0.952, text-only 0.949) and are also the gates
applied to the two new rewriter families without retraining. Original (private) development repository; this release reproduces its code.

## Gate checkpoints (git-ignored weights; sha256 of the files as they are on disk)

| role | directory | model.pt sha256 | gate.json sha256 | size (bytes) |
|---|---|---|---|---|
| **full gate** (our rewriters; uses the rewriter's token logprobs) | `runs/gates/qwen06_pooled` | `53d5335db289d1ce44362b5428d662910a6a90078554eabd7112907c54a1e906` | `bb2f41cb5a384dfdde0e467ae65d0323335c5ceca0053086acd73ef5f0da6ccb` | 1,193,273,291 |
| **text-only gate** (any system's output, incl. participants') | `runs/gates/qwen06_pooled_nolp` | `30d751dff9d0eb3424c3c3fd722ff13a1327e9662183a3d46cfa4ffea1f5406a` | `50c163f4fd9b540c0f75572f9279812be146e6fb9999c9821b3d30eebffc30c2` | 1,193,273,291 |

Both: encoder `Qwen/Qwen3-0.6B` (HF snapshot `c1899de289a04d12100db370d81485cdf75e47ca`), last-token pooling, bf16; head =
Linear(1024 + 15 -> 256) + GELU + Dropout(0.1) + Linear(256 -> 1); `max_len` 256 tokens; trained 3 epochs (batch 32, lr 3e-5
encoder / 1e-3 head, AdamW wd 0.01, 5% warm-up, linear decay, seed 1234) on the pooled zero-shot Qwen3-8B + Qwen3-32B TRAIN
blocks (130,165 fit / 30,077 held-out labelled blocks; the checkpoint is **epoch 1**, the held-out-best epoch for both).
`gate.json` holds the feature standardisation (mean/std) and the threshold. Training command:
`scripts/p1_train_gates.sh`-equivalent single calls, exactly:

```
$VLLM_PYTHON -m ocrgate.gate_train --blocks runs/20261030T010915Z_zeroshot_train_8b-lp/blocks.jsonl runs/20261030T012216Z_zeroshot_train_32b-lp/blocks.jsonl --encoder Qwen/Qwen3-0.6B --name qwen06_pooled --epochs 3 --lr 3e-5 --max-len 256
$VLLM_PYTHON -m ocrgate.gate_train --blocks runs/20261030T010915Z_zeroshot_train_8b-lp/blocks.jsonl runs/20261030T012216Z_zeroshot_train_32b-lp/blocks.jsonl --encoder Qwen/Qwen3-0.6B --name qwen06_pooled_nolp --no-lp --epochs 3 --lr 3e-5 --max-len 256
```

## Decision thresholds (single global value per gate; chosen on held-out TRAIN as the maximiser of the total labelled gain of kept blocks)

| gate | threshold on P(helpful) | kept block if |
|---|---|---|
| full `qwen06_pooled` | **0.5064541101455688** | score >= threshold |
| text-only `qwen06_pooled_nolp` | **0.4989624172449112** | score >= threshold |

No per-language, per-cell or per-rewriter thresholds.

## Feature set (`ocrgate/gate.py`, `NUM_FEATURES`, in order)

Text (both gates): `[<lang>|<dataset>] OCR: <up to 60 OCR chars left>⟦<OCR block>⟧<up to 60 OCR chars right>\nFIX: ⟦<proposed block>⟧`
where blocks are maximal runs of adjacent character edits between OCR and rewrite (`edits.edit_blocks`, rapidfuzz
Levenshtein editops), and `<lang>`/`<dataset>` come from `document_metadata.language` / `primary_dataset_name`.

Numeric (standardised with the stored mean/std): `lp_mean, lp_min, lp_sum, lp_n, lp_first, lp_unit_mean` (rewriter token
logprob statistics over the block; **all six are identically 0 in the text-only gate**), `len_ocr, len_new, len_diff,
log_n_blocks_unit, log_len_unit, frac_changed_unit, is_insert, is_delete` (from OCR + rewrite text only), `lp_missing`
(1 if no logprobs; 0 in the text-only gate by construction, since --no-lp zeroes it).

Output = OCR text with the kept blocks applied (`edits.apply_blocks`); rejected blocks revert to the OCR text.

## Rewriters (all zero-shot, same prompt `prompts.zeroshot_messages`, greedy, temperature 0, top_p 1, seed 1234, max_tokens =
max(512, 0.6 x input chars), fallback to the OCR input on empty/truncated/failed completion; served by vLLM 0.32.0 on
127.0.0.1:8000 with the bearer token, `--generation-config vllm --seed 1234`, `VLLM_USE_FLASHINFER_SAMPLER=0`)

| rewriter | HF model | serving | thinking | dev no gate -> full gate -> text-only gate (cMER / pref) |
|---|---|---|---|---|
| Qwen3-8B | `Qwen/Qwen3-8B` | DP=2 TP=1 | off (`enable_thinking=false`) | 0.0245 / 0.461 -> 0.0213 / 0.677 -> 0.0211 / 0.679 |
| Qwen3-32B | `Qwen/Qwen3-32B` | DP=2 TP=1 | off | 0.0220 / 0.600 -> 0.0160 / 0.810 -> 0.0161 / 0.811 |
| Mistral-Small-24B-Instruct-2501 | `mistralai/Mistral-Small-24B-Instruct-2501` | DP=2 TP=1 (vLLM mistral tokenizer mode; no chat_template_kwargs) | n/a | 0.0251 / 0.486 -> 0.0133 / 0.896 -> 0.0138 / 0.837 |
| Qwen2.5-72B-Instruct | `Qwen/Qwen2.5-72B-Instruct` | DP=1 TP=2 bf16 | n/a | 0.0216 / 0.563 -> 0.0134 / 0.833 -> 0.0133 / 0.815 |
| Qwen3-8B + LoRA (P1-b, optional) | `Qwen/Qwen3-8B` + adapter `runs/lora/qwen3-8b-p1` (served as `p1lora`) | DP=1, `--enable-lora` | off | 0.0176 / 0.677 -> 0.0143 / 0.898 -> (not run) |

Recommended primary systems for the test run: **Mistral-Small-24B-Instruct + full gate** (dev cMER 0.0133 / pref 0.896,
the best zero-shot result) and **Qwen2.5-72B-Instruct + text-only gate** (0.0133 / 0.815), with **Qwen3-32B + full gate**
(0.0160 / 0.810) as the all-Qwen3 configuration; the text-only gate is the one applied to participant outputs. All four
zero-shot rewriters are run with and without both gates in part A, so the test tables do not depend on this choice. The LoRA rewriter is optional (it was trained on train only, but
a gate trained on its own outputs does not exist; the frozen gates apply to it unchanged).

## Exact command lines of the test run (`scripts/test_run_frozen.sh`; refuses to run without `CONFIRM_TEST_RUN=1`)

Part A, per rewriter (MODEL DP TP TAG as in the table; servers started with the env vars *inside* the tmux string):

```
tmux new -d -s vllm_test "cd $REPO && VLLM_USE_FLASHINFER_SAMPLER=0 MODEL=<MODEL> DP=<DP> TP=<TP> ./serve_model.sh"
.venv/bin/python -m ocrgate.run --config configs/phase1.yaml --system zeroshot --split test --model <MODEL> --tag <TAG>-frozen
tmux kill-session -t vllm_test
.venv/bin/python -m ocrgate.blocks --config configs/phase1.yaml --run runs/<RUN>
CONFIRM_TEST_RUN=1 $VLLM_PYTHON -m ocrgate.gate_apply --run runs/<RUN> --mode none --allow-test                       # rewriter, no gate
CONFIRM_TEST_RUN=1 $VLLM_PYTHON -m ocrgate.gate_apply --run runs/<RUN> --mode gate --gate runs/gates/qwen06_pooled --allow-test
CONFIRM_TEST_RUN=1 $VLLM_PYTHON -m ocrgate.gate_apply --run runs/<RUN> --mode gate --gate runs/gates/qwen06_pooled_nolp --allow-test
CONFIRM_TEST_RUN=1 $VLLM_PYTHON -m ocrgate.gate_apply --run runs/<RUN> --mode oracle_label --allow-test             # upper bound only
```

Part B, the frozen text-only gate on every participant's public test output (bnf-mistral run1-3, blocr, l3i, Zakaria-ENSIAS,
baseline-no-correction; files `HIPE-OCRepair-2026-eval/data/systems/<team>_hipe-ocrepair-bench_v0.9_<cell>_masked-test_<lang>_run<N>.jsonl`,
references `HIPE-OCRepair-2026-eval/data/reference/hipe-ocrepair-bench_v0.9_<cell>_test_<lang>.jsonl`), each scored
ungated and gated with the official scorer:

```
$VLLM_PYTHON -m ocrgate.gate_external --gate runs/gates/qwen06_pooled_nolp --submission <SUB> --reference <REF> --out-dir runs/external_gated_test --team-suffix tgate
.venv/bin/python -I HIPE-OCRepair-2026-eval/lib/score_one.py --hypothesis runs/external_gated_test/<team>-tgate_<rest>.jsonl --reference-dir HIPE-OCRepair-2026-eval/data/reference --output ... --log-file ...
.venv/bin/python -I HIPE-OCRepair-2026-eval/lib/score_one.py --hypothesis <SUB> --reference-dir HIPE-OCRepair-2026-eval/data/reference --output ... --log-file ...
```

`gate_external` reads only `document_metadata`, `ocr_hypothesis` and `ocr_postcorrection_output` of the submission and
`document_metadata` + `ocr_hypothesis` of the reference (never `ground_truth`); outputs equal to `"None"` or missing are
replaced by the OCR input, as the scorer does.

## Dry-run evidence (dev only; no test file was opened)

- `SPLIT=dev ONLY=B DEVICE=cpu CONFIRM_TEST_RUN=1 scripts/test_run_frozen.sh` ran part B over all 8 dev prediction files
  of the 8B no-gate run as stand-ins for participant submissions (`logs/p1_frozen_dryrun_dev.log`,
  `runs/external_gated_dev/scores/`: 8 gated + 8 ungated official-scorer JSONs). Gated per-cell scores equal those of
  `gate_apply` with the same gate on the same run (`20261030T053325Z_zeroshot_gate_dev_8b-lp_qwen06_pooled_nolp`) in 6 of 8
  cells; the two DTA cells differed in the last decimal (dta19-l0 pref 0.14 vs 0.13, dta19-l2 cMER 0.0428 vs 0.0427) because
  that dry-run evaluated the gate on the CPU in fp32 while `gate_apply` used the GPU in bf16, which flips a few blocks whose
  score sits at the threshold. Re-running `gate_external` for those two cells on the GPU (`--device cuda`) reproduces
  `gate_apply` exactly (dta19-l0 0.0044 / 0.13, dta19-l2 0.0427 / 0.83). **The test run must therefore use the GPU path
  (`DEVICE=cuda`, the default), as all dev numbers did.**
- Ungated stand-in scores from the same dry-run equal the no-gate run's official scores in all 8 cells (e.g. icdar2017-en
  0.0288 / 0.6649, dta19-l0 0.0077 / −0.53), confirming the file-name -> reference resolution of `score_one.py`.
- Part A commands are the ones that produced every dev run in `results/runs_p1.json` (with `--split dev`); the Mistral and
  Qwen2.5-72B serving settings were exercised on dev on 2026-10-30 (segments 11-12 of `results/gpu_hours_p1.txt`).

## What the test run must NOT change

The gate checkpoints, thresholds, prompt, decoding parameters, block rule and feature set above. Any change requires a
new dev cycle. The test references (`data/reference/`) and participant files (`data/systems/`) were not read in Phase 1.
