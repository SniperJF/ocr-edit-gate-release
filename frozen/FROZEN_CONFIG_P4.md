# Phase 4 FROZEN configuration for the pre-declared THIRD TEST look (SECONDARY systems only; executed once by `scripts/test_chain_p4.sh`)

Frozen on 2026-10-31 after the dev-only Phase 4 (`results/FINAL_REPORT_P4.md`). **The paper's primary system remains the Phase 3
ensemble (`frozen/FROZEN_CONFIG_P3.md`, test 0.0069 / 0.928); nothing here replaces it. Every P4 test number is secondary,
post-hoc and part of a third test look.** Every choice below was made on the held-out part of TRAIN (`configs/p2_split.json`:
LoRA epoch on the 719-unit held-out subset, ensemble pool / levels / thresholds on the 360 `gate_heldout` units); dev was used
for reporting only. Original (private) development repository; this release reproduces its code. No test file was opened in Phase 4.

## Frozen artefacts (sha256 of the files on disk, `frozen/p4_frozen_hashes.tsv`; P2/P3 artefacts unchanged, their tables apply)

| role | path | size (bytes) | sha256 |
|---|---|---|---|
| **Qwen3.8-27B LoRA adapter** (P4-c, epoch 1, chosen on held-out) | `runs/lora/qwen38-27b-p4/ep1/adapter_model.safetensors` | 467,062,560 | `f19eb627e39dc4cd99d73fcfe7d7323190666dd6859b71223cdde6b1652a6c1d` |
|  | `runs/lora/qwen38-27b-p4/ep1/adapter_config.json` | 1,258 | `4e39d975b31bf9bfbe2751ca70cb6c2c5ddbf70bca819a7c6f6ff33e3baebe16` |
| **P4 ensemble configuration** (tracked) | `frozen/p4_frozen_ensemble.json` | 1,635 | `48da356b36a8d9b54aa6de2cfadefb503ae1f0c0b43d2428b03192d5339b031f` |
| epoch selection (tracked) | `frozen/p4_epoch_selection.json` | 755 | `5360784eda7f066a27afa4b84e10f1d071bf4c97fe2213459b4b0602fa8f25e8` |
| full gate (Phase 2, UNCHANGED) | `runs/gates/p2_pooled/{model.pt,gate.json,rules.json}` | | as in `frozen/p2_frozen_hashes.tsv` / `FROZEN_CONFIG_P3.md` |
| text-only gate (Phase 2, UNCHANGED) | `runs/gates/p2_pooled_nolp/{model.pt,gate.json,rules.json}` | | as in `frozen/p2_frozen_hashes.tsv` |
| LoRA-72B / LoRA-72B-syn adapters (Phase 3, members of the P4 ensemble) | `runs/lora/qwen25-72b-p3/ep2`, `runs/lora/qwen25-72b-syn-p3/ep1` | | as in `frozen/p3_frozen_hashes.tsv` |

Base model: `Qwen/Qwen3.8-27B` (HF snapshot `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, 18 safetensors shards, 55,563,006,776 bytes,
Apache-2.0; `results/p4_feasibility.md`). `scripts/test_hashes_p4.py` recomputes every hash above plus the P2/P3 tables and checks
the stored thresholds / members / epoch against `frozen/p4_frozen_hashes_expect.json` before the run; any mismatch stops the chain.

## Gate (unchanged since Phase 2)

`runs/gates/p2_pooled` rule `global`, threshold on P(helpful) **0.37570643424987793** (full gate; 15 numeric features incl.
log-probs + Qwen3-0.6B text encoder); `runs/gates/p2_pooled_nolp` rule `global` **0.29665425419807434** (text-only gate). Not
retrained, not re-thresholded: no Qwen3.8 block was ever in the gate's training data. GPU, bf16, as every dev number.

## Rewriters (vLLM 0.32 on 127.0.0.1:8000 with the bearer token; `serve_model.sh` with `EXTRA_ARGS='--language-model-only'`,
`--generation-config vllm --seed 1234`, `VLLM_USE_FLASHINFER_SAMPLER=0`; Phase 0 zero-shot prompt, greedy, temperature 0, top_p 1,
seed 1234, max_tokens = max(512, 0.6 x input chars), fallback to OCR on empty/truncated/failed; logprobs on; thinking OFF via
`chat_template_kwargs {"enable_thinking": false}`)

| family | rewriter | serving | thinking kwarg |
|---|---|---|---|
| `qwen38zs` | `Qwen/Qwen3.8-27B` zero-shot (Qwen3_5ForConditionalGeneration, text only) | ONE server, DP=1 TP=1 (one H200), `--enable-lora --max-lora-rank 16` with the adapter below attached; requests with `model = Qwen/Qwen3.8-27B` | auto (model name contains `qwen3`) |
| `qwen38lora` | `Qwen/Qwen3.8-27B` + `runs/lora/qwen38-27b-p4/ep1`, served as `qwen38-lora-ep1` | same server | `--send-thinking-kwarg on` |

LoRA recipe (`ocrgate/lora_train.py --model-class Qwen3_5ForConditionalGeneration --target-modules q_proj,k_proj,v_proj,o_proj,
in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj,gate_proj,up_proj,down_proj`): r=16, alpha=32, dropout 0.05, lr 1e-4,
grad-accum 16, 2 epochs on the 1,572 `fit` units (impresso-nzz excluded; 1 unit skipped as too long), seed 1234, bf16 on one H200
(no sharding), 197 optimizer steps, 4,592 s. **Epoch 1** chosen on the 719 held-out train units (0.0088 / 0.882; epoch 2
0.0089 / 0.869). Dev (reporting): ungated 0.0094 / 0.922, + full gate 0.0089 / 0.976, + text-only gate 0.0090 / 0.978, label
oracle 0.0061. Zero-shot dev: ungated 0.0156 / 0.729, + full gate 0.0128 / 0.813, + text-only gate 0.0130 / 0.810, oracle 0.0107.

## P4 ensemble (SECONDARY; runs on test ONLY because it beat the P3 pool on gate_heldout; `frozen/p4_frozen_ensemble.json`)

Pool of the edit blocks of **LoRA-72B + LoRA-72B-syn + Qwen3.8 zero-shot + Qwen3.8 LoRA** (families `lora72b lora72bsyn qwen38zs
qwen38lora`), **2 agreement levels**: a block is a candidate iff its full-gate score >= **0.6671476364135742** (n = 1 family) or
**0.1225232183933258** (n >= 2); overlapping candidates resolved by higher score (then higher agreement); accepted blocks applied
to the OCR (`ocrgate/ensemble.py`). Selection (`scripts/p4_select.sh`, protocol declared in `results/p4_feasibility.md` before any
number was read): 13 declared pools x {2, 3} levels, thresholds per level by labelled gain on gate_heldout (`ocrgate.p3_pool`),
ranked by the unrounded official weighted cMER of the actual ensemble on the 360 gate_heldout units (tie: pref). Winner
**0.007294 / 0.846** vs the P3 frozen pool re-run in the same batch **0.007578 / 0.883** (= the P3 report's 0.0076 / 0.8832) ->
`beats_p3 = true` (the 3-level variant of the same pool: 0.007300). Dev (reporting): **0.0082 / 0.975**
(`20261031T185341Z_ensemble_dev_dev-p4e-lora72b+lora72bsyn+qwen38zs+qwen38lora-L2` = `20261031T185416Z_ensemble_dev_dev-p4-best`);
P3 frozen ensemble dev 0.0084 / 0.971 (Δ −0.0002, p = 0.65, n.s.). On test the two P3 members are the frozen P3 test runs of look 2
(`runs/*_zeroshot_test_lora72b-p3frozen`, `*_lora72b-syn-p3frozen`); they are not regenerated.

## Systems to run on test (`scripts/test_run_frozen_p4.sh`, refuses without `CONFIRM_TEST_RUN=1`; ONE pass via `scripts/test_chain_p4.sh`)

Part A (one server): Qwen3.8-27B zero-shot {no gate, full gate, text-only gate, label oracle} and Qwen3.8-LoRA epoch 1 {same four}.
Part B: the P4 ensemble above (only because `beats_p3` is true). Nothing else. Comparisons (`scripts/test_stats_p4.py`, paired
bootstrap 10,000 / seed 1234): each of {zs + full gate, LoRA + full gate, P4 ensemble} vs the P3 primary ensemble (look 2), vs
Qwen2.5-72B + P1 gate (test run 1), vs no-correction, vs bnf-mistral run1 and run3 (ungated participant files, test run 1 scores);
gate-vs-no-gate for both rewriters; declared side-by-sides (LoRA vs zs; vs Qwen2.5-72B zs, LoRA-32B, LoRA-72B of look 2).
Estimated test cost ~0.45 GPU-h (one TP=1 server for 2 x 740 units + gate inference + ensemble).

```
# A. one server, both rewriters
tmux new -d -s vllm_test "cd $REPO && VLLM_USE_FLASHINFER_SAMPLER=0 MODEL=Qwen/Qwen3.8-27B DP=1 TP=1 LORA_MODULES='qwen38-lora-ep1=.../runs/lora/qwen38-27b-p4/ep1' EXTRA_ARGS='--language-model-only' ./serve_model.sh"
.venv/bin/python -m ocrgate.run --config configs/phase2.yaml --system zeroshot --split test --model Qwen/Qwen3.8-27B --tag qwen38-zs-p4frozen
.venv/bin/python -m ocrgate.run --config configs/phase2.yaml --system zeroshot --split test --model qwen38-lora-ep1 --tag qwen38-lora-p4frozen --send-thinking-kwarg on
# per run: blocks; gate_apply --mode none / --mode gate --gate runs/gates/p2_pooled --rule global / --gate runs/gates/p2_pooled_nolp --rule global / --mode oracle_label (all --allow-test)
# B. P4 ensemble
CONFIRM_TEST_RUN=1 $VLLM_PYTHON -m ocrgate.ensemble --config configs/phase2.yaml --runs <L72 test> <SYN test> <ZS test> <LORA test> --gate runs/gates/p2_pooled --agree-thresholds 0.6671476364135742 0.1225232183933258 --families lora72b lora72bsyn qwen38zs qwen38lora --tag test-p4-ens-p4frozen --allow-test
```

## Dry-run evidence (dev only; no test file opened)

`SPLIT=dev CONFIRM_TEST_RUN=1 ONLY=B SUF=dryrun ZS_RUN=runs/20261031T165529Z_zeroshot_dev_qwen38-zs LORA_RUN=runs/20261031T181336Z_zeroshot_dev_qwen38-lora-ep1 scripts/test_run_frozen_p4.sh`
(`logs/p4_frozen_dryrun_dev.log`) reproduced the dev ensemble exactly: 0.0082 / 0.9748, pooled 33,506, kept 23,730,
conflicts 2,071 (`20261031T185548Z_ensemble_dev_dev-p4-ens-dryrun` vs `20261031T185416Z_ensemble_dev_dev-p4-best`). Part A's commands
are the ones that produced the dev runs (`scripts/p4_zs.sh`, `scripts/p4_lora_infer.sh`: same serve flags, prompt, decoding; the dev
zero-shot pass was served with the P4-a smoke adapter attached instead of the frozen one - the base model's requests do not use an
adapter either way). Without `CONFIRM_TEST_RUN=1` the script exits 2.

## What the test run must NOT change

The adapter and its epoch, the gate checkpoints and thresholds, the two agreement thresholds, the family assignment, the prompt,
decoding parameters, block rule, conflict rule, feature set, the `--language-model-only` / thinking-off serving. **Not to be done on
test:** pool or level re-selection, epoch choice, gate retraining, participant-file gating, any fourth look.
