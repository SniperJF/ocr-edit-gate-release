# Rewrite, Then Veto: Rewriter-Agnostic Edit Gating for LLM-Based Post-OCR Correction

Jorge Ramón Fonseca Cacho, Department of Computer Science, University of Nevada, Las Vegas

arXiv: [arXiv:2611.05539](https://arxiv.org/abs/2611.05539)

Code for the paper *Rewrite, Then Veto: Rewriter-Agnostic Edit Gating for LLM-Based Post-OCR Correction* (HIPE-OCRepair-2026 benchmark). The method corrects
historical OCR text with LLM rewriters but only lets through the edits a small learned **edit gate** accepts; every other
character is copied from the OCR input.

1. **Rewriters.** LLMs fine-tuned with LoRA (Mistral-Small-24B, Qwen3-32B, Qwen2.5-72B; Qwen3.8-27B as a secondary 2026
   model) rewrite one transcription unit at a time (greedy decoding, thinking off).
2. **Edit blocks.** Each rewrite is aligned to the OCR and split into edit blocks.
3. **Edit gate.** Trained on labelled edit blocks from six rewriters on the train split (zero-shot Qwen3-8B, Qwen3-32B,
   Mistral-Small-24B and Qwen2.5-72B, plus LoRA-24B and LoRA-32B). A Qwen3-0.6B encoder with 15 numeric features (including the rewriter's token log-probs) scores
   P(helpful) per block. The frozen rule is one global threshold. A text-only variant needs no log-probs.
4. **Edit-pool ensemble.** Blocks from several rewriters are pooled. A block is accepted if its gate score clears the
   threshold for its agreement level (how many rewriter families proposed it). Overlaps go to the higher score.

Every model choice (epochs, thresholds, ensemble pool, agreement levels) was made on held-out parts of the **train** split
(`configs/p2_split.json`, including the 360 `gate_heldout` units). Dev was used only for reporting. The frozen
configurations are in `frozen/`, and their sha256 hash tables are next to them.

## Repository layout

| path | contents |
|---|---|
| `ocrgate/` | the package: data I/O, LLM client, rewriting systems, edit blocks, gate (model, training, application, rules), ensemble, pool search, LoRA training, synthetic noise model, official scoring wrapper, bootstrap statistics |
| `scripts/` | drivers per phase (`p2_*`, `p3_*`, `p4_*`), frozen test runners (`test_run_frozen_p{2,3,4}.sh`, which refuse to run without `CONFIRM_TEST_RUN=1`), hash checks, statistics, synthetic-data fetcher, `fetch_data.sh` |
| `configs/` | YAML configs, `p2_split.json` (unit roles of the train split: ids and hashes only, no text), held-out subsets |
| `frozen/` | `FROZEN_CONFIG_P1..P4.md`, sha256 tables of every frozen artefact, frozen ensemble/threshold/epoch JSONs |
| `tests/` | CPU-only unit tests |
| `serve_model.sh`, `run_all.sh`, `setup_server.sh` | vLLM server launcher, Phase 0 driver, one-time setup |

## Install

Two environments, as in the paper:

```bash
git clone https://github.com/SniperJF/ocr-edit-gate-release.git
cd ocr-edit-gate-release

# 1) client / scoring environment (CPU is enough): generation client, edit blocks, official scorer, tests
python3.12 -m venv .venv
.venv/bin/pip install -e ".[test]"
.venv/bin/python -m pytest -q tests/   # 4 tests need the HIPE data (see Data) and are skipped without it

# 2) GPU environment (vLLM serving, LoRA training, gate training/inference); pinned versions used for every number
python3.12 -m venv ~/vllm-env
~/vllm-env/bin/pip install -r requirements-gpu.txt     # vllm 0.32.0, transformers 5.19.0, peft 0.21.2, torch 2.13.0
~/vllm-env/bin/pip install -e .
export VLLM_ENV=~/vllm-env VLLM_PYTHON=~/vllm-env/bin/python
```

Hardware used: 2x NVIDIA H200 (141 GB). The 72B LoRA training and serving use both GPUs (bf16, TP = 2). Qwen3.8-27B runs
on one GPU.

The vLLM server binds to `127.0.0.1:8000` and requires a bearer token. Set `VLLM_API_KEY` in the shells that run
`serve_model.sh` and the clients (or point `VLLM_API_KEY_FILE` at a file holding the token). The key is never logged.

## Data (not redistributed)

This repository contains **no HIPE-OCRepair data**. Get the official data and evaluation repositories at the pinned
commits:

```bash
bash scripts/fetch_data.sh   # clones hipe-eval/HIPE-OCRepair-2026-data @ bd463f7 (tag v0.9.5) and hipe-eval/HIPE-OCRepair-2026-eval @ 47a71d8
# or use existing clones:
export HIPE_DATA_DIR=/path/to/HIPE-OCRepair-2026-data HIPE_EVAL_DIR=/path/to/HIPE-OCRepair-2026-eval
```

The datasets have their own licenses, and several are **non-commercial** (icdar2017 and impresso-snippets CC BY-NC-SA 4.0,
impresso-nzz CC BY-NC 4.0, overproof "research use only", dta19 CC BY-NC 4.0 / CC BY-SA 4.0 depending on the document;
see `NOTICE`). Test ground truth is public. Use it only for final reporting.

## System outputs

Our dev and test predictions are **not included**. They are corrected versions of the HIPE OCR text, so they inherit the
dataset licenses. overproof is "research use only", and the two sources for dta19 disagree (CC BY-NC 4.0 vs CC BY-SA 4.0,
which would require incompatible share-alike terms). No single license covers all cells, so the outputs can be regenerated
with the released code and weights instead.

## Scoring (official scorer)

Every number below comes from the official scorer (`hipe-ocrepair-scorer` 0.9.9, called through the eval repo's
`lib/score_one.py`). The overall number is the organisers' weighted mean of the per-cell `cmer_micro` (DTA cells weigh 1/3
each, the others 1). `pref` is the weighted `pref_score_cmer_macro`. Scoring one cell:

```bash
.venv/bin/python -m ocrgate.score --hypothesis <team>_hipe-ocrepair-bench_v0.9_icdar2017_v1.1_test_en_run1.jsonl
#   [--reference-dir DIR] (default $HIPE_EVAL_DIR/data/reference)  [--scorer PATH] (default $HIPE_EVAL_DIR/lib/score_one.py)  [--json]
```

Whole runs are scored automatically by `ocrgate.run`, `ocrgate.gate_apply` and `ocrgate.ensemble` (`ocrgate/score.py`).

## Model weights

The frozen gates and the released LoRA adapters are **not in this repository**. They are on the Hugging Face Hub
(https://huggingface.co/SJF7) under **CC BY-NC-SA 4.0**, because several of the training sets are
non-commercial / share-alike (see *Data* and `NOTICE`). The code in this repository is Apache-2.0. The Qwen2.5-72B adapter
is also subject to the Qwen LICENSE AGREEMENT (it ships `LICENSE-Qwen` and `NOTICE`; **Built with Qwen**). The look-2 gates
were trained on edit blocks that include Qwen2.5-72B-Instruct outputs (**Improved using Qwen**, Qwen license section 5b).
Please use the weights for non-commercial research only, as the overproof terms ask (a request, not an extra license condition).

| Hub repo | artefact (frozen path expected by the scripts) |
|---|---|
| [`SJF7/ocr-edit-gate-qwen3-0.6b-gate`](https://huggingface.co/SJF7/ocr-edit-gate-qwen3-0.6b-gate) | full gate, `runs/gates/p2_pooled/{model.safetensors,model.pt,gate.json,rules.json}` |
| [`SJF7/ocr-edit-gate-qwen3-0.6b-gate-textonly`](https://huggingface.co/SJF7/ocr-edit-gate-qwen3-0.6b-gate-textonly) | text-only gate, `runs/gates/p2_pooled_nolp/{model.safetensors,model.pt,gate.json,rules.json}` |
| [`SJF7/ocr-edit-gate-mistral-small-24b-lora`](https://huggingface.co/SJF7/ocr-edit-gate-mistral-small-24b-lora) | LoRA-24B epoch 2, `runs/lora/mistral24b-p2/ep2/` |
| [`SJF7/ocr-edit-gate-qwen3-32b-lora`](https://huggingface.co/SJF7/ocr-edit-gate-qwen3-32b-lora) | LoRA-32B epoch 2, `runs/lora/qwen3-32b-p2/ep2/` |
| [`SJF7/ocr-edit-gate-qwen2.5-72b-lora`](https://huggingface.co/SJF7/ocr-edit-gate-qwen2.5-72b-lora) | LoRA-72B epoch 2, `runs/lora/qwen25-72b-p3/ep2/` (Built with Qwen) |
| [`SJF7/ocr-edit-gate-qwen3.8-27b-lora`](https://huggingface.co/SJF7/ocr-edit-gate-qwen3.8-27b-lora) | Qwen3.8-27B LoRA epoch 1, `runs/lora/qwen38-27b-p4/ep1/` |
| [`SJF7/ocr-edit-gate-p1-qwen3-0.6b-gate`](https://huggingface.co/SJF7/ocr-edit-gate-p1-qwen3-0.6b-gate) | Phase 1 (test look 1) full gate, `runs/gates/qwen06_pooled/{model.safetensors,model.pt,gate.json}` |
| [`SJF7/ocr-edit-gate-p1-qwen3-0.6b-gate-textonly`](https://huggingface.co/SJF7/ocr-edit-gate-p1-qwen3-0.6b-gate-textonly) | Phase 1 (test look 1) text-only gate, `runs/gates/qwen06_pooled_nolp/{model.safetensors,model.pt,gate.json}` |
| [`SJF7/ocr-edit-gate-p1-qwen3-8b-lora`](https://huggingface.co/SJF7/ocr-edit-gate-p1-qwen3-8b-lora) | Phase 1 LoRA-8B (Qwen3-8B, epoch 1), `runs/lora/qwen3-8b-p1/` (served as `p1lora`) |

The Hub release is exactly nine repos (landing page: https://huggingface.co/SJF7):

- **Four edit gates, all Qwen3-0.6B:** the look-1 frozen gates (Phase 1, `p1` repos) and the look-2 retrained gates (Phase 2,
  used by the primary P3 system), each in a full (log-prob features) and a text-only version.
- **Five LoRA adapters:** Qwen3-8B LoRA (look 1), LoRA-24B (Mistral-Small-24B), LoRA-32B (Qwen3-32B), LoRA-72B (Qwen2.5-72B) and
  the Qwen3.8-27B LoRA.
- **Withheld:** the LoRA-72B-syn adapter and its synthetic training pairs. The recipe to regenerate them is below.

Each gate repo has `model.safetensors` (recommended) and the frozen `model.pt`. The safetensors file holds the identical
tensors, converted after the freeze (`scripts/gate_to_safetensors.py`). Its sha256 is in
`frozen/post_freeze_safetensors_hashes.tsv`, and `model.pt` is kept for verification against the frozen hash tables.
`ocrgate.gate.load_gate` uses `model.safetensors` when present and falls back to `model.pt`.

Put the downloaded files at the paths shown. `python scripts/test_hashes_p3.py` / `test_hashes_p4.py` check them
against `frozen/*_frozen_hashes.tsv`. The Phase 1 gate hashes are in `frozen/FROZEN_CONFIG_P1.md`; their `model.safetensors`
files are post-freeze conversions of `model.pt` like the Phase 2 ones (sha256 in `frozen/post_freeze_safetensors_hashes.tsv`).

**Not released: the LoRA-72B-syn adapter and its synthetic training data.** The synthetic pairs mix public-domain text with
ground truth from impresso-nzz (CC BY-NC 4.0). LoRA-72B-syn is a member of the primary P3 ensemble, so the primary system
can only be reproduced after regenerating it (see the paper, section on synthetic data):

```bash
python scripts/p3_fetch_clean.py fetch          # Project Gutenberg / Wikisource sources (polite, cached) -> data_synth/raw/
python scripts/p3_fetch_clean.py units          # + impresso-nzz fit-role ground truth -> data_synth/clean.jsonl
python -m ocrgate.p3_noise stats --config configs/phase2.yaml --split configs/p2_split.json --roles fit --out data_synth/noise_stats.json
python -m ocrgate.p3_noise apply --stats data_synth/noise_stats.json --clean data_synth/clean.jsonl --out data_synth/synth.jsonl   # seed 1234
MODEL=Qwen/Qwen2.5-72B-Instruct OUT=runs/lora/qwen25-72b-syn-p3 EXTRA=data_synth/synth.jsonl DEVMAP=auto scripts/p3_lora_synth_train.sh
```

The frozen adapter is epoch 1 (`runs/lora/qwen25-72b-syn-p3/ep1`). A regenerated adapter will not match the frozen sha256
because the web sources may have changed since, so expect small differences.

## Reproducing the main tables

All commands run from the repository root, with the client venv in `.venv` and `VLLM_PYTHON` pointing at the GPU env.
`serve_model.sh` starts vLLM with the settings used in the paper (`--generation-config vllm --seed 1234`; set
`VLLM_USE_FLASHINFER_SAMPLER=0`). The frozen test runners start and stop vLLM servers in `tmux`.

### PRIMARY: Phase 3 4-family, 3-level ensemble (`frozen/FROZEN_CONFIG_P3.md`)

Members: LoRA-24B + LoRA-32B + LoRA-72B + LoRA-72B-syn under the full gate. Agreement thresholds 0.7786493301391602
(n = 1), 0.3098590523004532 (n = 2), 0.011687257327139378 (n >= 3).

```bash
python scripts/test_hashes_p3.py                          # verify the frozen weights
SPLIT=dev  CONFIRM_TEST_RUN=1 scripts/test_run_frozen_p3.sh   # dev (reporting split)
SPLIT=test CONFIRM_TEST_RUN=1 scripts/test_run_frozen_p3.sh   # test: parts A (24B, 32B), B (72B + 72B-syn, one server), C (ensemble)
```

Part C on its own, given the four rewriter runs:

```bash
$VLLM_PYTHON -m ocrgate.ensemble --config configs/phase2.yaml --runs <L24_RUN> <L32_RUN> <L72_RUN> <SYN_RUN> \
  --gate runs/gates/p2_pooled --agree-thresholds 0.7786493301391602 0.3098590523004532 0.011687257327139378 \
  --families lora24b lora32b lora72b lora72bsyn --tag test-p3-4fam-L3 --allow-test
```

Single rewriter with the gate (any run):

```bash
.venv/bin/python -m ocrgate.blocks --config configs/phase2.yaml --run runs/<RUN>
$VLLM_PYTHON -m ocrgate.gate_apply --config configs/phase2.yaml --run runs/<RUN> --mode gate --gate runs/gates/p2_pooled --rule global [--allow-test]
```

### SECONDARY: Qwen3.8-27B rows and the P4 ensemble (`frozen/FROZEN_CONFIG_P4.md`)

```bash
python scripts/test_hashes_p4.py
SPLIT=test CONFIRM_TEST_RUN=1 scripts/test_run_frozen_p4.sh   # needs the P3 test runs (runs/*_zeroshot_test_lora72b{,-syn}-p3frozen) in runs/
SPLIT=dev  CONFIRM_TEST_RUN=1 L72_RUN=runs/<dev LoRA-72B run> SYN_RUN=runs/<dev LoRA-72B-syn run> scripts/test_run_frozen_p4.sh
```

The P4 ensemble (LoRA-72B + LoRA-72B-syn + Qwen3.8 zero-shot + Qwen3.8 LoRA, 2 levels: 0.6671476364135742 / 0.1225232183933258)
reuses the LoRA-72B and LoRA-72B-syn test runs of the P3 look; it is run only because it beat the P3 pool on gate_heldout
(`frozen/p4_frozen_ensemble.json`, `beats_p3`).

### Training (dev/train only)

```bash
# LoRA rewriters (recipe: r=16, alpha=32, dropout 0.05, lr 1e-4, grad-accum 16, 2 epochs, seed 1234, 1,572 fit units, impresso-nzz excluded)
$VLLM_PYTHON -m ocrgate.lora_train --config configs/phase2.yaml --model mistralai/Mistral-Small-24B-Instruct-2501 --out runs/lora/mistral24b-p2 --split configs/p2_split.json --roles fit --epochs 2 --save-each-epoch
$VLLM_PYTHON -m ocrgate.lora_train --config configs/phase2.yaml --model Qwen/Qwen3-32B --out runs/lora/qwen3-32b-p2 --split configs/p2_split.json --roles fit --epochs 2 --save-each-epoch
scripts/p3_lora72b_train.sh            # Qwen2.5-72B-Instruct, bf16 sharded over 2 GPUs (--device-map auto)
scripts/p4_lora_train.sh               # Qwen3.8-27B (Qwen3_5ForConditionalGeneration; attention + Gated-DeltaNet + MLP targets)
# rewriter generation on train / held-out / dev, epoch selection on held-out train
scripts/p2_generate.sh ; scripts/p3_lora72b_infer.sh ; scripts/p4_zs.sh ; scripts/p4_lora_infer.sh
# edit gates (full and text-only) on the pooled train blocks, then decision rules
scripts/p2_train_gates.sh
# ensemble selection on gate_heldout
scripts/p2_ensemble.sh ; scripts/p3_4_select.sh ; scripts/p4_select.sh
```

`frozen/FROZEN_CONFIG_P2.md` gives the exact gate-training command, including the six block files it pools.

## Expected numbers

Weighted cMER (lower is better) / pref, from the official scorer. **Dev** has 548 units in 7 official cells (there is no
icdar2017-fr dev file). **Test** has 590 scored units in all 8 official cells. The two are not directly comparable.

### Primary system and references, test (source: `FINAL_REPORT_TEST_P3.md`, `FINAL_REPORT_TEST_P4.md`)

| system | dev | test |
|---|---|---|
| **P3 primary ensemble (LoRA-24B + LoRA-32B + LoRA-72B + LoRA-72B-syn, 3 levels)** | 0.0084 / 0.971 | **0.0069 / 0.9282** |
| P2 3-pool ensemble (LoRA-24B + LoRA-32B + Qwen2.5-72B zero-shot) | 0.0089 / 0.971 | 0.0071 / 0.9244 |
| LoRA-72B-syn + full gate | 0.0099 / 0.964 | 0.0080 / 0.9320 |
| LoRA-72B + full gate | 0.0099 / 0.969 | 0.0085 / 0.8944 |
| LoRA-24B + full gate | 0.0112 / 0.930 | 0.0086 / 0.9169 |
| LoRA-32B + full gate | 0.0110 / 0.949 | 0.0088 / 0.8980 |
| Qwen2.5-72B zero-shot + full gate | 0.0130 / 0.821 | 0.0105 / 0.8343 |
| no correction (OCR) | 0.0275 / 0.000 | 0.0226 / 0.0000 |
| bnf-mistral run1 (published participant file, re-scored) | - | 0.0050 / 0.9000 |
| bnf-mistral run3 (published participant file, re-scored) | - | 0.0071 / 0.8737 |
| bnf-mistral run2 (published participant file, re-scored) | - | 0.0090 / 0.8739 |
| blocr run1 (published participant file, re-scored) | - | 0.0106 / 0.7028 |

Dev sources: `FINAL_REPORT_P2.md`, `FINAL_REPORT_P3.md`, `FINAL_REPORT_P4.md`, `FROZEN_CONFIG_P3.md`. Paired bootstrap on
test (`FINAL_REPORT_TEST_P3.md`): the primary ensemble beats the P2 3-pool (-0.0003, p = 0.003), bnf-mistral run2 (-0.0022)
and blocr (-0.0038). It ties bnf-mistral run3 (p = 0.49) and is worse than bnf-mistral run1 (+0.0019, p < 0.0002). It is not
state of the art on cMER.

Per-cell test values of the primary ensemble (`FINAL_REPORT_TEST_P3.md` A.2): icdar2017 en 0.0061, fr 0.0094;
impresso-snippets en 0.0063, fr 0.0055, de 0.0077; dta19 l0 0.0021, l1 0.0049, l2 0.0117.

### Secondary: Qwen3.8-27B, third test look (source: `FROZEN_CONFIG_P4.md`, `FINAL_REPORT_P4.md`, `FINAL_REPORT_TEST_P4.md`)

| system | dev | test |
|---|---|---|
| Qwen3.8-27B zero-shot, no gate | 0.0156 / 0.729 | 0.0114 / 0.7704 |
| Qwen3.8-27B zero-shot + full gate | 0.0128 / 0.813 | 0.0108 / 0.8289 |
| Qwen3.8-27B LoRA (epoch 1), no gate | 0.0094 / 0.922 | 0.0072 / 0.8928 |
| Qwen3.8-27B LoRA + full gate | 0.0089 / 0.976 | 0.0071 / 0.9250 |
| Qwen3.8-27B LoRA + text-only gate | 0.0090 / 0.978 | 0.0071 / 0.9372 |
| P4 ensemble (LoRA-72B + LoRA-72B-syn + Qwen3.8 zs + Qwen3.8 LoRA, 2 levels) | 0.0082 / 0.975 | 0.0066 / 0.9296 |

On test, the P4 ensemble is 0.0003 below the P3 primary (p = 0.018) and 0.0017 above bnf-mistral run1 (p < 0.0002).
Per cell it is not uniformly better: on dta19-l1 it is **+0.0001 worse** than the P3 primary (0.0049 -> 0.0050; unrounded
+0.000052, 95% CI [-0.00083, +0.00085], p = 0.896), and also +0.0001 on icdar2017-en (p = 0.505). Both differences are
not significant (`FINAL_REPORT_TEST_P4.md`, C.5 and the erratum). These
rows are post hoc and secondary. The P3 ensemble remains the primary system.

### GPU-hours (source: `gpu_hours_*.txt`; GPU-hours = server wall-clock x GPUs)

| phase | GPU-h |
|---|---|
| Phase 0 / 0b (feasibility) | 0.349 / 0.733 |
| Phase 1 (dev) | 5.245 |
| Test look 1 (Phase 1 frozen) | 1.096 |
| Phase 2 (dev) | 6.413 |
| Phase 3 (dev) | 16.572 |
| Test look 2 (Phase 3 frozen, primary) | 1.137 |
| Phase 4 (dev) | 3.196 |
| Test look 3 (Phase 4 frozen, secondary) | 0.180 |
| **Total, Phase 1 to test look 3** | **33.83** (`FINAL_REPORT_TEST_P4.md`; Phase 0/0b not included) |

Rough cost of one primary test run: about 1.1 GPU-h (`gpu_hours_test_p3.txt`, which also includes the secondary runs).

## Test-set usage (disclosure)

The test ground truth is public. The project scored the test split **three times**, and the paper discloses all three:

1. **Look 1**: the frozen Phase 1 gates on five rewriters and the published participant files.
2. **Look 2**: the frozen Phase 3 configuration, which is the **primary result** (0.0069 / 0.9282).
3. **Look 3**: a pre-declared, narrow look at the frozen Phase 4 configuration (Qwen3.8-27B and the P4 ensemble). It is
   **secondary** and post hoc.

Each look was one pass of a frozen configuration whose hashes were checked before the run. Nothing was selected,
re-thresholded or re-run after seeing a test number. Readers comparing with the leaderboard should count three looks.
No result was submitted to the official leaderboard.

## Licenses

- **Code:** Apache-2.0 (`LICENSE`). Third-party components and data are listed in `NOTICE`.
- **Data:** not included. Per-dataset HIPE-OCRepair licenses apply (several non-commercial).
- **Model weights (gates and adapters):** not included in this repository; on the Hugging Face Hub (https://huggingface.co/SJF7) under
  CC BY-NC-SA 4.0, because they were trained on non-commercial / share-alike data. Base models keep their own licenses
  (Qwen3-0.6B, Qwen3-8B, Qwen3-32B, Qwen3.8-27B and Mistral-Small-24B: Apache-2.0). The LoRA-72B adapter is also subject to
  the Qwen LICENSE AGREEMENT (**Built with Qwen**; `LICENSE-Qwen` and `NOTICE` in its repo), and the look-2 gates display
  **Improved using Qwen** because their training blocks include Qwen2.5-72B-Instruct outputs. LoRA-72B-syn is withheld.
- **Official scorer:** `hipe-ocrepair-scorer` (MIT) is a pip dependency. The eval repo (AGPL-3.0) is cloned at runtime and
  is not vendored.

## Citation

```bibtex
@misc{ocreditgate2026,
  title         = {Rewrite, Then Veto: Rewriter-Agnostic Edit Gating for LLM-Based Post-OCR Correction},
  author        = {Fonseca Cacho, Jorge Ram{\'o}n},
  year          = {2026},
  eprint        = {2611.05539},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

## Acknowledgements

We thank the HIPE-OCRepair-2026 organisers for the benchmark, the data and the official scorer.
