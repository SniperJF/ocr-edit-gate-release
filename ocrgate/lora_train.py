"""Phase 1 P1-b / Phase 2 P2-1: LoRA-tune a zero-shot rewriter on the TRAIN split (OCR -> ground truth).

    python -m ocrgate.lora_train --config configs/phase1.yaml --out runs/lora/qwen3-8b-p1 [--epochs 1] ...
    python -m ocrgate.lora_train --model mistralai/Mistral-Small-24B-Instruct-2501 --out runs/lora/mistral24b-p2
        --split configs/p2_split.json --roles fit --epochs 2 --save-each-epoch          (Phase 2)

Phase 2 additions: --split/--roles restrict training to the units whose role in configs/p2_split.json is listed (the
~70% LoRA-fit part; held-out and dev-overlap units are never seen); --save-each-epoch writes the adapter after every
epoch to <out>/ep<N> (the epoch is then selected on the held-out part of train, never on dev). Tokenisation goes through
`apply_chat_template(tokenize=True)` so that Mistral models use the mistral_common tokenizer (the one vLLM serves
them with); the answer is the gold unit + EOS (Qwen3: `<|im_end|>\n` as in Phase 1, verified identical to the Phase 1
string path; Mistral: `</s>`, verified equal to mistral_common's own assistant-turn encoding).

Each training example is exactly the Phase 0 zero-shot chat (system + user prompt from `prompts.zeroshot_messages`,
thinking disabled) with the ground-truth transcription unit as the assistant answer; the loss is taken on the
answer tokens only. Units whose chat exceeds --max-tokens tokens are skipped (counted). LoRA (peft) on all
attention and MLP projections, bf16, gradient checkpointing, one GPU. Nothing from dev or test is read.
The adapter is saved in --out and can be served by vLLM with `--enable-lora --lora-modules name=path`.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from .data import ROOT, load_config, split_cells
from .prompts import zeroshot_messages
from .score import select_refs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--exclude-cells", default="impresso-nzz", help="comma-separated substrings of cell labels to skip")
    ap.add_argument("--split", default=None, help="Phase 2: configs/p2_split.json (unit roles)")
    ap.add_argument("--roles", default="fit", help="comma-separated roles of --split to train on")
    ap.add_argument("--save-each-epoch", action="store_true", help="save the adapter after every epoch to <out>/ep<N>")
    ap.add_argument("--max-examples", type=int, default=0, help="debug: train on the first N examples only")
    ap.add_argument("--device-map", default="0", help="Phase 3: 'auto' shards the base model over all visible GPUs (bf16 LoRA on a 72B base needs two 141 GB GPUs); default = single GPU 0")
    ap.add_argument("--extra-jsonl", default=None, help="Phase 3 (P3-3): jsonl of extra training pairs {meta, ocr, gt} (synthetic noise-matched data), mixed into the fit examples")
    ap.add_argument("--model-class", default="AutoModelForCausalLM",
                    help="Phase 4: transformers class to load the base with (Qwen3_5ForConditionalGeneration for Qwen3.8-27B, so that the adapter's "
                         "weight names carry the model.language_model. prefix that vLLM's LoRA loader maps for that architecture); text-only input in any case")
    ap.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                    help="Phase 4: comma-separated LoRA target module names (default = the P1-P3 attention+MLP set; Qwen3.8 adds the Gated DeltaNet "
                         "projections in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj)")
    args = ap.parse_args()

    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer
    model_cls = getattr(transformers, args.model_class)

    random.seed(args.seed); torch.manual_seed(args.seed)
    cfg = load_config(ROOT / args.config)
    tok = AutoTokenizer.from_pretrained(args.model)
    is_qwen = "qwen" in args.model.lower()
    eos_ids = tok("<|im_end|>\n", add_special_tokens=False)["input_ids"] if is_qwen else [tok.eos_token_id]
    excl = [x for x in args.exclude_cells.split(",") if x]
    allowed = None
    if args.split:
        sp = json.loads((ROOT / args.split).read_text(encoding="utf-8"))
        roles = set(args.roles.split(","))
        allowed = {(u["cell"], u["unit_idx"]) for u in sp["units"] if u["role"] in roles}
    examples, skipped_long, skipped_cells, skipped_role = [], 0, 0, 0
    for cell in split_cells(cfg, "train"):
        if any(x in cell.label for x in excl):
            skipped_cells += 1
            continue
        for ui, ref in enumerate(select_refs(cell)):
            if allowed is not None and (cell.label, ui) not in allowed:
                skipped_role += 1
                continue
            msgs = zeroshot_messages(ref["document_metadata"], ref["ocr_hypothesis"]["transcription_unit"])
            kw = {"enable_thinking": False} if is_qwen else {}
            p_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **kw)
            if not isinstance(p_ids, list):
                p_ids = list(p_ids["input_ids"])
            a_ids = tok.encode(ref["ground_truth"]["transcription_unit"], add_special_tokens=False) + eos_ids
            if len(p_ids) + len(a_ids) > args.max_tokens:
                skipped_long += 1
                continue
            examples.append((p_ids, a_ids))
    n_real = len(examples); n_extra = 0
    if args.extra_jsonl:
        for line in open(ROOT / args.extra_jsonl, encoding="utf-8"):
            if not line.strip():
                continue
            ex = json.loads(line)
            msgs = zeroshot_messages(ex["meta"], ex["ocr"])
            kw = {"enable_thinking": False} if is_qwen else {}
            p_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **kw)
            if not isinstance(p_ids, list):
                p_ids = list(p_ids["input_ids"])
            a_ids = tok.encode(ex["gt"], add_special_tokens=False) + eos_ids
            if len(p_ids) + len(a_ids) > args.max_tokens:
                skipped_long += 1
                continue
            examples.append((p_ids, a_ids)); n_extra += 1
    random.shuffle(examples)
    if args.max_examples:
        examples = examples[:args.max_examples]
    info = {"model": args.model, "n_examples": len(examples), "skipped_long": skipped_long, "skipped_cells": skipped_cells,
            "skipped_role": skipped_role, "split": args.split, "roles": args.roles if args.split else None,
            "epochs": args.epochs, "lr": args.lr, "rank": args.rank, "alpha": args.alpha, "grad_accum": args.grad_accum,
            "max_tokens": args.max_tokens, "seed": args.seed, "exclude_cells": excl, "eos_ids": eos_ids,
            "n_real": n_real, "n_extra": n_extra, "extra_jsonl": args.extra_jsonl, "device_map": args.device_map,
            "model_class": args.model_class, "target_modules": args.target_modules.split(","),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    print(json.dumps(info), flush=True)

    dmap = "auto" if args.device_map == "auto" else {"": int(args.device_map)}
    model = model_cls.from_pretrained(args.model, dtype=torch.bfloat16, device_map=dmap)
    if args.device_map == "auto":
        print("device map:", {k: v for k, v in model.hf_device_map.items() if not k.startswith("model.layers.") or k.endswith((".0", ".39", ".40", ".79"))}, flush=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=args.target_modules.split(","))
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    total_steps = math.ceil(len(examples) * args.epochs / args.grad_accum)
    warm = max(1, int(0.05 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total_steps - s) / max(1, total_steps - warm)))
    t0 = time.perf_counter()
    step, micro, run_loss, n_loss = 0, 0, 0.0, 0
    model.train()
    for epoch in range(args.epochs):
        order = list(range(len(examples)))
        random.Random(args.seed + epoch).shuffle(order)
        for i in order:
            p_ids, a_ids = examples[i]
            ids = torch.tensor([p_ids + a_ids], device="cuda:0")
            labels = torch.tensor([[-100] * len(p_ids) + a_ids], device="cuda:0")
            out = model(input_ids=ids, labels=labels)
            (out.loss / args.grad_accum).backward()
            run_loss += out.loss.item(); n_loss += 1; micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 10 == 0:
                    print(f"epoch {epoch + 1} step {step}/{total_steps} loss {run_loss / n_loss:.4f} elapsed {time.perf_counter() - t0:.0f}s", flush=True)
                    run_loss, n_loss = 0.0, 0
        if args.save_each_epoch:
            ep_dir = ROOT / args.out / f"ep{epoch + 1}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ep_dir)
            (ep_dir / "train_info.json").write_text(json.dumps({**info, "epoch": epoch + 1, "steps": step,
                "wall_s": round(time.perf_counter() - t0, 1)}, indent=2), encoding="utf-8")
            print(f"saved epoch {epoch + 1} adapter -> {ep_dir} ({time.perf_counter() - t0:.0f}s)", flush=True)
    if micro % args.grad_accum:
        opt.step(); opt.zero_grad(set_to_none=True); step += 1
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    info.update(steps=step, wall_s=round(time.perf_counter() - t0, 1))
    (out_dir / "train_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
