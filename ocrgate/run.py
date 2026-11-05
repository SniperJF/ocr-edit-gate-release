"""Run one system on one split, score it with the official scorer, and log everything.

    python -m ocrgate.run --config configs/phase0.yaml --system zeroshot --split dev [--base-url URL]

Outputs (runs/ is git-ignored because it contains dataset text):
    runs/<run_id>/predictions/*.jsonl  submission files (official format)
    runs/<run_id>/log.jsonl            one line per transcription unit (inputs, outputs, LLM metadata)
    runs/<run_id>/scores/              official scorer JSON + logs per cell
    runs/<run_id>/summary.json         config snapshot, git commit, timings, scores
and appends the summary (no dataset text) to results/runs.jsonl, then regenerates results/phase0.md.
"""

import argparse
import asyncio
import json
import random
import subprocess
import time
from datetime import datetime, timezone

import yaml

from .data import ROOT, load_config, resolve, split_cells, submission_name, submission_record, write_jsonl
from .llm import LLM
from .report import write_report
from .score import score_run, select_refs
from .systems import NEEDS_LLM, SYSTEMS


COUNTER_KEYS = (
    "fallback_to_input", "n_spans", "n_fallback_spans", "n_echo", "n_changed_spans", "n_blocks", "n_blocks_kept",
    "n_ops", "n_ops_kept", "n_truncated", "prompt_tokens", "completion_tokens",
)


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def git_state() -> dict:
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }


def load_subset(path: str | None) -> dict | None:
    """Probe subset file: JSON {cell label: [document_id, ...]} (no dataset text)."""
    if not path:
        return None
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def new_run_dir(system: str, split: str, tag: str | None = None) -> tuple[str, "Path"]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}_{system}_{split}" + (f"_{tag}" if tag else "")
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True)
    return run_id, run_dir


def finalize_run(run_dir, summary: dict) -> None:
    """Write summary.json, append the summary (no dataset text) to results/runs.jsonl, refresh phase0.md."""
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "results" / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    write_report(ROOT / "results" / "runs.jsonl", ROOT / "results" / "phase0.md")


async def _run(args, cfg) -> dict:
    random.seed(cfg["seed"])
    cells = split_cells(cfg, args.split)
    sys_cfg = cfg["systems"][args.system]
    system = SYSTEMS[args.system]
    subset = load_subset(args.subset)

    llm = None
    models = None
    if args.system in NEEDS_LLM:
        llm = LLM(cfg["endpoint"], cfg["generation"], base_url=args.base_url)
        models = await llm.check()

    run_id, run_dir = new_run_dir(args.system, args.split, args.tag)
    stamp = run_id.split("_", 1)[0]
    pred_dir = run_dir / "predictions"

    t0 = time.perf_counter()
    names, n_docs, counters = {}, 0, {}
    with open(run_dir / "log.jsonl", "w", encoding="utf-8") as log_f:
        for cell in cells:
            refs = select_refs(cell, args.limit, subset)
            t_cell = time.perf_counter()
            outputs = await asyncio.gather(*(system(r, llm, sys_cfg) for r in refs))
            dt_cell = time.perf_counter() - t_cell
            records = []
            for ref, (out, doc_log) in zip(refs, outputs):
                records.append(submission_record(ref, out))
                for k in COUNTER_KEYS:
                    if k in doc_log and doc_log[k] is not None:
                        counters[k] = counters.get(k, 0) + int(doc_log[k])
                log_f.write(json.dumps({
                    "run_id": run_id,
                    "cell": cell.label,
                    "document_id": ref["document_metadata"]["document_id"],
                    "ocr": ref["ocr_hypothesis"]["transcription_unit"],
                    "output": out,
                    "changed": out != ref["ocr_hypothesis"]["transcription_unit"],
                    **doc_log,
                }, ensure_ascii=False) + "\n")
            names[cell.stem] = submission_name(args.system, cell)
            write_jsonl(pred_dir / names[cell.stem], records)
            n_docs += len(refs)
            print(f"[{args.system}/{args.split}] {cell.label}: {len(refs)} units in {dt_cell:.1f}s", flush=True)
    gen_wall = time.perf_counter() - t0

    if args.limit:
        print(f"NOTE: --limit {args.limit}: scored on the first {args.limit} units per cell only (debug).", flush=True)
    scores = score_run(cells, pred_dir, names, run_dir / "scores", cfg, limit=args.limit, subset=subset)
    wall = time.perf_counter() - t0

    summary = {
        "run_id": run_id,
        "phase": cfg.get("phase", 0),
        "system": args.system,
        "tag": args.tag,
        "split": args.split,
        "limit": args.limit,
        "subset": args.subset,
        "model": cfg["endpoint"]["model"] if llm else None,
        "thinking": cfg["endpoint"].get("enable_thinking", False) if llm else None,
        "send_thinking_kwarg": llm.send_thinking_kwarg if llm else None,
        "served_models": models,
        "base_url": llm.base_url if llm else None,
        **git_state(),
        "config_path": str(args.config),
        "seed": cfg["seed"],
        "generation": cfg["generation"] if llm else None,
        "system_config": sys_cfg,
        "started_utc": stamp,
        "n_units": n_docs,
        "generation_wall_s": round(gen_wall, 1),
        "wall_s": round(wall, 1),
        "counters": counters,
        **scores,
    }
    finalize_run(run_dir, summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase0.yaml")
    ap.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--base-url", default=None, help="override endpoint.base_url from the config")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N units per cell")
    ap.add_argument("--subset", default=None, help="JSON {cell label: [document ids]}: run/score only these units")
    ap.add_argument("--tag", default=None, help="suffix for the run id (e.g. model/thinking variant)")
    ap.add_argument("--model", default=None, help="override endpoint.model (e.g. Qwen/Qwen3-32B)")
    ap.add_argument("--thinking", choices=["on", "off"], default=None, help="override endpoint.enable_thinking")
    ap.add_argument("--send-thinking-kwarg", choices=["auto", "on", "off"], default=None,
                    help="override endpoint.send_thinking_kwarg (auto = only when the model name contains 'qwen3'; use 'on' for a Qwen3 LoRA served under another name)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a system config key")
    ap.add_argument("--gen", action="append", default=[], metavar="KEY=VALUE",
                    help="Phase 3: override a generation key (e.g. temperature=0.7 seed=1235 for sampled pool candidates)")
    args = ap.parse_args()
    cfg = load_config(resolve(args.config))
    if args.model:
        cfg["endpoint"]["model"] = args.model
    if args.thinking:
        cfg["endpoint"]["enable_thinking"] = args.thinking == "on"
    if args.send_thinking_kwarg:
        cfg["endpoint"]["send_thinking_kwarg"] = "auto" if args.send_thinking_kwarg == "auto" else args.send_thinking_kwarg == "on"
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg["systems"][args.system][k] = yaml.safe_load(v)
    for kv in args.gen:
        k, v = kv.split("=", 1)
        cfg["generation"][k] = yaml.safe_load(v)
    s = asyncio.run(_run(args, cfg))
    print(
        f"[{s['system']}/{s['split']}] weighted cMER={s['cmer_micro_weighted']:.4f} "
        f"pref={s['pref_score_cmer_macro_weighted']:.4f} wall={s['wall_s']}s -> runs/{s['run_id']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
