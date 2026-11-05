#!/usr/bin/env python3
"""Convert a frozen gate checkpoint (model.pt, a plain PyTorch state_dict) to model.safetensors next to it, on CPU,
and verify the conversion: same keys, dtypes and shapes, and torch.equal for every tensor. model.pt is not modified.

    python scripts/gate_to_safetensors.py runs/gates/p2_pooled [runs/gates/p2_pooled_nolp ...]

Prints the sha256 of both files. Exit code 1 if any check fails.
"""
import hashlib
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert(d: Path) -> bool:
    pt, st = d / "model.pt", d / "model.safetensors"
    pt_sha = sha256(pt)
    sd = torch.load(pt, map_location="cpu", weights_only=True)
    # detach shared storage so safetensors can serialise every key; values are unchanged
    save_file({k: v.detach().clone().contiguous() for k, v in sd.items()}, str(st),
              metadata={"format": "pt", "converted_from": "model.pt", "model_pt_sha256": pt_sha})
    back = load_file(str(st), device="cpu")
    ok = set(sd) == set(back)
    n_bad = 0
    for k, v in sd.items():
        w = back.get(k)
        if w is None or v.dtype != w.dtype or v.shape != w.shape or not torch.equal(v, w):
            n_bad += 1
    ok = ok and n_bad == 0 and sha256(pt) == pt_sha
    dtypes = sorted({str(v.dtype) for v in sd.values()})
    print(f"{d}: keys {len(sd)} vs {len(back)}, dtypes {dtypes}, mismatching tensors {n_bad}, model.pt unchanged {sha256(pt) == pt_sha}"
          f" -> {'OK' if ok else 'FAIL'}")
    print(f"  model.pt          sha256 {pt_sha}  {pt.stat().st_size} bytes")
    print(f"  model.safetensors sha256 {sha256(st)}  {st.stat().st_size} bytes")
    return ok


if __name__ == "__main__":
    results = [convert(Path(a)) for a in sys.argv[1:]]
    sys.exit(0 if results and all(results) else 1)
