"""Phase 1 edit-veto gate: a small classifier that accepts or rejects one edit block of a rewrite.

Input of one example = (OCR context window with the block marked, the proposed block, numeric features
incl. the rewriter's token logprobs). Output = P(block is helpful). Shared by gate_train.py / gate_apply.py.

Encoders (<= 8B rule applies to the gate; these are 0.6B / 0.58B / 1.7B):
  - Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B  (decoder; pooled = hidden state of the last non-pad token)
  - google/byt5-base                  (byte-level encoder; pooled = mean of non-pad positions)
The pooled vector is concatenated with the standardised numeric features and fed to a 2-layer MLP head.
`encoder = "lp"` is the training-free baseline: score = one logprob statistic, thresholded.
"""

import json
import math
from pathlib import Path

import torch
from torch import nn

OPEN, CLOSE = "⟦", "⟧"

# numeric features, in order; lp_* are null when the completion could not be mapped to tokens
NUM_FEATURES = [
    "lp_mean", "lp_min", "lp_sum", "lp_n", "lp_first", "lp_unit_mean",
    "len_ocr", "len_new", "len_diff", "log_n_blocks_unit", "log_len_unit", "frac_changed_unit",
    "is_insert", "is_delete", "lp_missing",
]
LP_FEATURES = {"lp_mean", "lp_min", "lp_sum", "lp_n", "lp_first", "lp_unit_mean", "lp_missing"}


def block_text(b: dict) -> str:
    return (
        f"[{b.get('lang')}|{b.get('dataset')}] OCR: {b['left']}{OPEN}{b['ocr_text']}{CLOSE}{b['right']}\n"
        f"FIX: {OPEN}{b['new_text']}{CLOSE}"
    )


def raw_features(b: dict, use_lp: bool = True) -> list[float]:
    lp_ok = bool(b.get("lp_ok")) and b.get("lp_mean") is not None
    f = {
        "lp_mean": b["lp_mean"] if lp_ok else 0.0,
        "lp_min": b["lp_min"] if lp_ok else 0.0,
        "lp_sum": b["lp_sum"] if lp_ok else 0.0,
        "lp_n": b["lp_n"] if lp_ok else 0.0,
        "lp_first": b["lp_first"] if lp_ok else 0.0,
        "lp_unit_mean": b["lp_unit_mean"] if lp_ok and b.get("lp_unit_mean") is not None else 0.0,
        "len_ocr": b["len_ocr"],
        "len_new": b["len_new"],
        "len_diff": b["len_new"] - b["len_ocr"],
        "log_n_blocks_unit": math.log1p(b["n_blocks_unit"]),
        "log_len_unit": math.log1p(b["len_unit"]),
        "frac_changed_unit": b["frac_changed_unit"],
        "is_insert": float(b["is_insert"]),
        "is_delete": float(b["is_delete"]),
        "lp_missing": 0.0 if lp_ok else 1.0,
    }
    if not use_lp:
        for k in LP_FEATURES:
            f[k] = 0.0
    return [float(f[k]) for k in NUM_FEATURES]


class Standardizer:
    def __init__(self, mean=None, std=None):
        self.mean, self.std = mean, std

    def fit(self, X: torch.Tensor) -> "Standardizer":
        self.mean = X.mean(0).tolist()
        self.std = [s if s > 1e-6 else 1.0 for s in X.std(0).tolist()]
        return self

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return (X - torch.tensor(self.mean)) / torch.tensor(self.std)


class GateModel(nn.Module):
    def __init__(self, encoder_name: str, n_num: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoConfig, AutoModel, T5EncoderModel

        self.encoder_name = encoder_name
        cfg = AutoConfig.from_pretrained(encoder_name)
        self.is_t5 = cfg.model_type in {"t5", "byt5", "mt5"}
        if self.is_t5:
            self.encoder = T5EncoderModel.from_pretrained(encoder_name, dtype=torch.bfloat16)
            dim = cfg.d_model
        else:
            self.encoder = AutoModel.from_pretrained(encoder_name, dtype=torch.bfloat16)
            dim = cfg.hidden_size
        self.head = nn.Sequential(
            nn.Linear(dim + n_num, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, input_ids, attention_mask, num_feats):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.is_t5:
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            last = attention_mask.sum(1) - 1  # right padding
            pooled = h[torch.arange(h.size(0), device=h.device), last]
        x = torch.cat([pooled.float(), num_feats.float()], dim=-1)
        return self.head(x).squeeze(-1)


def load_tokenizer(encoder_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(encoder_name)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def save_gate(dir_: Path, model: GateModel, std: Standardizer, meta: dict) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), dir_ / "model.pt")
    (dir_ / "gate.json").write_text(json.dumps({**meta, "std_mean": std.mean, "std_std": std.std}, indent=2), encoding="utf-8")


def load_state_dict_file(dir_: Path, fmt: str = "auto") -> dict:
    """State dict of a gate directory. fmt = "auto" (model.safetensors when present, else model.pt), "safetensors" or "pt".
    model.safetensors holds the identical tensors of the frozen model.pt, converted after the freeze
    (scripts/gate_to_safetensors.py); model.pt is kept for hash verification against frozen/p2_frozen_hashes.tsv."""
    dir_ = Path(dir_)
    st = dir_ / "model.safetensors"
    if fmt == "safetensors" or (fmt == "auto" and st.is_file()):
        from safetensors.torch import load_file

        return load_file(str(st), device="cpu")
    if fmt not in ("auto", "pt"):
        raise ValueError(f"unknown gate weight format {fmt!r}")
    return torch.load(dir_ / "model.pt", map_location="cpu", weights_only=True)


def load_gate(dir_: Path, device: str = "cuda", fmt: str = "auto"):
    dir_ = Path(dir_)
    meta = json.loads((dir_ / "gate.json").read_text(encoding="utf-8"))
    std = Standardizer(meta["std_mean"], meta["std_std"])
    model = GateModel(meta["encoder"], len(NUM_FEATURES), meta.get("hidden", 256), meta.get("dropout", 0.1))
    model.load_state_dict(load_state_dict_file(dir_, fmt))
    model.to(device).eval()
    return model, std, meta


@torch.no_grad()
def predict(model: GateModel, tok, std: Standardizer, blocks: list[dict], max_len: int, use_lp: bool,
            batch_size: int = 128, device: str = "cuda") -> list[float]:
    """P(helpful) for every block."""
    out = []
    for i in range(0, len(blocks), batch_size):
        chunk = blocks[i:i + batch_size]
        enc = tok([block_text(b) for b in chunk], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        feats = std(torch.tensor([raw_features(b, use_lp) for b in chunk]))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device), feats.to(device))
        out.extend(torch.sigmoid(logits.float()).cpu().tolist())
    return out
