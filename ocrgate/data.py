"""Reading HIPE-OCRepair reference files and writing submission files.

Path resolution. Every path in the YAML configs is relative to the repository root. The two upstream repositories
(cloned by `scripts/fetch_data.sh`) are looked up through the environment variables `HIPE_DATA_DIR` and `HIPE_EVAL_DIR`
when set; otherwise under `<root>/HIPE-OCRepair-2026-data` and `<root>/HIPE-OCRepair-2026-eval`. `OCRGATE_ROOT` overrides
the repository root (default: the directory containing the `ocrgate` package).
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(os.environ.get("OCRGATE_ROOT") or Path(__file__).resolve().parent.parent)
DATA_REPO_DIR = "HIPE-OCRepair-2026-data"
EVAL_REPO_DIR = "HIPE-OCRepair-2026-eval"

LANGUAGE_NAMES = {"en": "English", "fr": "French", "de": "German"}


def data_dir() -> Path:
    return Path(os.environ["HIPE_DATA_DIR"]).expanduser() if os.environ.get("HIPE_DATA_DIR") else ROOT / DATA_REPO_DIR


def eval_dir() -> Path:
    return Path(os.environ["HIPE_EVAL_DIR"]).expanduser() if os.environ.get("HIPE_EVAL_DIR") else ROOT / EVAL_REPO_DIR


def resolve(path: str | Path) -> Path:
    """Absolute path of a config-relative path; the two upstream repo prefixes follow HIPE_DATA_DIR / HIPE_EVAL_DIR."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    parts = p.parts
    if parts and parts[0] == DATA_REPO_DIR:
        return data_dir().joinpath(*parts[1:])
    if parts and parts[0] == EVAL_REPO_DIR:
        return eval_dir().joinpath(*parts[1:])
    return ROOT / p


def data_available(cfg: dict | None = None, split: str = "dev") -> bool:
    """True iff every reference file of `split` in `cfg` (default configs/phase0.yaml) exists."""
    if cfg is None:
        cfg_path = ROOT / "configs" / "phase0.yaml"
        if not cfg_path.is_file():
            return False
        cfg = load_config(cfg_path)
    return all(resolve(e["path"]).is_file() for e in cfg["splits"].get(split, []))


@dataclass
class Cell:
    """One reference file = one (dataset, language, split) cell of the benchmark."""

    path: Path
    weight: float
    official: bool

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def label(self) -> str:
        # hipe-ocrepair-bench_v0.9_<dataset>_<version>_<split>_<lang> -> <dataset>_<version>_<split>_<lang>
        return re.sub(r"^hipe-ocrepair-bench_v[\d.]+_", "", self.stem)


def load_config(path: str | Path) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_cells(cfg: dict, split: str) -> list[Cell]:
    if split not in cfg["splits"]:
        raise SystemExit(f"Unknown split {split!r}; configured: {list(cfg['splits'])}")
    cells = []
    for entry in cfg["splits"][split]:
        path = resolve(entry["path"])
        if not path.is_file():
            raise SystemExit(f"Missing reference file {path} (run scripts/fetch_data.sh or set HIPE_DATA_DIR / HIPE_EVAL_DIR)")
        cells.append(Cell(path=path, weight=float(entry["weight"]), official=entry.get("official", True)))
    return cells


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def submission_name(system: str, cell: Cell, run: int = 1) -> str:
    """`<team>_<reference-stem>_run<N>.jsonl`, as derived back by the official lib/score_one.py.

    The team part must not contain underscores (score_one.py strips everything up to the first `_`).
    """
    team = system.replace("_", "-")
    return f"{team}_{cell.stem}_run{run}.jsonl"


def submission_record(ref: dict, corrected: str) -> dict:
    """Submission record in the official format (no ground truth)."""
    meta = ref["document_metadata"]
    return {
        "document_metadata": {
            "document_id": meta["document_id"],
            "primary_dataset_name": meta["primary_dataset_name"],
            "language": meta["language"],
        },
        "ocr_hypothesis": {"transcription_unit": ref["ocr_hypothesis"]["transcription_unit"]},
        "ocr_postcorrection_output": {"transcription_unit": corrected},
    }
