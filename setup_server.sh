#!/usr/bin/env bash
# One-time setup on the GPU server: client venv (.venv) + pinned data/scorer repos + offline tests.
# The vLLM server itself runs from its own existing venv (set VLLM_ENV, default $HOME/vllm-env), see serve_model.sh.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3.12}
if [ ! -x .venv/bin/python ]; then
  "$PYTHON" -m venv .venv
fi
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

bash scripts/fetch_data.sh

# Offline sanity checks (no GPU): oracle alignment/splice + system code paths with a fake LLM.
.venv/bin/python -m pytest -q tests/

echo "Setup done. Next: ./serve_model.sh (in another shell / tmux), then ./run_all.sh"
