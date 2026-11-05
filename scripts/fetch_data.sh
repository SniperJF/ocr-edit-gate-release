#!/usr/bin/env bash
# Clone the official data and evaluation repos at the commits pinned in configs/phase0.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."

fetch() {  # url dir commit
  local url=$1 dir=$2 commit=$3
  if [ ! -d "$dir/.git" ]; then
    git clone --quiet "$url" "$dir"
  fi
  git -C "$dir" fetch --quiet --tags origin
  git -C "$dir" -c advice.detachedHead=false checkout --quiet "$commit"
  echo "$dir @ $(git -C "$dir" rev-parse --short HEAD)"
}

# Keep in sync with configs/phase0.yaml (data.*_commit).
fetch https://github.com/hipe-eval/HIPE-OCRepair-2026-data HIPE-OCRepair-2026-data bd463f7e340280473d9e5bab5e231f4922ea1a3b
fetch https://github.com/hipe-eval/HIPE-OCRepair-2026-eval HIPE-OCRepair-2026-eval 47a71d89e900ddff682b094e755b884271ef5244
