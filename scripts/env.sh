#!/bin/bash
# Source this file from any directory. Do not change HOME or CODEX_HOME.
JEV_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export JEV_PROJECT_ROOT
export XDG_CACHE_HOME="$JEV_PROJECT_ROOT/.cache"
export XDG_CONFIG_HOME="$JEV_PROJECT_ROOT/.config"
export XDG_DATA_HOME="$JEV_PROJECT_ROOT/.local/share"
export MPLCONFIGDIR="$JEV_PROJECT_ROOT/.cache/matplotlib"
export PLAYWRIGHT_BROWSERS_PATH="$JEV_PROJECT_ROOT/.cache/ms-playwright"
export PIP_CACHE_DIR="$JEV_PROJECT_ROOT/.cache/pip"
export UV_CACHE_DIR="$JEV_PROJECT_ROOT/.cache/uv-cache"
export HF_HOME="$JEV_PROJECT_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_XET_CACHE="$HF_HOME/xet"
export JAX_COMPILATION_CACHE_DIR="$JEV_PROJECT_ROOT/.cache/jax"
export TMPDIR="$JEV_PROJECT_ROOT/.cache/tmp"
export PYTHONPATH="$JEV_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$UV_CACHE_DIR" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_XET_CACHE" \
  "$JAX_COMPILATION_CACHE_DIR" "$TMPDIR"
