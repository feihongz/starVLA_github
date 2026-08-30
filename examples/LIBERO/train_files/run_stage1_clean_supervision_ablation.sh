#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-.venv310/bin/python}"
exec "${PYTHON_BIN}" scripts/stage1/run_stage1_ablation.py \
  --benchmark libero "$@"
