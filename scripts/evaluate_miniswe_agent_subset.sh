#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

INSTANCE_IDS_FILE="${1:-${ROOT_DIR}/swe_bench/subsets/small.json}"
RESULTS_DIR="${2:-${ROOT_DIR}/results/miniswe_subset}"
LLM="${LLM:-gpt-5}"
HOURS_PER_TASK="${HOURS_PER_TASK:-1}"
NUM_WORKERS="${NUM_WORKERS:-1}"

"${PYTHON_BIN}" "${ROOT_DIR}/evaluate_agent.py" \
  --agent_path "${ROOT_DIR}/initial_swe/miniswe_agent/src" \
  --results_dir "${RESULTS_DIR}" \
  --split Verified \
  --llm "${LLM}" \
  --hours_per_task "${HOURS_PER_TASK}" \
  --num_workers "${NUM_WORKERS}" \
  --instance_ids_file "${INSTANCE_IDS_FILE}"
