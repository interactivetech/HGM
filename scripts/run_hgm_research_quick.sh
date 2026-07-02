#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${ROOT_DIR}/output_research_quick_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

HGM_ARGS=(
  --no_polyglot
  --self_improve_llm vllm-qwen
  --downstream_llm vllm-qwen
  --diagnose_llm vllm-qwen
  --max_task_evals 30
  --initial_eval_tasks 5
  --max_workers 6
  --self_improve_timeout 1200
  --evaluation_timeout 1200
  --n_pseudo_descendant_evals 200
  --eval_random_level 1.0
  --output_dir "${OUTPUT_DIR}"
)

if [[ -n "${VLLM_BASE_URL:-}" ]]; then
  HGM_ARGS+=(--vllm_base_url "${VLLM_BASE_URL}")
fi
if [[ -n "${VLLM_API_KEY:-}" ]]; then
  HGM_ARGS+=(--vllm_api_key "${VLLM_API_KEY}")
fi
if [[ -n "${VLLM_MODEL:-}" ]]; then
  HGM_ARGS+=(--vllm_model "${VLLM_MODEL}")
fi

RUN_LOG="${OUTPUT_DIR}/run.log"
set +e
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u "${ROOT_DIR}/hgm.py" "${HGM_ARGS[@]}" |& tee "${RUN_LOG}"
HGM_EXIT=${PIPESTATUS[0]}
set -e

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/analyze_hgm_research_run.py" --output_dir "${OUTPUT_DIR}"

echo "Progress report: ${OUTPUT_DIR}/research/progress.md"
echo "Figures: ${OUTPUT_DIR}/research/figures"
exit "${HGM_EXIT}"
