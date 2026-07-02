#!/usr/bin/env bash
set -euo pipefail

MAX_WORKERS="${MAX_WORKERS:-1}"
SELF_IMPROVE_TIMEOUT="${SELF_IMPROVE_TIMEOUT:-3600}"
EVALUATION_TIMEOUT="${EVALUATION_TIMEOUT:-3600}"
INITIAL_EVAL_TASKS="${INITIAL_EVAL_TASKS:-}"
FULL_EVAL=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full_eval)
      FULL_EVAL=1
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

BUDGETS=(10 25 50 100 200)
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

for BUDGET in "${BUDGETS[@]}"; do
  OUTPUT_DIR="output_qwen_budget_${BUDGET}_${TIMESTAMP}"
  mkdir -p "${OUTPUT_DIR}"
  LOG_FILE="${OUTPUT_DIR}/run.log"
  CMD=(
    python hgm.py
    --no_polyglot
    --self_improve_llm vllm-qwen
    --downstream_llm vllm-qwen
    --diagnose_llm vllm-qwen
    --max_workers "${MAX_WORKERS}"
    --max_task_evals "${BUDGET}"
    --self_improve_timeout "${SELF_IMPROVE_TIMEOUT}"
    --evaluation_timeout "${EVALUATION_TIMEOUT}"
    --output_dir "${OUTPUT_DIR}"
  )

  if [[ -n "${INITIAL_EVAL_TASKS}" ]]; then
    CMD+=(--initial_eval_tasks "${INITIAL_EVAL_TASKS}")
  fi

  if [[ "${FULL_EVAL}" == "1" ]]; then
    CMD+=(--full_eval)
  fi
  if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
  fi

  echo "Running budget sweep for budget=${BUDGET} output_dir=${OUTPUT_DIR}"
  "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
done

echo
echo "Summarizing completed runs"
python scripts/summarize_budget_sweep.py

: <<'LARGE_RUN_EXAMPLE'
Large run example:
python hgm.py \
  --no_polyglot \
  --self_improve_llm vllm-qwen \
  --downstream_llm vllm-qwen \
  --diagnose_llm vllm-qwen \
  --full_eval \
  --max_task_evals 800 \
  --max_workers 4 \
  --self_improve_timeout 3600 \
  --evaluation_timeout 3600 \
  --output_dir output_qwen_budget_800_$(date +%Y%m%d_%H%M%S)

Scale to 8 workers only after validating endpoint stability and Docker throughput.
LARGE_RUN_EXAMPLE
