#!/usr/bin/env bash
set -euo pipefail

# Run from the directory containing this script.
cd "$(dirname "$0")"

# Safe defaults. Override these before running if desired:
#   export LLM="vllm" or "gpt-5" etc.
#   export HOURS_PER_TASK=1
#   export NUM_WORKERS=1
: "${LLM:=gpt-5}"
: "${HOURS_PER_TASK:=1}"
: "${NUM_WORKERS:=1}"

# Find Python even if PYTHON_BIN was not exported.
if [[ -z "${PYTHON_BIN:-}" ]]; then
	  if [[ -x "$(pwd)/.venv/bin/python" ]]; then
		      PYTHON_BIN="$(pwd)/.venv/bin/python"
		        elif command -v python3 >/dev/null 2>&1; then
				    PYTHON_BIN="$(command -v python3)"
				      else
					          PYTHON_BIN="$(command -v python)"
						    fi
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
	  echo "ERROR: PYTHON_BIN is not executable: ${PYTHON_BIN}"
	    exit 1
fi

RUN_ID="hgm_miniswe_verified_initial10_budget40_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$(pwd)/output_hgm/${RUN_ID}"
RUN_LOG="${OUTPUT_DIR}/run.log"
mkdir -p "${OUTPUT_DIR}"

echo "PYTHON_BIN=${PYTHON_BIN}"
echo "LLM=${LLM}"
echo "HOURS_PER_TASK=${HOURS_PER_TASK}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

HGM_ARGS=(
  --no_polyglot
    --full_eval
      --initial_agent_name miniswe_agent
        --initial_eval_tasks 10
	  --max_task_evals 40
	    --max_workers "${NUM_WORKERS}"
	      --self_improve_llm "${LLM}"
	        --downstream_llm "${LLM}"
		  --diagnose_llm "${LLM}"
		    --self_improve_timeout "$((HOURS_PER_TASK * 3600))"
		      --evaluation_timeout "$((HOURS_PER_TASK * 3600))"
		        --n_pseudo_descendant_evals 200
			  --eval_random_level 1.0
			    --output_dir "${OUTPUT_DIR}"
		    )

		    # Optional vLLM passthrough, only if these are set.
		    if [[ -n "${VLLM_BASE_URL:-}" ]]; then
			      HGM_ARGS+=(--vllm_base_url "${VLLM_BASE_URL}")
		    fi
		    if [[ -n "${VLLM_API_KEY:-}" ]]; then
			      HGM_ARGS+=(--vllm_api_key "${VLLM_API_KEY}")
		    fi
		    if [[ -n "${VLLM_MODEL:-}" ]]; then
			      HGM_ARGS+=(--vllm_model "${VLLM_MODEL}")
		    fi

		    set +e
		    PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u hgm.py "${HGM_ARGS[@]}" |& tee "${RUN_LOG}"
		    RUN_EXIT=${PIPESTATUS[0]}
		    set -e

		    if [[ -f scripts/analyze_hgm_research_run.py ]]; then
			      "${PYTHON_BIN}" scripts/analyze_hgm_research_run.py --output_dir "${OUTPUT_DIR}" || true
		    fi

		    echo "Run log: ${RUN_LOG}"
		    echo "Output dir: ${OUTPUT_DIR}"
		    exit "${RUN_EXIT}"
