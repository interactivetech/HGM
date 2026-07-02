#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <hgm_run_dir> [--filtered] [extra evaluate_agent.py args...]"
  exit 1
fi

RUN_DIR="$1"
shift

FILTERED=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --filtered)
      FILTERED=1
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

BEST_AGENT_PATH="$(python - "$RUN_DIR" <<'PY'
import json
import os
import sys

run_dir = sys.argv[1]
best = None
best_score = -1.0
for child in os.listdir(run_dir):
    child_dir = os.path.join(run_dir, child)
    if not os.path.isdir(child_dir) or child == "initial":
        continue
    metadata_path = os.path.join(child_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        continue
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    score = metadata.get("overall_performance", {}).get("accuracy_score", 0.0)
    if score > best_score:
        best_score = score
        best = child_dir
if not best:
    raise SystemExit("No evaluated child agents found.")
print(best)
PY
)"

RESULTS_ROOT="${RESULTS_ROOT:-results_qwen_swelite}"
mkdir -p "${RESULTS_ROOT}"

echo "Best agent path: ${BEST_AGENT_PATH}"
echo "Running SWE-Lite standard generalization evaluation"
PYTHONPATH=. python evaluate_agent.py \
  --agent_path "${BEST_AGENT_PATH}" \
  --results_dir "${RESULTS_ROOT}/standard" \
  --split Lite \
  --llm "${LLM:-vllm-qwen}" \
  "${EXTRA_ARGS[@]}"

if [[ "${FILTERED}" == "1" ]]; then
  echo "Running SWE-Lite filtered generalization evaluation"
  PYTHONPATH=. python evaluate_agent.py \
    --agent_path "${BEST_AGENT_PATH}" \
    --results_dir "${RESULTS_ROOT}/filtered" \
    --split Lite \
    --exclude_verified_overlap \
    --llm "${LLM:-vllm-qwen}" \
    "${EXTRA_ARGS[@]}"
fi
