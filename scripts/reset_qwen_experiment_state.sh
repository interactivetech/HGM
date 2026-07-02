#!/usr/bin/env bash
set -euo pipefail

REMOVE_DOCKER=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --docker)
      REMOVE_DOCKER=1
      shift
      ;;
    *)
      echo "Usage: $0 [--docker]"
      exit 1
      ;;
  esac
done

paths=(
  output_qwen_smoke
  output_qwen_budget_*
  qwen_budget_sweep.csv
  qwen_budget_sweep.json
  qwen_best_accuracy_vs_budget.png
  qwen_best_accuracy_vs_allocated_hours.png
  results_qwen_swelite
  initial_swe/default_agent
)

echo "Removing local experiment artifacts"
for path in "${paths[@]}"; do
  if compgen -G "$path" > /dev/null; then
    rm -rf $path
    echo "  removed $path"
  fi
done

if [[ "$REMOVE_DOCKER" == "1" ]]; then
  echo "Removing Docker image cache"
  docker rmi default_agent >/dev/null 2>&1 || true
  echo "  attempted docker rmi default_agent"
fi

echo "Reset complete"
