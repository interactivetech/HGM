# mini-SWE Agent 10-Task Runbook

This runbook covers:

1. Running the mini-SWE-style agent on 10 SWE-bench Verified tasks.
2. Running HGM with `miniswe_agent` selected and 10 initial evaluation tasks.
3. Resetting local run state and task containers if a run gets stuck.

Run all commands from the HGM repository root.

```bash
cd /mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM
```

## Prerequisites

Choose one model provider path.

OpenAI:

```bash
export OPENAI_API_KEY="..."
export LLM="gpt-5"
```

OpenRouter:

```bash
export OpenRouter_API_KEY="..."
export LLM="anthropic/claude-sonnet-4"
```

vLLM:

```bash
export VLLM_BASE_URL="http://173.73.39.103:8000/v1"
export VLLM_API_KEY="dummy"
export VLLM_MODEL="Qwen/Qwen3.6-35B-A3B-FP8"
export LLM="vllm"
```

Set shared evaluation defaults:

```bash
export PYTHON_BIN="${PYTHON_BIN:-$(pwd)/.venv/bin/python}"
test -x "${PYTHON_BIN}" || export PYTHON_BIN="python"
export HOURS_PER_TASK="${HOURS_PER_TASK:-1}"
export NUM_WORKERS="${NUM_WORKERS:-1}"
export HGM_MINISWE_STEP_LIMIT="${HGM_MINISWE_STEP_LIMIT:-250}"
export HGM_MINISWE_TIMEOUT_BUFFER="${HGM_MINISWE_TIMEOUT_BUFFER:-60}"
```

## 10 SWE-Bench Tasks Total

This runs `miniswe_agent` directly through HGM's `evaluate_agent.py` wrapper on
the first 10 SWE-bench Verified tasks.

```bash
RUN_ID="miniswe_verified_10_$(date +%Y%m%d_%H%M%S)"
RUN_LOG="results/${RUN_ID}.log"
mkdir -p results

set +e
"${PYTHON_BIN}" evaluate_agent.py \
  --agent_path initial_swe/miniswe_agent/src \
  --results_dir "results/${RUN_ID}" \
  --split Verified \
  --llm "${LLM}" \
  --hours_per_task "${HOURS_PER_TASK}" \
  --num_workers "${NUM_WORKERS}" \
  --n_tasks 10 |& tee "${RUN_LOG}"
RUN_EXIT=${PIPESTATUS[0]}
set -e
exit "${RUN_EXIT}"
```

Expected outputs:

```text
results/${RUN_ID}.log
results/${RUN_ID}/miniswe_agent_1/*.json
results/${RUN_ID}/miniswe_agent_1/*.md
results/${RUN_ID}/miniswe_agent_1/*_docker.log
results/${RUN_ID}/miniswe_agent_1/all_preds.jsonl
```

If you want to use the helper script instead:

```bash
RUN_LOG="results/miniswe_verified_10.log"
mkdir -p results
set +e
LLM="${LLM}" HOURS_PER_TASK="${HOURS_PER_TASK}" NUM_WORKERS="${NUM_WORKERS}" \
  bash scripts/evaluate_miniswe_agent.sh 10 "results/miniswe_verified_10" |& tee "${RUN_LOG}"
RUN_EXIT=${PIPESTATUS[0]}
set -e
exit "${RUN_EXIT}"
```

## 10 Tasks As HGM Initial Evaluation

This runs HGM with `miniswe_agent` selected as the initial agent and limits the
initial evaluation to 10 tasks.

```bash
RUN_ID="hgm_miniswe_initial10_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$(pwd)/output_hgm/${RUN_ID}"
RUN_LOG="${OUTPUT_DIR}/run.log"
mkdir -p "${OUTPUT_DIR}"

set +e
"${PYTHON_BIN}" -u hgm.py \
  --no_polyglot \
  --initial_agent_name miniswe_agent \
  --initial_eval_tasks 10 \
  --max_task_evals 10 \
  --max_workers "${NUM_WORKERS}" \
  --self_improve_llm "${LLM}" \
  --downstream_llm "${LLM}" \
  --diagnose_llm "${LLM}" \
  --self_improve_timeout "$((HOURS_PER_TASK * 3600))" \
  --evaluation_timeout "$((HOURS_PER_TASK * 3600))" \
  --output_dir "${OUTPUT_DIR}" |& tee "${RUN_LOG}"
RUN_EXIT=${PIPESTATUS[0]}
set -e
exit "${RUN_EXIT}"
```

Expected outputs:

```text
output_hgm/${RUN_ID}/run.log
output_hgm/${RUN_ID}/run_config.json
output_hgm/${RUN_ID}/init_evaluated_tasks.json
output_hgm/${RUN_ID}/initial/metadata.json
output_hgm/${RUN_ID}/hgm_metadata.jsonl
```

Important: `hgm.py` treats `initial_swe/miniswe_agent` as a cached initial
agent because this integration supplies that directory. For clean direct
benchmark numbers, prefer the `evaluate_agent.py --n_tasks 10` command above.

## Verify Progress

Show result files:

```bash
find results output_hgm -maxdepth 4 -type f \
  \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' -o -name '*docker.log' \) \
  | sort | tail -80
```

Inspect submitted patches:

```bash
find results -maxdepth 4 -path '*miniswe_agent*' -name '*.json' \
  -print | sort | tail -20
```

Inspect HGM metadata:

```bash
find output_hgm \( -name metadata.json -o -name hgm_metadata.jsonl \) \
  | sort | tail -20
```

## Reset Commands

Use the smallest reset that matches the problem. These commands intentionally
do not touch source files.

### Stop Running Task Containers

```bash
docker ps -aq \
  --filter "name=sweb" \
  --filter "name=hgm-container" \
  | xargs -r docker rm -f
```

### Remove Exited Containers

```bash
docker container prune -f
```

### Remove One Failed Result Directory

Replace the paths with the run you want to discard.

```bash
rm -rf results/miniswe_verified_10
rm -rf output_hgm/hgm_miniswe_initial10
```

### Reset All mini-SWE Evaluation Outputs

This removes only outputs following the naming convention in this runbook.

```bash
rm -rf results/miniswe_verified_10*
rm -rf output_hgm/hgm_miniswe_initial10*
```

### Rebuild SWE-Bench Images From Scratch

This is slower and should only be used if task images or environments are
corrupt.

```bash
docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' \
  | awk '/swebench|sweb/ {print $2}' \
  | sort -u \
  | xargs -r docker rmi -f
```

### Full Local Docker Cleanup

Use this only if disk space or Docker state is badly broken. It removes stopped
containers, unused networks, dangling images, and build cache.

```bash
docker system prune -f
docker builder prune -f
```

## Common Adjustments

Increase parallelism:

```bash
export NUM_WORKERS=4
```

Increase per-task timeout:

```bash
export HOURS_PER_TASK=2
```

Log and tune the gap between the outer shell timeout and the agent's internal
wall-time limit:

```bash
export HGM_MINISWE_TIMEOUT_BUFFER=300
```

Optionally set and log the model request timeout. Leave unset to use the
OpenAI-compatible client's default timeout behavior:

```bash
export HGM_MINISWE_REQUEST_TIMEOUT=300
```

Use a lower LLM call limit only for packaging smoke tests, not benchmark runs:

```bash
export HGM_MINISWE_STEP_LIMIT=30
```

Run without model calls for packaging smoke tests only:

```bash
export HGM_MINISWE_DRY_RUN=1
```

Unset dry-run before real evaluation:

```bash
unset HGM_MINISWE_DRY_RUN
```
