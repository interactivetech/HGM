# Local vLLM Qwen support for HGM

## vLLM server

Recommended launch command:

```bash
vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key dummy \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --generation-config vllm
```

If the agent runs inside Docker containers, `localhost` from inside the container is not your host machine. Use a routable host/IP, `host.docker.internal`, an SSH tunnel, or Docker networking. Your current routable example is `http://173.73.39.103:8000/v1`.

## Environment variables

```bash
export VLLM_BASE_URL="http://173.73.39.103:8000/v1"
export VLLM_API_KEY="dummy"
export VLLM_MODEL="Qwen/Qwen3.6-35B-A3B-FP8"
```

You can also override these from `hgm.py`:

```bash
--vllm_base_url http://173.73.39.103:8000/v1
--vllm_api_key dummy
--vllm_model Qwen/Qwen3.6-35B-A3B-FP8
```

Supported local-model aliases:

```bash
--self_improve_llm vllm-qwen
--downstream_llm vllm-qwen
--diagnose_llm vllm-qwen
```

or

```bash
--self_improve_llm vllm:Qwen/Qwen3.6-35B-A3B-FP8
```

The vLLM model resolution order is:

1. explicit model after the `vllm:` prefix
2. `VLLM_MODEL`
3. first model from `client.models.list()`
4. alias fallback

## Smoke test

```bash
python scripts/check_vllm_endpoint.py --test-tools
```

If tool calling is unavailable, run:

```bash
python scripts/check_vllm_endpoint.py --skip-tool-test
```

The tool tests fail loudly when auto tool calling is misconfigured.

## Small HGM smoke run

```bash
python hgm.py \
  --no_polyglot \
  --self_improve_llm vllm-qwen \
  --downstream_llm vllm-qwen \
  --diagnose_llm vllm-qwen \
  --max_workers 1 \
  --max_task_evals 10 \
  --self_improve_timeout 900 \
  --evaluation_timeout 900 \
  --output_dir output_qwen_smoke
```

Logs should show the vLLM base URL and resolved model ID. Local-vLLM mode should not require paid OpenAI calls.

## Budget sweep

Safe staged sweep:

```bash
bash scripts/run_qwen_budget_sweep.sh
```

This runs budgets `10 25 50 100 200`, stops on the first failing budget, and writes `output_qwen_budget_<budget>_<timestamp>` directories with a per-budget `run.log`.

Reset local experiment artifacts before a fresh rerun:

```bash
bash scripts/reset_qwen_experiment_state.sh
```

Also remove the cached bootstrap Docker image:

```bash
bash scripts/reset_qwen_experiment_state.sh --docker
```

Summarize after the sweep:

```bash
python scripts/summarize_budget_sweep.py
```

Artifacts:

- `qwen_budget_sweep.csv`
- `qwen_budget_sweep.json`
- `qwen_best_accuracy_vs_budget.png`
- `qwen_best_accuracy_vs_allocated_hours.png`

The allocated-hours plot is the paper-aligned Figure 1 style view: x-axis is allocated worker-hours, y-axis is best-found accuracy.

## Full-eval caution

Large runs are expensive. Do not start them first.

Paper-aligned larger run template:

```bash
python hgm.py \
  --no_polyglot \
  --self_improve_llm vllm-qwen \
  --downstream_llm vllm-qwen \
  --diagnose_llm vllm-qwen \
  --full_eval \
  --max_task_evals 800 \
  --max_workers 4 \
  --output_dir output_qwen_budget_800_$(date +%Y%m%d_%H%M%S)
```

## Generalization evaluation

Evaluate the best discovered Qwen agent on SWE-Lite:

```bash
bash scripts/evaluate_best_qwen_on_swelite.sh output_qwen_smoke
```

Filtered Lite variant:

```bash
bash scripts/evaluate_best_qwen_on_swelite.sh output_qwen_smoke --filtered
```

Treat this as generalization evaluation, not optimization-set performance.

## Paper-aligned targets

| Metric | Paper reference |
|---|---:|
| SWE-Verified-60 after 800 evals | about 56.7% |
| Allocated compute for that point | about 517 CPU-hours |
| Full SWE-bench Verified after 8000 evals | 61.4% |
| SWE-Lite Filtered with GPT-5-mini backbone | 40.1% |
| SWE-Lite Standard with GPT-5-mini backbone | 49.0% |
| SWE-Lite Standard after transfer to GPT-5 | 57.0% |

## Interpretation guide

- Claim **self-improvement transfer** if evolved Qwen agents reliably beat the initial Qwen agent and no-self-improvement/random-search controls under equal compute, ideally by roughly `+5` absolute points on SWE-Lite.
- Claim **frontier recovery** only if Qwen-evolved/Qwen-evaluated agents approach the SWE-Lite Standard `56.7%` to `57.0%` region under comparable evaluation settings.
- Compare runs using `best-found accuracy (%)` versus `allocated worker-hours`, not just raw budget.
