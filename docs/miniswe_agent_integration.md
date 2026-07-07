# mini-SWE-style HGM Agent

## Architecture

The new agent is packaged as `initial_swe/miniswe_agent/src`, matching HGM's
existing initial-agent layout. HGM's harness copies this directory into each
SWE-bench task container and runs `/hgm/coding_agent.py`, so the selectable
entrypoint is `initial_swe/miniswe_agent/src/coding_agent.py`.

The implementation follows mini-SWE-agent's minimal control flow:

1. Render a system message and task message.
2. Query an OpenAI-compatible model with a `bash` tool schema.
3. Parse bash tool calls from the model response.
4. Execute the action in the task repository.
5. Append tool observation messages and save a trajectory.
6. Stop when the action output starts with `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
   or when limits are reached.

HGM compatibility is preserved by always writing `/hgm/model_patch.diff`. On
successful submission this is the exact patch emitted by the agent; on failure
or timeout it is empty so setup-time repository diffs are not submitted.

## Files Changed

- `coding_agent_miniswe.py`: root convenience wrapper for local execution.
- `initial_swe/miniswe_agent/src/coding_agent.py`: standalone HGM-compatible
  mini-SWE-style agent.
- `initial_swe/miniswe_agent/src/utils/git_utils.py`: minimal patch extraction
  utility.
- `initial_swe/miniswe_agent/src/*`: minimal harness-required package files.
- `initial_swe/miniswe_agent/metadata.json`: initial-agent metadata stub.
- `scripts/evaluate_miniswe_agent.sh`: helper for 1-task and 3-task runs.
- `scripts/evaluate_miniswe_agent_subset.sh`: helper for subset runs.
- `docs/miniswe_agent_integration.md`: this note.

## How To Run

Run one Verified task:

```bash
LLM=gpt-5 HOURS_PER_TASK=1 NUM_WORKERS=1 \
  bash scripts/evaluate_miniswe_agent.sh 1 results/miniswe_1
```

Run three Verified tasks:

```bash
LLM=gpt-5 HOURS_PER_TASK=1 NUM_WORKERS=1 \
  bash scripts/evaluate_miniswe_agent.sh 3 results/miniswe_3
```

Run the HGM small subset:

```bash
LLM=gpt-5 HOURS_PER_TASK=1 NUM_WORKERS=1 \
  bash scripts/evaluate_miniswe_agent_subset.sh swe_bench/subsets/small.json results/miniswe_small
```

Run directly through `evaluate_agent.py`:

```bash
python evaluate_agent.py \
  --agent_path initial_swe/miniswe_agent/src \
  --results_dir results/miniswe_1 \
  --split Verified \
  --llm gpt-5 \
  --hours_per_task 1 \
  --num_workers 1 \
  --n_tasks 1
```

Select as HGM's initial agent:

```bash
python hgm.py --no_polyglot --initial_agent_name miniswe_agent
```

## Assumptions

- The task container has network access to the selected model endpoint.
- `OPENAI_API_KEY`, `OpenRouter_API_KEY`, or `VLLM_BASE_URL`/`VLLM_API_KEY` are
  set as needed by the selected `--llm`.
- HGM continues to own SWE-bench dataset loading, task container setup, and
  evaluation.

## Known Gaps

- The implementation intentionally does not vendor mini-SWE-agent or depend on
  the mini-extra SWE-bench CLI.
- The implementation uses the `openai` SDK directly against OpenAI-compatible
  providers instead of depending on LiteLLM, to keep per-task runtime installs
  small.
- `hgm.py --initial_agent_name miniswe_agent` uses the supplied metadata stub
  unless the agent is evaluated separately with `evaluate_agent.py`.
