# HGM Research Playbook

This document is the operational checklist for running HGM research experiments, analyzing them, and resetting a failed run without touching code or Docker images.

## 1. Prerequisites

- Run commands from the repo root.
- Use the local virtual environment when possible: `.venv/bin/python`.
- Set your vLLM environment before running:

```bash
export VLLM_BASE_URL="http://127.0.0.1:8000/v1"
export VLLM_API_KEY="dummy"
export VLLM_MODEL="Qwen/Qwen3.6-35B-A3B-FP8"
```

- The smoke and quick scripts respect these environment variables and pass them through to `hgm.py`.

## 2. Recommended run order

1. Start with smoke mode.
2. Inspect the generated report.
3. If the run looks healthy, move to quick mode.
4. Use the analyzer directly on any existing output directory when you need to re-read or regenerate the report.

## 3. Smoke run

Smoke mode is the smallest end-to-end run. It is meant to validate the stack quickly.

```bash
rtk bash scripts/run_hgm_research_smoke.sh
```

What it does:

- Creates a timestamped directory like `output_research_smoke_YYYYMMDD_HHMMSS`.
- Runs HGM with a small budget.
- Streams stdout and stderr through `tee` into `<output_dir>/run.log`.
- Uses unbuffered Python so the log stays live while the run is in progress.
- Runs `scripts/analyze_hgm_research_run.py` after the HGM job finishes.
- Prints the report and figures locations.

## 4. Quick run

Quick mode is the preferred short experiment when smoke mode is already working.

```bash
rtk bash scripts/run_hgm_research_quick.sh
```

What it does:

- Creates a timestamped directory like `output_research_quick_YYYYMMDD_HHMMSS`.
- Runs a larger budget than smoke mode, but still keeps the experiment bounded.
- Streams stdout and stderr through `tee` into `<output_dir>/run.log`.
- Uses unbuffered Python so the log stays live while the run is in progress.
- Runs the analyzer at the end.
- Prints the report and figures locations.

## 5. Analyzer only

Use this when a run already exists and you want to regenerate the report or inspect the logs again.

```bash
rtk .venv/bin/python scripts/analyze_hgm_research_run.py --output_dir <output_dir>
```

Optional flags:

```bash
rtk .venv/bin/python scripts/analyze_hgm_research_run.py \
  --output_dir <output_dir> \
  --min_descendant_evals_for_empirical_cmp 1 \
  --no_write_figures
```

Outputs written by the analyzer:

- `<output_dir>/research/progress.md`
- `<output_dir>/research/node_table.csv`
- `<output_dir>/research/expansion_events.csv`
- `<output_dir>/research/evaluation_events.csv`
- `<output_dir>/research/final_selection.csv`
- `<output_dir>/research/figures/*.png`

## 6. Where to look first

- `<output_dir>/run.log` for the raw execution log.
- `<output_dir>/research/progress.md` for the human-readable summary.
- `<output_dir>/research/events.jsonl` for the expansion and evaluation event stream.
- `<output_dir>/research/snapshots.jsonl` for periodic tree snapshots.
- `<output_dir>/hgm_metadata.jsonl` for compact per-snapshot run metadata.
- Per-child directories like `<output_dir>/<child_run_id>/metadata.json`, `self_evo.md`, and `model_patch.diff`.

## 7. Resetting a failed experiment

Reset means removing only the experiment output directory. Do not delete tracked source files. Do not delete Docker images.

For these HGM research runs, there is a dedicated reset script:

```bash
rtk bash scripts/reset_hgm_research_state.sh
rtk bash scripts/reset_hgm_research_state.sh --docker
```

Use the `--docker` form after changes to `.dockerignore` or any files that need to be rebuilt into the child image.

### 7.1 Stop the active job

If the HGM process is still running, stop it first:

```bash
rtk ps -eo pid,ppid,pgid,cmd | rg 'hgm.py|run_hgm_research'
```

Then send `TERM` to the matching PID:

```bash
rtk kill -TERM <pid>
```

If the wrapper script is still active and you know the exact shell PID, stop that process instead of guessing.

### 7.2 Remove only the failed output directory

Delete the specific run directory you want to rerun:

```bash
rtk rm -rf <output_dir>
```

Examples:

```bash
rtk rm -rf output_research_smoke_20260702_101500
rtk rm -rf output_research_quick_20260702_141230
```

If you want to clear all local research outputs from this repo, target only the generated experiment folders:

```bash
rtk rm -rf output_research_smoke_* output_research_quick_*
```

## 8. Rerun after failure

After cleanup, rerun the same script:

```bash
rtk bash scripts/run_hgm_research_smoke.sh
```

or:

```bash
rtk bash scripts/run_hgm_research_quick.sh
```

If the run failed because of a transient LLM or Docker issue, keep the code unchanged and rerun into a fresh timestamped output directory.

## 9. Common recovery workflow

1. Open `<output_dir>/run.log` and find the first real error.
2. Run the analyzer if the output directory still exists.
3. Stop the active process if it is still alive.
4. Delete the failed output directory.
5. Rerun smoke mode first if the failure looks environmental.
6. Move back to quick mode once smoke mode is clean.

## 10. Short command summary

```bash
rtk bash scripts/run_hgm_research_smoke.sh
rtk bash scripts/run_hgm_research_quick.sh
rtk .venv/bin/python scripts/analyze_hgm_research_run.py --output_dir <output_dir>
rtk ps -eo pid,ppid,pgid,cmd | rg 'hgm.py|run_hgm_research'
rtk kill -TERM <pid>
rtk rm -rf <output_dir>
```

## 11. Notes

- The scripts intentionally do not force-delete Docker images.
- The analyzer is safe to rerun on the same output directory.
- If logs are sparse or partial, the report will show `insufficient data` instead of crashing.
- If you run `hgm.py` manually, mirror the script behavior with `python -u ... |& tee <output_dir>/run.log`.
