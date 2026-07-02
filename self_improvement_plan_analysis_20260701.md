# Self-Improvement Plan Analysis

This note explains how the latest self-improvement plan was chosen in the run dated July 1, 2026, and which files were involved in that decision.

## Latest run examined

- Sweep log: [qwen_budget_sweep_20260701_103639.log](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/qwen_budget_sweep_20260701_103639.log)
- Mirrored run log: [output_qwen_budget_10_20260701_103639/run.log](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/output_qwen_budget_10_20260701_103639/run.log)
- Container-local self-improvement log: [output_qwen_budget_10_20260701_103639/20260701_110912_137823/self_improve.log](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/output_qwen_budget_10_20260701_103639/20260701_110912_137823/self_improve.log)

The key events in the latest run are logged around:

- [output_qwen_budget_10_20260701_103639/run.log](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/output_qwen_budget_10_20260701_103639/run.log:55202)
- [qwen_budget_sweep_20260701_103639.log](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/qwen_budget_sweep_20260701_103639.log:55203)

## How the plan was chosen

The self-improvement pipeline is:

1. Evaluate the current agent on a batch of benchmark tasks.
2. Build a report of resolved, unresolved, and empty-patch tasks.
3. Choose one target entry to optimize for.
4. Ask the diagnose LLM to turn that target into a GitHub-issue-style improvement prompt.
5. Feed that generated prompt into the self-improvement agent.

### 1. Task selection logic

The target task is chosen in [`choose_entry()` in hgm_utils.py](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/hgm_utils.py:117).

For SWE runs, the logic is heuristic:

- sometimes pick `solve_empty_patches`
- sometimes pick `solve_stochasticity`
- sometimes pick `solve_contextlength`
- otherwise pick a random unresolved instance

In the latest run, the selected target was:

- `sphinx-doc__sphinx-9281`

This is visible in:

- [output_qwen_budget_10_20260701_103639/run.log](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/output_qwen_budget_10_20260701_103639/run.log:55203)
- [output_qwen_budget_10_20260701_103639/20260701_110912_137823/self_improve.log](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/output_qwen_budget_10_20260701_103639/20260701_110912_137823/self_improve.log:43)

### 2. Diagnosis step

The diagnose step is implemented in [`self_improve_step.py`](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/self_improve_step.py:54).

It calls:

- [`get_diagnose_prompt_swe()`](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/prompts/self_improvement_prompt.py:362)
- then `get_response_from_llm(...)`
- then `extract_json_between_markers(...)`
- then `get_problem_description_prompt(...)`

If JSON extraction fails, it falls back to plain-text parsing using:

- [`_fallback_problem_statement_from_text()`](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/self_improve_step.py:25)

### 3. Prompt material used for diagnosis

For a concrete task like `sphinx-doc__sphinx-9281`, [`get_diagnose_prompt_swe()`](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/prompts/self_improvement_prompt.py:362) pulls in:

- the task markdown/chat log if present
- the task eval log
- the model patch from the prediction JSON
- the benchmark answer patch
- the benchmark test patch
- the original GitHub issue text
- a summary of the current agent code

The helper that locates the task artifacts is:

- [`find_selfimprove_eval_logs()`](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/prompts/self_improvement_prompt.py:245)

## Specific files used for this latest diagnosis

### A. Run-level files

- [output_qwen_budget_10_20260701_103639/run.log](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/output_qwen_budget_10_20260701_103639/run.log)
- [qwen_budget_sweep_20260701_103639.log](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/qwen_budget_sweep_20260701_103639.log)
- [output_qwen_budget_10_20260701_103639/20260701_110912_137823/self_improve.log](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/output_qwen_budget_10_20260701_103639/20260701_110912_137823/self_improve.log)

### B. Task artifacts for the selected target

The selected task was `sphinx-doc__sphinx-9281`, so the diagnosis path can use:

- [output_qwen_budget_10_20260701_103639/initial/predictions/default_agent_1/sphinx-doc__sphinx-9281.json](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/output_qwen_budget_10_20260701_103639/initial/predictions/default_agent_1/sphinx-doc__sphinx-9281.json)
- [output_qwen_budget_10_20260701_103639/initial/predictions/default_agent_1/sphinx-doc__sphinx-9281_docker.log](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/output_qwen_budget_10_20260701_103639/initial/predictions/default_agent_1/sphinx-doc__sphinx-9281_docker.log)
- [output_qwen_budget_10_20260701_103639/initial/predictions/default_agent_1/sphinx-doc__sphinx-9281_eval.sh](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/output_qwen_budget_10_20260701_103639/initial/predictions/default_agent_1/sphinx-doc__sphinx-9281_eval.sh)

Notes:

- `find_selfimprove_eval_logs()` looks for a per-task markdown file `sphinx-doc__sphinx-9281.md`, but in this run only the JSON, Docker log, and eval script are clearly present in the copied `initial` prediction set.
- For eval results, the code can also look under `logs/run_evaluation/.../report.json` if available.

### C. Agent code summary used by the diagnose prompt

The diagnose system prompt includes the current code snapshot from:

- [coding_agent.py](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/coding_agent.py)
- [tools/](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/tools)
- [utils/](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/utils)

with exclusions defined in:

- [prompts/self_improvement_prompt.py](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/prompts/self_improvement_prompt.py:392)

Excluded from the code summary:

- `utils/evo_utils.py`
- `utils/docker_utils.py`
- `utils/swe_log_parsers.py`
- `prompts/self_improvement_prompt.py`

## Why this plan was produced

This run did not choose the generic `solve_empty_patches` path. It chose a concrete unresolved task, `sphinx-doc__sphinx-9281`, and asked the diagnose model to generalize from that task’s failure.

Because of that, the resulting proposal focused on:

- iterative retries inside `AgenticSystem.forward()`
- test execution after each attempt
- feeding test failures back into the next attempt

That is why the latest plan is an iterative test-driven loop rather than the earlier empty-patch retry proposal.

## Final diagnosed self-improvement prompt

This is the exact `problem_statement` logged by the latest run before it launched self-improvement.

```md
# Coding Agent Summary

- **Main File**: `coding_agent.py`
  - Primary Class: `AgenticSystem`
  - The `forward()` function is the central entry point.
  - Prompts are located either within the `forward()` function or in the `prompts/` directory.
- **Tools**: `tools/`
  - The `tools/` directory contains various tools that LLMs can use to perform specific tasks.
  - Each tool must have a `tool_info()` function that returns a JSON object containing 'name', 'description', and 'input_schema'. The 'input_schema' should be a JSON object containing 'type', 'properties', and 'required'.
  - Each tool must have a `tool_function()` function that takes the arguments defined in input_schema, performs the tool's task, and returns a string.
  - See other tools for reference.
- **Utilities**: `utils/`
  - The `utils/` directory contains utility functions used across the codebase.

- **Additional Details**:
  - The agent is very good at automatically utilizing the right available tools at the right time. So do not have an agentic flow that explicitly forces a tool's usage.
  - Common tools, such as file editing and bash commands, are easy for the agent to recognize and use appropriately. However, more complex and niche tools may require explicit instructions in the prompt.
  - Tools should be designed to be as general as possible, ensuring they work across any GitHub repository. Avoid hardcoding repository-specific details or behaviors (e.g., paths).
  - Do not use 'while True' loops in the agent's code. This can cause the agent to get stuck and not respond.
  - Verify the implementation details of helper functions prior to usage to ensure proper integration and expected behavior.
  - Do not install additional packages or dependencies directly. Update `requirements.txt` if new dependencies are required and install them using `pip install -r requirements.txt`.


# To Implement

Modify `coding_agent.py` to introduce a bounded `for` loop with a max iteration limit in the `forward()` method. Within each iteration: (1) Call `chat_with_agent` with the problem statement, current diff, and any previous test feedback. (2) Use the `bash` tool to run the specific test file (e.g., `pytest tests/test_util_inspect.py::test_object_description_enum -v`). (3) Parse the test output to check for failures or errors. (4) If tests pass, break the loop and finalize the patch. (5) If tests fail, construct a new prompt containing the test output and current diff, asking the LLM to fix the identified issues. Ensure state is managed properly by tracking diff changes and using `reset_to_commit` from `utils/git_utils.py` if necessary. Add a helper function to format test results into a readable string for the LLM prompt. Strictly avoid `while True` loops by enforcing a fixed maximum iteration count.

Implement Iterative Test-Driven Development Loop in `AgenticSystem.forward()`

Currently, the agent makes a single LLM call and exits, preventing it from debugging or refining solutions based on test feedback. This issue proposes adding a bounded iterative loop to `forward()` that enables test-driven development. The agent should generate code, run relevant tests, evaluate results, and iteratively refine its implementation until tests pass or a maximum iteration limit is reached. This requires integrating test execution via the `bash` tool, parsing test outputs, and feeding failures back into subsequent LLM prompts. The implementation must avoid infinite loops by enforcing a strict iteration cap and properly managing repository state across iterations.
```

## Actual prompt wrapper passed into self-improvement

After the diagnosis step, the self-improvement agent does not receive only the block above. It is wrapped by the normal `coding_agent.py` task instruction:

```text
I have uploaded a Python code repository in the directory {self.git_tempdir}. Help solve the following problem.

<problem_description>
{self.problem_statement}
</problem_description>

<test_description>
{self.test_description}
</test_description>

Your task is to make changes to the files in the {self.git_tempdir} directory to address the <problem_description>. I have already taken care of the required dependencies.
```

This wrapper is constructed in:

- [coding_agent.py](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/coding_agent.py)

## Bottom line

The latest plan was driven by one sampled unresolved task, `sphinx-doc__sphinx-9281`, not by a global optimizer over all failures. The diagnose model turned that task’s failure context into a generic improvement proposal, and that proposal became the self-improvement prompt shown above.
