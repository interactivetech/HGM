# Self-Improvement Prompt Analysis

This note explains how the latest diagnosis prompt was assembled, what the diagnose LLM focused on, and what the self-improvement prompt asked the model to change in the July 2, 2026 run.

## Sources used

- Diagnosis prompt template: [prompts/self_improvement_prompt.py](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/prompts/self_improvement_prompt.py:70)
- Diagnosis assembly and fallback logic: [self_improve_step.py](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/self_improve_step.py:30)
- Self-improve wrapper: [coding_agent.py](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/coding_agent.py:189)
- Latest run log: [output_qwen_budget_10_20260702_121253/20260702_124549_727452/self_improve.log](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/output_qwen_budget_10_20260702_121253/20260702_124549_727452/self_improve.log:47)
- Final prompt payload: [output_qwen_budget_10_20260702_121253/20260702_124549_727452/metadata.json](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/output_qwen_budget_10_20260702_121253/20260702_124549_727452/metadata.json)
- Self-improvement prompt artifact: [output_qwen_budget_10_20260702_121253/20260702_124549_727452/self_evo.md](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/output_qwen_budget_10_20260702_121253/20260702_124549_727452/self_evo.md)
- Produced patch artifact: [output_qwen_budget_10_20260702_121253/20260702_124549_727452/model_patch.diff](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/output_qwen_budget_10_20260702_121253/20260702_124549_727452/model_patch.diff)

## 1. Diagnosis prompt template

The diagnosis prompt is built in [prompts/self_improvement_prompt.py](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/prompts/self_improvement_prompt.py:85).

It gives the diagnose LLM four kinds of input:

- the agent running log
- the GitHub issue text
- the predicted patch
- the private test patch plus the issue test results

It also forces the model to return a JSON object with five fields:

- `log_summarization`
- `potential_improvements`
- `improvement_proposal`
- `implementation_suggestion`
- `problem_description`

The key point is that the diagnosis stage is not asked to fix the specific benchmark issue. It is asked to turn one failure case into a general improvement plan for the coding agent.

The system message for this step is also explicit that the model should inspect the agent implementation and produce one general plan that improves overall coding ability. See [prompts/self_improvement_prompt.py](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/prompts/self_improvement_prompt.py:70).

## 2. What the diagnosis LLM focused on

The latest run selected `sphinx-doc__sphinx-9229` as the target task. That is visible in [self_improve.log](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/output_qwen_budget_10_20260702_121253/20260702_124549_727452/self_improve.log:46).

The diagnosis call itself started at line 47 of the same log, completed in about 41 seconds, and produced a non-empty problem statement. The logged problem statement is the important part: it did not narrow in on a single repo-specific bug. Instead, it generalized the failure into a missing agent workflow.

The diagnosis focused on this pattern:

- the agent should not do one patch and stop
- it should run tests after each edit
- it should read test failures and feed them into the next attempt
- it should keep state controlled by resetting to the base commit when needed
- it should expose a readable test-result summary to the LLM

In other words, the model decided the core weakness was lack of an iterative test-and-fix loop.

## 3. Self-improvement prompt template

The self-improvement prompt is assembled in two layers.

First, [prompts/self_improvement_prompt.py](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/prompts/self_improvement_prompt.py:182) builds the inner prompt:

- a short coding-agent summary
- a `# To Implement` block
- the detailed issue description for the proposed improvement

Second, [coding_agent.py](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/coding_agent.py:189) wraps that problem statement in the normal repository task shell:

- repository path
- `<problem_description>`
- `<test_description>`
- instruction to modify files in the repo and use the provided dependencies

So the final self-improvement model sees the diagnosis output as a normal coding task, not as a meta-analysis task.

The handoff logic in [self_improve_step.py](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/self_improve_step.py:64) does two things:

- it parses JSON from the diagnose model when possible
- it falls back to plain-text parsing if JSON extraction fails

## 4. What the self-improvement LLM decided to improve

The final diagnosed improvement, as logged in [metadata.json](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/output_qwen_budget_10_20260702_121253/20260702_124549_727452/metadata.json), was a bounded, test-driven loop inside `AgenticSystem.forward()`.

The requested behavior was:

- add a fixed iteration cap instead of an open-ended loop
- call `chat_with_agent` on each attempt
- capture the diff after each patch
- run a specific regression test after the edit
- parse the test output and detect pass/fail
- if tests fail, feed the failure back into the next prompt
- reset to the base commit when state needs to be cleared
- add a helper for formatting test results into an LLM-friendly summary

That is the exact improvement theme the model chose: make the agent verify itself after every edit, rather than relying on a single patch.

The generated patch artifact then tried to move in that direction. A quick search of [model_patch.diff](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/output_qwen_budget_10_20260702_121253/20260702_124549_727452/model_patch.diff) shows work around:

- `tools/bash.py`
- `utils/test_runner.py`

So the implementation attempt also aligned with the diagnosis: better test execution and better feedback plumbing.

## 5. Final prompt digest

The final self-improvement prompt, as recorded in [self_evo.md](/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/06302026_hgm_self_improve_agent/HGM/output_qwen_budget_10_20260702_121253/20260702_124549_727452/self_evo.md), asked the model to implement a bounded iterative workflow in the agent:

- generate a patch
- run the relevant test
- inspect failures
- revise the patch
- stop when tests pass or the iteration cap is reached

It also asked for a small helper to make test results readable inside the prompt, and it explicitly rejected unbounded `while True` style control flow.

## Bottom line

The diagnosis stage did not pick a repository-specific bug fix. It learned a general lesson from the selected failure case: the agent needs a tighter test-and-refine loop. The self-improvement prompt then turned that lesson into a concrete coding task, and the generated patch attempted to build the supporting test tooling around it.
