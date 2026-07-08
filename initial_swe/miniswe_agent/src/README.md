# HGM mini-SWE-style Agent

This initial agent package is selected with:

```bash
python evaluate_agent.py --agent_path initial_swe/miniswe_agent/src
```

The entrypoint is `coding_agent.py`. It implements a self-contained
mini-SWE-agent-style loop while preserving HGM's expected task artifact:
`/hgm/model_patch.diff`.
