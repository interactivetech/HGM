"""Convenience entrypoint for the standalone HGM mini-SWE-style agent.

The HGM harness selects agents by copying `initial_swe/<name>/src/coding_agent.py`
into task containers. This wrapper lets developers run the same implementation
directly from the repository root.
"""

import runpy
from pathlib import Path


if __name__ == "__main__":
    agent_path = (
        Path(__file__).resolve().parent
        / "initial_swe"
        / "miniswe_agent"
        / "src"
        / "coding_agent.py"
    )
    runpy.run_path(str(agent_path), run_name="__main__")
