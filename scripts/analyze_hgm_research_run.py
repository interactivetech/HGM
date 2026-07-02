#!/usr/bin/env python3
"""Analyze an HGM research run from the local research JSONL logs."""

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.hgm_research_analysis import analyze_hgm_research_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_descendant_evals_for_empirical_cmp", type=int, default=1)
    parser.add_argument("--write_markdown", dest="write_markdown", action="store_true")
    parser.add_argument(
        "--no_write_markdown", dest="write_markdown", action="store_false"
    )
    parser.add_argument("--write_figures", dest="write_figures", action="store_true")
    parser.add_argument("--no_write_figures", dest="write_figures", action="store_false")
    parser.set_defaults(write_markdown=True, write_figures=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = analyze_hgm_research_run(
        args.output_dir,
        write_markdown=args.write_markdown,
        write_figures=args.write_figures,
        min_descendant_evals_for_empirical_cmp=args.min_descendant_evals_for_empirical_cmp,
    )

    research_dir = Path(result["research_dir"])
    progress_path = research_dir / "progress.md"
    print(f"Wrote {progress_path}")
    for name, path in sorted(result.get("figure_paths", {}).items()):
        if path:
            print(f"Figure {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
