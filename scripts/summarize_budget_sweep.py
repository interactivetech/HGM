#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import re
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def infer_budget(run_dir: str, run_summary: Optional[Dict], run_config: Optional[Dict]) -> Optional[int]:
    if run_summary and run_summary.get("max_task_evals") is not None:
        return run_summary["max_task_evals"]
    if run_config:
        return run_config.get("config", {}).get("execution", {}).get("max_task_evals")
    match = re.search(r"output_qwen_budget_(\d+)", os.path.basename(run_dir))
    return int(match.group(1)) if match else None


def benchmark_mode(run_summary: Optional[Dict], run_config: Optional[Dict]) -> str:
    full_eval = False
    if run_summary:
        full_eval = bool(run_summary.get("full_eval"))
    if run_config:
        full_eval = bool(
            run_config.get("config", {}).get("evaluation", {}).get("full_eval", full_eval)
        )
    if full_eval:
        return "full SWE-Verified"
    return "small/medium subset"


def count_log_patterns(run_dir: str) -> Dict[str, int]:
    patterns = {
        "tool_call_failures": r"tool call|tool-call|Error in get_response_withtools",
        "json_parse_failures": r"Invalid tool-call JSON arguments|JSONDecodeError|failed to parse tool arguments",
        "timeout_count": r"Timeout reached|Timed out|timeout",
    }
    counts = {key: 0 for key in patterns}
    for root, _dirs, files in os.walk(run_dir):
        for fname in files:
            if not fname.endswith((".log", ".md", ".txt", ".json")):
                continue
            path = os.path.join(root, fname)
            try:
                with open(path, "r", errors="ignore") as f:
                    content = f.read()
            except OSError:
                continue
            for key, pattern in patterns.items():
                counts[key] += len(re.findall(pattern, content, flags=re.IGNORECASE))
    return counts


def collect_node_rows(run_dir: str) -> List[Dict]:
    rows = []
    for child in os.listdir(run_dir):
        child_dir = os.path.join(run_dir, child)
        if not os.path.isdir(child_dir) or child == "initial":
            continue
        metadata = load_json(os.path.join(child_dir, "metadata.json")) or {}
        overall = metadata.get("overall_performance", {})
        rows.append(
            {
                "commit_id": child,
                "status": metadata.get("status"),
                "submitted_instances": overall.get("total_submitted_instances", 0),
                "resolved_instances": overall.get("total_resolved_instances", 0),
                "accuracy_score": overall.get("accuracy_score", 0.0),
                "path": child_dir,
            }
        )
    return rows


def summarize_run(run_dir: str) -> Dict:
    run_summary = load_json(os.path.join(run_dir, "run_summary.json"))
    run_config = load_json(os.path.join(run_dir, "run_config.json"))
    node_rows = collect_node_rows(run_dir)
    log_counts = count_log_patterns(run_dir)

    best_node = max(node_rows, key=lambda row: row["accuracy_score"], default=None)
    wall_clock_seconds = (
        run_summary.get("wall_clock_seconds")
        if run_summary
        else None
    )
    max_workers = None
    if run_summary:
        max_workers = run_summary.get("max_workers")
    if max_workers is None and run_config:
        max_workers = run_config.get("config", {}).get("execution", {}).get("max_workers")
    max_workers = max_workers or 1

    row = {
        "run_directory": run_dir,
        "budget": infer_budget(run_dir, run_summary, run_config),
        "benchmark_mode": benchmark_mode(run_summary, run_config),
        "max_workers": max_workers,
        "wall_clock_seconds": wall_clock_seconds or 0,
        "allocated_worker_hours": ((wall_clock_seconds or 0) * max_workers) / 3600.0,
        "num_nodes": len(node_rows),
        "child_agents_created": sum(1 for row in node_rows if row["status"] == "success"),
        "failed_self_modifications": sum(1 for row in node_rows if row["status"] == "failed"),
        "total_submitted_task_evaluations": (
            run_summary.get("n_task_evals") if run_summary else sum(r["submitted_instances"] for r in node_rows)
        ),
        "total_resolved_task_evaluations": sum(r["resolved_instances"] for r in node_rows),
        "best_found_accuracy": (best_node["accuracy_score"] * 100.0) if best_node else 0.0,
        "best_agent_commit_id": best_node["commit_id"] if best_node else "",
        "best_agent_path": best_node["path"] if best_node else "",
        "tool_call_failure_count": log_counts["tool_call_failures"],
        "json_parse_failure_count": log_counts["json_parse_failures"],
        "timeout_count": log_counts["timeout_count"],
    }
    return row


def write_outputs(rows: List[Dict], csv_path: str, json_path: str):
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)


def make_plot(rows: List[Dict], x_key: str, y_key: str, out_path: str, xlabel: str):
    rows = sorted(rows, key=lambda row: (row[x_key], row["budget"] or 0))
    x = [row[x_key] for row in rows]
    y = [row[y_key] for row in rows]
    labels = [str(row["budget"]) for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    for x_val, y_val, label in zip(x, y, labels):
        plt.annotate(label, (x_val, y_val), textcoords="offset points", xytext=(5, 5))
    plt.xlabel(xlabel)
    plt.ylabel("Best-found accuracy (%)")
    plt.title("Qwen HGM budget scaling")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", dest="glob_pattern", default="output_qwen_budget_*")
    parser.add_argument("--csv", default="qwen_budget_sweep.csv")
    parser.add_argument("--json", default="qwen_budget_sweep.json")
    parser.add_argument(
        "--plot-budget", default="qwen_best_accuracy_vs_budget.png"
    )
    parser.add_argument(
        "--plot-hours", default="qwen_best_accuracy_vs_allocated_hours.png"
    )
    args = parser.parse_args()

    run_dirs = sorted(
        path for path in glob.glob(args.glob_pattern) if os.path.isdir(path)
    )
    rows = [summarize_run(run_dir) for run_dir in run_dirs]
    rows = [row for row in rows if row["budget"] is not None]
    if not rows:
        raise SystemExit("No matching output_qwen_budget_* directories found.")

    write_outputs(rows, args.csv, args.json)
    make_plot(rows, "budget", "best_found_accuracy", args.plot_budget, "Max task evals")
    make_plot(
        rows,
        "allocated_worker_hours",
        "best_found_accuracy",
        args.plot_hours,
        "Allocated worker-hours",
    )
    print(f"Wrote {args.csv}, {args.json}, {args.plot_budget}, and {args.plot_hours}")


if __name__ == "__main__":
    main()
