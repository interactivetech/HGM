import json
import math
import subprocess
import sys
from pathlib import Path

from utils.hgm_research_analysis import (
    analyze_hgm_research_run,
    build_final_selection_rows,
    compute_runtime_estimates,
    pearson,
    weighted_pearson,
)
from utils.evo_utils import load_hgm_metadata


def _write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _write_synthetic_run(output_dir: Path):
    research_dir = output_dir / "research"
    research_dir.mkdir(parents=True, exist_ok=True)

    snapshots = [
        {
            "timestamp": 1.0,
            "run_id": "synthetic",
            "output_dir": str(output_dir),
            "n_task_evals": 0,
            "num_nodes": 1,
            "nodes": [
                {
                    "node_id": 0,
                    "commit_id": "initial",
                    "parent_id": None,
                    "depth": 0,
                    "num_children": 0,
                    "subtree_size": 1,
                    "direct_successes": 2,
                    "direct_failures": 1,
                    "direct_num_evals": 3,
                    "direct_mean_utility": 2 / 3,
                    "clade_successes": 2,
                    "clade_failures": 1,
                    "clade_num_evals": 3,
                    "estimated_cmp": 2 / 3,
                    "best_descendant_direct_mean_utility_excluding_self": None,
                    "num_descendants": 0,
                    "created_count": 0,
                    "evaluated_count": 0,
                    "expanded_count": 0,
                }
            ],
        },
        {
            "timestamp": 2.0,
            "run_id": "synthetic",
            "output_dir": str(output_dir),
            "n_task_evals": 1,
            "num_nodes": 3,
            "nodes": [
                {
                    "node_id": 0,
                    "commit_id": "initial",
                    "parent_id": None,
                    "depth": 0,
                    "num_children": 2,
                    "subtree_size": 3,
                    "direct_successes": 2,
                    "direct_failures": 1,
                    "direct_num_evals": 3,
                    "direct_mean_utility": 2 / 3,
                    "clade_successes": 5,
                    "clade_failures": 2,
                    "clade_num_evals": 7,
                    "estimated_cmp": 5 / 7,
                    "best_descendant_direct_mean_utility_excluding_self": 1.0,
                    "num_descendants": 2,
                    "created_count": 0,
                    "evaluated_count": 0,
                    "expanded_count": 0,
                },
                {
                    "node_id": 1,
                    "commit_id": "c1",
                    "parent_id": 0,
                    "depth": 1,
                    "num_children": 0,
                    "subtree_size": 1,
                    "direct_successes": 1,
                    "direct_failures": 1,
                    "direct_num_evals": 2,
                    "direct_mean_utility": 0.5,
                    "clade_successes": 1,
                    "clade_failures": 1,
                    "clade_num_evals": 2,
                    "estimated_cmp": 0.5,
                    "best_descendant_direct_mean_utility_excluding_self": None,
                    "num_descendants": 0,
                    "created_count": 1,
                    "evaluated_count": 0,
                    "expanded_count": 1,
                },
                {
                    "node_id": 2,
                    "commit_id": "c2",
                    "parent_id": 0,
                    "depth": 1,
                    "num_children": 0,
                    "subtree_size": 1,
                    "direct_successes": 3,
                    "direct_failures": 0,
                    "direct_num_evals": 3,
                    "direct_mean_utility": 1.0,
                    "clade_successes": 3,
                    "clade_failures": 0,
                    "clade_num_evals": 3,
                    "estimated_cmp": 1.0,
                    "best_descendant_direct_mean_utility_excluding_self": None,
                    "num_descendants": 0,
                    "created_count": 0,
                    "evaluated_count": 0,
                    "expanded_count": 0,
                },
            ],
        },
        {
            "timestamp": 3.0,
            "run_id": "synthetic",
            "output_dir": str(output_dir),
            "n_task_evals": 2,
            "num_nodes": 4,
            "nodes": [
                {
                    "node_id": 0,
                    "commit_id": "initial",
                    "parent_id": None,
                    "depth": 0,
                    "num_children": 2,
                    "subtree_size": 4,
                    "direct_successes": 2,
                    "direct_failures": 1,
                    "direct_num_evals": 3,
                    "direct_mean_utility": 2 / 3,
                    "clade_successes": 6,
                    "clade_failures": 4,
                    "clade_num_evals": 10,
                    "estimated_cmp": 0.6,
                    "best_descendant_direct_mean_utility_excluding_self": 1.0,
                    "num_descendants": 3,
                    "created_count": 0,
                    "evaluated_count": 0,
                    "expanded_count": 0,
                },
                {
                    "node_id": 1,
                    "commit_id": "c1",
                    "parent_id": 0,
                    "depth": 1,
                    "num_children": 1,
                    "subtree_size": 2,
                    "direct_successes": 2,
                    "direct_failures": 1,
                    "direct_num_evals": 3,
                    "direct_mean_utility": 2 / 3,
                    "clade_successes": 4,
                    "clade_failures": 2,
                    "clade_num_evals": 6,
                    "estimated_cmp": 2 / 3,
                    "best_descendant_direct_mean_utility_excluding_self": 1.0,
                    "num_descendants": 1,
                    "created_count": 1,
                    "evaluated_count": 1,
                    "expanded_count": 1,
                },
                {
                    "node_id": 2,
                    "commit_id": "c2",
                    "parent_id": 0,
                    "depth": 1,
                    "num_children": 0,
                    "subtree_size": 1,
                    "direct_successes": 3,
                    "direct_failures": 0,
                    "direct_num_evals": 3,
                    "direct_mean_utility": 1.0,
                    "clade_successes": 3,
                    "clade_failures": 0,
                    "clade_num_evals": 3,
                    "estimated_cmp": 1.0,
                    "best_descendant_direct_mean_utility_excluding_self": None,
                    "num_descendants": 0,
                    "created_count": 0,
                    "evaluated_count": 0,
                    "expanded_count": 0,
                },
                {
                    "node_id": 3,
                    "commit_id": "c3",
                    "parent_id": 1,
                    "depth": 2,
                    "num_children": 0,
                    "subtree_size": 1,
                    "direct_successes": 1,
                    "direct_failures": 0,
                    "direct_num_evals": 1,
                    "direct_mean_utility": 1.0,
                    "clade_successes": 1,
                    "clade_failures": 0,
                    "clade_num_evals": 1,
                    "estimated_cmp": 1.0,
                    "best_descendant_direct_mean_utility_excluding_self": None,
                    "num_descendants": 0,
                    "created_count": 1,
                    "evaluated_count": 0,
                    "expanded_count": 0,
                },
            ],
        },
    ]

    events = [
        {
            "timestamp": 1.5,
            "run_id": "synthetic",
            "output_dir": str(output_dir),
            "event_type": "expansion_decision",
            "payload": {
                "current_n_task_evals": 1,
                "selected_node_id": 1,
                "selected_commit_id": "c1",
                "selected_index": 0,
                "greedy_current_performance_node_id": 2,
                "greedy_estimated_cmp_node_id": 2,
                "selected_same_as_greedy_current_performance": False,
                "selected_same_as_greedy_estimated_cmp": False,
                "thompson": {
                    "alphas": [2.0, 1.0],
                    "betas": [2.0, 2.0],
                    "thetas": [0.2, 0.8],
                    "selected_index": 0,
                },
                "config": {
                    "alpha": 0.6,
                    "beta": 1.0,
                    "cool_down": False,
                    "n_pseudo_descendant_evals": 100,
                },
                "n_pending_expands": 0,
                "n_pending_measures": 0,
            },
        },
        {
            "timestamp": 2.2,
            "run_id": "synthetic",
            "output_dir": str(output_dir),
            "event_type": "expansion_result",
            "payload": {
                "parent_node_id": 1,
                "parent_commit_id": "c1",
                "child_commit_id": "c3",
                "status": "success",
                "elapsed_seconds": 5.0,
                "new_child_id": 3,
            },
        },
        {
            "timestamp": 2.5,
            "run_id": "synthetic",
            "output_dir": str(output_dir),
            "event_type": "evaluation_result",
            "payload": {
                "current_n_task_evals": 1,
                "global_n_task_evals": 2,
                "selected_node_id": 1,
                "selected_commit_id": "c1",
                "selected_task_id": "task-1",
                "result_values": [1],
                "updated_direct_mean_utility": 2 / 3,
                "updated_direct_eval_count": 3,
                "updated_estimated_cmp": 2 / 3,
                "greedy_current_performance_node_id": 2,
                "selected_same_as_greedy_current_performance": False,
                "elapsed_seconds": 2.0,
                "available_task_count": 4,
                "task_choice_mode": "random",
                "task_choice_random": True,
            },
        },
    ]

    _write_jsonl(research_dir / "snapshots.jsonl", snapshots)
    _write_jsonl(research_dir / "events.jsonl", events)

    (output_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "run_id": "synthetic",
                "wall_clock_seconds": 12.0,
                "run_started_at_epoch": 1.0,
                "run_finished_at_epoch": 13.0,
                "max_task_evals": 8,
                "max_workers": 3,
                "polyglot": False,
                "full_eval": False,
                "n_task_evals": 2,
                "num_non_initial_nodes": 3,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "config": {
                    "execution": {
                        "max_workers": 3,
                        "self_improve_timeout": 600,
                        "evaluation_timeout": 600,
                        "max_task_evals": 8,
                    },
                    "optimization": {
                        "alpha": 0.6,
                    },
                    "research": {
                        "enabled": True,
                        "log_policy_details": True,
                    },
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    child_dir = output_dir / "c3"
    child_dir.mkdir(parents=True, exist_ok=True)
    (child_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": "child-123",
                "entry": "solve_stochasticity",
                "problem_statement": "Investigate an off-by-one error in the evaluation branch.",
                "overall_performance": {
                    "accuracy_score": 0.0,
                    "total_resolved_instances": 0,
                    "total_submitted_instances": 0,
                    "files": [],
                    "total_submitted_ids": [],
                    "total_unresolved_ids": [],
                    "total_emptypatch_ids": [],
                    "total_resolved_ids": [],
                },
                "status": "success",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (child_dir / "self_evo.md").write_text("Self improvement notes", encoding="utf-8")
    (child_dir / "model_patch.diff").write_text(
        "\n".join(
            [
                "diff --git a/foo.py b/foo.py",
                "--- a/foo.py",
                "+++ b/foo.py",
                "@@ -1 +1 @@",
                "-print('old')",
                "+print('new')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_analyzer_writes_outputs_and_report(tmp_path):
    output_dir = tmp_path / "synthetic_run"
    _write_synthetic_run(output_dir)

    result = analyze_hgm_research_run(str(output_dir))

    research_dir = output_dir / "research"
    assert (research_dir / "progress.md").exists()
    assert (research_dir / "node_table.csv").exists()
    assert (research_dir / "expansion_events.csv").exists()
    assert (research_dir / "evaluation_events.csv").exists()
    assert (research_dir / "final_selection.csv").exists()
    assert (research_dir / "figures" / "progress_best_utility.png").exists()
    assert result["runtime_estimates"]["avg_expansion_seconds"] > 0
    assert result["runtime_estimates"]["avg_evaluation_seconds"] > 0

    report = (research_dir / "progress.md").read_text(encoding="utf-8")
    assert "HGM Research Analysis" in report
    assert "Diagnosis and self-improvement changes" in report
    assert "run summary" in report.lower()
    assert "self_evo.md" in report
    assert "model_patch.diff" in report
    assert "metadata.json" in report


def test_correlation_helpers_and_runtime_estimates():
    assert math.isclose(pearson([1, 2, 3], [2, 4, 6]), 1.0)
    assert math.isclose(weighted_pearson([1, 2, 3], [2, 4, 6], [1, 1, 1]), 1.0)

    runtime = compute_runtime_estimates(
        {"config": {"execution": {"max_workers": 4, "self_improve_timeout": 30, "evaluation_timeout": 10}, "optimization": {"alpha": 0.5}}},
        {"max_workers": 4},
        [{"elapsed_seconds": 6.0}],
        [{"elapsed_seconds": 3.0}],
        budgets=(10,),
    )
    assert runtime["budget_rows"][0]["lower_bound_parallel_hours"] > 0
    assert runtime["budget_rows"][0]["timeout_upper_bound_hours"] > 0


def test_final_selection_policies_work_on_synthetic_node_table():
    node_rows = [
        {"node_id": 0, "commit_id": "initial", "direct_mean_utility": 0.6, "estimated_cmp": 0.7, "direct_num_evals": 3, "clade_num_evals": 7, "depth": 0, "subtree_size": 4, "expanded_count": 0},
        {"node_id": 1, "commit_id": "c1", "direct_mean_utility": 0.75, "estimated_cmp": 0.8, "direct_num_evals": 4, "clade_num_evals": 6, "depth": 1, "subtree_size": 2, "expanded_count": 3},
        {"node_id": 2, "commit_id": "c2", "direct_mean_utility": 0.9, "estimated_cmp": 0.85, "direct_num_evals": 5, "clade_num_evals": 5, "depth": 1, "subtree_size": 1, "expanded_count": 1},
    ]
    expansion_events = [
        {"status": "success", "child_commit_id": "c1", "timestamp": 1.0, "event_index": 0},
        {"status": "failed", "child_commit_id": "failed", "timestamp": 2.0, "event_index": 1},
    ]

    rows = build_final_selection_rows(node_rows, expansion_events)
    by_policy = {row["policy_name"]: row for row in rows}

    assert by_policy["best_direct_mean_utility"]["selected_node_id"] == 2
    assert by_policy["best_estimated_cmp"]["selected_node_id"] == 2
    assert by_policy["most_evaluated_node"]["selected_node_id"] == 2
    assert by_policy["most_expanded_node"]["selected_node_id"] == 1
    assert by_policy["latest_successful_child"]["commit_id"] == "c1"


def test_cli_entrypoint_writes_progress_md(tmp_path):
    output_dir = tmp_path / "synthetic_cli_run"
    _write_synthetic_run(output_dir)

    subprocess.run(
        [
            sys.executable,
            str(Path("scripts") / "analyze_hgm_research_run.py"),
            "--output_dir",
            str(output_dir),
        ],
        check=True,
        cwd=Path.cwd(),
    )

    assert (output_dir / "research" / "progress.md").exists()


def test_load_hgm_metadata_supports_jsonl_and_legacy_multiline(tmp_path):
    jsonl_path = tmp_path / "metadata.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps({"n_task_evals": 0, "nodes": []}),
                json.dumps({"n_task_evals": 1, "nodes": [{"id": 1}]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_hgm_metadata(str(jsonl_path), last_only=True)["n_task_evals"] == 1

    legacy_path = tmp_path / "metadata_legacy.jsonl"
    legacy_path.write_text(
        json.dumps({"n_task_evals": 0, "nodes": []}, indent=2)
        + "\n"
        + json.dumps({"n_task_evals": 2, "nodes": [{"id": 2}]}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    assert load_hgm_metadata(str(legacy_path), last_only=True)["n_task_evals"] == 2
