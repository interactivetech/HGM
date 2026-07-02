import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import hgm
import hgm_utils
from tree import Node
from utils.hgm_research_logger import (
    ResearchLogger,
    collect_tree_stats,
    select_greedy_current_performance,
)


@pytest.fixture
def isolated_tree():
    original_nodes = hgm_utils.nodes
    original_n_task_evals = hgm_utils.n_task_evals
    hgm_utils.nodes = {}
    hgm_utils.n_task_evals = 0
    try:
        yield
    finally:
        hgm_utils.nodes = original_nodes
        hgm_utils.n_task_evals = original_n_task_evals


def test_tree_clade_stats(isolated_tree):
    root = Node("root", utility_measures=[1.0, 0.0])
    left = Node("left", utility_measures=[1.0], parent_id=root.id)
    right = Node("right", utility_measures=[0.0, 1.0], parent_id=root.id)
    leaf = Node("leaf", utility_measures=[1.0, 1.0], parent_id=left.id)
    root.add_child(left)
    root.add_child(right)
    left.add_child(leaf)

    stats = {item["node_id"]: item for item in collect_tree_stats(hgm_utils.nodes)}
    root_stats = stats[root.id]

    assert root_stats["depth"] == 0
    assert root_stats["subtree_size"] == 4
    assert root_stats["num_descendants"] == 3
    assert root_stats["direct_num_evals"] == 2
    assert root_stats["clade_num_evals"] == 7
    assert root_stats["clade_successes"] == 5
    assert root_stats["clade_failures"] == 2
    assert math.isclose(root_stats["estimated_cmp"], 5 / 7)


def test_best_descendant_excludes_self(isolated_tree):
    root = Node("root", utility_measures=[1.0, 1.0])
    child = Node("child", utility_measures=[0.0, 0.0], parent_id=root.id)
    grandchild = Node("grandchild", utility_measures=[0.5, 0.5], parent_id=child.id)
    root.add_child(child)
    child.add_child(grandchild)

    stats = {item["node_id"]: item for item in collect_tree_stats(hgm_utils.nodes)}

    assert math.isclose(stats[root.id]["direct_mean_utility"], 1.0)
    assert math.isclose(stats[root.id]["best_descendant_direct_mean_utility_excluding_self"], 0.5)
    assert math.isclose(stats[child.id]["best_descendant_direct_mean_utility_excluding_self"], 0.5)
    assert stats[grandchild.id]["best_descendant_direct_mean_utility_excluding_self"] is None


def test_greedy_current_performance_selection(isolated_tree):
    root = Node("root", utility_measures=[0.0, 0.0])
    left = Node("left", utility_measures=[1.0], parent_id=root.id)
    right = Node("right", utility_measures=[0.5], parent_id=root.id)
    root.add_child(left)
    root.add_child(right)

    selected = select_greedy_current_performance([root, left, right])

    assert selected is left
    assert selected.id == left.id


def test_thompson_sampling_diagnostics(monkeypatch):
    monkeypatch.setattr(hgm.np.random, "beta", lambda alphas, betas: np.array([0.2, 0.8]))

    selected_index, details = hgm.TS_sample(
        [[1, 0], [0, 0]],
        return_details=True,
        opt_cfg=SimpleNamespace(cool_down=False, beta=1.0),
        exec_cfg=SimpleNamespace(max_task_evals=10),
        current_n_task_evals=3,
    )

    assert selected_index == 1
    assert details["alphas"] == [2.0, 1.0]
    assert details["betas"] == [2.0, 3.0]
    assert details["thetas"] == [0.2, 0.8]
    assert details["selected_index"] == 1


def test_research_logger_writes_valid_jsonl(isolated_tree):
    root = Node("root", utility_measures=[1.0])
    child = Node("child", utility_measures=[0.0, 1.0], parent_id=root.id)
    root.add_child(child)

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ResearchLogger(tmpdir, enabled=True)
        logger.log_event(
            "diagnostic_event",
            {"value": 7, "nested": {"flag": True}, "bad": float("inf")},
        )
        logger.write_snapshot(hgm_utils.nodes.values(), 3, extra={"phase": "test"})

        events_path = Path(tmpdir) / "research" / "events.jsonl"
        snapshots_path = Path(tmpdir) / "research" / "snapshots.jsonl"

        event_record = json.loads(events_path.read_text(encoding="utf-8").strip())
        snapshot_record = json.loads(snapshots_path.read_text(encoding="utf-8").strip())

        assert event_record["event_type"] == "diagnostic_event"
        assert event_record["payload"]["value"] == 7
        assert event_record["payload"]["bad"] is None
        assert snapshot_record["n_task_evals"] == 3
        assert snapshot_record["extra"]["phase"] == "test"
        assert snapshot_record["nodes"][0]["node_id"] == root.id
