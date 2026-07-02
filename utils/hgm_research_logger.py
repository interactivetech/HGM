# This file is adapted from https://github.com/jennyzzt/dgm.

import json
import math
import os
import time
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


def _as_node_list(nodes: Iterable[Any], sort_by_node_id: bool = False) -> List[Any]:
    if nodes is None:
        return []
    if isinstance(nodes, Mapping):
        node_list = list(nodes.values())
    else:
        node_list = list(nodes)
    if sort_by_node_id:
        node_list.sort(key=lambda node: getattr(node, "id", 0))
    return node_list


def _build_node_map(nodes: Iterable[Any]) -> Dict[Any, Any]:
    node_list = _as_node_list(nodes, sort_by_node_id=False)
    return {getattr(node, "id"): node for node in node_list}


def _to_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _utility_values(node: Any) -> List[float]:
    values = getattr(node, "utility_measures", []) or []
    output = []
    for value in values:
        numeric = _to_float(value)
        if numeric is not None:
            output.append(numeric)
    return output


def _count_successes(values: Sequence[float]) -> int:
    return sum(1 for value in values if value > 0)


def _compute_depth(node: Any, node_map: Mapping[Any, Any], depth_cache: MutableMapping[Any, Optional[int]]) -> Optional[int]:
    node_id = getattr(node, "id", None)
    if node_id in depth_cache:
        return depth_cache[node_id]

    parent_id = getattr(node, "parent_id", None)
    if parent_id is None or parent_id not in node_map:
        depth = 0 if parent_id is None else None
    else:
        parent = node_map[parent_id]
        parent_depth = _compute_depth(parent, node_map, depth_cache)
        depth = None if parent_depth is None else parent_depth + 1

    depth_cache[node_id] = depth
    return depth


def _summarize_node_recursive(
    node: Any,
    node_map: Optional[Mapping[Any, Any]],
    depth_cache: MutableMapping[Any, Optional[int]],
    memo: MutableMapping[Any, Tuple[Dict[str, Any], Optional[float]]],
) -> Tuple[Dict[str, Any], Optional[float]]:
    node_id = getattr(node, "id", None)
    if node_id in memo:
        return memo[node_id]

    direct_values = _utility_values(node)
    direct_num_evals = len(direct_values)
    direct_successes = _count_successes(direct_values)
    direct_failures = direct_num_evals - direct_successes
    direct_mean_utility = (
        sum(direct_values) / direct_num_evals if direct_num_evals > 0 else None
    )
    children = list(getattr(node, "children", []) or [])

    subtree_size = 1
    clade_successes = direct_successes
    clade_failures = direct_failures
    clade_num_evals = direct_num_evals
    best_descendant_direct_mean = None
    best_direct_mean_in_subtree = direct_mean_utility

    for child in children:
        child_stats, child_best_in_subtree = _summarize_node_recursive(
            child, node_map, depth_cache, memo
        )
        subtree_size += child_stats["subtree_size"]
        clade_successes += child_stats["clade_successes"]
        clade_failures += child_stats["clade_failures"]
        clade_num_evals += child_stats["clade_num_evals"]
        if child_best_in_subtree is not None:
            if (
                best_descendant_direct_mean is None
                or child_best_in_subtree > best_descendant_direct_mean
            ):
                best_descendant_direct_mean = child_best_in_subtree
            if (
                best_direct_mean_in_subtree is None
                or child_best_in_subtree > best_direct_mean_in_subtree
            ):
                best_direct_mean_in_subtree = child_best_in_subtree

    depth = None
    if node_map is not None:
        depth = _compute_depth(node, node_map, depth_cache)
    else:
        parent_id = getattr(node, "parent_id", None)
        if parent_id is None:
            depth = 0

    stats = {
        "node_id": node_id,
        "commit_id": getattr(node, "commit_id", None),
        "parent_id": getattr(node, "parent_id", None),
        "depth": depth,
        "num_children": len(children),
        "subtree_size": subtree_size,
        "direct_successes": direct_successes,
        "direct_failures": direct_failures,
        "direct_num_evals": direct_num_evals,
        "direct_mean_utility": direct_mean_utility,
        "clade_successes": clade_successes,
        "clade_failures": clade_failures,
        "clade_num_evals": clade_num_evals,
        "estimated_cmp": (
            clade_successes / clade_num_evals if clade_num_evals > 0 else None
        ),
        "best_descendant_direct_mean_utility_excluding_self": best_descendant_direct_mean,
        "num_descendants": max(subtree_size - 1, 0),
    }
    memo[node_id] = (stats, best_direct_mean_in_subtree)
    return memo[node_id]


def build_node_stats(node: Any, node_map: Optional[Mapping[Any, Any]] = None) -> Dict[str, Any]:
    memo: Dict[Any, Tuple[Dict[str, Any], Optional[float]]] = {}
    depth_cache: Dict[Any, Optional[int]] = {}
    stats, _ = _summarize_node_recursive(node, node_map, depth_cache, memo)
    return stats


def collect_tree_stats(nodes: Iterable[Any]) -> List[Dict[str, Any]]:
    node_list = _as_node_list(nodes, sort_by_node_id=True)
    node_map = _build_node_map(node_list)
    memo: Dict[Any, Tuple[Dict[str, Any], Optional[float]]] = {}
    depth_cache: Dict[Any, Optional[int]] = {}
    return [
        _summarize_node_recursive(node, node_map, depth_cache, memo)[0]
        for node in node_list
    ]


def select_greedy_current_performance(nodes: Iterable[Any]) -> Optional[Any]:
    best_node = None
    best_key = None
    for node in _as_node_list(nodes, sort_by_node_id=False):
        direct_values = _utility_values(node)
        if not direct_values:
            continue
        direct_mean = sum(direct_values) / len(direct_values)
        key = (direct_mean, -int(getattr(node, "id", 0)))
        if best_key is None or key > best_key:
            best_key = key
            best_node = node
    return best_node


def select_greedy_estimated_cmp(nodes: Iterable[Any]) -> Optional[Any]:
    best_node = None
    best_key = None
    for node in _as_node_list(nodes, sort_by_node_id=False):
        stats = build_node_stats(node)
        estimated_cmp = stats.get("estimated_cmp")
        if estimated_cmp is None:
            continue
        key = (estimated_cmp, -int(getattr(node, "id", 0)))
        if best_key is None or key > best_key:
            best_key = key
            best_node = node
    return best_node


def build_candidate_policy_summary(
    nodes: Iterable[Any], node_map: Optional[Mapping[Any, Any]] = None
) -> Dict[str, Any]:
    node_list = _as_node_list(nodes, sort_by_node_id=False)
    node_map = node_map or _build_node_map(node_list)
    stats = [build_node_stats(node, node_map=node_map) for node in node_list]

    greedy_current_stats = None
    greedy_current_key = None
    greedy_estimated_stats = None
    greedy_estimated_key = None
    for item in stats:
        node_id = item.get("node_id")
        direct_mean = item.get("direct_mean_utility")
        estimated_cmp = item.get("estimated_cmp")
        if direct_mean is not None:
            direct_key = (direct_mean, -int(node_id) if node_id is not None else 0)
            if greedy_current_key is None or direct_key > greedy_current_key:
                greedy_current_key = direct_key
                greedy_current_stats = item
        if estimated_cmp is not None:
            cmp_key = (estimated_cmp, -int(node_id) if node_id is not None else 0)
            if greedy_estimated_key is None or cmp_key > greedy_estimated_key:
                greedy_estimated_key = cmp_key
                greedy_estimated_stats = item

    return {
        "candidate_stats": stats,
        "candidate_node_ids": [item["node_id"] for item in stats],
        "candidate_commit_ids": [item["commit_id"] for item in stats],
        "candidate_direct_mean_utilities": [
            item["direct_mean_utility"] for item in stats
        ],
        "candidate_direct_eval_counts": [item["direct_num_evals"] for item in stats],
        "candidate_direct_successes": [item["direct_successes"] for item in stats],
        "candidate_direct_failures": [item["direct_failures"] for item in stats],
        "candidate_clade_successes": [item["clade_successes"] for item in stats],
        "candidate_clade_failures": [item["clade_failures"] for item in stats],
        "candidate_clade_num_evals": [item["clade_num_evals"] for item in stats],
        "candidate_estimated_cmp_values": [item["estimated_cmp"] for item in stats],
        "greedy_current_performance_node_id": (
            greedy_current_stats["node_id"] if greedy_current_stats is not None else None
        ),
        "greedy_current_performance_commit_id": (
            greedy_current_stats["commit_id"] if greedy_current_stats is not None else None
        ),
        "greedy_estimated_cmp_node_id": (
            greedy_estimated_stats["node_id"] if greedy_estimated_stats is not None else None
        ),
        "greedy_estimated_cmp_commit_id": (
            greedy_estimated_stats["commit_id"]
            if greedy_estimated_stats is not None
            else None
        ),
    }


def _sanitize_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_jsonable(item) for item in value]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _sanitize_jsonable(value.item())
        except Exception:
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, bool) or value is None or isinstance(value, int) or isinstance(value, str):
        return value
    if hasattr(value, "as_posix"):
        try:
            return value.as_posix()
        except Exception:
            pass
    return str(value)


class ResearchLogger:
    def __init__(self, output_dir: str, enabled: bool = True):
        self.output_dir = os.path.abspath(output_dir)
        self.run_id = os.path.basename(os.path.normpath(self.output_dir))
        self.enabled = bool(enabled)
        self.research_dir = os.path.join(self.output_dir, "research")
        self.events_path = os.path.join(self.research_dir, "events.jsonl")
        self.snapshots_path = os.path.join(self.research_dir, "snapshots.jsonl")
        if self.enabled:
            try:
                os.makedirs(self.research_dir, exist_ok=True)
            except Exception:
                self.enabled = False

    def _append_jsonl(self, path: str, record: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            payload = json.dumps(_sanitize_jsonable(record), ensure_ascii=False, allow_nan=False)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(payload + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            return

    def log_event(self, event_type: str, payload: Any) -> None:
        if not self.enabled:
            return
        record = {
            "timestamp": time.time(),
            "run_id": self.run_id,
            "output_dir": self.output_dir,
            "event_type": event_type,
            "payload": payload,
        }
        self._append_jsonl(self.events_path, record)

    def write_snapshot(
        self, nodes: Iterable[Any], n_task_evals: int, extra: Optional[Any] = None
    ) -> None:
        if not self.enabled:
            return
        try:
            node_stats = collect_tree_stats(nodes)
            record = {
                "timestamp": time.time(),
                "run_id": self.run_id,
                "output_dir": self.output_dir,
                "n_task_evals": n_task_evals,
                "num_nodes": len(node_stats),
                "nodes": node_stats,
            }
            if extra is not None:
                record["extra"] = extra
            self._append_jsonl(self.snapshots_path, record)
        except Exception:
            return
