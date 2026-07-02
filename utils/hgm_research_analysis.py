# This file is adapted from https://github.com/jennyzzt/dgm.

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from utils.common_utils import load_json_file
from utils.evo_utils import load_hgm_metadata


def is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def to_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def to_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def safe_text(value: Any, limit: int = 240) -> str:
    if value is None:
        return ""
    text = str(value).strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def format_float(value: Any, digits: int = 4) -> str:
    numeric = to_float_or_none(value)
    if numeric is None:
        return ""
    return f"{numeric:.{digits}f}"


def format_int(value: Any) -> str:
    integer = to_int_or_none(value)
    return "" if integer is None else str(integer)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def load_optional_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return load_json_file(str(path))
    except Exception:
        return None


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if is_finite_number(x) and is_finite_number(y)
    ]
    if len(pairs) < 2:
        return None
    x_vals = [x for x, _ in pairs]
    y_vals = [y for _, y in pairs]
    x_mean = mean(x_vals)
    y_mean = mean(y_vals)
    cov = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    var_x = sum((x - x_mean) ** 2 for x in x_vals)
    var_y = sum((y - y_mean) ** 2 for y in y_vals)
    if var_x <= 0 or var_y <= 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def weighted_pearson(
    xs: Sequence[float], ys: Sequence[float], weights: Sequence[float]
) -> Optional[float]:
    triples = [
        (float(x), float(y), float(w))
        for x, y, w in zip(xs, ys, weights)
        if is_finite_number(x) and is_finite_number(y) and is_finite_number(w) and float(w) > 0
    ]
    if len(triples) < 2:
        return None
    total_weight = sum(w for _, _, w in triples)
    if total_weight <= 0:
        return None
    x_mean = sum(x * w for x, _, w in triples) / total_weight
    y_mean = sum(y * w for _, y, w in triples) / total_weight
    cov = sum(w * (x - x_mean) * (y - y_mean) for x, y, w in triples) / total_weight
    var_x = sum(w * (x - x_mean) ** 2 for x, _, w in triples) / total_weight
    var_y = sum(w * (y - y_mean) ** 2 for _, y, w in triples) / total_weight
    if var_x <= 0 or var_y <= 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    cleaned = sorted(v for v in values if is_finite_number(v))
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return float(cleaned[0])
    q = max(0.0, min(100.0, float(q)))
    position = (len(cleaned) - 1) * (q / 100.0)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(cleaned[lower])
    lower_weight = upper - position
    upper_weight = position - lower
    return float(cleaned[lower] * lower_weight + cleaned[upper] * upper_weight)


def summarize_durations(values: Sequence[float]) -> Dict[str, Optional[float]]:
    cleaned = [float(v) for v in values if is_finite_number(v) and float(v) >= 0]
    if not cleaned:
        return {"avg": None, "p50": None, "p90": None}
    return {
        "avg": mean(cleaned),
        "p50": percentile(cleaned, 50),
        "p90": percentile(cleaned, 90),
    }


def load_run_artifacts(output_dir: Path) -> Dict[str, Any]:
    research_dir = output_dir / "research"
    events = read_jsonl(research_dir / "events.jsonl")
    snapshots = read_jsonl(research_dir / "snapshots.jsonl")
    run_summary = load_optional_json(output_dir / "run_summary.json")
    run_config = load_optional_json(output_dir / "run_config.json")
    hgm_metadata = None
    hgm_metadata_path = output_dir / "hgm_metadata.jsonl"
    if hgm_metadata_path.exists():
        try:
            hgm_metadata = load_hgm_metadata(str(hgm_metadata_path), last_only=True)
        except Exception:
            hgm_metadata = None
    return {
        "output_dir": output_dir,
        "research_dir": research_dir,
        "events": events,
        "snapshots": snapshots,
        "run_summary": run_summary,
        "run_config": run_config,
        "hgm_metadata": hgm_metadata,
    }


def _final_snapshot_nodes(snapshots: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not snapshots:
        return []
    final_snapshot = max(
        snapshots,
        key=lambda item: (
            to_int_or_none(item.get("n_task_evals")) or -1,
            item.get("timestamp", 0),
        ),
    )
    nodes = final_snapshot.get("nodes", []) or []
    return [dict(node) for node in nodes]


def _metadata_node_rows(output_dir: Path, hgm_metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    node_entries = (hgm_metadata or {}).get("nodes", []) or []
    rows: List[Dict[str, Any]] = []
    for node in node_entries:
        node = dict(node)
        node_id = node.get("id")
        commit_id = node.get("commit_id")
        parent_id = node.get("parent_id")
        metadata = load_optional_json(output_dir / str(commit_id) / "metadata.json")
        overall = (metadata or {}).get("overall_performance", {})
        direct_successes = to_int_or_none(overall.get("total_resolved_instances")) or 0
        direct_num_evals = to_int_or_none(overall.get("total_submitted_instances")) or 0
        direct_failures = max(direct_num_evals - direct_successes, 0)
        rows.append(
            {
                "node_id": node_id,
                "commit_id": commit_id,
                "parent_id": parent_id,
                "depth": None,
                "num_children": 0,
                "subtree_size": 1,
                "direct_successes": direct_successes,
                "direct_failures": direct_failures,
                "direct_num_evals": direct_num_evals,
                "direct_mean_utility": (
                    direct_successes / direct_num_evals if direct_num_evals > 0 else None
                ),
                "clade_successes": direct_successes,
                "clade_failures": direct_failures,
                "clade_num_evals": direct_num_evals,
                "estimated_cmp": (
                    direct_successes / direct_num_evals if direct_num_evals > 0 else None
                ),
                "best_descendant_direct_mean_utility_excluding_self": None,
                "num_descendants": 0,
            }
        )
    return rows


def _compute_tree_fallback(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    by_id = {row.get("node_id"): row for row in rows}
    children: Dict[Any, List[Any]] = defaultdict(list)
    for row in rows:
        parent_id = row.get("parent_id")
        node_id = row.get("node_id")
        if parent_id in by_id:
            children[parent_id].append(node_id)

    depth_cache: Dict[Any, Optional[int]] = {}
    memo: Dict[Any, Tuple[Dict[str, Any], Optional[float]]] = {}

    def compute_depth(node_id: Any) -> Optional[int]:
        if node_id in depth_cache:
            return depth_cache[node_id]
        row = by_id.get(node_id, {})
        parent_id = row.get("parent_id")
        if parent_id is None or parent_id not in by_id:
            depth_cache[node_id] = 0 if parent_id is None else None
            return depth_cache[node_id]
        parent_depth = compute_depth(parent_id)
        depth_cache[node_id] = None if parent_depth is None else parent_depth + 1
        return depth_cache[node_id]

    def rec(node_id: Any) -> Tuple[Dict[str, Any], Optional[float]]:
        if node_id in memo:
            return memo[node_id]
        row = dict(by_id[node_id])
        direct_successes = to_int_or_none(row.get("direct_successes")) or 0
        direct_failures = to_int_or_none(row.get("direct_failures"))
        direct_num_evals = to_int_or_none(row.get("direct_num_evals")) or (
            direct_successes + (direct_failures or 0)
        )
        if direct_failures is None:
            direct_failures = max(direct_num_evals - direct_successes, 0)
        direct_mean_utility = (
            direct_successes / direct_num_evals if direct_num_evals > 0 else None
        )
        subtree_size = 1
        clade_successes = direct_successes
        clade_failures = direct_failures
        clade_num_evals = direct_num_evals
        best_descendant = None
        best_in_subtree = direct_mean_utility
        for child_id in children.get(node_id, []):
            child_stats, child_best = rec(child_id)
            subtree_size += child_stats["subtree_size"]
            clade_successes += child_stats["clade_successes"]
            clade_failures += child_stats["clade_failures"]
            clade_num_evals += child_stats["clade_num_evals"]
            if child_best is not None:
                if best_descendant is None or child_best > best_descendant:
                    best_descendant = child_best
                if best_in_subtree is None or child_best > best_in_subtree:
                    best_in_subtree = child_best
        stats = {
            "node_id": node_id,
            "commit_id": row.get("commit_id"),
            "parent_id": row.get("parent_id"),
            "depth": compute_depth(node_id),
            "num_children": len(children.get(node_id, [])),
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
            "best_descendant_direct_mean_utility_excluding_self": best_descendant,
            "num_descendants": max(subtree_size - 1, 0),
        }
        memo[node_id] = (stats, best_in_subtree)
        return memo[node_id]

    return [rec(node_id)[0] for node_id in by_id]


def _merge_event_counts(rows: List[Dict[str, Any]], events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    created = defaultdict(int)
    evaluated = defaultdict(int)
    expanded = defaultdict(int)
    for event in events:
        payload = event.get("payload", {}) or {}
        event_type = event.get("event_type")
        if event_type == "expansion_result":
            node_id = payload.get("new_child_id")
            if node_id is not None and payload.get("status") == "success":
                created[node_id] += 1
        elif event_type == "evaluation_result":
            node_id = payload.get("selected_node_id")
            if node_id is not None:
                evaluated[node_id] += 1
        elif event_type == "expansion_decision":
            node_id = payload.get("selected_node_id")
            if node_id is not None:
                expanded[node_id] += 1
    for row in rows:
        node_id = row.get("node_id")
        row["created_count"] = created.get(node_id, 0)
        row["evaluated_count"] = evaluated.get(node_id, 0)
        row["expanded_count"] = expanded.get(node_id, 0)
    return rows


def build_node_table(output_dir: Path, artifacts: Dict[str, Any], min_descendant_evals_for_empirical_cmp: int = 1) -> List[Dict[str, Any]]:
    rows = _final_snapshot_nodes(artifacts["snapshots"])
    if not rows:
        rows = _metadata_node_rows(output_dir, artifacts.get("hgm_metadata"))
        rows = _compute_tree_fallback(rows)
    rows = [dict(row) for row in rows]

    for row in rows:
        direct_successes = to_int_or_none(row.get("direct_successes")) or 0
        direct_failures = to_int_or_none(row.get("direct_failures"))
        direct_num_evals = to_int_or_none(row.get("direct_num_evals")) or (
            direct_successes + (direct_failures or 0)
        )
        if direct_failures is None:
            direct_failures = max(direct_num_evals - direct_successes, 0)
        clade_successes = to_int_or_none(row.get("clade_successes")) or direct_successes
        clade_failures = to_int_or_none(row.get("clade_failures"))
        clade_num_evals = to_int_or_none(row.get("clade_num_evals")) or direct_num_evals
        if clade_failures is None:
            clade_failures = max(clade_num_evals - clade_successes, 0)
        row["direct_successes"] = direct_successes
        row["direct_failures"] = direct_failures
        row["direct_num_evals"] = direct_num_evals
        row["direct_mean_utility"] = (
            direct_successes / direct_num_evals if direct_num_evals > 0 else None
        )
        row["clade_successes"] = clade_successes
        row["clade_failures"] = clade_failures
        row["clade_num_evals"] = clade_num_evals
        row["estimated_cmp"] = (
            clade_successes / clade_num_evals if clade_num_evals > 0 else None
        )
        row["subtree_size"] = to_int_or_none(row.get("subtree_size")) or 1
        row["num_descendants"] = to_int_or_none(row.get("num_descendants")) or max(
            row["subtree_size"] - 1, 0
        )
        row["num_children"] = to_int_or_none(row.get("num_children")) or 0
        if row.get("depth") is not None:
            row["depth"] = to_int_or_none(row.get("depth"))
        descendant_evals = max(clade_num_evals - direct_num_evals, 0)
        descendant_successes = max(clade_successes - direct_successes, 0)
        if descendant_evals >= min_descendant_evals_for_empirical_cmp and descendant_evals > 0:
            row["empirical_future_cmp_excluding_self"] = descendant_successes / descendant_evals
        else:
            row["empirical_future_cmp_excluding_self"] = None
    rows = _merge_event_counts(rows, artifacts["events"])
    rows.sort(key=lambda row: (to_int_or_none(row.get("node_id")) or -1))
    return rows


def _lookup_candidate_stats(payload: Dict[str, Any], node_id: Any) -> Dict[str, Any]:
    candidate_stats = payload.get("candidate_stats", []) or []
    for stat in candidate_stats:
        if stat.get("node_id") == node_id:
            return stat
    return {}


def _load_child_artifacts_from_disk(output_dir: Path, child_commit: Any) -> Dict[str, Any]:
    if child_commit is None:
        return {}
    child_dir = output_dir / str(child_commit)
    metadata = load_optional_json(child_dir / "metadata.json") or {}
    diff_path = child_dir / "model_patch.diff"
    diff_summary = {"patch_line_count": None, "changed_files": []}
    if diff_path.exists():
        try:
            diff_text = diff_path.read_text(encoding="utf-8", errors="replace")
            diff_summary["patch_line_count"] = len(diff_text.splitlines())
            changed_files = []
            for line in diff_text.splitlines():
                if line.startswith("diff --git a/"):
                    parts = line.split()
                    if len(parts) >= 4 and parts[3].startswith("b/"):
                        changed_files.append(parts[3][2:])
            diff_summary["changed_files"] = list(dict.fromkeys(changed_files))
        except Exception:
            pass
    return {
        "child_run_dir": str(child_dir) if child_dir.exists() else None,
        "child_metadata_path": str(child_dir / "metadata.json")
        if (child_dir / "metadata.json").exists()
        else None,
        "self_evo_path": str(child_dir / "self_evo.md")
        if (child_dir / "self_evo.md").exists()
        else None,
        "model_patch_path": str(diff_path) if diff_path.exists() else None,
        "diagnosis_entry": metadata.get("entry"),
        "problem_statement_preview": safe_text(metadata.get("problem_statement")),
        "patch_line_count": diff_summary["patch_line_count"],
        "changed_files": diff_summary["changed_files"],
    }


def build_expansion_events(artifacts: Dict[str, Any], node_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    node_lookup = {row.get("node_id"): row for row in node_rows}
    output_dir = artifacts.get("output_dir")
    rows = []
    for index, event in enumerate(artifacts["events"]):
        if event.get("event_type") != "expansion_decision":
            continue
        payload = event.get("payload", {}) or {}
        selected_node_id = payload.get("selected_node_id")
        selected_row = node_lookup.get(selected_node_id, {})
        selected_stat = _lookup_candidate_stats(payload, selected_node_id)
        greedy_current_id = payload.get("greedy_current_performance_node_id")
        greedy_estimated_id = payload.get("greedy_estimated_cmp_node_id")
        greedy_current_stat = _lookup_candidate_stats(payload, greedy_current_id)
        greedy_estimated_stat = _lookup_candidate_stats(payload, greedy_estimated_id)
        rows.append(
            {
                "event_index": index,
                "timestamp": event.get("timestamp"),
                "n_task_evals": payload.get("current_n_task_evals"),
                "selected_node_id": selected_node_id,
                "selected_commit_id": payload.get("selected_commit_id"),
                "greedy_current_performance_node_id": greedy_current_id,
                "greedy_estimated_cmp_node_id": greedy_estimated_id,
                "selected_same_as_greedy_current_performance": payload.get(
                    "selected_same_as_greedy_current_performance"
                ),
                "selected_same_as_greedy_estimated_cmp": payload.get(
                    "selected_same_as_greedy_estimated_cmp"
                ),
                "selected_node_direct_mean_utility": selected_row.get("direct_mean_utility")
                if selected_row
                else selected_stat.get("direct_mean_utility"),
                "selected_node_estimated_cmp": selected_row.get("estimated_cmp")
                if selected_row
                else selected_stat.get("estimated_cmp"),
                "greedy_current_performance_direct_mean_utility": greedy_current_stat.get(
                    "direct_mean_utility"
                ),
                "greedy_current_performance_estimated_cmp": greedy_current_stat.get(
                    "estimated_cmp"
                ),
                "greedy_estimated_cmp_direct_mean_utility": greedy_estimated_stat.get(
                    "direct_mean_utility"
                ),
                "greedy_estimated_cmp_estimated_cmp": greedy_estimated_stat.get(
                    "estimated_cmp"
                ),
                "child_commit_id": payload.get("child_commit_id"),
                "child_run_dir": payload.get("child_run_dir"),
                "child_metadata_path": payload.get("child_metadata_path"),
                "self_evo_path": payload.get("self_evo_path"),
                "model_patch_path": payload.get("model_patch_path"),
                "status": None,  # filled below if a matching result exists
                "elapsed_seconds": payload.get("elapsed_seconds"),
                "diagnosis_entry": payload.get("diagnosis_entry"),
                "problem_statement_preview": safe_text(
                    payload.get("problem_statement_preview")
                ),
                "patch_line_count": payload.get("patch_line_count"),
                "changed_files": ";".join(payload.get("changed_files", []) or []),
                "parent_node_id": payload.get("parent_node_id"),
                "parent_commit_id": payload.get("parent_commit_id"),
            }
        )
    used_result_events = set()
    for row in rows:
        selected_node_id = row.get("selected_node_id")
        selected_commit_id = row.get("selected_commit_id")
        matched_payload = None
        matched_index = None
        for index, event in enumerate(artifacts["events"]):
            if event.get("event_type") != "expansion_result":
                continue
            payload = event.get("payload", {}) or {}
            if index in used_result_events:
                continue
            if payload.get("parent_node_id") != selected_node_id and payload.get("parent_commit_id") != selected_commit_id:
                continue
            matched_payload = payload
            matched_index = index
            break
        if matched_payload is None:
            continue
        used_result_events.add(matched_index)
        child_commit = matched_payload.get("child_commit_id") or row.get("child_commit_id")
        fallback = (
            _load_child_artifacts_from_disk(output_dir, child_commit)
            if isinstance(output_dir, Path)
            else {}
        )
        row["child_commit_id"] = child_commit
        row["status"] = matched_payload.get("status") or row.get("status")
        row["elapsed_seconds"] = (
            row.get("elapsed_seconds") or matched_payload.get("elapsed_seconds")
        )
        row["diagnosis_entry"] = (
            matched_payload.get("diagnosis_entry")
            or fallback.get("diagnosis_entry")
            or row.get("diagnosis_entry")
        )
        row["problem_statement_preview"] = (
            row.get("problem_statement_preview")
            or safe_text(matched_payload.get("problem_statement_preview"))
            or fallback.get("problem_statement_preview")
        )
        row["patch_line_count"] = (
            row.get("patch_line_count")
            or matched_payload.get("patch_line_count")
            or fallback.get("patch_line_count")
        )
        if not row["changed_files"]:
            changed_files = (
                matched_payload.get("changed_files") or fallback.get("changed_files") or []
            )
            row["changed_files"] = ";".join(changed_files)
        if not row.get("child_run_dir"):
            row["child_run_dir"] = (
                matched_payload.get("child_run_dir") or fallback.get("child_run_dir")
            )
        if not row.get("child_metadata_path"):
            row["child_metadata_path"] = (
                matched_payload.get("child_metadata_path")
                or fallback.get("child_metadata_path")
            )
        if not row.get("self_evo_path"):
            row["self_evo_path"] = (
                matched_payload.get("self_evo_path") or fallback.get("self_evo_path")
            )
        if not row.get("model_patch_path"):
            row["model_patch_path"] = (
                matched_payload.get("model_patch_path")
                or fallback.get("model_patch_path")
            )
    return rows


def build_evaluation_events(artifacts: Dict[str, Any], node_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    node_lookup = {row.get("node_id"): row for row in node_rows}
    rows = []
    for index, event in enumerate(artifacts["events"]):
        if event.get("event_type") != "evaluation_result":
            continue
        payload = event.get("payload", {}) or {}
        selected_node_id = payload.get("selected_node_id")
        selected_row = node_lookup.get(selected_node_id, {})
        result_values = payload.get("result_values", []) or []
        result_sum = sum(v for v in result_values if is_finite_number(v))
        result_count = len(result_values)
        after_count = to_int_or_none(payload.get("updated_direct_eval_count"))
        after_mean = to_float_or_none(payload.get("updated_direct_mean_utility"))
        before_count = None
        before_mean = None
        if after_count is not None and result_count is not None:
            before_count = after_count - result_count
            if before_count > 0 and after_mean is not None:
                before_sum = after_mean * after_count - result_sum
                before_mean = before_sum / before_count
        rows.append(
            {
                "event_index": index,
                "timestamp": event.get("timestamp"),
                "n_task_evals_before": payload.get("current_n_task_evals"),
                "n_task_evals_after": payload.get("global_n_task_evals"),
                "selected_node_id": selected_node_id,
                "selected_commit_id": payload.get("selected_commit_id"),
                "selected_task_id": payload.get("selected_task_id"),
                "result_values": json.dumps(result_values),
                "selected_node_direct_mean_before": before_mean,
                "selected_node_direct_mean_after": after_mean,
                "selected_node_direct_eval_count_before": before_count,
                "selected_node_direct_eval_count_after": after_count,
                "selected_node_estimated_cmp_after": payload.get("updated_estimated_cmp"),
                "greedy_current_performance_node_id": payload.get(
                    "greedy_current_performance_node_id"
                ),
                "same_as_greedy_current_performance": payload.get(
                    "selected_same_as_greedy_current_performance"
                ),
                "elapsed_seconds": payload.get("elapsed_seconds"),
                "available_task_count": payload.get("available_task_count"),
                "task_choice_mode": payload.get("task_choice_mode"),
                "task_choice_random": payload.get("task_choice_random"),
                "direct_mean_utility_after": selected_row.get("direct_mean_utility"),
                "estimated_cmp_after": selected_row.get("estimated_cmp"),
                "direct_eval_count_after": selected_row.get("direct_num_evals"),
            }
        )
    return rows


def _best_row(rows: Sequence[Dict[str, Any]], metric: str) -> Optional[Dict[str, Any]]:
    candidates = [row for row in rows if is_finite_number(row.get(metric))]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            float(row.get(metric)),
            float(row.get("direct_num_evals") or row.get("direct_eval_count_after") or 0),
            -int(row.get("node_id") or 0),
        ),
    )


def _latest_successful_child(expansion_events: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    successes = [row for row in expansion_events if row.get("status") == "success" and row.get("child_commit_id")]
    if not successes:
        return None
    return max(
        successes,
        key=lambda row: (
            to_float_or_none(row.get("timestamp")) or -1,
            row.get("event_index") or -1,
        ),
    )


def build_final_selection_rows(
    node_rows: Sequence[Dict[str, Any]],
    expansion_events: Sequence[Dict[str, Any]],
    min_descendant_evals_for_empirical_cmp: int = 1,
) -> List[Dict[str, Any]]:
    rows = list(node_rows)
    selected_rows: List[Dict[str, Any]] = []
    policies = [
        ("best_direct_mean_utility", "direct_mean_utility"),
        ("best_estimated_cmp", "estimated_cmp"),
        ("best_empirical_future_cmp", "empirical_future_cmp_excluding_self"),
        ("most_evaluated_node", "direct_num_evals"),
        ("most_expanded_node", "expanded_count"),
    ]
    for policy_name, metric in policies:
        best = _best_row(rows, metric)
        selected_rows.append(_policy_row(policy_name, best))
    latest_success = _latest_successful_child(expansion_events)
    if latest_success:
        child_commit = latest_success.get("child_commit_id")
        selected = next((row for row in rows if row.get("commit_id") == child_commit), None)
        selected_rows.append(_policy_row("latest_successful_child", selected))
    else:
        selected_rows.append(_policy_row("latest_successful_child", None))
    return selected_rows


def _policy_row(policy_name: str, row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not row:
        return {
            "policy_name": policy_name,
            "selected_node_id": None,
            "commit_id": None,
            "direct_mean_utility": None,
            "estimated_cmp": None,
            "direct_num_evals": None,
            "clade_num_evals": None,
            "depth": None,
            "subtree_size": None,
        }
    return {
        "policy_name": policy_name,
        "selected_node_id": row.get("node_id"),
        "commit_id": row.get("commit_id"),
        "direct_mean_utility": row.get("direct_mean_utility"),
        "estimated_cmp": row.get("estimated_cmp"),
        "direct_num_evals": row.get("direct_num_evals"),
        "clade_num_evals": row.get("clade_num_evals"),
        "depth": row.get("depth"),
        "subtree_size": row.get("subtree_size"),
    }


def compute_disagreement_metrics(expansion_events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    decisions = [row for row in expansion_events if row.get("selected_node_id") is not None]
    if not decisions:
        return {
            "num_expansion_decisions": 0,
            "disagreement_current_count": 0,
            "disagreement_current_rate": None,
            "disagreement_cmp_count": 0,
            "disagreement_cmp_rate": None,
            "avg_selected_direct_utility": None,
            "avg_greedy_current_direct_utility": None,
            "avg_selected_estimated_cmp": None,
            "avg_greedy_current_estimated_cmp": None,
            "avg_selected_minus_greedy_estimated_cmp": None,
        }
    current_disagreements = 0
    cmp_disagreements = 0
    selected_direct = []
    greedy_direct = []
    selected_cmp = []
    greedy_cmp = []
    cmp_delta = []
    for row in decisions:
        selected_direct.append(to_float_or_none(row.get("selected_node_direct_mean_utility")))
        greedy_direct.append(to_float_or_none(row.get("greedy_current_performance_direct_mean_utility")))
        selected_cmp.append(to_float_or_none(row.get("selected_node_estimated_cmp")))
        greedy_cmp.append(to_float_or_none(row.get("greedy_current_performance_estimated_cmp")))
        if row.get("selected_same_as_greedy_current_performance") is False:
            current_disagreements += 1
        if row.get("selected_same_as_greedy_estimated_cmp") is False:
            cmp_disagreements += 1
        if is_finite_number(row.get("selected_node_estimated_cmp")) and is_finite_number(
            row.get("greedy_current_performance_estimated_cmp")
        ):
            cmp_delta.append(
                float(row.get("selected_node_estimated_cmp"))
                - float(row.get("greedy_current_performance_estimated_cmp"))
            )
    return {
        "num_expansion_decisions": len(decisions),
        "disagreement_current_count": current_disagreements,
        "disagreement_current_rate": current_disagreements / len(decisions),
        "disagreement_cmp_count": cmp_disagreements,
        "disagreement_cmp_rate": cmp_disagreements / len(decisions),
        "avg_selected_direct_utility": mean([v for v in selected_direct if v is not None])
        if any(v is not None for v in selected_direct)
        else None,
        "avg_greedy_current_direct_utility": mean([v for v in greedy_direct if v is not None])
        if any(v is not None for v in greedy_direct)
        else None,
        "avg_selected_estimated_cmp": mean([v for v in selected_cmp if v is not None])
        if any(v is not None for v in selected_cmp)
        else None,
        "avg_greedy_current_estimated_cmp": mean([v for v in greedy_cmp if v is not None])
        if any(v is not None for v in greedy_cmp)
        else None,
        "avg_selected_minus_greedy_estimated_cmp": mean(cmp_delta) if cmp_delta else None,
    }


def compute_correlations(
    node_rows: Sequence[Dict[str, Any]],
    min_descendant_evals_for_empirical_cmp: int = 1,
) -> List[Dict[str, Any]]:
    direct = []
    est = []
    empirical = []
    direct_weights = []
    clade_weights = []
    for row in node_rows:
        direct_mean = to_float_or_none(row.get("direct_mean_utility"))
        estimated = to_float_or_none(row.get("estimated_cmp"))
        empirical_cmp = to_float_or_none(row.get("empirical_future_cmp_excluding_self"))
        if direct_mean is not None and estimated is not None and empirical_cmp is not None:
            direct.append(direct_mean)
            est.append(estimated)
            empirical.append(empirical_cmp)
            direct_weights.append(max(to_int_or_none(row.get("direct_num_evals")) or 0, 1))
            clade_weights.append(max(to_int_or_none(row.get("clade_num_evals")) or 0, 1))

    rows = []
    if len(empirical) >= 3:
        rows.append(
            {
                "metric_pair": "direct_mean_utility vs empirical_future_cmp",
                "n": len(empirical),
                "pearson_r": pearson(direct, empirical),
                "weighted_r": weighted_pearson(direct, empirical, direct_weights),
            }
        )
        rows.append(
            {
                "metric_pair": "estimated_cmp vs empirical_future_cmp",
                "n": len(empirical),
                "pearson_r": pearson(est, empirical),
                "weighted_r": weighted_pearson(est, empirical, clade_weights),
            }
        )
        rows.append(
            {
                "metric_pair": "direct_mean_utility vs empirical_future_cmp (weighted by direct eval count)",
                "n": len(empirical),
                "pearson_r": pearson(direct, empirical),
                "weighted_r": weighted_pearson(direct, empirical, direct_weights),
            }
        )
        rows.append(
            {
                "metric_pair": "estimated_cmp vs empirical_future_cmp (weighted by clade eval count)",
                "n": len(empirical),
                "pearson_r": pearson(est, empirical),
                "weighted_r": weighted_pearson(est, empirical, clade_weights),
            }
        )
    else:
        rows.append(
            {
                "metric_pair": "insufficient data",
                "n": len(empirical),
                "pearson_r": None,
                "weighted_r": None,
            }
        )
    return rows


def compute_runtime_estimates(
    run_config: Optional[Dict[str, Any]],
    run_summary: Optional[Dict[str, Any]],
    expansion_events: Sequence[Dict[str, Any]],
    evaluation_events: Sequence[Dict[str, Any]],
    budgets: Sequence[int] = (30, 80, 200, 800),
) -> Dict[str, Any]:
    config = (run_config or {}).get("config", {})
    execution = config.get("execution", {})
    optimization = config.get("optimization", {})
    max_workers = (
        (run_summary or {}).get("max_workers")
        or execution.get("max_workers")
        or 1
    )
    self_improve_timeout = (
        execution.get("self_improve_timeout")
        or (run_summary or {}).get("self_improve_timeout")
        or 3600
    )
    evaluation_timeout = (
        execution.get("evaluation_timeout")
        or (run_summary or {}).get("evaluation_timeout")
        or 3600
    )
    alpha = optimization.get("alpha", 0.6)

    expansion_durations = [
        to_float_or_none(row.get("elapsed_seconds"))
        for row in expansion_events
        if is_finite_number(row.get("elapsed_seconds"))
    ]
    evaluation_durations = [
        to_float_or_none(row.get("elapsed_seconds"))
        for row in evaluation_events
        if is_finite_number(row.get("elapsed_seconds"))
    ]
    expansion_durations = [v for v in expansion_durations if v is not None]
    evaluation_durations = [v for v in evaluation_durations if v is not None]
    avg_expansion = mean(expansion_durations) if expansion_durations else None
    avg_evaluation = mean(evaluation_durations) if evaluation_durations else None
    p50_expansion = percentile(expansion_durations, 50)
    p90_expansion = percentile(expansion_durations, 90)
    p50_evaluation = percentile(evaluation_durations, 50)
    p90_evaluation = percentile(evaluation_durations, 90)

    budget_rows = []
    for budget in budgets:
        expansion_count = budget ** alpha
        evaluation_count = budget
        lower = None
        upper = None
        if avg_expansion is not None and avg_evaluation is not None:
            lower = (
                expansion_count * avg_expansion + evaluation_count * avg_evaluation
            ) / max_workers
        if self_improve_timeout and evaluation_timeout:
            upper = (
                expansion_count * self_improve_timeout + evaluation_count * evaluation_timeout
            ) / max_workers
        budget_rows.append(
            {
                "budget": budget,
                "estimated_expansion_count": expansion_count,
                "estimated_evaluation_count": evaluation_count,
                "lower_bound_parallel_hours": lower / 3600.0 if lower is not None else None,
                "timeout_upper_bound_hours": upper / 3600.0 if upper is not None else None,
            }
        )
    return {
        "max_workers": max_workers,
        "self_improve_timeout": self_improve_timeout,
        "evaluation_timeout": evaluation_timeout,
        "alpha": alpha,
        "avg_expansion_seconds": avg_expansion,
        "avg_evaluation_seconds": avg_evaluation,
        "p50_expansion_seconds": p50_expansion,
        "p90_expansion_seconds": p90_expansion,
        "p50_evaluation_seconds": p50_evaluation,
        "p90_evaluation_seconds": p90_evaluation,
        "budget_rows": budget_rows,
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[Tuple[str, str]], max_rows: Optional[int] = None) -> str:
    if not rows:
        return "insufficient data"
    selected = list(rows[: max_rows or len(rows)])
    header = " | ".join(label for _key, label in columns)
    separator = " | ".join("---" for _key, _label in columns)
    lines = [header, separator]
    for row in selected:
        cells = []
        for key, _label in columns:
            value = row.get(key)
            if isinstance(value, float):
                cells.append(format_float(value))
            else:
                cells.append(_csv_value(value))
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def _nodes_for_plot(node_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in node_rows if is_finite_number(row.get("direct_mean_utility"))]


def generate_figures(
    output_dir: Path,
    node_rows: Sequence[Dict[str, Any]],
    snapshot_rows: Sequence[Dict[str, Any]],
    expansion_events: Sequence[Dict[str, Any]],
    evaluation_events: Sequence[Dict[str, Any]],
    runtime_estimates: Dict[str, Any],
) -> Dict[str, Optional[Path]]:
    figures_dir = output_dir / "research" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: Dict[str, Optional[Path]] = {
        "progress_best_utility": None,
        "node_cmp_vs_direct": None,
        "cmp_vs_empirical_future_cmp": None,
        "direct_vs_empirical_future_cmp": None,
        "selection_disagreement": None,
        "tree_depth_vs_utility": None,
    }

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return figure_paths

    # Figure 1: best utility over time
    if snapshot_rows:
        points = sorted(
            [
                (
                    to_int_or_none(snapshot.get("n_task_evals")) or index,
                    snapshot,
                )
                for index, snapshot in enumerate(snapshot_rows)
            ],
            key=lambda item: item[0],
        )
        xs = [point[0] for point in points]
        best_direct = []
        best_cmp = []
        running_direct = None
        running_cmp = None
        for _x, snapshot in points:
            nodes = snapshot.get("nodes", []) or []
            direct_vals = [
                to_float_or_none(node.get("direct_mean_utility"))
                for node in nodes
                if is_finite_number(node.get("direct_mean_utility"))
            ]
            cmp_vals = [
                to_float_or_none(node.get("estimated_cmp"))
                for node in nodes
                if is_finite_number(node.get("estimated_cmp"))
            ]
            if direct_vals:
                current_direct = max(direct_vals)
                running_direct = current_direct if running_direct is None else max(running_direct, current_direct)
            if cmp_vals:
                current_cmp = max(cmp_vals)
                running_cmp = current_cmp if running_cmp is None else max(running_cmp, current_cmp)
            best_direct.append(running_direct)
            best_cmp.append(running_cmp)
        if any(v is not None for v in best_direct):
            plt.figure(figsize=(8, 5))
            plt.plot(xs, best_direct, marker="o", label="best direct mean utility so far")
            if any(v is not None for v in best_cmp):
                plt.plot(xs, best_cmp, marker="o", label="best estimated CMP so far")
            plt.xlabel("task evals")
            plt.ylabel("value")
            plt.title("Progress of best observed utility")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            out_path = figures_dir / "progress_best_utility.png"
            plt.savefig(out_path, dpi=150)
            plt.close()
            figure_paths["progress_best_utility"] = out_path

    scatter_rows = _nodes_for_plot(node_rows)
    annotate = len(scatter_rows) <= 20
    if scatter_rows:
        plt.figure(figsize=(7, 6))
        xs = [row.get("direct_mean_utility") for row in scatter_rows]
        ys = [row.get("estimated_cmp") for row in scatter_rows]
        plt.scatter(xs, ys, alpha=0.75)
        if annotate:
            for row in scatter_rows:
                plt.annotate(
                    str(row.get("node_id")),
                    (row.get("direct_mean_utility"), row.get("estimated_cmp")),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                )
        plt.xlabel("direct mean utility")
        plt.ylabel("estimated CMP")
        plt.title("Estimated CMP vs direct utility")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = figures_dir / "node_cmp_vs_direct.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        figure_paths["node_cmp_vs_direct"] = out_path

    empirical_rows = [
        row
        for row in node_rows
        if is_finite_number(row.get("empirical_future_cmp_excluding_self"))
    ]
    if len(empirical_rows) >= 3:
        plt.figure(figsize=(7, 6))
        xs = [row.get("estimated_cmp") for row in empirical_rows]
        ys = [row.get("empirical_future_cmp_excluding_self") for row in empirical_rows]
        plt.scatter(xs, ys, alpha=0.75)
        if len(empirical_rows) <= 20:
            for row in empirical_rows:
                plt.annotate(
                    str(row.get("node_id")),
                    (row.get("estimated_cmp"), row.get("empirical_future_cmp_excluding_self")),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                )
        plt.xlabel("estimated CMP")
        plt.ylabel("empirical future CMP")
        plt.title("Estimated CMP vs empirical future CMP")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = figures_dir / "cmp_vs_empirical_future_cmp.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        figure_paths["cmp_vs_empirical_future_cmp"] = out_path

        plt.figure(figsize=(7, 6))
        xs = [row.get("direct_mean_utility") for row in empirical_rows]
        ys = [row.get("empirical_future_cmp_excluding_self") for row in empirical_rows]
        plt.scatter(xs, ys, alpha=0.75)
        if len(empirical_rows) <= 20:
            for row in empirical_rows:
                plt.annotate(
                    str(row.get("node_id")),
                    (row.get("direct_mean_utility"), row.get("empirical_future_cmp_excluding_self")),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                )
        plt.xlabel("direct mean utility")
        plt.ylabel("empirical future CMP")
        plt.title("Direct utility vs empirical future CMP")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = figures_dir / "direct_vs_empirical_future_cmp.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        figure_paths["direct_vs_empirical_future_cmp"] = out_path

    if expansion_events:
        cumulative = []
        running = 0
        for idx, row in enumerate(expansion_events, start=1):
            if row.get("selected_same_as_greedy_current_performance") is False:
                running += 1
            cumulative.append((idx, running / idx))
        if cumulative:
            plt.figure(figsize=(8, 5))
            xs = [item[0] for item in cumulative]
            ys = [item[1] for item in cumulative]
            plt.plot(xs, ys, marker="o")
            plt.xlabel("expansion decision index")
            plt.ylabel("cumulative disagreement rate")
            plt.title("HGM vs greedy-current-performance disagreement")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out_path = figures_dir / "selection_disagreement.png"
            plt.savefig(out_path, dpi=150)
            plt.close()
            figure_paths["selection_disagreement"] = out_path

    depth_rows = [row for row in node_rows if row.get("depth") is not None and is_finite_number(row.get("direct_mean_utility"))]
    if depth_rows:
        plt.figure(figsize=(7, 6))
        xs = [row.get("depth") for row in depth_rows]
        ys = [row.get("direct_mean_utility") for row in depth_rows]
        plt.scatter(xs, ys, alpha=0.75)
        if len(depth_rows) <= 20:
            for row in depth_rows:
                plt.annotate(
                    str(row.get("node_id")),
                    (row.get("depth"), row.get("direct_mean_utility")),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                )
        plt.xlabel("depth")
        plt.ylabel("direct mean utility")
        plt.title("Depth vs direct utility")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = figures_dir / "tree_depth_vs_utility.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        figure_paths["tree_depth_vs_utility"] = out_path

    return figure_paths


def build_report_text(
    output_dir: Path,
    artifacts: Dict[str, Any],
    snapshot_rows: Sequence[Dict[str, Any]],
    node_rows: Sequence[Dict[str, Any]],
    expansion_events: Sequence[Dict[str, Any]],
    evaluation_events: Sequence[Dict[str, Any]],
    final_selection_rows: Sequence[Dict[str, Any]],
    correlations: Sequence[Dict[str, Any]],
    disagreement_metrics: Dict[str, Any],
    runtime_estimates: Dict[str, Any],
    figure_paths: Dict[str, Optional[Path]],
) -> str:
    run_summary = artifacts.get("run_summary") or {}
    run_config = artifacts.get("run_config") or {}
    config = run_config.get("config", {}) if run_config else {}
    execution = config.get("execution", {})
    optimization = config.get("optimization", {})
    research = config.get("research", {})
    wall_clock = run_summary.get("wall_clock_seconds")
    if wall_clock is None:
        wall_clock = None
        if artifacts["snapshots"]:
            timestamps = [s.get("timestamp") for s in artifacts["snapshots"] if is_finite_number(s.get("timestamp"))]
            if timestamps:
                wall_clock = max(timestamps) - min(timestamps)

    num_success = sum(1 for row in expansion_events if row.get("status") == "success")
    num_failed = sum(1 for row in expansion_events if row.get("status") == "failed")
    num_nodes = len(node_rows)

    report: List[str] = []
    report.append("# HGM Research Analysis")
    report.append("")
    report.append("## 1. Run summary")
    report.append("")
    summary_rows = [
        ("output dir", str(output_dir)),
        ("wall-clock time (s)", format_float(wall_clock, 2) if wall_clock is not None else "insufficient data"),
        (
            "max task evals",
            format_int(run_summary.get("max_task_evals") or execution.get("max_task_evals"))
            if (run_summary.get("max_task_evals") or execution.get("max_task_evals")) is not None
            else "insufficient data",
        ),
        (
            "max workers",
            format_int(run_summary.get("max_workers") or execution.get("max_workers"))
            if (run_summary.get("max_workers") or execution.get("max_workers")) is not None
            else "insufficient data",
        ),
        ("number of nodes", str(num_nodes)),
        ("number of expansion decisions", str(disagreement_metrics.get("num_expansion_decisions", 0))),
        ("number of evaluation decisions", str(len(evaluation_events))),
        ("number of successful children", str(num_success)),
        ("number of failed children", str(num_failed)),
    ]
    for key, value in summary_rows:
        report.append(f"- {key}: {value}")
    report.append("")
    report.append("## 2. How to read this report")
    report.append("")
    report.append(
        "Expansion policy: HGM chooses a parent node using Thompson sampling over descendant-evaluation utility, with a bias toward nodes whose clade looks promising."
    )
    report.append(
        "Evaluation policy: HGM chooses a node to measure on a downstream task using Thompson sampling over direct utility measures, then picks a task from the node's remaining task pool."
    )
    report.append(
        "Estimated CMP: the research logger records a clade-level success rate estimate for each node using the observed utility measures available at decision time."
    )
    report.append(
        "Greedy current-performance baseline: the report compares HGM's sampled choice against the node with the highest finite direct mean utility among eligible candidates."
    )
    report.append("")
    report.append("## 3. Final selected agents under different policies")
    report.append("")
    report.append(
        markdown_table(
            list(final_selection_rows),
            [
                ("policy_name", "policy"),
                ("selected_node_id", "selected_node_id"),
                ("commit_id", "commit_id"),
                ("direct_mean_utility", "direct_mean_utility"),
                ("estimated_cmp", "estimated_cmp"),
                ("direct_num_evals", "direct_num_evals"),
                ("clade_num_evals", "clade_num_evals"),
                ("depth", "depth"),
                ("subtree_size", "subtree_size"),
            ],
        )
    )
    report.append("")
    report.append("## 4. HGM vs greedy expansion behavior")
    report.append("")
    disagreement_rate = disagreement_metrics.get("disagreement_current_rate")
    report.append(
        f"- Disagreement rate vs greedy current-performance: {format_float(disagreement_rate, 3) if disagreement_rate is not None else 'insufficient data'}"
    )
    report.append(
        f"- Disagreement rate vs greedy estimated-CMP: {format_float(disagreement_metrics.get('disagreement_cmp_rate'), 3) if disagreement_metrics.get('disagreement_cmp_rate') is not None else 'insufficient data'}"
    )
    report.append(
        markdown_table(
            list(expansion_events[-10:]),
            [
                ("event_index", "event_index"),
                ("n_task_evals", "n_task_evals"),
                ("selected_node_id", "selected_node_id"),
                ("selected_commit_id", "selected_commit_id"),
                ("greedy_current_performance_node_id", "greedy_current_node"),
                ("greedy_estimated_cmp_node_id", "greedy_cmp_node"),
                ("selected_same_as_greedy_current_performance", "same_current?"),
                ("selected_same_as_greedy_estimated_cmp", "same_cmp?"),
                ("selected_node_direct_mean_utility", "selected_direct"),
                ("selected_node_estimated_cmp", "selected_cmp"),
                ("child_commit_id", "child_commit"),
                ("status", "status"),
                ("elapsed_seconds", "elapsed_s"),
            ],
        )
    )
    report.append("")
    report.append(
        "Interpretation: if HGM frequently disagrees with the greedy baseline, then the posterior sampler is exploring beyond immediate direct utility; if it often agrees, the search is behaving more exploitatively."
    )
    report.append("")
    report.append("## 5. Metaproductivity / CMP analysis")
    report.append("")
    report.append(
        markdown_table(
            correlations,
            [
                ("metric_pair", "metric_pair"),
                ("n", "n"),
                ("pearson_r", "pearson_r"),
                ("weighted_r", "weighted_r"),
            ],
        )
    )
    report.append("")
    report.append("### Node table sorted by estimated CMP")
    report.append("")
    cmp_sorted = sorted(
        [row for row in node_rows if is_finite_number(row.get("estimated_cmp"))],
        key=lambda row: (-float(row.get("estimated_cmp")), int(row.get("node_id") or 0)),
    )
    report.append(
        markdown_table(
            cmp_sorted[:15],
            [
                ("node_id", "node_id"),
                ("commit_id", "commit_id"),
                ("direct_mean_utility", "direct_mean_utility"),
                ("estimated_cmp", "estimated_cmp"),
                ("empirical_future_cmp_excluding_self", "empirical_future_cmp"),
                ("direct_num_evals", "direct_num_evals"),
                ("clade_num_evals", "clade_num_evals"),
                ("depth", "depth"),
                ("subtree_size", "subtree_size"),
            ],
        )
    )
    report.append("")
    report.append("### Node table sorted by direct utility")
    report.append("")
    direct_sorted = sorted(
        [row for row in node_rows if is_finite_number(row.get("direct_mean_utility"))],
        key=lambda row: (-float(row.get("direct_mean_utility")), int(row.get("node_id") or 0)),
    )
    report.append(
        markdown_table(
            direct_sorted[:15],
            [
                ("node_id", "node_id"),
                ("commit_id", "commit_id"),
                ("direct_mean_utility", "direct_mean_utility"),
                ("estimated_cmp", "estimated_cmp"),
                ("empirical_future_cmp_excluding_self", "empirical_future_cmp"),
                ("direct_num_evals", "direct_num_evals"),
                ("clade_num_evals", "clade_num_evals"),
                ("depth", "depth"),
                ("subtree_size", "subtree_size"),
            ],
        )
    )
    report.append("")
    report.append(
        "Empirical future CMP is post-hoc and path-dependent: it is derived from descendant evaluations that happened after a node existed in the evolving tree."
    )
    report.append("")
    report.append("## 6. Diagnosis and self-improvement changes")
    report.append("")
    report.append(
        markdown_table(
            list(expansion_events),
            [
                ("status", "status"),
                ("parent_node_id", "parent_node"),
                ("child_commit_id", "child_commit"),
                ("diagnosis_entry", "diagnosis_entry"),
                ("problem_statement_preview", "problem_statement_preview"),
                ("changed_files", "changed_files"),
                ("patch_line_count", "patch_line_count"),
                ("child_run_dir", "child_run_dir"),
                ("child_metadata_path", "child_metadata_path"),
                ("self_evo_path", "self_evo_path"),
                ("model_patch_path", "model_patch_path"),
            ],
            max_rows=15,
        )
    )
    report.append("")
    report.append(
        "Paths for each successful child are recorded in the per-child metadata directory and can be inspected through the linked artifacts in the event log."
    )
    report.append("")
    report.append("## 7. Evaluation progress")
    report.append("")
    report.append(
        markdown_table(
            list(evaluation_events[-15:]),
            [
                ("event_index", "event_index"),
                ("n_task_evals_before", "n_task_evals_before"),
                ("n_task_evals_after", "n_task_evals_after"),
                ("selected_node_id", "selected_node_id"),
                ("selected_task_id", "selected_task_id"),
                ("result_values", "result_values"),
                ("selected_node_direct_mean_before", "direct_before"),
                ("selected_node_direct_mean_after", "direct_after"),
                ("same_as_greedy_current_performance", "same_greedy?"),
                ("elapsed_seconds", "elapsed_s"),
            ],
        )
    )
    if snapshot_rows:
        best_over_time = []
        running = None
        for snapshot in snapshot_rows:
            nodes = snapshot.get("nodes", []) or []
            direct_values = [to_float_or_none(node.get("direct_mean_utility")) for node in nodes]
            direct_values = [v for v in direct_values if v is not None]
            if direct_values:
                current_best = max(direct_values)
                running = current_best if running is None else max(running, current_best)
            best_over_time.append(running)
        final_best = next((v for v in reversed(best_over_time) if v is not None), None)
        report.append("")
        report.append(
            f"Best direct utility over time ended at {format_float(final_best, 4) if final_best is not None else 'insufficient data'}."
        )
    report.append("")
    report.append("## 8. Runtime estimate")
    report.append("")
    report.append(
        f"- observed average expansion duration: {format_float(runtime_estimates.get('avg_expansion_seconds'), 2) if runtime_estimates.get('avg_expansion_seconds') is not None else 'insufficient data'}"
    )
    report.append(
        f"- observed average evaluation duration: {format_float(runtime_estimates.get('avg_evaluation_seconds'), 2) if runtime_estimates.get('avg_evaluation_seconds') is not None else 'insufficient data'}"
    )
    report.append(
        f"- expansion duration p50/p90: {format_float(runtime_estimates.get('p50_expansion_seconds'), 2) if runtime_estimates.get('p50_expansion_seconds') is not None else 'insufficient data'} / {format_float(runtime_estimates.get('p90_expansion_seconds'), 2) if runtime_estimates.get('p90_expansion_seconds') is not None else 'insufficient data'}"
    )
    report.append(
        f"- evaluation duration p50/p90: {format_float(runtime_estimates.get('p50_evaluation_seconds'), 2) if runtime_estimates.get('p50_evaluation_seconds') is not None else 'insufficient data'} / {format_float(runtime_estimates.get('p90_evaluation_seconds'), 2) if runtime_estimates.get('p90_evaluation_seconds') is not None else 'insufficient data'}"
    )
    report.append("")
    report.append(
        markdown_table(
            runtime_estimates.get("budget_rows", []),
            [
                ("budget", "budget"),
                ("estimated_expansion_count", "exp_count"),
                ("estimated_evaluation_count", "eval_count"),
                ("lower_bound_parallel_hours", "lower_bound_hours"),
                ("timeout_upper_bound_hours", "timeout_upper_bound_hours"),
            ],
        )
    )
    report.append("")
    report.append(
        "These runtime estimates are rough. They assume average observed expansion/evaluation duration, scale expansion count as max_task_evals ** alpha, and divide by max_workers; Docker overhead, LLM latency, task difficulty, and retries can move the true wall-clock time substantially."
    )
    report.append("")
    report.append("## Artifact links")
    report.append("")
    report.append("- [node_table.csv](node_table.csv)")
    report.append("- [expansion_events.csv](expansion_events.csv)")
    report.append("- [evaluation_events.csv](evaluation_events.csv)")
    report.append("- [final_selection.csv](final_selection.csv)")
    if figure_paths.get("progress_best_utility"):
        report.append("- [progress_best_utility.png](figures/progress_best_utility.png)")
    if figure_paths.get("node_cmp_vs_direct"):
        report.append("- [node_cmp_vs_direct.png](figures/node_cmp_vs_direct.png)")
    if figure_paths.get("cmp_vs_empirical_future_cmp"):
        report.append("- [cmp_vs_empirical_future_cmp.png](figures/cmp_vs_empirical_future_cmp.png)")
    if figure_paths.get("direct_vs_empirical_future_cmp"):
        report.append("- [direct_vs_empirical_future_cmp.png](figures/direct_vs_empirical_future_cmp.png)")
    if figure_paths.get("selection_disagreement"):
        report.append("- [selection_disagreement.png](figures/selection_disagreement.png)")
    if figure_paths.get("tree_depth_vs_utility"):
        report.append("- [tree_depth_vs_utility.png](figures/tree_depth_vs_utility.png)")
    return "\n".join(report).strip() + "\n"


def analyze_hgm_research_run(
    output_dir: str,
    write_markdown: bool = True,
    write_figures: bool = True,
    min_descendant_evals_for_empirical_cmp: int = 1,
) -> Dict[str, Any]:
    output_path = Path(output_dir).resolve()
    artifacts = load_run_artifacts(output_path)
    node_rows = build_node_table(
        output_path, artifacts, min_descendant_evals_for_empirical_cmp=min_descendant_evals_for_empirical_cmp
    )
    expansion_events = build_expansion_events(artifacts, node_rows)
    evaluation_events = build_evaluation_events(artifacts, node_rows)
    final_selection_rows = build_final_selection_rows(
        node_rows,
        expansion_events,
        min_descendant_evals_for_empirical_cmp=min_descendant_evals_for_empirical_cmp,
    )
    disagreement_metrics = compute_disagreement_metrics(expansion_events)
    correlations = compute_correlations(
        node_rows, min_descendant_evals_for_empirical_cmp=min_descendant_evals_for_empirical_cmp
    )
    runtime_estimates = compute_runtime_estimates(
        artifacts.get("run_config"),
        artifacts.get("run_summary"),
        expansion_events,
        evaluation_events,
    )

    research_dir = output_path / "research"
    research_dir.mkdir(parents=True, exist_ok=True)
    write_csv(research_dir / "node_table.csv", node_rows)
    write_csv(research_dir / "expansion_events.csv", expansion_events)
    write_csv(research_dir / "evaluation_events.csv", evaluation_events)
    write_csv(research_dir / "final_selection.csv", final_selection_rows)

    figure_paths = {}
    if write_figures:
        figure_paths = generate_figures(
            output_path,
            node_rows,
            artifacts["snapshots"],
            expansion_events,
            evaluation_events,
            runtime_estimates,
        )
    if write_markdown:
        report_text = build_report_text(
            output_path,
            artifacts,
            artifacts["snapshots"],
            node_rows,
            expansion_events,
            evaluation_events,
            final_selection_rows,
            correlations,
            disagreement_metrics,
            runtime_estimates,
            figure_paths,
        )
        (research_dir / "progress.md").write_text(report_text, encoding="utf-8")

    return {
        "output_dir": output_path,
        "research_dir": research_dir,
        "node_rows": node_rows,
        "expansion_events": expansion_events,
        "evaluation_events": evaluation_events,
        "final_selection_rows": final_selection_rows,
        "correlations": correlations,
        "disagreement_metrics": disagreement_metrics,
        "runtime_estimates": runtime_estimates,
        "figure_paths": figure_paths,
    }
