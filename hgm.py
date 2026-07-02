# This file is adapted from https://github.com/jennyzzt/dgm.

import argparse
import datetime
import json
import math
import os
import random
import string
import sys
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                TimeoutError, as_completed)
from statistics import stdev
from types import SimpleNamespace

import numpy as np
from datasets import load_dataset
from utils.docker_utils import copy_src_files

import hgm_utils
from config import load_config
from tree import Node
from utils.common_utils import load_json_file
from utils.docker_utils import copy_src_files, setup_logger
from utils.evo_utils import load_hgm_metadata
from utils.hgm_research_logger import (
    ResearchLogger,
    build_candidate_policy_summary,
    build_node_stats,
    _sanitize_jsonable,
)


def apply_vllm_env_overrides(llm_cfg):
    env_updates = {
        "VLLM_BASE_URL": llm_cfg.vllm_base_url,
        "VLLM_API_KEY": llm_cfg.vllm_api_key,
        "VLLM_MODEL": llm_cfg.vllm_model,
    }
    for key, value in env_updates.items():
        if value:
            os.environ[key] = value
    return {key: os.getenv(key) for key in env_updates}


research_logger = None


def _append_jsonl_record(path, record):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                _sanitize_jsonable(record),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        handle.flush()


def update_metadata(output_dir, n_task_evals, snapshot_extra=None, research_logger_obj=None):
    _append_jsonl_record(
        os.path.join(output_dir, "hgm_metadata.jsonl"),
        {
            "timestamp": time.time(),
            "run_id": os.path.basename(os.path.normpath(output_dir)),
            "output_dir": output_dir,
            "n_task_evals": n_task_evals,
            "nodes": [
                node.save_as_dict()
                for node in hgm_utils.nodes.values()
                if node.commit_id != "initial"
            ],
        },
    )
    json.dump(
        hgm_utils.init_evaluated_tasks,
        open(os.path.join(output_dir, "init_evaluated_tasks.json"), "w"),
    )
    logger_obj = research_logger_obj if research_logger_obj is not None else research_logger
    if logger_obj is not None:
        try:
            logger_obj.write_snapshot(
                hgm_utils.nodes.values(),
                n_task_evals,
                extra=snapshot_extra,
            )
        except Exception:
            pass


def _preview_text(value, limit=400):
    if value is None:
        return None
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def _summarize_diff_file(diff_path):
    summary = {"patch_line_count": None, "changed_files": []}
    if not diff_path or not os.path.exists(diff_path):
        return summary
    try:
        with open(diff_path, "r", encoding="utf-8", errors="replace") as handle:
            diff_text = handle.read()
        summary["patch_line_count"] = len(diff_text.splitlines())
        changed_files = []
        for line in diff_text.splitlines():
            if line.startswith("diff --git a/"):
                parts = line.split()
                if len(parts) >= 4 and parts[3].startswith("b/"):
                    changed_files.append(parts[3][2:])
        summary["changed_files"] = list(dict.fromkeys(changed_files))
    except Exception:
        pass
    return summary


def _summarize_child_artifacts(output_dir, child_commit):
    if not child_commit or child_commit == "failed":
        return {
            "child_run_dir": None,
            "child_metadata_path": None,
            "self_evo_path": None,
            "model_patch_path": None,
            "diagnosis_entry": None,
            "problem_statement_preview": None,
            "patch_line_count": None,
            "changed_files": [],
        }
    child_run_dir = os.path.join(output_dir, child_commit)
    metadata_path = os.path.join(child_run_dir, "metadata.json")
    self_evo_path = os.path.join(child_run_dir, "self_evo.md")
    model_patch_path = os.path.join(child_run_dir, "model_patch.diff")
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            metadata = load_json_file(metadata_path)
        except Exception:
            metadata = {}
    diff_summary = _summarize_diff_file(model_patch_path)
    return {
        "child_run_dir": child_run_dir if os.path.exists(child_run_dir) else None,
        "child_metadata_path": metadata_path if os.path.exists(metadata_path) else None,
        "self_evo_path": self_evo_path if os.path.exists(self_evo_path) else None,
        "model_patch_path": model_patch_path if os.path.exists(model_patch_path) else None,
        "diagnosis_entry": metadata.get("entry"),
        "problem_statement_preview": _preview_text(metadata.get("problem_statement")),
        "patch_line_count": diff_summary["patch_line_count"],
        "changed_files": diff_summary["changed_files"],
    }


def TS_sample(
    evals,
    return_details=False,
    opt_cfg=None,
    exec_cfg=None,
    current_n_task_evals=None,
):
    if len(evals) == 0:
        raise ValueError("TS_sample received an empty evaluation list")

    opt_cfg = opt_cfg or SimpleNamespace(cool_down=False, beta=1.0)
    if exec_cfg is None:
        exec_cfg = SimpleNamespace(max_task_evals=max(len(evals), 1))
    if current_n_task_evals is None:
        current_n_task_evals = hgm_utils.n_task_evals

    alphas = [1 + np.sum(de) for de in evals]
    betas = [1 + len(de) - np.sum(de) for de in evals]
    if opt_cfg.cool_down:
        cooldown_scale = (
            10000
            if exec_cfg.max_task_evals == current_n_task_evals
            else exec_cfg.max_task_evals**opt_cfg.beta
            / (exec_cfg.max_task_evals - current_n_task_evals) ** opt_cfg.beta
        )
        alphas = np.array(alphas) * cooldown_scale
        betas = np.array(betas) * cooldown_scale
    thetas = np.random.beta(alphas, betas)
    selected_index = int(np.argmax(thetas))
    if return_details:
        return selected_index, {
            "alphas": [float(value) for value in np.asarray(alphas).tolist()],
            "betas": [float(value) for value in np.asarray(betas).tolist()],
            "thetas": [float(value) for value in np.asarray(thetas).tolist()],
            "selected_index": selected_index,
        }
    return selected_index


def initialize_run(
    output_dir,
    self_improve_llm,
    tasks,
    initial_agent_name,
    initial_eval_tasks=None,
    prevrun_dir=None,
    polyglot=False,
    timeout=3600,
    max_workers=20
):
    hgm_utils.init(polyglot, output_dir, tasks, 0, self_improve_llm, timeout)

    # Copy cached initial version into experiment dir
    initial_folder = "initial_swe/" if not polyglot else "initial_polyglot/"
    if not prevrun_dir:
        if not os.path.exists(f"{initial_folder}/{initial_agent_name}"):
            copy_src_files(f"{initial_folder}/{initial_agent_name}/src", build_image=True)
            hgm_utils.output_dir = initial_folder
            initial_eval_task_list = tasks
            if initial_eval_tasks is not None:
                initial_eval_task_list = initial_eval_task_list[:initial_eval_tasks]
            hgm_utils.eval_agent(
                initial_agent_name,
                tasks=initial_eval_task_list,
                max_workers=max_workers,
                init_agent_path=f"{initial_folder}/{initial_agent_name}/src",
            )
            hgm_utils.init_evaluated_tasks = list(initial_eval_task_list)
            hgm_utils.output_dir = output_dir

    os.system(f"cp -r {initial_folder}/{initial_agent_name} {output_dir}/initial")

    Node(commit_id="initial")
    if prevrun_dir:
        # Load previous run's archive
        hgm_utils.init_evaluated_tasks = load_json_file(
            os.path.join(prevrun_dir, "init_evaluated_tasks.json")
        )
        metadata_path = os.path.join(prevrun_dir, "hgm_metadata.jsonl")
        metadata = load_hgm_metadata(metadata_path, last_only=True)
        for node in metadata["nodes"]:
            commit_id = node["commit_id"]
            parent_id = node["parent_id"]
            Node(commit_id, parent_id=parent_id, id=node["id"])
        for node in hgm_utils.nodes.values():
            if node.parent_id is not None:
                parent = hgm_utils.nodes[node.parent_id]
                parent.add_child(node)

    n_task_evals = 0
    submitted_ids = defaultdict(set)  # node_id -> set of submitted task ids
    for node in hgm_utils.nodes.values():
        metadata = load_json_file(
            os.path.join(output_dir, node.commit_id, "metadata.json")
        )
        submitted_ids[node.id] = set(
            metadata["overall_performance"]["total_submitted_ids"]
        )
        node.utility_measures = [
            1
            for _ in range(metadata["overall_performance"]["total_resolved_instances"])
        ] + [
            0
            for _ in range(
                metadata["overall_performance"]["total_submitted_instances"]
                - metadata["overall_performance"]["total_resolved_instances"]
            )
        ]
        if node.commit_id != "initial":
            n_task_evals += metadata["overall_performance"]["total_submitted_instances"]
    hgm_utils.n_task_evals = n_task_evals
    return os.path.join(initial_folder, initial_agent_name, "src"), submitted_ids


def main():
    parser = argparse.ArgumentParser(description="Optimistic Tree Search")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--max_task_evals",
        type=int,
        default=None,
        help="Maximum number of evolution iterations.",
    )
    parser.add_argument(
        "--initial_eval_tasks",
        type=int,
        default=None,
        help="Cap the initial baseline evaluation to the first N tasks.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Number of parallel workers for self-improvement attempts.",
    )
    parser.add_argument(
        "--continue_from",
        type=str,
        default=None,
        help="Directory to continue the run from.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for this run (overrides config).",
    )
    parser.add_argument(
        "--polyglot",
        dest="polyglot",
        action="store_true",
        help="Run Polyglot benchmark instead of SWE-bench.",
    )
    parser.add_argument(
        "--no_polyglot",
        dest="polyglot",
        action="store_false",
        help="Disable Polyglot benchmark even if enabled in config.",
    )
    parser.add_argument(
        "--self_improve_llm",
        type=str,
        default=None,
        help="LLM model to use for self-improvement",
    )
    parser.add_argument(
        "--downstream_llm",
        type=str,
        default=None,
        help="LLM model to use for downstream tasks",
    )
    parser.add_argument(
        "--diagnose_llm",
        type=str,
        default=None,
        help="LLM model to use for diagnosis",
    )
    parser.add_argument(
        "--vllm_base_url",
        type=str,
        default=None,
        help="Base URL for a local OpenAI-compatible vLLM endpoint.",
    )
    parser.add_argument(
        "--vllm_api_key",
        type=str,
        default=None,
        help="API key for a local OpenAI-compatible vLLM endpoint.",
    )
    parser.add_argument(
        "--vllm_model",
        type=str,
        default=None,
        help="Default model ID served by the local vLLM endpoint.",
    )
    parser.add_argument(
        "--alpha", type=float, default=None, help="Alpha parameter for node expansion."
    )
    parser.add_argument(
        "--cool_down",
        dest="cool_down",
        action="store_true",
        help="Use a decreasing temperature over iterations.",
    )
    parser.add_argument(
        "--no_cool_down",
        dest="cool_down",
        action="store_false",
        help="Disable decreasing temperature over iterations even if enabled in config.",
    )
    parser.add_argument(
        "--beta", type=float, default=None, help="Cooling down factor beta."
    )
    parser.add_argument(
        "--full_eval",
        dest="full_eval",
        action="store_true",
        help="Run full evaluation on SWE even if disabled in config.",
    )

    parser.add_argument(
        "--self_improve_timeout",
        type=int,
        default=None,
        help="Timeout for self-improvement attempts.",
    )
    parser.add_argument(
        "--evaluation_timeout",
        type=int,
        default=None,
        help="Timeout for evaluation attempts.",
    )
    parser.add_argument(
        "--n_pseudo_descendant_evals",
        type=int,
        default=None,
        help="Number of pseudo descendant evaluations.",
    )
    parser.add_argument(
        "--eval_random_level",
        type=float,
        default=None,
        help="Randomness level for evaluation task selection.",
    )
    parser.add_argument(
        "--initial_agent_name",
        type=str,
        default="default_agent",
        help="Name of the initial agent.",
    )

    parser.set_defaults(polyglot=None, cool_down=None, full_eval=None)

    args = parser.parse_args()

    overrides = {}
    if args.max_task_evals is not None:
        overrides["execution.max_task_evals"] = args.max_task_evals
    if args.initial_eval_tasks is not None:
        overrides["execution.initial_eval_tasks"] = args.initial_eval_tasks
    if args.max_workers is not None:
        overrides["execution.max_workers"] = args.max_workers
    if args.continue_from is not None:
        overrides["paths.continue_from"] = args.continue_from
    if args.output_dir is not None:
        overrides["paths.output_dir"] = args.output_dir
    if args.self_improve_llm is not None:
        overrides["llm.self_improve_llm"] = args.self_improve_llm
    if args.downstream_llm is not None:
        overrides["llm.downstream_llm"] = args.downstream_llm
    if args.diagnose_llm is not None:
        overrides["llm.diagnose_llm"] = args.diagnose_llm
    if args.vllm_base_url is not None:
        overrides["llm.vllm_base_url"] = args.vllm_base_url
    if args.vllm_api_key is not None:
        overrides["llm.vllm_api_key"] = args.vllm_api_key
    if args.vllm_model is not None:
        overrides["llm.vllm_model"] = args.vllm_model
    if args.alpha is not None:
        overrides["optimization.alpha"] = args.alpha
    if args.cool_down is not None:
        overrides["optimization.cool_down"] = args.cool_down
    if args.beta is not None:
        overrides["optimization.beta"] = args.beta
    if args.full_eval is not None:
        overrides["evaluation.full_eval"] = args.full_eval
    if args.self_improve_timeout is not None:
        overrides["execution.self_improve_timeout"] = args.self_improve_timeout
    if args.evaluation_timeout is not None:
        overrides["execution.evaluation_timeout"] = args.evaluation_timeout
    if args.n_pseudo_descendant_evals is not None:
        overrides["optimization.n_pseudo_descendant_evals"] = args.n_pseudo_descendant_evals
    if args.eval_random_level is not None:
        overrides["optimization.eval_random_level"] = args.eval_random_level
    if args.polyglot is not None:
        overrides["evaluation.polyglot"] = args.polyglot
    if args.initial_agent_name is not None:
        overrides["paths.initial_agent_name"] = args.initial_agent_name

    config = load_config(args.config, **overrides)

    if not config.paths.initial_agent_name:
        parser.error(
            "Initial agent name must be provided either in config.yaml or via --initial_agent_name."
        )

    llm_cfg = config.llm
    opt_cfg = config.optimization
    exec_cfg = config.execution
    eval_cfg = config.evaluation
    research_cfg = config.research
    path_cfg = config.paths
    vllm_env = apply_vllm_env_overrides(llm_cfg)

    # Variables for this HGM run
    if path_cfg.output_dir:
        output_dir = os.path.abspath(path_cfg.output_dir)
        run_id = os.path.basename(os.path.normpath(output_dir))
    elif not path_cfg.continue_from:
        run_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S_%f")
        output_dir = os.path.abspath(os.path.join("./output_hgm", run_id))
    else:
        run_id = os.path.basename(os.path.normpath(path_cfg.continue_from))
        output_dir = os.path.abspath(os.path.join("./output_hgm", run_id))

    # Ensure output directory exists and log path info
    os.makedirs(output_dir, exist_ok=True)
    print(f"Working directory: {os.getcwd()}")
    print(f"Using config file: {args.config}")
    print(f"Output directory: {output_dir}")
    print(f"Output directory exists: {os.path.exists(output_dir)}")
    print(
        "vLLM env: "
        f"base_url={vllm_env.get('VLLM_BASE_URL')} "
        f"model={vllm_env.get('VLLM_MODEL')}"
    )

    run_started_at = time.time()
    config_snapshot = config.to_dict()
    if config_snapshot["llm"].get("vllm_api_key"):
        config_snapshot["llm"]["vllm_api_key"] = "***"
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(
            {
                "config": config_snapshot,
                "env": {
                    "VLLM_BASE_URL": vllm_env.get("VLLM_BASE_URL"),
                    "VLLM_MODEL": vllm_env.get("VLLM_MODEL"),
                    "OPENAI_API_KEY_set": bool(os.getenv("OPENAI_API_KEY")),
                },
                "run_started_at_epoch": run_started_at,
            },
            f,
            indent=2,
        )

    import self_improve_step
    import swe_bench.harness

    if eval_cfg.polyglot:
        import polyglot.harness

        polyglot.harness.llm = llm_cfg.downstream_llm
        polyglot.harness.timeout = exec_cfg.evaluation_timeout
    swe_bench.harness.llm = (
        llm_cfg.downstream_llm
    )  # Set the LLM model for downstream tasks
    swe_bench.harness.timeout = exec_cfg.evaluation_timeout
    self_improve_step.diagnose_llm = llm_cfg.diagnose_llm
    self_improve_step.self_improve_llm = llm_cfg.self_improve_llm
    # Initialize logger early
    logger = setup_logger(os.path.join(output_dir, "hgm_outer.log"))
    global research_logger
    research_logger = ResearchLogger(output_dir, enabled=research_cfg.enabled)
    research_logger.log_event(
        "run_started",
        {
            "max_task_evals": exec_cfg.max_task_evals,
            "max_workers": exec_cfg.max_workers,
            "initial_eval_tasks": exec_cfg.initial_eval_tasks,
            "alpha": opt_cfg.alpha,
            "beta": opt_cfg.beta,
            "cool_down": opt_cfg.cool_down,
            "eval_random_level": opt_cfg.eval_random_level,
            "n_pseudo_descendant_evals": opt_cfg.n_pseudo_descendant_evals,
            "research_enabled": research_cfg.enabled,
            "log_policy_details": research_cfg.log_policy_details,
        },
    )
    # SWE issues to consider
    if not eval_cfg.polyglot:
        if eval_cfg.full_eval:
            tasks = [
                task["instance_id"]
                for task in load_dataset("princeton-nlp/SWE-bench_Verified")["test"]
            ]
        else:
            tasks = load_json_file("./swe_bench/subsets/small.json") \
                    + load_json_file("./swe_bench/subsets/medium.json") 
        random.seed(42)
        random.shuffle(tasks)
    else:
        tasks = load_json_file("./polyglot/subsets/medium.json") + load_json_file(
            "./polyglot/subsets/small.json"
        )

    src_path, submitted_ids = initialize_run(
        output_dir,
        llm_cfg.self_improve_llm,
        tasks,
        path_cfg.initial_agent_name,
        initial_eval_tasks=exec_cfg.initial_eval_tasks,
        prevrun_dir=path_cfg.continue_from,
        polyglot=eval_cfg.polyglot,
        timeout=exec_cfg.self_improve_timeout,
        max_workers=exec_cfg.max_workers
    )
    total_num_tasks = len(hgm_utils.total_tasks)

    # Set up logger
    logger.info(
        f"Starting HGM run {run_id} with configuration: {config.to_dict()}"
    )
    logger.info(
        "Local provider configuration: "
        f"VLLM_BASE_URL={vllm_env.get('VLLM_BASE_URL')} "
        f"VLLM_MODEL={vllm_env.get('VLLM_MODEL')}"
    )

    n_pending_expands = 0
    n_pending_measures = 0
    lock = threading.Lock()

    def expand():
        decision_payload = None
        selected_node = None
        selected_index = None
        with lock:
            nodes = [
                node
                for node in hgm_utils.nodes.values()
                if np.isfinite(node.mean_utility) and node.mean_utility > 0
            ]
            if len(nodes) == 0:
                nodes = [hgm_utils.nodes[0]]
            candidate_summary = (
                build_candidate_policy_summary(nodes, node_map=hgm_utils.nodes)
                if research_cfg.enabled and research_cfg.log_policy_details
                else None
            )
            decendant_evals = [
                node.get_decendant_evals(num_pseudo=opt_cfg.n_pseudo_descendant_evals)
                for node in nodes
            ]
            selected_index, ts_details = TS_sample(
                decendant_evals,
                return_details=True,
                opt_cfg=opt_cfg,
                exec_cfg=exec_cfg,
                current_n_task_evals=hgm_utils.n_task_evals,
            )
            selected_node = nodes[selected_index]
            if candidate_summary is not None and research_logger is not None:
                decision_payload = {
                    "current_n_task_evals": hgm_utils.n_task_evals,
                    **candidate_summary,
                    "thompson": ts_details,
                    "selected_node_id": getattr(selected_node, "id", None),
                    "selected_commit_id": getattr(selected_node, "commit_id", None),
                    "selected_index": selected_index,
                    "selected_same_as_greedy_current_performance": (
                        getattr(selected_node, "id", None)
                        == candidate_summary["greedy_current_performance_node_id"]
                    ),
                    "selected_same_as_greedy_estimated_cmp": (
                        getattr(selected_node, "id", None)
                        == candidate_summary["greedy_estimated_cmp_node_id"]
                    ),
                    "config": {
                        "alpha": opt_cfg.alpha,
                        "beta": opt_cfg.beta,
                        "cool_down": opt_cfg.cool_down,
                        "n_pseudo_descendant_evals": opt_cfg.n_pseudo_descendant_evals,
                    },
                    "n_pending_expands": n_pending_expands,
                    "n_pending_measures": n_pending_measures,
                }
                research_logger.log_event("expansion_decision", decision_payload)
        child_start = time.time()
        child_commit = hgm_utils.sample_child(
            selected_node.commit_id,
            image_name=path_cfg.initial_agent_name + ":latest",
        )
        child_elapsed = time.time() - child_start
        child_artifacts = _summarize_child_artifacts(output_dir, child_commit)
        child_node = None
        with lock:
            if child_commit != "failed":
                child_node = Node(child_commit, parent_id=selected_node.id)
                selected_node.children.append(child_node)
                update_metadata(
                    output_dir,
                    hgm_utils.n_task_evals,
                    snapshot_extra={
                        "trigger_event": "expansion_result",
                        "parent_node_id": getattr(selected_node, "id", None),
                        "parent_commit_id": getattr(selected_node, "commit_id", None),
                        "child_commit_id": child_commit,
                    },
                )
        if research_logger is not None and research_cfg.log_policy_details:
            research_logger.log_event(
                "expansion_result",
                {
                    "parent_node_id": getattr(selected_node, "id", None),
                    "parent_commit_id": getattr(selected_node, "commit_id", None),
                    "child_commit_id": child_commit,
                    "status": "success" if child_commit != "failed" else "failed",
                    "elapsed_seconds": child_elapsed,
                    "new_child_id": getattr(child_node, "id", None),
                    **child_artifacts,
                },
            )

    def sample():
        time.sleep(random.random())
        with lock:
            nonlocal n_pending_expands, n_pending_measures
            if hgm_utils.n_task_evals >= exec_cfg.max_task_evals:
                return

            if (
                hgm_utils.n_task_evals**opt_cfg.alpha
                >= len(hgm_utils.nodes) - 1 + n_pending_expands
            ):
                n_pending_expands += 1
                is_expand = True
            else:
                is_expand = False
        if is_expand:
            expand()
            with lock:
                n_pending_expands -= 1
                return

        decision_payload = None
        with lock:
            nodes = hgm_utils.nodes[0].get_sub_tree(fn=lambda node: node)
            nodes = [
                node for node in nodes if len(submitted_ids[node.id]) < total_num_tasks
            ]
            evals = [node.utility_measures for node in nodes]
            if len(evals) == 0:
                return
            candidate_summary = (
                build_candidate_policy_summary(nodes, node_map=hgm_utils.nodes)
                if research_cfg.enabled and research_cfg.log_policy_details
                else None
            )
            selected_index, ts_details = TS_sample(
                evals,
                return_details=True,
                opt_cfg=opt_cfg,
                exec_cfg=exec_cfg,
                current_n_task_evals=hgm_utils.n_task_evals,
            )
            selected_node = nodes[selected_index]
            available_tasks = list(
                [
                    task
                    for task in hgm_utils.total_tasks
                    if task not in submitted_ids[selected_node.id]
                ]
            )
            if len(available_tasks) == 0:
                return
            task_choice_random = random.random() < opt_cfg.eval_random_level
            if task_choice_random:
                selected_node_tasks = random.choice(available_tasks)
                task_choice_mode = "random"
            else:
                selected_node_tasks = available_tasks[0]
                task_choice_mode = "fixed"
            submitted_ids[selected_node.id].add(selected_node_tasks)
            n_pending_measures += 1
            if candidate_summary is not None and research_logger is not None:
                decision_payload = {
                    "current_n_task_evals": hgm_utils.n_task_evals,
                    **candidate_summary,
                    "thompson": ts_details,
                    "selected_node_id": getattr(selected_node, "id", None),
                    "selected_commit_id": getattr(selected_node, "commit_id", None),
                    "selected_task_id": selected_node_tasks,
                    "selected_index": selected_index,
                    "greedy_current_performance_node_id": candidate_summary[
                        "greedy_current_performance_node_id"
                    ],
                    "selected_same_as_greedy_current_performance": (
                        getattr(selected_node, "id", None)
                        == candidate_summary["greedy_current_performance_node_id"]
                    ),
                    "available_task_count": len(available_tasks),
                    "eval_random_level": opt_cfg.eval_random_level,
                    "task_choice_random": task_choice_random,
                    "task_choice_mode": task_choice_mode,
                }
                research_logger.log_event("evaluation_decision", decision_payload)

        eval_start = time.time()
        evals = hgm_utils.eval_agent(
            selected_node.commit_id,
            tasks=[selected_node_tasks],
            init_agent_path=src_path,
        )
        eval_elapsed = time.time() - eval_start
        selected_node_stats = None
        with lock:
            selected_node.utility_measures += evals
            n_pending_measures -= 1
            selected_node_stats = build_node_stats(selected_node, node_map=hgm_utils.nodes)
            update_metadata(
                output_dir,
                hgm_utils.n_task_evals,
                snapshot_extra={
                    "trigger_event": "evaluation_result",
                    "selected_node_id": getattr(selected_node, "id", None),
                    "selected_commit_id": getattr(selected_node, "commit_id", None),
                    "selected_task_id": selected_node_tasks,
                },
            )
        if research_logger is not None and research_cfg.log_policy_details:
            research_logger.log_event(
                "evaluation_result",
                {
                    "selected_node_id": getattr(selected_node, "id", None),
                    "selected_commit_id": getattr(selected_node, "commit_id", None),
                    "selected_task_id": selected_node_tasks,
                    "result_values": evals,
                    "elapsed_seconds": eval_elapsed,
                    "updated_direct_mean_utility": (
                        None
                        if selected_node_stats is None
                        else selected_node_stats.get("direct_mean_utility")
                    ),
                    "updated_direct_eval_count": (
                        None
                        if selected_node_stats is None
                        else selected_node_stats.get("direct_num_evals")
                    ),
                    "updated_estimated_cmp": (
                        None
                        if selected_node_stats is None
                        else selected_node_stats.get("estimated_cmp")
                    ),
                    "global_n_task_evals": hgm_utils.n_task_evals,
                },
            )

    had_error = False
    try:
        with ThreadPoolExecutor(max_workers=exec_cfg.max_workers) as executor:
            futures = [
                executor.submit(expand)
                for _ in range(
                    len(hgm_utils.nodes) - 1,
                    min(5, int(exec_cfg.max_workers**opt_cfg.alpha)),
                )
            ]
            for future in as_completed(futures):
                future.result()

        with ThreadPoolExecutor(max_workers=exec_cfg.max_workers) as executor:
            futures = [
                executor.submit(sample)
                for _ in range(int(exec_cfg.max_task_evals * 100))
            ]
            for future in as_completed(futures):
                future.result()

    except Exception as e:
        had_error = True
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        print(repr(e))
    finally:
        run_finished_at = time.time()
        try:
            non_initial_nodes = [
                node.save_as_dict()
                for node in hgm_utils.nodes.values()
                if getattr(node, "commit_id", None) != "initial"
            ]
            with open(os.path.join(output_dir, "run_summary.json"), "w") as f:
                json.dump(
                    {
                        "run_id": run_id,
                        "wall_clock_seconds": run_finished_at - run_started_at,
                        "run_started_at_epoch": run_started_at,
                        "run_finished_at_epoch": run_finished_at,
                        "max_task_evals": exec_cfg.max_task_evals,
                        "max_workers": exec_cfg.max_workers,
                        "polyglot": eval_cfg.polyglot,
                        "full_eval": eval_cfg.full_eval,
                        "n_task_evals": hgm_utils.n_task_evals,
                        "num_non_initial_nodes": len(non_initial_nodes),
                        "vllm_base_url": vllm_env.get("VLLM_BASE_URL"),
                        "vllm_model": vllm_env.get("VLLM_MODEL"),
                    },
                    f,
                    indent=2,
                )
        except Exception:
            logger.error("Failed to write run_summary.json")
            logger.error(traceback.format_exc())
    if had_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
