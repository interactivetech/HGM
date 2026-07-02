# This file is adapted from https://github.com/jennyzzt/dgm.

import argparse
import datetime
import json
import os
import random
import re
import subprocess
import traceback

import docker

from llm import create_client, extract_json_between_markers, get_response_from_llm

from prompts.self_improvement_prompt import (build_problem_description_prompt,
                                             get_diagnose_prompt_polyglot,
                                             get_diagnose_prompt_swe,
                                             get_problem_description_prompt)

from utils.docker_utils import safe_log

dataset = None
diagnose_llm = ""
self_improve_llm = ""
timeout = 3600
n_evals = 0


def _fallback_problem_statement_from_text(response_text, polyglot=False):
    if response_text is None:
        return None
    if not isinstance(response_text, str):
        response_text = str(response_text)

    response_text = response_text.strip()
    if not response_text:
        return None

    if "# To Implement" in response_text:
        if polyglot:
            return build_problem_description_prompt("", response_text, is_polyglot=True)
        return build_problem_description_prompt("", response_text, is_polyglot=False)

    title_match = re.search(r"(?im)^title:\s*(.+)$", response_text)
    title = title_match.group(1).strip() if title_match else None
    implementation_match = re.search(
        r"(?is)(?:#\s*To Implement\s*|implementation_suggestion\s*[:\-]\s*)(.+?)(?:\n\s*\n|$)",
        response_text,
    )
    implementation_suggestion = (
        implementation_match.group(1).strip() if implementation_match else ""
    )

    if title:
        problem_description = f"Title: {title}\n\n{response_text}"
        return build_problem_description_prompt(
            implementation_suggestion, problem_description, is_polyglot=polyglot
        )

    return None


def diagnose_problem(
    entry, commit, root_dir, out_dir, patch_files=[], max_attempts=2, polyglot=False
):
    client, resolved_model = create_client(diagnose_llm)
    safe_log(
        f"Diagnose problem with model alias={diagnose_llm} resolved_model={resolved_model}"
    )
    if polyglot:
        diagnose_sys_message, diagnose_prompt = get_diagnose_prompt_polyglot(
            entry,
            commit,
            root_dir,
            out_dir,
            dataset,
            patch_files=patch_files,
        )
    else:
        diagnose_sys_message, diagnose_prompt = get_diagnose_prompt_swe(
            entry,
            commit,
            root_dir,
            out_dir,
            dataset,
            patch_files=patch_files,
        )
    try:
        try:
            response, msg_history = get_response_from_llm(
                msg=diagnose_prompt,
                client=client,
                model=resolved_model,
                system_message=diagnose_sys_message,
                print_debug=False,
                msg_history=None,
                logging=safe_log,
            )
        except Exception as e:
            safe_log(f"Error with get_response_from_llm: {e}")
            safe_log(traceback.format_exc())
            raise
        # safe_log(f"Message history: {msg_history}")
        response_json = extract_json_between_markers(response)
        if response_json:
            problem_statement = get_problem_description_prompt(response_json, polyglot)
        else:
            safe_log(
                "Diagnose response was not valid JSON; attempting plain-text fallback."
            )
            problem_statement = _fallback_problem_statement_from_text(
                response, polyglot=polyglot
            )
            assert problem_statement, "empty response json and plain-text fallback failed"
    except Exception as e:
        safe_log(f"Error while diagnosing the problem: {e}")
        safe_log(traceback.format_exc())
        if max_attempts > 0:
            return diagnose_problem(
                entry,
                commit,
                root_dir,
                out_dir,
                patch_files=patch_files,
                max_attempts=max_attempts - 1,
                polyglot=polyglot,
            )
        else:
            return None
    return problem_statement


def save_metadata(metadata, output_dir):
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)
