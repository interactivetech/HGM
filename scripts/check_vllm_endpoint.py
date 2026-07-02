#!/usr/bin/env python3
import argparse
import json
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from llm import create_client, get_client_provider_config
from llm_withtools import convert_tool_info
from tools import load_all_tools


def pass_fail(name, ok, detail):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")
    return ok


def response_debug(response):
    if response is None:
        return "<no response>"
    if hasattr(response, "model_dump_json"):
        try:
            return response.model_dump_json(indent=2)
        except Exception:
            pass
    if hasattr(response, "model_dump"):
        try:
            return json.dumps(response.model_dump(mode="json"), indent=2, default=str)
        except Exception:
            pass
    return str(response)


def ensure_env(args):
    if args.vllm_base_url:
        os.environ["VLLM_BASE_URL"] = args.vllm_base_url
    if args.vllm_api_key:
        os.environ["VLLM_API_KEY"] = args.vllm_api_key
    if args.vllm_model:
        os.environ["VLLM_MODEL"] = args.vllm_model


def normal_chat(client, model):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: HGM_VLLM_OK"}],
        temperature=0,
    )
    content = response.choices[0].message.content or ""
    return response, "HGM_VLLM_OK" in content


def json_chat(client, model):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": 'Return a JSON object with {"status":"ok","value":7}.',
            }
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or ""
    parsed = json.loads(content)
    return response, parsed.get("status") == "ok"


def simple_tool_chat(client, model):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "echo_value",
                "description": "Echo a string value.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                    },
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Call echo_value with value set to HGM_TOOL_OK.",
            }
        ],
        tools=tools,
        tool_choice="auto",
        temperature=0,
    )
    tool_calls = response.choices[0].message.tool_calls or []
    if not tool_calls:
        return response, False, "No tool call returned."
    args = tool_calls[0].function.arguments
    parsed = json.loads(args)
    return response, parsed.get("value") == "HGM_TOOL_OK", args


def hgm_tool_schema_chat(client, model):
    all_tools = load_all_tools(logging=lambda *_args, **_kwargs: None)
    preferred_names = {"bash", "edit", "file_editor", "python_exec"}
    selected_tools = [
        convert_tool_info(tool["info"], model=f"vllm:{model}")
        for tool in all_tools
        if tool["info"]["name"] in preferred_names
    ]
    if not selected_tools:
        selected_tools = [
            convert_tool_info(all_tools[0]["info"], model=f"vllm:{model}")
        ]
    tool_name = selected_tools[0]["function"]["name"]
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Use the {tool_name} tool once. Do not answer in plain text until "
                    "after the tool call."
                ),
            }
        ],
        tools=selected_tools,
        tool_choice="auto",
        temperature=0,
    )
    tool_calls = response.choices[0].message.tool_calls or []
    if not tool_calls:
        return response, False, "No HGM-style tool call returned."
    raw_args = tool_calls[0].function.arguments
    parsed = json.loads(raw_args)
    return response, isinstance(parsed, dict), raw_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias", default="vllm-qwen")
    parser.add_argument("--vllm_base_url", default=None)
    parser.add_argument("--vllm_api_key", default=None)
    parser.add_argument("--vllm_model", default=None)
    parser.add_argument("--test-tools", action="store_true")
    parser.add_argument("--skip-tool-test", action="store_true")
    args = parser.parse_args()

    ensure_env(args)
    client, resolved_model = create_client(args.alias)
    provider = get_client_provider_config(client, resolved_model)
    print(
        "Resolved endpoint "
        f"provider={provider.provider} "
        f"base_url={provider.base_url} "
        f"resolved_model={resolved_model} "
        f"model_source={provider.model_source}"
    )

    failures = 0

    try:
        models = client.models.list()
        model_ids = [m.id for m in getattr(models, "data", [])]
        if not pass_fail("/models", bool(model_ids), model_ids[:5]):
            failures += 1
    except Exception as exc:
        failures += 1
        pass_fail("/models", False, exc)
        print(traceback.format_exc())

    try:
        response, ok = normal_chat(client, resolved_model)
        if not pass_fail("chat completion", ok, response.choices[0].message.content):
            failures += 1
    except Exception as exc:
        failures += 1
        pass_fail("chat completion", False, exc)
        print(traceback.format_exc())

    try:
        response, ok = json_chat(client, resolved_model)
        if not pass_fail("JSON output", ok, response.choices[0].message.content):
            failures += 1
    except Exception as exc:
        failures += 1
        pass_fail("JSON output", False, exc)
        print(traceback.format_exc())

    if args.skip_tool_test:
        pass_fail(
            "tool calling",
            True,
            "Skipped by user. Use --test-tools to validate auto tool calling.",
        )
    elif args.test_tools:
        tool_hint = (
            "If this fails with no tool calls, restart vLLM with "
            "--enable-auto-tool-choice --tool-call-parser hermes --generation-config vllm"
        )
        try:
            response, ok, detail = simple_tool_chat(client, resolved_model)
            if not pass_fail("tool calling auto", ok, detail):
                failures += 1
                print(response_debug(response))
                print(tool_hint)
        except Exception as exc:
            failures += 1
            pass_fail("tool calling auto", False, exc)
            print(traceback.format_exc())
            print(tool_hint)

        try:
            response, ok, detail = hgm_tool_schema_chat(client, resolved_model)
            if not pass_fail("HGM tool schema", ok, detail):
                failures += 1
                print(response_debug(response))
                print(tool_hint)
        except Exception as exc:
            failures += 1
            pass_fail("HGM tool schema", False, exc)
            print(traceback.format_exc())
            print(tool_hint)

    if failures:
        print(f"Endpoint check failed with {failures} failing checks.")
        sys.exit(1)
    print("All endpoint checks passed.")


if __name__ == "__main__":
    main()
