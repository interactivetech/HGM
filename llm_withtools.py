# This file is adapted from https://github.com/jennyzzt/dgm.

import copy
import json
import threading
import traceback
from time import time

import openai

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import backoff
except ImportError:
    class _BackoffShim:
        @staticmethod
        def expo(*args, **kwargs):
            return None

        @staticmethod
        def on_exception(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

    backoff = _BackoffShim()

from llm import create_client, get_client_provider_config, is_vllm_model
from tools import load_all_tools

CLAUDE_MODEL = "anthropic/claude-sonnet-4"
OPENAI_MODEL = "gpt-5"
STALL_INSPECTION_LIMIT = 4
INSPECTION_BASH_PREFIXES = (
    "git status",
    "git diff",
    "git show",
    "git log",
    "sed -n",
    "cat -n",
    "ls ",
    "find ",
    "pwd",
)
WORKTREE_CHANGE_MARKERS = (
    "diff --git",
    "modified:",
    "new file mode",
    "deleted file mode",
    "untracked files:",
    "changes not staged for commit",
)
FIX_COMPLETE_MARKERS = (
    "fix is complete",
    "the fix is complete",
    "fix complete",
)


def provider_retry_exceptions():
    exceptions = []
    for attr in ("RateLimitError", "APITimeoutError"):
        exc = getattr(openai, attr, None)
        if isinstance(exc, type) and issubclass(exc, BaseException):
            exceptions.append(exc)
    if anthropic is not None:
        exceptions.extend([anthropic.RateLimitError, anthropic.APIStatusError])
    if not exceptions:
        exceptions = [Exception]
    return tuple(exceptions)


def _logger(logging_fn):
    return logging_fn or print


def _to_plain_data(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _to_plain_data(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain_data(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_plain_data(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return _to_plain_data(obj.model_dump(mode="json"))
    if hasattr(obj, "to_dict"):
        return _to_plain_data(obj.to_dict())
    return str(obj)


def _message_to_dict(message):
    message = _to_plain_data(message)
    if isinstance(message, dict):
        return message
    return {"role": "assistant", "content": str(message)}


def _response_to_debug_string(response):
    try:
        return json.dumps(_to_plain_data(response), indent=2, default=str)
    except Exception:
        return str(response)


def _log_request(logging_fn, provider_config, request_type, tool_choice, model):
    log = _logger(logging_fn)
    log(
        "LLM request "
        f"provider={provider_config.provider} "
        f"base_url={provider_config.base_url or '<default>'} "
        f"resolved_model={provider_config.resolved_model or model} "
        f"request_type={request_type} "
        f"tool_choice={tool_choice}"
    )


def _get_value(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    value = getattr(obj, key, default)
    if value is not default:
        return value
    try:
        return obj.get(key, default)
    except Exception:
        return default


def _response_usage_summary(response, elapsed=None):
    usage = _get_value(response, "usage")
    if usage is None:
        return "usage=unavailable"
    parts = []
    for field, label in (
        ("prompt_tokens", "prompt"),
        ("input_tokens", "input"),
        ("completion_tokens", "completion"),
        ("output_tokens", "output"),
        ("total_tokens", "total"),
    ):
        value = _get_value(usage, field)
        if value is not None:
            parts.append(f"{label}={value}")
    if elapsed and elapsed > 0:
        total_tokens = (
            _get_value(usage, "total_tokens")
            or _get_value(usage, "output_tokens")
            or _get_value(usage, "completion_tokens")
            or _get_value(usage, "input_tokens")
            or _get_value(usage, "prompt_tokens")
        )
        if total_tokens is not None:
            parts.append(f"rate={float(total_tokens) / elapsed:.2f}tok/s")
    if not parts:
        return "usage=unavailable"
    return "usage(" + ", ".join(parts) + ")"


def _start_progress_heartbeat(label, interval=30):
    stop_event = threading.Event()
    started = time()

    def _heartbeat():
        while not stop_event.wait(interval):
            elapsed = time() - started
            print(f"{label} still running after {elapsed:.1f}s", flush=True)

    thread = threading.Thread(target=_heartbeat, daemon=True)
    thread.start()
    return stop_event, thread, started


def _response_text_content(response):
    choices = _get_value(response, "choices") or []
    if choices:
        message = _get_value(choices[0], "message")
        content = _get_value(message, "content", "")
        if content is not None:
            return content
    output_items = _get_value(response, "output") or []
    if isinstance(output_items, list):
        texts = []
        for item in output_items:
            text = _get_value(item, "text")
            if text:
                texts.append(str(text))
        if texts:
            return "".join(texts)
    return ""


def _tool_command_text(tool_name, tool_input):
    if not isinstance(tool_input, dict):
        return ""
    if tool_name == "bash":
        return str(tool_input.get("command", ""))
    if tool_name == "editor":
        return " ".join(
            part for part in [tool_input.get("command", ""), tool_input.get("path", "")] if part
        )
    return ""


def _is_inspection_tool_call(tool_name, tool_input):
    command = _tool_command_text(tool_name, tool_input).lower()
    if not command:
        return False
    if tool_name == "editor":
        return tool_input.get("command") == "view"
    if tool_name == "bash":
        return any(command.startswith(prefix) or prefix in command for prefix in INSPECTION_BASH_PREFIXES)
    return False


def _tool_result_shows_worktree_changes(tool_name, tool_input, tool_result):
    result_text = str(tool_result).lower()
    if "nothing to commit, working tree clean" in result_text:
        return False
    if tool_name == "editor" and isinstance(tool_input, dict):
        if tool_input.get("command") in {"edit", "create"}:
            return True
    return any(marker in result_text for marker in WORKTREE_CHANGE_MARKERS)


def _response_mentions_fix_complete(response):
    response_text = _response_to_debug_string(response).lower()
    return any(marker in response_text for marker in FIX_COMPLETE_MARKERS)


def _update_stall_guard_state(
    tool_name,
    tool_input,
    tool_result,
    response,
    saw_nonempty_worktree,
    inspection_streak,
    stall_inspection_limit,
):
    saw_nonempty_worktree = saw_nonempty_worktree or _tool_result_shows_worktree_changes(
        tool_name, tool_input, tool_result
    )
    if _is_inspection_tool_call(tool_name, tool_input):
        inspection_streak += 1
    else:
        inspection_streak = 0

    should_stop = False
    if saw_nonempty_worktree:
        if inspection_streak >= stall_inspection_limit:
            should_stop = True
        elif _response_mentions_fix_complete(response):
            should_stop = True

    return saw_nonempty_worktree, inspection_streak, should_stop


def process_tool_call(tools_dict, tool_name, tool_input):
    try:
        if tool_name in tools_dict:
            return tools_dict[tool_name]["function"](**tool_input)
        return f"Error: Tool '{tool_name}' not found"
    except Exception as exc:
        return f"Error executing tool '{tool_name}': {exc}"


@backoff.on_exception(
    backoff.expo,
    provider_retry_exceptions(),
    max_time=600,
    max_value=60,
)
def get_response_withtools(
    client, model, messages, tools, tool_choice, logging=None, max_retry=3
):
    log = _logger(logging)
    provider_config = get_client_provider_config(client, model)
    request_type = (
        "responses.create"
        if provider_config.provider == "openai" and is_vllm_model(provider_config.alias) is False and (model.startswith("o") or "gpt" in model.lower())
        else "chat.completions.create"
    )
    message_chars = len(json.dumps(_to_plain_data(messages), ensure_ascii=False, default=str))
    log(
        "LLM request "
        f"provider={provider_config.provider} "
        f"base_url={provider_config.base_url or '<default>'} "
        f"resolved_model={provider_config.resolved_model or model} "
        f"request_type={request_type} "
        f"tool_choice={tool_choice} "
        f"message_count={len(messages)} "
        f"approx_input_chars={message_chars} "
        f"tool_count={len(tools)}"
    )
    stop_event, heartbeat_thread, started = _start_progress_heartbeat(
        f"LLM request model={provider_config.resolved_model or model}"
    )
    response = None
    try:
        if model.startswith("o") or "gpt" in model.lower():
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "system",
                        "content": "You are the best coder in the world!",
                    }
                ]
                + messages,
                tool_choice=tool_choice,
                tools=tools,
                parallel_tool_calls=False,
            )
        else:
            chat_kwargs = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are the best coder in the world!",
                    }
                ]
                + messages,
                "tool_choice": tool_choice,
                "tools": tools,
            }
            if provider_config.provider != "vllm":
                chat_kwargs["parallel_tool_calls"] = False
            response = client.chat.completions.create(**chat_kwargs)
    except Exception as exc:
        elapsed = time() - started
        log(
            "Error in get_response_withtools "
            f"provider={provider_config.provider} "
            f"base_url={provider_config.base_url or '<default>'} "
            f"resolved_model={provider_config.resolved_model or model} "
            f"request_type={request_type} "
            f"after={elapsed:.2f}s: {exc}"
        )
        log(traceback.format_exc())
        if max_retry > 0:
            stop_event.set()
            heartbeat_thread.join(timeout=1.0)
            return get_response_withtools(
                client, model, messages, tools, tool_choice, logging, max_retry - 1
            )
        raise
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=1.0)

    content = _response_text_content(response)
    log(
        "LLM request completed "
        f"provider={provider_config.provider} "
        f"base_url={provider_config.base_url or '<default>'} "
        f"resolved_model={provider_config.resolved_model or model} "
        f"request_type={request_type} "
        f"elapsed={time() - started:.2f}s "
        f"response_chars={len(content) if content else 0} "
        f"{_response_usage_summary(response, elapsed=time() - started)}"
    )
    return response


def check_for_tool_use(response, model="", logging=None):
    log = _logger(logging)
    if model.startswith("o") or "gpt" in model.lower():
        output_items = getattr(response, "output", []) or []
        for item in output_items:
            item_type = getattr(item, "type", None) or (
                item.get("type") if isinstance(item, dict) else None
            )
            if item_type == "function_call":
                arguments = getattr(item, "arguments", None)
                try:
                    tool_input = json.loads(arguments)
                except Exception as exc:
                    log(f"Failed to parse tool arguments: {arguments}")
                    raise ValueError(f"Invalid tool-call JSON arguments: {exc}") from exc
                return {
                    "tool_id": getattr(item, "call_id", None),
                    "tool_name": getattr(item, "name", None),
                    "tool_input": tool_input,
                }
        return None

    message = response.choices[0].message
    tool_calls = getattr(message, "tool_calls", None) or []
    if not tool_calls:
        return None
    call = tool_calls[0]
    raw_arguments = call.function.arguments
    try:
        tool_input = json.loads(raw_arguments)
    except Exception as exc:
        log("vLLM returned invalid tool-call arguments.")
        log(f"Raw tool arguments: {raw_arguments}")
        log(f"Raw response: {_response_to_debug_string(response)}")
        raise ValueError(f"Invalid tool-call JSON arguments: {exc}") from exc
    return {
        "tool_id": call.id,
        "tool_name": call.function.name,
        "tool_input": tool_input,
    }


def convert_tool_info(tool_info, model=None):
    if is_vllm_model(model or ""):
        required = list(tool_info["input_schema"]["properties"].keys())
        return {
            "type": "function",
            "function": {
                "name": tool_info["name"],
                "description": tool_info["description"],
                "parameters": {
                    "type": "object",
                    "properties": tool_info["input_schema"]["properties"],
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    if model.startswith("o") or "gpt" in model.lower():

        def add_additional_properties(schema):
            if isinstance(schema, dict):
                if "properties" in schema:
                    schema["additionalProperties"] = False
                for value in schema.values():
                    add_additional_properties(value)

        input_schema = copy.deepcopy(tool_info["input_schema"])
        add_additional_properties(input_schema)
        for prop_name in input_schema["properties"].keys():
            if prop_name not in input_schema["required"]:
                input_schema["required"].append(prop_name)
                prop_type = copy.deepcopy(input_schema["properties"][prop_name]["type"])
                if isinstance(prop_type, str):
                    input_schema["properties"][prop_name]["type"] = [prop_type, "null"]
                elif isinstance(prop_type, list):
                    input_schema["properties"][prop_name]["type"] = prop_type + [
                        "null"
                    ]

        return {
            "type": "function",
            "name": tool_info["name"],
            "description": tool_info["description"],
            "parameters": input_schema,
            "strict": True,
        }

    required = list(tool_info["input_schema"]["properties"].keys())
    return {
        "type": "function",
        "function": {
            "name": tool_info["name"],
            "description": tool_info["description"],
            "parameters": {
                "type": "object",
                "properties": tool_info["input_schema"]["properties"],
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def chat_with_agent_openai(
    msg,
    model=OPENAI_MODEL,
    msg_history=None,
    logging=print,
    max_llm_calls=100,
    timeout=3600,
    stall_inspection_limit=STALL_INSPECTION_LIMIT,
):
    start_time = time()
    if msg_history is None:
        msg_history = []
    new_msg_history = [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": msg}],
        }
    ]
    separator = "=" * 10
    logging(f"\n{separator} User Instruction {separator}\n{msg}")

    client, client_model = create_client(model)
    provider_config = get_client_provider_config(client, client_model)
    logging(
        "LLM session "
        f"provider={provider_config.provider} "
        f"base_url={provider_config.base_url or '<default>'} "
        f"resolved_model={client_model}"
    )

    all_tools = load_all_tools(logging=logging)
    tools_dict = {tool["info"]["name"]: tool for tool in all_tools}
    tools = [convert_tool_info(tool["info"], model=client_model) for tool in all_tools]
    saw_nonempty_worktree = False
    inspection_streak = 0

    for i in range(max_llm_calls):
        if timeout * 0.9 < time() - start_time:
            logging("Timeout reached, stopping further LLM calls.")
            return new_msg_history, i

        logging(
            f"Agent turn {i + 1}/{max_llm_calls} starting; "
            f"elapsed={time() - start_time:.1f}s; "
            f"messages={len(msg_history + new_msg_history)}"
        )
        response = get_response_withtools(
            client=client,
            model=client_model,
            messages=msg_history + new_msg_history,
            tool_choice="auto",
            tools=tools,
            logging=logging,
        )
        logging(f"Tool Response: {_response_to_debug_string(response)}")
        tool_use = check_for_tool_use(response, model=client_model, logging=logging)
        new_msg_history.extend([_message_to_dict(item) for item in response.output])
        if not tool_use:
            return new_msg_history, i + 1

        tool_name = tool_use["tool_name"]
        tool_input = tool_use["tool_input"]
        tool_result = process_tool_call(tools_dict, tool_name, tool_input)

        logging(f"Tool Used: {tool_name}")
        logging(f"Tool Input: {tool_input}")
        logging(f"Tool Result: {tool_result}")

        new_msg_history.append(
            {
                "type": "function_call_output",
                "call_id": tool_use["tool_id"],
                "output": tool_result,
            }
        )
        saw_nonempty_worktree, inspection_streak, should_stop = _update_stall_guard_state(
            tool_name,
            tool_input,
            tool_result,
            response,
            saw_nonempty_worktree,
            inspection_streak,
            stall_inspection_limit,
        )
        if should_stop:
            logging(
                "Early stopping agent loop after repeated inspection-only turns "
                "with a non-empty worktree or a completion signal."
            )
            return new_msg_history, i + 1

    return new_msg_history, max_llm_calls


def chat_with_agent_open_router(
    msg,
    model=CLAUDE_MODEL,
    msg_history=None,
    logging=print,
    max_llm_calls=100,
    timeout=3600,
    stall_inspection_limit=STALL_INSPECTION_LIMIT,
):
    start_time = time()
    if msg_history is None:
        msg_history = []
    new_msg_history = [{"role": "user", "content": msg}]
    separator = "=" * 10
    logging(f"\n{separator} User Instruction {separator}\n{msg}")

    client, client_model = create_client(model)
    provider_config = get_client_provider_config(client, client_model)
    logging(
        "LLM session "
        f"provider={provider_config.provider} "
        f"base_url={provider_config.base_url or '<default>'} "
        f"resolved_model={client_model}"
    )

    all_tools = load_all_tools(logging=logging)
    tools_dict = {tool["info"]["name"]: tool for tool in all_tools}
    tools = [convert_tool_info(tool["info"], model=client_model) for tool in all_tools]
    saw_nonempty_worktree = False
    inspection_streak = 0

    for i in range(max_llm_calls):
        if timeout * 0.9 < time() - start_time:
            logging("Timeout reached, stopping further LLM calls.")
            return new_msg_history, i

        logging(
            f"Agent turn {i + 1}/{max_llm_calls} starting; "
            f"elapsed={time() - start_time:.1f}s; "
            f"messages={len(msg_history + new_msg_history)}"
        )
        response = get_response_withtools(
            client=client,
            model=client_model,
            messages=msg_history + new_msg_history,
            tool_choice="auto",
            tools=tools,
            logging=logging,
        )
        logging(f"Tool Response: {_response_to_debug_string(response)}")
        tool_use = check_for_tool_use(response, model=client_model, logging=logging)
        new_msg_history.append(_message_to_dict(response.choices[0].message))
        if not tool_use:
            return new_msg_history, i + 1

        tool_name = tool_use["tool_name"]
        tool_input = tool_use["tool_input"]
        tool_result = process_tool_call(tools_dict, tool_name, tool_input)

        logging(f"Tool Used: {tool_name}")
        logging(f"Tool Input: {tool_input}")
        logging(f"Tool Result: {tool_result}")

        new_msg_history.append(
            {
                "role": "tool",
                "tool_call_id": tool_use["tool_id"],
                "name": tool_use["tool_name"],
                "content": f"{tool_result}",
            }
        )
        saw_nonempty_worktree, inspection_streak, should_stop = _update_stall_guard_state(
            tool_name,
            tool_input,
            tool_result,
            response,
            saw_nonempty_worktree,
            inspection_streak,
            stall_inspection_limit,
        )
        if should_stop:
            logging(
                "Early stopping agent loop after repeated inspection-only turns "
                "with a non-empty worktree or a completion signal."
            )
            return new_msg_history, i + 1

    return new_msg_history, max_llm_calls


def convert_msg_history_openai(msg_history):
    new_msg_history = []
    for msg in msg_history:
        plain_msg = _message_to_dict(msg)
        role = plain_msg.get("role", "assistant")
        content = plain_msg.get("content")
        if content is None and "output" in plain_msg:
            content = "Tool Result: " + str(plain_msg.get("output", ""))
        new_msg_history.append({"role": role, "content": content})
    return new_msg_history


def convert_msg_history_open_router(msg_history):
    new_msg_history = []
    for msg in msg_history:
        plain_msg = _message_to_dict(msg)
        role = plain_msg.get("role", "")
        if "content" in plain_msg:
            if role == "tool":
                content = "Tool Result: " + str(plain_msg["content"])
            else:
                content = plain_msg["content"]
        else:
            tool_calls = plain_msg.get("tool_calls", [])
            if tool_calls:
                tool_call = tool_calls[0]
                function = tool_call.get("function", {})
                content = (
                    f"Function: {function.get('name')}\n"
                    f"Arguments: {function.get('arguments')}"
                )
            else:
                content = str(plain_msg)
        new_msg_history.append({"role": role, "content": content})
    return new_msg_history


def convert_msg_history(msg_history, model=None):
    if model.startswith("o") or "gpt" in model.lower():
        return convert_msg_history_openai(msg_history)
    return convert_msg_history_open_router(msg_history)


def chat_with_agent(
    msg,
    model=CLAUDE_MODEL,
    msg_history=None,
    logging=print,
    convert=False,
    max_llm_calls=100,
    timeout=3600,
    stall_inspection_limit=STALL_INSPECTION_LIMIT,
):
    session_started = time()
    if msg_history is None:
        msg_history = []

    if model.startswith("o") or "gpt" in model.lower():
        new_msg_history, n_llm_calls = chat_with_agent_openai(
            msg,
            model=model,
            msg_history=msg_history,
            logging=logging,
            max_llm_calls=max_llm_calls,
            timeout=timeout,
            stall_inspection_limit=stall_inspection_limit,
        )
    else:
        new_msg_history, n_llm_calls = chat_with_agent_open_router(
            msg,
            model=model,
            msg_history=msg_history,
            logging=logging,
            max_llm_calls=max_llm_calls,
            timeout=timeout,
            stall_inspection_limit=stall_inspection_limit,
        )

    new_msg_history = msg_history + new_msg_history
    logging(
        f"LLM session completed in {time() - session_started:.2f}s; "
        f"calls={n_llm_calls}; messages={len(new_msg_history)}"
    )
    return new_msg_history, n_llm_calls


if __name__ == "__main__":
    msg = (
        "First create the current directory. Then implement a function that returns "
        "the current directory and save it in the directory just created. Finally "
        "call the function and return the result. In the end, summarize what you did."
    )
    model = "vllm-qwen"
    history, _ = chat_with_agent(msg, model=model, max_llm_calls=2)
    from utils.eval_utils import msg_history_to_report

    print(msg_history_to_report("hgm", history, model=model))
