import argparse
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import openai
except ImportError:  # pragma: no cover - requirements install should provide it.
    openai = None

try:
    from tenacity import (
        Retrying,
        retry_if_not_exception_type,
        stop_after_attempt,
        wait_exponential,
    )
except ImportError:  # pragma: no cover - requirements install should provide it.
    Retrying = None
    retry_if_not_exception_type = None
    stop_after_attempt = None
    wait_exponential = None

from utils.git_utils import diff_versus_commit


DEFAULT_MODEL = "gpt-5"
MAX_OUTPUT_CHARS = 20000
DEFAULT_STEP_LIMIT = int(os.getenv("HGM_MINISWE_STEP_LIMIT", "250"))
DEFAULT_COMMAND_TIMEOUT = int(os.getenv("HGM_MINISWE_COMMAND_TIMEOUT", "120"))
DEFAULT_TIMEOUT_BUFFER = int(os.getenv("HGM_MINISWE_TIMEOUT_BUFFER", "60"))
MODEL_REQUEST_TIMEOUT = os.getenv("HGM_MINISWE_REQUEST_TIMEOUT") or os.getenv(
    "HGM_MINISWE_LITELLM_TIMEOUT"
)
PROGRESS_PREVIEW_CHARS = int(os.getenv("HGM_MINISWE_LOG_PREVIEW_CHARS", "240"))
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}


class AgentFlow(Exception):
    def __init__(self, message: Dict[str, Any]):
        self.message = message
        super().__init__(message.get("content", ""))


class Submitted(AgentFlow):
    pass


class FormatError(AgentFlow):
    pass


class LimitsExceeded(AgentFlow):
    pass


class TimeExceeded(LimitsExceeded):
    pass


def log_progress(message: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[hgm-miniswe {timestamp}] {message}", file=sys.stderr, flush=True)


def preview_text(value: Any, limit: int = PROGRESS_PREVIEW_CHARS) -> str:
    text = str(value).replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[:limit] + f"... <truncated {len(text) - limit} chars>"


def request_timeout_value() -> Optional[float]:
    if not MODEL_REQUEST_TIMEOUT:
        return None
    try:
        return float(MODEL_REQUEST_TIMEOUT)
    except ValueError:
        log_progress(
            "invalid_model_request_timeout "
            f"value={preview_text(MODEL_REQUEST_TIMEOUT)} using=openai_default"
        )
        return None


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump(mode="json"))
        except Exception:
            return repr(value)
    return repr(value)


def _extract_text(response) -> str:
    choices = getattr(response, "choices", None) or []
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if content is not None:
            return content
    return ""


def _legacy_vllm_base_url(alias: str) -> Optional[str]:
    match = re.match(r"^vllm-[^-]+-(?P<host>.+)$", alias, re.IGNORECASE)
    if not match:
        return None
    host = match.group("host").strip()
    if not re.match(r"^[A-Za-z0-9_.:-]+$", host):
        return None
    return f"http://{host}:8000/v1"


def _is_openai_direct_model(model: str) -> bool:
    lower = model.lower()
    return lower.startswith("gpt") or lower.startswith("o")


def _retry_attempts(logger: logging.Logger):
    if Retrying is None:
        return [None]
    abort_exceptions = [KeyboardInterrupt]
    if openai is not None:
        abort_exceptions.extend(
            [
                openai.AuthenticationError,
                openai.BadRequestError,
                openai.NotFoundError,
                openai.PermissionDeniedError,
            ]
        )
    return Retrying(
        reraise=True,
        stop=stop_after_attempt(
            int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))
        ),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=log_model_retry,
        retry=retry_if_not_exception_type(tuple(abort_exceptions)),
    )


def log_model_retry(retry_state):
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    next_sleep = getattr(retry_state.next_action, "sleep", None)
    log_progress(
        "model_query_retry "
        f"attempt={retry_state.attempt_number} "
        f"next_sleep={next_sleep} "
        f"exception={type(exception).__name__ if exception else ''}:"
        f"{preview_text(exception) if exception else ''}"
    )


def _message_to_dict(value) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return dict(value)
    return {
        key: getattr(value, key)
        for key in ("role", "content", "tool_calls")
        if hasattr(value, key)
    }


def _tool_attr(tool_call, name: str, default=None):
    if isinstance(tool_call, dict):
        return tool_call.get(name, default)
    return getattr(tool_call, name, default)


def _function_attr(function, name: str, default=None):
    if isinstance(function, dict):
        return function.get(name, default)
    return getattr(function, name, default)


def parse_toolcall_actions(tool_calls: List[Any]) -> List[Dict[str, str]]:
    if not tool_calls:
        raise FormatError(
            {
                "role": "user",
                "content": (
                    "Tool call error: no tool calls found. Every response must "
                    "call the bash tool at least once."
                ),
                "extra": {"interrupt_type": "FormatError", "n_actions": 0},
            }
        )
    actions = []
    for tool_call in tool_calls:
        function = _tool_attr(tool_call, "function", {})
        name = _function_attr(function, "name", "")
        arguments = _function_attr(function, "arguments", "{}")
        error_msg = ""
        try:
            args = json.loads(arguments or "{}")
        except Exception as exc:
            args = {}
            error_msg = f"Error parsing tool call arguments: {exc}. "
        if name != "bash":
            error_msg += f"Unknown tool '{name}'. "
        if not isinstance(args, dict) or "command" not in args:
            error_msg += "Missing 'command' argument in bash tool call."
        if error_msg:
            raise FormatError(
                {
                    "role": "user",
                    "content": f"Tool call error: {error_msg.strip()}",
                    "extra": {"interrupt_type": "FormatError"},
                }
            )
        actions.append(
            {
                "command": str(args["command"]),
                "tool_call_id": str(_tool_attr(tool_call, "id", "")),
            }
        )
    return actions


@dataclass
class MiniModel:
    model_name: str

    def __post_init__(self):
        if os.getenv("HGM_MINISWE_DRY_RUN") == "1":
            self.provider = "dry-run"
            self.resolved_model = self.model_name
            self.base_url = None
            self.api_key = None
            self.client = None
            return
        if openai is None:
            raise RuntimeError("The openai package is required for coding_agent.py")
        self.provider = "openai"
        self.resolved_model = self.model_name
        self.base_url = None
        self.api_key = None
        self.client = None

        if self.model_name.lower().startswith("vllm"):
            self.provider = "vllm"
            self.base_url = os.getenv("VLLM_BASE_URL") or _legacy_vllm_base_url(
                self.model_name
            )
            if not self.base_url:
                raise ValueError(
                    "vLLM model selected but VLLM_BASE_URL is not set."
                )
            self.base_url = self.base_url.rstrip("/")
            self.api_key = os.getenv("VLLM_API_KEY") or "dummy"
            explicit = self.model_name.split(":", 1)[1].strip() if ":" in self.model_name else ""
            self.resolved_model = explicit or os.getenv("VLLM_MODEL") or self.model_name
            self.client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)
            return

        if _is_openai_direct_model(self.model_name):
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.client = openai.OpenAI(api_key=self.api_key)
            return

        self.provider = "openrouter"
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_key = os.getenv("OpenRouter_API_KEY")
        if self.model_name.startswith("openrouter/"):
            self.resolved_model = self.model_name[len("openrouter/") :]
        else:
            self.resolved_model = self.model_name
        self.client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)

    def format_message(self, role: str, content: str, extra: Optional[dict] = None):
        message = {"role": role, "content": content}
        if extra:
            message["extra"] = extra
        return message

    def _api_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        api_messages = []
        for message in messages:
            role = message.get("role")
            if role == "exit":
                continue
            if role not in {"system", "user", "assistant", "tool"}:
                role = "user"
            api_message = {
                key: value
                for key, value in message.items()
                if key in {"role", "content", "tool_calls", "tool_call_id", "name"}
            }
            api_message["role"] = role
            if api_message.get("content") is None:
                api_message["content"] = ""
            if role != "assistant":
                api_message.pop("tool_calls", None)
            api_messages.append(api_message)
        return api_messages

    def query(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        started = time.time()
        kwargs = {
            "model": self.resolved_model,
            "messages": self._api_messages(messages),
            "tools": [BASH_TOOL],
        }
        request_timeout = request_timeout_value()
        if request_timeout is not None:
            kwargs["timeout"] = request_timeout
        if not self.resolved_model.lower().startswith("o"):
            kwargs["temperature"] = 0.2
        logger = logging.getLogger("hgm_miniswe.model")
        response = None
        log_progress(
            "model_query_start "
            f"provider={self.provider} model={self.resolved_model} "
            f"messages={len(messages)} "
            f"request_timeout={request_timeout if request_timeout is not None else 'openai_default'} "
            f"retry_attempts={os.getenv('MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT', '10')}"
        )
        try:
            for attempt in _retry_attempts(logger):
                if attempt is None:
                    response = self.client.chat.completions.create(**kwargs)
                else:
                    with attempt:
                        response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            log_progress(
                "model_query_failed "
                f"elapsed={time.time() - started:.2f}s "
                f"request_timeout={request_timeout if request_timeout is not None else 'openai_default'} "
                f"exception={type(exc).__name__}:{preview_text(exc)}"
            )
            raise
        message_obj = response.choices[0].message
        message = _message_to_dict(message_obj)
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls") or []
        if os.getenv("HGM_MINISWE_TEXT_FALLBACK") == "1" and not tool_calls:
            actions = parse_actions(content)
        else:
            actions = parse_toolcall_actions(tool_calls)
        usage = _jsonable(getattr(response, "usage", None))
        elapsed = time.time() - started
        action_preview = "; ".join(
            preview_text(action.get("command", "")) for action in actions[:3]
        )
        log_progress(
            "model_query_done "
            f"elapsed={elapsed:.2f}s actions={len(actions)} "
            f"content_chars={len(content)} usage={preview_text(usage, 300)} "
            f"action_preview={action_preview}"
        )
        message["role"] = "assistant"
        message["content"] = content
        message["extra"] = {
            "actions": actions,
            "provider": self.provider,
            "resolved_model": self.resolved_model,
            "usage": usage,
            "elapsed": elapsed,
            "timestamp": time.time(),
        }
        return message

    def format_observation_messages(
        self, message: Dict[str, Any], outputs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        messages = []
        actions = message.get("extra", {}).get("actions", [])
        not_executed = {
            "output": "",
            "returncode": -1,
            "exception_info": "action was not executed",
        }
        padded_outputs = outputs + [not_executed] * (len(actions) - len(outputs))
        for action, output in zip(actions, padded_outputs):
            content = (
                f"<returncode>{output.get('returncode')}</returncode>\n"
                f"<output>\n{output.get('output', '')}</output>"
            )
            if output.get("exception_info"):
                content = f"<exception>{output['exception_info']}</exception>\n{content}"
            observation = {
                "role": "tool" if action.get("tool_call_id") else "user",
                "content": content,
                "extra": {
                    "raw_output": output.get("output", ""),
                    "returncode": output.get("returncode"),
                    "exception_info": output.get("exception_info", ""),
                    "timestamp": time.time(),
                },
            }
            if action.get("tool_call_id"):
                observation["tool_call_id"] = action["tool_call_id"]
            messages.append(observation)
        return messages

    def serialize(self):
        return {
            "info": {
                "config": {
                    "model": {
                        "model_name": self.model_name,
                        "resolved_model": self.resolved_model,
                        "provider": self.provider,
                        "base_url": self.base_url,
                    },
                    "model_type": "hgm_miniswe.OpenAICompatibleModel",
                }
            }
        }


@dataclass
class LocalEnvironment:
    cwd: str
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT

    def execute(self, action: Dict[str, Any]) -> Dict[str, Any]:
        command = str(action.get("command", ""))
        started = time.time()
        log_progress(f"tool_start command={preview_text(command)}")
        try:
            result = run_command(command, self.cwd, self.command_timeout)
            output = {
                "output": truncate_output(result.stdout),
                "returncode": result.returncode,
                "exception_info": "",
                "command": command,
            }
        except subprocess.TimeoutExpired as exc:
            raw_output = exc.output or ""
            if isinstance(raw_output, bytes):
                raw_output = raw_output.decode("utf-8", errors="replace")
            output = {
                "output": truncate_output(raw_output),
                "returncode": -1,
                "exception_info": f"Command timed out after {self.command_timeout}s",
                "command": command,
            }
        except Exception as exc:
            output = {
                "output": "",
                "returncode": -1,
                "exception_info": f"{type(exc).__name__}: {exc}",
                "command": command,
            }
        log_progress(
            "tool_done "
            f"returncode={output.get('returncode')} "
            f"elapsed={time.time() - started:.2f}s "
            f"output_chars={len(output.get('output', '') or '')} "
            f"exception={preview_text(output.get('exception_info', ''))}"
        )
        self._check_finished(output)
        return output

    def _check_finished(self, output: Dict[str, Any]):
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if (
            output.get("returncode") == 0
            and lines
            and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
        ):
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {
                        "exit_status": "Submitted",
                        "submission": submission,
                        "timestamp": time.time(),
                    },
                }
            )

    def serialize(self):
        return {
            "info": {
                "config": {
                    "environment": asdict(self),
                    "environment_type": "hgm_miniswe.LocalEnvironment",
                }
            }
        }


def run_command(command: str, cwd: str, timeout: int) -> subprocess.CompletedProcess:
    process = subprocess.Popen(
        command,
        shell=True,
        text=True,
        cwd=cwd,
        env=os.environ.copy(),
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        stdout, _ = process.communicate()
        raise subprocess.TimeoutExpired(command, timeout, output=stdout)
    return subprocess.CompletedProcess(command, process.returncode, stdout=stdout)


def truncate_output(output: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(output) <= limit:
        return output
    head = output[: limit // 2]
    tail = output[-limit // 2 :]
    return f"{head}\n\n... <truncated {len(output) - limit} chars> ...\n\n{tail}"


def parse_actions(content: str) -> List[Dict[str, str]]:
    patterns = [
        r"```(?:bash|sh|shell)?\s*\n(.*?)```",
        r"<bash>\s*(.*?)\s*</bash>",
    ]
    actions = []
    for pattern in patterns:
        actions.extend(match.strip() for match in re.findall(pattern, content, re.DOTALL))
    actions = [action for action in actions if action]
    if len(actions) != 1:
        raise FormatError(
            {
                "role": "user",
                "content": (
                    "Format error: expected exactly one shell action in a fenced "
                    "```bash block or <bash>...</bash> block. "
                    f"Found {len(actions)} actions. Try again with one action only."
                ),
                "extra": {
                    "interrupt_type": "FormatError",
                    "n_actions": len(actions),
                    "model_response": content,
                    "timestamp": time.time(),
                },
            }
        )
    return [{"command": actions[0]}]


def recursive_merge(*dicts):
    result = {}
    for item in dicts:
        for key, value in (item or {}).items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = recursive_merge(result[key], value)
            else:
                result[key] = value
    return result


@dataclass
class MiniAgentConfig:
    system_template: str
    instance_template: str
    step_limit: int = DEFAULT_STEP_LIMIT
    wall_time_limit_seconds: int = 0
    max_consecutive_format_errors: int = 3
    output_path: Optional[str] = None


class MiniSweStyleAgent:
    def __init__(self, model: MiniModel, env: LocalEnvironment, config: MiniAgentConfig):
        self.model = model
        self.env = env
        self.config = config
        self.messages: List[Dict[str, Any]] = []
        self.n_calls = 0
        self.n_consecutive_format_errors = 0
        self._start_time = time.time()
        self.extra_template_vars: Dict[str, Any] = {}

    def get_template_vars(self, **kwargs):
        return {
            "repo_path": self.env.cwd,
            "n_model_calls": self.n_calls,
            "elapsed_seconds": int(time.time() - self._start_time),
            **self.extra_template_vars,
            **kwargs,
        }

    def render(self, template: str) -> str:
        rendered = template
        for key, value in self.get_template_vars().items():
            rendered = rendered.replace("{{ " + key + " }}", str(value))
            rendered = rendered.replace("{{" + key + "}}", str(value))
        return rendered

    def add_messages(self, *messages: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.messages.extend(messages)
        return list(messages)

    def run(self, task: str, **kwargs) -> Dict[str, Any]:
        self.extra_template_vars = {"task": task, **kwargs}
        self.messages = []
        log_progress(
            "agent_start "
            f"instance_id={kwargs.get('instance_id', '')} "
            f"step_limit={self.config.step_limit} "
            f"wall_time_limit_seconds={self.config.wall_time_limit_seconds}"
        )
        self.add_messages(
            self.model.format_message("system", self.render(self.config.system_template)),
            self.model.format_message("user", self.render(self.config.instance_template)),
        )

        if os.getenv("HGM_MINISWE_DRY_RUN") == "1":
            self.add_messages(
                {
                    "role": "exit",
                    "content": "DryRun",
                    "extra": {"exit_status": "DryRun", "submission": ""},
                }
            )
            self.save(self.config.output_path)
            return self.messages[-1]["extra"]

        while True:
            try:
                self.step()
                self.n_consecutive_format_errors = 0
            except FormatError as exc:
                log_progress(
                    "agent_format_error "
                    f"consecutive={self.n_consecutive_format_errors + 1} "
                    f"message={preview_text(exc.message.get('content', ''))}"
                )
                self.n_consecutive_format_errors += 1
                if (
                    0 < self.config.max_consecutive_format_errors
                    <= self.n_consecutive_format_errors
                ):
                    self.add_messages(
                        exc.message,
                        {
                            "role": "exit",
                            "content": "RepeatedFormatError",
                            "extra": {
                                "exit_status": "RepeatedFormatError",
                                "submission": "",
                                "timestamp": time.time(),
                            },
                        },
                    )
                else:
                    self.add_messages(exc.message)
            except AgentFlow as exc:
                log_progress(
                    "agent_exit_signal "
                    f"status={exc.message.get('extra', {}).get('exit_status', type(exc).__name__)}"
                )
                self.add_messages(exc.message)
            except Exception as exc:
                log_progress(
                    "agent_exception "
                    f"type={type(exc).__name__} message={preview_text(exc)}"
                )
                self.add_messages(
                    {
                        "role": "exit",
                        "content": str(exc),
                        "extra": {
                            "exit_status": type(exc).__name__,
                            "submission": "",
                            "exception_str": str(exc),
                            "traceback": traceback.format_exc(),
                            "timestamp": time.time(),
                        },
                    }
                )
                raise
            finally:
                self.save(self.config.output_path)

            if self.messages and self.messages[-1].get("role") == "exit":
                extra = self.messages[-1].get("extra", {})
                log_progress(
                    "agent_done "
                    f"status={extra.get('exit_status', '')} "
                    f"api_calls={self.n_calls} "
                    f"submission_chars={len(extra.get('submission', '') or '')}"
                )
                return self.messages[-1].get("extra", {})

    def step(self) -> List[Dict[str, Any]]:
        return self.execute_actions(self.query())

    def query(self) -> Dict[str, Any]:
        elapsed = int(time.time() - self._start_time)
        if 0 < self.config.step_limit <= self.n_calls:
            log_progress(
                "agent_step_limit_exceeded "
                f"api_calls={self.n_calls} step_limit={self.config.step_limit}"
            )
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        if 0 < self.config.wall_time_limit_seconds <= elapsed:
            log_progress(
                "agent_time_exceeded "
                f"elapsed={elapsed}s "
                f"wall_time_limit_seconds={self.config.wall_time_limit_seconds}"
            )
            raise TimeExceeded(
                {
                    "role": "exit",
                    "content": "TimeExceeded",
                    "extra": {"exit_status": "TimeExceeded", "submission": ""},
                }
            )
        self.n_calls += 1
        log_progress(
            "agent_step_start "
            f"step={self.n_calls} elapsed={elapsed}s messages={len(self.messages)}"
        )
        message = self.model.query(self.messages)
        self.add_messages(message)
        log_progress(
            "agent_step_model_done "
            f"step={self.n_calls} actions={len(message.get('extra', {}).get('actions', []))}"
        )
        return message

    def execute_actions(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        actions = message.get("extra", {}).get("actions", [])
        log_progress(f"agent_step_execute step={self.n_calls} actions={len(actions)}")
        outputs = [
            self.env.execute(action)
            for action in actions
        ]
        log_progress(f"agent_step_done step={self.n_calls} observations={len(outputs)}")
        return self.add_messages(*self.model.format_observation_messages(message, outputs))

    def serialize(self, *extra_dicts):
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        data = {
            "info": {
                "model_stats": {
                    "api_calls": self.n_calls,
                },
                "config": {
                    "agent": asdict(self.config),
                    "agent_type": "hgm_miniswe.MiniSweStyleAgent",
                },
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            },
            "messages": self.messages,
            "trajectory_format": "hgm-miniswe-agent-1.0",
        }
        return recursive_merge(data, self.model.serialize(), self.env.serialize(), *extra_dicts)

    def save(self, path: Optional[str], *extra_dicts):
        data = self.serialize(*extra_dicts)
        if path:
            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            path_obj.write_text(json.dumps(_jsonable(data), indent=2), encoding="utf-8")
        return data


SYSTEM_TEMPLATE = """You are a SWE-bench repair agent running inside an HGM task container.

You interact with the repository by calling the bash tool.
Every response must include reasoning text and at least one bash tool call.

The working directory for every command is:
{{ repo_path }}

Use inspection commands, edit files with small targeted shell/Python snippets, and run relevant tests when feasible.
Do not overwrite whole files unless a small targeted edit is impractical.

Important boundaries:
- Modify regular source code files in the task repository only.
- Do not modify tests, reproduction scripts, helper scripts, generated files, pyproject.toml, setup.cfg, setup.py, tox.ini, noxfile.py, or other build/configuration files unless the issue explicitly requires that exact file.
- Do not include temporary files, test files, patch.txt, or setup/config changes in the final patch.

When the fix is complete, submit a git patch. Follow these steps in separate commands:
1. Create patch.txt with git diff, listing only the source files you intentionally modified:
   git diff -- path/to/file1 path/to/file2 > patch.txt
2. Inspect patch.txt and confirm it contains only intended source changes.
3. Submit with this exact command:
   echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt

Example final action:
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
"""


INSTANCE_TEMPLATE = """Fix the repository for this task.

<problem_statement>
{{ task }}
</problem_statement>

<test_description>
{{ test_description }}
</test_description>

<instance_id>
{{ instance_id }}
</instance_id>

Start by inspecting the repository and identifying the smallest source change likely to fix the issue.
Before submitting, create and inspect patch.txt so the submitted output contains the patch itself.
"""


class AgenticSystem:
    def __init__(
        self,
        problem_statement,
        git_tempdir,
        base_commit,
        chat_history_file="./chat_history.md",
        test_description=None,
        self_improve=False,
        instance_id=None,
        model=DEFAULT_MODEL,
    ):
        self.problem_statement = problem_statement
        self.git_tempdir = git_tempdir
        self.base_commit = base_commit
        self.chat_history_file = chat_history_file
        self.test_description = test_description or ""
        self.self_improve = self_improve
        self.instance_id = instance_id if not self_improve else "hgm"
        self.code_model = model
        Path(chat_history_file).parent.mkdir(parents=True, exist_ok=True)
        Path(chat_history_file).write_text("", encoding="utf-8")

    def forward(self, timeout=3600):
        out_path = str(Path(self.chat_history_file).with_suffix(".traj.json"))
        timeout_buffer = max(0, DEFAULT_TIMEOUT_BUFFER)
        wall_time_limit = max(0, int(timeout) - timeout_buffer)
        model_request_timeout = request_timeout_value()
        log_progress(
            "timeout_config "
            f"outer_timeout_seconds={int(timeout)} "
            f"buffer_seconds={timeout_buffer} "
            f"agent_wall_time_limit_seconds={wall_time_limit} "
            f"command_timeout_seconds={DEFAULT_COMMAND_TIMEOUT} "
            f"model_request_timeout={model_request_timeout if model_request_timeout is not None else 'openai_default'}"
        )
        if wall_time_limit <= 0:
            log_progress(
                "timeout_config_warning "
                "agent wall-time limit is zero; check HGM_MINISWE_TIMEOUT_BUFFER"
            )
        agent = MiniSweStyleAgent(
            model=MiniModel(self.code_model),
            env=LocalEnvironment(self.git_tempdir),
            config=MiniAgentConfig(
                system_template=SYSTEM_TEMPLATE,
                instance_template=INSTANCE_TEMPLATE,
                wall_time_limit_seconds=wall_time_limit,
                output_path=out_path,
            ),
        )
        info = agent.run(
            self.problem_statement,
            test_description=self.test_description,
            instance_id=self.instance_id or "",
        )
        write_markdown_log(self.chat_history_file, agent, info)
        return info


def write_markdown_log(path: str, agent: MiniSweStyleAgent, info: Dict[str, Any]):
    lines = [
        "# mini-SWE-style HGM trajectory",
        "",
        f"- exit_status: `{info.get('exit_status', '')}`",
        f"- submission_chars: `{len(info.get('submission', '') or '')}`",
        f"- api_calls: `{agent.n_calls}`",
        "",
    ]
    for idx, message in enumerate(agent.messages, start=1):
        role = message.get("role", "")
        lines.append(f"## {idx}. {role}")
        lines.append("")
        lines.append("```text")
        lines.append(str(message.get("content", "")))
        lines.append("```")
        extra = message.get("extra") or {}
        if extra.get("actions"):
            lines.append("")
            lines.append("Actions:")
            for action in extra["actions"]:
                lines.append("")
                lines.append("```bash")
                lines.append(str(action.get("command", "")))
                lines.append("```")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_model_patch(outdir: str, patch: str):
    os.makedirs(outdir or ".", exist_ok=True)
    model_patch_outfile = os.path.join(outdir or ".", "model_patch.diff")
    with open(model_patch_outfile, "w", encoding="utf-8") as f:
        f.write(patch)
    log_progress(f"model_patch_written path={model_patch_outfile} bytes={len(patch)}")


def patch_from_agent_info(info: Dict[str, Any], git_dir: str) -> str:
    if info.get("exit_status") == "Submitted":
        log_progress(
            "patch_source=submission "
            f"submission_chars={len(info.get('submission') or '')}"
        )
        return info.get("submission") or ""
    if os.getenv("HGM_MINISWE_ALLOW_FALLBACK_DIFF") == "1":
        log_progress("patch_source=fallback_diff base=HEAD")
        return diff_versus_commit(git_dir, "HEAD")
    log_progress(f"patch_source=empty exit_status={info.get('exit_status', '')}")
    return ""


def main():
    parser = argparse.ArgumentParser(description="Run HGM mini-SWE-style agent.")
    parser.add_argument("--problem_statement", required=True)
    parser.add_argument("--git_dir", required=True)
    parser.add_argument("--base_commit", required=True)
    parser.add_argument("--chat_history_file", required=True)
    parser.add_argument("--outdir", required=False, default="/hgm/")
    parser.add_argument("--test_description", default=None, required=False)
    parser.add_argument("--self_improve", default=False, action="store_true")
    parser.add_argument("--instance_id", default=None)
    parser.add_argument("--model", required=False, default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    outdir = args.outdir or "."
    try:
        agentic_system = AgenticSystem(
            problem_statement=args.problem_statement,
            git_tempdir=args.git_dir,
            base_commit=args.base_commit,
            chat_history_file=args.chat_history_file,
            test_description=args.test_description,
            self_improve=args.self_improve,
            instance_id=args.instance_id,
            model=args.model,
        )
        info = agentic_system.forward(args.timeout)
    except Exception:
        write_model_patch(outdir, "")
        raise
    write_model_patch(outdir, patch_from_agent_info(info, args.git_dir))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"mini-SWE-style agent failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise
