# This file is adapted from https://github.com/jennyzzt/dgm.
#
# Code adapted from https://github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/llm.py.
import json
import os
import re
import traceback
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

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

MAX_OUTPUT_TOKENS = 4096
AVAILABLE_LLMS = [
    "gpt-5",
    "o4-mini",
    "o3",
    "deepseek/deepseek-chat-v3.1",
    "anthropic/claude-sonnet-4",
]
VLLM_ENV_VARS = ("VLLM_BASE_URL", "VLLM_API_KEY", "VLLM_MODEL")


def _existing_openai_retry_exceptions():
    exceptions = []
    for attr in ("RateLimitError", "APITimeoutError"):
        exc = getattr(openai, attr, None)
        if isinstance(exc, type) and issubclass(exc, BaseException):
            exceptions.append(exc)
    return exceptions


def provider_retry_exceptions():
    exceptions = _existing_openai_retry_exceptions()
    if anthropic is not None:
        exceptions.extend([anthropic.RateLimitError, anthropic.APIStatusError])
    if not exceptions:
        exceptions = [Exception]
    return tuple(exceptions)


class _CompatNamespace:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _to_namespace(value):
    if isinstance(value, dict):
        return _CompatNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def _sanitize_text_for_json(value):
    if isinstance(value, str):
        return value.encode("utf-8", "replace").decode("utf-8")
    if isinstance(value, dict):
        return {k: _sanitize_text_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_text_for_json(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_text_for_json(v) for v in value)
    return value


class LegacyOpenAICompatClient:
    def __init__(self, base_url=None, api_key=None):
        self.base_url = base_url.rstrip("/") if base_url else "https://api.openai.com/v1"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.chat = _CompatNamespace(completions=_CompatNamespace(create=self._chat_completions_create))
        self.models = _CompatNamespace(list=self._models_list)

    def _headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request_json(self, method, path, payload=None):
        url = f"{self.base_url}{path}"
        payload = _sanitize_text_for_json(payload)
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=self._headers(), method=method)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8", "replace")
            except Exception:
                pass
            raise RuntimeError(
                f"HTTP {exc.code} {exc.msg} from {url}; body={error_body}"
            ) from exc
        return json.loads(body)

    def _chat_completions_create(self, **kwargs):
        return _to_namespace(self._request_json("POST", "/chat/completions", kwargs))

    def _models_list(self):
        return _to_namespace(self._request_json("GET", "/models"))


@dataclass
class ProviderConfig:
    provider: str
    alias: str
    resolved_model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model_source: str = "alias"
    requested_model: Optional[str] = None


def is_openai_model(model: str) -> bool:
    return "gpt" in model.lower() or model.startswith("o")


def is_vllm_model(model: str) -> bool:
    return model.lower().startswith("vllm")


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def get_provider_env() -> Dict[str, Optional[str]]:
    env = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OpenRouter_API_KEY": os.getenv("OpenRouter_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    }
    for key in VLLM_ENV_VARS:
        env[key] = os.getenv(key)
    return env


def _extract_explicit_vllm_model(alias: str) -> Optional[str]:
    if alias.lower().startswith("vllm:"):
        explicit = alias.split(":", 1)[1].strip()
        return explicit or None
    return None


def _legacy_vllm_base_url(alias: str) -> Optional[str]:
    if os.getenv("VLLM_BASE_URL"):
        return None
    match = re.match(r"^vllm-[^-]+-(?P<host>.+)$", alias, re.IGNORECASE)
    if not match:
        return None
    host = match.group("host").strip()
    if "." not in host and ":" not in host and host != "localhost":
        return None
    if not re.match(r"^[A-Za-z0-9_.:-]+$", host):
        return None
    return normalize_base_url(f"http://{host}:8000/v1")


def _resolve_vllm_model(
    client: Any, alias: str, explicit_model: Optional[str]
) -> Tuple[str, str]:
    if explicit_model:
        return explicit_model, "alias_prefix"

    env_model = os.getenv("VLLM_MODEL")
    if env_model:
        return env_model, "env"

    try:
        models = client.models.list()
        if getattr(models, "data", None):
            first_model = models.data[0].id
            if first_model:
                return first_model, "models.list"
    except Exception as exc:
        print(f"Warning: failed to list models from vLLM endpoint: {exc}")
        print(traceback.format_exc())

    return alias, "alias_fallback"


def get_client_provider_config(
    client: Any, default_model: Optional[str] = None
) -> ProviderConfig:
    config = getattr(client, "_hgm_provider_config", None)
    if config is None:
        return ProviderConfig(
            provider="unknown",
            alias=default_model or "",
            resolved_model=default_model or "",
        )
    if isinstance(config, dict):
        return ProviderConfig(**config)
    return config


def _attach_provider_config(client: Any, config: ProviderConfig) -> None:
    client._hgm_provider_config = asdict(config)


def create_client(model: str):
    if is_openai_model(model):
        if hasattr(openai, "OpenAI"):
            client = openai.OpenAI()
        else:
            client = LegacyOpenAICompatClient()
        config = ProviderConfig(
            provider="openai",
            alias=model,
            resolved_model=model,
            requested_model=model,
        )
        _attach_provider_config(client, config)
        print(f"Using OpenAI API with model {model}.")
        return client, model

    if is_vllm_model(model):
        explicit_model = _extract_explicit_vllm_model(model)
        base_url = os.getenv("VLLM_BASE_URL") or _legacy_vllm_base_url(model)
        if not base_url:
            raise ValueError(
                "vLLM mode selected but VLLM_BASE_URL is not set. "
                "Set VLLM_BASE_URL or use a legacy alias like vllm-qwenS-10.0.0.1."
            )
        base_url = normalize_base_url(base_url)
        api_key = os.getenv("VLLM_API_KEY") or "dummy"
        if hasattr(openai, "OpenAI"):
            client = openai.OpenAI(base_url=base_url, api_key=api_key)
        else:
            client = LegacyOpenAICompatClient(base_url=base_url, api_key=api_key)
        resolved_model, model_source = _resolve_vllm_model(
            client, model, explicit_model
        )
        config = ProviderConfig(
            provider="vllm",
            alias=model,
            resolved_model=resolved_model,
            base_url=base_url,
            api_key=api_key,
            model_source=model_source,
            requested_model=explicit_model or os.getenv("VLLM_MODEL") or model,
        )
        _attach_provider_config(client, config)
        print(
            "Using vLLM API with alias "
            f"{model}, base_url={base_url}, resolved_model={resolved_model} "
            f"(source={model_source})."
        )
        return client, resolved_model

    if hasattr(openai, "OpenAI"):
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OpenRouter_API_KEY"),
        )
    else:
        client = LegacyOpenAICompatClient(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OpenRouter_API_KEY"),
        )
    config = ProviderConfig(
        provider="openrouter",
        alias=model,
        resolved_model=model,
        base_url="https://openrouter.ai/api/v1",
        requested_model=model,
    )
    _attach_provider_config(client, config)
    print(f"Using OpenRouter API with model {model}.")
    return client, model


@backoff.on_exception(
    backoff.expo,
    provider_retry_exceptions(),
    max_time=120,
)
def get_json_response_from_llm(
    msg,
    client,
    model,
    system_message,
):
    new_msg_history = [{"role": "user", "content": msg}]
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            *new_msg_history,
        ],
        n=1,
        stop=None,
        seed=0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    content_json = json.loads(content)
    new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]

    return content_json, new_msg_history


def get_response_from_llm(
    msg,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
):
    if msg_history is None:
        msg_history = []

    if model.startswith("o"):
        new_msg_history = msg_history + [
            {"role": "user", "content": system_message + msg}
        ]
        response = client.chat.completions.create(
            model=model,
            messages=new_msg_history,
            temperature=1,
            n=1,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif "gpt" in model.lower():
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            n=1,
            stop=None,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    else:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_OUTPUT_TOKENS,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        print(f'User: {new_msg_history[-2]["content"]}')
        print(f'Assistant: {new_msg_history[-1]["content"]}')
        print("*" * 21 + " LLM END " + "*" * 21)
        print()
    return content, new_msg_history


def extract_json_between_markers(llm_output):
    inside_json_block = False
    json_lines = []

    for line in llm_output.split("\n"):
        striped_line = line.strip()

        if striped_line.startswith("```json"):
            inside_json_block = True
            continue

        if inside_json_block and striped_line.startswith("```"):
            inside_json_block = False
            break

        if inside_json_block:
            json_lines.append(line)

    if not json_lines:
        fallback_pattern = r"\{.*?\}"
        matches = re.findall(fallback_pattern, llm_output, re.DOTALL)
        for candidate in matches:
            candidate = candidate.strip()
            if candidate:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    candidate_clean = re.sub(r"[\x00-\x1F\x7F]", "", candidate)
                    try:
                        return json.loads(candidate_clean)
                    except json.JSONDecodeError:
                        continue
        return None

    json_string = "\n".join(json_lines).strip()

    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
        try:
            return json.loads(json_string_clean)
        except json.JSONDecodeError:
            return None
