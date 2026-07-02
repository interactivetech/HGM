import importlib.util
import pathlib
import sys

import pytest

import hgm
import llm


class FakeModelsAPI:
    def __init__(self, model_ids):
        self._model_ids = model_ids

    def list(self):
        return type(
            "ModelList",
            (),
            {"data": [type("Model", (), {"id": model_id})() for model_id in self._model_ids]},
        )()


class FakeChatCompletionsAPI:
    def create(self, **kwargs):
        if kwargs.get("tools"):
            tool_call = type(
                "ToolCall",
                (),
                {
                    "function": type(
                        "Fn", (), {"arguments": '{"value":"HGM_TOOL_OK"}', "name": "echo_value"}
                    )(),
                    "id": "tool_1",
                },
            )()
            message = type(
                "Message", (), {"content": "", "tool_calls": [tool_call]}
            )()
        elif kwargs.get("response_format"):
            message = type(
                "Message", (), {"content": '{"status":"ok","value":7}', "tool_calls": None}
            )()
        else:
            message = type(
                "Message", (), {"content": "HGM_VLLM_OK", "tool_calls": None}
            )()
        return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()


class FakeClient:
    def __init__(self, base_url=None, api_key=None, model_ids=None):
        self.base_url = base_url
        self.api_key = api_key
        self.models = FakeModelsAPI(model_ids or ["listed-model"])
        self.chat = type("Chat", (), {"completions": FakeChatCompletionsAPI()})()


def load_check_vllm_module():
    script_path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "check_vllm_endpoint.py"
    spec = importlib.util.spec_from_file_location("check_vllm_endpoint", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_create_client_resolves_vllm_from_env(monkeypatch):
    created = {}

    def fake_openai_client(**kwargs):
        created.update(kwargs)
        return FakeClient(**kwargs)

    monkeypatch.setattr(llm.openai, "OpenAI", fake_openai_client)
    monkeypatch.setenv("VLLM_BASE_URL", "http://example.test:8000/v1")
    monkeypatch.setenv("VLLM_API_KEY", "dummy")
    monkeypatch.setenv("VLLM_MODEL", "Qwen/FromEnv")

    client, resolved_model = llm.create_client("vllm-qwen")

    assert created["base_url"] == "http://example.test:8000/v1"
    assert created["api_key"] == "dummy"
    assert resolved_model == "Qwen/FromEnv"
    provider = llm.get_client_provider_config(client, resolved_model)
    assert provider.provider == "vllm"
    assert provider.base_url == "http://example.test:8000/v1"


def test_create_client_prefers_explicit_vllm_model(monkeypatch):
    monkeypatch.setattr(
        llm.openai, "OpenAI", lambda **kwargs: FakeClient(**kwargs, model_ids=["listed-model"])
    )
    monkeypatch.setenv("VLLM_BASE_URL", "http://example.test:8000/v1")
    monkeypatch.delenv("VLLM_MODEL", raising=False)

    _client, resolved_model = llm.create_client("vllm:Qwen/Explicit")

    assert resolved_model == "Qwen/Explicit"


def test_apply_vllm_env_overrides_and_no_openai_key_required(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(llm.openai, "OpenAI", lambda **kwargs: FakeClient(**kwargs))
    cfg = type(
        "Cfg",
        (),
        {
            "vllm_base_url": "http://example.test:8000/v1",
            "vllm_api_key": "dummy",
            "vllm_model": "Qwen/EnvModel",
        },
    )()

    env = hgm.apply_vllm_env_overrides(cfg)
    _client, resolved_model = llm.create_client("vllm-qwen")

    assert env["VLLM_BASE_URL"] == "http://example.test:8000/v1"
    assert resolved_model == "Qwen/EnvModel"


def test_check_vllm_endpoint_skip_tool_test(monkeypatch, capsys):
    module = load_check_vllm_module()

    monkeypatch.setattr(module, "create_client", lambda alias: (FakeClient(), "listed-model"))
    monkeypatch.setattr(
        module,
        "get_client_provider_config",
        lambda client, model: type(
            "Provider", (), {"provider": "vllm", "base_url": "http://example.test:8000/v1", "model_source": "env"}
        )(),
    )

    monkeypatch.setattr(sys, "argv", ["check_vllm_endpoint.py", "--skip-tool-test"])
    module.main()
    output = capsys.readouterr().out
    assert "[PASS] /models" in output
    assert "[PASS] chat completion" in output
    assert "[PASS] JSON output" in output
