import re
import sys
import types


def _make_module(name, attrs=None, package=False):
    module = types.ModuleType(name)
    if package:
        module.__path__ = []
    for key, value in (attrs or {}).items():
        setattr(module, key, value)
    return module


def _import_hgm_utils_with_stubs():
    utils_pkg = _make_module("utils", package=True)
    prompts_pkg = _make_module("prompts", package=True)
    swe_bench_pkg = _make_module("swe_bench", package=True)

    stubs = {
        "numpy": _make_module(
            "numpy",
            {
                "inf": float("inf"),
                "isfinite": lambda value: True,
                "sum": lambda values: 0,
                "argmax": lambda values: 0,
                "random": types.SimpleNamespace(beta=lambda *args, **kwargs: [1.0]),
            },
        ),
        "docker": _make_module("docker", {"from_env": lambda: None}),
        "self_improve_step": _make_module(
            "self_improve_step",
            {
                "dataset": None,
                "diagnose_problem": lambda *args, **kwargs: None,
                "save_metadata": lambda *args, **kwargs: None,
            },
        ),
        "prompts": prompts_pkg,
        "prompts.self_improvement_prompt": _make_module(
            "prompts.self_improvement_prompt",
            {
                "find_selfimprove_eval_logs": lambda *args, **kwargs: ([], None, None, None),
            },
        ),
        "prompts.testrepo_prompt": _make_module(
            "prompts.testrepo_prompt",
            {"get_test_description": lambda *args, **kwargs: "test-description"},
        ),
        "swe_bench": swe_bench_pkg,
        "swe_bench.harness": _make_module(
            "swe_bench.harness", {"harness": lambda *args, **kwargs: []}
        ),
        "swe_bench.report": _make_module(
            "swe_bench.report", {"make_report": lambda *args, **kwargs: None}
        ),
        "utils": utils_pkg,
        "utils.common_utils": _make_module(
            "utils.common_utils", {"load_json_file": lambda *args, **kwargs: {}}
        ),
        "utils.docker_utils": _make_module(
            "utils.docker_utils",
            {
                "build_hgm_container": lambda *args, **kwargs: None,
                "cleanup_container": lambda *args, **kwargs: None,
                "copy_from_container": lambda *args, **kwargs: None,
                "copy_to_container": lambda *args, **kwargs: None,
                "ensure_psql_client": lambda *args, **kwargs: None,
                "log_container_output": lambda *args, **kwargs: None,
                "remove_existing_container": lambda *args, **kwargs: None,
                "safe_log": lambda *args, **kwargs: None,
                "setup_logger": lambda *args, **kwargs: None,
            },
        ),
        "utils.eval_utils": _make_module(
            "utils.eval_utils", {"get_acc_on_tasks": lambda *args, **kwargs: []}
        ),
        "utils.evo_utils": _make_module(
            "utils.evo_utils",
            {
                "get_all_performance": lambda *args, **kwargs: (None, {}),
                "get_model_patch_paths": lambda *args, **kwargs: [],
                "is_compiled_self_improve": lambda *args, **kwargs: False,
            },
        ),
    }

    utils_pkg.common_utils = stubs["utils.common_utils"]
    utils_pkg.docker_utils = stubs["utils.docker_utils"]
    utils_pkg.eval_utils = stubs["utils.eval_utils"]
    utils_pkg.evo_utils = stubs["utils.evo_utils"]
    prompts_pkg.self_improvement_prompt = stubs["prompts.self_improvement_prompt"]
    prompts_pkg.testrepo_prompt = stubs["prompts.testrepo_prompt"]
    swe_bench_pkg.harness = stubs["swe_bench.harness"]
    swe_bench_pkg.report = stubs["swe_bench.report"]

    originals = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        import hgm_utils as module
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return module


hgm_utils = _import_hgm_utils_with_stubs()


class FakeExecResult:
    def __init__(self, exit_code, output):
        self.exit_code = exit_code
        self.output = output


class FakeContainer:
    def __init__(self, available_paths):
        self.available_paths = set(available_paths)
        self.calls = []

    def exec_run(self, command, workdir="/"):
        self.calls.append((command, workdir))
        match = re.search(r"command -v ([^']+)", command)
        if not match:
            return FakeExecResult(1, b"")
        candidate = match.group(1)
        if candidate in self.available_paths:
            return FakeExecResult(0, (candidate + "\n").encode("utf-8"))
        return FakeExecResult(1, b"")


def test_resolve_container_agent_python_prefers_usr_local_python():
    container = FakeContainer({"/usr/local/bin/python"})

    resolved = hgm_utils.resolve_container_agent_python(container)

    assert resolved == "/usr/local/bin/python"
    assert len(container.calls) >= 2
    assert "/opt/miniconda3/envs/testbed/bin/python" in container.calls[0][0]
    assert "/usr/local/bin/python" in container.calls[1][0]


def test_resolve_container_agent_python_raises_when_missing():
    container = FakeContainer(set())

    try:
        hgm_utils.resolve_container_agent_python(container)
    except RuntimeError as exc:
        message = str(exc)
        assert "No Python executable found in container" in message
        assert "/opt/miniconda3/envs/testbed/bin/python" in message
        assert "/usr/local/bin/python" in message
    else:
        raise AssertionError("Expected RuntimeError when no Python executable is available")
