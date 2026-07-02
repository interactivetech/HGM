import sys
import types


def _import_llm_withtools_with_stubs():
    stubs = {
        "openai": types.SimpleNamespace(
            RateLimitError=Exception, APITimeoutError=Exception, OpenAI=object
        ),
        "llm": types.SimpleNamespace(
            create_client=lambda *args, **kwargs: None,
            get_client_provider_config=lambda *args, **kwargs: None,
            is_vllm_model=lambda *args, **kwargs: False,
        ),
        "tools": types.SimpleNamespace(load_all_tools=lambda logging=print: []),
    }
    originals = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        import llm_withtools as module
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return module


lwt = _import_llm_withtools_with_stubs()


def test_inspection_tool_calls_are_detected():
    assert lwt._is_inspection_tool_call("bash", {"command": "git status"})
    assert lwt._is_inspection_tool_call(
        "bash", {"command": "git diff sphinx/builders/gettext.py"}
    )
    assert lwt._is_inspection_tool_call(
        "editor", {"command": "view", "path": "/tmp/example.py"}
    )
    assert not lwt._is_inspection_tool_call(
        "bash", {"command": "python -c 'print(1)'"}
    )
    assert not lwt._is_inspection_tool_call(
        "editor", {"command": "edit", "path": "/tmp/example.py"}
    )


def test_worktree_changes_are_detected_from_tool_results():
    assert lwt._tool_result_shows_worktree_changes(
        "bash",
        {"command": "git status"},
        "modified: sphinx/builders/gettext.py",
    )
    assert lwt._tool_result_shows_worktree_changes(
        "bash",
        {"command": "git diff"},
        "diff --git a/sphinx/builders/gettext.py b/sphinx/builders/gettext.py",
    )
    assert lwt._tool_result_shows_worktree_changes(
        "editor",
        {"command": "edit", "path": "/tmp/example.py"},
        "File at /tmp/example.py has been overwritten with new content.",
    )
    assert not lwt._tool_result_shows_worktree_changes(
        "bash",
        {"command": "git status"},
        "nothing to commit, working tree clean",
    )


def test_stall_guard_stops_after_repeated_inspection():
    saw_worktree, streak, should_stop = lwt._update_stall_guard_state(
        "bash",
        {"command": "git diff"},
        "diff --git a/file.py b/file.py",
        {"reasoning": "keep going"},
        True,
        3,
        4,
    )

    assert saw_worktree is True
    assert streak == 4
    assert should_stop is True


def test_stall_guard_stops_when_completion_signal_is_present():
    saw_worktree, streak, should_stop = lwt._update_stall_guard_state(
        "bash",
        {"command": "git status"},
        "modified: file.py",
        {"reasoning": "The fix is complete."},
        True,
        0,
        4,
    )

    assert saw_worktree is True
    assert streak == 1
    assert should_stop is True


def test_stall_guard_resets_on_non_inspection_command():
    saw_worktree, streak, should_stop = lwt._update_stall_guard_state(
        "bash",
        {"command": "python -c 'print(1)'"},
        "1",
        {"reasoning": "keep going"},
        True,
        3,
        4,
    )

    assert saw_worktree is True
    assert streak == 0
    assert should_stop is False
