import asyncio
import sys
import types

from verl.tools.filesdsl_tool import FilesDSLTool
from verl.tools.schemas import OpenAIFunctionSchema, OpenAIFunctionToolSchema


def _build_schema() -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(name="filesdsl_execute", description="Execute FilesDSL"),
    )


def test_execute_runs_in_isolated_process_for_python_by_default(monkeypatch):
    fake_module = types.SimpleNamespace(execute_fdsl=lambda *args, **kwargs: "direct")
    monkeypatch.setitem(sys.modules, "filesdsl", fake_module)

    tool = FilesDSLTool(config={}, tool_schema=_build_schema())
    instance_id, _ = asyncio.run(tool.create())

    called = {"value": False}

    def _fake_subprocess(code, cwd, sandbox_root, timeout):
        called["value"] = True
        assert code == "print(1)"
        return "ok-from-subprocess"

    monkeypatch.setattr(tool, "_execute_in_subprocess", _fake_subprocess)

    response, _, meta = asyncio.run(tool.execute(instance_id, {"code": "print(1)", "language": "python"}))

    assert called["value"] is True
    assert response.text == "ok-from-subprocess"
    assert meta["status"] == "ok"


def test_execute_reports_timeout_for_isolated_execution(monkeypatch):
    fake_module = types.SimpleNamespace(execute_fdsl=lambda *args, **kwargs: "direct")
    monkeypatch.setitem(sys.modules, "filesdsl", fake_module)

    tool = FilesDSLTool(config={}, tool_schema=_build_schema())
    instance_id, _ = asyncio.run(tool.create())

    def _fake_subprocess(code, cwd, sandbox_root, timeout):
        raise TimeoutError

    monkeypatch.setattr(tool, "_execute_in_subprocess", _fake_subprocess)

    response, _, meta = asyncio.run(tool.execute(instance_id, {"code": "print(1)", "language": "python", "timeout": 1}))

    assert "timed out" in response.text
    assert meta["status"] == "timeout"


def test_fdsl_uses_persistent_isolated_worker_by_default(monkeypatch):
    fake_module = types.SimpleNamespace(execute_fdsl=lambda *args, **kwargs: "direct")
    monkeypatch.setitem(sys.modules, "filesdsl", fake_module)

    tool = FilesDSLTool(config={}, tool_schema=_build_schema())
    instance_id, _ = asyncio.run(tool.create())
    called = {"value": False}

    def _fake_persistent_subprocess(instance_id, code, cwd, sandbox_root, timeout):
        called["value"] = True
        return "fdsl-from-persistent"

    def _fake_subprocess(code, cwd, sandbox_root, timeout):
        raise AssertionError("fdsl should not use one-shot subprocess when persistent worker is enabled")

    monkeypatch.setattr(tool, "_execute_in_persistent_subprocess", _fake_persistent_subprocess)
    monkeypatch.setattr(tool, "_execute_in_subprocess", _fake_subprocess)

    response, _, meta = asyncio.run(tool.execute(instance_id, {"code": "print(1)", "language": "fdsl"}))

    assert called["value"] is True
    assert response.text == "fdsl-from-persistent"
    assert meta["status"] == "ok"


def test_fdsl_can_be_forced_inprocess(monkeypatch):
    fake_module = types.SimpleNamespace(execute_fdsl=lambda *args, **kwargs: "direct")
    monkeypatch.setitem(sys.modules, "filesdsl", fake_module)

    tool = FilesDSLTool(config={"prefer_inprocess_languages": ["fdsl"]}, tool_schema=_build_schema())
    instance_id, _ = asyncio.run(tool.create())

    def _fake_persistent_subprocess(instance_id, code, cwd, sandbox_root, timeout):
        raise AssertionError("fdsl should run in-process when explicitly configured")

    def _fake_subprocess(code, cwd, sandbox_root, timeout):
        raise AssertionError("fdsl should run in-process when explicitly configured")

    monkeypatch.setattr(tool, "_execute_in_persistent_subprocess", _fake_persistent_subprocess)
    monkeypatch.setattr(tool, "_execute_in_subprocess", _fake_subprocess)

    response, _, meta = asyncio.run(tool.execute(instance_id, {"code": "print(1)", "language": "fdsl"}))

    assert response.text == "direct"
    assert meta["status"] == "ok"


def test_fdsl_can_disable_persistent_worker_and_use_one_shot_subprocess(monkeypatch):
    fake_module = types.SimpleNamespace(execute_fdsl=lambda *args, **kwargs: "direct")
    monkeypatch.setitem(sys.modules, "filesdsl", fake_module)

    tool = FilesDSLTool(config={"use_persistent_isolated_worker": False}, tool_schema=_build_schema())
    instance_id, _ = asyncio.run(tool.create())
    called = {"value": False}

    def _fake_subprocess(code, cwd, sandbox_root, timeout):
        called["value"] = True
        return "fdsl-in-subprocess"

    monkeypatch.setattr(tool, "_execute_in_subprocess", _fake_subprocess)

    response, _, meta = asyncio.run(tool.execute(instance_id, {"code": "print(1)", "language": "fdsl"}))

    assert called["value"] is True
    assert response.text == "fdsl-in-subprocess"
    assert meta["status"] == "ok"
