import asyncio
import sys
import types

from verl.tools import filesdsl_tool
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


def test_execute_reports_soft_timeout_details_inprocess(monkeypatch):
    class _FakeDSLTimeoutError(Exception):
        def __init__(self, message):
            super().__init__(message)
            self.elapsed_s = 1.234
            self.phase = "file.search"
            self.partial_output = "before-timeout"

    def _raise_timeout(code, cwd=None, sandbox_root=None, timeout_s=None):
        raise _FakeDSLTimeoutError("Timed out")

    fake_module = types.SimpleNamespace(execute_fdsl=_raise_timeout)
    fake_errors_module = types.SimpleNamespace(DSLTimeoutError=_FakeDSLTimeoutError)
    monkeypatch.setitem(sys.modules, "filesdsl", fake_module)
    monkeypatch.setitem(sys.modules, "filesdsl.errors", fake_errors_module)

    tool = FilesDSLTool(config={"prefer_inprocess_languages": ["fdsl"]}, tool_schema=_build_schema())
    instance_id, _ = asyncio.run(tool.create())

    response, _, meta = asyncio.run(tool.execute(instance_id, {"code": "print(1)", "language": "fdsl", "timeout": 2}))

    assert "timed out" in response.text.lower()
    assert "Partial output" in response.text
    assert meta["status"] == "timeout"
    assert meta["phase"] == "file.search"
    assert meta["elapsed_s"] == 1.234


def test_execute_reports_timeout_from_structured_subprocess_error(monkeypatch):
    fake_module = types.SimpleNamespace(execute_fdsl=lambda *args, **kwargs: "direct")
    monkeypatch.setitem(sys.modules, "filesdsl", fake_module)

    tool = FilesDSLTool(config={"use_persistent_isolated_worker": False}, tool_schema=_build_schema())
    instance_id, _ = asyncio.run(tool.create())

    def _fake_subprocess(code, cwd, sandbox_root, timeout):
        raise filesdsl_tool._FilesDSLTimeoutDetails(
            message="Timed out",
            elapsed_s=3.5,
            phase="directory.search.file",
            partial_output="partial",
        )

    monkeypatch.setattr(tool, "_execute_in_subprocess", _fake_subprocess)

    response, _, meta = asyncio.run(tool.execute(instance_id, {"code": "print(1)", "language": "python", "timeout": 2}))

    assert "timed out" in response.text.lower()
    assert meta["status"] == "timeout"
    assert meta["phase"] == "directory.search.file"
    assert meta["elapsed_s"] == 3.5
