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


def test_execute_runs_in_isolated_process_by_default(monkeypatch):
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

    response, _, meta = asyncio.run(tool.execute(instance_id, {"code": "print(1)", "language": "fdsl"}))

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

    response, _, meta = asyncio.run(tool.execute(instance_id, {"code": "print(1)", "timeout": 1}))

    assert "timed out" in response.text
    assert meta["status"] == "timeout"


def test_fdsl_language_forces_isolation_even_when_global_isolation_disabled(monkeypatch):
    fake_module = types.SimpleNamespace(execute_fdsl=lambda *args, **kwargs: "direct")
    monkeypatch.setitem(sys.modules, "filesdsl", fake_module)

    tool = FilesDSLTool(config={"isolate_execution_process": False}, tool_schema=_build_schema())
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


def test_subprocess_execution_reads_queue_before_join(monkeypatch):
    fake_module = types.SimpleNamespace(execute_fdsl=lambda *args, **kwargs: "direct")
    monkeypatch.setitem(sys.modules, "filesdsl", fake_module)

    tool = FilesDSLTool(config={}, tool_schema=_build_schema())

    class _FakeProcess:
        def __init__(self, target=None, args=None):
            self._alive = False
            self.exitcode = 0

        def start(self):
            return None

        def join(self, timeout=None):
            return None

        def is_alive(self):
            return self._alive

        def terminate(self):
            self._alive = False

    class _FakeQueue:
        def __init__(self, maxsize=1):
            self.get_called = False

        def get(self, timeout=None):
            self.get_called = True
            return ("ok", "value")

    class _FakeCtx:
        queue = None

        def Queue(self, maxsize=1):
            self.queue = _FakeQueue(maxsize=maxsize)
            return self.queue

        def Process(self, target=None, args=None):
            return _FakeProcess(target=target, args=args)

    fake_ctx = _FakeCtx()
    monkeypatch.setattr("verl.tools.filesdsl_tool.mp.get_context", lambda method: fake_ctx)

    output = tool._execute_in_subprocess("print(1)", ".", ".", timeout=5)

    assert output == "value"
    assert fake_ctx.queue.get_called is True
