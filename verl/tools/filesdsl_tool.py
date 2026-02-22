# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import inspect
import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from typing import Any, Optional
from uuid import uuid4

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
_INPROCESS_EXECUTION_LOCK = threading.Lock()


class _FilesDSLTimeoutDetails(Exception):
    def __init__(
        self,
        *,
        message: str,
        elapsed_s: float | None = None,
        phase: str | None = None,
        partial_output: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.elapsed_s = elapsed_s
        self.phase = phase
        self.partial_output = partial_output


def _load_fdsl_runtime():
    from filesdsl import execute_fdsl

    try:
        from filesdsl.errors import DSLTimeoutError
    except Exception:  # pragma: no cover - compatibility fallback
        DSLTimeoutError = None

    try:
        supports_soft_timeout = "timeout_s" in inspect.signature(execute_fdsl).parameters
    except (TypeError, ValueError):  # pragma: no cover - very uncommon
        supports_soft_timeout = False
    return execute_fdsl, DSLTimeoutError, supports_soft_timeout


def _invoke_execute_fdsl(
    execute_fdsl,
    *,
    code: str,
    cwd: str | None,
    sandbox_root: str | None,
    timeout: float | None,
    supports_soft_timeout: bool,
) -> str:
    if timeout is not None and supports_soft_timeout:
        return execute_fdsl(code, cwd=cwd, sandbox_root=sandbox_root, timeout_s=timeout)
    return execute_fdsl(code, cwd=cwd, sandbox_root=sandbox_root)


def _serialize_timeout_exception(exc: Exception, dsl_timeout_error_cls) -> dict[str, Any] | None:
    if dsl_timeout_error_cls is None or not isinstance(exc, dsl_timeout_error_cls):
        return None

    elapsed_s = getattr(exc, "elapsed_s", None)
    phase = getattr(exc, "phase", None)
    partial_output = getattr(exc, "partial_output", None)
    return {
        "message": str(exc),
        "elapsed_s": elapsed_s if isinstance(elapsed_s, (int, float)) else None,
        "phase": phase if isinstance(phase, str) else None,
        "partial_output": partial_output if isinstance(partial_output, str) else None,
    }


def _execute_fdsl_worker(
    code: str,
    cwd: str | None,
    sandbox_root: str | None,
    timeout: float | None,
    result_queue: mp.Queue,
) -> None:
    execute_fdsl, dsl_timeout_error_cls, supports_soft_timeout = _load_fdsl_runtime()

    try:
        output = _invoke_execute_fdsl(
            execute_fdsl,
            code=code,
            cwd=cwd,
            sandbox_root=sandbox_root,
            timeout=timeout,
            supports_soft_timeout=supports_soft_timeout,
        )
    except Exception as exc:
        timeout_payload = _serialize_timeout_exception(exc, dsl_timeout_error_cls)
        if timeout_payload is not None:
            result_queue.put(("timeout", timeout_payload))
            return
        result_queue.put(("error", str(exc)))
        return

    result_queue.put(("ok", str(output)))


def _execute_fdsl_persistent_worker(task_queue: mp.Queue, result_queue: mp.Queue) -> None:
    execute_fdsl, dsl_timeout_error_cls, supports_soft_timeout = _load_fdsl_runtime()

    while True:
        task = task_queue.get()
        if task is None:
            return

        request_id, code, cwd, sandbox_root, timeout = task
        try:
            output = _invoke_execute_fdsl(
                execute_fdsl,
                code=code,
                cwd=cwd,
                sandbox_root=sandbox_root,
                timeout=timeout,
                supports_soft_timeout=supports_soft_timeout,
            )
        except Exception as exc:
            timeout_payload = _serialize_timeout_exception(exc, dsl_timeout_error_cls)
            if timeout_payload is not None:
                result_queue.put((request_id, "timeout", timeout_payload))
                continue
            result_queue.put((request_id, "error", str(exc)))
            continue
        result_queue.put((request_id, "ok", str(output)))


class FilesDSLTool(BaseTool):
    """Execute FilesDSL snippets using ``filesdsl.execute_fdsl``."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.default_cwd = config.get("cwd", ".")
        self.default_sandbox_root = config.get("sandbox_root", self.default_cwd)
        self.default_timeout = config.get("default_timeout", None)
        self.max_output_chars = int(config.get("max_output_chars", 4000))
        accepted_languages = config.get("accepted_languages", ["fdsl", "filesdsl", "python"]) or []
        self.accepted_languages = {str(lang).lower() for lang in accepted_languages}
        self.isolate_execution_process = bool(config.get("isolate_execution_process", True))
        prefer_inprocess_languages = config.get("prefer_inprocess_languages", []) or []
        self.prefer_inprocess_languages = {str(lang).lower() for lang in prefer_inprocess_languages}
        force_isolate_languages = config.get("force_isolate_languages", ["python"]) or []
        self.force_isolate_languages = {str(lang).lower() for lang in force_isolate_languages}
        self.use_persistent_isolated_worker = bool(config.get("use_persistent_isolated_worker", True))
        persistent_worker_languages = config.get("persistent_worker_languages", ["fdsl", "filesdsl"]) or []
        self.persistent_worker_languages = {str(lang).lower() for lang in persistent_worker_languages}
        self.subprocess_timeout_grace_seconds = self._normalize_non_negative_float(
            config.get("subprocess_timeout_grace_seconds", 5.0),
            default=5.0,
        )
        self.subprocess_start_method = str(config.get("subprocess_start_method", "spawn"))
        self._instance_dict: dict[str, dict[str, Any]] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", {})
        cwd = create_kwargs.get("cwd", self.default_cwd)
        sandbox_root = create_kwargs.get("sandbox_root", self.default_sandbox_root)
        self._instance_dict[instance_id] = self._build_instance_state(cwd=cwd, sandbox_root=sandbox_root)
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        code = parameters.get("code", "")
        if not isinstance(code, str):
            code = str(code)

        language = str(parameters.get("language", "fdsl")).lower()
        if language and language not in self.accepted_languages:
            return (
                ToolResponse(
                    text=(
                        f"Unsupported language '{language}'. "
                        f"Expected one of: {sorted(self.accepted_languages)}."
                    )
                ),
                None,
                {"status": "unsupported_language", "language": language},
            )

        timeout = self._normalize_timeout(parameters.get("timeout", self.default_timeout))
        instance_state = self._instance_dict.get(instance_id, {})
        cwd = parameters.get("cwd", instance_state.get("cwd", self.default_cwd))
        sandbox_root = parameters.get("sandbox_root", instance_state.get("sandbox_root", self.default_sandbox_root))

        try:
            execute_fdsl, dsl_timeout_error_cls, supports_soft_timeout = _load_fdsl_runtime()
        except ImportError:
            install_hint = "pip install git+https://github.com/Viagounet/FilesDSL.git"
            return (
                ToolResponse(text=f"filesdsl is not installed. Install it with: `{install_hint}`"),
                None,
                {"status": "missing_dependency"},
            )

        try:
            output = await self._execute_with_timeout(
                execute_fdsl=execute_fdsl,
                instance_id=instance_id,
                code=code,
                cwd=cwd,
                sandbox_root=sandbox_root,
                timeout=timeout,
                language=language,
                dsl_timeout_error_cls=dsl_timeout_error_cls,
                supports_soft_timeout=supports_soft_timeout,
            )
        except _FilesDSLTimeoutDetails as timeout_exc:
            timeout_text = timeout if timeout is not None else timeout_exc.elapsed_s
            timeout_message = f"FilesDSL execution timed out after {timeout_text} seconds."
            if timeout_exc.phase:
                timeout_message = f"{timeout_message} phase={timeout_exc.phase}"
            if timeout_exc.partial_output:
                timeout_message = f"{timeout_message}\nPartial output:\n{timeout_exc.partial_output}"
            timeout_message, truncated = self._truncate_text(timeout_message)
            return (
                ToolResponse(text=timeout_message),
                None,
                {
                    "status": "timeout",
                    "timeout": timeout,
                    "elapsed_s": timeout_exc.elapsed_s,
                    "phase": timeout_exc.phase,
                    "truncated": truncated,
                },
            )
        except TimeoutError:
            return (
                ToolResponse(
                    text=(
                        f"FilesDSL execution timed out after {timeout} seconds "
                        "(hard timeout guard)."
                    )
                ),
                None,
                {"status": "timeout", "timeout": timeout, "hard_timeout": True},
            )
        except Exception as exc:
            logger.warning(f"FilesDSL execution failed: {exc}")
            return (
                ToolResponse(text=f"FilesDSL execution failed: {exc}"),
                None,
                {"status": "error"},
            )

        output, truncated = self._truncate_text(output)

        return ToolResponse(text=output), None, {"status": "ok", "truncated": truncated}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        instance_state = self._instance_dict.pop(instance_id, None)
        self._shutdown_persistent_worker(instance_state)

    async def _execute_with_timeout(
        self,
        execute_fdsl,
        *,
        instance_id: str,
        code: str,
        cwd: str | None,
        sandbox_root: str | None,
        timeout: float | None,
        language: str,
        dsl_timeout_error_cls,
        supports_soft_timeout: bool,
    ) -> str:
        should_isolate = self._should_isolate_execution(language)
        if timeout is not None and not should_isolate and not supports_soft_timeout:
            logger.warning(
                "filesdsl.execute_fdsl does not expose timeout_s. "
                "Falling back to isolated execution so timeout remains enforceable."
            )
            should_isolate = True

        if should_isolate:
            should_use_persistent_worker = (
                self.use_persistent_isolated_worker and language in self.persistent_worker_languages
            )
            if should_use_persistent_worker:
                run_coro = asyncio.to_thread(
                    self._execute_in_persistent_subprocess,
                    instance_id,
                    code,
                    cwd,
                    sandbox_root,
                    timeout,
                )
            else:
                run_coro = asyncio.to_thread(self._execute_in_subprocess, code, cwd, sandbox_root, timeout)
            return await run_coro

        try:
            return await asyncio.to_thread(self._execute_inprocess, execute_fdsl, code, cwd, sandbox_root, timeout)
        except Exception as exc:
            timeout_payload = _serialize_timeout_exception(exc, dsl_timeout_error_cls)
            if timeout_payload is None:
                raise
            raise _FilesDSLTimeoutDetails(
                message=timeout_payload.get("message", "FilesDSL execution timed out."),
                elapsed_s=timeout_payload.get("elapsed_s"),
                phase=timeout_payload.get("phase"),
                partial_output=timeout_payload.get("partial_output"),
            ) from exc

    def _build_instance_state(self, cwd: str | None, sandbox_root: str | None) -> dict[str, Any]:
        return {
            "cwd": cwd,
            "sandbox_root": sandbox_root,
            "worker": None,
            "worker_lock": threading.Lock(),
        }

    def _execute_inprocess(
        self,
        execute_fdsl,
        code: str,
        cwd: str | None,
        sandbox_root: str | None,
        timeout: float | None,
    ) -> str:
        # filesdsl.execute_fdsl currently redirects process-wide stdout; serialize
        # in-process executions to avoid cross-thread stdio races.
        supports_soft_timeout = True
        try:
            supports_soft_timeout = "timeout_s" in inspect.signature(execute_fdsl).parameters
        except (TypeError, ValueError):  # pragma: no cover
            pass

        with _INPROCESS_EXECUTION_LOCK:
            return _invoke_execute_fdsl(
                execute_fdsl,
                code=code,
                cwd=cwd,
                sandbox_root=sandbox_root,
                timeout=timeout,
                supports_soft_timeout=supports_soft_timeout,
            )

    def _execute_in_persistent_subprocess(
        self,
        instance_id: str,
        code: str,
        cwd: str | None,
        sandbox_root: str | None,
        timeout: float | None,
    ) -> str:
        instance_state = self._instance_dict.get(instance_id)
        if instance_state is None:
            instance_state = self._build_instance_state(cwd=cwd, sandbox_root=sandbox_root)
            self._instance_dict[instance_id] = instance_state

        with instance_state["worker_lock"]:
            worker = self._ensure_persistent_worker(instance_state)
            request_id = str(uuid4())
            worker["task_queue"].put((request_id, code, cwd, sandbox_root, timeout))

            hard_timeout = self._get_hard_timeout(timeout)
            deadline = None if hard_timeout is None else time.monotonic() + hard_timeout
            while True:
                wait_timeout = 0.1
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._shutdown_persistent_worker(instance_state)
                        raise TimeoutError
                    wait_timeout = min(wait_timeout, remaining)

                try:
                    response_request_id, status, payload = worker["result_queue"].get(timeout=wait_timeout)
                except queue.Empty:
                    if not worker["process"].is_alive():
                        exitcode = worker["process"].exitcode
                        self._shutdown_persistent_worker(instance_state)
                        raise RuntimeError(f"FilesDSL worker exited with code {exitcode}")
                    continue

                if response_request_id != request_id:
                    # Ignore stale responses from previous requests.
                    continue

                if status == "ok":
                    return payload
                if status == "timeout":
                    payload_dict = payload if isinstance(payload, dict) else {}
                    raise _FilesDSLTimeoutDetails(
                        message=payload_dict.get("message", "FilesDSL execution timed out."),
                        elapsed_s=payload_dict.get("elapsed_s"),
                        phase=payload_dict.get("phase"),
                        partial_output=payload_dict.get("partial_output"),
                    )
                raise RuntimeError(payload)

    def _ensure_persistent_worker(self, instance_state: dict[str, Any]) -> dict[str, Any]:
        worker = instance_state.get("worker")
        if worker is not None and worker["process"].is_alive():
            return worker

        self._shutdown_persistent_worker(instance_state)

        ctx = mp.get_context(self.subprocess_start_method)
        task_queue = ctx.Queue(maxsize=1)
        result_queue = ctx.Queue(maxsize=1)
        process = ctx.Process(target=_execute_fdsl_persistent_worker, args=(task_queue, result_queue))
        process.start()

        worker = {
            "task_queue": task_queue,
            "result_queue": result_queue,
            "process": process,
        }
        instance_state["worker"] = worker
        return worker

    def _shutdown_persistent_worker(self, instance_state: dict[str, Any] | None) -> None:
        if not instance_state:
            return

        worker = instance_state.get("worker")
        if worker is None:
            return

        process = worker.get("process")
        task_queue = worker.get("task_queue")
        result_queue = worker.get("result_queue")

        try:
            if process is not None and process.is_alive():
                try:
                    task_queue.put_nowait(None)
                except Exception:
                    pass
                process.join(timeout=0.2)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1)
        finally:
            try:
                if task_queue is not None:
                    task_queue.close()
            except Exception:
                pass
            try:
                if result_queue is not None:
                    result_queue.close()
            except Exception:
                pass
            instance_state["worker"] = None

    def _execute_in_subprocess(
        self,
        code: str,
        cwd: str | None,
        sandbox_root: str | None,
        timeout: float | None,
    ) -> str:
        ctx = mp.get_context(self.subprocess_start_method)
        result_queue = ctx.Queue(maxsize=1)
        process = ctx.Process(target=_execute_fdsl_worker, args=(code, cwd, sandbox_root, timeout, result_queue))
        process.start()
        hard_timeout = self._get_hard_timeout(timeout)
        try:
            # Read result first (with timeout) to avoid queue/pipe backpressure
            # deadlocks when large outputs are emitted from the subprocess.
            # Joining before draining the queue can block until timeout.
            if hard_timeout is None:
                status, payload = result_queue.get()
            else:
                status, payload = result_queue.get(timeout=hard_timeout)
        except queue.Empty:
            if process.is_alive():
                process.terminate()
                process.join()
                raise TimeoutError
            if process.exitcode == 0:
                return ""
            raise RuntimeError(f"FilesDSL worker exited with code {process.exitcode}")
        finally:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join()

        if status == "ok":
            return payload
        if status == "timeout":
            payload_dict = payload if isinstance(payload, dict) else {}
            raise _FilesDSLTimeoutDetails(
                message=payload_dict.get("message", "FilesDSL execution timed out."),
                elapsed_s=payload_dict.get("elapsed_s"),
                phase=payload_dict.get("phase"),
                partial_output=payload_dict.get("partial_output"),
            )
        raise RuntimeError(payload)

    def _should_isolate_execution(self, language: str) -> bool:
        normalized_language = str(language).lower()
        if normalized_language in self.force_isolate_languages:
            return True
        if normalized_language in self.prefer_inprocess_languages:
            return False
        return self.isolate_execution_process

    def __del__(self):
        for instance_state in list(self._instance_dict.values()):
            self._shutdown_persistent_worker(instance_state)

    def _truncate_text(self, text: str) -> tuple[str, bool]:
        text = text if isinstance(text, str) else str(text)
        if self.max_output_chars > 0 and len(text) > self.max_output_chars:
            return text[: self.max_output_chars] + "...(truncated)", True
        return text, False

    def _get_hard_timeout(self, soft_timeout: float | None) -> float | None:
        if soft_timeout is None:
            return None
        return soft_timeout + self.subprocess_timeout_grace_seconds

    @staticmethod
    def _normalize_non_negative_float(value: Any, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if parsed < 0:
            return default
        return parsed

    def _normalize_timeout(self, timeout: Any) -> float | None:
        if timeout is None or timeout == "":
            return None
        try:
            timeout = float(timeout)
        except (TypeError, ValueError):
            logger.warning(f"Invalid timeout value: {timeout}; ignoring it.")
            return None
        if timeout <= 0:
            return None
        return timeout
