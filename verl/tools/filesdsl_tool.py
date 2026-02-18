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
import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FilesDSLTool(BaseTool):
    """Execute FilesDSL snippets using ``filesdsl.execute_fdsl``."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.default_cwd = config.get("cwd", ".")
        self.default_sandbox_root = config.get("sandbox_root", self.default_cwd)
        self.default_timeout = config.get("default_timeout", None)
        self.max_output_chars = int(config.get("max_output_chars", 4000))
        accepted_languages = config.get("accepted_languages", ["fdsl", "filesdsl", "python"])
        self.accepted_languages = {str(lang).lower() for lang in accepted_languages}
        self._instance_dict: dict[str, dict[str, Any]] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", {})
        cwd = create_kwargs.get("cwd", self.default_cwd)
        sandbox_root = create_kwargs.get("sandbox_root", self.default_sandbox_root)
        self._instance_dict[instance_id] = {
            "cwd": cwd,
            "sandbox_root": sandbox_root,
        }
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
            from filesdsl import execute_fdsl
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
                code=code,
                cwd=cwd,
                sandbox_root=sandbox_root,
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return (
                ToolResponse(text=f"FilesDSL execution timed out after {timeout} seconds."),
                None,
                {"status": "timeout", "timeout": timeout},
            )
        except Exception as exc:
            logger.warning(f"FilesDSL execution failed: {exc}")
            return (
                ToolResponse(text=f"FilesDSL execution failed: {exc}"),
                None,
                {"status": "error"},
            )

        truncated = False
        if self.max_output_chars > 0 and len(output) > self.max_output_chars:
            output = output[: self.max_output_chars] + "...(truncated)"
            truncated = True

        return ToolResponse(text=output), None, {"status": "ok", "truncated": truncated}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)

    async def _execute_with_timeout(
        self,
        execute_fdsl,
        *,
        code: str,
        cwd: str | None,
        sandbox_root: str | None,
        timeout: float | None,
    ) -> str:
        run_coro = asyncio.to_thread(
            execute_fdsl,
            code,
            cwd=cwd,
            sandbox_root=sandbox_root,
        )
        if timeout is None:
            return await run_coro
        return await asyncio.wait_for(run_coro, timeout=timeout)

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
