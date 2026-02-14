# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

import pytest
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import DictConfigWrap
from verl.experimental.agent_loop.code_interrupt_agent_loop import CodeInterruptAgentLoop
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.dataset.rl_dataset import RLHFDataset


class _FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict]] = None,
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        **kwargs,
    ) -> list[int]:
        del messages, tools, add_generation_prompt, tokenize, kwargs
        return self.encode("PROMPT:", add_special_tokens=False)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr(x) for x in ids)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(c) for c in text]


@dataclass
class _FakeTokenOutput:
    token_ids: list[int]
    log_probs: Optional[list[float]] = None
    routed_experts: Any = None
    num_preempted: Optional[int] = None


class _FakeServerManager:
    def __init__(self, tokenizer: _FakeTokenizer):
        self.tokenizer = tokenizer
        self.turn = 0

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> _FakeTokenOutput:
        del request_id, prompt_ids, image_data, video_data
        self.turn += 1
        assert sampling_params["stop"] == ["</code>"]
        assert sampling_params["include_stop_str_in_output"] is True
        if self.turn == 1:
            text = "<code>print(1)</code>"
        else:
            text = " final-answer"
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return _FakeTokenOutput(token_ids=token_ids, log_probs=[0.0] * len(token_ids), num_preempted=0)


class FakeCodeExecutorTool(BaseTool):
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "code_executor",
                    "description": "Execute python code",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"},
                            "language": {"type": "string"},
                        },
                        "required": ["code"],
                    },
                },
            }
        )

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        del instance_id, kwargs
        return ToolResponse(text=f"trace:{parameters['code']}"), 0.5, {"ok": True}


@pytest.mark.asyncio
async def test_code_interrupt_agent_loop_executes_and_injects_result(tmp_path):
    tool_config = {
        "tools": [
            {
                "class_name": "tests.experimental.agent_loop.test_code_interrupt_agent_loop_on_cpu.FakeCodeExecutorTool",
                "config": {"type": "native"},
            }
        ]
    }
    tool_config_path = tmp_path / "tool_config.json"
    with open(tool_config_path, "w") as f:
        json.dump(tool_config, f)

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": 64,
                    "response_length": 256,
                    "multi_turn": {
                        "max_tool_response_length": 128,
                        "tool_config_path": str(tool_config_path),
                        "code_interrupt": {
                            "tool_name": "code_executor",
                            "max_interrupts": 3,
                            "stop_sequences": ["</code>"],
                        },
                    },
                }
            },
            "data": {
                "tool_config_path": None,
                "apply_chat_template_kwargs": {},
            },
        }
    )

    tokenizer = _FakeTokenizer()
    loop = CodeInterruptAgentLoop(
        trainer_config=DictConfigWrap(config),
        server_manager=_FakeServerManager(tokenizer),
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=RLHFDataset,
        dataset_config=DictConfigWrap(config.data),
    )

    output = await loop.run(sampling_params={}, raw_prompt=[{"role": "user", "content": "compute"}])
    output_text = tokenizer.decode(output.response_ids)

    assert "<code>print(1)</code>" in output_text
    assert "<result>trace:print(1)</result>" in output_text
    assert output_text.endswith(" final-answer")

    code_end = output_text.find("</code>") + len("</code>")
    result_end = output_text.find("</result>") + len("</result>")
    assert all(v == 1 for v in output.response_mask[:code_end])
    assert all(v == 0 for v in output.response_mask[code_end:result_end])
    assert any(v == 1 for v in output.response_mask[result_end:])
