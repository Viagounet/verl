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

import logging
import os
import re
from typing import Any
from uuid import uuid4

from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    AsyncLLMServerManager,
    DictConfigWrap,
    register,
)
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_CODE_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL)
_CODE_RE_OPEN_ENDED = re.compile(r"<code>(.*)$", re.DOTALL)


@register("code_interrupt_agent")
class CodeInterruptAgentLoop(AgentLoopBase):
    """Agent loop that interrupts generation on <code>...</code>, executes code, injects <result>...</result>,
    then resumes generation."""

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        rollout_cfg = trainer_config.config.actor_rollout_ref.rollout
        mt_cfg = rollout_cfg.multi_turn

        self.prompt_length = rollout_cfg.prompt_length
        self.response_length = rollout_cfg.response_length

        code_interrupt_cfg = mt_cfg.get("code_interrupt", {})
        self.max_interrupts = code_interrupt_cfg.get("max_interrupts", 8)
        self.language = code_interrupt_cfg.get("language", "python")
        self.timeout = code_interrupt_cfg.get("timeout", None)
        self.result_max_length = code_interrupt_cfg.get("result_max_length", mt_cfg.max_tool_response_length)
        self.stop_sequences = code_interrupt_cfg.get("stop_sequences", ["</code>"])
        self.include_stop_str_in_output = code_interrupt_cfg.get("include_stop_str_in_output", True)
        # Some SGLang versions do not support include_stop_str_in_output in SamplingParams.
        self._include_stop_param_supported = True
        self._stop_on_code_close = "</code>" in self.stop_sequences

        tool_config_path = mt_cfg.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        if not tool_list:
            raise ValueError("code_interrupt_agent requires at least one tool in rollout.multi_turn.tool_config_path")

        target_tool_name = code_interrupt_cfg.get("tool_name")
        if target_tool_name is None:
            self.code_tool = tool_list[0]
        else:
            tool_map = {tool.name: tool for tool in tool_list}
            if target_tool_name not in tool_map:
                raise ValueError(
                    f"code_interrupt_agent cannot find tool '{target_tool_name}'. Available: {list(tool_map.keys())}"
                )
            self.code_tool = tool_map[target_tool_name]

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        prompt_ids = await self.apply_chat_template(messages)

        metrics = {}
        request_id = uuid4().hex

        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        tool_rewards: list[float] = []
        turn_scores: list[float] = []
        assistant_turns = 0
        interrupts = 0

        while len(response_mask) < self.response_length:
            interrupt_params = dict(sampling_params)
            interrupt_params.setdefault("stop", self.stop_sequences)
            if self.include_stop_str_in_output and self._include_stop_param_supported:
                interrupt_params.setdefault("include_stop_str_in_output", True)

            with simple_timer("generate_sequences", metrics):
                try:
                    output = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=prompt_ids,
                        sampling_params=interrupt_params,
                        image_data=None,
                        video_data=None,
                    )
                except Exception as e:
                    if self._is_include_stop_param_unsupported(e, interrupt_params):
                        self._include_stop_param_supported = False
                        logger.warning(
                            "SamplingParams does not support include_stop_str_in_output; "
                            "retrying without it for code_interrupt_agent."
                        )
                        interrupt_params.pop("include_stop_str_in_output", None)
                        output = await self.server_manager.generate(
                            request_id=request_id,
                            prompt_ids=prompt_ids,
                            sampling_params=interrupt_params,
                            image_data=None,
                            video_data=None,
                        )
                    else:
                        raise

            if metrics.get("num_preempted") is None:
                metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
            else:
                metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

            assistant_turns += 1
            generated_ids = output.token_ids
            if not generated_ids:
                break

            prompt_ids += generated_ids
            response_ids += generated_ids
            response_mask += [1] * len(generated_ids)
            if output.log_probs:
                response_logprobs += output.log_probs

            generated_text = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(generated_ids, skip_special_tokens=True),
            )
            match = _CODE_RE.search(generated_text)
            # Fallback when stop string is stripped from output by backend.
            if (
                match is None
                and self._stop_on_code_close
                and "<code>" in generated_text
                and not (self.include_stop_str_in_output and self._include_stop_param_supported)
            ):
                match = _CODE_RE_OPEN_ENDED.search(generated_text)
            if match is None or interrupts >= self.max_interrupts:
                break

            code = match.group(1).strip()
            result, tool_reward = await self._execute_code(code=code, tools_kwargs=kwargs.get("tools_kwargs", {}))
            if tool_reward is not None:
                tool_rewards.append(tool_reward)

            if len(result) > self.result_max_length:
                result = result[: self.result_max_length] + "...(truncated)"
            result_text = f"<result>{result}</result>"
            result_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.encode(result_text, add_special_tokens=False),
            )

            if len(response_mask) + len(result_ids) > self.response_length:
                break

            prompt_ids += result_ids
            response_ids += result_ids
            response_mask += [0] * len(result_ids)
            if response_logprobs:
                response_logprobs += [0.0] * len(result_ids)

            interrupts += 1

        output = AgentLoopOutput(
            prompt_ids=prompt_ids[: len(prompt_ids) - len(response_ids)],
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            num_turns=assistant_turns + 1,
            metrics=metrics,
            extra_fields={"turn_scores": turn_scores, "tool_rewards": tool_rewards},
        )
        return output

    async def _execute_code(self, code: str, tools_kwargs: dict[str, Any]) -> tuple[str, float | None]:
        tool = self.code_tool
        kwargs = tools_kwargs.get(tool.name, {})
        instance_id = None
        try:
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            parameters = {"code": code, "language": self.language}
            if self.timeout is not None:
                parameters["timeout"] = self.timeout
            tool_resp, tool_reward, _ = await tool.execute(instance_id, parameters, **kwargs)
            if tool_reward is not None:
                logger.debug(f"code tool reward: {tool_reward}")
            if isinstance(tool_resp, ToolResponse):
                return tool_resp.text or "", tool_reward
            return str(tool_resp), tool_reward
        except Exception as e:
            logger.warning(f"Error executing code interrupt tool: {e}")
            return f"Error when executing code: {e}", 0.0
        finally:
            if instance_id is not None:
                await tool.release(instance_id)

    @staticmethod
    def _is_include_stop_param_unsupported(error: Exception, sampling_params: dict[str, Any]) -> bool:
        if "include_stop_str_in_output" not in sampling_params:
            return False
        message = str(error).lower()
        if "include_stop_str_in_output" not in message:
            return False
        unsupported_signatures = (
            "unexpected keyword argument",
            "unexpected keywork argument",
            "unexpected key",
        )
        return any(sig in message for sig in unsupported_signatures)
