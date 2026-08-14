"""The computer-use action language as a trainer-independent core tool."""

from __future__ import annotations

import asyncio

from bedrock_rl.adapters.netherite.chat import tool_schema
from bedrock_rl.adapters.netherite.computer import (
    NEXT_COMPUTER_MESSAGE, TOOL_NAME, cu_payload,
)
from bedrock_rl.core.messages import data_image_part, text_part
from bedrock_rl.core.tools import ToolCall, ToolContext, ToolResult
from bedrock_rl.data import png_bytes
from bedrock_rl.env.episode import tool_response_text


def malformed_computer_step(episode, error, *, count_turn=True):
    """Charge and render one malformed call identically in every adapter."""
    observation, frame, step_reward, done = episode.step(
        "{", count_turn=count_turn)
    text = tool_response_text(observation, done, NEXT_COMPUTER_MESSAGE)
    return (f"Malformed tool call: {error}\n{text}", frame,
            step_reward, done)


class NetheriteComputerTool:
    name = TOOL_NAME
    schema = tool_schema()[0]
    next_message = NEXT_COMPUTER_MESSAGE
    # Unknown tool names still attempted to act in this one stateful episode.
    # The generic registry uses this opt-in to charge them through the same
    # malformed transition without knowing anything about Minecraft.
    malformed_fallback = True

    async def execute(self, call: ToolCall, context: ToolContext):
        episode = context.session.episode
        payload = cu_payload(call.arguments)
        count_turn = bool(context.metadata.get("count_turn", True))
        observation, frame, step_reward, done = await asyncio.to_thread(
            episode.step, payload, count_turn=count_turn)
        text = tool_response_text(observation, done, self.next_message)
        content = []
        encoded = png_bytes(frame)
        if encoded is not None:
            content.append(data_image_part(encoded))
        content.append(text_part(text))
        return ToolResult(
            content,
            {"turns": episode.turns, "success": episode.success,
             "step_reward": float(step_reward)},
            done=bool(done), reward=float(step_reward),
            attachments={"image": frame})

    async def malformed(self, error: Exception | str,
                        context: ToolContext) -> ToolResult:
        """Charge a malformed model call without terminating its worker."""
        episode = context.session.episode
        count_turn = bool(context.metadata.get("count_turn", True))
        message, frame, step_reward, done = await asyncio.to_thread(
            malformed_computer_step, episode, error, count_turn=count_turn)
        content = []
        encoded = png_bytes(frame)
        if encoded is not None:
            content.append(data_image_part(encoded))
        content.append(text_part(message))
        return ToolResult(
            content,
            {"turns": episode.turns, "success": episode.success,
             "step_reward": float(step_reward), "malformed": str(error)},
            done=bool(done), reward=float(step_reward),
            attachments={"image": frame})
