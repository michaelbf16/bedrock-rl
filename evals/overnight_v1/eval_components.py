"""Eval-local agent components; run_evals.py puts this dir on PYTHONPATH."""

import json

from bedrock_rl.adapters.chat_completions import ChatCompletionsModel
from bedrock_rl.adapters.netherite.computer import (TOOL_CALL_RE,
                                                    tool_call_attempts)


class TextEnvelopeModel(ChatCompletionsModel):
    """Promote Hermes-style ``<tool_call>`` text envelopes to structured calls.

    Some providers serve models whose tool calls arrive as text in message
    content rather than as structured ``tool_calls`` (seen with
    moonshotai/kimi-k2.6 via Chutes once its reasoning is disabled). The
    harness executes only structured calls, so such a turn would read as a
    voluntary stop. A response that already carries structured calls, or has
    no parseable envelope, passes through untouched, so this is a no-op for
    well-behaved providers and every model runs behind the same adapter.
    """

    async def complete(self, messages, tools, sampling):
        raw = await super().complete(messages, tools, sampling)
        try:
            message = raw["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return raw
        content = message.get("content")
        if message.get("tool_calls") or not isinstance(content, str):
            return raw
        calls = [
            {"id": f"textenv_{index}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(arguments)}}
            for index, (name, arguments, error)
            in enumerate(tool_call_attempts(content)) if error is None]
        if calls:
            message["tool_calls"] = calls
            message["content"] = TOOL_CALL_RE.sub("", content).strip() or None
        return raw
