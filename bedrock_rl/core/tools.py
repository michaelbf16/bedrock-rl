"""Trainer-independent tool calls, results, registries, and function tools."""

from __future__ import annotations

import copy
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol

from bedrock_rl.core.components import resolve_callable
from bedrock_rl.core.messages import validate_message
from bedrock_rl.core.state import json_value


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    @classmethod
    def from_openai(cls, value: Mapping[str, Any]):
        if not isinstance(value, Mapping):
            raise TypeError("tool call must be a mapping")
        fn = value.get("function")
        if not isinstance(fn, Mapping) or not fn.get("name"):
            raise ValueError("tool call needs function.name")
        call_id = value.get("id")
        if not call_id:
            raise ValueError("tool call needs an id")
        arguments = fn.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError as exc:
                raise ValueError(f"tool call {call_id!r} has invalid JSON "
                                 f"arguments: {exc}") from exc
        if not isinstance(arguments, Mapping):
            raise TypeError("tool arguments must be a JSON object")
        return cls(str(call_id), str(fn["name"]),
                   json_value(dict(arguments), what="tool arguments"))

    def to_openai(self) -> dict[str, Any]:
        return {"id": self.id, "type": "function",
                "function": {"name": self.name,
                             "arguments": json.dumps(
                                 self.arguments, separators=(",", ":"))}}


@dataclass
class ToolResult:
    content: str | list[dict] = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    done: bool = False
    reward: float | None = None
    # Runtime-only objects for an adapter (PIL frames, file handles, tensors).
    # They never enter the OpenAI message or the serialized trajectory.
    attachments: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if not isinstance(self.content, (str, list)):
            raise TypeError("tool result content must be text or content parts")
        self.content = copy.deepcopy(self.content)
        self.metadata = json_value(self.metadata, what="tool result metadata")
        self.attachments = dict(self.attachments)
        if self.reward is not None:
            self.reward = float(self.reward)

    def message(self, call_id: str, *, name: str | None = None) -> dict:
        messages = self.messages(call_id, name=name)
        if len(messages) != 1:
            raise ValueError(
                "visual tool results expand to a text tool message followed "
                "by a user image message; use ToolResult.messages()")
        return messages[0]

    def messages(self, call_id: str, *, name: str | None = None) -> list[dict]:
        """Return Chat-Completions-valid result and observation messages.

        Chat Completions permits only text in a ``tool`` message. A visual
        result therefore keeps its matching text result under ``role=tool``
        and carries exact ``image_url`` parts in the immediately following
        ``role=user`` observation. This retains normal tool-call linkage and
        remains directly acceptable to multimodal OpenAI-compatible clients.
        """
        content = copy.deepcopy(self.content)
        images = []
        if isinstance(content, list):
            text = []
            for part in content:
                if isinstance(part, Mapping) and part.get("type") == "image_url":
                    images.append(part)
                else:
                    text.append(part)
            content = text or ""
        message = {"role": "tool", "tool_call_id": str(call_id),
                   "content": content}
        if name is not None:
            message["name"] = str(name)
        validate_message(message)
        messages = [message]
        if images:
            observation = {"role": "user", "content": images}
            validate_message(observation)
            messages.append(observation)
        return messages


@dataclass
class ToolContext:
    task: Any
    session: Any
    trajectory: Any
    turn: int
    state: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class Tool(Protocol):
    name: str
    schema: Mapping[str, Any]

    def execute(self, call: ToolCall, context: ToolContext) -> ToolResult: ...


class FunctionTool:
    """Adapt a sync or async ``(arguments, context)`` callable."""

    def __init__(self, name: str, description: str, parameters: Mapping,
                 function):
        self.name = str(name)
        self.function = resolve_callable(function, what=f"tool {name} function")
        self.schema = {
            "type": "function",
            "function": {"name": self.name, "description": str(description),
                         "parameters": copy.deepcopy(dict(parameters))},
        }

    def execute(self, call: ToolCall, context: ToolContext):
        return self.function(copy.deepcopy(call.arguments), context)


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()):
        self._tools = {}
        for tool in tools:
            name = getattr(tool, "name", None)
            if not name:
                schema = getattr(tool, "schema", {})
                name = (schema.get("function") or {}).get("name")
            if not name:
                raise ValueError("tool has neither name nor schema function name")
            if name in self._tools:
                raise ValueError(f"duplicate tool name {name!r}")
            self._tools[str(name)] = tool
        fallbacks = [
            (name, tool) for name, tool in self._tools.items()
            if bool(getattr(tool, "malformed_fallback", False))
        ]
        if len(fallbacks) > 1:
            raise ValueError(
                "tool registry has multiple malformed fallbacks: "
                + ", ".join(name for name, _tool in fallbacks))
        self._malformed_fallback = fallbacks[0] if fallbacks else None
        if (self._malformed_fallback is not None
                and not callable(getattr(
                    self._malformed_fallback[1], "malformed", None))):
            raise TypeError(
                f"malformed fallback tool "
                f"{self._malformed_fallback[0]!r} exposes no handler")

    def __contains__(self, name: str):
        return name in self._tools

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    @property
    def schemas(self) -> list[dict]:
        return [copy.deepcopy(dict(tool.schema))
                for tool in self._tools.values()]

    @property
    def malformed_fallback_name(self) -> str | None:
        return (None if self._malformed_fallback is None else
                self._malformed_fallback[0])

    @staticmethod
    def _result(name: str, value: Any) -> ToolResult:
        if isinstance(value, ToolResult):
            return value
        if isinstance(value, (str, list)):
            return ToolResult(value)
        if isinstance(value, Mapping):
            return ToolResult(**dict(value))
        raise TypeError(f"tool {name!r} returned {type(value).__name__}; "
                        "expected ToolResult, text, content parts, or mapping")

    async def execute(self, call: ToolCall,
                      context: ToolContext) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            raise KeyError(f"model called unknown tool {call.name!r}; available: "
                           f"{sorted(self._tools) or 'none'}")
        value = tool.execute(call, context)
        if inspect.isawaitable(value):
            value = await value
        return self._result(call.name, value)

    async def malformed(self, call: ToolCall, error: Exception | str,
                        context: ToolContext) -> ToolResult | None:
        """Let a registered tool account for its malformed call.

        Most tools have no environment transition to make and return ``None``
        here, which preserves the harness's generic error result. Stateful
        tools may expose ``malformed(error, context)`` to spend the same turn,
        observation, and penalty their other runtimes do.
        """
        tool = self._tools.get(call.name)
        handler = getattr(tool, "malformed", None)
        if not callable(handler):
            return None
        value = handler(error, context)
        if inspect.isawaitable(value):
            value = await value
        return self._result(call.name, value)

    async def account_malformed(
            self, call: ToolCall, error: Exception | str,
            context: ToolContext) -> tuple[ToolResult | None, str | None]:
        """Account for malformed/unknown calls through one stateful tool.

        An exact-name tool owns its malformed inputs.  An unknown name has no
        such owner, so a registry may nominate one tool with
        ``malformed_fallback = True`` to spend the environment turn.  This
        keeps the core harness generic while letting a stateful environment
        make invented tool names comparable with its normal malformed calls.
        """
        result = await self.malformed(call, error, context)
        if result is not None:
            return result, call.name
        # A known extension owns its own malformed-input semantics. Routing a
        # recipe lookup or unrelated API bug through the Minecraft computer
        # fallback would mutate the game for a call that never targeted it.
        if call.name in self._tools:
            return None, None
        if self._malformed_fallback is None:
            return None, None
        name, tool = self._malformed_fallback
        handler = getattr(tool, "malformed", None)
        value = handler(error, context)
        if inspect.isawaitable(value):
            value = await value
        return self._result(name, value), name
