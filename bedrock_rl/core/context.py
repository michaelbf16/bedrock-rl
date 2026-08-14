"""Composable, model-independent views over an immutable chat history."""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol

from bedrock_rl.core.components import object_reference, resolve_callable
from bedrock_rl.core.messages import (copy_messages, image_locations,
                                        is_image_part, text_part,
                                        validate_messages)


@dataclass(frozen=True)
class ContextTransform:
    """Auditable description of one change made to a model view."""

    type: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "details": copy.deepcopy(self.details)}


@dataclass
class ContextView:
    """The exact messages a model saw plus how they were derived."""

    messages: list[dict]
    transforms: list[ContextTransform] = field(default_factory=list)

    @classmethod
    def from_messages(cls, messages: Iterable[Mapping[str, Any]]):
        return cls(validate_messages(messages))

    def copy(self):
        return ContextView(copy_messages(self.messages), list(self.transforms))


@dataclass(frozen=True)
class ContextRequest:
    turn: int
    task: Any = None
    state: Mapping[str, Any] | None = None
    name: str = "policy"
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ContextPolicy(Protocol):
    async def apply(self, view: ContextView,
                    request: ContextRequest) -> ContextView: ...


async def _apply(policy: Any, view: ContextView,
                 request: ContextRequest) -> ContextView:
    result = policy.apply(view, request)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, ContextView):
        raise TypeError(f"context policy {object_reference(policy)} returned "
                        f"{type(result).__name__}, expected ContextView")
    validate_messages(result.messages)
    return result


class Identity:
    async def apply(self, view: ContextView,
                    request: ContextRequest) -> ContextView:
        return view.copy()


class Compose:
    """Apply policies in declaration order."""

    def __init__(self, policies: Iterable[ContextPolicy] = ()):
        self.policies = tuple(policies)

    async def apply(self, view: ContextView,
                    request: ContextRequest) -> ContextView:
        out = view.copy()
        for policy in self.policies:
            out = await _apply(policy, out, request)
        return out


class KeepLastImages:
    """Remove old typed image parts while retaining canonical history.

    This policy deliberately does not inspect model token strings such as
    Qwen's ``vision_start`` token and does not mutate the trajectory.  It
    operates on OpenAI multimodal content before a model adapter renders or
    tokenizes it, so the same policy works for every model family.
    """

    def __init__(self, count: int, placeholder: str | None = None):
        count = int(count)
        if count < 0:
            raise ValueError("KeepLastImages.count must be at least zero")
        self.count = count
        self.placeholder = placeholder

    async def apply(self, view: ContextView,
                    request: ContextRequest) -> ContextView:
        selected = keep_last_images(view.messages, self.count,
                                    placeholder=self.placeholder)
        selected.transforms = [*view.transforms, *selected.transforms]
        return selected


def keep_last_images(messages, count: int, *, placeholder: str | None = None):
    """Synchronous typed-message primitive behind ``KeepLastImages``.

    Adapters that render synchronously can use this function without owning a
    second image-window algorithm. It returns a ``ContextView`` and never
    mutates canonical history.
    """
    count = int(count)
    if count < 0:
        raise ValueError("image count must be at least zero")
    copied = copy_messages(messages)
    locations = image_locations(copied)
    cut = max(0, len(locations) - count)
    removing = set(locations[:cut])
    if not removing:
        return ContextView(copied)
    removed = []
    for message_index, message in enumerate(copied):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        kept = []
        for part_index, part in enumerate(content):
            if ((message_index, part_index) in removing
                    and is_image_part(part)):
                removed.append([message_index, part_index])
                if placeholder is not None:
                    kept.append(text_part(placeholder))
            else:
                kept.append(part)
        message["content"] = kept
    transform = ContextTransform(
        "keep_last_images",
        {"count": count, "removed": removed, "placeholder": placeholder})
    return ContextView(copied, [transform])


def _leading_system_end(messages: list[dict]) -> int:
    index = 0
    while index < len(messages) and messages[index]["role"] in (
            "system", "developer"):
        index += 1
    return index


def _task_prefix_end(messages: list[dict]) -> int:
    """End of the immutable rules + opening task instruction prefix."""
    index = _leading_system_end(messages)
    if index < len(messages) and messages[index]["role"] == "user":
        index += 1
    return index


def _recent_turn_start(messages: list[dict], count: int) -> int:
    """Start of the last N assistant/tool groups without orphaning tools."""
    assistants = [i for i, message in enumerate(messages)
                  if message["role"] == "assistant"]
    if count >= len(assistants):
        return _leading_system_end(messages)
    if count == 0:
        if not assistants:
            return _leading_system_end(messages)
        tail = messages[assistants[-1] + 1:]
        # A tool result cannot stand without the assistant tool call it
        # answers, so retain that pair even at a zero-history setting.
        return (assistants[-1] if any(message["role"] == "tool"
                                      for message in tail)
                else assistants[-1] + 1)
    return assistants[-count]


class KeepLastTurns:
    """Retain rules, the opening task instruction, and recent turns."""

    def __init__(self, count: int):
        count = int(count)
        if count < 0:
            raise ValueError("KeepLastTurns.count must be at least zero")
        self.count = count

    async def apply(self, view: ContextView,
                    request: ContextRequest) -> ContextView:
        messages = copy_messages(view.messages)
        head = _task_prefix_end(messages)
        start = _recent_turn_start(messages, self.count)
        if start <= head:
            return ContextView(messages, list(view.transforms))
        out = messages[:head] + messages[start:]
        transform = ContextTransform(
            "keep_last_turns",
            {"count": self.count, "removed_range": [head, start]})
        return ContextView(out, [*view.transforms, transform])


class SummarizeOldTurns:
    """Replace old turns with a summary produced by an injected callable.

    The summarizer receives a deep copy of the messages being replaced and
    the ``ContextRequest``.  It may be synchronous or asynchronous and must
    return text.  Model choice, prompting, cache policy, and privacy therefore
    remain application decisions rather than framework hardcoding.
    """

    def __init__(self, summarizer, keep_last: int = 2, *,
                 role: str = "system",
                 prefix: str = "Summary of earlier context:\n"):
        if role not in ("system", "developer", "user"):
            raise ValueError("summary role must be system, developer, or user")
        self.summarizer = resolve_callable(summarizer,
                                           what="context summarizer")
        self.keep_last = int(keep_last)
        if self.keep_last < 0:
            raise ValueError("keep_last must be at least zero")
        self.role = role
        self.prefix = str(prefix)

    async def apply(self, view: ContextView,
                    request: ContextRequest) -> ContextView:
        messages = copy_messages(view.messages)
        head = _task_prefix_end(messages)
        start = _recent_turn_start(messages, self.keep_last)
        if start <= head:
            return ContextView(messages, list(view.transforms))
        old = copy_messages(messages[head:start])
        summary = self.summarizer(old, request)
        if inspect.isawaitable(summary):
            summary = await summary
        if not isinstance(summary, str) or not summary.strip():
            raise TypeError("context summarizer must return non-empty text")
        summary_message = {"role": self.role,
                           "content": self.prefix + summary.strip()}
        out = messages[:head] + [summary_message] + messages[start:]
        transform = ContextTransform(
            "summarize_old_turns",
            {"keep_last": self.keep_last, "removed_range": [head, start],
             "summarizer": object_reference(self.summarizer),
             "summary_role": self.role})
        return ContextView(out, [*view.transforms, transform])


async def apply_context(policy: ContextPolicy | None,
                        messages: Iterable[Mapping[str, Any]],
                        request: ContextRequest) -> ContextView:
    view = ContextView.from_messages(messages)
    return view if policy is None else await _apply(policy, view, request)
