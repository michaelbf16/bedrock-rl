"""Static and per-turn guidance as reversible model-view transforms."""

from __future__ import annotations

import inspect
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from bedrock_rl.core.components import object_reference, resolve_callable
from bedrock_rl.core.context import ContextTransform, ContextView
from bedrock_rl.core.messages import copy_messages, text_part


@dataclass(frozen=True)
class GuidanceRequest:
    turn: int
    task: Any = None
    state: Mapping[str, Any] | None = None
    name: str = "policy"
    metadata: Mapping[str, Any] = field(default_factory=dict)


class GuidanceProvider(Protocol):
    def static_hint(self, request: GuidanceRequest,
                    level: str) -> str | None: ...

    def dynamic_hint(self, request: GuidanceRequest,
                     level: str) -> str | None: ...


async def _hint(provider: Any, method: str, request: GuidanceRequest,
                level: str | None) -> str | None:
    if level in (None, "", "off", "none"):
        return None
    fn = getattr(provider, method, None)
    if fn is None:
        return None
    value = fn(request, level)
    if inspect.isawaitable(value):
        value = await value
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{object_reference(provider)}.{method} returned "
                        f"{type(value).__name__}; expected text or None")
    return value.strip() or None


class NoGuidance:
    async def apply(self, view: ContextView,
                    request: GuidanceRequest) -> ContextView:
        return view.copy()


class StaticGuidance:
    """A convenient provider for declarative static hint ladders."""

    def __init__(self, hints: Mapping[str, str] | None = None,
                 dynamic=None):
        self.hints = {str(k): str(v) for k, v in (hints or {}).items()}
        self.dynamic = (None if dynamic is None else
                        resolve_callable(dynamic,
                                         what="dynamic guidance provider"))

    def static_hint(self, request: GuidanceRequest, level: str):
        if level not in self.hints:
            raise ValueError(f"no static guidance level {level!r}; available: "
                             f"{sorted(self.hints) or 'none'}")
        return self.hints[level]

    def dynamic_hint(self, request: GuidanceRequest, level: str):
        if self.dynamic is None:
            return None
        return self.dynamic(request, level)


class GuidancePolicy:
    """Render privileged guidance into a view, never canonical history.

    Since hints are transformations of a saved canonical trajectory, SDFT
    and SDPO no longer remove strings heuristically.  They select the hinted
    teacher view or the unhinted student/deployment view of the same history.
    """

    def __init__(self, provider: GuidanceProvider, *,
                 static_level: str | None = None,
                 dynamic_level: str | None = None,
                 dynamic_separator: str = "\n\n"):
        self.provider = provider
        self.static_level = static_level
        self.dynamic_level = dynamic_level
        self.dynamic_separator = str(dynamic_separator)

    async def apply(self, view: ContextView,
                    request: GuidanceRequest) -> ContextView:
        messages = copy_messages(view.messages)
        transforms = list(view.transforms)
        static = await _hint(self.provider, "static_hint", request,
                             self.static_level)
        if static:
            messages.insert(0, {"role": "system", "content": static})
            transforms.append(ContextTransform(
                "static_guidance",
                {"provider": object_reference(self.provider),
                 "level": self.static_level, "message_index": 0}))
        dynamic = await _hint(self.provider, "dynamic_hint", request,
                              self.dynamic_level)
        if dynamic:
            candidates = [i for i, message in enumerate(messages)
                          if message["role"] != "assistant"]
            if not candidates:
                raise ValueError("dynamic guidance has no non-assistant "
                                 "message to annotate")
            index = candidates[-1]
            message = messages[index]
            block = self.dynamic_separator + dynamic
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = content + block
            elif isinstance(content, list):
                message["content"] = [*content, text_part(block)]
            elif content is None:
                message["content"] = block.lstrip("\n")
            else:                              # validated views never reach it
                raise TypeError("dynamic guidance target has invalid content")
            transforms.append(ContextTransform(
                "dynamic_guidance",
                {"provider": object_reference(self.provider),
                 "level": self.dynamic_level, "message_index": index,
                 "turn": request.turn}))
        return ContextView(messages, transforms)


def group_guidance(groups, probability: float, *, rng=None) -> list[bool]:
    """One hint decision per group, expanded back to rollout order."""
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("guidance probability must be in [0, 1]")
    rng = rng or random
    decisions = {}
    out = []
    for group in groups:
        if group not in decisions:
            decisions[group] = (probability >= 1.0 or
                                (probability > 0.0
                                 and rng.random() < probability))
        out.append(decisions[group])
    return out
