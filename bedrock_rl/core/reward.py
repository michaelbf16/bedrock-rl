"""Composable reward/verifier contracts over trajectories and state traces."""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, TYPE_CHECKING

from bedrock_rl.core.components import object_reference, resolve_callable
from bedrock_rl.core.state import StateTrace, json_value

if TYPE_CHECKING:                              # avoid a runtime import cycle
    from bedrock_rl.core.trajectory import Trajectory


@dataclass
class RewardResult:
    total: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float | int | bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.total = float(self.total)
        self.components = {str(k): float(v)
                           for k, v in self.components.items()}
        self.metrics = {str(k): v for k, v in self.metrics.items()}
        self.details = json_value(self.details, what="reward details")

    def to_dict(self) -> dict[str, Any]:
        return {"total": self.total,
                "components": dict(self.components),
                "metrics": copy.deepcopy(self.metrics),
                "details": copy.deepcopy(self.details)}

    @classmethod
    def parse(cls, value: Any, *, default_name: str = "reward"):
        if isinstance(value, cls):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return cls(float(value), {default_name: float(value)})
        if not isinstance(value, Mapping):
            raise TypeError(f"verifier returned {type(value).__name__}; "
                            "expected RewardResult, number, or mapping")
        if "total" not in value:
            raise ValueError("reward mapping needs a numeric 'total'")
        return cls(value["total"], dict(value.get("components") or {}),
                   dict(value.get("metrics") or {}),
                   dict(value.get("details") or {}))


class Verifier(Protocol):
    def evaluate(self, trajectory: "Trajectory",
                 state: StateTrace) -> RewardResult: ...


async def evaluate(verifier: Any, trajectory: "Trajectory",
                   state: StateTrace) -> RewardResult:
    fn = getattr(verifier, "evaluate", verifier)
    value = fn(trajectory, state)
    if inspect.isawaitable(value):
        value = await value
    return RewardResult.parse(value,
                              default_name=object_reference(verifier))


class FunctionVerifier:
    """Adapt any sync or async ``(trajectory, state)`` callable."""

    def __init__(self, function):
        self.function = resolve_callable(function, what="verifier function")

    def evaluate(self, trajectory: "Trajectory", state: StateTrace):
        return self.function(trajectory, state)


class CompositeVerifier:
    """Sum independent verifiers while retaining every named component."""

    def __init__(self, verifiers):
        self.verifiers = tuple(verifiers)

    async def evaluate(self, trajectory: "Trajectory",
                       state: StateTrace) -> RewardResult:
        total = 0.0
        components = {}
        metrics = {}
        details = {}
        for index, verifier in enumerate(self.verifiers):
            result = await evaluate(verifier, trajectory, state)
            name = getattr(verifier, "name", None) or object_reference(verifier)
            total += result.total
            if result.components:
                for key, value in result.components.items():
                    if key in components:
                        raise ValueError(f"duplicate reward component {key!r}")
                    components[key] = value
            else:
                key = str(name)
                if key in components:
                    key += f"#{index}"
                components[key] = result.total
            for key, value in result.metrics.items():
                if key in metrics:
                    raise ValueError(f"duplicate reward metric {key!r}")
                metrics[key] = value
            if result.details:
                details[str(name)] = result.details
        return RewardResult(total, components, metrics, details)


def state_value(state: StateTrace, path: str, *, sample: str = "last"):
    if not state.samples:
        raise ValueError("reward requested state but the trace has no samples")
    values = state.samples[0 if sample == "first" else -1].values
    value = values
    for part in path.split("."):
        if not part:
            continue
        value = value[part] if isinstance(value, Mapping) else getattr(
            value, part)
    return value


class StateValueReward:
    """Reward a JSON state path satisfying an injected predicate."""

    def __init__(self, name: str, path: str, *, equals: Any = None,
                 predicate=None, reward: float = 1.0,
                 otherwise: float = 0.0, sample: str = "last"):
        if predicate is None:
            predicate = lambda value: value == equals
        self.name = str(name)
        self.path = str(path)
        self.predicate = (None if predicate is None else
                          resolve_callable(predicate,
                                           what="state reward predicate"))
        self.reward = float(reward)
        self.otherwise = float(otherwise)
        self.sample = sample

    def evaluate(self, trajectory: "Trajectory", state: StateTrace):
        value = state_value(state, self.path, sample=self.sample)
        passed = bool(self.predicate(value))
        score = self.reward if passed else self.otherwise
        return RewardResult(score, {self.name: score},
                            {f"{self.name}.passed": passed},
                            {"path": self.path, "value": value})


class EventCountReward:
    def __init__(self, event: str, *, at_least: int = 1,
                 reward: float = 1.0, otherwise: float = 0.0,
                 name: str | None = None, predicate=None):
        self.event = str(event)
        self.at_least = int(at_least)
        self.reward = float(reward)
        self.otherwise = float(otherwise)
        self.name = name or f"event:{self.event}"
        self.predicate = (None if predicate is None else
                          resolve_callable(predicate,
                                           what="event reward predicate"))

    def evaluate(self, trajectory: "Trajectory", state: StateTrace):
        matches = [event for event in state.events
                   if event.name == self.event and
                   (self.predicate is None or self.predicate(event))]
        passed = len(matches) >= self.at_least
        score = self.reward if passed else self.otherwise
        return RewardResult(score, {self.name: score},
                            {f"{self.name}.count": len(matches),
                             f"{self.name}.passed": passed})


class EventSequenceReward:
    """Reward an ordered subsequence of named environment events."""

    def __init__(self, events, *, reward: float = 1.0,
                 otherwise: float = 0.0, name: str = "event_sequence"):
        self.events = tuple(str(event) for event in events)
        if not self.events:
            raise ValueError("event sequence cannot be empty")
        self.reward = float(reward)
        self.otherwise = float(otherwise)
        self.name = str(name)

    def evaluate(self, trajectory: "Trajectory", state: StateTrace):
        wanted = iter(self.events)
        current = next(wanted)
        found = 0
        for event in state.events:
            if event.name != current:
                continue
            found += 1
            try:
                current = next(wanted)
            except StopIteration:
                break
        passed = found == len(self.events)
        score = self.reward if passed else self.otherwise
        return RewardResult(score, {self.name: score},
                            {f"{self.name}.matched": found,
                             f"{self.name}.passed": passed})
