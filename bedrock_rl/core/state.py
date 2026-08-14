"""Versioned state probes and event traces for environment-independent rewards."""

from __future__ import annotations

import copy
import inspect
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Protocol

from bedrock_rl.core.components import object_reference, resolve_callable


def json_value(value: Any, *, what: str = "value") -> Any:
    """Deep-copy a value and refuse state that cannot be persisted."""
    value = copy.deepcopy(value)
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{what} is not JSON-serializable: {exc}") from exc
    return value


@dataclass(frozen=True)
class StateEvent:
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    tick: int | None = None
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def parse(cls, value: "StateEvent | Mapping[str, Any]"):
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping) or not value.get("name"):
            raise TypeError("state event must be a mapping with a name")
        return cls(str(value["name"]),
                   json_value(dict(value.get("payload") or {}),
                              what="event payload"),
                   None if value.get("tick") is None else int(value["tick"]),
                   int(value.get("version", 1)))


@dataclass(frozen=True)
class StateSample:
    turn: int
    phase: str
    values: dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StateTrace:
    schema_version: int = 1
    samples: list[StateSample] = field(default_factory=list)
    events: list[StateEvent] = field(default_factory=list)

    def record(self, turn: int, phase: str,
               values: Mapping[str, Any]) -> StateSample:
        sample = StateSample(int(turn), str(phase),
                             json_value(dict(values), what="state sample"))
        self.samples.append(sample)
        return sample

    def extend_events(self, events: Iterable[StateEvent | Mapping[str, Any]]):
        self.events.extend(StateEvent.parse(event) for event in events)

    @property
    def latest(self) -> dict[str, Any]:
        return copy.deepcopy(self.samples[-1].values) if self.samples else {}

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version,
                "samples": [sample.to_dict() for sample in self.samples],
                "events": [event.to_dict() for event in self.events]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        trace = cls(int(value.get("schema_version", 1)))
        for sample in value.get("samples", ()):
            trace.samples.append(StateSample(
                int(sample["turn"]), str(sample["phase"]),
                json_value(dict(sample.get("values") or {})),
                float(sample.get("timestamp", 0.0))))
        trace.extend_events(value.get("events", ()))
        return trace


class StateProbe(Protocol):
    name: str

    def capture(self, source: Any) -> Any: ...


class FunctionProbe:
    def __init__(self, name: str, function):
        self.name = str(name)
        self.function = resolve_callable(function, what="state probe function")

    def capture(self, source: Any):
        return self.function(source)


class AttributeProbe:
    """Capture a dotted attribute/mapping path from an environment."""

    def __init__(self, name: str, path: str):
        self.name = str(name)
        self.path = tuple(part for part in str(path).split(".") if part)
        if not self.path:
            raise ValueError("attribute probe path cannot be empty")

    def capture(self, source: Any):
        value = source
        for part in self.path:
            value = (value[part] if isinstance(value, Mapping)
                     else getattr(value, part))
        return value


class ProbeSet:
    """Named probes captured together at reset, turns, and termination."""

    def __init__(self, probes: Iterable[StateProbe] = ()):
        self.probes = tuple(probes)
        names = [probe.name for probe in self.probes]
        if len(set(names)) != len(names):
            raise ValueError(f"state probe names must be unique, got {names}")

    async def capture(self, source: Any) -> dict[str, Any]:
        out = {}
        for probe in self.probes:
            value = probe.capture(source)
            if inspect.isawaitable(value):
                value = await value
            out[probe.name] = json_value(
                value, what=f"state probe {object_reference(probe)}")
        return out
