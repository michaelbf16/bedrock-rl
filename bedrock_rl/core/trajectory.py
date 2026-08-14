"""Canonical OpenAI-chat trajectories, exact model views, state, and reward."""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

from bedrock_rl.core.context import ContextView
from bedrock_rl.core.messages import (validate_message,
                                        validate_messages,
                                        validate_transcript)
from bedrock_rl.core.reward import RewardResult
from bedrock_rl.core.state import StateTrace, json_value


@dataclass
class TrajectoryView:
    name: str
    turn: int
    messages: list[dict]
    transforms: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "turn": self.turn,
                "messages": validate_transcript(self.messages),
                "transforms": copy.deepcopy(self.transforms)}


@dataclass
class Trajectory:
    """One portable trial trajectory.

    ``messages`` is normal OpenAI chat history and is the canonical account
    of what happened.  ``views`` stores the exact derived messages presented
    to teacher, student, deployment, or evaluator models.  Environment state
    and reward are siblings rather than pseudo-chat messages.
    """

    messages: list[dict] = field(default_factory=list)
    views: list[TrajectoryView] = field(default_factory=list)
    state: StateTrace = field(default_factory=StateTrace)
    reward: RewardResult | None = None
    task: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)
    schema: str = "bedrock.trajectory.v1"
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        messages = validate_transcript(self.messages)
        return {"schema": self.schema, "id": self.id,
                "created_at": self.created_at,
                "finished_at": self.finished_at,
                "task": json_value(self.task, what="trajectory task"),
                "messages": messages,
                "views": [view.to_dict() for view in self.views],
                "state": self.state.to_dict(),
                "reward": None if self.reward is None else self.reward.to_dict(),
                "provenance": json_value(self.provenance,
                                         what="trajectory provenance"),
                "metadata": json_value(self.metadata,
                                       what="trajectory metadata")}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        if value.get("schema") != "bedrock.trajectory.v1":
            raise ValueError(f"unsupported trajectory schema "
                             f"{value.get('schema')!r}")
        views = [TrajectoryView(
            str(view["name"]), int(view["turn"]),
            validate_transcript(view.get("messages", ())),
            copy.deepcopy(list(view.get("transforms") or ())))
                 for view in value.get("views", ())]
        reward = value.get("reward")
        return cls(
            messages=validate_transcript(value.get("messages", ())),
            views=views,
            state=StateTrace.from_dict(value.get("state") or {}),
            reward=None if reward is None else RewardResult.parse(reward),
            task=json_value(dict(value.get("task") or {})),
            provenance=json_value(dict(value.get("provenance") or {})),
            metadata=json_value(dict(value.get("metadata") or {})),
            id=str(value["id"]), schema="bedrock.trajectory.v1",
            created_at=float(value.get("created_at", 0.0)),
            finished_at=(None if value.get("finished_at") is None else
                         float(value["finished_at"])))

    def save(self, path: str | Path):
        """Atomically save one readable JSON artifact."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("w") as stream:
            json.dump(self.to_dict(), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
        return path

    def append_jsonl(self, path: str | Path):
        """Append one compact record for batched training artifacts."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(self.to_dict(), ensure_ascii=False,
                          separators=(",", ":"))
        with path.open("a") as stream:
            stream.write(line + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path):
        with Path(path).open() as stream:
            return cls.from_dict(json.load(stream))


class TrajectoryRecorder:
    def __init__(self, *, task: Mapping[str, Any] | None = None,
                 provenance: Mapping[str, Any] | None = None,
                 metadata: Mapping[str, Any] | None = None,
                 trajectory_id: str | None = None):
        self.trajectory = Trajectory(
            task=json_value(dict(task or {})),
            provenance=json_value(dict(provenance or {})),
            metadata=json_value(dict(metadata or {})),
            id=trajectory_id or uuid4().hex)

    @property
    def messages(self) -> list[dict]:
        return self.trajectory.messages

    @property
    def state(self) -> StateTrace:
        return self.trajectory.state

    def append(self, message: Mapping[str, Any]):
        copied = copy.deepcopy(dict(message))
        validate_message(copied, len(self.messages))
        self.messages.append(copied)
        return copied

    def extend(self, messages: Iterable[Mapping[str, Any]]):
        for message in messages:
            self.append(message)

    def record_view(self, name: str, turn: int, view: ContextView):
        saved = TrajectoryView(
            str(name), int(turn), validate_messages(view.messages),
            [transform.to_dict() if hasattr(transform, "to_dict")
             else copy.deepcopy(dict(transform))
             for transform in view.transforms])
        self.trajectory.views.append(saved)
        return saved

    def record_state(self, turn: int, phase: str,
                     values: Mapping[str, Any]):
        return self.state.record(turn, phase, values)

    def record_events(self, events):
        self.state.extend_events(events)

    def record_tool_result(self, turn: int, call, result):
        """Persist non-chat tool evidence without leaking it to the model."""
        records = self.trajectory.metadata.setdefault("tool_results", [])
        if not isinstance(records, list):
            raise TypeError("trajectory metadata.tool_results must be a list")
        record = {
            "turn": int(turn), "call_id": str(call.id),
            "name": str(call.name), "done": bool(result.done),
            "reward": (None if result.reward is None else
                       float(result.reward)),
            "metadata": json_value(result.metadata,
                                   what="tool result metadata"),
        }
        records.append(record)
        return record

    def finish(self, reward: RewardResult | Mapping[str, Any] | float | None):
        self.trajectory.reward = (None if reward is None else
                                  RewardResult.parse(reward))
        self.trajectory.finished_at = time.time()
        return self.trajectory
