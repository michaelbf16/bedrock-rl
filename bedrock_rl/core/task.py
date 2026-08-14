"""A small task definition: instruction, environment, and verifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from bedrock_rl.core.components import ComponentSpec
from bedrock_rl.core.state import json_value

TASK_KEYS = frozenset(("schema_version", "name", "instruction",
                       "instruction_file", "environment", "verifier",
                       "metadata"))


@dataclass(frozen=True)
class TaskSpec:
    """Trainer- and harness-independent task configuration."""

    name: str
    instruction: str
    environment: ComponentSpec
    verifier: ComponentSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1
    path: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, path: str | None = None):
        if not isinstance(value, Mapping):
            raise TypeError("task document must be a mapping")
        raw = dict(value)
        unknown = sorted(set(raw) - TASK_KEYS)
        if unknown:
            raise ValueError(f"unknown task keys: {', '.join(unknown)}")
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ValueError("task needs a non-empty name")
        direct = raw.get("instruction")
        instruction_file = raw.get("instruction_file")
        if direct is not None and instruction_file is not None:
            raise ValueError("task has both instruction and instruction_file")
        if instruction_file is not None:
            if path is None:
                raise ValueError("instruction_file needs a task file path to "
                                 "resolve against")
            instruction_path = Path(path).resolve().parent / str(
                instruction_file)
            instruction = instruction_path.read_text()
        else:
            instruction = str(direct or "")
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("task needs a non-empty instruction")
        if "environment" not in raw:
            raise ValueError("task needs an environment component")
        environment = ComponentSpec.parse(raw["environment"],
                                          what="task environment")
        verifier = (None if raw.get("verifier") is None else
                    ComponentSpec.parse(raw["verifier"],
                                        what="task verifier"))
        metadata = json_value(dict(raw.get("metadata") or {}),
                              what="task metadata")
        return cls(name, instruction, environment, verifier, metadata,
                   int(raw.get("schema_version", 1)), path)

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path).resolve()
        if path.is_dir():
            candidates = [path / "task.yaml", path / "task.yml"]
            path = next((candidate for candidate in candidates
                         if candidate.is_file()), candidates[0])
        try:
            import yaml
        except ImportError as exc:                    # pragma: no cover
            raise RuntimeError("loading YAML tasks requires pyyaml") from exc
        with path.open() as stream:
            value = yaml.safe_load(stream)
        return cls.from_dict(value, path=str(path))

    def build_environment(self, **injected):
        return self.environment.create(**injected)

    def build_verifier(self, **injected):
        return (None if self.verifier is None else
                self.verifier.create(**injected))

    def to_dict(self) -> dict[str, Any]:
        out = {"schema_version": self.schema_version, "name": self.name,
               "instruction": self.instruction,
               "environment": {"type": self.environment.type,
                               **self.environment.config},
               "metadata": dict(self.metadata)}
        if self.verifier is not None:
            out["verifier"] = {"type": self.verifier.type,
                               **self.verifier.config}
        return out
