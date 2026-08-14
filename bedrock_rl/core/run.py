"""One file describing a harness run, assembled entirely by import path."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from bedrock_rl.core.components import ComponentSpec
from bedrock_rl.core.context import Compose
from bedrock_rl.core.guidance import GuidancePolicy, NoGuidance
from bedrock_rl.core.harness import ToolHarness
from bedrock_rl.core.state import ProbeSet, json_value
from bedrock_rl.core.task import TaskSpec

RUN_KEYS = frozenset((
    "schema_version", "task", "model", "harness", "tools", "context",
    "guidance", "probes", "verifier", "max_turns", "view_name",
    "sampling", "output", "metadata", "provenance",
))
GUIDANCE_KEYS = frozenset(("provider", "static_level", "dynamic_level",
                           "dynamic_separator"))


def _component_list(value, what):
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError(f"{what} must be a list of components")
    return tuple(ComponentSpec.parse(item, what=f"{what} item")
                 for item in value)


@dataclass(frozen=True)
class RunSpec:
    task: TaskSpec
    model: ComponentSpec
    harness: ComponentSpec = field(default_factory=lambda: ComponentSpec(
        "bedrock_rl.core.harness:ToolHarness"))
    tools: tuple[ComponentSpec, ...] = ()
    context: tuple[ComponentSpec, ...] = ()
    guidance_provider: ComponentSpec | None = None
    guidance_options: dict[str, Any] = field(default_factory=dict)
    probes: tuple[ComponentSpec, ...] = ()
    verifier: ComponentSpec | None = None
    max_turns: int = 8
    view_name: str = "policy"
    sampling: dict[str, Any] = field(default_factory=dict)
    output: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1
    path: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, path: str | None = None):
        if not isinstance(value, Mapping):
            raise TypeError("run document must be a mapping")
        raw = dict(value)
        unknown = sorted(set(raw) - RUN_KEYS)
        if unknown:
            raise ValueError(f"unknown run keys: {', '.join(unknown)}")
        schema_version = int(raw.get("schema_version", 1))
        if schema_version != 1:
            raise ValueError(
                f"unsupported run schema_version {schema_version}; "
                "this release supports only 1")
        base = Path(path).resolve().parent if path else Path.cwd()
        task_value = raw.get("task")
        if isinstance(task_value, TaskSpec):
            task = task_value
        elif isinstance(task_value, Mapping):
            task = TaskSpec.from_dict(task_value, path=path)
        elif isinstance(task_value, str):
            task_path = Path(task_value)
            if not task_path.is_absolute():
                task_path = base / task_path
            task = TaskSpec.load(task_path)
        else:
            raise ValueError("run needs a task path or inline task mapping")
        if "model" not in raw:
            raise ValueError("run needs a model component")
        guidance_provider = None
        guidance_options = {}
        guidance = raw.get("guidance")
        if guidance is not None:
            if not isinstance(guidance, Mapping):
                raise TypeError("guidance must be a mapping")
            guidance = dict(guidance)
            bad = sorted(set(guidance) - GUIDANCE_KEYS)
            if bad:
                raise ValueError(f"unknown guidance keys: {', '.join(bad)}")
            if "provider" not in guidance:
                raise ValueError("guidance needs a provider component")
            guidance_provider = ComponentSpec.parse(
                guidance.pop("provider"), what="guidance provider")
            guidance_options = guidance
        verifier = (None if raw.get("verifier") is None else
                    ComponentSpec.parse(raw["verifier"],
                                        what="run verifier"))
        output = raw.get("output")
        if output is not None:
            output_path = Path(str(output))
            if path and not output_path.is_absolute():
                output_path = base / output_path
            output = str(output_path)
        max_turns = int(raw.get("max_turns", 8))
        if max_turns < 1:
            raise ValueError("run.max_turns must be at least one")
        return cls(
            task=task,
            model=ComponentSpec.parse(raw["model"], what="run model"),
            harness=ComponentSpec.parse(
                raw.get("harness", "bedrock_rl.core.harness:ToolHarness"),
                what="run harness"),
            tools=_component_list(raw.get("tools"), "run.tools"),
            context=_component_list(raw.get("context"), "run.context"),
            guidance_provider=guidance_provider,
            guidance_options=guidance_options,
            probes=_component_list(raw.get("probes"), "run.probes"),
            verifier=verifier, max_turns=max_turns,
            view_name=str(raw.get("view_name", "policy")),
            sampling=json_value(dict(raw.get("sampling") or {}),
                                what="run sampling"),
            output=output,
            metadata=json_value(dict(raw.get("metadata") or {}),
                                what="run metadata"),
            provenance=json_value(dict(raw.get("provenance") or {}),
                                  what="run provenance"),
            schema_version=schema_version, path=path)

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path).resolve()
        try:
            import yaml
        except ImportError as exc:                    # pragma: no cover
            raise RuntimeError("loading YAML runs requires pyyaml") from exc
        with path.open() as stream:
            value = yaml.safe_load(stream)
        return cls.from_dict(value, path=str(path))

    def build(self):
        model = self.model.create()
        environment = self.task.build_environment()
        tools = [spec.create() for spec in self.tools]
        policies = [spec.create() for spec in self.context]
        context = (None if not policies else policies[0]
                   if len(policies) == 1 else Compose(policies))
        guidance = NoGuidance()
        if self.guidance_provider is not None:
            guidance = GuidancePolicy(self.guidance_provider.create(),
                                      **self.guidance_options)
        probes = ProbeSet(spec.create() for spec in self.probes)
        verifier = (self.verifier.create() if self.verifier is not None else
                    self.task.build_verifier())
        injected = dict(model=model, environment=environment, tools=tools,
                        context=context, guidance=guidance, probes=probes,
                        verifier=verifier, max_turns=self.max_turns,
                        view_name=self.view_name, provenance=self.provenance)
        if self.harness.type == "bedrock_rl.core.harness:ToolHarness" and not (
                self.harness.config):
            return ToolHarness(**injected)
        return self.harness.create(**injected)

    def component_specs(self):
        """Every selected component, named for validation and diagnostics."""
        items = [("task.environment", self.task.environment),
                 ("model", self.model), ("harness", self.harness)]
        if self.task.verifier is not None:
            items.append(("task.verifier", self.task.verifier))
        items.extend((f"tool[{index}]", spec)
                     for index, spec in enumerate(self.tools))
        items.extend((f"context[{index}]", spec)
                     for index, spec in enumerate(self.context))
        if self.guidance_provider is not None:
            items.append(("guidance.provider", self.guidance_provider))
        items.extend((f"probe[{index}]", spec)
                     for index, spec in enumerate(self.probes))
        if self.verifier is not None:
            items.append(("verifier", self.verifier))
        return tuple(items)

    def validate_components(self):
        """Resolve every import path without constructing any component."""
        return tuple((name, spec.resolve())
                     for name, spec in self.component_specs())

    async def execute(self):
        trajectory = await self.build().run(
            self.task, sampling=self.sampling, metadata=self.metadata)
        if self.output:
            if self.output.endswith(".jsonl"):
                trajectory.append_jsonl(self.output)
            else:
                trajectory.save(self.output)
        return trajectory
