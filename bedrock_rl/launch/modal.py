"""Build the local command that submits one training config to Modal.

This module deliberately does not import :mod:`modal`.  The normal CLI and
its tests therefore remain lightweight; Modal is needed only when a user
selects it as the launch backend. ``bedrock_rl.launch.training_app`` is the remote
application imported by the generated command.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import time
import uuid
from pathlib import Path


class ModalLaunchError(ValueError):
    """A launch request Modal cannot execute as written."""


@dataclasses.dataclass(frozen=True)
class ModalScoutLaunch:
    """One durable CPU-fleet seed-scout submission."""

    config: str
    job: str | None
    sets: tuple[str, ...]
    target: int
    max_candidates: int
    containers: int
    results: str
    run_id: str

    def environment(self) -> dict[str, str]:
        return {
            "BRL_MODAL_SDG_CONFIG": self.config,
            "BRL_MODAL_SDG_JOB": self.job or "",
            "BRL_MODAL_SDG_SETS": json.dumps(self.sets),
            "BRL_MODAL_SDG_CONTAINERS": str(self.containers),
            "BRL_MODAL_SDG_RESULTS": self.results,
        }

    def command(self) -> list[str]:
        return [
            "modal", "run", "--detach", "-m", "bedrock_rl.launch.sdg_app",
            "--max-candidates", str(self.max_candidates),
            "--target", str(self.target), "--run-id", self.run_id,
        ]


def build_scout_launch(
    *,
    root: str | os.PathLike[str],
    config: str | os.PathLike[str],
    job: str | None,
    sets: list[str] | tuple[str, ...],
    target: int,
    max_candidates: int,
    containers: int,
    results: str,
    run_id: str | None = None,
) -> ModalScoutLaunch:
    """Validate and build a detached Modal CPU seed-scout invocation."""
    root_path = Path(root).resolve()
    config_path = Path(config).resolve()
    try:
        relative = config_path.relative_to(root_path)
    except ValueError as exc:
        raise ModalLaunchError(
            "Modal configs must live inside this checkout so relative tasks "
            f"and components enter the image: {config_path}") from exc
    target = int(target)
    max_candidates = int(max_candidates)
    containers = int(containers)
    if target < 1:
        raise ModalLaunchError("--scout-worlds must be at least 1")
    if max_candidates < target:
        raise ModalLaunchError(
            "--scout-attempts cannot be smaller than --scout-worlds")
    if not 1 <= containers <= 5000:
        raise ModalLaunchError(
            "--scout-containers must be between 1 and 5000")
    results = str(results).strip()
    if not results:
        raise ModalLaunchError("--scout-results cannot be empty")
    run_id = str(run_id or (
        time.strftime("sdg-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]))
    if not run_id.strip() or ":" in run_id:
        raise ModalLaunchError(
            "--scout-run must be non-empty and cannot contain ':'")
    return ModalScoutLaunch(
        config=str(relative), job=job, sets=tuple(sets), target=target,
        max_candidates=max_candidates, containers=containers,
        results=results, run_id=run_id)


def task_turn_budget(task: str | None, base_dir: str | os.PathLike[str]) -> int:
    """Resolve Modal's episode budget through the ordinary task contract."""
    from bedrock_rl.config.paths import DEFAULT_TASK, task_yaml
    from bedrock_rl.env.task import Task

    path = task_yaml(task or DEFAULT_TASK, base_dir=str(base_dir))
    return Task.load(path).max_turns


def modal_gpu_dockerfile(source: str) -> str:
    """Select the GPU default for Modal's all-stage Dockerfile builder.

    Modal builds the final Dockerfile stage and currently resolves a stage
    name used by ``FROM`` from the Dockerfile default before applying
    forwarded build arguments.  Keep the repository's lightweight CPU
    default for ordinary ``docker build`` while changing that one default in
    the temporary Dockerfile submitted to Modal.
    """
    marker = "ARG FINAL_STAGE=cpu"
    if source.count(marker) != 1:
        raise ModalLaunchError(
            f"Dockerfile must contain exactly one {marker!r}")
    return source.replace(marker, "ARG FINAL_STAGE=gpu")


def modal_cpu_dockerfile(source: str) -> str:
    """Return only the CPU half of the repository Dockerfile for Modal.

    Modal's Dockerfile frontend resolves the final ``FROM`` through every
    named stage before applying build arguments.  The ordinary multi-target
    Dockerfile consequently pulls the CUDA training base even when its
    ``FINAL_STAGE`` is ``cpu``.  A seed scout needs neither CUDA nor verl, so
    cut at the documented GPU boundary and make the already-built CPU stage
    the literal final stage.
    """
    marker = "# ── gpu: the training stack on top"
    if source.count(marker) != 1:
        raise ModalLaunchError(
            f"Dockerfile must contain exactly one {marker!r}")
    cpu_source, _ = source.split(marker, 1)
    return cpu_source.rstrip() + "\n\nFROM cpu AS default\n"


# Since 2026-05-31 Modal clustered Functions require a full host.  These are
# the full-host shapes for the GPU families the public Modal API exposes.
# Keeping this launch-time fact here means train.yaml never names device ids.
_FULL_NODE_GPUS = {
    "A10": 4,
    "A100": 8,
    "A100-40GB": 8,
    "A100-80GB": 8,
    "B200": 8,
    "H100": 8,
    "H100!": 8,
    "H200": 8,
    "L4": 8,
    "L40S": 8,
    "RTX-PRO-6000": 8,
    "T4": 8,
}
_GPU = re.compile(r"^(?P<kind>[A-Za-z0-9+!_-]+)(?::(?P<count>[1-8]))?$")
_CUDA_INCOMPATIBLE_GPUS = {
    "B300": "B300 requires a CUDA 13.1 image",
    "B200+": "B200+ may schedule a B300 host",
}


def normalize_gpu(value: str | None, nodes: int) -> tuple[str, int]:
    """Return a Modal GPU spec and the number of visible devices per node."""
    value = value or "H100"
    match = _GPU.fullmatch(value)
    if match is None:
        raise ModalLaunchError(
            f"invalid Modal GPU {value!r}; use a type such as H100 or "
            "H100:8")
    kind = match.group("kind").upper()
    if kind in _CUDA_INCOMPATIBLE_GPUS:
        raise ModalLaunchError(
            f"Modal GPU {kind} is incompatible with this image's locked "
            f"CUDA 12.8 stack: {_CUDA_INCOMPATIBLE_GPUS[kind]}")
    if kind not in _FULL_NODE_GPUS:
        raise ModalLaunchError(
            f"unsupported Modal GPU type {match.group('kind')!r}; choose "
            + ", ".join(_FULL_NODE_GPUS))
    count = int(match.group("count") or 1)
    if nodes > 1:
        full = _FULL_NODE_GPUS[kind]
        if match.group("count") is None:
            count = full
        elif count != full:
            raise ModalLaunchError(
                f"Modal multi-node clusters require the full {kind} host "
                f"({kind}:{full}), not {value}")
    return (f"{kind}:{count}" if count > 1 else kind), count


@dataclasses.dataclass(frozen=True)
class ModalLaunch:
    config: str
    preset: str | None
    nodes: int
    gpu: str
    gpus_per_node: int
    volume: str
    secrets: tuple[str, ...]
    sets: tuple[str, ...]
    train_env: tuple[str, ...]
    timeout: int
    rdma: bool
    smoke: bool
    detach: bool
    run_id: str | None

    def environment(self) -> dict[str, str]:
        return {
            "BRL_MODAL_NODES": str(self.nodes),
            "BRL_MODAL_GPU": self.gpu,
            "BRL_MODAL_GPUS_PER_NODE": str(self.gpus_per_node),
            "BRL_MODAL_VOLUME": self.volume,
            "BRL_MODAL_SECRETS": ",".join(self.secrets),
            "BRL_MODAL_SETS": json.dumps(self.sets),
            "BRL_MODAL_TRAIN_ENV": json.dumps(self.train_env),
            "BRL_MODAL_TIMEOUT": str(self.timeout),
            "BRL_MODAL_RDMA": "1" if self.rdma else "0",
            "BRL_MODAL_MODE": "smoke" if self.smoke else "train",
        }

    def command(self) -> list[str]:
        command = ["modal", "run"]
        if self.detach:
            command.append("--detach")
        command += ["-m", "bedrock_rl.launch.training_app",
                    "--config", self.config]
        if self.preset:
            command += ["--preset", self.preset]
        if self.smoke:
            command.append("--smoke")
        if self.run_id:
            command += ["--run-id", self.run_id]
        return command


def build_launch(
    *,
    root: str | os.PathLike[str],
    config: str | os.PathLike[str],
    preset: str | None,
    target: str,
    nodes: int,
    gpu: str | None,
    volume: str,
    secrets: list[str] | tuple[str, ...],
    sets: list[str] | tuple[str, ...],
    train_env: list[str] | tuple[str, ...],
    timeout: int,
    rdma: bool,
    smoke: bool,
    detach: bool,
    run_id: str | None = None,
) -> ModalLaunch:
    """Validate provider settings and make a portable Modal invocation."""
    root_path = Path(root).resolve()
    config_path = Path(config).resolve()
    try:
        remote_config = config_path.relative_to(root_path)
    except ValueError as exc:
        raise ModalLaunchError(
            "Modal configs must live inside this checkout so the task and "
            f"its relative files are included in the image: {config_path}"
        ) from exc
    if nodes < 1:
        raise ModalLaunchError("--nodes must be at least 1")
    if nodes > 1 and target not in {"rl", "opd", "mopd"}:
        raise ModalLaunchError(
            "multi-node launch is supported by verl trainers rl/opd/mopd; "
            f"trainer/workflow {target!r} has no distributed launcher")
    if timeout < 60 or timeout > 86_400:
        raise ModalLaunchError("--timeout must be between 60 and 86400 seconds")
    if not volume.strip():
        raise ModalLaunchError("--volume cannot be empty")
    if run_id is not None:
        run_id = str(run_id).strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
            raise ModalLaunchError(
                "--modal-run must contain only letters, digits, dot, "
                "underscore, and dash")
    gpu_spec, count = normalize_gpu(gpu, nodes)
    return ModalLaunch(
        config=str(remote_config),
        preset=preset,
        nodes=nodes,
        gpu=gpu_spec,
        gpus_per_node=count,
        volume=volume,
        secrets=tuple(secrets),
        sets=tuple(sets),
        train_env=tuple(train_env),
        timeout=timeout,
        rdma=rdma and nodes > 1,
        smoke=smoke,
        detach=detach,
        run_id=run_id,
    )
