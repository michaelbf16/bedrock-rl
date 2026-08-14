"""Validated configuration bridge to verl 0.8's OPD and MOPD trainers.

Netherite owns the agent environment, task rewards, model views, and saved
trajectories.  It does not reimplement on-policy distillation: this module
turns the compact ``teachers:`` block into the exact Hydra fields consumed by
``verl.workers.config.DistillationConfig`` at the repository's pinned commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any


_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENGINES = {"vllm", "sglang"}


@dataclass(frozen=True)
class Teacher:
    name: str
    model: str
    route: str
    replicas: int = 1
    tensor_parallel: int = 1
    data_parallel: int = 1
    pipeline_parallel: int = 1
    engine: str = "vllm"
    gpu_memory_utilization: float = 0.4
    max_model_len: int | None = None

    @property
    def world_size(self) -> int:
        return (self.replicas * self.tensor_parallel * self.data_parallel
                * self.pipeline_parallel)


def _positive(raw: Any, field: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def load_teachers(payload: str, mode: str) -> list[Teacher]:
    """Parse and validate the config schema shared by the CLI and launcher."""
    if mode not in ("opd", "mopd"):
        raise ValueError("distillation mode is opd or mopd")
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"teachers is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise ValueError("teachers must be a non-empty mapping")
    if mode == "opd" and len(raw) != 1:
        raise ValueError("OPD needs exactly one teacher")
    if mode == "mopd" and len(raw) < 2:
        raise ValueError("MOPD needs at least two teachers")

    teachers = []
    routes = set()
    for name, values in raw.items():
        if not _NAME.fullmatch(str(name)) or name == "teacher_model":
            raise ValueError(
                f"teacher name {name!r} must be a Hydra-safe identifier and "
                "cannot be the reserved name 'teacher_model'")
        if not isinstance(values, dict) or not values.get("model"):
            raise ValueError(f"teacher {name!r} needs model")
        engine = str(values.get("engine", "vllm"))
        if engine not in _ENGINES:
            raise ValueError(
                f"teacher {name!r} engine is {engine!r}; use vllm or sglang")
        route = str(values.get("route", name))
        if not route:
            raise ValueError(f"teacher {name!r} route cannot be empty")
        if route in routes:
            raise ValueError(f"duplicate teacher route {route!r}")
        routes.add(route)
        try:
            memory = float(values.get("gpu_memory_utilization", 0.4))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"teacher {name!r} gpu_memory_utilization must be a number") from exc
        if not 0.0 < memory < 1.0:
            raise ValueError(
                f"teacher {name!r} gpu_memory_utilization must be in (0, 1)")
        maximum = values.get("max_model_len")
        pipeline_parallel = _positive(
            values.get("pipeline_parallel", 1),
            f"teachers.{name}.pipeline_parallel")
        # The pinned verl RolloutConfig exposes this field, but its own
        # validation raises for every value above one.  Accepting it here
        # would make a compact config look supported and fail only after the
        # teacher pool had been allocated.
        if pipeline_parallel != 1:
            raise ValueError(
                f"teacher {name!r} pipeline_parallel={pipeline_parallel} is "
                "not implemented by the pinned verl rollout engines; use 1")
        teachers.append(Teacher(
            name=str(name), model=str(values["model"]), route=route,
            replicas=_positive(values.get("replicas", 1),
                                f"teachers.{name}.replicas"),
            tensor_parallel=_positive(values.get("tensor_parallel", 1),
                                      f"teachers.{name}.tensor_parallel"),
            data_parallel=_positive(values.get("data_parallel", 1),
                                    f"teachers.{name}.data_parallel"),
            pipeline_parallel=pipeline_parallel,
            engine=engine, gpu_memory_utilization=memory,
            max_model_len=(None if maximum is None else
                           _positive(maximum,
                                     f"teachers.{name}.max_model_len")),
        ))
    return teachers


def stage_teachers(payload: str, mode: str,
                   provenance_out: str | None = None) -> str:
    """Resolve every teacher once and return config bound to local bytes.

    The tokenizer preflight and verl must consume the same immutable path;
    merely recording a Hub revision while passing the moving Hub id onward
    does not provide that guarantee.
    """
    from bedrock_rl import models

    teachers = load_teachers(payload, mode)
    raw = json.loads(payload)
    provenance = {}
    for teacher in teachers:
        requested = teacher.model
        base, adapter = models.policy_paths(requested)
        if adapter is not None:
            raise ValueError(
                f"teacher {teacher.name!r} is a LoRA adapter. The pinned verl "
                "teacher server accepts a full policy only; merge the adapter "
                "into a standalone checkpoint first")
        resolved = models.weights_path(base)
        if not os.path.isdir(resolved):
            raise ValueError(
                f"teacher {teacher.name!r} did not resolve to a model "
                f"directory: {resolved}")
        raw[teacher.name]["model"] = resolved
        revision = models.model_revision(base)
        snapshot = os.path.basename(os.path.normpath(resolved))
        if revision is None and re.fullmatch(r"[0-9a-f]{40,64}", snapshot):
            revision = snapshot
        provenance[teacher.name] = {
            "requested": requested,
            "resolved": resolved,
            "revision": revision,
            "files": models.artifact_files(resolved),
        }

    if provenance_out:
        output = os.path.abspath(os.path.expanduser(provenance_out))
        os.makedirs(os.path.dirname(output), exist_ok=True)
        temporary = f"{output}.tmp.{os.getpid()}"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump({"schema": "bedrock.teacher-policies.v1",
                       "mode": mode, "teachers": provenance}, stream,
                      indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, output)
    return json.dumps(raw, separators=(",", ":"), sort_keys=True)


def default_gpus_per_node(teachers: list[Teacher], mode: str,
                          nnodes: int) -> int:
    """Smallest teacher pool described by the config, evenly across nodes."""
    nnodes = _positive(nnodes, "teacher nnodes")
    total = (teachers[0].tensor_parallel * teachers[0].data_parallel
             * teachers[0].pipeline_parallel * nnodes
             if mode == "opd" else sum(t.world_size for t in teachers))
    if total % nnodes:
        raise ValueError(
            f"configured teachers occupy {total} GPUs, which cannot be "
            f"spread evenly over {nnodes} nodes; change replicas or parallelism")
    return total // nnodes


def hydra_overrides(teachers: list[Teacher], mode: str, *,
                    teacher_gpus_per_node: int, teacher_nnodes: int,
                    teacher_key: str, max_model_len: int,
                    loss_mode: str, topk: int, use_task_rewards: str,
                    use_policy_gradient: str, coefficient: float,
                    loss_max_clamp: str, log_prob_min_clamp: str) -> list[str]:
    """Return one argv item per line; no shell evaluation is involved."""
    pool = _positive(teacher_gpus_per_node, "teacher_gpus_per_node")
    nodes = _positive(teacher_nnodes, "teacher_nnodes")
    expected = pool * nodes
    if mode == "mopd":
        actual = sum(t.world_size for t in teachers)
        if actual != expected:
            raise ValueError(
                f"MOPD teachers occupy {actual} GPUs but the teacher pool is "
                f"{pool} x {nodes} = {expected}; change replicas or the pool")
    else:
        per_replica = (teachers[0].tensor_parallel
                       * teachers[0].data_parallel
                       * teachers[0].pipeline_parallel)
        if expected % per_replica:
            raise ValueError(
                f"OPD teacher replica size {per_replica} does not divide "
                f"the teacher pool of {expected} GPUs")

    args = [
        "distillation.enabled=True",
        f"distillation.n_gpus_per_node={pool}",
        f"distillation.nnodes={nodes}",
        f"distillation.teacher_key={teacher_key}",
        f"distillation.distillation_loss.loss_mode={loss_mode}",
        f"distillation.distillation_loss.topk={_positive(topk, 'topk')}",
        f"distillation.distillation_loss.use_task_rewards={use_task_rewards}",
        f"distillation.distillation_loss.use_policy_gradient={use_policy_gradient}",
        f"distillation.distillation_loss.distillation_loss_coef={float(coefficient)}",
        f"distillation.distillation_loss.loss_max_clamp={loss_max_clamp}",
        f"distillation.distillation_loss.log_prob_min_clamp={log_prob_min_clamp}",
    ]
    for teacher in teachers:
        prefix = ("distillation.teacher_models.teacher_model" if mode == "opd"
                  else f"+distillation.teacher_models.{teacher.name}")
        if mode == "mopd":
            args.extend((
                f"{prefix}.key={teacher.route}",
                f"{prefix}.num_replicas={teacher.replicas}",
            ))
        pipeline_prefix = (prefix if mode == "mopd" else "+" + prefix)
        args.extend((
            f"{prefix}.model_path={teacher.model}",
            f"{prefix}.inference.name={teacher.engine}",
            f"{prefix}.inference.tensor_model_parallel_size={teacher.tensor_parallel}",
            f"{prefix}.inference.data_parallel_size={teacher.data_parallel}",
            f"{pipeline_prefix}.inference.pipeline_model_parallel_size={teacher.pipeline_parallel}",
            f"{prefix}.inference.gpu_memory_utilization={teacher.gpu_memory_utilization}",
            f"{prefix}.inference.max_model_len={teacher.max_model_len or max_model_len}",
        ))
    return args


def _vocab_fingerprint(tokenizer) -> str:
    encoded = json.dumps(tokenizer.get_vocab(), sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def check_tokenizers(student: str, teachers: list[Teacher]) -> None:
    """Fail before GPU allocation if teacher scoring would decode wrong IDs."""
    from transformers import AutoTokenizer
    student_tok = AutoTokenizer.from_pretrained(student,
                                                trust_remote_code=False)
    student_fp = _vocab_fingerprint(student_tok)
    for teacher in teachers:
        teacher_tok = AutoTokenizer.from_pretrained(
            teacher.model, trust_remote_code=False)
        if (_vocab_fingerprint(teacher_tok) != student_fp
                or teacher_tok.eos_token_id != student_tok.eos_token_id):
            raise ValueError(
                f"teacher {teacher.name!r} does not share the student's exact "
                "token-id vocabulary. verl scores the student's response IDs "
                "under the teacher, so these models are incompatible")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("default-gpus", "hydra", "check-tokenizers",
                    "stage-teachers"):
        p = sub.add_parser(command)
        p.add_argument("--mode", choices=("opd", "mopd"), required=True)
        p.add_argument("--teachers-json", required=True)
        if command == "default-gpus":
            p.add_argument("--nnodes", type=int, required=True)
        elif command == "check-tokenizers":
            p.add_argument("--student", required=True)
        elif command == "stage-teachers":
            p.add_argument("--provenance-out")
        else:
            p.add_argument("--teacher-gpus-per-node", type=int, required=True)
            p.add_argument("--teacher-nnodes", type=int, required=True)
            p.add_argument("--teacher-key", default="data_source")
            p.add_argument("--max-model-len", type=int, required=True)
            p.add_argument("--loss-mode", default="k1")
            p.add_argument("--topk", type=int, default=64)
            p.add_argument("--use-task-rewards", default="False")
            p.add_argument("--use-policy-gradient", default="True")
            p.add_argument("--coefficient", type=float, default=1.0)
            p.add_argument("--loss-max-clamp", default="10.0")
            p.add_argument("--log-prob-min-clamp", default="-10.0")
    args = parser.parse_args(argv)
    try:
        teachers = load_teachers(args.teachers_json, args.mode)
        if args.command == "default-gpus":
            print(default_gpus_per_node(teachers, args.mode, args.nnodes))
        elif args.command == "check-tokenizers":
            check_tokenizers(args.student, teachers)
        elif args.command == "stage-teachers":
            print(stage_teachers(args.teachers_json, args.mode,
                                 args.provenance_out))
        else:
            for value in hydra_overrides(
                    teachers, args.mode,
                    teacher_gpus_per_node=args.teacher_gpus_per_node,
                    teacher_nnodes=args.teacher_nnodes,
                    teacher_key=args.teacher_key,
                    max_model_len=args.max_model_len,
                    loss_mode=args.loss_mode, topk=args.topk,
                    use_task_rewards=args.use_task_rewards,
                    use_policy_gradient=args.use_policy_gradient,
                    coefficient=args.coefficient,
                    loss_max_clamp=args.loss_max_clamp,
                    log_prob_min_clamp=args.log_prob_min_clamp):
                if "\n" in value:
                    raise ValueError("distillation values cannot contain newlines")
                print(value)
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
