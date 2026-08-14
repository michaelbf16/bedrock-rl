"""On-policy Self-Distillation Fine-Tuning (SDFT).

SDFT is learning from demonstrations without sampling those demonstrations as
the student policy.  The student generates an unconditioned trajectory.  A
EMA copy of the same policy then re-evaluates the exact student tokens with
an expert demonstration in context.  The forward KL from that conditioned
self-teacher to the student is optimized at the states the student visited.

This is the algorithm described by Shenfeld et al. (2026), not the separate
self-distillation workflow in :mod:`bedrock_rl.train.self_distill`.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SDFTConfig:
    """Paper defaults used by the released SDFT implementation."""

    teacher_mixup_alpha: float = 0.01
    loss_tokens_to_skip: int = 3
    teacher_template: str = (
        "An expert demonstration for this task follows. Use it to judge how "
        "the current attempt should continue.\n\n{demonstration}"
    )

    def __post_init__(self):
        if not 0.0 < self.teacher_mixup_alpha <= 1.0:
            raise ValueError("teacher_mixup_alpha must be in (0, 1]")
        if self.loss_tokens_to_skip < 0:
            raise ValueError("loss_tokens_to_skip cannot be negative")


@dataclass(frozen=True)
class Demonstration:
    """Privileged response paired to the world that produced it."""

    text: str
    seed: int | None = None


def _messages(value: Any) -> list[dict]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise TypeError("demonstration messages must be a list")
    return [dict(message) for message in value]


def demonstration_text(messages: Iterable[Mapping[str, Any]]) -> str:
    """Render the expert's assistant actions as stable, model-readable text."""
    actions = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        compact = {}
        if message.get("content") not in (None, ""):
            compact["content"] = message["content"]
        if message.get("tool_calls"):
            compact["tool_calls"] = message["tool_calls"]
        if compact:
            actions.append(json.dumps(compact, ensure_ascii=False,
                                      separators=(",", ":")))
    if not actions:
        raise ValueError("demonstration has no assistant action")
    return "\n".join(actions)


def _demo_from_row(row: Mapping[str, Any]) -> str:
    if row.get("messages") is not None:
        return demonstration_text(_messages(row["messages"]))
    if row.get("answer") not in (None, ""):
        return str(row["answer"])
    raise ValueError("demonstration row needs messages or answer")


def _demo_seed(row: Mapping[str, Any]) -> int | None:
    """Read the world identity from SFT rows or canonical trajectories."""
    value = row.get("seed")
    metadata = row.get("metadata")
    if value is None and isinstance(metadata, Mapping):
        generation = metadata.get("generation_case")
        episode = metadata.get("episode")
        if isinstance(generation, Mapping):
            value = generation.get("seed")
        if value is None and isinstance(episode, Mapping):
            value = episode.get("seed")
    if value is None:
        return None
    if isinstance(value, bool) or int(value) != value:
        raise ValueError(f"demonstration seed {value!r} is not an integer")
    return int(value)


def load_demonstrations(path: str | Path) -> list[Demonstration]:
    """Load expert demonstrations from trajectory JSON/JSONL or SFT parquet."""
    path = Path(path)
    suffix = path.suffix.lower()
    rows: list[Mapping[str, Any]]
    if suffix == ".jsonl":
        with path.open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    elif suffix == ".json":
        with path.open() as stream:
            value = json.load(stream)
        rows = value if isinstance(value, list) else [value]
    elif suffix in (".parquet", ".pq"):
        import pandas as pd
        frame = pd.read_parquet(path)
        rows = frame.to_dict("records")
    else:
        raise ValueError("demonstrations must be .json, .jsonl, or .parquet")
    demos = []
    for row in rows:
        # Canonical trajectory records and SFT rows both carry messages.
        demos.append(Demonstration(_demo_from_row(row), _demo_seed(row)))
    if not demos:
        raise ValueError(f"no demonstrations in {path}")
    return demos


def demonstrations_for(specs, demonstrations, rng, match="seed"):
    """Pair every sampled world with its privileged demonstration.

    SDFT's teacher context is an answer to the same prompt, not an unrelated
    successful row. Minecraft prompt rows identify that prompt by world seed.
    Cross-world demonstrations remain available, but only through the explicit
    ``random`` ablation.
    """
    if match not in ("seed", "random"):
        raise ValueError("demonstration match must be seed or random")
    demos = [d if isinstance(d, Demonstration) else Demonstration(str(d))
             for d in demonstrations]
    if match == "random":
        return [demos[rng.randrange(len(demos))].text for _ in specs]
    by_seed = {}
    for demo in demos:
        if demo.seed is not None:
            by_seed.setdefault(demo.seed, []).append(demo.text)
    selected = []
    for spec in specs:
        seed = spec.get("seed")
        choices = by_seed.get(seed, ())
        if not choices:
            raise ValueError(
                f"no demonstration matches world seed {seed!r}; generate "
                "paired SFT rows or set demonstration_match: random "
                "explicitly for a cross-world ablation")
        selected.append(choices[rng.randrange(len(choices))])
    return selected


def teacher_messages(student_messages: Iterable[Mapping[str, Any]],
                     demonstration: str, template: str) -> list[dict]:
    """Add privileged demonstration context without changing student input."""
    messages = [dict(message) for message in student_messages]
    context = template.format(demonstration=demonstration.strip())
    if messages and messages[0].get("role") == "system":
        messages[0] = dict(messages[0])
        messages[0]["content"] = (str(messages[0].get("content") or "")
                                  + "\n\n" + context).strip()
    else:
        messages.insert(0, {"role": "system", "content": context})
    return messages


def build_views(coder, rollouts, demonstrations, cfg: SDFTConfig):
    """Build student/teacher token views over identical student actions."""
    if not demonstrations:
        raise ValueError("SDFT needs at least one expert demonstration")
    views = []
    for index, rollout in enumerate(rollouts):
        demo = demonstrations[index % len(demonstrations)]
        tokens_left_to_skip = cfg.loss_tokens_to_skip
        for turn in rollout.sampled_turns:
            teacher = teacher_messages(
                turn.teacher_messages, demo, cfg.teacher_template)
            view = coder.sampled_view(
                turn.prompt_ids, turn.response_ids, teacher,
                turn.image_pads)
            if view is not None:
                view["images"] = list(turn.images)
                # Carry the one trajectory-prefix skip across turn
                # boundaries. A two-token first turn and a three-token skip
                # means the first token of turn two is skipped too.
                view["skip_tokens"] = min(
                    tokens_left_to_skip, len(turn.response_ids))
                tokens_left_to_skip -= view["skip_tokens"]
                view["trajectory"] = index
                view["token_count"] = max(
                    len(turn.response_ids) - view["skip_tokens"], 0)
            views.append(view)
    totals = {}
    for view in views:
        if view is not None:
            key = view["trajectory"]
            totals[key] = totals.get(key, 0) + view["token_count"]
    for view in views:
        if view is not None:
            total = totals[view["trajectory"]]
            view["trajectory_weight"] = (
                view["token_count"] / total if total else 0.0)
    return views


def forward_kl(student_log_probs, teacher_log_probs, response_mask,
               *, skip_tokens: int = 3):
    """Exact per-token forward KL ``KL(teacher || student)``."""
    if student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError("student and teacher distributions must align")
    if response_mask.shape != student_log_probs.shape[:-1]:
        raise ValueError("response mask does not match token distributions")
    mask = response_mask.to(student_log_probs.dtype)
    if skip_tokens:
        ordinal = torch.cumsum(mask, dim=-1)
        mask = mask * (ordinal > skip_tokens)
    per_token = F.kl_div(student_log_probs, teacher_log_probs,
                         reduction="none", log_target=True).sum(-1)
    count = mask.sum().clamp(min=1.0)
    loss = (per_token * mask).sum() / count
    metrics = {
        "sdft/forward_kl": float(loss.detach()),
        "sdft/distilled_tokens": float(mask.sum().detach()),
    }
    return loss, metrics


@torch.no_grad()
def update_teacher(teacher, student, alpha: float):
    """Paper release's moving self-teacher: ``t = a*s + (1-a)*t``."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("teacher update alpha must be in (0, 1]")
    for teacher_parameter, student_parameter in zip(
            teacher.parameters(), student.parameters()):
        if teacher_parameter is student_parameter:
            continue
        teacher_parameter.mul_(1.0 - alpha).add_(
            student_parameter.detach().to(teacher_parameter.dtype),
            alpha=alpha)
    for teacher_buffer, student_buffer in zip(teacher.buffers(),
                                               student.buffers()):
        teacher_buffer.copy_(student_buffer.to(teacher_buffer.dtype))


def ema_teacher(student):
    """Copy trainable EMA state while sharing an immutable frozen base.

    LoRA SDFT therefore allocates independent fp32 adapter state without a
    second full vision-language model. Full fine-tuning still copies every
    parameter because every parameter can change.
    """
    shared = {id(parameter): parameter for parameter in student.parameters()
              if not parameter.requires_grad}
    teacher = copy.deepcopy(student, shared)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher
