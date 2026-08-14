"""SDPO: Self-Distilled Policy Optimization, the algorithm core.

Upstream is lasgroup/SDPO (arXiv 2601.20802), a fork of verl 0.7. Despite
the name it is NOT a preference-pair method. There are no pairs and no
DPO loss. It is on-policy group sampling plus a distillation term in
which the SELF-TEACHER is the same policy conditioned on privileged
in-context information the student does not get. The teacher runs over
the student's own response tokens and its next-token distribution is
distilled back into the student. That loss REPLACES the policy-gradient
term, so grpo advantages are computed and then never read.

The divergence, the top-k tail handling, the alpha interpolation (0
forward KL, 1 reverse KL, generalized Jensen-Shannon between), the
importance clamp at is_clip, the aggregation and the demonstration
selection are ports of

  verl/trainer/ppo/core_algos.py::compute_self_distillation_loss
  verl/trainer/ppo/ray_trainer.py::_maybe_build_self_distillation_batch
  verl/workers/actor/dp_actor.py::update_policy
  verl/trainer/config/sdpo.yaml

Two upstream assumptions do not survive a multi-turn computer-use
rollout. Upstream refuses multimodal batches and flattens a trajectory
into a single user message, so the privileged channel here is a task
guide in the SYSTEM message. Upstream gates success on sequence reward,
which here also carries per-action costs and malformed-turn penalties, so
the gate reads the engine's terminal check instead.

The trainer module owns rollout construction and model updates.
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class SDPOConfig:
    """Distillation knobs. Defaults follow upstream sdpo.yaml and
    run_local_sdpo.sh (alpha 0.5, top-k 100, is_clip 2.0)."""

    full_logit_distillation: bool = True
    alpha: float = 0.5                  # 0 forward KL, 1 reverse KL, else JSD
    distillation_topk: Optional[int] = 100
    distillation_add_tail: bool = True
    is_clip: Optional[float] = 2.0
    loss_agg_mode: str = "token-mean"
    # Released SDPO regularizes the privileged self-teacher as an EMA of the
    # student rather than scoring with the just-mutated student weights.
    teacher_update_rate: float = 0.05
    # teacher-context channels
    teacher_hint: bool = True           # static task YAML system_hint
    teacher_turn_hint: bool = False     # state-dependent per-turn feedback
    teacher_demo: bool = True           # successful sibling's tool calls
    dont_reprompt_on_self_success: bool = True
    demo_template: str = ("\n\nA previous attempt that solved this task "
                          "made exactly these tool calls:\n\n{demo}\n")

    def __post_init__(self):
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0,1], got {self.alpha}")
        if not self.full_logit_distillation and self.alpha != 1.0:
            raise ValueError("only reverse KL (alpha=1) is defined without "
                             "full_logit_distillation")
        if self.distillation_topk is not None and self.distillation_topk <= 0:
            raise ValueError("distillation_topk must be positive")
        if self.is_clip is not None and self.is_clip <= 0:
            raise ValueError("is_clip must be positive")
        if not 0.0 <= self.teacher_update_rate <= 1.0:
            raise ValueError("teacher_update_rate must be in [0, 1]")
        if not (self.teacher_hint or self.teacher_turn_hint
                or self.teacher_demo):
            raise ValueError("at least one teacher channel must be enabled")


def agg_loss(loss_mat, loss_mask, loss_agg_mode="token-mean"):
    """Port of verl/upstream agg_loss for the modes this trainer uses."""
    if loss_agg_mode == "token-mean":
        return (loss_mat * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
    if loss_agg_mode == "seq-mean-token-mean":
        per_seq = (loss_mat * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1e-8)
        alive = (loss_mask.sum(-1) > 0).float()
        return (per_seq * alive).sum() / alive.sum().clamp(min=1.0)
    if loss_agg_mode == "seq-mean-token-sum":
        per_seq = (loss_mat * loss_mask).sum(-1)
        alive = (loss_mask.sum(-1) > 0).float()
        return (per_seq * alive).sum() / alive.sum().clamp(min=1.0)
    raise ValueError(f"unknown loss_agg_mode {loss_agg_mode!r}")


def _add_tail(log_probs):
    """Append a bucket holding the log-probability mass outside top-k.
    log(1 - sum p_i) via expm1 for stability, exactly as upstream."""
    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_s = torch.clamp(log_s, max=-1e-7)
    tail = torch.log(-torch.expm1(log_s))
    return torch.cat([log_probs, tail], dim=-1)


def _renorm(log_probs):
    return log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)


def self_distillation_loss(student_log_probs, teacher_log_probs, response_mask,
                           cfg, old_log_probs=None,
                           student_topk_log_probs=None,
                           teacher_topk_log_probs=None,
                           self_distillation_mask=None):
    """Per-token divergence between the hint-free student and the
    hint-conditioned self-teacher over the same response tokens.

    student_log_probs / teacher_log_probs: (B, T) sampled-token log-probs.
    *_topk_log_probs: (B, T, k) at the STUDENT's top-k indices, which is
    the support upstream distills over.
    self_distillation_mask: (B,) 1 for rollouts whose teacher context
    carries privileged information.
    """
    metrics = {}
    loss_mask = response_mask
    if self_distillation_mask is not None:
        loss_mask = loss_mask * self_distillation_mask.unsqueeze(-1)

    if cfg.full_logit_distillation:
        if student_topk_log_probs is None or teacher_topk_log_probs is None:
            raise ValueError("full-logit distillation needs top-k log-probs")
        s = student_topk_log_probs
        t = teacher_topk_log_probs
        if cfg.distillation_add_tail:
            s, t = _add_tail(s), _add_tail(t)
        else:
            s, t = _renorm(s), _renorm(t)
        if cfg.alpha == 0.0:
            kl = F.kl_div(s, t, reduction="none", log_target=True)
        elif cfg.alpha == 1.0:
            kl = F.kl_div(t, s, reduction="none", log_target=True)
        else:
            a = torch.tensor(cfg.alpha, dtype=s.dtype, device=s.device)
            mixture = torch.logsumexp(
                torch.stack([s + torch.log1p(-a), t + torch.log(a)]), dim=0)
            kl_t = F.kl_div(mixture, t, reduction="none", log_target=True)
            kl_s = F.kl_div(mixture, s, reduction="none", log_target=True)
            kl = torch.lerp(kl_s, kl_t, a)
        per_token = kl.sum(-1)
    else:
        log_ratio = student_log_probs - teacher_log_probs
        per_token = log_ratio.detach() * student_log_probs

    if cfg.is_clip is not None:
        if old_log_probs is None:
            raise ValueError("is_clip needs old_log_probs")
        neg_kl = (student_log_probs - old_log_probs).detach().clamp(-20.0, 20.0)
        per_token = per_token * torch.exp(neg_kl).clamp(max=cfg.is_clip)

    loss = agg_loss(per_token, loss_mask, cfg.loss_agg_mode)
    with torch.no_grad():
        n = loss_mask.sum().clamp(min=1.0)
        metrics["sdpo/per_token_loss"] = float((per_token * loss_mask).sum() / n)
        metrics["sdpo/teacher_minus_student_logp"] = float(
            ((teacher_log_probs - student_log_probs) * loss_mask).sum() / n)
        metrics["sdpo/distilled_tokens"] = float(loss_mask.sum())
    return loss, metrics


def demo_for(index, group_ids, successes, demos, cfg):
    """Pick a successful sibling's demonstration text for one rollout,
    mirroring upstream _get_solution: first successful index in the same
    group, optionally excluding the rollout itself."""
    gid = group_ids[index]
    cand = [j for j, (g, ok) in enumerate(zip(group_ids, successes))
            if g == gid and ok]
    if cfg.dont_reprompt_on_self_success:
        cand = [j for j in cand if j != index]
    if not cand:
        return None
    return demos[cand[0]]
