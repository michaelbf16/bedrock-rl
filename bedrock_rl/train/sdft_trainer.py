"""Multi-turn on-policy SDFT trainer for Netherite tasks."""

from __future__ import annotations

import argparse
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from bedrock_rl.sdg import load_episode_specs
from bedrock_rl import reporting as console
from bedrock_rl.adapters.netherite.chat import Coder
from bedrock_rl.env.task import Task
from bedrock_rl.train import peft_util
from bedrock_rl.train.rollout import (add_sampler_args, build_sampler,
                                        check_engine, load_model,
                                        model_banner, play)
from bedrock_rl.train.sdft import (SDFTConfig, build_views, ema_teacher,
                                     demonstrations_for, forward_kl,
                                     load_demonstrations, update_teacher)
from bedrock_rl.train.sdpo_trainer import save


def response_distributions(model, coder, ids, mask, images, device):
    """Exact next-token log distributions at assistant-token positions."""
    pixels, grid, _ = coder.encode_images(images)
    input_ids = torch.tensor([ids], device=device)
    values = {"input_ids": input_ids,
              "attention_mask": torch.ones_like(input_ids)}
    if pixels is not None:
        values["pixel_values"] = pixels.to(device, model.dtype)
        values["image_grid_thw"] = grid.to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**values).logits[0]
    positions = torch.tensor(
        [index - 1 for index, active in enumerate(mask)
         if active and index > 0], device=device)
    return torch.log_softmax(logits[positions].float(), dim=-1)


def sdft_step(student, teacher, coder, views, cfg, device, metrics):
    """Distill every usable student-generated trajectory; reward is not a gate."""
    live = [view for view in views if view is not None]
    metrics["sdft/usable_views"] = len(live) / max(len(views), 1)
    active = [view for view in live if view["trajectory_weight"] > 0]
    trajectories = {view["trajectory"] for view in active}
    if not trajectories:
        metrics["sdft/distilled"] = 0
        return 0.0
    metrics["sdft/distilled"] = len(trajectories)
    total = 0.0
    token_total = 0.0
    for view in active:
        student_ids, student_mask = view["student"]
        teacher_ids, teacher_mask = view["teacher"]
        student_log_probs = response_distributions(
            student, coder, student_ids, student_mask, view["images"],
            device)
        with torch.no_grad():
            teacher_log_probs = response_distributions(
                teacher, coder, teacher_ids, teacher_mask, view["images"],
                device)
        response_mask = torch.ones(
            1, student_log_probs.shape[0], device=device)
        loss, values = forward_kl(
            student_log_probs.unsqueeze(0),
            teacher_log_probs.unsqueeze(0), response_mask,
            skip_tokens=view.get("skip_tokens", cfg.loss_tokens_to_skip))
        scale = view["trajectory_weight"] / len(trajectories)
        (loss * scale).backward()
        total += float(loss.detach()) * scale
        metrics["sdft/forward_kl"] = (
            metrics.get("sdft/forward_kl", 0.0)
            + values["sdft/forward_kl"] * scale)
        token_total += values["sdft/distilled_tokens"]
    metrics["sdft/distilled_tokens"] = token_total / len(trajectories)
    return total


def _arguments():
    parser = argparse.ArgumentParser(
        description="on-policy SDFT from expert demonstrations")
    parser.add_argument("--task", required=True)
    parser.add_argument("--data", required=True,
                        help="RL prompt parquet containing episode specs")
    parser.add_argument("--demonstrations", required=True,
                        help="expert trajectory JSON/JSONL or SFT parquet")
    parser.add_argument("--demonstration-match", choices=("seed", "random"),
                        default="seed", help="pair teacher demonstrations by "
                        "world seed; random is an explicit cross-world ablation")
    parser.add_argument("--model", default="qwen3-vl")
    parser.add_argument("--out", default=None)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--prompts", type=int, default=8,
                        help="unhinted student trajectories per step")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--teacher-mixup-alpha", type=float, default=0.01)
    parser.add_argument("--skip-tokens", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--netherite-home", default=None)
    parser.add_argument("--frames-root", default="/tmp/netherite_sdft_frames")
    parser.add_argument("--run", default=None)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--save-every", type=int, default=0)
    peft_util.add_args(parser)
    add_sampler_args(parser)
    return parser.parse_args()


def main():
    args = _arguments()
    if args.lora_4bit:
        raise SystemExit(
            "trainer: sdft does not support --lora-4bit: its EMA teacher "
            "path is not implemented for a quantized base. Use ordinary "
            "LoRA here, or trainer: sft for QLoRA.")
    task = Task.load(args.task)
    cfg = SDFTConfig(teacher_mixup_alpha=args.teacher_mixup_alpha,
                     loss_tokens_to_skip=args.skip_tokens)
    demonstrations = load_demonstrations(args.demonstrations)
    out = args.out or os.path.join(os.path.expanduser("~"), "ckpts",
                                   f"netherite_{task.name}_sdft")
    os.makedirs(out, exist_ok=True)

    # The paper release corrects vLLM sampling/training log-prob mismatch.
    # This compact trainer does not yet persist sampler log-probs, so calling
    # that path paper-faithful would be false. HF generation is genuinely
    # on-policy because it samples from the student module directly.
    engine = check_engine(args, trainable=True)
    if engine != "hf":
        raise SystemExit(
            "trainer: sdft currently requires rollout_engine: hf; the vLLM "
            "path needs sampling log-probs for the paper's importance "
            "correction and is refused rather than mislabeled on-policy")

    from bedrock_rl import models as registry
    base, found = registry.policy_paths(args.model)
    if found and not args.lora_adapter:
        args.lora_adapter = found
    use_lora = peft_util.requested(args)
    student, processor, family = load_model(
        args.model, args.device,
        torch.bfloat16 if use_lora else torch.float32,
        adapter=args.lora_adapter or None, trainable=True)
    if use_lora and not args.lora_adapter:
        student, note = peft_util.apply(
            student, family, args, trainable=True, trainable_dtype=torch.float32)
        student.to(args.device)
    else:
        note = None
    if use_lora and args.lora_adapter:
        peft_util.lift_trainable(student, torch.float32)
        note = (f"lora: continuing {args.lora_adapter}; "
                f"{peft_util.trainable_note(student)}")

    # A distinct no-grad target is what makes the teacher stable during a
    # step.  It starts as the same policy and tracks it with the released
    # implementation's 0.01 Polyak update after every optimizer step.
    # Preserve the student's master dtypes.  In the LoRA case that means a
    # bf16 frozen base plus fp32 adapter values; casting the copy wholesale to
    # bf16 makes a 0.01 EMA update smaller than one ulp and the teacher stops
    # tracking anything.  A full fine-tune is fp32 on both sides.
    teacher = ema_teacher(student)

    coder = Coder(processor, chat_template=family.read_chat_template(),
                  spec=family)
    specs = load_episode_specs(args.data)
    sampler = build_sampler(student, coder, args, task, base=base,
                            adapter=args.lora_adapter, trainable=True)
    trainable = peft_util.trainable_parameters(student)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    pool = ThreadPoolExecutor(max_workers=args.workers)
    rng = random.Random(args.seed)

    ui = console.Reporter(
        trainer="sdft", run=args.run or os.path.basename(out.rstrip("/")),
        total=args.steps,
        metrics=args.metrics or os.path.join(out, "metrics.jsonl"),
        config={"task": task.name, "task file": args.task,
                "model": model_banner(args.model, args.lora_adapter),
                "data": args.data, "demonstrations": args.demonstrations,
                "demonstration match": args.demonstration_match,
                "rollouts": "unhinted student / HuggingFace",
                "loss": "exact per-token forward KL",
                "teacher update": args.teacher_mixup_alpha,
                "device": args.device, "out": out})
    ui.start()
    if note:
        ui.note(note)

    for step in range(1, args.steps + 1):
        started = time.time()
        sampler.eval()
        picked = [dict(specs[rng.randrange(len(specs))], group=index)
                  for index in range(args.prompts)]
        # This line is the defining on-policy property: no static or dynamic
        # demonstration/hint is in the policy's generation context.
        rollouts = play(sampler, coder, picked, task, None, args, pool,
                        turn_hint="off", turn_hint_p=0.0)
        rollout_seconds = time.time() - started
        metrics = {
            "reward": sum(item.reward for item in rollouts) / len(rollouts),
            "success": sum(float(item.success) for item in rollouts)
                       / len(rollouts),
            "turns": sum(item.turns for item in rollouts) / len(rollouts),
        }
        selected = demonstrations_for(
            picked, demonstrations, rng, args.demonstration_match)
        views = build_views(coder, rollouts, selected, cfg)
        student.train()
        optimizer.zero_grad(set_to_none=True)
        loss = sdft_step(student, teacher, coder, views, cfg, args.device,
                         metrics)
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        update_teacher(teacher, student, cfg.teacher_mixup_alpha)
        metrics.update({"loss": loss, "grad_norm": float(gradient_norm),
                        "sec_rollout": round(rollout_seconds, 1),
                        "sec_total": round(time.time() - started, 1)})
        ui.step(step, metrics)
        if args.save_every and step % args.save_every == 0:
            ui.note(f"checkpoint {save(student, processor, out, f'step_{step}', base)}")

    final = save(student, processor, out, "final", base)
    ui.finish({"checkpoint": final,
               "kind": "lora adapter" if use_lora else "full checkpoint"})


if __name__ == "__main__":
    main()
