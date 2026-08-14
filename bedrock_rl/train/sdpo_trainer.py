"""SDPO trainer: a standalone multi-turn RL loop for Bedrock RL.

This is a peer of the verl trainer (bin/rl.sh), not a mode of it. It owns
its loss and its optimizer, shares the rollout driver in
bedrock_rl/train/rollout.py and the hint machinery in
bedrock_rl/sdg/guidance.py, and reads the algorithm from
bedrock_rl/train/sdpo.py, which also records why the upstream trainer
cannot be driven directly.

Per step:
  1. Draw episode specs and expand each into a group of `group-size`
     rollouts, the group structure SDPO needs to find a successful
     sibling.
  2. Play every rollout multi-turn from the unhinted student policy and
     score its live episode. State-dependent feedback is recorded from the
     live world but is not shown while the student acts.
  3. Build two token views over identical response tokens. The teacher
     view carries the hint and a successful sibling's tool calls. The
     student view carries neither and is byte for byte the prompt a
     hint-free deployment renders.
  4. Distill the teacher's next-token distribution into the student and
     take one gradient step per generation batch.

Gradients only ever reach the student, so the trained policy is hint-free
and hint-free evaluation measures the artifact.

  python -m bedrock_rl.train.sdpo_trainer \
      --task examples/equip_pickaxe_grpo/task.yaml \
      --data data/equip_pickaxe_grpo_cu/rl_prompts.parquet --steps 30
  python -m bedrock_rl.train.sdpo_trainer --task ... --data ... \
      --model <ckpt> --eval-only --hint-level none
  python -m bedrock_rl.train.sdpo_trainer --task ... --data ... \
      --lora-rank 32

--lora-rank trains an adapter instead of the whole model. It keeps the
frozen base in bf16 and optimizer state only for the fp32 adapter.
"""
import argparse
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from bedrock_rl.sdg import load_episode_specs
from bedrock_rl import lora, reporting as console
from bedrock_rl.sdg import guidance as hints
from bedrock_rl.adapters.netherite.chat import Coder
from bedrock_rl.train import peft_util
from bedrock_rl.train.rollout import (add_sampler_args, build_sampler,
                                        check_engine, load_coder, load_model,
                                        model_banner, play)
from bedrock_rl.train.sdpo import (SDPOConfig, demo_for,
                                     self_distillation_loss)
from bedrock_rl.train.sdft import ema_teacher, update_teacher
from bedrock_rl.env.task import Task


def check_hint_exact(coder, task, hint):
    """Prove on real rendered prompts that removing the hint message
    reproduces the hint-free prompt exactly and that nothing of the hint
    survives it. Runs once per run, before any gradient is taken.

    The proof is `hints.assert_hint_removable`, which asks the question
    of THIS model family's chat template rather than assuming one
    family's system-message markup. The family key goes in so a template
    that cannot carry the channel is refused by name.

    This is about the STATIC channel. The per-turn channel cannot be
    checked this way, because a suffix that changes every turn is not one
    block; it gets its exactness from being recorded verbatim as it is
    written and removed by that record (hints.without_turn_hints), and
    its safety from never sitting in a loss-carrying position."""
    body = [{"role": "user", "content": "probe"}]
    spec = getattr(coder, "spec", None)
    return hints.assert_hint_removable(
        lambda msgs: coder.render(msgs, []), body, hint,
        family=getattr(spec, "key", None))


def build_views(coder, rollouts, hint, cfg):
    """One (student, teacher) token view per sampled turn over identical
    response token IDs.

    Rollouts are already the unhinted student's own trajectories. Recorded
    state-dependent feedback and the static guide are added only to the
    teacher copy after generation, so both views still run over identical
    assistant tokens."""
    demos = [r.response_text() for r in rollouts]
    groups = [r.group for r in rollouts]
    oks = [r.success for r in rollouts]
    views = []
    for i, r in enumerate(rollouts):
        demo = demo_for(i, groups, oks, demos, cfg) if cfg.teacher_demo else None
        sys_txt = (hint or "") if cfg.teacher_hint else ""
        if demo:
            sys_txt += cfg.demo_template.format(demo=demo.strip())
        for turn in r.sampled_turns:
            teacher_body = (list(turn.teacher_messages)
                            if cfg.teacher_turn_hint
                            else list(turn.messages))
            turn_feedback = (
                teacher_body != list(turn.messages)
                if cfg.teacher_turn_hint
                else False)
            # A rollout is active exactly when its teacher view has private
            # information. The static guide is itself such a channel; gating
            # it on a successful sibling made static-hint-only runs silently
            # take zero gradient for every all-failure group.
            static_guidance = bool(cfg.teacher_hint and hint)
            distil = bool(static_guidance or (cfg.teacher_demo and demo)
                          or turn_feedback)
            tmsgs = hints.with_hint(teacher_body, sys_txt.strip() or None)
            view = coder.sampled_view(
                turn.prompt_ids, turn.response_ids, tmsgs,
                turn.image_pads)
            if view is not None:
                view.update({"images": list(turn.images), "distil": distil,
                             "trajectory": i,
                             "token_count": len(turn.response_ids),
                             "sampling_log_probs": (
                                 list(turn.sampling_log_probs)
                                 if turn.sampling_log_probs else None)})
            views.append(view)
    totals = {}
    for view in views:
        if view is not None and view["distil"]:
            key = view["trajectory"]
            totals[key] = totals.get(key, 0) + view["token_count"]
    for view in views:
        if view is not None:
            total = totals.get(view["trajectory"], 0)
            view["trajectory_weight"] = (
                view["token_count"] / total
                if view["distil"] and total else 0.0)
    return views


def resp_logprobs(model, coder, ids, mask, images, device, topk=None,
                  topk_indices=None):
    """Log-probs of the masked (assistant) tokens from one forward pass,
    plus their top-k. Positions shift by one: the logits at j-1 predict
    the token at j. When topk_indices is given the log-probs are gathered
    there instead of at this model's own top-k, which is how upstream puts
    teacher and student on the same support."""
    pv, grid, _ = coder.encode_images(images)
    input_ids = torch.tensor([ids], device=device)
    kw = {"input_ids": input_ids,
          "attention_mask": torch.ones_like(input_ids)}
    if pv is not None:
        kw["pixel_values"] = pv.to(device, model.dtype)
        kw["image_grid_thw"] = grid.to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**kw).logits[0]
    pos = torch.tensor([j - 1 for j, m in enumerate(mask) if m and j > 0],
                       device=device)
    tgt = torch.tensor([ids[j] for j, m in enumerate(mask) if m and j > 0],
                       device=device)
    logp = torch.log_softmax(logits[pos].float(), dim=-1)
    sampled = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    if topk_indices is not None:
        return sampled, logp.gather(-1, topk_indices), topk_indices
    if topk is None:
        return sampled, None, None
    top = logp.topk(topk, dim=-1)
    return sampled, top.values, top.indices


def sdpo_step(model, teacher, coder, views, cfg, device, metrics):
    """One gradient step per generation batch, accumulated over
    single-trajectory micro batches. The vLLM path uses the sampled-token
    log-probabilities returned by its serving engine for upstream's
    importance correction. Hugging Face samples and trains through the same
    module, so its old log-probability is the detached recomputation."""
    live = [v for v in views if v is not None]
    usable_trajectories = {v["trajectory"] for v in live}
    distilled_trajectories = {
        v["trajectory"] for v in live if v["trajectory_weight"] > 0}
    n_distil = len(distilled_trajectories)
    metrics["sdpo/usable_views"] = len(live) / max(len(views), 1)
    metrics["sdpo/mask_fraction"] = (
        n_distil / max(len(usable_trajectories), 1))
    metrics["sdpo/distilled"] = n_distil
    if n_distil == 0:
        # No rollout received private teacher context, so there is no
        # distillation target and no backward pass. The caller counts these
        # steps and stops the run rather than letting them pass unseen.
        return 0.0
    topk = cfg.distillation_topk if cfg.full_logit_distillation else None
    total = 0.0
    token_total = 0.0
    for v in live:
        if v["trajectory_weight"] <= 0:
            continue
        s_ids, s_mask = v["student"]
        s_lp, s_topk, s_idx = resp_logprobs(model, coder, s_ids, s_mask,
                                            v["images"], device, topk=topk)
        t_ids, t_mask = v["teacher"]
        # The released method's default teacher regularization is an EMA
        # reference policy.  It sees the privileged context, while the live
        # student sees the deployment context. SDPO defines the top-k support
        # under the student, so the teacher is gathered at those same indices.
        with torch.no_grad():
            t_lp, t_topk, _ = resp_logprobs(
                teacher, coder, t_ids, t_mask, v["images"], device,
                topk_indices=s_idx)
        old_lp = s_lp.detach()
        if v["sampling_log_probs"] is not None:
            old_lp = torch.tensor(v["sampling_log_probs"], device=device)
            if old_lp.shape != s_lp.shape:
                raise ValueError(
                    "sampled and recomputed log-probabilities do not align")
        rmask = torch.ones(1, s_lp.shape[0], device=device)
        loss, m = self_distillation_loss(
            s_lp.unsqueeze(0), t_lp.unsqueeze(0), rmask, cfg,
            old_log_probs=old_lp.unsqueeze(0),
            student_topk_log_probs=None if s_topk is None else s_topk.unsqueeze(0),
            teacher_topk_log_probs=None if t_topk is None else t_topk.unsqueeze(0),
            self_distillation_mask=torch.ones(1, device=device))
        scale = v["trajectory_weight"] / n_distil
        (loss * scale).backward()
        total += float(loss.detach()) * scale
        for key in ("sdpo/per_token_loss",
                    "sdpo/teacher_minus_student_logp"):
            metrics[key] = metrics.get(key, 0.0) + m[key] * scale
        token_total += m["sdpo/distilled_tokens"]
    metrics["sdpo/distilled_tokens"] = token_total / n_distil
    return total


def batch_specs(specs, rng, n_prompts, group_size):
    out = []
    for g in range(n_prompts):
        s = specs[rng.randrange(len(specs))]
        for _ in range(group_size):
            out.append(dict(s, group=g))
    return out


def evaluate(sampler, coder, specs, task, hint, args, pool, n,
             turn_hint="off"):
    """Paired evaluation: the same RNG picks the same episodes whatever
    the hint level, so hinted and hint-free numbers compare directly.

    `turn_hint` is passed EXPLICITLY rather than inherited from `args`,
    because a run whose rollouts are hinted must still be able to measure
    a hint-free number, and inheriting it would have made the "none"
    half of the paired eval carry a per-turn hint and report a hinted
    number under a hint-free name. The DEFAULT is "off" and not None for
    the same reason: None means "inherit from args" down in
    rollout.play, so a caller that forgets the argument would reintroduce
    exactly the failure this docstring says is designed out."""
    rng = random.Random(args.seed + 12345)
    picked = [dict(specs[rng.randrange(len(specs))], group=i)
              for i in range(n)]
    # p=1.0, not the run's p: an eval that mixed hinted and un-hinted
    # episodes would report a mixture under the rung's name
    rolls = play(sampler, coder, picked, task, hint, args, pool,
                 turn_hint=turn_hint, turn_hint_p=1.0)
    return {"reward": sum(r.reward for r in rolls) / len(rolls),
            "success": sum(1.0 for r in rolls if r.success) / len(rolls),
            "stopped": sum(1.0 for r in rolls if r.stopped) / len(rolls),
            "n": len(rolls)}


def dry_run_message(task, args):
    """What to say when SDPO has taken no gradient for several steps.

    The loss only reaches a rollout whose teacher view actually contains a
    private static guide, per-turn feedback, or a successful sibling."""
    return (
        f"\nstopping: {args.max_dry_steps} consecutive steps on "
        f"{task.name} produced no rollout to distil, so no gradient was "
        "taken.\n\n"
        "No sampled teacher view contained private information: a static "
        "guide, state-dependent per-turn feedback, or a successful sibling "
        "demonstration. Choose one of these.\n"
        "  enable a task system_hint with --hint-level procedure\n"
        "  enable state feedback with --turn-hint direction or oracle\n"
        "  run SFT first and pass that checkpoint as --model\n"
        "    bash bin/sft.sh " + task.name + "\n"
        "  or pass --max-dry-steps 0 to run anyway\n")


def add_common_args(ap):
    ap.add_argument("--task", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="qwen3-vl",
                    help="registry alias, HuggingFace id, or local path")
    ap.add_argument("--hint-level", default="procedure",
                    choices=list(hints.LEVELS))
    ap.add_argument("--finish-hint", default="instruct",
                    choices=list(hints.FINISH_LEVELS))
    hints.add_turn_hint_args(ap)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--netherite-home", default=None)
    ap.add_argument("--frames-root", default="/tmp/netherite_sd_frames")
    ap.add_argument("--run", default=None,
                    help="run name for the header and for wandb; defaults "
                         "to the output directory's name")
    # The console is for a person and the metrics file is for everything
    # else. --log is the name this flag had when the console WAS the
    # record, and it still works.
    ap.add_argument("--metrics", "--log", dest="metrics", default=None,
                    help="per-step metrics, one JSON object per line; "
                         "defaults to metrics.jsonl under --out")
    add_sampler_args(ap)


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--prompts", type=int, default=8, help="prompts per step")
    ap.add_argument("--group-size", type=int, default=4,
                    help="rollouts per prompt; SDPO needs a group to find a "
                         "successful sibling")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--is-clip", type=float, default=2.0)
    ap.add_argument("--teacher-update-rate", type=float, default=0.05,
                    help="EMA update rate for the privileged self-teacher")
    ap.add_argument("--no-teacher-demo", action="store_true")
    ap.add_argument("--no-teacher-hint", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--eval-episodes", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--max-dry-steps", type=int, default=5,
                    help="stop after this many consecutive steps with no "
                         "distillable rollout; 0 never stops")
    peft_util.add_args(ap)
    args = ap.parse_args()
    if args.lora_4bit:
        raise SystemExit(
            "trainer: sdpo does not support --lora-4bit: its EMA teacher "
            "path is not implemented for a quantized base. Use ordinary "
            "LoRA here, or trainer: sft for QLoRA.")

    task = Task.load(args.task)
    hint = hints.hint_text(task, args.hint_level, args.finish_hint)
    cfg = SDPOConfig(
        alpha=args.alpha, distillation_topk=args.topk, is_clip=args.is_clip,
        teacher_update_rate=args.teacher_update_rate,
        teacher_demo=not args.no_teacher_demo,
        # Static and state-dependent hints are independent privilege
        # channels. ``--hint-level none`` disables only the former; the
        # latter remains available to a turn-hint-only SDPO run.
        teacher_hint=not args.no_teacher_hint and hint is not None,
        teacher_turn_hint=args.turn_hint != "off")
    out = args.out or os.path.join(os.path.expanduser("~"), "ckpts",
                                   f"netherite_{task.name}_sdpo")
    os.makedirs(out, exist_ok=True)
    from bedrock_rl import models as reg
    base, found = reg.policy_paths(args.model)
    if found and not args.lora_adapter:
        args.lora_adapter = found
    use_lora = peft_util.requested(args)
    # Before the run header, and a long way before the weights: whether
    # the engine asked for is importable and whether this run's flags can
    # keep it on-policy are both knowable now, and finding out after a
    # 35 GiB fp32 load is finding out by running out of memory.
    engine = check_engine(args, trainable=not args.eval_only)
    ui = console.Reporter(
        trainer="sdpo", run=args.run or os.path.basename(out.rstrip("/")),
        total=None if args.eval_only else args.steps,
        metrics=args.metrics or os.path.join(
            out, "eval.jsonl" if args.eval_only else "metrics.jsonl"),
        config={"task": task.name, "task file": args.task,
                "model": model_banner(args.model, args.lora_adapter),
                "data": args.data,
                "hint": f"{args.hint_level} (finish {args.finish_hint})",
                "turn hint": (f"{args.turn_hint} (p={args.turn_hint_p:g} "
                              f"per episode)" if args.turn_hint != "off"
                              else "off"),
                "teacher": f"EMA self-teacher ({args.teacher_update_rate:g})",
                "tuning": "full fine-tune" if not use_lora else lora.describe(
                    *peft_util.config_of(args, reg.resolve(args.model)),
                    adapter=args.lora_adapter),
                "device": args.device, "out": out})
    ui.start()

    specs = load_episode_specs(args.data)
    # An evaluation on the vLLM engine never forwards through this
    # module: the engine holds the same weights and samples from those.
    # Loading them here as well is a whole checkpoint read and, for an
    # 8B, about 18 GiB held for the life of the run, to obtain a
    # processor and a family spec that need neither.
    model = None
    if args.eval_only and engine == "vllm":
        proc, spec, _base, _adapter = load_coder(args.model,
                                                 args.lora_adapter)
    else:
        # Full fine-tuning keeps fp32 master weights across the whole
        # model. LoRA keeps the frozen base in bf16 and lifts only the
        # adapter to fp32, which is where the memory saving in this
        # trainer comes from.
        model, proc, spec = load_model(
            args.model, args.device,
            torch.bfloat16 if (args.eval_only or use_lora) else torch.float32,
            adapter=args.lora_adapter if args.lora_adapter else None,
            trainable=not args.eval_only)
        if use_lora and not args.lora_adapter:
            model, note = peft_util.apply(
                model, spec, args, trainable=not args.eval_only,
                trainable_dtype=None if args.eval_only else torch.float32)
            model.to(args.device)
            ui.note(note)
        elif use_lora:
            peft_util.lift_trainable(model, torch.float32)
            ui.note(f"lora: continuing {args.lora_adapter}; "
                    f"{peft_util.trainable_note(model)}")
    coder = Coder(proc, chat_template=spec.read_chat_template(), spec=spec)
    pool = ThreadPoolExecutor(max_workers=args.workers)

    if args.eval_only:
        sampler = build_sampler(model, coder, args, task, base=base,
                                adapter=args.lora_adapter, trainable=False,
                                say=ui.note)
        res = {"hint_level": args.hint_level,
               "turn_hint": args.turn_hint,
               **evaluate(sampler, coder, specs, task, hint, args, pool,
                          args.eval_episodes, turn_hint=args.turn_hint)}
        ui.record(res)
        ui.finish({k: v for k, v in res.items()})
        return

    if hint is not None:
        check_hint_exact(coder, task, hint)

    sampler = build_sampler(model, coder, args, task, base=base,
                            adapter=args.lora_adapter, trainable=True,
                            say=ui.note)
    teacher = ema_teacher(model)
    # Under LoRA the sampler IS the policy, so it has to be put back in
    # eval mode before each batch of rollouts; sdpo_step below leaves it
    # in train mode. On the HF path the sampler is the model itself; the
    # vLLM sampler answers it as a no-op so
    # that it is substitutable for the module here.
    trained = peft_util.trainable_parameters(model)
    opt = torch.optim.AdamW(trained, lr=args.lr, weight_decay=0.0)
    rng = random.Random(args.seed)
    dry = 0                       # consecutive steps with nothing to distil
    for step in range(1, args.steps + 1):
        t0 = time.time()
        sampler.eval()
        rolls = play(
            sampler, coder,
            batch_specs(specs, rng, args.prompts, args.group_size),
            task, None, args, pool, turn_hint=args.turn_hint,
            turn_hint_p=args.turn_hint_p, turn_hint_visible=False)
        t_roll = time.time() - t0
        metrics = {"reward": sum(r.reward for r in rolls) / len(rolls),
                   "success": sum(1.0 for r in rolls if r.success) / len(rolls),
                   "stopped": sum(1.0 for r in rolls if r.stopped) / len(rolls),
                   "turns": sum(r.turns for r in rolls) / len(rolls)}
        model.train()
        views = build_views(coder, rolls, hint, cfg)
        opt.zero_grad(set_to_none=True)
        loss = sdpo_step(model, teacher, coder, views, cfg, args.device,
                         metrics)
        gnorm = torch.nn.utils.clip_grad_norm_(trained, 1.0)
        opt.step()
        if cfg.teacher_update_rate:
            update_teacher(teacher, model, cfg.teacher_update_rate)
        metrics.update({"loss": loss, "grad_norm": float(gnorm),
                        "sec_rollout": round(t_roll, 1),
                        "sec_total": round(time.time() - t0, 1)})
        ui.step(step, metrics)
        # A step with nothing to distil took no gradient at all. One or two
        # of those early is normal; a run of them means the model never
        # reaches a success on this task, so every remaining step would
        # burn rollout for a zero update.
        dry = dry + 1 if metrics.get("sdpo/distilled", 0) == 0 else 0
        if args.max_dry_steps and dry >= args.max_dry_steps:
            ui.record({"step": step, "error": "no_distillable_rollouts",
                       "dry_steps": dry})
            ui.error(f"{dry} consecutive steps with no rollout to distil, "
                     f"so no gradient was taken")
            ui.finish(failed=True)
            raise SystemExit(dry_run_message(task, args))
        if args.eval_every and step % args.eval_every == 0:
            sampler.eval()
            # Explicit (static, per-turn) PAIRS, not a static level with
            # the per-turn rung inferred from it. Inferring it with
            # `level == args.hint_level` was wrong at --hint-level none,
            # where the loop is ("none", "none"), both iterations match,
            # and the half recorded as hint-free carried the per-turn
            # hint. --hint-level none is the natural config for a
            # turn-hint-only ablation, so it is exactly the case that would
            # have been mis-reported.
            for level, th in ((args.hint_level, args.turn_hint),
                              ("none", "off")):
                h = hints.hint_text(task, level, args.finish_hint)
                res = evaluate(sampler, coder, specs, task, h, args, pool,
                               args.eval_episodes, turn_hint=th)
                ui.record({"step": step, "eval_hint": level,
                           "eval_turn_hint": th, **res})
                ui.note(f"eval hint={level} turn_hint={th} "
                        f"reward={res['reward']:+.3f} "
                        f"success={res['success'] * 100:.1f}% n={res['n']}")
        if args.save_every and step % args.save_every == 0:
            ui.note(f"checkpoint {save(model, proc, out, f'step_{step}', base)}")
    # The per-step adapters the vLLM sampler stages are scratch, and the
    # checkpoint here is the one that is meant to outlive the run. The
    # sampler registers its own atexit, so the exits that are not this
    # one -- --max-dry-steps, a StalePolicyError, a crash -- clean up too.
    final = save(model, proc, out, "final", base)
    ui.finish({"checkpoint": final,
               "kind": "lora adapter" if use_lora else "full checkpoint"})


def save(model, proc, out, name, base):
    """One checkpoint. Under LoRA save_pretrained writes the adapter
    rather than the model, so it also has to record the weights that
    adapter belongs on top of."""
    d = os.path.join(out, name)
    if getattr(model, "peft_config", None):
        peft_util.save_adapter(model, d, base, proc)
    else:
        model.save_pretrained(d)
        proc.save_pretrained(d)
    return d


if __name__ == "__main__":
    main()
