"""Self-distillation harvester: hinted successful trajectories for SFT.

Cloning a scripted oracle is not the only way to initialize a policy, and
on this substrate it is not the best one. A sibling study measured
golden-oracle SFT coming in WORSE than the untrained model, while
self-distillation beat oracle demonstrations by 12.7 points. The
mechanism is Zelikman et al.'s rationalisation. Reveal enough for the
model to produce a correct trajectory, then train on that trajectory with
the revelation deleted from the input.

This module does the harvest half.

  1. Sample trajectories from the CURRENT model with the hint present, at
     a chosen level, through bedrock_rl/train/rollout.py.
  2. Keep only the ones whose live engine episode reached success. No
     self-labeling, model judge, or decoded-text replay.
  3. Delete every hint by exact removal (bedrock_rl/sdg/guidance.py), so
     the training input is the prompt a hint-free deployment renders.
     Both channels come out: the static system block, and the per-turn
     observation suffixes a `--turn-hint` rung wrote, each of which was
     recorded verbatim as it was written.
  4. Write one causal conversation prefix per assistant action, selecting
     that action's observation window and applying loss only to its response.

The yield is reported per hint level, because a low or empty yield is a
measurement of task difficulty and of how much the hint is carrying, not
a failure to hide.

  python -m bedrock_rl.train.self_distill \
      --task examples/equip_pickaxe_grpo/task.yaml \
      --data data/equip_pickaxe_grpo_cu/rl_prompts.parquet \
      --episodes 256 --hint-level procedure
"""
import argparse
import json
import math
import os
import random
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from bedrock_rl.sdg import load_episode_specs
from bedrock_rl import reporting as console
from bedrock_rl.data import png_bytes, write_rows
from bedrock_rl.sdg import guidance as hints
from bedrock_rl.adapters.netherite.chat import Coder, sft_action_prefix
from bedrock_rl.train import peft_util
from bedrock_rl.train.rollout import (add_sampler_args, build_sampler,
                                        check_engine, load_coder, load_model,
                                        model_banner, play,
                                        specs_from_task)
from bedrock_rl.env.task import Task


def harvest_rows(rollouts, hint):
    """Successful trajectories with EVERY hint deleted, as SFT rows.

    `hints.strip` removes both channels, the static system block and the
    per-turn observation suffixes, in the one order that works. The
    per-turn ones are never in a loss-carrying position to begin with,
    but leaving them in the row would still train the model to condition
    on a sentence a hint-free deployment does not render, so they come
    out of the INPUT too and the row is byte for byte the hint-free
    prompt."""
    rows, failed = [], []
    for r in rollouts:
        if not r.success:
            continue
        try:
            # The completed episode's model_view keeps terminal frames. Start
            # from the full exact history and window each supervised action.
            msgs = hints.strip(r.exact_messages(), hint, r.hint_marks)
            images = [{"bytes": png_bytes(im), "path": None}
                      for im in r.images]
            keep_images = int(os.environ.get("BRL_KEEP_LATEST_IMAGES", "0"))
            targets = [index for index, message in enumerate(msgs)
                       if message.get("role") == "assistant"]
            if not targets:
                raise ValueError("successful rollout has no assistant action")
            prefixes = [sft_action_prefix(
                msgs, images, target, keep_latest_images=keep_images)
                for target in targets]
        except ValueError as e:
            # A row whose hints cannot be removed EXACTLY must not ship,
            # and that is not in question. Dropping it rather than
            # raising is, because raising here throws away every episode
            # of an already-spent batch for one bad trajectory. Dropping
            # cannot leak: the row is gone. The count is reported.
            failed.append(f"seed {r.spec.get('seed')}: {e}")
            continue
        for messages, causal_images in prefixes:
            rows.append({
                "messages": json.dumps(messages),
                "images": causal_images,
                "loss_last_assistant_only": True,
                "reward": float(r.reward),
                "seed": int(r.spec["seed"]),
                "turns": int(r.turns),
                # Keep the episode's hint population after removing its text.
                "turn_hinted": bool(r.turn_hinted)})
    return rows, failed


def write_receipt(path, summary):
    """Atomically acknowledge only this invocation's completed harvest."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=parent, prefix=".harvest-receipt-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump({**summary, "out": os.path.abspath(summary["out"])},
                      stream)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--data", default=None,
                    help="rl parquet for episode specs; without it the "
                         "seeds and start poses come from the task YAML")
    ap.add_argument("--model", default="qwen3-vl",
                    help="registry alias, HuggingFace id, or local path")
    ap.add_argument("--out", default=None)
    ap.add_argument("--receipt", default=None,
                    help="write a JSON receipt for this invocation; an empty "
                         "harvest records examples=0 and preserves old data")
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
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
                         "to the harvest file's name")
    ap.add_argument("--metrics", default=None,
                    help="per-batch harvest metrics, one JSON object per "
                         "line; defaults to the harvest file's name with a "
                         ".jsonl suffix")
    # This half takes no gradient, so it takes only the adapter. A rank
    # is a question the harvester cannot answer, and the fine-tune half
    # that can is bedrock_rl/train/sft.py, which takes the whole set.
    peft_util.add_args(ap, train=False)
    add_sampler_args(ap)
    args = ap.parse_args()

    task = Task.load(args.task)
    # Before the run header and a long way before the weights: an engine
    # this environment cannot import is worth saying so about here rather
    # than after a full checkpoint read.
    engine = check_engine(args, trainable=False)
    hint = hints.hint_text(task, args.hint_level, args.finish_hint)
    # The rung is a PATH input, not only a flag, for the reason the hint
    # level already is: two arms of the same ablation that write one file
    # overwrite each other silently, and the whole point of the dial is
    # running `off` against `direction` at the same static level.
    rung = "" if args.turn_hint == "off" else f"_turn{args.turn_hint}"
    out = args.out or os.path.join(
        "data", f"{task.name}_cu",
        f"self_distill_{args.hint_level}{rung}.parquet")
    batches = max(1, math.ceil(args.episodes / max(args.batch, 1)))
    ui = console.Reporter(
        trainer="self_distill", total=batches,
        run=args.run or os.path.splitext(os.path.basename(out))[0],
        metrics=args.metrics or os.path.splitext(out)[0] + ".jsonl",
        config={"task": task.name, "task file": args.task,
                "model": model_banner(args.model, args.lora_adapter),
                "hint": f"{args.hint_level} (finish {args.finish_hint})",
                "turn hint": (f"{args.turn_hint} (p={args.turn_hint_p:g} "
                              f"per episode)" if args.turn_hint != "off"
                              else "off"),
                "episodes": f"{args.episodes} in batches of {args.batch}",
                "device": args.device, "out": out})
    ui.start()

    rng0 = random.Random(args.seed)
    specs = (load_episode_specs(args.data) if args.data
             else specs_from_task(task, max(args.episodes, 32), rng0,
                                  args.netherite_home))
    # The harvest takes no gradient, so on the vLLM engine this module is
    # never forwarded through: the engine holds the same weights and
    # samples from those. Loading them here too is a whole checkpoint
    # read and, for an 8B, about 18 GiB held for the run, to obtain a
    # processor and a family spec that need neither.
    model, base, adapter = None, None, args.lora_adapter
    if engine == "vllm":
        proc, spec, base, adapter = load_coder(args.model, args.lora_adapter)
    else:
        model, proc, spec = load_model(args.model, args.device,
                                       torch.bfloat16,
                                       adapter=args.lora_adapter)
        model.eval()
    coder = Coder(proc, chat_template=spec.read_chat_template(), spec=spec)
    sampler = build_sampler(model, coder, args, task, base=base,
                            adapter=adapter, trainable=False, say=ui.note)
    pool = ThreadPoolExecutor(max_workers=args.workers)
    rng = random.Random(args.seed)

    rows, n_seen, n_stop, batch, n_kept = [], 0, 0, 0, 0
    n_hinted, n_dropped = 0, 0
    while n_seen < args.episodes:
        t0 = time.time()
        k = min(args.batch, args.episodes - n_seen)
        picked = [dict(specs[rng.randrange(len(specs))], group=i)
                  for i in range(k)]
        # the sampler, not the module: on the vLLM engine `model` is None
        # here, because the harvest takes no gradient and never forwards it
        rolls = play(sampler, coder, picked, task, hint, args, pool)
        kept, failed = harvest_rows(rolls, hint)
        rows += kept
        n_kept += sum(bool(r.success) for r in rolls) - len(failed)
        n_seen += k
        n_stop += sum(1 for r in rolls if r.stopped)
        n_hinted += sum(1 for r in rolls if r.turn_hinted)
        for why in failed:
            # loud, because it means a recorded hint was not where the
            # record said and that is a bug rather than a measurement
            ui.warn(f"dropped a solved trajectory: its per-turn hints "
                    f"could not be removed exactly ({why})")
        n_dropped += len(failed)
        batch += 1
        ui.step(batch, {"seen": n_seen, "kept": n_kept,
                        "examples": len(rows),
                        "voluntary_stop": n_stop,
                        "turn_hinted": n_hinted,
                        "strip_failed": n_dropped,
                        "yield": n_kept / max(n_seen, 1),
                        "sec_total": round(time.time() - t0, 1)})

    summary = {"task": task.name, "hint_level": args.hint_level,
               "finish_hint": args.finish_hint,
               "turn_hint": args.turn_hint, "turn_hint_p": args.turn_hint_p,
               "episodes": n_seen,
               "kept": n_kept, "examples": len(rows),
               "yield": round(n_kept / max(n_seen, 1), 4),
               "voluntary_stop": round(n_stop / max(n_seen, 1), 4),
               "turn_hinted": round(n_hinted / max(n_seen, 1), 4),
               "strip_failed": n_dropped,
               "out": out}
    if rows:
        write_rows(rows, out)
    else:
        summary["note"] = ("empty harvest; the model never solved the task "
                           "under this hint level")
        ui.warn("empty harvest; the model never solved the task under this "
                "hint level, so no training set was written")
    ui.record(summary)
    if args.receipt:
        write_receipt(args.receipt, summary)
    # An empty harvest is a measurement of the task and of what the hint
    # is carrying, not a crash, so the summary does not claim the run
    # failed. The warning above says what happened.
    ui.finish({"harvest": out if rows else "(nothing written)"})


if __name__ == "__main__":
    main()
