"""Turn a verl LoRA checkpoint into an adapter something else can load.

A verl checkpoint is FSDP shards of a peft model, one `.pt` per rank,
plus a `huggingface/` directory holding a config and a processor and no
weights at all. Nothing outside verl reads that. So a LoRA RL run that
stopped there would leave an artifact no eval path, no other trainer and
no vLLM could open, which is the same as leaving nothing.

verl already knows how to reassemble it. `verl.model_merger` walks the
shards, lifts the `lora_A`/`lora_B` tensors out into
`lora_adapter/adapter_model.safetensors` with an `adapter_config.json`
beside them, and writes the remaining weights as an ordinary
HuggingFace model. This module drives that and then fixes the one thing
it leaves out.

**The base weights in the merged directory are the base model.** LoRA
never writes into them. So loading the merged directory on its own
evaluates the untrained model and looks like a run that learned
nothing, which is exactly the kind of quiet wrong answer this repo tries
not to produce. The adapter beside it is the trained artifact.

**verl's `adapter_config.json` does not say what it sits on top of.** It
carries the rank, the alpha and the target modules and nothing else,
because the merger builds it from a bare `peft.LoraConfig`. This module
writes `base_model_name_or_path` in, so the result is a self-describing
adapter directory like the one SFT writes, and one rule in
bedrock_rl/lora.py covers both.

  python -m bedrock_rl.train.export_adapter \
      --checkpoints ~/ckpts/bedrock_<task>_grpo_cu_<model> \
      --base Qwen/Qwen3-VL-2B-Instruct
"""
import argparse
import glob
import os
import re
import subprocess
import sys

from bedrock_rl import lora

MERGED = "merged"


def step_dirs(root):
    """verl's `global_step_N/actor` directories, oldest first."""
    out = []
    for path in glob.glob(os.path.join(root, "global_step_*")):
        m = re.search(r"global_step_(\d+)$", path)
        actor = os.path.join(path, "actor")
        if m and os.path.isdir(actor):
            out.append((int(m.group(1)), actor))
    return [p for _, p in sorted(out)]


def merge(actor_dir, target_dir):
    """verl's own merger, in this interpreter's environment."""
    argv = [sys.executable, "-m", "verl.model_merger", "merge",
            "--backend", "fsdp",
            "--local_dir", actor_dir, "--target_dir", target_dir]
    print("+ " + " ".join(argv))
    return subprocess.call(argv)


def export(checkpoints, base, step=None):
    """The newest checkpoint under `checkpoints`, as an adapter directory.

    Returns the adapter path, or None when there is nothing to export.
    A run that crashed before its first save is the common case for that
    and is not an error here: the trainer already said what happened.
    """
    steps = step_dirs(checkpoints)
    if not steps:
        print(f"no verl checkpoint under {checkpoints}, so no adapter was "
              f"written. A LoRA run saves on trainer.save_freq; SAVE_FREQ "
              f"names it.")
        return None
    actor = steps[-1] if step is None else next(
        (p for p in steps if p.endswith(os.path.join(
            f"global_step_{step}", "actor"))), None)
    if actor is None:
        raise SystemExit(f"no global_step_{step} under {checkpoints}")
    target = os.path.join(os.path.dirname(actor), MERGED)
    rc = merge(actor, target)
    if rc != 0:
        raise SystemExit(f"verl.model_merger exited {rc} on {actor}")
    adapter = os.path.join(target, lora.MERGED_ADAPTER)
    if not lora.is_adapter_dir(adapter):
        raise SystemExit(
            f"{actor} merged, but no adapter came out of it. That means the "
            f"checkpoint holds no lora_A/lora_B tensors, so the run was a "
            f"full fine-tune however it was launched.")
    from bedrock_rl.train import peft_util
    peft_util.set_adapter_base(adapter, base)
    print(f"adapter  {adapter}")
    print(f"base     {base}")
    print(f"evaluate it with\n"
          f"  bedrock eval trials --model {target}")
    return adapter


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoints", required=True,
                    help="the trainer's output directory, which holds "
                         "global_step_N/actor")
    ap.add_argument("--base", required=True,
                    help="the base model the adapter was trained on")
    ap.add_argument("--step", type=int, default=None,
                    help="a specific step; the newest one by default")
    a = ap.parse_args()
    export(a.checkpoints, a.base, a.step)


if __name__ == "__main__":
    main()
