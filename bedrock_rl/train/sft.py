"""Vision-language SFT for Netherite tool-call trajectories.

This compact trainer uses Transformers directly so its OpenAI-chat masking,
multimodal collation, LoRA artifacts, and checkpoint format are shared with
the repository's standalone distillation trainers. The pinned verl release
also has a multimodal SFT trainer; this is an intentionally smaller local
path, not a compatibility workaround. Loss lands on response tokens only and
checkpoints save with save_pretrained for direct vLLM loading.

  python -m bedrock_rl.train.sft --data <parquet> --out <dir> \
      [--model qwen3-vl] [--epochs 2] [--lr LR] [--freeze-vision]

  torchrun --standalone --nproc_per_node=2 -m bedrock_rl.train.sft \
      --data ... --out ... --deepspeed z3

  python -m bedrock_rl.train.sft --data ... --out ... --lora-rank 32

--lora-rank trains an adapter instead of the whole model and writes an
adapter directory rather than a checkpoint, which is the difference
between SFT that needs an 80 GB card and SFT that does not. The
output names its own base model, so `bin/rl.sh` takes it as MODEL and
continues the same adapter under RL. bedrock_rl/train/peft_util.py holds
the shared LoRA wiring.

--deepspeed takes a shipped ZeRO preset from
bedrock_rl/templates/deepspeed (z2, z2_offload, z3, z3_offload) or a
path to your own config, and needs a torchrun launch. It shards the
optimizer, and at ZeRO-3 the parameters, which is what lets a model
larger than one GPU to train. It applies to this stage only,
because RL runs on verl, which shards with FSDP and has no DeepSpeed
backend.

--model takes anything bedrock_rl/models.py resolves. The model class,
the chat template, the vision-tower parameter names, the attention
implementation and the image band all come from that entry, so this
script has no per-family branches.

Two data shapes are accepted. `question`, `answer`, `images` is one turn
with one frame. `messages`, `images` is a multi-turn trajectory with a
frame per observation, and loss lands on assistant turns only.
"""
import argparse
import json
import os
import time

import torch
import transformers
from datasets import load_dataset
from transformers import (Trainer, TrainerCallback,
                          TrainingArguments)

from bedrock_rl import lora, models as reg, reporting as console, resources
from bedrock_rl.train import peft_util
from bedrock_rl.train.sft_data import LazyEncodedDataset, validate_columns

DS_DIR = str(resources.template_path("deepspeed"))


class ReporterCallback(TrainerCallback):
    """transformers' own progress is a tqdm bar and a printed dict, and
    under tmux with stdout tee'd to a file that bar is a page of carriage
    returns per step. This routes the same numbers through the repo's one
    reporter instead: the metrics file gets transformers' keys unchanged,
    the console gets one aligned line."""

    def __init__(self, ui):
        self.ui = ui
        self.t = time.time()

    def on_train_begin(self, args, state, control, **kw):
        del control
        self.ui.set_total(int(state.max_steps or 0))
        self.t = time.time()

    def on_log(self, args, state, control, logs=None, **kw):
        del control
        if not logs:
            return
        now = time.time()
        if "loss" not in logs:
            # the end-of-training summary transformers logs once
            self.ui.record(dict(logs))
            return
        m = dict(logs)
        if "learning_rate" in m:
            m["lr"] = m["learning_rate"]
        m["sec_total"] = round(now - self.t, 2)
        self.t = now
        self.ui.step(int(state.global_step), m)


def ds_presets():
    if not os.path.isdir(DS_DIR):
        return []
    return sorted(f[3:-5] for f in os.listdir(DS_DIR)
                  if f.startswith("ds_") and f.endswith(".json"))


def resolve_deepspeed(value):
    """Shipped preset name or a config path -> path to a DeepSpeed JSON."""
    if value is None:
        return None
    preset = os.path.join(DS_DIR, f"ds_{value}.json")
    if os.path.isfile(preset):
        return preset
    if os.path.isfile(value):
        return value
    raise SystemExit(
        f"--deepspeed {value!r} is neither a shipped preset nor a readable "
        f"file. Presets are {', '.join(ds_presets())} in "
        f"bedrock_rl/templates/deepspeed.")


def check_deepspeed_launch(cfg):
    """DeepSpeed is optional here, so demand it explicitly once it is asked for.

    It also cannot run outside a distributed launch, and finding that out
    after a large model has loaded wastes minutes, so check both up front.
    """
    try:
        import deepspeed                                  # noqa: F401
    except ImportError:
        raise SystemExit(
            "--deepspeed needs the deepspeed package, which this repo keeps "
            "optional because only SFT uses it. Install it into "
            "the training venv with\n"
            "  INSTALL_DEEPSPEED=1 bash bin/setup_deps.sh\n"
            "or directly with\n"
            '  uv pip install --python "$VERL_ENV/bin/python" deepspeed')
    if "LOCAL_RANK" not in os.environ:
        raise SystemExit(
            f"--deepspeed {os.path.basename(cfg)} needs a distributed launch "
            f"and this process was started without one. bin/sft.sh "
            f"switches to torchrun for you, or launch it yourself with\n"
            f"  torchrun --standalone --nproc_per_node=<gpus> "
            f"python -m bedrock_rl.train.sft ...")

# prompt parity with RL rollouts: render the SAME tools block the agent
# loop renders, or the SFT imprint lands on a different prompt and does
# not transfer at all. One implementation, in bedrock_rl/adapters/netherite/chat.py,
# and BOTH data shapes below go through it. Coder supplies the schema
# itself, so this file no longer keeps a second handle on it.
from bedrock_rl.adapters.netherite.chat import (                          # noqa: E402
    Coder, prepare_sft_multiturn,
)

IGNORE = -100


# ── what makes a checkpoint resumable ────────────────────────────────────
# A checkpoint directory is written file by file over seconds to minutes.
# A run killed in the middle of that leaves one that LOOKS like a
# checkpoint: it has the name, the step number and some of the contents.
# `--resume auto` picks the highest step number, which is exactly the one
# most likely to be the partial write, and Trainer then fails somewhere
# inside a load with a message about a missing key -- or, worse, resumes
# from weights with no optimizer state and silently restarts the moment
# schedule.
#
# So "complete" is defined here rather than discovered there. The names
# are transformers' own (TRAINER_STATE_NAME, OPTIMIZER_NAME, the weight
# names) spelled out rather than imported, because they have moved
# modules between releases and a checkpoint written by one release is
# read by the next; a wrong import would break the check itself.
STATE_FILE = "trainer_state.json"
# Any ONE of these is a complete set of weights. The sharded forms name
# their pieces in an index, and a partial write is precisely the case
# where the index exists and a shard does not, so the index is FOLLOWED
# rather than counted.
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin",
                "adapter_model.safetensors", "adapter_model.bin")
WEIGHT_INDEXES = ("model.safetensors.index.json",
                  "pytorch_model.bin.index.json")
# Weights alone make a checkpoint loadable, not resumable: without
# optimizer state a resume restarts Adam's moments and the LR schedule,
# which is a different run wearing the same step number. DeepSpeed writes
# its optimizer partitions into a `global_step<N>` directory instead of
# one file, so either shape counts.
OPTIMIZER_FILE = "optimizer.pt"


def checkpoint_problem(path):
    """Why this checkpoint directory cannot be resumed from, or None.

    A sentence rather than a boolean, because every caller has to be able
    to say WHICH part of the write did not finish.
    """
    if not os.path.isdir(path):
        return "it is not a directory"
    have = set(os.listdir(path))
    if STATE_FILE not in have:
        return (f"no {STATE_FILE}, so the step, the schedule and the "
                f"metric history the resume needs are not there")
    weights = [f for f in WEIGHT_FILES if f in have]
    indexes = [f for f in WEIGHT_INDEXES if f in have]
    if not weights and not indexes:
        return ("no model weights: none of "
                + ", ".join(WEIGHT_FILES + WEIGHT_INDEXES))
    for index in indexes:
        try:
            with open(os.path.join(path, index)) as f:
                shards = set(json.load(f).get("weight_map", {}).values())
        except (OSError, ValueError) as e:
            return f"{index} is not readable as an index ({e})"
        gone = sorted(s for s in shards if s not in have)
        if gone:
            return (f"{index} names {len(shards)} shard(s) and "
                    f"{len(gone)} of them were never written, first "
                    f"{gone[0]!r}")
    if OPTIMIZER_FILE not in have and not any(
            d.startswith("global_step") and os.path.isdir(
                os.path.join(path, d)) for d in have):
        return (f"no {OPTIMIZER_FILE} and no DeepSpeed global_step "
                f"directory, so a resume would restart the optimizer and "
                f"the learning-rate schedule from scratch")
    return None


def checkpoints_under(out):
    """(step, path) for every `checkpoint-<step>` directory under `out`,
    lowest step first. Ordered by STEP NUMBER and not by mtime, because a
    partial write from a killed process is the most recent thing on
    disk."""
    if not os.path.isdir(out):
        return []
    return sorted(
        (int(d.split("-")[-1]), os.path.join(out, d))
        for d in os.listdir(out)
        if d.startswith("checkpoint-") and d.split("-")[-1].isdigit())


def newest_resumable(out, note=None):
    """The highest-numbered COMPLETE checkpoint under `out`, or None.

    Incomplete ones are skipped and every skip is announced. Skipping
    rather than refusing is what `save_total_limit=2` is for: the newest
    is the one that can be a partial write, so the previous one is kept
    precisely so a resume has somewhere to land. Silence would turn
    "resumed 200 steps back" into something a reader finds out from a
    step counter.
    """
    for step, path in reversed(checkpoints_under(out)):
        why = checkpoint_problem(path)
        if why is None:
            return path
        if note:
            note(f"skipping checkpoint-{step}: {why}")
    return None


def build_example(coder, question, answer, image):
    """One `question`/`answer` row with one frame, as a one-turn
    trajectory through the SAME Coder everything else renders with.

    It used to render its own prompt: strip the `<image>` marker out of
    the question, hand the remainder to the processor's content-parts
    path, and let the template put the vision span back. That path drops
    the newline the marker was followed by, and this file's header names
    the consequence: an SFT imprint on a prompt the rollout never
    renders does not transfer. Measured on Qwen/Qwen3-VL-4B-Instruct
    with a real processor, the two renders differ by exactly that byte
    (7715 vs 7716 chars, first difference right after
    `<|vision_start|><|image_pad|><|vision_end|>`), which is one token
    of 1887 and shifts every token after it.

    So a single-turn row is now a two-message trajectory and
    `build_multiturn_example` renders it. `Coder` is what the RL rollout,
    the SDPO trainer and self-distillation harvester render with, and it expands
    `<image>` IN PLACE, so the surrounding bytes survive. Returns None on
    the same condition the multi-turn path does.
    """
    return build_multiturn_example(
        coder,
        [{"role": "user", "content": question},
         {"role": "assistant", "content": answer}],
        [image])


def build_multiturn_example(coder, messages, images,
                            last_assistant_only=False,
                            keep_latest_images=0):
    """One whole trajectory. The hint is already gone from `messages`,
    and the label mask covers assistant tokens only, so no privileged
    text ever sits in a loss-carrying position.

    The Collator below reads pixel_values and image_grid_thw straight out
    of what this returns; Coder.encode_images is what refuses a family
    that cannot produce them, once, for every trainer."""
    messages, images = prepare_sft_multiturn(
        messages, images, last_assistant_only=last_assistant_only,
        keep_latest_images=keep_latest_images)
    pv, grid, npads = coder.encode_images(images)
    selected = None
    if last_assistant_only:
        selected = [next(
            index for index in range(len(messages) - 1, -1, -1)
            if messages[index].get("role") == "assistant")]
    ids, mask = coder.encode_trajectory(
        messages, npads, assistant_indices=selected)
    if ids is None:
        return None
    input_ids = torch.tensor(ids)
    labels = torch.where(torch.tensor(mask).bool(), input_ids,
                         torch.full_like(input_ids, IGNORE))
    return {"input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels,
            "pixel_values": pv,
            "image_grid_thw": grid}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        maxlen = max(f["input_ids"].shape[0] for f in feats)
        ids, att, lab = [], [], []
        for f in feats:
            n = maxlen - f["input_ids"].shape[0]
            ids.append(torch.nn.functional.pad(f["input_ids"], (0, n),
                                               value=self.pad_id))
            att.append(torch.nn.functional.pad(f["attention_mask"], (0, n)))
            lab.append(torch.nn.functional.pad(f["labels"], (0, n),
                                               value=IGNORE))
        return {"input_ids": torch.stack(ids),
                "attention_mask": torch.stack(att),
                "labels": torch.stack(lab),
                "pixel_values": torch.cat([f["pixel_values"]
                                           for f in feats]),
                # one row per image. Every example is a trajectory now, so
                # the grid arrives (n_frames, 3) and a single-turn row is
                # (1, 3); the unsqueeze holds for a (3,) an older
                # producer might still hand over.
                "image_grid_thw": torch.cat(
                    [f["image_grid_thw"] if f["image_grid_thw"].dim() == 2
                     else f["image_grid_thw"].unsqueeze(0) for f in feats])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="qwen3-vl",
                    help="registry alias, HuggingFace id, or local path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=-1)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=None,
                    help="learning rate (default: 1e-6 for full tuning, "
                         "1e-5 for LoRA)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--keep-latest-images", type=int, default=0,
                    help="newest causal frames retained per trajectory; "
                         "0 keeps all (default: 0, matching RL and eval)")
    ap.add_argument("--save-every", type=int, default=0,
                    help="write a resumable checkpoint every N optimizer "
                         "steps, and keep the two most recent. 0 keeps the "
                         "old behaviour of saving only at the end, which "
                         "loses the whole run if it is interrupted")
    ap.add_argument("--resume", default=None,
                    help="resume from a checkpoint directory, or `auto` to "
                         "pick up the newest one under --out")
    ap.add_argument("--attn-impl", default=None)
    ap.add_argument("--freeze-vision", action="store_true")
    ap.add_argument("--deepspeed", default=None,
                    help="ZeRO preset from "
                         "bedrock_rl/templates/deepspeed (%s) or a "
                         "config path; needs torchrun"
                         % ", ".join(ds_presets()))
    ap.add_argument("--run", "--run-name", dest="run", default=None,
                    help="run name for the header and for wandb; defaults "
                         "to the output directory's name")
    ap.add_argument("--metrics", default=None,
                    help="per-step metrics, one JSON object per line; "
                         "defaults to metrics.jsonl under --out")
    peft_util.add_args(ap)
    a = ap.parse_args()
    if a.keep_latest_images < 0:
        ap.error("--keep-latest-images cannot be negative")

    ds_config = resolve_deepspeed(a.deepspeed)
    if ds_config is not None:
        check_deepspeed_launch(ds_config)

    spec = reg.resolve(a.model)
    # --model may itself be an adapter directory, in which case the
    # weights to load are its base and the adapter is what gets continued.
    path, named_adapter = reg.policy_paths(a.model)
    if named_adapter and not a.lora_adapter:
        a.lora_adapter = named_adapter
    use_lora = peft_util.requested(a)
    if a.lr is None:
        a.lr = 1e-5 if use_lora else 1e-6
    attn = reg.attn_impl(spec, a.attn_impl)
    os.makedirs(a.out, exist_ok=True)
    # Reporter is silent on every rank but zero, which is what the old
    # log() helper was for.
    ui = console.Reporter(
        trainer="sft", run=a.run or os.path.basename(a.out.rstrip("/")),
        total=a.steps if a.steps > 0 else None,
        metrics=a.metrics or os.path.join(a.out, "metrics.jsonl"),
        config={"model": reg.banner(a.model, attn, a.lora_adapter),
                "data": a.data, "epochs": a.epochs, "lr": a.lr,
                "batch": f"{a.batch} x {a.grad_accum} accum",
                "image window": (a.keep_latest_images or "all"),
                "tuning": lora.describe(*peft_util.config_of(a, spec),
                                        adapter=a.lora_adapter)
                          if use_lora else "full fine-tune",
                "vision tower": "frozen" if a.freeze_vision else "trained",
                "out": a.out})
    ui.start()
    log = ui.note

    # TrainingArguments has to exist BEFORE from_pretrained. Building it with
    # deepspeed= is what registers the config that transformers checks in
    # is_deepspeed_zero3_enabled, and under ZeRO-3 that check is what makes
    # from_pretrained shard parameters as it loads them. Construct the model
    # first and every rank materializes the whole thing, which defeats ZeRO-3
    # and puts a large model straight into OOM.
    args = TrainingArguments(
        output_dir=a.out, per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum,
        num_train_epochs=a.epochs,
        max_steps=a.steps, learning_rate=a.lr, bf16=True,
        logging_steps=1, report_to=[],
        # An interrupted run used to lose everything, because the only
        # write was trainer.save_model after trainer.train returned. A
        # SFT killed at step 25 of 45 left an empty directory and
        # threw away twenty-five steps. --save-every writes optimizer
        # state as well as weights, so --resume continues rather than
        # restarts; two are kept because the newest can be a partial
        # write if the process dies mid-save.
        save_strategy=("steps" if a.save_every > 0 else "no"),
        save_steps=(a.save_every if a.save_every > 0 else 500),
        save_total_limit=2,
        remove_unused_columns=False, dataloader_num_workers=0,
        # Every trainable adapter/language parameter participates in this
        # ordinary causal-LM forward. DDP's unused-parameter graph walk was
        # pure overhead on every long multimodal sequence (and PyTorch said
        # so on every rank of the measured six-GPU run).
        ddp_find_unused_parameters=False,
        # the bar is a page of carriage returns in a tee'd log, and
        # ReporterCallback prints the same numbers as one line
        disable_tqdm=True,
        deepspeed=ds_config)
    if ds_config is not None:
        log(f"deepspeed: {ds_config} across {args.world_size} rank(s)")

    console.quiet_transformers_repeats()
    processor = reg.load_processor(
        spec, lora.processor_path(path, a.lora_adapter))
    # the family's tool-call template, so the SFT prompt is byte-identical
    # to the one verl renders at rollout time
    custom = spec.read_chat_template()
    if custom is not None:
        processor.chat_template = custom
        log(f"chat template: bedrock_rl/templates/chat_template/"
            f"{spec.chat_template}")
    quant = peft_util.quantization_config(a) if use_lora else None
    if getattr(a, "lora_4bit", False) and not use_lora:
        raise SystemExit(
            "--lora-4bit quantizes the frozen base an adapter sits on, so it "
            "needs --lora-rank as well. On its own it would quantize the "
            "weights this run is about to train.")
    weight_dtype = torch.bfloat16 if use_lora else torch.float32
    model = reg.load_weights(spec, path, weight_dtype, attn,
                             quantization_config=quant)
    model.config.use_cache = False
    if a.freeze_vision:
        n_frozen = 0
        for name, p in model.named_parameters():
            if any(pat in name for pat in spec.vision_patterns):
                p.requires_grad = False
                n_frozen += 1
        log(f"froze {n_frozen} vision-tower params "
            f"(patterns {list(spec.vision_patterns)})")
        if n_frozen == 0:
            raise SystemExit(
                f"--freeze-vision matched nothing: registry patterns "
                f"{list(spec.vision_patterns)} do not appear in this "
                f"checkpoint's parameter names")
    if use_lora:
        # After the freeze, so the freeze still counts real parameter
        # names rather than peft's rewritten ones, and so its check that
        # the registry patterns match something still runs.
        model, note = peft_util.apply(model, spec, a)
        log(note)
        if a.freeze_vision:
            log("--freeze-vision adds nothing under LoRA: peft freezes "
                "every base parameter, and the tower carries no adapter "
                "either")

    raw = load_dataset("parquet", data_files=a.data)["train"]
    # One Coder for both data shapes: a single-turn row is a one-turn
    # trajectory, and rendering it any other way puts the imprint on a
    # prompt the rollout never produces.
    coder = Coder(processor, chat_template=custom, spec=spec)
    validate_columns(raw.column_names)
    if "messages" in raw.column_names:
        def encode(row):
            return build_multiturn_example(
                coder, json.loads(row["messages"]), row["images"],
                bool(row.get("loss_last_assistant_only", False)),
                a.keep_latest_images)
    else:
        def encode(row):
            return build_example(coder, row["question"], row["answer"],
                                 row["images"][0])
    ds = LazyEncodedDataset(raw, encode)
    if not ds:
        ui.finish(failed=True)
        raise SystemExit("no usable training examples")
    log(f"{len(ds)} examples (tensorized lazily by the data loader)")

    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      processing_class=processor,
                      data_collator=Collator(processor.tokenizer.pad_token_id),
                      callbacks=[ReporterCallback(ui)])
    # transformers keeps one of these two whatever disable_tqdm says, and
    # PrinterCallback is the thing that printed {'loss': 0.7982,
    # 'grad_norm': 122.0, ...} under the step line. ReporterCallback says
    # the same in one aligned line and writes the record.
    for cb in (transformers.trainer_callback.PrinterCallback,
               transformers.trainer_callback.ProgressCallback):
        trainer.remove_callback(cb)
    resume = a.resume
    if resume == "auto":
        # Trainer writes checkpoint-<step> under output_dir. Highest step
        # first, and every one that did not finish being written is
        # skipped out loud (checkpoint_problem says which part is
        # missing) rather than handed to Trainer to fail on.
        resume = newest_resumable(a.out, ui.warn)
        log(f"resuming from {resume}" if resume
            else "nothing complete to resume from, starting fresh")
    elif resume:
        # A path the caller NAMED is refused rather than skipped. Quietly
        # starting from scratch because the directory they asked for was
        # half-written is the one outcome they cannot see in the logs
        # until the step counter is back at zero.
        why = checkpoint_problem(resume)
        if why is not None:
            ui.finish(failed=True)
            raise SystemExit(
                f"--resume {resume} is not a complete checkpoint: {why}. "
                f"Point --resume at a finished one, or use `--resume auto` "
                f"to take the newest complete checkpoint under --out.")
    trainer.train(resume_from_checkpoint=resume)
    model.config.use_cache = True
    # ZeRO-3 leaves every parameter sharded across ranks, so model.save_pretrained
    # writes tensors of length zero. Trainer.save_model is the sharding-aware
    # path; it gathers the 16-bit weights first, which is what
    # stage3_gather_16bit_weights_on_model_save in the presets turns on. It
    # writes config.json too, so the registry can still resolve the result by
    # model_type. On one process it does exactly what save_pretrained did.
    trainer.save_model(a.out)
    if trainer.is_world_process_zero():
        processor.save_pretrained(a.out)
        if use_lora:
            # save_model wrote the adapter, which is a few megabytes and
            # not a model. It has to name the weights it sits on top of,
            # or the next stage has nothing to load it onto; peft's own
            # guess is the path transformers happened to load from, which
            # is a hub cache directory as often as not.
            peft_util.set_adapter_base(a.out, path)
    ui.finish({"checkpoint": a.out, "examples": len(ds),
               "kind": "lora adapter" if use_lora else "full checkpoint"})


if __name__ == "__main__":
    main()
