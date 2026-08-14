"""LoRA for the three trainers this repo writes itself.

`bin/rl.sh` needs none of this: verl builds its own peft model from
hydra settings. SFT and the two self-distillation
trainers are hand-written, so they wire peft themselves, and they do it
through this module so that "LoRA" means one thing across all four
entry points.

The policy that does not need torch, which is the defaults, the
vision-tower exclusion and how an adapter directory names its base, is in
bedrock_rl/lora.py and is shared with the registry and the CLI.

Two decisions are worth naming here.

**The base loads in bf16 and only the adapter is fp32.** That is where
the memory goes: a full fine-tune holds bf16 weights, bf16 gradients and
two fp32 Adam moments for every parameter. With LoRA the frozen base
keeps only its weights and the optimizer state covers a fraction of a
percent of the parameters. `trainable_dtype` is what the SDPO trainer
uses to keep its master weights in fp32 without paying for them on the
whole model.

**Attaching an existing adapter and creating a new one are one call.**
`apply` dispatches on whether an adapter path was given, because every
caller wants the same thing: a model that is training a LoRA adapter,
whether that adapter is new or came from an earlier SFT run.
"""
import json
import os

from bedrock_rl import lora


def add_args(ap, train=True):
    """The LoRA flags, identical in every trainer that takes them.

    `train=False` is for the self-distillation harvester, which samples from a policy
    and takes no gradient: an adapter is something it can be pointed at,
    and a rank is not a question it can answer.
    """
    if train:
        ap.add_argument("--lora-rank", type=int, default=0,
                        help="train a LoRA adapter of this rank instead of "
                             "a full fine-tune; 0 is a full fine-tune")
        ap.add_argument("--lora-alpha", type=int, default=None,
                        help="LoRA alpha; defaults to the rank, which is "
                             "scaling 1.0")
        ap.add_argument("--lora-targets", default=lora.DEFAULT_TARGETS,
                        help="all-linear, or a comma-separated list of "
                             "module name suffixes")
        ap.add_argument("--lora-exclude", default=None,
                        help="regex of modules to keep adapter-free; "
                             "defaults to the family's own vision tower, "
                             "which the rollout engine would drop anyway")
    ap.add_argument("--lora-adapter", default=None,
                    help="an existing adapter directory to continue from")
    if train:
        ap.add_argument("--lora-4bit", action="store_true",
                        help="quantize the frozen base to 4-bit NF4 (QLoRA). "
                             "Saves memory and costs time. Available to SFT; "
                             "the EMA SDFT/SDPO and verl paths reject it")
    return ap


def quantization_config(a):
    """The BitsAndBytesConfig for --lora-4bit, or None.

    NF4 with double quantization and a bf16 compute dtype, which is the
    combination QLoRA measured and the one peft's own helpers assume.
    The adapter stays in fp32 on top; only the frozen base is quantized,
    so what is saved is the base weights and nothing else.
    """
    if not getattr(a, "lora_4bit", False):
        return None
    import torch
    try:
        import bitsandbytes                                 # noqa: F401
    except ImportError:
        raise SystemExit(
            "--lora-4bit needs bitsandbytes, which this repo keeps optional "
            "because only the trainers outside verl can use it. Install it "
            "into the training venv with\n"
            '  uv pip install --python "$VERL_ENV/bin/python" bitsandbytes')
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16)


def requested(a):
    """Whether the flags on this namespace ask for LoRA at all."""
    return lora.enabled(getattr(a, "lora_rank", 0),
                        getattr(a, "lora_adapter", None))


def config_of(a, spec):
    """(rank, alpha, targets, exclude) for a fresh adapter, off the flags."""
    return lora.settings(spec,
                         rank=getattr(a, "lora_rank", None),
                         alpha=getattr(a, "lora_alpha", None),
                         targets=getattr(a, "lora_targets", None),
                         exclude=getattr(a, "lora_exclude", None))


def attach(model, adapter, trainable):
    """Load an existing adapter onto a base model."""
    from peft import PeftModel
    if trainable:
        model.enable_input_require_grads()
    return PeftModel.from_pretrained(model, str(adapter),
                                     is_trainable=bool(trainable))


def create(model, spec, rank, alpha, targets, exclude):
    """Wrap a base model in a new adapter.

    Every setting comes from bedrock_rl/lora.py, which is where they
    can be checked without a training venv; this only turns the task
    type into peft's enum and calls peft.
    """
    from peft import LoraConfig, TaskType, get_peft_model
    # Gradient checkpointing recomputes activations, and with every
    # parameter feeding it frozen there is no input to the first block
    # that carries grad_fn, so the recomputed graph is empty and the
    # backward pass reaches nothing. This is the hook that fixes it, and
    # verl calls it in the same place for the same reason.
    model.enable_input_require_grads()
    kwargs = lora.config_kwargs(rank, alpha, targets, exclude)
    kwargs["task_type"] = TaskType(kwargs["task_type"])
    return get_peft_model(model, LoraConfig(**kwargs))


def apply(model, spec, a, trainable=True, trainable_dtype=None):
    """Put this model on the LoRA path the flags describe.

    Returns (model, note), where `note` is the one line every trainer
    prints: what was wrapped, and how much of the model is now trainable.
    A caller that got here has already checked `requested`.
    """
    adapter = getattr(a, "lora_adapter", None)
    if getattr(a, "lora_4bit", False):
        # Casts the layer norms and the head back to fp32 and turns on
        # input gradients, without which a quantized frozen base gives
        # the adapter nothing to attach a graph to.
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=False)
    if adapter:
        model = attach(model, adapter, trainable)
        what = f"continuing adapter {adapter}"
    else:
        rank, alpha, targets, exclude = config_of(a, spec)
        model = create(model, spec, rank, alpha, targets, exclude)
        what = lora.describe(rank, alpha, targets, exclude)
    lora.assert_no_trainable_vision(model, spec)
    if trainable_dtype is not None:
        # Master weights for the adapter only. An lr of 1e-4 on a bf16
        # weight is representable, but the SDPO trainer runs at 1e-6,
        # which is below a bf16 ulp and would round away to nothing.
        lift_trainable(model, trainable_dtype)
    if getattr(a, "lora_4bit", False):
        what += ", frozen base in 4-bit NF4"
    return model, f"lora: {what}; {trainable_note(model)}"


def trainable_note(model):
    """"N trainable of M parameters (x%)", the number that says whether
    the wrap did what it was asked to."""
    total = sum(p.numel() for p in model.parameters())
    live = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * live / max(total, 1)
    return f"{live:,} trainable of {total:,} ({pct:.3f}%)"


def trainable_parameters(model):
    """What the optimizer and the gradient clipper should see.

    Handing AdamW every parameter of a peft model works by accident,
    because a frozen parameter never gets a gradient and never gets
    optimizer state. It also makes the parameter count in a run header
    wrong and the clip norm cover tensors that carry no gradient, so the
    trainers ask for this instead.
    """
    return [p for p in model.parameters() if p.requires_grad]


def lift_trainable(model, dtype):
    """Keep optimizer/EMA master values representable at small learning rates."""
    for parameter in trainable_parameters(model):
        parameter.data = parameter.data.to(dtype)
    return model


def save_adapter(model, out, base, processor=None):
    """Write the adapter, and make it say what it was trained on.

    peft fills `base_model_name_or_path` from the loaded model's
    `_name_or_path`, which is right when the base came from the hub and
    is a local snapshot directory when it did not. Writing the path the
    registry resolved keeps the chain from SFT into
    `bin/rl.sh` working, because that is the field the launcher reads to
    find the weights the adapter goes on top of.
    """
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    if processor is not None:
        processor.save_pretrained(out)
    return set_adapter_base(out, base)


def set_adapter_base(out, base):
    """Point an adapter directory's config at the base it belongs to."""
    path = os.path.join(str(out), lora.ADAPTER_CONFIG)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        conf = json.load(f)
    conf["base_model_name_or_path"] = str(base)
    with open(path, "w") as f:
        json.dump(conf, f, indent=2, sort_keys=True)
    return path
