"""Framework-independent LoRA configuration and adapter discovery.

This module defines defaults, vision-tower exclusions, and adapter-directory
resolution without importing Torch or PEFT. Training-specific PEFT utilities
live in :mod:`bedrock_rl.train.peft_util`.

Vision modules are excluded because vLLM serves language-model LoRA weights but
does not apply adapters to the multimodal connector or vision tower. Exclusion
patterns are derived from each model family's ``vision_patterns`` declaration.
PEFT's ``adapter_config.json`` identifies an adapter's base model, allowing a
later training stage to resolve the correct family and weights automatically.
"""
import json
import os
import re

# Rank 32 with alpha equal to the rank is scaling 1.0, which is the
# setting that does not need explaining alongside a learning rate. It is a
# starting point and not a measured optimum; nothing here claims a quality
# result for LoRA. Alpha has no constant of its own on purpose: a caller
# who changes the rank and not the alpha has changed the scaling without
# meaning to, so an unset alpha follows the rank.
DEFAULT_RANK = 32
# peft's own name for "every linear layer except the output head". Passed
# through as a string; anything else is a comma-separated list of module
# name suffixes.
ALL_LINEAR = "all-linear"
DEFAULT_TARGETS = ALL_LINEAR

# vLLM sizes its LoRA slots from a fixed ladder and refuses anything
# else, so a rank that is not on it is rounded UP to the next one. The
# ladder is vllm.config.lora.MaxLoRARanks; verl rounds the same way in
# verl/workers/rollout/vllm_rollout/utils.py::get_vllm_max_lora_rank.
VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)

ADAPTER_CONFIG = "adapter_config.json"
# What verl's model merger names the adapter it writes beside a merged
# checkpoint (verl/model_merger/base_model_merger.py::save_lora_adapter).
MERGED_ADAPTER = "lora_adapter"


def enabled(rank, adapter=None):
    """Is LoRA on. Rank alone decides for a fresh adapter; continuing an
    existing one is LoRA whatever the rank says."""
    if adapter:
        return True
    try:
        return int(rank or 0) > 0
    except (TypeError, ValueError):
        return False


def exclude_regex(spec):
    """The `exclude_modules` regex for a family.

    Two sources, joined. The vision tower comes from the family's own
    `vision_patterns`, so a family that names its tower correctly gets
    its LoRA exclusion for free. `lora_exclude_extra` is for anything
    else a family must not adapt, and today that is one thing: the
    router of a mixture of experts.

    peft matches a string `exclude_modules` with `re.fullmatch` against
    the module's dotted name, so each alternative has to cover a whole
    name. The registry's vision patterns are substrings with cosmetic
    dots on the ends (".visual.", "visual."), which collapse to one
    alternative here.

    Returns None for a family that declares neither, which is not a case
    the registry has today; peft warns loudly when an exclusion matches
    nothing, so a wrong pattern is not silent.
    """
    seen = []
    for pattern in getattr(spec, "vision_patterns", ()) or ():
        token = str(pattern).strip(".")
        if token and token not in seen:
            seen.append(token)
    alts = [".*" + re.escape(t) + ".*" for t in seen]
    alts += list(getattr(spec, "lora_exclude_extra", ()) or ())
    if not alts:
        return None
    return "|".join(alts)


def vllm_max_rank(rank):
    """The LoRA slot size vLLM has to be told to reserve for this rank."""
    rank = int(rank)
    for allowed in VLLM_LORA_RANKS:
        if rank <= allowed:
            return allowed
    raise SystemExit(
        f"LoRA rank {rank} is above {VLLM_LORA_RANKS[-1]}, which is the "
        f"largest the rollout engine can serve, so the adapter would train "
        f"and never be sampled from.")


def parse_targets(value):
    """A LORA_TARGETS knob into what peft wants.

    `all-linear` is peft's own keyword and stays a string. Anything else
    is a comma-separated list of module name suffixes, such as
    `q_proj,k_proj,v_proj,o_proj`, and becomes a list.
    """
    if value is None:
        return DEFAULT_TARGETS
    value = str(value).strip()
    if not value or value == ALL_LINEAR:
        return DEFAULT_TARGETS
    return [part.strip() for part in value.split(",") if part.strip()]


# ── adapter directories ──────────────────────────────────────────────────

def read_adapter_config(path):
    """An adapter directory's own config, or None when it is not one."""
    cfg = os.path.join(str(path), ADAPTER_CONFIG)
    if not os.path.isfile(cfg):
        return None
    try:
        with open(cfg) as f:
            conf = json.load(f)
    except (OSError, ValueError):
        return None
    return conf if isinstance(conf, dict) else None


def is_adapter_dir(path):
    return read_adapter_config(path) is not None


def adapter_rank(path):
    """(rank, alpha) an adapter directory was trained at, or (0, 0).

    verl needs the rank on the command line even when it is loading an
    existing adapter, because `lora_rank > 0` is the only thing that
    switches its LoRA path on and because the vLLM engine sizes its LoRA
    slots from it.
    """
    conf = read_adapter_config(path) or {}
    try:
        return int(conf.get("r") or 0), int(conf.get("lora_alpha") or 0)
    except (TypeError, ValueError):
        return 0, 0


def adapter_base(path):
    """The base model an adapter directory was trained against.

    peft writes this itself. verl's model merger does not, so
    bedrock_rl/train/export_adapter.py fills it in for the adapters
    that come out of an RL run; without it an adapter is a file nobody
    can say what to load underneath.
    """
    conf = read_adapter_config(path)
    if conf is None:
        return None
    base = conf.get("base_model_name_or_path")
    return str(base) if base else None


def policy_paths(name):
    """Anything MODEL can be, as (base weights, adapter directory or None).

    Three shapes reach this:

      a HuggingFace id, a registry alias or an ordinary checkpoint
        -> (name, None)
      an adapter directory written by SFT or by SDPO
        -> (its base_model_name_or_path, the directory)
      a merged directory written by an RL run, which holds the base
      weights and a `lora_adapter/` beside them
        -> (the directory, the directory/lora_adapter)

    The third shape is what verl's model merger produces, and its base
    weights are the UNCHANGED base: LoRA never writes into them. Loading
    that directory without its adapter therefore evaluates the base model
    and looks like training that did nothing.
    """
    name = str(name)
    if not os.path.isdir(name):
        return name, None
    merged = os.path.join(name, MERGED_ADAPTER)
    if is_adapter_dir(merged):
        return name, merged
    if is_adapter_dir(name):
        base = adapter_base(name)
        if not base:
            raise SystemExit(
                f"{name} is a LoRA adapter directory whose "
                f"{ADAPTER_CONFIG} does not say which base model it was "
                f"trained against, so there is nothing to load it onto.\n"
                f"  add \"base_model_name_or_path\" to "
                f"{os.path.join(name, ADAPTER_CONFIG)}\n"
                f"  or name the base model and pass the adapter separately "
                f"with LORA_ADAPTER")
        return base, name
    return name, None


def settings(spec, rank=None, alpha=None, targets=None, exclude=None):
    """The four numbers a fresh adapter is built from, defaults filled in.

    Alpha follows the rank rather than a constant of its own, because
    alpha over rank is the scaling: a caller who raises the rank and
    leaves a constant alpha has quietly lowered the scaling without
    touching anything they can see. Exclude falls back to the family's
    own vision tower, which is the setting that keeps a run training
    what the rollout engine serves.
    """
    rank = int(rank or DEFAULT_RANK)
    return (rank, int(alpha or rank), parse_targets(targets),
            exclude or exclude_regex(spec))


def config_kwargs(rank, alpha, targets, exclude):
    """Everything a `peft.LoraConfig` for this repo is built from.

    Kept here, away from peft, so the settings that are correctness
    requirements rather than preferences can be asserted without a
    training venv. Two of them are:

    `bias` must be "none". vLLM refuses an adapter carrying bias terms
    outright (`PEFTHelper.validate_legal`, "Adapter bias is not
    supported"), so an adapter trained with them is one the rollout
    engine will not serve, and the run only finds out at engine start.

    `lora_dropout` is zero. The SDPO trainer samples from the policy it
    is training rather than from a separate bf16 shadow, and dropout
    would make the sampled policy differ from the trained one in a way
    nothing downstream accounts for.
    """
    return {"task_type": "CAUSAL_LM", "r": int(rank),
            "lora_alpha": int(alpha), "lora_dropout": 0.0, "bias": "none",
            "target_modules": targets, "exclude_modules": exclude}


def assert_no_trainable_vision(model, spec):
    """Refuse an adapter that would be partially discarded by vLLM.

    A user-supplied exclusion may be syntactically valid while matching only
    the tower root (``visual``) rather than its linear descendants. PEFT then
    creates trainable vision adapters and vLLM drops them at serving time.
    Checking the wrapped model catches that semantic mismatch immediately.
    """
    patterns = tuple(getattr(spec, "vision_patterns", ()) or ())
    bad = [name for name, parameter in model.named_parameters()
           if parameter.requires_grad
           and any(pattern in name for pattern in patterns)]
    if bad:
        preview = ", ".join(bad[:3])
        more = "" if len(bad) <= 3 else f" (+{len(bad) - 3} more)"
        raise ValueError(
            "LoRA created trainable vision parameters that vLLM cannot "
            f"serve: {preview}{more}. Use the model registry's full vision "
            f"exclusion regex ({exclude_regex(spec)!r}).")


def processor_path(base, adapter=None):
    """Where to load the processor from.

    It rides with a saved policy when possible because image sizing, special
    tokens and the chat template are part of that policy. Transformers writes
    step checkpoints below the run directory, however, and older Netherite
    runs saved the processor only in that parent after training finished.
    Recognize that layout so an intermediate ``checkpoint-N`` can feed eval,
    SDFT or SDPO directly instead of requiring an unrelated tokenizer flag.
    """
    for candidate in (adapter, base):
        if not candidate:
            continue
        candidate = os.path.abspath(os.path.expanduser(str(candidate)))
        if os.path.isfile(os.path.join(candidate,
                                       "preprocessor_config.json")):
            return candidate
        name = os.path.basename(candidate)
        parent = os.path.dirname(candidate)
        if (name.startswith("checkpoint-")
                and name.removeprefix("checkpoint-").isdigit()
                and os.path.isfile(os.path.join(
                    parent, "preprocessor_config.json"))):
            return parent
    return base


def describe(rank, alpha, targets, exclude, adapter=None):
    """One line for a run header. Every trainer prints the same one."""
    if adapter:
        return f"continuing adapter {adapter}"
    return (f"rank {rank}, alpha {alpha}, targets {targets}, "
            f"excluding {exclude or 'nothing'}")
