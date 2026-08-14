"""Supported-model registry: everything that differs between VLMs.

bedrock-rl trains a computer-use agent that looks at rendered game
frames, so only models that take an image input belong here. One entry
per family, one lookup, and everything downstream reads that entry rather
than branching on a model name.

  MODEL=qwen3-vl-4b bash bin/rl.sh
  MODEL=Qwen/Qwen3-VL-8B-Instruct bash bin/rl.sh
  MODEL=/local/path/to/a/checkpoint bash bin/rl.sh

A local directory resolves by the model_type in its config.json, so a
SFT checkpoint lands on the same entry as its base model. Anything
else matches the family key, the size aliases, the HuggingFace ids, then
the id prefixes. An unrecognised name fails before any GPU is touched.

Two fields carry a correctness requirement rather than a preference.
chat_template must render the `computer` tool schema and emit assistant
tool calls in the form tool_parser reads back. image_span is the family's
own image tokens. A template or a span borrowed from another family
trains the model on a prompt or a view it never sees at rollout time, and
nothing raises.

Adding a model is one ModelSpec entry plus a run of bedrock models check.
"""
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from bedrock_rl import lora, resources
from bedrock_rl.core.sampling import (MAX_TOOL_CALLS_PER_TURN,
                                      multimodal_image_limit)

CHAT_TEMPLATE_DIR = str(resources.template_path("chat_template"))

# ChatML tools template: renders the <tools> block, assistant tool_calls as
# <tool_call>{json}</tool_call>, and a tool response as a user turn whose
# image content becomes a real <|vision_start|><|image_pad|><|vision_end|>
# span. Byte for byte what Qwen3-VL's shipped template produces.
QWEN_CHATML_TOOLS = "qwen_chatml_tools.jinja"

# Default image band.
DEFAULT_MIN_PIXELS = 32 * 32
DEFAULT_MAX_PIXELS = 768 * 768


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display: str
    provider: str
    hub: str
    ids: dict
    model_types: tuple
    id_prefixes: tuple = ()
    aliases: tuple = ()
    chat_template: Optional[str] = None
    tool_parser: str = "hermes"
    auto_class: str = "AutoModelForImageTextToText"
    vision_patterns: tuple = (".visual.", "visual.")
    # Extra `exclude_modules` alternatives for a LoRA run, each a regex
    # that must FULL-match a module name. The vision tower is derived
    # from vision_patterns above and does not belong here; this is for
    # anything else a family must not adapt.
    lora_exclude_extra: tuple = ()
    attn_impl: str = "auto"
    vit_attn_backend: Optional[str] = None
    image_span: tuple = ()
    image_min_pixels: int = DEFAULT_MIN_PIXELS
    image_max_pixels: int = DEFAULT_MAX_PIXELS
    max_prompt_length: int = 3072
    # HuggingFace id -> the immutable commit to fetch. A repo id is a
    # MOVING pointer: `snapshot_download(id)` and
    # `from_pretrained(id)` both take whatever main happens to be, so
    # two boxes provisioned a week apart, or the same box before and
    # after an upstream re-upload, can train and evaluate different
    # weights while every log says the same name. Keyed by id rather
    # than by size key so a reader can see which commit belongs to
    # which repo without holding the ids dict in their head.
    #
    # Empty means "not pinned", and a local path is already fixed bytes,
    # so neither is an error. What IS an error is a pin that does not
    # exist: the hub refuses it rather than silently serving main.
    revisions: dict = field(default_factory=dict)
    # Executing the model repo's OWN python at load time. Opt-in at run
    # time as well as here; see remote_code_allowed() below.
    trust_remote_code: bool = False
    verl_overrides: tuple = ()
    verl_override_fn: str = ""
    status: str = "untested"
    notes: str = ""
    # Parameter count in billions, per size key. Optional, because it is
    # only used to answer "will this fit", and a spec that does not say
    # gets no answer rather than a guessed one.
    params_b: dict = field(default_factory=dict)

    @property
    def sizes(self):
        return tuple(self.ids.keys())

    @property
    def default_id(self):
        return next(iter(self.ids.values()))

    def revision(self, ident):
        """The pinned commit for one of this family's ids, or None."""
        return self.revisions.get(str(ident))

    @property
    def chat_template_path(self):
        if self.chat_template is None:
            return None
        return os.path.join(CHAT_TEMPLATE_DIR, self.chat_template)

    def read_chat_template(self):
        p = self.chat_template_path
        if p is None:
            return None
        with open(p, encoding="utf-8") as stream:
            return stream.read()


_QWEN_VISION = (".visual.", "visual.")

# One image, as the family's own chat template writes it: what wraps the
# span, what repeats once per vision pad token, and what closes it.
QWEN_IMAGE_SPAN = ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>")

MODELS = (
    ModelSpec(
        key="qwen3-vl",
        image_span=QWEN_IMAGE_SPAN,
        display="Qwen3-VL",
        provider="Qwen",
        hub="https://huggingface.co/Qwen",
        ids={
            "2B": "Qwen/Qwen3-VL-2B-Instruct",
            "4B": "Qwen/Qwen3-VL-4B-Instruct",
            "8B": "Qwen/Qwen3-VL-8B-Instruct",
            "32B": "Qwen/Qwen3-VL-32B-Instruct",
        },
        revisions={
            "Qwen/Qwen3-VL-2B-Instruct":
                "89644892e4d85e24eaac8bacfd4f463576704203",
            "Qwen/Qwen3-VL-4B-Instruct":
                "ebb281ec70b05090aa6165b016eac8ec08e71b17",
            "Qwen/Qwen3-VL-8B-Instruct":
                "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
            "Qwen/Qwen3-VL-32B-Instruct":
                "0cfaf48183f594c314753d30a4c4974bc75f3ccb",
        },
        model_types=("qwen3_vl",),
        id_prefixes=("Qwen/Qwen3-VL-",),
        aliases=("qwen3vl",),
        chat_template=None,
        vision_patterns=_QWEN_VISION,
        # vLLM's bundled ViT FlashAttention kernel can contain PTX newer
        # than an otherwise supported NVIDIA driver. The vision encoder is
        # small; SDPA avoids that host-specific startup failure while the
        # language model keeps its selected high-throughput backend.
        vit_attn_backend="TORCH_SDPA",
        status="tested",
        params_b={"2B": 2.2, "4B": 4.4, "8B": 8.8, "32B": 32.8},
        notes="Shipped template renders tool schemas natively. The vLLM "
              "vision encoder uses portable Torch SDPA. Default model of "
              "this repo.",
    ),
    ModelSpec(
        key="qwen2-vl",
        image_span=QWEN_IMAGE_SPAN,
        display="Qwen2-VL",
        provider="Qwen",
        hub="https://huggingface.co/Qwen",
        ids={
            "2B": "Qwen/Qwen2-VL-2B-Instruct",
            "7B": "Qwen/Qwen2-VL-7B-Instruct",
            "72B": "Qwen/Qwen2-VL-72B-Instruct",
        },
        revisions={
            "Qwen/Qwen2-VL-2B-Instruct":
                "895c3a49bc3fa70a340399125c650a463535e71c",
            "Qwen/Qwen2-VL-7B-Instruct":
                "eed13092ef92e448dd6875b2a00151bd3f7db0ac",
            "Qwen/Qwen2-VL-72B-Instruct":
                "2ac26c967836fbb5729c709ad8f8b5548e1f88aa",
        },
        model_types=("qwen2_vl",),
        id_prefixes=("Qwen/Qwen2-VL-",),
        aliases=("qwen2vl",),
        verl_override_fn="qwen2_vl_flops_config",
        chat_template=QWEN_CHATML_TOOLS,
        vision_patterns=_QWEN_VISION,
        vit_attn_backend="TORCH_SDPA",
        status="tested",
        notes="Shipped template drops the tools argument and drops "
              "assistant tool_calls, so it trains with the ChatML tools "
              "template. verl's FLOP estimator also needs three "
              "vision_config attributes this family does not carry, "
              "which verl_override_fn derives from the checkpoint.",
    ),
)

BY_KEY = {m.key: m for m in MODELS}


class UnknownModel(ValueError):
    pass


class PolicyError(ValueError):
    """A base/adapter combination that cannot name one loadable policy."""


@dataclass(frozen=True)
class PolicyResolution:
    """The validated files and adapter shape for one policy."""

    base: str
    adapter: Optional[str] = None
    adapter_base: Optional[str] = None
    adapter_rank: int = 0
    adapter_alpha: int = 0


def supported_table():
    """Plain list used by --list and by the unknown-model error."""
    w = max(len(m.key) for m in MODELS)
    return "\n".join(f"  {m.key:<{w}}  {m.default_id}  [{m.status}]"
                     for m in MODELS)


def _alias_map():
    """Every string that names a family, lowercased."""
    out = {}
    for m in MODELS:
        for a in (m.key,) + tuple(m.aliases):
            out[a.lower()] = (m, m.default_id)
        for size, hf_id in m.ids.items():
            out[f"{m.key}-{size}".lower()] = (m, hf_id)
            out[hf_id.lower()] = (m, hf_id)
    return out


ALIASES = _alias_map()


def _by_model_type(mt):
    for m in MODELS:
        if mt in m.model_types:
            return m
    return None


def _by_name(name):
    low = str(name).lower().rstrip("/")
    if low in ALIASES:
        return ALIASES[low]
    best = None
    for m in MODELS:
        for p in m.id_prefixes:
            if low.startswith(p.lower()) and (best is None
                                              or len(p) > len(best[1])):
                best = (m, p)
    return (best[0], str(name)) if best else None


def _unknown(what, extra=""):
    return UnknownModel(
        f"{what} is not in the bedrock-rl model registry.\n"
        f"{extra}Supported families:\n{supported_table()}\n"
        "Pass a family alias, a size alias such as qwen3-vl-4b, a "
        "HuggingFace id from a listed family, or a local path whose "
        "config.json model_type matches one.\n"
        "To add a family, add one ModelSpec to bedrock_rl/models.py and "
        "run bedrock models check.")


def _local_spec(path):
    cfg = os.path.join(path, "config.json")
    try:
        conf = json.load(open(cfg))
    except (OSError, ValueError) as e:
        raise UnknownModel(f"cannot read {cfg}: {e}")
    mt = conf.get("model_type")
    spec = _by_model_type(mt)
    if spec is None:
        raise _unknown(path, f"Its config.json says model_type {mt!r}.\n")
    return spec


def resolve(name):
    """Family alias, size alias, HuggingFace id, or local path -> ModelSpec.

    A LoRA adapter directory carries no config.json, because peft writes
    only the adapter. It does carry the base model it was trained
    against, so it resolves to whatever that resolves to, which is what
    lets a LoRA SFT checkpoint be handed straight to bin/rl.sh.
    """
    name = str(name)
    if os.path.isdir(name):
        if os.path.exists(os.path.join(name, "config.json")):
            return _local_spec(name)
        base, adapter = lora.policy_paths(name)
        if adapter is not None and base != name:
            return resolve(base)
    hit = _by_name(name)
    if hit is None:
        raise _unknown(repr(name))
    return hit[0]


def model_path(name):
    """A registry alias stands in for a HuggingFace id. Ids and local paths
    pass through untouched."""
    name = str(name)
    if os.path.isdir(name):
        return name
    hit = _by_name(name)
    return hit[1] if hit else name


def resolve_model_location(name, resolve_path=None):
    """Resolve a config-relative checkpoint without rewriting model names.

    A registry alias or HuggingFace repo id is not a filesystem path merely
    because repo ids contain a slash.  Explicit path spellings are always
    paths; otherwise an existing path beside the owning config wins and a
    non-existing value stays a model name.  Callers use the same rule for the
    top-level policy, distillation teachers, and doctor.
    """
    name = str(name)
    if not name:
        return name
    resolver = resolve_path or (
        lambda value: os.path.abspath(os.path.expanduser(str(value))))
    explicit = (os.path.isabs(os.path.expanduser(name))
                or name.startswith(("~", "." + os.sep, "./")))
    candidate = str(resolver(name))
    if explicit or os.path.exists(candidate):
        return candidate
    return name


def model_revision(name):
    """The immutable commit to fetch for `name`, or None.

    None for a local directory, because that is already fixed bytes, and
    None for a hub id this registry does not pin, because there is
    nothing honest to return. Every caller that reaches the hub passes
    the result straight to `revision=`, where None means the same thing
    it has always meant.
    """
    name = str(name)
    if os.path.isdir(name):
        return None
    hit = _by_name(name)
    if hit is None:
        return None
    return hit[0].revision(hit[1])


def download(name, **kw):
    """snapshot_download for a registry name, AT ITS PIN.

    One function, because `snapshot_download(model)` with no revision
    appeared in three places and each one would have had to remember.
    """
    from huggingface_hub import snapshot_download
    ident = model_path(name)
    return snapshot_download(ident, revision=model_revision(ident), **kw)


def weights_path(name):
    """A LOCAL DIRECTORY holding exactly the weights `name` names.

    A hub id is not a set of weights, it is a pointer, and a trainer
    handed one resolves it itself at load time against whatever the
    branch says then. So a run pinned at one revision here and started
    twice can train two different models and report one name for both,
    and a run that reads the pin, prints it, and then passes the id on is
    the worst version of that: it says the right thing and does the other
    one.

    Resolving it to a snapshot directory is what closes that, because a
    directory of files cannot move underneath the process holding it, and
    every consumer -- the trainer, the rollout engine that loads the same
    path, an evaluation of the checkpoint afterwards -- takes a path
    already. Local paths and checkpoint directories pass straight
    through: they are fixed bytes and there is nothing to resolve.

    Downloading here rather than inside the trainer moves no bytes that
    were not going to move; it moves them before any GPU is held, and it
    is what makes `revision=` binding instead of advisory.
    """
    name = str(name)
    if os.path.isdir(name):
        return name
    try:
        return download(name)
    except Exception:
        # The cache alone, in case what failed was the network and not
        # the request: a box that already holds the pinned snapshot is
        # entitled to run offline.
        return download(name, local_files_only=True)


# ── remote code ──────────────────────────────────────────────────────────
# A custom family may need transformers to execute Python from its model
# repository. That is arbitrary code fetched over the network, so the family
# declaration and the operator must both opt in. Neither bundled family needs
# this capability.
REMOTE_CODE_ENV = "BRL_ALLOW_REMOTE_CODE"


def remote_code_allowed():
    return os.environ.get(REMOTE_CODE_ENV, "0").strip().lower() not in (
        "", "0", "no", "false", "off")


def trust_remote_code(spec):
    """The EFFECTIVE flag: what a loader should actually be told.

    Never raises, because the model tables, `bedrock models show` and
    the config generator all ask this about families nobody is loading.
    require_remote_code() is the one that stops a run.
    """
    return bool(spec.trust_remote_code) and remote_code_allowed()


def require_remote_code(spec, what="load this model"):
    """Stop, loudly, at the point where remote code would actually run."""
    if spec.trust_remote_code and not remote_code_allowed():
        raise SystemExit(
            f"{spec.key} needs trust_remote_code, which executes python "
            f"from the model repo ({spec.default_id}) in this "
            f"environment, so it is off unless you say otherwise.\n"
            f"Read what that repo runs, then:\n"
            f"  {REMOTE_CODE_ENV}=1 <your command>\n"
            f"Nothing was loaded, so nothing from that repo has run.")


# ── consumers ────────────────────────────────────────────────────────────

def fit_pixels(image, spec):
    """Rescale a frame into the family's pixel band before the processor
    sees it, the processor resolution control. The shipped 428x240
    frame already sits inside every band here, so this is a no-op until a
    task renders larger frames."""
    import math
    w, h = image.width, image.height
    n = w * h
    factor = 1.0
    if n > spec.image_max_pixels:
        factor = math.sqrt(spec.image_max_pixels / n)
    elif n < spec.image_min_pixels:
        factor = math.sqrt(spec.image_min_pixels / n)
    if factor == 1.0:
        return image
    return image.resize((max(1, int(w * factor)), max(1, int(h * factor))))


def _q(v):
    return "'" + str(v).replace("'", "'\\''") + "'"


def model_config(path):
    """`path`'s config.json as a dict, or None.

    A local directory is read off disk. A hub id is answered from the
    hub's cache AT THE PIN, and only from the cache: this is asked by
    override tables and model listings that have to work for families
    nobody is loading, so it may not depend on a network. The path that
    is about to train does not rely on the cache -- bin/rl.sh resolves
    its weights to a snapshot directory first (weights_path), and a
    directory of weights has its config.json in it.

    None rather than a raise, for the same reason. What a caller may not
    do is treat None as an empty answer: a config nobody could read means
    the caller knows less, not that there is nothing to say.
    """
    import json
    path = str(path)
    local = os.path.join(path, "config.json")
    if os.path.exists(local):
        with open(local) as f:
            return json.load(f)
    if os.path.isdir(path):
        return None
    try:
        from huggingface_hub import hf_hub_download
        got = hf_hub_download(model_path(path), "config.json",
                              revision=model_revision(path),
                              local_files_only=True)
        with open(got) as f:
            return json.load(f)
    except Exception:
        return None


def qwen2_vl_flops_config(spec, path):
    """verl 0.8 scores every Qwen VL family with its Qwen3-VL FLOP
    estimator, which reads vision_config fields that Qwen2-VL does not
    have, and the actor dies at the first log-prob step with
    AttributeError on vision_config.intermediate_size. These three
    attributes exist only so that estimator can run. They change no
    weights and no computation, and they leave the reported MFU
    approximate for this family because the estimator also reads
    vision_config.hidden_size, which means the projection width on
    Qwen2-VL and the tower width on Qwen3-VL.

    An unreadable config is NOT an empty override list. `path` is
    whatever the run named, and a bare hub id has no config.json beside
    it until something downloads one, which is the ordinary way to start
    a run; returning [] there drops the override silently and the actor
    dies at its first log-prob step, after the rollout has already been
    generated and paid for. So the launcher resolves the weights onto
    this disk before asking (bin/rl.sh, weights_path), model_config falls
    back to the hub cache when they are not, and when even that comes up
    empty the two fields the estimator REQUIRES are still emitted from
    this family's own tower shape, which is the same across its sizes.
    Only out_hidden_size, which varies by size and only makes the MFU
    report more accurate, is dropped when nothing can be read."""
    cfg = model_config(path) or {}
    vis = cfg.get("vision_config", {})
    embed = vis.get("embed_dim", 1280)
    ratio = vis.get("mlp_ratio", 4)
    out = vis.get("hidden_size") or cfg.get("text_config", {}).get(
        "hidden_size") or cfg.get("hidden_size")
    base = "+actor_rollout_ref.model.override_config.vision_config."
    ov = [f"{base}intermediate_size={int(embed * ratio)}",
          f"{base}in_channels={int(vis.get('in_chans', 3))}"]
    if out:
        ov.append(f"{base}out_hidden_size={int(out)}")
    return ov


def verl_overrides(spec, path=None):
    """Extra hydra overrides a family needs, registry fields first."""
    ov = list(spec.verl_overrides)
    if spec.vit_attn_backend:
        ov.append("+actor_rollout_ref.rollout.engine_kwargs.vllm."
                  f"mm_encoder_attn_backend={spec.vit_attn_backend}")
    if spec.verl_override_fn and path:
        ov += globals()[spec.verl_override_fn](spec, path)
    return ov


def banner(name, attn=None, adapter=None):
    """The one line a run prints about the model it is about to load.

    Written twice, and the copies could not agree: the trainer's could
    not reflect --attn-impl because it re-resolved the kernel itself.

    Under LoRA there are two paths to name, the weights and the adapter,
    and a header that showed only one of them would not say which policy
    was about to train.
    """
    policy = resolve_policy(name, adapter)
    s = resolve(policy.base)
    base, found = policy.base, policy.adapter
    tail = f" + adapter {found}" if found else ""
    return f"{s.key} ({s.status}) -> {base}{tail} [{attn or attn_impl(s)}]"


def load_processor(spec, path):
    """The family's processor, ready to tokenize.

    The pad-token fix-up is part of loading it, not something one caller
    remembers. The RL path did it and the SFT path did not, and the SFT
    collator hands `tokenizer.pad_token_id` straight to F.pad as the fill
    value, so on any family whose tokenizer ships no pad token that fill
    value was None.
    """
    from transformers import AutoProcessor
    require_remote_code(spec, "load a processor")
    proc = AutoProcessor.from_pretrained(
        path, revision=model_revision(path),
        trust_remote_code=trust_remote_code(spec))
    if proc.tokenizer.pad_token_id is None:
        proc.tokenizer.pad_token = proc.tokenizer.eos_token
    custom = spec.read_chat_template()
    if custom is not None:
        owner = proc if hasattr(proc, "chat_template") else proc.tokenizer
        owner.chat_template = custom
    return proc


def load_weights(spec, path, dtype, attn, quantization_config=None):
    """The transformers class this entry names, loaded from `path`.

    Both trainers resolved `auto_class` and called from_pretrained
    themselves, and the two had drifted: one hardcoded bfloat16 with no
    way to ask for anything else, one honoured an attention override the
    other could not take, and only one of the two failure messages named
    the transformers version that came up short.

    `quantization_config` is how QLoRA gets its 4-bit frozen base
    (bedrock_rl/train/peft_util.py). It is passed through rather than
    built here, because the registry decides what a family IS and not
    how a particular run wants to hold it.
    """
    import transformers
    cls = getattr(transformers, spec.auto_class, None)
    if cls is None:
        raise SystemExit(
            f"registry entry {spec.key} names transformers class "
            f"{spec.auto_class}, which this transformers "
            f"({transformers.__version__}) does not have")
    extra = {} if quantization_config is None else {
        "quantization_config": quantization_config}
    require_remote_code(spec, "load weights")
    return cls.from_pretrained(
        path, dtype=dtype, attn_implementation=attn,
        revision=model_revision(path),
        trust_remote_code=trust_remote_code(spec), **extra)


def attn_impl(spec, override=None):
    """The attention kernel to load a family's weights with.

    "auto" is resolved HERE and nowhere else, so a caller never has to
    probe for flash-attn itself. Callers outside the training venv must
    ask through this function rather than importing flash_attn, because
    the answer belongs to the venv the weights will load in.
    """
    impl = override or spec.attn_impl
    if impl != "auto":
        return impl
    try:
        import flash_attn                                  # noqa: F401
    except ImportError:
        return "sdpa"
    return "flash_attention_2"


def _model_identity(name):
    """A strict identity suitable for checking an adapter's declared base."""
    value = model_path(name)
    if os.path.isdir(value):
        return "path", os.path.realpath(value)
    return "name", str(value).rstrip("/").lower()


def _merged_adapter_for(base, adapter):
    if not os.path.isdir(base):
        return False
    expected = os.path.join(os.path.realpath(base), lora.MERGED_ADAPTER)
    return os.path.realpath(adapter) == expected


def resolve_policy(name, adapter=None):
    """Resolve and validate the base plus optional LoRA adapter once.

    Every consumer needs the same answer.  In particular, an explicit
    ``LORA_ADAPTER`` must not turn an arbitrary string into "LoRA enabled"
    and then fail after the actor has occupied the GPUs.  The adapter's own
    config is also the authority for its rank and for which base it belongs
    on.  The sole non-identical base is verl's merged checkpoint layout: its
    unchanged base weights intentionally sit beside ``lora_adapter/``.
    """
    base, inferred = lora.policy_paths(model_path(name))
    base = model_path(base)
    found = str(adapter) if adapter else inferred
    if not found:
        return PolicyResolution(base)
    found = os.path.abspath(os.path.expanduser(str(found)))
    if not os.path.isdir(found):
        raise PolicyError(
            f"LoRA adapter directory does not exist: {found}")
    config_path = os.path.join(found, lora.ADAPTER_CONFIG)
    if not os.path.isfile(config_path):
        raise PolicyError(
            f"LoRA adapter {found} has no {lora.ADAPTER_CONFIG}")
    try:
        with open(config_path) as stream:
            config = json.load(stream)
    except (OSError, ValueError) as exc:
        raise PolicyError(
            f"cannot read LoRA adapter config {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise PolicyError(f"LoRA adapter config {config_path} is not a mapping")
    declared = config.get("base_model_name_or_path")
    if not declared:
        raise PolicyError(
            f"LoRA adapter {found} does not declare base_model_name_or_path")
    try:
        rank = int(config.get("r") or 0)
        alpha = int(config.get("lora_alpha") or rank)
    except (TypeError, ValueError) as exc:
        raise PolicyError(
            f"LoRA adapter {found} has a non-integer rank or alpha") from exc
    if rank <= 0:
        raise PolicyError(f"LoRA adapter {found} has invalid rank {rank}")
    if rank > lora.VLLM_LORA_RANKS[-1]:
        raise PolicyError(
            f"LoRA adapter {found} has rank {rank}, above the rollout "
            f"engine maximum {lora.VLLM_LORA_RANKS[-1]}")
    if alpha <= 0:
        raise PolicyError(f"LoRA adapter {found} has invalid alpha {alpha}")
    declared = model_path(str(declared))
    if (adapter and _model_identity(base) != _model_identity(declared)
            and not _merged_adapter_for(base, found)):
        raise PolicyError(
            f"LoRA adapter {found} was trained on {declared}, not {base}. "
            "Use its declared base or pass the adapter directory as MODEL.")
    return PolicyResolution(base, found, declared, rank, alpha)


def policy_paths(name, adapter=None):
    """Compatibility tuple for callers that need only ``(base, adapter)``."""
    policy = resolve_policy(name, adapter)
    return policy.base, policy.adapter


def shell_env(name, max_turns=4, adapter=None, snapshot=False):
    """Assignments sourced by the launchers in bin/.

    BRL_MODEL_PATH is the WEIGHTS to load, which for an adapter argument
    is the base model rather than the adapter directory. The adapter
    comes back separately in BRL_LORA_ADAPTER, because every consumer
    needs the two apart: verl takes `model.path` and `lora_adapter_path`,
    and the trainers in bedrock_rl/train/ load one and then attach the
    other. Resolving them in one lookup keeps the rule in one place.

    `snapshot` decides whether BRL_MODEL_PATH is allowed to be a hub id.
    A launcher that is about to TRAIN asks for it, and gets back the
    local snapshot of the pinned revision (weights_path), because a
    trainer handed an id resolves the id itself, later, against a branch
    that may have moved: the pin then describes the run without binding
    it. A launcher that is only reporting -- a table, a resolve, a
    doctor row -- leaves it off, so naming a model never downloads one.

    Everything else in the block is derived AFTER that resolution, which
    is the second half of the point: the family overrides below are read
    out of the checkpoint's own config.json, and that file exists once
    the weights are somewhere on this disk.
    """
    policy = resolve_policy(name, adapter)
    s = resolve(policy.base)
    base, found = policy.base, policy.adapter
    # Both taken before base is replaced by a directory. A snapshot
    # directory is named after a commit hash: it carries no revision of
    # its own, and it does not carry the SIZE either, which the registry
    # reads out of a name. So the id stays available beside the path, and
    # the two are different questions -- what to load, and what it is
    # called -- that were one string only for as long as nothing pinned
    # anything.
    revision = model_revision(base) or ""
    ident = base
    if snapshot:
        base = weights_path(base)
    rank, alpha = policy.adapter_rank, policy.adapter_alpha
    return "\n".join([
        f"BRL_MODEL_KEY={s.key}",
        f"BRL_MODEL_PATH={_q(base)}",
        # The HuggingFace id or checkpoint path this run NAMES, which is
        # BRL_MODEL_PATH again unless the weights were snapshotted. Ask
        # this one anything that reads a name -- the size a family entry
        # is, the label a run directory takes -- and ask BRL_MODEL_PATH
        # for the bytes.
        f"BRL_MODEL_ID={_q(ident)}",
        f"BRL_TOOL_PARSER={s.tool_parser}",
        f"BRL_ATTN_IMPL={attn_impl(s)}",
        f"BRL_CHAT_TEMPLATE={_q(s.chat_template_path or '')}",
        # vLLM's limit_mm_per_prompt is an acceptance cap, not a context
        # policy. One assistant turn may execute several calls and each call
        # returns a frame, so a one-frame-per-turn cap is not sound.
        f"BRL_LIMIT_IMAGES={multimodal_image_limit(max_turns)}",
        f"BRL_MAX_TOOL_CALLS_PER_TURN={MAX_TOOL_CALLS_PER_TURN}",
        f"BRL_MAX_PROMPT_LEN={s.max_prompt_length}",
        # The EFFECTIVE flag, so what bin/rl.sh hands verl is what this
        # operator actually consented to and not what the registry
        # wishes for. A family that needs it and has not been allowed it
        # fails in the loader, with a message that names the variable.
        f"BRL_TRUST_REMOTE_CODE={'True' if trust_remote_code(s) else 'False'}",
        # What the registry pinned, for the run's own record. Empty for a
        # local path and for an unpinned id. It is a LABEL and not the
        # mechanism: nothing downstream is trusted to apply it, because a
        # revision that has to be threaded through every loader is a
        # revision that gets dropped by one of them. BRL_MODEL_PATH above
        # is the mechanism -- under `snapshot` it is already the
        # directory this revision unpacks to.
        f"BRL_MODEL_REVISION={_q(revision)}",
        f"BRL_IMAGE_MIN_PIXELS={s.image_min_pixels}",
        f"BRL_IMAGE_MAX_PIXELS={s.image_max_pixels}",
        f"BRL_STATUS={s.status}",
        # The vision tower's exclusion regex, derived from this family's
        # own vision_patterns. It applies whenever LoRA is on and does
        # nothing when it is off, so it is unconditional here.
        f"BRL_LORA_EXCLUDE={_q(lora.exclude_regex(s) or '')}",
        f"BRL_LORA_ADAPTER={_q(found or '')}",
        # Zero unless an adapter is being continued, in which case it is
        # that adapter's own shape. verl needs a positive lora_rank on
        # the command line even when the shape comes from the adapter.
        f"BRL_LORA_RANK={rank}",
        f"BRL_LORA_ALPHA={alpha}",
        f"BRL_VERL_OVERRIDES="
        # One override per line lets the launcher rebuild a Bash array. A
        # space-joined scalar cannot preserve a local model/config path that
        # itself contains spaces when expanded into Hydra arguments.
        f"{_q(chr(10).join(verl_overrides(s, base)))}",
    ])


CONFIG_HEADER = """\
# GENERATED by bedrock_rl/models.py from the {key} registry entry.
# Edits here are overwritten on the next run. Change the ModelSpec in
# bedrock_rl/models.py instead, or copy this file somewhere of your
# own and hand it to bin/rl.sh with CONFIG_DIR and CONFIG_NAME.
hydra:
  searchpath:
    - file://${{oc.env:VERL_HOME,/opt/verl}}/verl/trainer/config

defaults:
  - ppo_trainer
  - _self_
"""


# Files small enough to hash for free, and the ones that decide what the
# weights MEAN: the architecture, the tokenizer, the processor and the
# chat template. Two checkouts that agree on all of these and on the
# shard list are the same model.
MANIFEST_FILES = (
    "config.json", "generation_config.json", "preprocessor_config.json",
    "processor_config.json", "tokenizer_config.json", "tokenizer.json",
    "special_tokens_map.json", "chat_template.json", "chat_template.jinja",
    "adapter_config.json", "vocab.json", "merges.txt",
)
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf")


def _digest(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_files(path):
    """What a run should record about the bytes on disk.

    Small files are hashed. Weight shards are recorded by NAME AND SIZE
    and not hashed, deliberately: a 235B checkpoint is most of a terabyte
    and sha256 of it at launch would cost more than the run. Name and
    size catch a truncated or half-synced download, which is the failure
    that actually happens; a revision pin covers the rest, because the
    hub resolves a commit to fixed content.
    """
    if not os.path.isdir(path):
        return {}
    out = {}
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if not os.path.isfile(full):
            continue
        try:
            if name in MANIFEST_FILES:
                out[name] = {"sha256": _digest(full),
                             "bytes": os.path.getsize(full)}
            elif name.endswith(WEIGHT_SUFFIXES):
                out[name] = {"bytes": os.path.getsize(full)}
        except OSError as e:
            out[name] = {"unreadable": str(e)}
    return out


def provenance(name, adapter=None):
    """The weights a run is about to load, as a record it can keep.

    Written beside the generated config by emit_config, so every run gets
    one without a launcher having to remember. `bedrock_rl.models
    --provenance MODEL` prints the same thing.
    """
    policy = resolve_policy(name, adapter)
    s = resolve(policy.base)
    base, found = policy.base, policy.adapter
    local = os.path.isdir(str(base))
    ident = None if local else str(base)
    rec = {
        "requested": str(name),
        "family": s.key,
        "resolved": str(base),
        "hub_id": ident,
        # None here is the thing to notice in a manifest: it means the
        # weights were whatever that repo's main was at fetch time
        "revision": None if ident is None else model_revision(ident),
        "trust_remote_code": trust_remote_code(s),
        "adapter": found,
        "files": artifact_files(str(base)),
    }
    if found:
        rec["adapter_base"] = policy.adapter_base
        rec["adapter_rank"] = policy.adapter_rank
        rec["adapter_alpha"] = policy.adapter_alpha
        rec["adapter_files"] = artifact_files(str(found))
    return rec


def emit_config(name, out, adapter=None):
    """Write the hydra config for a model. The chat template is the one
    setting that cannot ride on the hydra command line, because it is a
    multi-line jinja document.

    It also drops a `<config>.provenance.json` beside it, which is the
    run manifest: id, pinned revision, resolved path and the hashes of
    everything small enough to hash. A run that cannot say which weights
    it loaded is a run whose numbers cannot be compared with anything.
    """
    # Validate the complete policy before writing either half of the run
    # manifest.  An invalid adapter must not leave a plausible base-only
    # config behind.
    policy = resolve_policy(name, adapter)
    s = resolve(policy.base)
    body = CONFIG_HEADER.format(key=s.key)
    tmpl = s.read_chat_template()
    if tmpl is not None:
        import yaml
        body += (f"\n# chat template: bedrock_rl/templates/chat_template/"
                 f"{s.chat_template}\n")
        body += yaml.safe_dump(
            {"actor_rollout_ref": {"model": {"custom_chat_template": tmpl}}},
            default_flow_style=False, width=10 ** 6, allow_unicode=True)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write(body)
    # Best effort, and it says so: a manifest is worth having and is
    # never worth failing a launch over, and the weights may not be on
    # this node yet when the config is written.
    try:
        man = os.path.splitext(out)[0] + ".provenance.json"
        with open(man, "w") as f:
            json.dump(provenance(name, adapter), f, indent=2, sort_keys=True)
    except (OSError, ValueError, UnknownModel) as e:
        print(f"[models] could not write a run manifest beside {out}: {e}")
    return out


def readme_table():
    """The supported-models table in README.md, exactly as it ships.
    Regenerate with `python -m bedrock_rl.models --readme-table`.

    Three columns, and no `Status`. The README answers "can I train
    this", and a tested/untested column there reads as a scoreboard on
    families that are supported either way. status_table() answers the
    separate question of which families have been exercised.
    """
    head = "| Model family | Sizes | HuggingFace id |\n|---|---|---|"
    rows = [f"| [{m.display}]({m.hub}) | {'/'.join(m.sizes)} | "
            f"`{m.default_id}` |" for m in MODELS]
    return "\n".join([head] + rows)


def status_table():
    """How far each family has been exercised."""
    head = ("| Model family | Chat template | Exercised |\n"
            "|---|---|---|")
    rows = []
    for m in MODELS:
        tmpl = "shipped by the model" if m.chat_template is None \
            else f"`bedrock_rl/templates/chat_template/{m.chat_template}`"
        note = {"tested": "trained end to end in this repo",
                "untested": "registry entry verified by `bedrock models "
                            "check`, no training run"}.get(m.status, m.status)
        rows.append(f"| [{m.display}]({m.hub}) | {tmpl} | {note} |")
    return "\n".join([head] + rows)


# ── will it fit ──────────────────────────────────────────────────────────
# Parameter bytes alone understate actor memory because activations, optimizer
# state, collectives, and the CUDA runtime do not divide cleanly across cards.
LORA_ACTOR_OVERHEAD = 1.5
BYTES_PER_PARAM_BF16 = 2
# Full FT keeps fp32 parameters, fp32 gradients, and two fp32 Adam moments on
# GPU when OFFLOAD=False: 16 bytes per trainable parameter before activations,
# CUDA/NCCL workspaces, and allocator fragmentation.  The extra 20% is a
# conservative fixed allowance, not a claim that activations scale perfectly
# with parameter count.
BYTES_PER_PARAM_FULL_FT = 16
FULL_FT_RUNTIME_OVERHEAD = 1.2
BYTES_PER_GIB = 1024 ** 3


def card_memory_gib():
    """The smallest visible card, in GiB, or None if nvidia-smi cannot say."""
    import subprocess
    try:
        out = subprocess.run(
            ("nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"),
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    vals = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
    return min(vals) / 1024.0 if vals else None


def sizing(spec, size, n_gpus, card_gib=None, lora=False, gpu_mem_util=0.45,
           offload=False):
    """What a cell of this shape would need, and whether it fits.

    Returns None when the registry does not know this model's parameter
    count, because a spec that cannot answer should say so rather than
    produce a number nobody measured.
    """
    # Param+optimizer offload has a layer-dependent peak and no calibration in
    # this project. Returning no estimate makes the fit gate advisory instead
    # of blessing an invented number.
    if offload:
        return None
    params = (spec.params_b or {}).get(size)
    if not params or not n_gpus:
        return None
    card = card_gib or card_memory_gib()
    if not card:
        return None
    weights = params * 1_000_000_000 * BYTES_PER_PARAM_BF16 / BYTES_PER_GIB
    if lora:
        actor = weights / n_gpus * LORA_ACTOR_OVERHEAD
        actor_basis = ("bf16 frozen base plus measured LoRA runtime "
                       f"overhead ({LORA_ACTOR_OVERHEAD:.2f}x)")
    else:
        train_state = (params * 1_000_000_000 * BYTES_PER_PARAM_FULL_FT
                       / BYTES_PER_GIB)
        actor = train_state / n_gpus * FULL_FT_RUNTIME_OVERHEAD
        actor_basis = ("fp32 weights + gradients + two Adam moments "
                       f"({BYTES_PER_PARAM_FULL_FT} bytes/parameter), plus "
                       f"{FULL_FT_RUNTIME_OVERHEAD:.2f}x runtime allowance")
    # verl colocates the vLLM rollout on the same cards and vLLM takes a
    # FRACTION OF THE WHOLE CARD, not of what is left, which is why a
    # cell can load its actor and then die
    rollout = card * gpu_mem_util
    need = actor + rollout
    return {
        "size": size, "params_b": params, "weights_gib": round(weights, 1),
        "gpus": n_gpus, "card_gib": round(card, 1),
        "actor_per_card_gib": round(actor, 1),
        "rollout_per_card_gib": round(rollout, 1),
        "needed_per_card_gib": round(need, 1),
        "fits": need <= card * 0.95,
        "lora": bool(lora),
        "actor_basis": actor_basis,
        "actor_total_gib": round(actor * n_gpus, 1),
        "note": actor_basis,
    }


def size_key(spec, name):
    """Which entry of spec.ids `name` is, by id, by path or by suffix.

    A local checkpoint directory is named after the weights it came from
    often enough to be worth trying, and when it is not this returns None
    and the caller reports that the size is unknown rather than sizing
    the wrong one.

    The suffix pass matches a size as a WHOLE TOKEN, delimited by a
    non-alphanumeric or by an end of the name. A bare substring test read
    "2B" out of "Qwen3-VL-32B-Instruct", because 2B comes first in
    spec.ids and "32b" contains "2b", so a 32B run resolved as 2B and the
    --fit gate green-lit a cell that cannot hold it. Longest size first
    for the same reason, so "30B-A3B" is not beaten to a name by a
    shorter key that also sits on a boundary in it.

    Exact ids are a whole pass of their own, ahead of the suffix pass, so
    a real id can never lose to another entry's loose match. That pass
    compares basenames case-insensitively, because two sizes of one
    family never differ by case alone and a checkpoint directory written
    in lower case is still named after its weights.
    """
    name = str(name)
    base = os.path.basename(name.rstrip("/"))
    ids = spec.ids or {}
    for size, hub_id in ids.items():
        if (name in (size, hub_id)
                or base.lower() == os.path.basename(hub_id).lower()):
            return size
    for size in sorted(ids, key=len, reverse=True):
        if re.search(r"(?<![0-9A-Za-z])%s(?![0-9A-Za-z])" % re.escape(size),
                     base, re.IGNORECASE):
            return size
    return None


def fit_report(spec, size, n_gpus, lora=False, gpu_mem_util=0.45,
               offload=False):
    """A human line per fact, and whether to refuse. (text, ok)."""
    gpu_mem_util = float(gpu_mem_util)
    if not 0 < gpu_mem_util < 0.95:
        return (
            "sizing  GPU_MEM_UTIL must be greater than 0 and below 0.95; "
            "the actor needs part of each card too",
            False,
        )
    d = sizing(spec, size, n_gpus, lora=lora,
               gpu_mem_util=gpu_mem_util, offload=offload)
    if d is None:
        why = ("parameter/optimizer offload has no calibrated peak estimate"
               if offload else
               "the registry or visible GPU did not provide enough information")
        return ("sizing  advisory unavailable for %s %s: %s; no fit claim "
                "was made" % (spec.key, size, why), True)
    lines = [
        "sizing  %s %s, %.1fB params, %.1f GiB in bf16"
        % (spec.key, size, d["params_b"], d["weights_gib"]),
        "        %d x %.1f GiB cards: actor %.1f + rollout %.1f = %.1f "
        "per card" % (d["gpus"], d["card_gib"], d["actor_per_card_gib"],
                      d["rollout_per_card_gib"], d["needed_per_card_gib"]),
        "        actor estimate: " + d["actor_basis"],
    ]
    if not d["fits"]:
        want = max(1, int(-(-d["actor_total_gib"]
                            // (d["card_gib"] * 0.95 - d["rollout_per_card_gib"]))))
        lines += [
            "        DOES NOT FIT. %.1f needed of %.1f available."
            % (d["needed_per_card_gib"], d["card_gib"]),
            "        try more cards (about %d), a lower GPU_MEM_UTIL than "
            "%.2f, or a smaller size." % (want, gpu_mem_util),
            "        bf16 weights alone are %.1f per card; they are not the "
            "training footprint."
            % (d["weights_gib"] / d["gpus"]),
        ]
    return ("\n".join(lines), d["fits"])


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--readme-table", action="store_true")
    ap.add_argument("--status-table", action="store_true")
    ap.add_argument("--resolve", metavar="MODEL")
    ap.add_argument("--env", metavar="MODEL")
    ap.add_argument("--emit-config", metavar="MODEL")
    ap.add_argument("--provenance", metavar="MODEL",
                    help="the id, the pinned revision, the resolved path "
                         "and the hashes of what is about to be loaded, "
                         "as json")
    ap.add_argument("--fit", metavar="MODEL",
                    help="ask whether a cell of this shape would fit, from "
                         "the registry and the cards in front of it, and "
                         "exit non-zero when it would not. This is a "
                         "twenty-minute OOM turned into a one-line message")
    ap.add_argument("--gpus", type=int, default=0)
    ap.add_argument("--gpu-mem-util", type=float, default=0.45)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--offload", action="store_true",
                    help="parameter and optimizer offload are uncalibrated, "
                         "so --fit reports an advisory instead of a fit claim")
    ap.add_argument("--out")
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--lora-adapter", metavar="DIR", default=None,
                    help="a LoRA adapter directory to load onto MODEL; "
                         "reported in the --env block beside the base "
                         "weights")
    ap.add_argument("--snapshot", action="store_true",
                    help="with --env, download MODEL at the revision this "
                         "registry pins and report the local snapshot "
                         "directory as BRL_MODEL_PATH, so what trains is "
                         "the pinned bytes rather than whatever the hub "
                         "branch resolves to at load time. For launchers "
                         "that are about to train; leave it off to name a "
                         "model without fetching one")
    a = ap.parse_args()
    if a.fit:
        spec = resolve(a.fit)
        size = size_key(spec, a.fit)
        text, ok = fit_report(spec, size, a.gpus or 1,
                              lora=a.lora,
                              gpu_mem_util=a.gpu_mem_util,
                              offload=a.offload)
        print(text)
        return 0 if ok else 1
    try:
        _dispatch(a)
    except (UnknownModel, PolicyError) as e:
        raise SystemExit(str(e))


def _dispatch(a):
    if a.list:
        print(supported_table())
    if a.readme_table:
        print(readme_table())
    if a.status_table:
        print(status_table())
    if a.resolve:
        s = resolve(a.resolve)
        d = dict(s.__dict__)
        d["resolved_path"] = model_path(a.resolve)
        print(json.dumps(d, indent=2, default=list))
    if a.env:
        print(shell_env(a.env, a.max_turns, a.lora_adapter,
                        snapshot=a.snapshot))
    if a.provenance:
        print(json.dumps(provenance(a.provenance, a.lora_adapter),
                         indent=2, sort_keys=True))
    if a.emit_config:
        if not a.out:
            raise SystemExit("--emit-config needs --out")
        print(emit_config(a.emit_config, a.out, a.lora_adapter))


if __name__ == "__main__":
    # main() returns a status for --fit, and a check that cannot fail the
    # process is a check nothing has to obey
    raise SystemExit(main() or 0)
