"""Shared training-run configuration: one selected config, one run.

`bedrock train <config.yaml>` and `bedrock generate <config.yaml>`
read a file instead of a line of flags. A training file may contain one config
directly or independent named ``presets`` under shared ``defaults``; selecting
a preset still launches exactly one run. This module is that config's whole
definition: the schema, the mapping onto the launchers in `bin/`, and the
precedence between the file, the environment and the command line.

WHY A FILE. The launchers' real interface is about forty environment
knobs, and the CLI used to carry a hand-written flag for each one it had
got around to naming. A knob with no flag was unreachable through the
CLI, a flag that reached a launcher which does not read that knob failed
by doing nothing, and a run was reproducible only from a shell line
somebody had to keep. A file is the record of the run, it can name every
knob including the ones nobody tabulated, and it sits next to the
checkpoint.

## The rule

    a YAML key is an environment knob, lowercased.

`train_bs: 8` is `TRAIN_BS=8`. `gpu_mem_util: 0.45` is
`GPU_MEM_UTIL=0.45`. `cuda_visible_devices: "0,1"` is exactly what it
looks like. There is no rename table to read and none to keep in step,
which is the point: `KNOBS` below carries no second spelling, only which
launchers read each knob, whether its value is a path, and how a switch
spells ON. The names stay the launchers' own, so config and `bin/*.sh` use the
same vocabulary.

## A train config

    trainer: rl                     # rl | opd | mopd | sft | sdft | sdpo
    task: equip_pickaxe_grpo        # an example name, or a path
    model: qwen3-vl                 # registry alias, hub id, checkpoint
    steps: 30                       # the launchers' third positional
    algorithm: grpo                 # rl only: grpo | rloo | ...
    train_bs: 8                     # ... and any number of knobs
    env:                            # the escape hatch, verbatim
      RAY_ADDRESS: ray://head:10001

Hinted success harvesting followed by SFT is a workflow, so it says so:

    workflow: self_distill
    task: equip_pickaxe_grpo
    model: qwen3-vl
    steps: 30

`trainer` and `workflow` are mutually exclusive and select a launcher.
`task`, `model` and `steps` are the three positionals every launcher takes;
`algorithm` selects the RL advantage estimator; `env` is passed through
verbatim, name and value, so those names are the variables themselves and
not keys to lowercase.

Two genuinely separate stages can share one file without introducing a
pipeline command::

    defaults:
      task: task.yaml
      model: qwen3-vl-2b
    presets:
      warmup: {trainer: sft, steps: 100, data: sft.parquet}
      sdpo: {trainer: sdpo, steps: 50, data: prompts.parquet}

They remain explicit runs: ``bedrock train train.yaml warmup`` and
``bedrock train train.yaml sdpo --set model=PATH_TO_WARMUP``.

Everything else is validated, because an environment variable nothing
reads fails SILENTLY. An unknown key is a typo or a knob this table has
not heard of, and both are worth stopping for; `env:` is how you say you
meant it. A knob the chosen training target does not read is refused for the same
reason -- `--data` used to reach `bin/rl.sh`, which reads `DATA_DIR`, and
setting it there did nothing at all.

`bedrock generate` has its own smaller import-path schema in
``bedrock_rl.sdg.generation``. It does not map through training knobs
or this module.

## Precedence

Later wins, and this list is the only statement of it.

1. the environment `bedrock` itself was started with
2. the config file: its knob keys, then its `env:` block
3. `--set key=value`, which is the config file's namespace, so it can
   also set `trainer`, `workflow`, `task`, `model`, `steps` and `algorithm`
4. `--netherite-home` / `--verl-env` / `--verl-home`, and `-e KNOB=VALUE`,
   which name the variable itself and are applied last

So a config that names its GPUs gets them whatever the shell exported,
and a knob the config does not mention is still inherited. `--dry-run`
prints the result of all four layers, which is the point of it.
"""
import collections
import copy
import difflib
import os

import yaml

# Every runnable training target, and what its `steps:` counts. `script` is
# the launcher's file name in bin/, which is the execution layer this module
# only ever builds an environment for. Trainers and workflows are separate
# public namespaces even though both eventually select one launcher.
RunSpec = collections.namedtuple("RunSpec", "script what steps")

TRAINERS = {
    "rl": RunSpec("rl.sh", "multi-turn computer-use RL on verl",
                  "RL steps"),
    "opd": RunSpec("rl.sh", "on-policy distillation from one teacher",
                   "distillation steps"),
    "mopd": RunSpec("rl.sh", "multi-teacher on-policy distillation",
                    "distillation steps"),
    "sft": RunSpec("sft.sh", "supervised fine-tuning on tool calls",
                   "SFT steps"),
    "sdft": RunSpec("sdft.sh", "on-policy self-distillation from demos",
                    "SDFT steps"),
    "sdpo": RunSpec("sdpo.sh", "self-distillation policy optimisation",
                    "SDPO steps"),
}

WORKFLOWS = {
    "self_distill": RunSpec(
        "self_distill.sh",
        "harvest hinted successes, strip hints, then SFT",
        "harvest episodes"),
}

TARGETS = {**TRAINERS, **WORKFLOWS}
ALL_TARGETS = tuple(TARGETS)
RL_ALGORITHMS = ("grpo", "rloo", "reinforce_plus_plus", "remax")
VERL_TARGETS = ("rl", "opd", "mopd")

RESERVED = ("trainer", "workflow", "task", "model", "steps",
            "algorithm", "teachers", "env")


class Knob:
    """One environment knob, as a config key can name it.

    `key` is not stored: it is `env.lower()`, always, so there is nothing
    here that could disagree with the rule at the top of this file.
    """

    __slots__ = ("env", "targets", "path", "many", "on", "off", "what",
                 "ambient")

    def __init__(self, env, targets, what, path=False, switch=None,
                 ambient=False, many=False):
        self.env = env
        self.targets = targets
        self.what = what
        # Ambient knobs describe the BOX rather than the run: where the
        # engine is, which venv, how the console prints. Exporting them
        # once per machine is the documented way to use them, so a run
        # that inherits one is the normal case. --dry-run warns about an
        # inherited knob the config did not name, and warning about
        # these would fire on every run on a set-up box, which is how a
        # warning stops being read.
        self.ambient = ambient
        # A path written in a config is resolved beside that config. This
        # makes a task directory portable and matches data.yaml semantics.
        self.path = path
        self.many = bool(many)
        # A switch says how the launcher spells ON **and OFF**, because
        # they do not agree and cannot be made to. `OFFLOAD=True` becomes
        # a hydra value verl parses as a bool, while `HARVEST_ONLY` is tested
        # with `[ -n ... ]`, so its OFF is the EMPTY STRING and a "0"
        # there turns it on. Rendering YAML `true`/`false` the same way
        # for both would stop a self-distillation run before its fine-tune while the
        # config says to run the whole thing.
        self.on, self.off = switch or (None, None)

    @property
    def key(self):
        return self.env.lower()

    @property
    def switch(self):
        return self.on is not None

    def render(self, value):
        """The string the launcher will read, from what YAML parsed.

        Strings pass through except on switches, where command-line and YAML
        boolean spellings must resolve to the same launcher value.
        """
        if self.many:
            import json
            values = value if isinstance(value, (list, tuple)) else [value]
            if not values or any(isinstance(v, (list, tuple, dict, bool))
                                 or v is None for v in values):
                raise ValueError(
                    f"{self.key} takes one or more scalar values")
            return json.dumps([str(v) for v in values], separators=(",", ":"))
        if isinstance(value, str) and self.switch:
            from bedrock_rl import reporting
            if value.lower() in reporting.TRUE:
                return self.on
            if value.lower() in reporting.FALSE:
                return self.off
            raise ValueError(
                f"{self.key} is a switch, and {value!r} is neither on nor "
                f"off\n  on:  " + ", ".join(t for t in reporting.TRUE if t)
                + "\n  off: " + ", ".join(f for f in reporting.FALSE if f))
        if isinstance(value, bool):
            if not self.switch:
                raise ValueError(
                    f"{self.key} takes a value, not true/false; it becomes "
                    f"{self.env} and the launcher reads it as text.\n"
                    f"  YAML reads a bare on, off, yes and no as a BOOLEAN, "
                    f"so quote the word you meant: {self.key}: "
                    f'"{"on" if value else "off"}"')
            return self.on if value else self.off
        if self.switch and isinstance(value, (int, float)):
            if value == 0:
                return self.off
            if value == 1:
                return self.on
            raise ValueError(
                f"{self.key} is a switch, so a numeric value must be 0 "
                f"(off) or 1 (on), not {value!r}")
        # A knob is one string to one shell variable. `[0, 1]` is a
        # natural thing to write in YAML, and str() would hand CUDA the
        # python repr `[0, 1]` and let the run find out.
        if isinstance(value, (list, tuple, dict)):
            shown = (",".join(str(v) for v in value)
                     if not isinstance(value, dict) else "")
            raise ValueError(
                f"{self.key} takes one value, and this is a "
                f"{type(value).__name__}; it becomes {self.env}, which is "
                f"one shell variable"
                + (f'\n  write it as text: {self.key}: "{shown}"'
                   if shown else ""))
        return str(value)


def _k(*a, **kw):
    return Knob(*a, **kw)


# ── the knobs ────────────────────────────────────────────────────────────
# The tuple after each name is the training targets that read it. Nothing
# compares this table with the launchers, so
# so a knob listed here that no launcher reads prints as a supported
# setting that does nothing, and a knob a launcher reads but does not
# claim here is invisible to `bedrock config`. Grep bin/ for the name
# when you add or rename one.
KNOBS = [
    # shared by every launcher, through bin/_common.sh
    _k("NETHERITE_HOME", ALL_TARGETS, "the engine checkout", path=True,
       ambient=True),
    _k("NETHERITE_HOME_PROCEDURAL", ALL_TARGETS,
       "engine checkout built with Netherite procedural textures",
       path=True, ambient=True),
    _k("NETHERITE_HOME_JAR", ALL_TARGETS,
       "engine checkout built from a Minecraft client jar you own",
       path=True, ambient=True),
    _k("VERL_ENV", ALL_TARGETS, "the training venv", path=True, ambient=True),
    _k("CUDA_VISIBLE_DEVICES", ALL_TARGETS, "the training GPUs"),
    _k("RUN_TAG", ALL_TARGETS, "names the run's output and log directory"),
    _k("OUT", ALL_TARGETS, "the checkpoint directory", path=True),
    _k("METRICS", ALL_TARGETS, "where the per-step JSON lines go", path=True),
    _k("MAGMA_GUI_RL", ALL_TARGETS,
       "the engine's mouse-driven container screens",
       switch=("1", "0")),
    _k("BEDROCK_PLAIN", ALL_TARGETS,
       "force plain console output on a terminal",
       switch=("1", "0"), ambient=True),
    _k("WANDB", ALL_TARGETS, "log this run to Weights and Biases",
       switch=("1", "0"), ambient=True),
    _k("WANDB_PROJECT", ALL_TARGETS, "the wandb project, default bedrock-rl",
       ambient=True),
    _k("WANDB_ENTITY", ALL_TARGETS, "the wandb entity", ambient=True),
    _k("TENSORBOARD", VERL_TARGETS, "log this verl run to TensorBoard",
       switch=("1", "0"), ambient=True),
    _k("TENSORBOARD_DIR", VERL_TARGETS, "TensorBoard event directory",
       path=True),

    # LoRA, spelled one way in all four launchers on purpose: a knob that
    # means something different per trainer cannot be carried from a warm
    # start into RL, which is the sequence most runs take.
    _k("LORA_RANK", ALL_TARGETS,
       "train a LoRA adapter of this rank instead of a full fine-tune"),
    _k("LORA_ALPHA", ALL_TARGETS, "LoRA alpha, default the rank"),
    _k("LORA_TARGETS", ALL_TARGETS, "all-linear, or a comma-separated list"),
    _k("LORA_EXCLUDE", ALL_TARGETS, "regex of modules to keep adapter-free; "
                            "defaults to the family's own vision tower"),
    _k("LORA_ADAPTER", ALL_TARGETS, "an existing adapter directory to continue",
       path=True),
    _k("LORA_4BIT", ("sft", "self_distill"),
       "quantize the frozen base to 4-bit NF4 (QLoRA). Available to SFT "
       "and the self-distillation workflow's SFT phase; EMA SDFT/SDPO and "
       "verl refuse it rather than ignoring it",
       switch=("1", "0")),

    # bin/rl.sh
    _k("LR", ("rl", "opd", "mopd", "sft", "sdft", "sdpo",
              "self_distill"),
       "learning rate; bin/rl.sh defaults to 1e-6 full and 1e-4 LoRA"),
    _k("VERL_HOME", VERL_TARGETS, "the verl 0.8 checkout", path=True,
       ambient=True),
    _k("TRAIN_BS", VERL_TARGETS, "training batch size"),
    _k("ROLLOUT_N", VERL_TARGETS, "rollouts per prompt"),
    _k("AGENT_WORKERS", VERL_TARGETS, "parallel agent-loop workers"),
    _k("TEMP", VERL_TARGETS, "rollout temperature"),
    _k("ENTROPY_COEFF", VERL_TARGETS,
       "actor entropy bonus coefficient; 0 disables it"),
    _k("TP", VERL_TARGETS, "rollout tensor parallel size"),
    _k("GPU_MEM_UTIL", VERL_TARGETS, "vLLM memory fraction of the whole device"),
    _k("OFFLOAD", VERL_TARGETS, "offload parameters and the optimizer to host "
                           "memory", switch=("True", "False")),
    _k("SAVE_FREQ", VERL_TARGETS, "steps between checkpoints, default one at "
                             "the end"),
    _k("DATA_DIR", VERL_TARGETS, "directory holding rl_prompts.parquet",
       path=True),
    _k("TRAIN_FILES", VERL_TARGETS,
       "one or more training parquets; MOPD routes rows by data_source",
       path=True, many=True),
    _k("RESPONSE_LEN", VERL_TARGETS, "response token budget, default 512 per "
                                "turn floored at 4096"),
    _k("MODEL_LEN", VERL_TARGETS, "rollout context length, default prompt plus "
                             "response floored at 8192"),
    _k("LOSS_MODE", VERL_TARGETS, "policy loss mode, a second axis to ADV: "
                             "vanilla, or gspo for a sequence-level ratio"),
    _k("LOSS_AGG_MODE", VERL_TARGETS,
       "how the loss aggregates; gspo defaults it to seq-mean-token-mean "
       "and everything else to token-mean"),
    _k("USE_KL_IN_REWARD", VERL_TARGETS,
       "subtract reference-policy KL from rewards",
       switch=("True", "False")),
    _k("KL_COEF", VERL_TARGETS,
       "reference KL coefficient when KL is in reward"),
    _k("USE_KL_LOSS", VERL_TARGETS,
       "add reference-policy KL directly to the actor loss",
       switch=("True", "False")),
    _k("KL_LOSS_COEF", VERL_TARGETS, "actor KL-loss coefficient"),
    _k("KL_LOSS_TYPE", VERL_TARGETS,
       "verl KL estimator, default low_var_kl"),
    _k("MICRO_BS", VERL_TARGETS, "per-GPU micro batch, ignored when DYN_BSZ is on"),
    _k("DYN_BSZ", VERL_TARGETS, "pack sequences to a token budget instead of a "
                           "fixed micro batch", switch=("1", "0")),
    _k("DYN_MAX_TOKEN", VERL_TARGETS,
       "that token budget per GPU; 32768 on a 2-GPU 8B craft arm ran out of "
       "memory waking the rollout engine, so bisect down from 16384"),
    _k("CONFIG_NAME", VERL_TARGETS, "a hand-written hydra config"),
    _k("CONFIG_DIR", VERL_TARGETS, "where that config lives", path=True),
    _k("AGENT_LOOP_CONFIG", VERL_TARGETS,
       "verl agent-loop component file; use this to replace the harness, "
       "tools, probes, or verifier", path=True),
    _k("AGENT_LOOP", VERL_TARGETS,
       "name selected from agent_loop_config, default bedrock"),
    _k("ATTN_IMPL", VERL_TARGETS, "the model's own attention kernel"),
    _k("LOAD_FORMAT", VERL_TARGETS, "how the rollout engine gets its weights"),
    _k("LAYERED_SUMMON", VERL_TARGETS, "gather FSDP shards a layer at a time",
       switch=("True", "False")),
    _k("EXPORT_ADAPTER", VERL_TARGETS, "write the adapter beside the checkpoint "
                                  "when the run ends", switch=("1", "0")),
    _k("NETHERITE_RASTER", VERL_TARGETS, "env rasterizer, cpu or cuda"),
    _k("NETHERITE_ENV_GPUS", VERL_TARGETS,
       "devices environment workers may use, cuda raster only; must not "
       "overlap CUDA_VISIBLE_DEVICES"),
    _k("NNODES", VERL_TARGETS, "nodes training spans; above 1 needs RAY_ADDRESS"),
    _k("GPUS_PER_NODE", VERL_TARGETS, "per-node device count"),
    _k("RAY_ADDRESS", VERL_TARGETS, "the Ray head, required above one node",
       ambient=True),
    _k("RAY_NUM_CPUS", VERL_TARGETS, "CPU resources advertised by a local Ray "
       "runtime; defaults to four per training GPU"),
    _k("DATALOADER_WORKERS", VERL_TARGETS, "host workers for train/validation "
       "row collation; defaults to 0 because Ray already parallelizes "
       "rollouts and child workers make actor teardown unreliable"),
    _k("BRL_TRAJECTORY_DIR", ("rl", "opd", "mopd", "sdft", "sdpo",
                              "self_distill"),
       "directory for canonical bedrock.trajectory.v1 artifacts; on by "
       "default beside metrics, empty disables it", path=True),
    _k("BRL_KEEP_LATEST_IMAGES",
       ("rl", "opd", "mopd", "sft", "sdft", "sdpo", "self_distill"),
       "dynamic policy view: retain the newest N images; 0 keeps all"),
    _k("BRL_FRAMES_ROOT", VERL_TARGETS,
       "optional per-turn render scratch root; a private temporary root is "
       "created automatically", path=True),
    _k("ROLLOUT_DATA_DIR", VERL_TARGETS,
       "verl's own dump of the decoded prompt, response and score per "
       "rollout (trainer.rollout_data_dir), off by default because it is "
       "~35x the bytes and carries no success flag", path=True),
    _k("BRL_SKIP_FIT", VERL_TARGETS, "start even when the fit estimate refuses",
       switch=("1", "0")),
    _k("VERL_LOGGING_LEVEL", VERL_TARGETS, "verl's own log level, default WARN",
       ambient=True),
    _k("BRL_VERL_VERBOSE", VERL_TARGETS, "put verl's unfiltered stream back on "
                                    "the console", switch=("1", "0"),
       ambient=True),

    # verl's first-class on-policy distillation path. `teachers:` is a
    # structured reserved key; these are its scalar loss/resource controls.
    _k("ACTOR_GPUS_PER_NODE", ("opd", "mopd"),
       "student actor GPUs per node; defaults to visible minus teachers"),
    _k("TEACHER_GPUS_PER_NODE", ("opd", "mopd"),
       "teacher-pool GPUs per node"),
    _k("TEACHER_NNODES", ("opd", "mopd"),
       "nodes occupied by teacher replicas; must match NNODES"),
    _k("TEACHER_KEY", ("mopd",),
       "dataset field used to route each row, default data_source"),
    _k("DISTILLATION_LOSS_MODE", ("opd", "mopd"),
       "verl distillation loss, default k1 for policy-gradient OPD"),
    _k("DISTILLATION_TOPK", ("opd", "mopd"),
       "teacher top-k log probabilities"),
    _k("USE_POLICY_GRADIENT", ("opd", "mopd"),
       "use on-policy distillation as a policy-gradient reward",
       switch=("True", "False")),
    _k("USE_TASK_REWARDS", ("opd", "mopd"),
       "combine Minecraft reward with distillation",
       switch=("True", "False")),
    _k("DISTILLATION_LOSS_COEF", ("opd", "mopd"),
       "distillation coefficient when task rewards are enabled"),
    _k("LOSS_MAX_CLAMP", ("opd", "mopd"),
       "maximum distillation loss, or null through env:"),
    _k("LOG_PROB_MIN_CLAMP", ("opd", "mopd"),
       "minimum teacher/student log probability"),
    _k("SKIP_TEACHER_TOKENIZER_CHECK", ("opd", "mopd"),
       "skip exact student/teacher vocabulary compatibility preflight",
       switch=("1", "0")),

    # the trainers and workflow this repo writes itself
    _k("DATA", ("sdft", "sdpo", "self_distill", "sft"),
       "the training rows, prompt parquet or harvest input", path=True),
    _k("SAVE_EVERY", ("sdft", "sdpo", "sft"),
       "steps between checkpoints"),
    _k("DEEPSPEED", ("sft",),
       "ZeRO preset (z2, z2_offload, z3, z3_offload) or a config path"),
    _k("RESUME", ("sft",), "auto, or a checkpoint to continue"),
    _k("BATCH", ("sft",), "examples per device"),
    _k("GRAD_ACCUM", ("sft",), "SFT gradient accumulation steps"),
    _k("PROMPTS", ("sdft", "sdpo"), "episode specs per step"),
    _k("TEMPERATURE", ("sdft", "sdpo"), "rollout sampling temperature"),
    _k("MAX_NEW_TOKENS", ("sdft", "sdpo"),
       "maximum generated tokens per assistant turn"),
    _k("WORKERS", ("sdft", "sdpo"),
       "parallel live episode workers"),
    _k("DEMONSTRATIONS", ("sdft",),
       "expert trajectory JSON/JSONL or SFT parquet", path=True),
    _k("DEMONSTRATION_MATCH", ("sdft",),
       "pair demonstrations by world seed (default), or random for an "
       "explicit cross-world ablation"),
    _k("TEACHER_MIXUP_ALPHA", ("sdft",),
       "moving self-teacher update; SDFT release default 0.01"),
    _k("TEACHER_UPDATE_RATE", ("sdpo",),
       "EMA privileged self-teacher update; SDPO release default 0.05"),
    _k("SKIP_TOKENS", ("sdft",),
       "assistant tokens skipped at each trajectory start"),
    _k("GROUP_SIZE", ("sdpo",), "rollouts per prompt"),
    _k("ALPHA", ("sdpo",), "divergence interpolation"),
    _k("TOPK", ("sdpo",), "top-k support for the teacher distribution"),
    _k("IS_CLIP", ("sdpo",), "importance-sampling clamp"),
    _k("EVAL_EVERY", ("sdpo",), "steps between evaluations"),
    _k("EVAL_EPISODES", ("sdpo",), "episodes per evaluation"),
    _k("MAX_DRY_STEPS", ("sdpo",),
       "stop after this many consecutive steps that took no gradient"),
    _k("HINT_LEVEL", ("sdpo", "self_distill"),
       "none, coarse or procedure"),
    _k("FINISH_HINT", ("sdpo", "self_distill"),
       "stop scaffold, off or instruct"),
    # The per-turn ladder. A separate channel from HINT_LEVEL, which is
    # the static system block: this one is recomputed from the world
    # state each turn and rides on the observation, and the two rungs are
    # set independently because the ablation varies one at a time.
    _k("TURN_HINT", ("sdpo", "self_distill"),
       "per-turn hint rung: off, weak, direction, oracle"),
    _k("TURN_HINT_P", ("sdpo", "self_distill"),
       "per-EPISODE probability the per-turn hint is shown, so a run can "
       "anneal it from 1 to 0"),
    _k("EPOCHS", ("self_distill",), "fine-tuning epochs"),
    _k("HARVEST_BATCH", ("self_distill",),
       "episodes played per harvest batch"),
    # OFF is the empty string and not "0": self_distill.sh tests it with
    # `[ -n "${HARVEST_ONLY:-}" ]`, which reads any non-empty value,
    # "0" included, as on.
    _k("HARVEST_ONLY", ("self_distill",),
       "harvest and print the yield, then stop",
       switch=("1", "")),
    _k("ROLLOUT_ENGINE", ("sdft", "sdpo", "self_distill"),
       "how rollouts are sampled: hf, or vllm for the fast path"),
    _k("VLLM_GPU_FRAC", ("sdft", "sdpo", "self_distill"),
       "card fraction that engine may take beside the training model"),
    _k("VLLM_MAX_MODEL_LEN", ("sdft", "sdpo", "self_distill"),
       "rollout context length for that engine"),
]

BY_KEY = {k.key: k for k in KNOBS}
assert len(BY_KEY) == len(KNOBS), "two knobs with the same name"


def knobs_for(target):
    """Every environment knob one trainer or workflow reads."""
    return [k for k in KNOBS if target in k.targets]


class ConfigError(SystemExit):
    """A user error in a config file. SystemExit with a string is how
    this repo's modules report one: the first line is the message and the
    rest is the fix, which is the shape bedrock_rl/cli/console.py
    prints."""


def _merged(base, overlay):
    if not isinstance(base, dict) or not isinstance(overlay, dict):
        return copy.deepcopy(overlay)
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        out[key] = (_merged(out[key], value)
                    if key in out else copy.deepcopy(value))
    return out


def load(path, preset=None):
    """Load one train config, optionally from a named-preset manifest."""
    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except OSError as e:
        raise ConfigError(f"cannot read the config at {path}: {e}")
    except yaml.YAMLError as e:
        raise ConfigError(f"{path} is not valid YAML\n  {e}")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must be a mapping of keys to values, "
                          f"and it parsed as {type(raw).__name__}")
    if "presets" not in raw:
        if preset is not None:
            raise ConfigError(f"{path} has no named presets; remove {preset!r}")
        return raw
    unknown = sorted(set(raw) - {"defaults", "presets"})
    if unknown:
        raise ConfigError(f"{path}: unknown training manifest keys: "
                          + ", ".join(unknown))
    defaults = raw.get("defaults") or {}
    presets = raw.get("presets")
    if not isinstance(defaults, dict):
        raise ConfigError(f"{path}: defaults must be a mapping")
    if not isinstance(presets, dict) or not presets:
        raise ConfigError(f"{path}: presets must be a non-empty mapping")
    if preset is None:
        if len(presets) != 1:
            raise ConfigError(f"{path} has named training presets: "
                              + ", ".join(str(name) for name in presets)
                              + "; choose one after the filename")
        preset = next(iter(presets))
    if preset not in presets:
        raise ConfigError(f"{path} has no training preset {preset!r}; choose "
                          + ", ".join(str(name) for name in presets))
    selected = presets[preset]
    if not isinstance(selected, dict):
        raise ConfigError(f"{path}: preset {preset!r} must be a mapping")
    return _merged(defaults, selected)


def apply_sets(cfg, sets, where="--set"):
    """Layer 3: the command line writing into the config's own namespace.

    Values stay strings. A knob's value reaches the launcher as text
    anyway, and the four reserved keys are text on the command line too,
    so there is nothing here worth guessing a type for.

    Returns the merged config and the keys layer 3 supplied, because a
    file's `env:` block is applied after the knob keys and would
    otherwise land on top of the command line.
    """
    out = dict(cfg)
    named = set()
    for item in sets or []:
        if "=" not in item:
            raise ConfigError(f"{where} wants key=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v
        named.add(k.strip())
    return out, named


def _unknown(origin, key, allowed, extra=""):
    near = difflib.get_close_matches(key, allowed, n=1)
    fix = f"did you mean {near[0]}?\n  " if near else ""
    return ConfigError(f"{origin}: unknown key {key!r}\n  {fix}{extra}")


def _near_knob(origin, key, kind, target):
    """An unknown key on a train config.

    The near miss is drawn from the keys THIS target can read, because
    suggesting one it cannot is a two-step failure: `train_bsz` on an
    self-distillation config suggested `train_bs`, which is then refused as
    an rl knob. A near miss that belongs to another trainer is still
    worth naming, as that fact rather than as a suggestion.
    """
    if key == "adv":
        return ConfigError(
            f"{origin}: adv: was renamed because it selects an RL "
            "algorithm, not an environment knob\n  algorithm: grpo")
    if key == "mode":
        return ConfigError(
            f"{origin}: mode: belonged to the removed whole-loop runner\n"
            "  use algorithm: for an RL trainer, or run data generation "
            "and training explicitly")
    mine = [k.key for k in KNOBS if target in k.targets]
    near = difflib.get_close_matches(key, mine + list(RESERVED), n=1)
    if near:
        return ConfigError(f"{origin}: unknown key {key!r}\n"
                           f"  did you mean {near[0]}?")
    other = difflib.get_close_matches(key, list(BY_KEY), n=1)
    if other:
        knob = BY_KEY[other[0]]
        return ConfigError(
            f"{origin}: unknown key {key!r}\n"
            f"  {other[0]} is close, and it is a "
            + ", ".join(knob.targets) + f" knob, not a {target} one")
    return ConfigError(
        f"{origin}: unknown key {key!r}\n"
        "  a key is an environment knob lowercased, and anything this "
        "schema has not named goes under env: verbatim\n"
        f"  {PROG_KNOBS} {target}  lists every key this {kind} reads")


PROG_KNOBS = "bedrock train --knobs"


class TrainPlan:
    """What a train config resolves to, before any path or default the
    CLI itself owns is filled in."""

    def __init__(self, kind, target, script, task, model, steps, algorithm,
                 env):
        self.kind = kind
        self.target = target
        self.script = script
        self.task = task
        self.model = model
        self.steps = steps
        self.algorithm = algorithm
        # knob -> value, already rendered as the launcher will read it,
        # in the order layer 2 and 3 set them
        self.env = env


def train_plan(cfg, sets=(), resolve=str, origin="the config"):
    """A train config, checked, as an environment plus three positionals.

    `resolve` turns a config-owned path into an absolute one. It is passed in
    rather than done here because this module does not know the config file's
    directory.
    """
    cfg, from_sets = apply_sets(cfg, sets)

    trainer = cfg.pop("trainer", None)
    workflow = cfg.pop("workflow", None)
    if trainer is not None and workflow is not None:
        raise ConfigError(
            f"{origin} names both trainer: and workflow:\n"
            "  choose one: a trainer performs one training method; a "
            "workflow coordinates more than one phase")
    if trainer is None and workflow is None:
        if "stages" in cfg:
            raise ConfigError(
                f"{origin} has stages: and no trainer: or workflow:, so it "
                "is a data "
                f"config rather than a training one\n"
                f"  bedrock generate {origin}")
        raise ConfigError(
            f"{origin} names no trainer or workflow\n"
            "  trainer: " + " | ".join(TRAINERS) + "\n"
            "  workflow: " + " | ".join(WORKFLOWS))

    kind = "trainer" if trainer is not None else "workflow"
    target = str(trainer if trainer is not None else workflow)
    choices = TRAINERS if kind == "trainer" else WORKFLOWS
    if target not in choices:
        migrations = {
            ("trainer", "warmstart"):
                "warmstart describes why SFT is run, not the trainer\n"
                "  trainer: sft",
            ("trainer", "loop"):
                "the whole-loop runner was removed; generate data and run "
                "each trainer explicitly",
            ("workflow", "star"):
                "use the descriptive workflow name\n"
                "  workflow: self_distill",
        }
        migration = migrations.get((kind, target))
        if migration:
            raise ConfigError(f"{origin}: {kind}: {target} was removed\n  "
                              + migration)
        near = difflib.get_close_matches(target, tuple(choices), n=1)
        raise ConfigError(
            f"{origin} has {kind}: {target}, which is not one of "
            + ", ".join(choices)
            + (f"\n  did you mean {near[0]}?" if near else ""))
    script = choices[target].script

    task = cfg.pop("task", None)
    model = cfg.pop("model", None)
    steps = cfg.pop("steps", None)
    algorithm = cfg.pop("algorithm", None)
    teachers = cfg.pop("teachers", None)
    env_block = cfg.pop("env", None) or {}
    if not isinstance(env_block, dict):
        raise ConfigError(f"{origin}: env: must be a mapping of "
                          f"KNOB: value")

    if algorithm is not None:
        algorithm = str(algorithm)
        if target != "rl":
            raise ConfigError(
                f"{origin}: algorithm selects the RL advantage estimator, "
                f"so it does not apply to {kind}: {target}")
        if algorithm not in RL_ALGORITHMS:
            near = difflib.get_close_matches(algorithm, RL_ALGORITHMS, n=1)
            raise ConfigError(
                f"{origin}: algorithm: {algorithm} is not one of "
                + ", ".join(RL_ALGORITHMS)
                + (f"\n  did you mean {near[0]}?" if near else ""))

    if teachers is not None:
        if target not in ("opd", "mopd"):
            raise ConfigError(
                f"{origin}: teachers: configures on-policy distillation, "
                f"so it does not apply to {kind}: {target}")
        if not isinstance(teachers, dict) or not teachers:
            raise ConfigError(f"{origin}: teachers: must be a non-empty mapping")
        expected = 1 if target == "opd" else 2
        relation = "exactly" if target == "opd" else "at least"
        if ((target == "opd" and len(teachers) != expected)
                or (target == "mopd" and len(teachers) < expected)):
            raise ConfigError(
                f"{origin}: trainer: {target} needs {relation} {expected} "
                f"teacher{'s' if expected != 1 else ''}; got {len(teachers)}")
        normalized = {}
        from bedrock_rl import models
        for name, raw_teacher in teachers.items():
            name = str(name)
            if not isinstance(raw_teacher, dict):
                raise ConfigError(
                    f"{origin}: teachers.{name} must be a mapping")
            teacher = dict(raw_teacher)
            unknown = sorted(set(teacher) - {
                "model", "route", "replicas", "tensor_parallel",
                "data_parallel", "pipeline_parallel", "engine",
                "gpu_memory_utilization", "max_model_len",
            })
            if unknown:
                raise ConfigError(
                    f"{origin}: teachers.{name} has unknown keys: "
                    + ", ".join(unknown))
            if not teacher.get("model"):
                raise ConfigError(
                    f"{origin}: teachers.{name} needs model:")
            teacher_model = str(teacher["model"])
            # The top-level student, teachers, and doctor share this exact
            # distinction between a config-relative checkpoint and a Hub id.
            teacher_model = models.resolve_model_location(
                teacher_model, resolve)
            teacher["model"] = teacher_model
            normalized[name] = teacher
        import json
        teachers_json = json.dumps(normalized, separators=(",", ":"),
                                   sort_keys=True)
    elif target in ("opd", "mopd"):
        raise ConfigError(f"{origin}: trainer: {target} needs teachers:")
    else:
        teachers_json = None

    env = {}
    for key, value in cfg.items():
        if value is None:                 # a commented-out setting
            continue
        # YAML 1.1 reads a bare `on:` or `no:` as a BOOLEAN key, and an
        # unquoted number as an int, so a key here is not always a
        # string. It is reported as one either way rather than tripping
        # difflib several frames from the mistake.
        key = str(key)
        knob = BY_KEY.get(key)
        if knob is None:
            raise _near_knob(origin, key, kind, target)
        if target not in knob.targets:
            raise ConfigError(
                f"{origin}: {key} becomes {knob.env} ({knob.what}), which "
                f"bin/{script} does not read, so setting it here would do "
                f"nothing\n  the training targets that read it are "
                + ", ".join(knob.targets))
        try:
            env[knob.env] = knob.render(value)
        except ValueError as e:
            raise ConfigError(f"{origin}: {e}")
        if knob.path:
            if knob.many:
                import json
                env[knob.env] = json.dumps(
                    [resolve(value) for value in json.loads(env[knob.env])],
                    separators=(",", ":"))
            else:
                env[knob.env] = resolve(env[knob.env])

    # The `env:` block is the second half of LAYER 2, so it lands on top
    # of the knob keys beside it -- and then the command line lands on
    # top of both. Applying it last and stopping there put layer 2 over
    # layer 3: a file with `env: {TRAIN_BS: 99}` quietly won against
    # `--set train_bs=3`, which is the command line saying one thing and
    # the child getting another, with a --dry-run that agreed with the
    # file. `--set` is the only way to vary a run without editing it.
    #
    # A name given BOTH ways in the same file is refused rather than
    # ordered, because there is no reading of one file that says two
    # things about one variable -- unless the command line names it too,
    # which settles it and is the layer that is allowed to.
    for name in env_names(env_block, origin):
        if name == "ADV" and target == "rl":
            raise ConfigError(
                f"{origin}: ADV is selected by algorithm:, not env:\n"
                "  algorithm: grpo")
        knob = BY_KEY.get(name.lower())
        if (knob is not None and knob.env in env
                and name.lower() not in from_sets):
            raise ConfigError(
                f"{origin} sets {name} twice, as {knob.key}: and again "
                f"under env:\n  say it once, as {knob.key}:")
    env.update(env_names(env_block, origin))
    for key in from_sets:
        knob = BY_KEY.get(key)
        if knob is not None and knob.env in env:
            env[knob.env] = knob.render(cfg[key])
            if knob.path:
                if knob.many:
                    import json
                    env[knob.env] = json.dumps(
                        [resolve(value) for value in
                         json.loads(env[knob.env])], separators=(",", ":"))
                else:
                    env[knob.env] = resolve(env[knob.env])
    if algorithm is not None:
        env["ADV"] = algorithm
    if teachers_json is not None:
        env["BRL_TEACHERS_JSON"] = teachers_json
        env["BRL_DISTILLATION_MODE"] = target
    partition = (env.get("TRAIN_BS"), env.get("ROLLOUT_N"),
                 env.get("AGENT_WORKERS"))
    if all(value is not None for value in partition):
        try:
            train_bs, rollout_n, workers = map(int, partition)
        except ValueError as exc:
            raise ConfigError(
                f"{origin}: train_bs, rollout_n, and agent_workers must be "
                "integers") from exc
        if min(train_bs, rollout_n, workers) < 1:
            raise ConfigError(
                f"{origin}: train_bs, rollout_n, and agent_workers must be "
                "positive")
        rollouts = train_bs * rollout_n
        if rollouts % workers:
            raise ConfigError(
                f"{origin}: agent_workers ({workers}) must divide "
                f"train_bs * rollout_n ({train_bs} * {rollout_n} = "
                f"{rollouts})")
    return TrainPlan(kind, target, script, task, model, steps, algorithm, env)


def env_names(block, origin):
    """The `env:` block, verbatim.

    A name that is not already upper case is refused rather than
    uppercased for you. Every knob these launchers read is upper case, so
    a lower-case name here is almost always someone writing a CONFIG key
    in the wrong place, and it is the one mistake `env:` cannot catch on
    its own: it exports a variable nothing reads and says nothing.
    """
    out = {}
    for name, value in block.items():
        if value is None:
            continue
        name = str(name)
        if name != name.upper():
            fix = (f"{name.lower()}: {value}   as an ordinary key"
                   if name.lower() in BY_KEY
                   else f"{name.upper()}: {value}")
            raise ConfigError(
                f"{origin}: env: names variables exactly as written, and "
                f"{name} is not upper case, so nothing would read it\n"
                f"  {fix}")
        out[name] = str(value)
    return out


def example_dir():
    """Where the worked examples live, or None from an installed wheel."""
    from bedrock_rl import resources
    root = resources.repo_root()
    if root is None:
        return None
    d = root / "examples"
    return d if d.is_dir() else None


def examples():
    """The shipped configs, as paths, for an error message to point at."""
    d = example_dir()
    if d is None:
        return []
    return sorted(str(p) for p in d.glob("*/train.yaml"))


def example_for(kind, target):
    """The shipped config for one trainer or workflow, when present."""
    for path in examples():
        try:
            if load(path).get(kind) == target:
                return path
        except SystemExit:
            continue
    return None


def relative(path):
    from bedrock_rl import resources
    root = resources.repo_root()
    if root is None:
        return path
    rel = os.path.relpath(path, str(root))
    return path if rel.startswith("..") else rel
