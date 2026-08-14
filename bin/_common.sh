#!/usr/bin/env bash
# Shared preamble for every launcher in bin/. Source it, do not run it.
#
# It does what every launcher has to do first: find the repo, point the
# engine and PYTHONPATH at it, check that uv and the training venv are
# there, and resolve the model through the registry BEFORE any GPU is
# touched. A launcher that skips the registry lookup trains a family with
# another family's chat template, which is silent and expensive, so this
# is shared rather than copied.
#
# One argument convention across all five launchers:
#
#   bash bin/<name>.sh [TASK] [MODEL] [N]
#
#   TASK   example name, or a path to a task YAML
#   MODEL  registry family alias (qwen3-vl), size alias (qwen3-vl-4b),
#          HuggingFace id, or a local checkpoint path
#   N      the size of the run: RL and SDPO steps, SFT steps,
#          self-distillation harvest episodes
#
# Everything else is an environment knob, and each launcher lists its own
# at the top. Variables this file sets for the launchers are prefixed
# BRL_, so they cannot collide with a knob or leak meaning into a child
# process; the knobs themselves keep their documented names.
set -euo pipefail
# `foo "$(bar)"` does NOT stop under set -e alone: bash ignores the
# failure of a command substitution and hands the caller an empty string.
# Every path in these launchers is built that way, so without this a
# lookup that failed became an empty argument and the run carried on with
# it.
#
# It arrived in bash 4.4 and macOS still ships 3.2.57, where `shopt -s`
# on an unknown option FAILS and would take this whole file, and every
# launcher that sources it, down with it. The guarantee simply does not
# exist on 3.2, which is where it already was, so its absence is not
# worth refusing to run over.
shopt -s inherit_errexit 2>/dev/null || true

BRL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NETHERITE_HOME="${NETHERITE_HOME:-$HOME/netherite}"
export PYTHONPATH="$BRL_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# tasks solved through the inventory screen need the engine's virtual
# mouse and the GUI overlay composited into the captured frame
export MAGMA_GUI_RL="${MAGMA_GUI_RL:-1}"

VERL_ENV="${VERL_ENV:-$HOME/verl-env}"

brl_require_venv() {
    if ! command -v uv >/dev/null 2>&1; then
        echo "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
        exit 1
    fi
    if [ ! -x "$VERL_ENV/bin/python" ]; then
        echo "training venv missing at $VERL_ENV; run:" \
             "bash $BRL_ROOT/bin/setup_deps.sh" >&2
        exit 1
    fi
}

# the training stack is CUDA-ABI-locked and lives outside this uv project,
# so --no-project keeps uv from syncing the repo env over it
brl_run() { uv run --no-project --python "$VERL_ENV/bin/python" "$@"; }

# ── the CPU half of a launcher ───────────────────────────────────────────
# Two things these scripts do before any GPU exists are pure CPU work on
# a text file: which file a task NAME means, and what a key inside it
# says. Both were running through brl_run, so they needed the training
# venv, so asking where a task file lives required a torch install. On a
# box that has no ~/verl-env they did not fail with "no such task", they
# failed with "No interpreter found", which names the wrong problem.
#
# So they get an interpreter that can merely import this package. The
# training venv is tried FIRST because a training box always has it and
# it is already warm, then whatever python is on PATH, and uv's project
# environment last, since that is the one that may have to build itself.
brl_cpu_python() {
    local cand
    for cand in "$VERL_ENV/bin/python" python3 python; do
        if command -v "$cand" >/dev/null 2>&1 \
                && "$cand" -c "import bedrock_rl.resources" 2>/dev/null;
        then
            echo "$cand"
            return 0
        fi
    done
    return 1
}

brl_cpu_run() {
    local py
    if py="$(brl_cpu_python)"; then
        "$py" "$@"
    else
        # nothing reachable can import the package, so let uv build the
        # CPU project environment, which is what `uv sync` documents
        (cd "$BRL_ROOT" && uv run python "$@")
    fi
}

# For the launchers that run verl. verl 0.8 unpads a batch through
# flash_attn.bert_padding when it computes log probs, and that import has
# no CUDA fallback, so a training venv without the package gets all the
# way through model load, vLLM warmup and a whole rollout batch before
# dying inside verl/utils/attention_utils.py with a ModuleNotFoundError.
# That is a whole model load, warmup and rollout batch of GPU time to
# learn something knowable in a tenth of a second, so it is asked first.
#
# ATTN_IMPL=sdpa does NOT make it optional. That knob picks the model's
# own attention kernel and nothing else; setup_deps.sh says the same thing
# from the other end when the wheel fails to build. A box with more than
# one training venv is where this bites, because the one that resolves
# first is not always the one that was finished.
brl_require_flash_attn() {
    brl_run python -c "import flash_attn" >/dev/null 2>&1 && return 0
    cat >&2 <<EOF

training venv $VERL_ENV has no flash_attn, and verl reaches into
flash_attn.bert_padding to compute log probs. This run would die at the
first log-prob step, after the first rollout batch, not here.

  uv pip install --python "$VERL_ENV/bin/python" --no-build-isolation \\
      flash-attn==2.8.3

or point VERL_ENV at the venv that has it. Everything except bin/rl.sh
works without it, including SFT.

EOF
    exit 1
}

# One registry lookup, before any GPU is touched. The assignment comes
# first so that an unknown model trips set -e with the supported list.
# Sets BRL_MODEL_KEY, BRL_MODEL_PATH and the rest of the BRL_ block, and
# rewrites MODEL to the resolved path.
#
# It prints nothing. Every trainer opens with a header that names the
# model as the registry resolved it, so a line here was the same fact
# twice, once in a format nothing else used.
#
# The same lookup settles LoRA, because the two questions are one
# question: MODEL may itself be an adapter directory, in which case the
# WEIGHTS to load are its base and the adapter is a second path. After
# this returns, MODEL is always the base weights and LORA_ADAPTER is
# always the adapter or empty, whichever way the caller named them.
brl_resolve_model() {
    local env_block
    env_block="$(brl_run python -m bedrock_rl.models --env "$MODEL" \
                 ${LORA_ADAPTER:+--lora-adapter "$LORA_ADAPTER"} "$@")"
    eval "$env_block"
    MODEL="$BRL_MODEL_PATH"
    LORA_ADAPTER="${LORA_ADAPTER:-$BRL_LORA_ADAPTER}"
    # Continuing an adapter is LoRA whatever LORA_RANK says, and verl
    # needs the rank on the command line even then, so an unset one takes
    # the adapter's own shape rather than zero.
    LORA_RANK="${LORA_RANK:-$BRL_LORA_RANK}"
    # Alpha follows the rank, which is scaling 1.0, unless an adapter
    # being continued says otherwise or the caller does. Defaulting it to
    # a constant instead means a rank the caller chose and a scaling they
    # did not, and the two are the same knob to anyone reading a run.
    if [ "${LORA_ALPHA:-0}" -le 0 ]; then
        if [ "${BRL_LORA_ALPHA:-0}" -gt 0 ]; then
            LORA_ALPHA="$BRL_LORA_ALPHA"
        else
            LORA_ALPHA="$LORA_RANK"
        fi
    fi
    # The family's own vision-tower exclusion, from its vision_patterns.
    # LORA_EXCLUDE overrides it; nothing reads either unless LoRA is on.
    LORA_EXCLUDE="${LORA_EXCLUDE:-$BRL_LORA_EXCLUDE}"
}

# Is LoRA on for this run. Rank decides for a fresh adapter, and
# continuing an existing one is LoRA whatever the rank says.
brl_lora_on() {
    [ "${LORA_RANK:-0}" -gt 0 ] || [ -n "${LORA_ADAPTER:-}" ]
}

# A path is taken as given; a bare name is looked up by
# bedrock_rl/resources.py::find_task, which is the same function
# `bedrock` uses.
#
# One rule, in one place, and an unknown name stops here.
brl_task_yaml() {
    case "$1" in
        *.yaml|*.yml|*/*) echo "$1" ;;
        *) brl_cpu_run -c "
import sys
from bedrock_rl.resources import find_task, TASK_DIRS
p = find_task(sys.argv[1])
if not p:
    sys.exit('no example named %r under %s'
             % (sys.argv[1], ' or '.join(d + '/' for d in TASK_DIRS)))
print(p)" "$1" ;;
    esac
}

# One key out of a task file, with a fallback for an absent key.
brl_task_get() {
    brl_cpu_run -c \
        "import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
print(d.get(sys.argv[2], sys.argv[3]))" \
        "$(brl_task_yaml "$1")" "$2" "$3"
}

# The `name:` inside the task file. Every data, log and checkpoint path is
# built from this and not from the path used to locate the task.
brl_task_name() {
    brl_task_get "$1" name ""
}

# ── the launcher preamble ────────────────────────────────────────────────
# Every launcher opened with the same block and then diverged in small
# ways that were not decisions: one wrote `tee` and the rest `tee -a`, one
# looked the model up in the wrong venv, one built its checkpoint path
# from the raw positional instead of the task's own name. The three
# functions below are that block.

# Positionals, GPUs and the venv check. $1 is `auto`, an explicit
# CUDA_VISIBLE_DEVICES fallback, or `-` for a launcher that touches no GPU
# itself. A scheduler (including Modal) owns CUDA_VISIBLE_DEVICES when it is
# already set; otherwise `auto` selects every GPU nvidia-smi reports. The rest is
# the launcher's own "$@": TASK and MODEL are the shared pair, the third
# positional means something different per launcher, so each reads it.
brl_setup() {
    local devices="$1"
    shift
    if [ "$devices" != - ]; then
        if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$devices" = auto ]; then
            local detected=""
            if command -v nvidia-smi >/dev/null 2>&1; then
                detected="$(nvidia-smi --query-gpu=index \
                    --format=csv,noheader 2>/dev/null \
                    | sed 's/[[:space:]]//g' | paste -sd, -)"
            fi
            devices="${detected:-0}"
        fi
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$devices}"
        if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
            echo "CUDA_VISIBLE_DEVICES is empty; expose at least one GPU" >&2
            exit 1
        fi
        BRL_NGPU=$(( $(echo "$CUDA_VISIBLE_DEVICES" | tr -cd , | wc -c) + 1 ))
    fi
    brl_require_venv
    TASK="${1:-equip_pickaxe_grpo}"
    MODEL="${2:-${MODEL:-qwen3-vl}}"
}

# Run identity: BRL_TASK_NAME, the slug every path is built from, the
# checkpoint dir, the console log and the metrics file. Arguments are the
# slug pieces after the task name, so `brl_paths sdpo "$BRL_TAG"` gives
# <task>_sdpo_<tag>.
#
# BRL_LOG and BRL_METRICS are two files on purpose. BRL_LOG is whatever
# scrolled past, for a person reading a finished run. BRL_METRICS is one
# JSON object per training step and nothing else, because that is what
# later analysis parses, and mixing the two put JSON lines in the middle
# of a console transcript and called it a record.
brl_paths() {
    BRL_TASK_NAME="$(brl_task_name "$TASK")"
    local IFS=_
    BRL_SLUG="${BRL_TASK_NAME}_$*"
    OUT="${OUT:-$HOME/ckpts/bedrock_$BRL_SLUG}"
    local stamp
    stamp="$(date +%s)"
    BRL_LOG="$BRL_ROOT/logs/${BRL_SLUG}_$stamp.log"
    BRL_METRICS="${METRICS:-$BRL_ROOT/logs/${BRL_SLUG}_$stamp.jsonl}"
    # Canonical OpenAI-chat trajectories are evidence for every trainer that
    # plays an environment, not a verl-only debug feature. An explicitly empty
    # value disables them for a throughput-only run.
    export BRL_TRAJECTORY_DIR="${BRL_TRAJECTORY_DIR-${BRL_METRICS%.jsonl}_trajectories}"
    mkdir -p "$OUT" "$BRL_ROOT/logs"
}

# Append `--flag <value>` to BRL_ARGS, but only when the named variable is
# set and non-empty.
#
# The launchers used to spell every flag `--prompts "${PROMPTS:-8}"`,
# which put a copy of the trainer's argparse default in the shell.
# Fourteen of them did. A copy is a thing that goes stale silently,
# because the shell value always wins: change the default in Python and
# nothing happens, and nothing says so. An unset knob now passes no flag
# at all and argparse decides.
#
# Which means BRL_ARGS is routinely EMPTY, and an empty array must be
# expanded as `"${BRL_ARGS[@]+"${BRL_ARGS[@]}"}"` rather than
# `"${BRL_ARGS[@]}"`. Under `set -u` bash treats an empty array as unset
# and aborts on the bare form, in every bash before 4.4 -- which includes
# the 3.2.57 macOS still ships and this file already accounts for above.
# The `+` form expands to nothing at all when the array is empty and is
# otherwise identical, so every launcher spells it that way.
brl_flag() {
    local var="$2"
    if [ -n "${!var:-}" ]; then
        BRL_ARGS+=("$1" "${!var}")
    fi
}

# The LoRA flags for the three trainers this repo writes itself. One
# spelling for all of them, and none at all when LoRA is off, so a full
# fine-tune stays the default and each trainer's own argparse default is
# still the only place those values are written down.
#
# bin/rl.sh does not use this: verl takes hydra overrides rather than
# flags, so its block spells the same four knobs as
# actor_rollout_ref.model.* settings.
brl_lora_flags() {
    # Forward 4-bit even when no adapter was requested so the trainer can
    # reject that invalid combination explicitly instead of this helper
    # silently swallowing it.
    if ! brl_lora_on; then
        case "${LORA_4BIT:-}" in
            1|true|yes|on) BRL_ARGS+=(--lora-4bit) ;;
        esac
        return 0
    fi
    brl_flag --lora-rank LORA_RANK
    brl_flag --lora-alpha LORA_ALPHA
    brl_flag --lora-targets LORA_TARGETS
    brl_flag --lora-exclude LORA_EXCLUDE
    brl_flag --lora-adapter LORA_ADAPTER
    case "${LORA_4BIT:-}" in
        1|true|yes|on) BRL_ARGS+=(--lora-4bit) ;;
    esac
}
