#!/usr/bin/env bash
# Tool-call-format supervised fine-tuning, often used to initialize RL but
# also a trainer in its own right. Trains rows written by `bedrock generate`
# sft with bedrock_rl/train/sft.py and freezes the vision tower.
#   bash bin/sft.sh [TASK] [MODEL] [STEPS]
# MODEL takes any registry alias, HuggingFace id, or local path (see
# `python -m bedrock_rl.models --list`).
# SAVE_EVERY writes a resumable checkpoint every N optimizer steps and
# keeps the two most recent, and RESUME=auto continues from the newest one
# under OUT. Both default to off, which saves only at the end. A run
# interrupted without them loses everything it did.
#
# Env knobs: SAVE_EVERY, RESUME, VERL_ENV (training venv from
# bin/setup_deps.sh), NETHERITE_HOME, CUDA_VISIBLE_DEVICES, RUN_TAG, DATA,
# OUT, DEEPSPEED, LORA_RANK, LORA_ALPHA, LORA_TARGETS, LORA_EXCLUDE,
# LORA_ADAPTER, METRICS, WANDB, WANDB_PROJECT, WANDB_ENTITY and
# BEDROCK_PLAIN.
#
# LORA_RANK trains an adapter instead of the whole model, which is what
# lets SFT run on a card that cannot hold weights, gradients and fp32 Adam
# moments at once. OUT is then an adapter directory naming its own base
# model, and bin/rl.sh takes it as MODEL and carries on. The same LoRA knobs
# mean the same thing in every launcher here.
#
# DEEPSPEED names a ZeRO preset in bedrock_rl/templates/deepspeed
# (z2, z2_offload, z3, z3_offload) or a config path. It covers this SFT
# trainer only; RL uses verl and FSDP.
#   DEEPSPEED=z3 CUDA_VISIBLE_DEVICES=0,1 bash bin/sft.sh
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=bin/_common.sh
# `set -euo pipefail` lives in that file, so until it is sourced NOTHING
# here is fatal. A source that fails leaves every function undefined and
# every variable unset, and the script runs to the end on empty strings
# and reaches the trainer invocation with no model, no task and no output
# directory. So the source is the one line that checks itself.
. "$SOURCE_DIR/_common.sh" || {
    echo "cannot source $SOURCE_DIR/_common.sh, which every launcher in" \
         "bin/ needs; run this from a checkout of bedrock-rl" >&2
    exit 1
}

brl_setup auto "$@"
STEPS="${3:-30}"
BRL_TAG="${RUN_TAG:-sft}"

brl_resolve_model
brl_paths "$BRL_TAG"
DATA="${DATA:-$BRL_ROOT/data/${BRL_TASK_NAME}_cu/sft.parquet}"

# DeepSpeed refuses to run outside a distributed launch even on one GPU,
# and more than one GPU needs torchrun whether or not DeepSpeed is in play.
# One GPU with no DeepSpeed stays on plain python.
BRL_ARGS=()
brl_flag --lr LR
brl_flag --batch BATCH
brl_flag --grad-accum GRAD_ACCUM
brl_flag --keep-latest-images BRL_KEEP_LATEST_IMAGES
if [ -n "${DEEPSPEED:-}" ]; then
    BRL_ARGS+=(--deepspeed "$DEEPSPEED")
fi
brl_lora_flags
if [ -n "${DEEPSPEED:-}" ] || [ "$BRL_NGPU" -gt 1 ]; then
    BRL_LAUNCH=(torchrun --standalone --nproc_per_node="$BRL_NGPU")
else
    BRL_LAUNCH=(python)
fi

# `torchrun` enables argparse abbreviations and consumes `--run` as an
# ambiguous prefix of its own --run-path before the training module can see
# it. The unambiguous long alias works for both plain Python and DDP.
brl_run "${BRL_LAUNCH[@]}" -m bedrock_rl.train.sft \
    --data "$DATA" \
    --model "$MODEL" \
    --out "$OUT" \
    --steps "$STEPS" --freeze-vision "${BRL_ARGS[@]+"${BRL_ARGS[@]}"}" \
    ${SAVE_EVERY:+--save-every "$SAVE_EVERY"} \
    ${RESUME:+--resume "$RESUME"} \
    --run-name "$BRL_SLUG" \
    --metrics "$BRL_METRICS" \
    2>&1 | tee -a "$BRL_LOG"
echo "pass $OUT as MODEL to bin/rl.sh"
