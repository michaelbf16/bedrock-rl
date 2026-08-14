#!/usr/bin/env bash
# SDPO trainer: standalone multi-turn self-distillation RL, a peer of the
# verl trainer rather than a mode of it.
#   bash bin/sdpo.sh [TASK] [MODEL] [STEPS]
# MODEL is a registry family alias (qwen3-vl), a size alias
# (qwen3-vl-4b), a HuggingFace id, or a local checkpoint directory; see
# `python -m bedrock_rl.models --list`.
# Episodes come from data/<task>_cu/rl_prompts.parquet, which supplies
# seeds and start poses only, so the hint level stays a trainer flag
# rather than a property of the data. Student rollouts are hint-free;
# static and per-turn feedback are added only to the self-teacher view.
# Reward comes directly from the live episode that received the actions.
# Env knobs: VERL_ENV, NETHERITE_HOME, CUDA_VISIBLE_DEVICES, RUN_TAG,
# PROMPTS, GROUP_SIZE, LR, ALPHA, TOPK, IS_CLIP, HINT_LEVEL, FINISH_HINT,
# TURN_HINT (off|weak|direction|oracle) and TURN_HINT_P,
# TEMPERATURE, MAX_NEW_TOKENS, WORKERS,
# EVAL_EVERY, EVAL_EPISODES, SAVE_EVERY, MAX_DRY_STEPS, OUT, DATA,
# LORA_RANK, LORA_ALPHA, LORA_TARGETS, LORA_EXCLUDE, LORA_ADAPTER,
# ROLLOUT_ENGINE, VLLM_GPU_FRAC, VLLM_MAX_MODEL_LEN,
# METRICS (per-step JSON lines, defaults under logs/), WANDB=1 with
# WANDB_PROJECT and WANDB_ENTITY, BEDROCK_PLAIN=1 for plain output.
# LORA_RANK trains an adapter instead of the whole model, and the
# checkpoints this writes are then adapter directories.
# ROLLOUT_ENGINE=vllm uses paged KV caching and continuous batching instead of
# HuggingFace generate. It requires LORA_RANK because the public per-step
# refresh API serves adapters, not arbitrary full-model weight replacements.
# VLLM_GPU_FRAC controls how much memory the colocated engine may reserve.
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

# Context selection happens before model rendering in the shared rollout
# driver, so the same BRL_KEEP_LATEST_IMAGES policy reaches sampling,
# student/teacher views, harvesting, and standalone evaluation.
STEPS="${3:-30}"
BRL_TAG="${RUN_TAG:-sdpo}"

brl_resolve_model
brl_paths sdpo "$BRL_TAG"
DATA="${DATA:-$BRL_ROOT/data/${BRL_TASK_NAME}_cu/rl_prompts.parquet}"

# Every knob below is unset by default, so bedrock_rl/train/sdpo_trainer.py's
# own argparse default is the only place that value is written down.
# SAVE_EVERY is the one that diverges on purpose: the trainer defaults to
# never saving, and a launcher run has to leave a checkpoint behind.
BRL_ARGS=()
for brl_kv in --prompts:PROMPTS --group-size:GROUP_SIZE --lr:LR \
              --alpha:ALPHA --topk:TOPK --is-clip:IS_CLIP \
              --teacher-update-rate:TEACHER_UPDATE_RATE \
              --hint-level:HINT_LEVEL --finish-hint:FINISH_HINT \
              --turn-hint:TURN_HINT --turn-hint-p:TURN_HINT_P \
              --temperature:TEMPERATURE \
              --max-new-tokens:MAX_NEW_TOKENS --workers:WORKERS \
              --eval-every:EVAL_EVERY --eval-episodes:EVAL_EPISODES \
              --max-dry-steps:MAX_DRY_STEPS \
              --rollout-engine:ROLLOUT_ENGINE --vllm-gpu-frac:VLLM_GPU_FRAC \
              --vllm-max-model-len:VLLM_MAX_MODEL_LEN; do
    brl_flag "${brl_kv%%:*}" "${brl_kv##*:}"
done
brl_lora_flags

cd "$BRL_ROOT"
brl_run python -m bedrock_rl.train.sdpo_trainer \
    --task "$(brl_task_yaml "$TASK")" \
    --data "$DATA" \
    --model "$MODEL" \
    --out "$OUT" \
    --steps "$STEPS" \
    --save-every "${SAVE_EVERY:-$STEPS}" \
    "${BRL_ARGS[@]+"${BRL_ARGS[@]}"}" \
    --run "$BRL_SLUG" \
    --metrics "$BRL_METRICS" \
    2>&1 | tee -a "$BRL_LOG"
