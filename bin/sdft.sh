#!/usr/bin/env bash
# On-policy Self-Distillation Fine-Tuning (Shenfeld et al., 2026).
# Student trajectories are generated without demonstrations. The self-teacher
# re-evaluates those exact tokens with an expert demonstration in context.
#   bash bin/sdft.sh [TASK] [MODEL] [STEPS]
# Required data:
#   DATA             RL prompt parquet with episode specs
#   DEMONSTRATIONS   expert trajectory JSON/JSONL or SFT parquet
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SOURCE_DIR/_common.sh" || {
    echo "cannot source $SOURCE_DIR/_common.sh" >&2
    exit 1
}

brl_setup auto "$@"
STEPS="${3:-30}"
BRL_TAG="${RUN_TAG:-sdft}"
brl_resolve_model
brl_paths sdft "$BRL_TAG"
DATA="${DATA:-$BRL_ROOT/data/${BRL_TASK_NAME}_cu/rl_prompts.parquet}"
DEMONSTRATIONS="${DEMONSTRATIONS:-$BRL_ROOT/data/${BRL_TASK_NAME}_cu/sft.parquet}"

BRL_ARGS=()
for brl_kv in --prompts:PROMPTS --lr:LR \
              --teacher-mixup-alpha:TEACHER_MIXUP_ALPHA \
              --demonstration-match:DEMONSTRATION_MATCH \
              --skip-tokens:SKIP_TOKENS \
              --temperature:TEMPERATURE \
              --max-new-tokens:MAX_NEW_TOKENS --workers:WORKERS \
              --rollout-engine:ROLLOUT_ENGINE \
              --vllm-gpu-frac:VLLM_GPU_FRAC \
              --vllm-max-model-len:VLLM_MAX_MODEL_LEN; do
    brl_flag "${brl_kv%%:*}" "${brl_kv##*:}"
done
brl_lora_flags

cd "$BRL_ROOT"
brl_run python -m bedrock_rl.train.sdft_trainer \
    --task "$(brl_task_yaml "$TASK")" \
    --data "$DATA" \
    --demonstrations "$DEMONSTRATIONS" \
    --model "$MODEL" \
    --out "$OUT" \
    --steps "$STEPS" \
    --save-every "${SAVE_EVERY:-$STEPS}" \
    "${BRL_ARGS[@]+"${BRL_ARGS[@]}"}" \
    --run "$BRL_SLUG" \
    --metrics "$BRL_METRICS" \
    2>&1 | tee -a "$BRL_LOG"
