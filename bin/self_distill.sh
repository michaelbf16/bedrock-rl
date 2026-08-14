#!/usr/bin/env bash
# Self-distillation workflow: harvest successful trajectories under a hint, then
# fine-tune on them with the hint deleted.
#   bash bin/self_distill.sh [TASK] [MODEL] [EPISODES]
# MODEL is a registry family alias, a size alias, a HuggingFace id, or a
# local checkpoint directory. Harvest gating is the
# live engine-state reward, never a self-label. The harvest yield is a
# result in its own right, and it lands in the metrics file as well as in
# the closing summary.
# Env knobs: VERL_ENV, NETHERITE_HOME, CUDA_VISIBLE_DEVICES, RUN_TAG,
# HINT_LEVEL, FINISH_HINT, TURN_HINT (off|weak|direction|oracle),
# TURN_HINT_P, EPOCHS, LR, OUT, DATA, HARVEST_ONLY,
# LORA_RANK, LORA_ALPHA, LORA_TARGETS, LORA_EXCLUDE, LORA_ADAPTER,
# ROLLOUT_ENGINE, VLLM_GPU_FRAC, VLLM_MAX_MODEL_LEN,
# METRICS (per-step JSON lines, defaults under logs/), WANDB=1 with
# WANDB_PROJECT and WANDB_ENTITY, BEDROCK_PLAIN=1 for plain output.
# The two halves take different LoRA knobs because they are different
# jobs. The harvest takes no gradient, so only LORA_ADAPTER reaches it,
# naming the adapter it samples with. The fine-tune takes them all.
# ROLLOUT_ENGINE reaches the harvest for the same reason: it says how
# trajectories are SAMPLED, and the fine-tune samples nothing.
# ROLLOUT_ENGINE=vllm harvests with vLLM instead of HuggingFace generate,
# which is the whole of the harvest's cost. Nothing here takes a
# gradient, so it needs no weight refresh and no LoRA.
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
# driver, including harvest output and standalone evaluation.
EPISODES="${3:-256}"
# The hint level names the harvest file, so it is a path input here and
# not only a flag, which is why this one keeps its default in the shell.
BRL_LEVEL="${HINT_LEVEL:-procedure}"
BRL_TAG="${RUN_TAG:-self_distill}"

brl_resolve_model
brl_paths self_distill "$BRL_TAG"
DATA="${DATA:-$BRL_ROOT/data/${BRL_TASK_NAME}_cu/rl_prompts.parquet}"
# The per-turn rung is a PATH input for the same reason the hint level is:
# `off` and `direction` at one HINT_LEVEL and RUN_TAG are the two arms of
# the ablation the rung exists for, and one file name would have the
# second silently replace the first.
# an `if`, not a `&&` chain: `set -e` is on in bin/_common.sh and a
# chain whose first test fails returns 1, which would abort the launcher
BRL_RUNG=""
if [ -n "${TURN_HINT:-}" ] && [ "${TURN_HINT}" != off ]; then
    BRL_RUNG="_turn${TURN_HINT}"
fi
BRL_SET="$BRL_ROOT/data/${BRL_TASK_NAME}_cu/self_distill_${BRL_LEVEL}${BRL_RUNG}_${BRL_TAG}.parquet"

BRL_ARGS=()
brl_flag --finish-hint FINISH_HINT
# the per-turn, state-conditional channel (bedrock_rl/sdg/guidance.py).
# Unset means off, so nothing changes for a caller that does not ask.
brl_flag --turn-hint TURN_HINT
brl_flag --turn-hint-p TURN_HINT_P
brl_flag --lora-adapter LORA_ADAPTER
brl_flag --rollout-engine ROLLOUT_ENGINE
brl_flag --vllm-gpu-frac VLLM_GPU_FRAC
brl_flag --vllm-max-model-len VLLM_MAX_MODEL_LEN

cd "$BRL_ROOT"
brl_run python -m bedrock_rl.train.self_distill \
    --task "$(brl_task_yaml "$TASK")" \
    --data "$DATA" \
    --model "$MODEL" \
    --out "$BRL_SET" \
    --episodes "$EPISODES" \
    --batch "${HARVEST_BATCH:-8}" \
    --hint-level "$BRL_LEVEL" \
    "${BRL_ARGS[@]+"${BRL_ARGS[@]}"}" \
    --run "${BRL_SLUG}_harvest" \
    --metrics "${BRL_METRICS%.jsonl}_harvest.jsonl" \
    2>&1 | tee -a "$BRL_LOG"

if [ ! -f "$BRL_SET" ]; then
    # A full run cannot train without accepted trajectories. Harvest-only mode
    # reports the measured yield without treating an empty result as an error.
    cat >&2 <<EOF

empty harvest after $EPISODES episodes on $TASK at hint level $BRL_LEVEL.
Nothing was written to $BRL_SET.

Self-distillation trains only on trajectories that pass the task verifier, and
this harvest produced none.

Self-distillation needs a starting point whose success rate is above
zero. Choose one of these.
  run SFT first, then harvest from that checkpoint
    bash bin/sft.sh $TASK $MODEL
    MODEL=\$HOME/ckpts/bedrock_${BRL_TASK_NAME}_sft bash bin/self_distill.sh
  harvest on a simpler task the model can already finish sometimes
  read the yield per hint level before committing a run
    HARVEST_ONLY=1 HINT_LEVEL=coarse bash bin/self_distill.sh
    The harvest yield is a result; inspect it before committing a run.

EOF
    [ -n "${HARVEST_ONLY:-}" ] && exit 0
    echo "no fine-tune ran" >&2
    exit 1
fi
[ -n "${HARVEST_ONLY:-}" ] && exit 0

BRL_ARGS=()
brl_flag --epochs EPOCHS
brl_flag --lr LR
brl_lora_flags

brl_run python -m bedrock_rl.train.sft \
    --data "$BRL_SET" \
    --model "$MODEL" \
    --out "$OUT" \
    --freeze-vision \
    "${BRL_ARGS[@]+"${BRL_ARGS[@]}"}" \
    --run "${BRL_SLUG}_finetune" \
    --metrics "${BRL_METRICS%.jsonl}_finetune.jsonl" \
    2>&1 | tee -a "$BRL_LOG"
