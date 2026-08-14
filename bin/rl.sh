#!/usr/bin/env bash
# COMPUTER-USE multi-turn RL: the model sees the rendered frame, acts
# through the `computer` tool, and gets a fresh frame and observation back
# after every call.
#   bash bin/rl.sh [TASK] [MODEL] [STEPS]
# MODEL is a registry family alias, size alias, HuggingFace id or local
# checkpoint path; everything family-specific comes from that entry.
# Data: `bedrock generate` -> data/<task>_cu/rl_prompts.parquet.
# Hydra config is generated per family into outputs/configs/, untracked
# runtime output. CONFIG_NAME and CONFIG_DIR take a hand-written config
# instead; none ships, so copy a generated one.
# Reward comes from the live episode owned by the custom agent loop.  There is
# no post-hoc text replay and no executor/scorer ledger to keep synchronized.
#
# Everything else is an environment knob, read where it is used below.
#
# NNODES              nodes training spans, default 1. Above one it needs
#                     RAY_ADDRESS and an identically provisioned cluster.
# GPUS_PER_NODE       GPUs per node, default the count this box derives
#                     from CUDA_VISIBLE_DEVICES.
# NETHERITE_ENV_GPUS  read only when NETHERITE_RASTER=cuda, and it MUST
#                     NOT overlap CUDA_VISIBLE_DEVICES: env processes
#                     rasterizing on the trainer's GPU cost a large
#                     multiple of the step time.
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

# BEFORE anything is set up, because this is a refusal and not a
# setting. LORA_4BIT quantizes the frozen base an adapter sits on, and
# there is nowhere to put it here: verl 0.8 declares no quantization
# field on actor_rollout_ref.model, its FSDP wrapper cannot shard
# bitsandbytes Params4bit, and the vLLM engine on the same card holds
# its own bf16 copy of the base either way, so a 4-bit actor would save
# a third of the smaller half. SFT, including self_distill's SFT phase,
# is the standalone path that implements it.
#
# Refused rather than ignored: a knob that silently does nothing on one
# of four trainers is the kind of thing this repo is careful about.
case "${LORA_4BIT:-}" in
    1|true|yes|on)
        echo "error  LORA_4BIT is not available on the verl RL path." \
             "verl 0.8 declares no quantization setting for the actor and" \
             "its FSDP wrapper cannot shard 4-bit parameters." \
             "bin/sft.sh and bin/self_distill.sh's SFT phase do take it." >&2
        exit 2;;
esac

# FIRST, because the raster block below reads CUDA_VISIBLE_DEVICES and
# brl_setup is what exports it. Under `set -u` reading it first is not a
# wrong default, it is an unbound-variable abort.
brl_setup auto "$@"
# before any GPU is touched, because the alternative is finding out after
# the first rollout batch
brl_require_flash_attn

export NETHERITE_RASTER="${NETHERITE_RASTER:-cpu}"
if [ "$NETHERITE_RASTER" = cuda ]; then
    if [ -z "${NETHERITE_ENV_GPUS:-}" ]; then
        echo "error  NETHERITE_RASTER=cuda needs NETHERITE_ENV_GPUS set to" \
             "gpu ids the trainer does not use" \
             "(CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)" >&2
        exit 1
    fi
    export NETHERITE_ENV_GPUS
    for g in ${NETHERITE_ENV_GPUS//,/ }; do
        case ",$CUDA_VISIBLE_DEVICES," in
            *",$g,"*)
                echo "error  NETHERITE_ENV_GPUS=$NETHERITE_ENV_GPUS overlaps" \
                     "the trainer's CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES;" \
                     "use disjoint devices to avoid simulator contention" >&2
                exit 1;;
        esac
    done
fi
export VERL_HOME="${VERL_HOME:-$HOME/verl}"
export HYDRA_FULL_ERROR=1
# ── how many nodes training spans ────────────────────────────────────────
# A third axis, independent of where the game runs and of where the
# trainer runs. verl has always driven several nodes through Ray; this
# launcher pinned nnodes to 1, so the path was unreachable from here.
# NNODES defaults to 1 and GPUS_PER_NODE defaults to the count this box
# derived from CUDA_VISIBLE_DEVICES, so a single-node run is unchanged.
#
# Above one node you start Ray yourself and point this at it. Every node
# needs the SAME engine at PINNED_SHA with every shipped patch, the same
# training venv, and the same model weights on a path that resolves
# there; `bedrock doctor` on each node is the check. RAY_ADDRESS is
# what the driver connects to.
NNODES="${NNODES:-1}"
BRL_NGPU="${GPUS_PER_NODE:-$BRL_NGPU}"
BRL_VISIBLE_GPUS_PER_NODE="$BRL_NGPU"
BRL_METHOD="${BRL_DISTILLATION_MODE:-rl}"
BRL_DISTILLATION_ARGS=()
if [ "$BRL_METHOD" != rl ]; then
    if [ "$BRL_METHOD" != opd ] && [ "$BRL_METHOD" != mopd ]; then
        echo "unknown BRL_DISTILLATION_MODE=$BRL_METHOD" >&2
        exit 2
    fi
    if [ -z "${BRL_TEACHERS_JSON:-}" ]; then
        echo "trainer: $BRL_METHOD needs a non-empty teachers: mapping" >&2
        exit 2
    fi
    TEACHER_NNODES="${TEACHER_NNODES:-$NNODES}"
    if [ "$TEACHER_NNODES" -ne "$NNODES" ]; then
        echo "TEACHER_NNODES must equal NNODES because the student and" \
             "teacher pools share one Ray/Modal allocation." >&2
        exit 2
    fi
    BRL_MIN_TEACHER_GPUS="$(brl_run python -m \
        bedrock_rl.adapters.verl.distillation default-gpus \
        --mode "$BRL_METHOD" --teachers-json "$BRL_TEACHERS_JSON" \
        --nnodes "$TEACHER_NNODES")"
    BRL_TARGET_TRAIN_BS="${TRAIN_BS:-8}"
    BRL_TEACHER_GPUS_EXPLICIT=0
    [ -z "${TEACHER_GPUS_PER_NODE:-}" ] || BRL_TEACHER_GPUS_EXPLICIT=1
    if [ -z "${TEACHER_GPUS_PER_NODE:-}" ]; then
        if [ -n "${ACTOR_GPUS_PER_NODE:-}" ] && [ "$BRL_METHOD" = opd ]; then
            TEACHER_GPUS_PER_NODE=$((BRL_VISIBLE_GPUS_PER_NODE - ACTOR_GPUS_PER_NODE))
        else
            TEACHER_GPUS_PER_NODE="$BRL_MIN_TEACHER_GPUS"
        fi
    fi
    if [ -z "${ACTOR_GPUS_PER_NODE:-}" ]; then
        ACTOR_GPUS_PER_NODE=$((BRL_VISIBLE_GPUS_PER_NODE - TEACHER_GPUS_PER_NODE))
        while (( ACTOR_GPUS_PER_NODE > 0 )); do
            brl_teacher_candidate="$TEACHER_GPUS_PER_NODE"
            if [ "$BRL_METHOD" = opd ] \
               && [ "$BRL_TEACHER_GPUS_EXPLICIT" = 0 ]; then
                brl_teacher_candidate=$((BRL_VISIBLE_GPUS_PER_NODE - ACTOR_GPUS_PER_NODE))
            fi
            if (( BRL_TARGET_TRAIN_BS % (ACTOR_GPUS_PER_NODE * NNODES) == 0
                  && (brl_teacher_candidate * TEACHER_NNODES)
                     % BRL_MIN_TEACHER_GPUS == 0 )); then
                TEACHER_GPUS_PER_NODE="$brl_teacher_candidate"
                break
            fi
            ACTOR_GPUS_PER_NODE=$((ACTOR_GPUS_PER_NODE - 1))
        done
    fi
    if (( ACTOR_GPUS_PER_NODE < 1 || TEACHER_GPUS_PER_NODE < 1
          || ACTOR_GPUS_PER_NODE + TEACHER_GPUS_PER_NODE > BRL_VISIBLE_GPUS_PER_NODE )); then
        echo "student ($ACTOR_GPUS_PER_NODE) + teacher" \
             "($TEACHER_GPUS_PER_NODE) GPUs must fit the" \
             "$BRL_VISIBLE_GPUS_PER_NODE visible GPUs on every node" >&2
        exit 2
    fi
    if (( BRL_TARGET_TRAIN_BS % (ACTOR_GPUS_PER_NODE * NNODES) != 0 )); then
        echo "TRAIN_BS=$BRL_TARGET_TRAIN_BS must be divisible by the student" \
             "world size ACTOR_GPUS_PER_NODE*NNODES=" \
             "$((ACTOR_GPUS_PER_NODE * NNODES))." >&2
        exit 2
    fi
    BRL_NGPU="$ACTOR_GPUS_PER_NODE"
fi
if [ "$NNODES" -gt 1 ]; then
    if [ -z "${RAY_ADDRESS:-}" ]; then
        echo "NNODES=$NNODES needs RAY_ADDRESS pointing at the head node" \
             "(ray start --head on one box, ray start --address on the" \
             "rest, then RAY_ADDRESS=ray://<head>:10001 or the GCS" \
             "address)" >&2
        exit 1
    fi
    # Every node owns local engine processes; no environment service sits
    # between reward code and the game state it reads.
    echo "note: NNODES=$NNODES means every node needs the pinned, patched" \
         "Netherite engine built locally" >&2
    if [ -n "${NETHERITE_ENV_GPUS:-}" ]; then
        echo "note: NETHERITE_ENV_GPUS names device ids on EVERY node, and" \
             "the overlap check below only sees this one" >&2
    fi
fi
# Ray sees that the driver started under uv run and tries to rebuild the
# driver env on every worker. verl already passes runtime_env.working_dir
# as None, which that path rejects, and the training venv is shared by the
# workers anyway, so turn the rebuild off.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
# Quieting, not silencing. verl reads VERL_LOGGING_LEVEL in 94 modules and
# defaults it to WARN in most and INFO in the checkpoint managers; naming
# it once makes the whole trainer consistent. RAY_DEDUP_LOGS is what stops
# one worker warning from arriving once per rank. Every one of these is an
# override, so VERL_LOGGING_LEVEL=INFO bash bin/rl.sh gets it all back,
# and nothing here touches stderr from a crash.
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-WARN}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-warning}"
export DATASETS_VERBOSITY="${DATASETS_VERBOSITY:-warning}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
STEPS="${3:-30}"
ADV="${ADV:-grpo}"
# POLICY LOSS MODE, a second and orthogonal axis to ADV.
#
# ADV picks the ADVANTAGE ESTIMATOR (grpo, rloo, reinforce_plus_plus,
# remax, ...). LOSS_MODE picks how the policy ratio is formed and clipped.
# They compose: GSPO is not a replacement for GRPO, it is GRPO advantages
# with a SEQUENCE-level importance ratio instead of a token-level one.
#
# GSPO (arXiv 2507.18071) forms one importance ratio per sequence and can be
# useful for long multi-turn trajectories where token-level ratios accumulate
# variance. verl 0.8 implements it at
#   verl/trainer/ppo/core_algos.py:1538  @register_policy_loss("gspo")
# and it wants loss_agg_mode="seq-mean-token-mean", which its own
# docstring recommends and which this sets automatically.
#
# verl also exposes cispo, geo_mean, clip_cov, kl_cov, dppo_kl, dppo_tv, gpg,
# and sapo; set LOSS_MODE explicitly to select one.
LOSS_MODE="${LOSS_MODE:-vanilla}"
# GSPO's own docstring: "For GSPO, it is recommended to use
# seq-mean-token-mean". Anything else silently changes what the loss means.
if [ "$LOSS_MODE" = gspo ]; then
    LOSS_AGG_MODE="${LOSS_AGG_MODE:-seq-mean-token-mean}"
else
    LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"
fi

BRL_TAG="${RUN_TAG:-cu}"
# The task owns the turn budget. verl's rollout loop and the episode have
# to agree on it: a loop that stops at 4 turns on a task whose YAML says 8
# cuts every trajectory before the episode can finish, and the reward
# replay then scores a truncated one.
TASK_MAX_TURNS="$(brl_task_get "$TASK" max_turns 6)"
if [ -n "${MAX_TURNS:-}" ] && [ "$MAX_TURNS" != "$TASK_MAX_TURNS" ]; then
    echo "error  MAX_TURNS=$MAX_TURNS disagrees with the task's" \
         "max_turns: $TASK_MAX_TURNS. Put the episode budget in the task" \
         "YAML so rollout, reward and evaluation share one contract." >&2
    exit 2
fi
MAX_TURNS="$TASK_MAX_TURNS"
# The custom loop reads the budget from this same task YAML and executes up to
# the shared per-turn call cap, resolving any excess as skipped and counting
# that assistant message as one turn. The live episode is therefore the sole
# authority for termination and reward. verl's
# multi-turn schema still requires its own assistant/user ceilings even though
# the custom loop does not use them for episode termination.  Keep those one
# above the live cap so the episode, not a framework pre-check, is binding.
BRL_VERL_TURNS="$((MAX_TURNS + 1))"

# One registry lookup decides every model-specific setting below.
#
# --snapshot because this launcher TRAINS. Without it MODEL stays a
# HuggingFace id and every loader downstream resolves that id for itself,
# at load time, against a branch that can move between two runs of the
# same config: the registry's pinned revision would then describe the run
# without binding it, which is the failure that looks most like success.
# With it, MODEL is the local directory that revision unpacks to, and
# that one path is what the actor loads, what the colocated rollout
# engine serves, and what the family's override table below is read out
# of. The bytes were going to be fetched either way; this fetches them
# before a GPU is held, and pins which ones.
brl_resolve_model --max-turns "$MAX_TURNS" --snapshot
# BRL_MODEL_ID and not MODEL, because MODEL is now a snapshot directory
# named after a commit hash and this names the run: its checkpoint
# directory, its logs and its experiment name.
brl_paths "$([ "$BRL_METHOD" = rl ] && echo "$ADV" || echo "$BRL_METHOD")" \
    "$BRL_TAG" "$(basename "$BRL_MODEL_ID")"

# ── per-rollout evidence, ON BY DEFAULT ──────────────────────────────────
# Aggregate metrics cannot distinguish policies that receive the same score
# through different behaviors, so portable rollout artifacts are enabled by
# default. Each JSON file contains the
# canonical OpenAI chat, exact per-turn model views, state/event traces,
# reward components, and provenance. Point BRL_TRAJECTORY_DIR at an empty
# string to disable them for a throughput-only run.
# No colon in ${name-default}: an explicitly empty value means disabled.
export BRL_TRAJECTORY_DIR="${BRL_TRAJECTORY_DIR-${BRL_METRICS%.jsonl}_trajectories}"
# Every record names its run, so dumps concatenated across arms stay
# separable. BRL_ prefixed because it is the launcher's own identity for
# the run rather than anybody's to set.
export BRL_SLUG
# verl's OWN rollout dump, off by default. trainer.rollout_data_dir makes
# RayPPOTrainer._dump_generations write <step>.jsonl of the decoded
# prompt, decoded response and score for every rollout. The canonical
# artifact above is usually the more useful record because it also carries
# exact views, structured calls, engine state and reward components. `++`
# works whether the generated config already declares the key or not.
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-}"

# A trajectory carries one frame at t0 plus one per executed tool call.
# BRL_LIMIT_IMAGES does not trim that list and never did: it tells vLLM
# how many to ACCEPT, and one too few ends the rollout with "At most N
# image(s) may be provided in one prompt". The shared per-turn call cap
# supplies the safe ceiling without making vLLM profile thousands of images.
#
# BRL_KEEP_LATEST_IMAGES is the trimmer. Unset or 0 keeps every frame; a
# positive value sends only the newest N frames. Smaller windows reduce repeated
# vision encoding, but tasks that depend on visual history should keep enough
# frames to preserve that context.
#
# It is exported because both the custom verl adapter and standalone
# sampling read it. The policy operates on typed OpenAI image parts before
# model-specific rendering; no tokenizer constants or upstream patch are
# involved.
export BRL_KEEP_LATEST_IMAGES="${BRL_KEEP_LATEST_IMAGES:-0}"
if [ "$BRL_KEEP_LATEST_IMAGES" -gt 0 ] \
        && [ "$BRL_KEEP_LATEST_IMAGES" -lt "$BRL_LIMIT_IMAGES" ]; then
    BRL_LIMIT_IMAGES="$BRL_KEEP_LATEST_IMAGES"
fi
#
# The family's own budget is about CONTEXT rather than about acceptance,
# so exceeding it is a prompt-length risk and not a cap.
if [ "${BRL_KEEP_LATEST_IMAGES:-0}" != 0 ]; then
    echo "frames   keeping the newest $BRL_KEEP_LATEST_IMAGES per trajectory" \
         "(BRL_KEEP_LATEST_IMAGES)" >&2
fi
# verl truncates response_ids to data.max_response_length. Scale the default
# with the turn budget while keeping a 4096-token floor for short tasks.
RESPONSE_LEN="${RESPONSE_LEN:-$(( MAX_TURNS * 512 > 4096 ? MAX_TURNS * 512 : 4096 ))}"
# vLLM's rollout context has to hold the prompt and the whole response, so
# it follows those two budgets rather than a constant that silently caps
# them.
BRL_MODEL_LEN="${MODEL_LEN:-$(( BRL_MAX_PROMPT_LEN + RESPONSE_LEN > 8192 ? BRL_MAX_PROMPT_LEN + RESPONSE_LEN : 8192 ))}"
BRL_VERL_OVERRIDE_ARGS=()
while IFS= read -r override; do
    [ -z "$override" ] || BRL_VERL_OVERRIDE_ARGS+=("$override")
done <<< "$BRL_VERL_OVERRIDES"
if [ "$BRL_METHOD" != rl ]; then
    BRL_TEACHER_PROVENANCE="$OUT/teachers.provenance.json"
    BRL_TEACHERS_JSON="$(brl_run python -m \
        bedrock_rl.adapters.verl.distillation stage-teachers \
        --mode "$BRL_METHOD" --teachers-json "$BRL_TEACHERS_JSON" \
        --provenance-out "$BRL_TEACHER_PROVENANCE")"
    if [ "${SKIP_TEACHER_TOKENIZER_CHECK:-0}" != 1 ]; then
        brl_run python -m bedrock_rl.adapters.verl.distillation check-tokenizers \
            --mode "$BRL_METHOD" --teachers-json "$BRL_TEACHERS_JSON" \
            --student "$MODEL"
    fi
    BRL_DISTILLATION_TEXT="$(brl_run python -m \
        bedrock_rl.adapters.verl.distillation hydra \
        --mode "$BRL_METHOD" --teachers-json "$BRL_TEACHERS_JSON" \
        --teacher-gpus-per-node "$TEACHER_GPUS_PER_NODE" \
        --teacher-nnodes "$TEACHER_NNODES" \
        --teacher-key "${TEACHER_KEY:-data_source}" \
        --max-model-len "$BRL_MODEL_LEN" \
        --loss-mode "${DISTILLATION_LOSS_MODE:-k1}" \
        --topk "${DISTILLATION_TOPK:-64}" \
        --use-task-rewards "${USE_TASK_REWARDS:-False}" \
        --use-policy-gradient "${USE_POLICY_GRADIENT:-True}" \
        --coefficient "${DISTILLATION_LOSS_COEF:-1.0}" \
        --loss-max-clamp "${LOSS_MAX_CLAMP:-10.0}" \
        --log-prob-min-clamp "${LOG_PROB_MIN_CLAMP:--10.0}")"
    while IFS= read -r brl_override; do
        [ -n "$brl_override" ] && BRL_DISTILLATION_ARGS+=("$brl_override")
    done <<< "$BRL_DISTILLATION_TEXT"
fi
DATA_DIR="${DATA_DIR:-$BRL_ROOT/data/${BRL_TASK_NAME}_cu}"
AGENT_LOOP_CONFIG="${AGENT_LOOP_CONFIG:-$BRL_ROOT/bedrock_rl/templates/verl/bedrock_agent.yaml}"
AGENT_LOOP="${AGENT_LOOP:-bedrock}"
# Token-budget batching. verl/workers/config/actor.py gates the fixed
# micro-batch path on
#   if not self.use_dynamic_bsz:
# so dynamic mode ignores ``*_micro_batch_size_per_gpu`` and packs sequences
# up to DYN_MAX_TOKEN. Tune the budget against the longest prompt, response,
# and image sequence while leaving memory for vLLM to restore its KV cache
# between training and rollout phases.
DYN_BSZ="${DYN_BSZ:-0}"
DYN_MAX_TOKEN="${DYN_MAX_TOKEN:-32768}"
MICRO_BS="${MICRO_BS:-4}"
TRAIN_BS="${TRAIN_BS:-8}"
ROLLOUT_N="${ROLLOUT_N:-4}"
if [ "$BRL_METHOD" = rl ] \
   && { [ "$ADV" = grpo ] || [ "$ADV" = rloo ]; } \
   && (( ROLLOUT_N < 2 )); then
    echo "$ADV needs ROLLOUT_N>=2. With one sample its group baseline" \
         "removes the entire learning signal." >&2
    exit 2
fi
# verl chunks the generated batch equally across agent-loop workers. Its
# upstream default is always eight, which crashes a legitimate 2x2 RLOO
# smoke only after model startup. Auto chooses the largest worker count up
# to eight that divides this run's actual generated batch.
if [[ "${AGENT_WORKERS:-auto}" == auto ]]; then
    _brl_generated_batch=$((TRAIN_BS * ROLLOUT_N))
    AGENT_WORKERS=$((_brl_generated_batch < 8 ? _brl_generated_batch : 8))
    while (( _brl_generated_batch % AGENT_WORKERS != 0 )); do
        AGENT_WORKERS=$((AGENT_WORKERS - 1))
    done
elif ! [[ "$AGENT_WORKERS" =~ ^[1-9][0-9]*$ ]] \
        || (( (TRAIN_BS * ROLLOUT_N) % AGENT_WORKERS != 0 )); then
    echo "AGENT_WORKERS must be a positive divisor of TRAIN_BS*ROLLOUT_N" >&2
    exit 2
fi

# CONFIG_NAME selects a hand-written config out of CONFIG_DIR. Without
# it the family's config is generated into that same directory, which is
# the only way to carry a multi-line jinja chat template into hydra. No
# hand-written config ships; the generated ones are the worked example,
# header and all, and copying one out is how you make yours. They are
# per-run output derived from the registry, not source, so they default
# under outputs/ and nothing about them is tracked.
BRL_CONFIG_DIR="${CONFIG_DIR:-$BRL_ROOT/outputs/configs}"
if [ -z "${CONFIG_NAME:-}" ]; then
    CONFIG_NAME="$BRL_MODEL_KEY"
    # BRL_MODEL_ID again: this also drops the run's provenance manifest,
    # whose whole content is the hub id and the revision it was pinned
    # to, and a snapshot directory has neither to give.
    brl_run python -m bedrock_rl.models --emit-config "$BRL_MODEL_ID" \
        ${LORA_ADAPTER:+--lora-adapter "$LORA_ADAPTER"} \
        --out "$BRL_CONFIG_DIR/$CONFIG_NAME.yaml" >/dev/null
fi

# ── LoRA ─────────────────────────────────────────────────────────────────
# Off by default: with no knob set this launcher still runs a full
# fine-tune. LORA_RANK turns it on, and MODEL pointing at an adapter
# directory continues that adapter instead of starting a new one. That is
# how a separate `trainer: rl` run continues the output of `trainer: sft`.
#
# Four things have to hold for a LoRA RL run to train the policy it
# samples from, and all four are decided here rather than left to a
# reader who would find out minutes into a run, or not at all.
#
# the vision tower gets no adapters. vLLM serves LoRA for the language
#   model only and drops tower adapters silently
#   (vllm/lora/models.py::_filter_unsupported_mm_module), so an adapter
#   trained there is trained against a policy the rollout never samples.
#   LORA_EXCLUDE carries the family's own vision_patterns as a regex.
# the rollout engine loads real base weights rather than verl's dummy
#   ones, which is what lets every sync afterwards carry the adapter
#   alone instead of the whole model on the first step.
# layered_summon gathers the FSDP shards a layer at a time, and verl
#   refuses it unless the line above is set.
# the learning rate is LoRA's. A fresh adapter's B matrix starts at zero
#   and the full fine-tune's 1e-6 moves it by nothing measurable, so the
#   default follows the mode rather than staying one number.
#
# verl folds the reference policy into the actor under LoRA and computes
# its log probs with the adapters disabled, so actor_rollout_ref.ref.*
# stops meaning anything here. That is verl's design, not a setting.
# Will it fit. Asked of the registry and the cards present, before
# a single weight is loaded, because the alternative is twenty minutes of
# model load ending in an OOM that names nothing. Same shape as the
# flash-attn check above: knowable in a tenth of a second, so know it.
#
# Unknown sizes pass. A registry entry with no parameter count should say
# it cannot answer rather than block a run on a number nobody wrote down.
# BRL_MODEL_ID, because the size of an entry is read out of its NAME and
# a snapshot directory is named after a commit. Asked about the path, the
# registry would report the size as unknown and this gate would pass
# everything.
BRL_FIT_OFFLOAD=""
case "${OFFLOAD:-False}" in
    1|true|True|yes|Yes|on|On) BRL_FIT_OFFLOAD=--offload ;;
esac
if ! brl_run python -m bedrock_rl.models --fit "$BRL_MODEL_ID" \
        --gpus "$((BRL_NGPU * NNODES))" \
        --gpu-mem-util "${GPU_MEM_UTIL:-0.45}" \
        $(brl_lora_on && echo --lora) \
        ${BRL_FIT_OFFLOAD:+$BRL_FIT_OFFLOAD}; then
    echo >&2
    if [ "${BRL_SKIP_FIT:-0}" = 1 ]; then
        echo "warning: BRL_SKIP_FIT=1 is overriding the failed model-fit" >&2
        echo "check; continuing at the risk of an out-of-memory failure." >&2
    else
        echo "refusing to start: this model does not fit the cards it was" >&2
        echo "given. Nothing was loaded. Override with BRL_SKIP_FIT=1 if" >&2
        echo "you believe the estimate rather than the arithmetic." >&2
        exit 1
    fi
fi

BRL_LORA_ARGS=()
if brl_lora_on; then
    LORA_TARGETS="${LORA_TARGETS:-all-linear}"
    BRL_LORA_ARGS+=(
        "actor_rollout_ref.model.lora_rank=$LORA_RANK"
        "actor_rollout_ref.model.lora_alpha=${LORA_ALPHA:-$LORA_RANK}"
        # LoRA keeps the frozen base in bf16 to avoid the memory cost of fp32
        # weights. Full fine-tuning retains verl's fp32 master-weight default.
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16"
        "actor_rollout_ref.ref.fsdp_config.model_dtype=bf16"
        # vLLM returns its own sampling logprobs, and verl's
        # calculate_debug_metrics compares them against the old_log_probs
        # the FSDP actor recomputes WITH the adapter active. If the
        # rollout serves base weights while the actor holds the adapter,
        # those are two different models and
        # training/rollout_actor_probs_pearson_corr falls well below 1.
        #
        # Correlation confirms that rollout and actor probabilities match once
        # the adapter has moved away from its zero-output initialization. It is
        # therefore informative only after the first update.
        "actor_rollout_ref.rollout.calculate_log_probs=True"
        "actor_rollout_ref.rollout.load_format=${LOAD_FORMAT:-safetensors}"
        "actor_rollout_ref.rollout.layered_summon=${LAYERED_SUMMON:-True}")
    if [ "$LORA_TARGETS" = all-linear ]; then
        BRL_LORA_ARGS+=("actor_rollout_ref.model.target_modules=all-linear")
    else
        BRL_LORA_ARGS+=("actor_rollout_ref.model.target_modules=[$LORA_TARGETS]")
    fi
    if [ -n "${LORA_EXCLUDE:-}" ]; then
        # quoted for hydra, whose override grammar reads an unquoted
        # value and would trip over the regex's own punctuation
        BRL_LORA_ARGS+=(
            "actor_rollout_ref.model.exclude_modules='$LORA_EXCLUDE'")
    fi
    if [ -n "${LORA_ADAPTER:-}" ]; then
        BRL_LORA_ARGS+=(
            "actor_rollout_ref.model.lora_adapter_path=$LORA_ADAPTER")
    fi
    LR="${LR:-1e-4}"
    if [ -n "${LORA_ADAPTER:-}" ]; then
        BRL_TUNING="lora continuing $LORA_ADAPTER, lr $LR"
    else
        BRL_TUNING="lora rank $LORA_RANK alpha ${LORA_ALPHA:-$LORA_RANK} on"
        BRL_TUNING="$BRL_TUNING $LORA_TARGETS excluding ${LORA_EXCLUDE:-nothing},"
        BRL_TUNING="$BRL_TUNING lr $LR"
    fi
else
    LR="${LR:-1e-6}"
    BRL_TUNING="full fine-tune, lr $LR"
fi

# ATTN_IMPL is the model's own attention kernel. The registry already
# resolved it, in the training venv, during brl_resolve_model: it reports
# flash_attention_2 when flash-attn imports there and sdpa otherwise, so
# this script no longer spawns a second interpreter to ask the same
# question. The sdpa branch keeps a FLASH_ATTN=0 setup running up to the
# log-prob step.
ATTN_IMPL="${ATTN_IMPL:-$BRL_ATTN_IMPL}"

# Validation is a separate, paired `bedrock eval` workflow. Older releases
# accepted VAL_DATA/VAL_FILES here while unconditionally disabling verl
# validation, which made both settings lie. Refuse the old spellings loudly.
if [ -n "${VAL_DATA:-}" ] || [ -n "${VAL_FILES:-}" ]; then
    echo "VAL_DATA and VAL_FILES are not training knobs; use 'bedrock eval' for held-out trials" >&2
    exit 2
fi

# verl demands data.val_files even with validation off, which it is here
# (test_freq=-1, val_before_train=False). The datasets and verl readers both
# refuse a zero-row parquet, so a dedicated synthetic image/prompt sentinel
# satisfies their schema without aliasing a real training row if validation is
# enabled by accident.
TRAIN_FILES="${TRAIN_FILES:-[\"$DATA_DIR/rl_prompts.parquet\"]}"
BRL_DISABLED_VAL_DATA="$OUT/validation_disabled.parquet"
brl_run python - "$TRAIN_FILES" "$BRL_DISABLED_VAL_DATA" <<'PY'
import os
import sys
import pyarrow.parquet as pq
from bedrock_rl.data import disabled_validation_table, first_data_file

files, path = sys.argv[1:]
source = first_data_file(files)
os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
pq.write_table(disabled_validation_table(source), path)
PY
BRL_DISABLED_VAL_FILES="[\"$BRL_DISABLED_VAL_DATA\"]"

# The streaming console adapter owns W&B. Running a second W&B client inside
# Ray creates duplicate runs and can lose buffered steps when the TaskRunner
# actor is torn down. TensorBoard remains a verl backend because it writes
# local event files rather than owning a network service. JSONL is always on.
BRL_LOGGERS='"console"'
case "${TENSORBOARD:-}" in
    1|true|yes|on)
        BRL_LOGGERS="$BRL_LOGGERS,\"tensorboard\""
        export TENSORBOARD_DIR="${TENSORBOARD_DIR:-$OUT/tensorboard}"
        ;;
esac
BRL_PROJECT="${WANDB_PROJECT:-bedrock-rl}"

# Ray otherwise advertises every CPU on the host. On large GPU boxes that can
# prestart hundreds of Python workers at once; each worker imports torch and
# the site hook before it registers, producing an import storm that can keep
# the actual GPU actors from starting. Four CPU resources per training GPU is
# ample for the driver, rollout workers, and local environment processes. A
# cluster operator can still name a different ceiling explicitly.
RAY_NUM_CPUS="${RAY_NUM_CPUS:-$((BRL_VISIBLE_GPUS_PER_NODE * 4))}"
BRL_RAY_INIT_ARGS=()
if [ -z "${RAY_ADDRESS:-}" ]; then
    BRL_RAY_INIT_ARGS+=(ray_kwargs.ray_init.num_cpus="$RAY_NUM_CPUS")
fi
# A driver that connects to an already-running Ray cluster cannot rely on the
# raylet inheriting variables exported by this launcher. Keep the worker-side
# agent loop on the exact same context, artifact, and rendering settings as the
# driver. Serialize a small allowlist as one Hydra mapping so paths containing
# commas or other override punctuation remain exact. Secrets and unrelated
# ambient variables are deliberately not copied.
BRL_RAY_ENV_JSON="$(brl_run python - <<'PY'
import json
import os

defaults = {
    "BRL_KEEP_LATEST_IMAGES": "0",
    "BRL_TRAJECTORY_DIR": "",
    "BRL_SLUG": "",
    "BRL_FRAMES_ROOT": "",
    "NETHERITE_RASTER": "cpu",
    "NETHERITE_ENV_GPUS": "",
    "NETHERITE_FRAME_SCALE": "1",
}
items = (f"{key}:{json.dumps(os.environ.get(key, default))}"
         for key, default in defaults.items())
print("{" + ",".join(items) + "}")
PY
)"
BRL_RAY_ENV_ARGS=(
    "+ray_kwargs.ray_init.runtime_env.env_vars=$BRL_RAY_ENV_JSON"
)

cd "$VERL_HOME"
# shellcheck disable=SC2086
brl_run python -m verl.trainer.main_ppo \
    --config-path="$BRL_CONFIG_DIR" \
    --config-name="$CONFIG_NAME" \
    "${BRL_RAY_INIT_ARGS[@]+"${BRL_RAY_INIT_ARGS[@]}"}" \
    "${BRL_RAY_ENV_ARGS[@]}" \
    algorithm.adv_estimator="$ADV" \
    actor_rollout_ref.actor.policy_loss.loss_mode="$LOSS_MODE" \
    actor_rollout_ref.actor.loss_agg_mode="$LOSS_AGG_MODE" \
    algorithm.use_kl_in_reward="${USE_KL_IN_REWARD:-False}" \
    algorithm.kl_ctrl.kl_coef="${KL_COEF:-0.001}" \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$BRL_DISABLED_VAL_FILES" \
    data.train_batch_size="$TRAIN_BS" \
    data.dataloader_num_workers="${DATALOADER_WORKERS:-0}" \
    data.max_prompt_length="$BRL_MAX_PROMPT_LEN" \
    data.max_response_length="$RESPONSE_LEN" \
    data.image_key=images \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.trust_remote_code="$BRL_TRUST_REMOTE_CODE" \
    +actor_rollout_ref.model.override_config.attn_implementation="$ATTN_IMPL" \
    actor_rollout_ref.actor.optim.lr="$LR" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BS" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BS" \
    actor_rollout_ref.actor.use_dynamic_bsz="$([ "$DYN_BSZ" = 1 ] && echo True || echo False)" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$DYN_MAX_TOKEN" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="$([ "$DYN_BSZ" = 1 ] && echo True || echo False)" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$DYN_MAX_TOKEN" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="$([ "$DYN_BSZ" = 1 ] && echo True || echo False)" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$DYN_MAX_TOKEN" \
    actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS:-False}" \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.001}" \
    actor_rollout_ref.actor.kl_loss_type="${KL_LOSS_TYPE:-low_var_kl}" \
    actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF:-0}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD:-False} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD:-False} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.temperature="${TEMP:-1.0}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP:-1}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL:-0.45}" \
    actor_rollout_ref.rollout.max_model_len="$BRL_MODEL_LEN" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$MICRO_BS" \
    +actor_rollout_ref.rollout.limit_images="$BRL_LIMIT_IMAGES" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format="$BRL_TOOL_PARSER" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$BRL_VERL_TURNS" \
    actor_rollout_ref.rollout.multi_turn.max_user_turns="$BRL_VERL_TURNS" \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls="$BRL_MAX_TOOL_CALLS_PER_TURN" \
    actor_rollout_ref.rollout.agent.num_workers="$AGENT_WORKERS" \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=null \
    actor_rollout_ref.rollout.agent.default_agent_loop="$AGENT_LOOP" \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_LOOP_CONFIG" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$MICRO_BS" \
    actor_rollout_ref.ref.fsdp_config.param_offload=${OFFLOAD:-False} \
    ++trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
    trainer.critic_warmup=0 \
    trainer.logger="[$BRL_LOGGERS]" \
    trainer.resume_mode="${BRL_RESUME_MODE:-auto}" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node="$BRL_NGPU" \
    trainer.nnodes="$NNODES" \
    trainer.save_freq="${SAVE_FREQ:-$STEPS}" \
    trainer.test_freq=-1 \
    trainer.total_training_steps="$STEPS" \
    trainer.total_epochs=1000 \
    trainer.project_name="$BRL_PROJECT" \
    trainer.experiment_name="$BRL_SLUG" \
    trainer.default_local_dir="$OUT" \
    "${BRL_DISTILLATION_ARGS[@]+"${BRL_DISTILLATION_ARGS[@]}"}" \
    "${BRL_LORA_ARGS[@]+"${BRL_LORA_ARGS[@]}"}" \
    "${BRL_VERL_OVERRIDE_ARGS[@]}" \
    2>&1 \
  | brl_run python -m bedrock_rl.adapters.verl.console \
        --run "$BRL_SLUG" --steps "$STEPS" \
        --raw-log "$BRL_LOG" --metrics "$BRL_METRICS" \
        --config task="$BRL_TASK_NAME" \
        --config "task file=$(brl_task_yaml "$TASK")" \
        --config "model=$BRL_MODEL_KEY ($BRL_STATUS) -> $MODEL [$ATTN_IMPL]" \
        --config "data=$TRAIN_FILES" \
        --config "trainer=$BRL_METHOD, advantage=$ADV" \
        --config "batch=$TRAIN_BS x $ROLLOUT_N rollouts, $MAX_TURNS turns, $AGENT_WORKERS agent workers" \
        --config "tuning=$BRL_TUNING" \
        --config "gpus=$CUDA_VISIBLE_DEVICES; student=$BRL_NGPU$([ "$BRL_METHOD" = rl ] || echo ", teacher=$TEACHER_GPUS_PER_NODE")" \
        --config "trajectories=$BRL_TRAJECTORY_DIR" \
        --config "out=$OUT"

# A LoRA run's checkpoints are FSDP shards of a peft model, which nothing
# outside verl reads, so the run ends by writing the adapter beside them.
# Without this the artifact of a LoRA run is a directory that no eval
# path, and no other trainer, can load.
if brl_lora_on && [ "${EXPORT_ADAPTER:-1}" != 0 ]; then
    brl_run python -m bedrock_rl.train.export_adapter \
        --checkpoints "$OUT" --base "$MODEL" 2>&1 | tee -a "$BRL_LOG"
fi
