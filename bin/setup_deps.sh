#!/usr/bin/env bash
# Dependency setup, uv end to end: clone and patch the pinned Netherite,
# build the sim, and, unless BRL_RUNTIME_ONLY=1, clone and patch verl and
# create the Linux/CUDA training venv.
#
# Deliberately NOT git submodules: both trees are large, netherite carries
# local patches (patches/netherite/), and a user may already have working
# checkouts. Pinned SHAs plus patch files keep provenance exact without
# submodule clone and update friction. Override locations with
# NETHERITE_HOME, VERL_HOME and VERL_ENV, and the interpreter with
# VERL_PYTHON. The CLI's `setup --runtime-only` sets BRL_RUNTIME_ONLY.
#
# Both checkouts may be yours and may hold work this script did not put
# there, so it moves one to its pin only when that is unambiguously safe and
# otherwise stops without touching anything.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# One source of truth for the engine pin. This is the only reader that
# opens the file: every python caller goes through
# bedrock_rl/env/engine.py::pinned_sha().
NETHERITE_SHA="$(cat "$ROOT/patches/netherite/PINNED_SHA")"
# verl is pinned to a SHA, not a tag, and carries every diff in
# patches/verl/. Two are LoRA fixes that landed after this revision and
# are needed for a LoRA RL run to be correct rather than merely to start;
# the third fixes Qwen-VL's multi-axis position IDs in the no-padding path:
#
#   0001  #6688  clones LoRA weights out of the reused IPC buffer before
#                add_lora, so the adapter vLLM serves is not overwritten
#                by the next batch
#   0002  #7014  syncs merged LoRA weights before the FSDP context exits
#   0003         packs the sequence, rather than the four MRoPE axes, as
#                the ragged dimension during actor log-prob/update passes
#
# Frame/context selection lives in this repo's custom agent loop, so it no
# longer patches verl or assumes a model family's image-token constants.
#
# The upstream two are cherry-picked rather than taken by moving the
# checkout, because there is no revision that has both and still works
# here. verl main requires vLLM 0.18.0+ and this stack is on 0.12.0, and
# every commit between the two PRs and that floor imports
# `transfer_queue`, which is not on PyPI. The window is empty. Both diffs
# are small, isolated changes and apply clean.
#
# Written in full, and not as 7aed6b23, because the Dockerfile fetches
# this object from the remote by name and a git server cannot resolve an
# abbreviation. The two files carry the same string so they can be
# compared by eye.
VERL_SHA=7aed6b230776f963fa09509c10d9c3a767d1102c
FLASH_ATTN_VERSION=2.8.3
DEEPSPEED_VERSION=0.19.4
NETHERITE_HOME="${NETHERITE_HOME:-$HOME/netherite}"
VERL_HOME="${VERL_HOME:-$HOME/verl}"
VERL_ENV="${VERL_ENV:-$HOME/verl-env}"
# 3.12 is the documented combo. 3.10 also resolves but needs a source-built
# flash-attn if you want flash attention, which 3.12 gets from a wheel.
VERL_PYTHON="${VERL_PYTHON:-3.12}"

if ! command -v uv >/dev/null 2>&1; then
    cat >&2 <<'EOF'
uv not found. Install it and re-run:

  curl -LsSf https://astral.sh/uv/install.sh | sh

EOF
    exit 1
fi

# Moving someone else's checkout can destroy work that is not recoverable
# from a remote, and this repo invites you to point NETHERITE_HOME at a tree
# you have modified. So a checkout is only moved to its pin when the move
# orphans nothing, and every other case stops the run and reports what it
# found. FORCE_PINNED_CHECKOUT=1 restores the old unconditional behavior.
sync_pinned_checkout() {
    local dir="$1" ref="$2" label="$3"
    local head target problems=""

    head="$(git -C "$dir" rev-parse HEAD)"
    if ! target="$(git -C "$dir" rev-parse -q --verify "$ref^{commit}")"; then
        git -C "$dir" fetch -q --tags origin
        target="$(git -C "$dir" rev-parse -q --verify "$ref^{commit}")"
    fi
    [ "$head" = "$target" ] && return 0

    if [ "${FORCE_PINNED_CHECKOUT:-0}" = 1 ]; then
        git -C "$dir" checkout -q --detach "$target"
        return 0
    fi

    if [ -n "$(git -C "$dir" status --porcelain --untracked-files=no)" ]; then
        problems="$problems
  uncommitted changes to tracked files"
    fi
    # commits reachable from HEAD but from no remote and not from the pin are
    # local work that a checkout would leave unreachable
    if [ -n "$(git -C "$dir" rev-list HEAD --not --remotes "$target")" ]; then
        problems="$problems
  local commits that no remote carries, which the move would strand"
    fi
    if ! git -C "$dir" merge-base --is-ancestor "$target" "$head" &&
       ! git -C "$dir" merge-base --is-ancestor "$head" "$target"; then
        problems="$problems
  a HEAD that is neither an ancestor nor a descendant of the pin"
    fi

    if [ -n "$problems" ]; then
        cat >&2 <<EOF

$label checkout $dir cannot be moved to $ref safely.
Found$problems

Nothing was changed. Choose one of these.
  point the env var at a new directory and let this script clone it fresh
  keep your work with git branch or git stash, then run this again
  overwrite the checkout and lose the above with FORCE_PINNED_CHECKOUT=1

EOF
        exit 1
    fi
    git -C "$dir" checkout -q --detach "$target"
}

# ---- netherite (C sim) ----------------------------------------------------
if [ ! -d "$NETHERITE_HOME/.git" ]; then
    git clone https://github.com/Infatoshi/netherite.git "$NETHERITE_HOME"
fi
sync_pinned_checkout "$NETHERITE_HOME" "$NETHERITE_SHA" netherite
# Patches apply in order. Deciding which are already in the tree needs one
# fact about them: the reverse-check is only meaningful for the TOPMOST
# applied patch. A later patch edits lines that are CONTEXT in an earlier
# one, so once the top patch is applied, reverse-checking 0001 may fail even
# 0001 is plainly there. Testing each patch independently therefore made
# a fully set-up checkout look like a conflicted one, and re-running this
# script on it stopped with an error telling the user their tree carried
# conflicting changes. It did not.
#
# A checkout can only ever hold a PREFIX of this stack, because that is
# the only thing that applies. So find the highest patch that reverse-
# checks clean, and apply everything above it.
NETHERITE_PATCHES=()
for NETHERITE_PATCH in "$ROOT"/patches/netherite/*.diff; do
    [ -f "$NETHERITE_PATCH" ] || continue
    NETHERITE_PATCHES+=("$(basename "$NETHERITE_PATCH" .diff)")
done
if [ "${#NETHERITE_PATCHES[@]}" -eq 0 ]; then
    echo "no engine patches found under $ROOT/patches/netherite" >&2
    exit 1
fi
first_missing=0
for ((i=${#NETHERITE_PATCHES[@]} - 1; i >= 0; i--)); do
    if git -C "$NETHERITE_HOME" apply --reverse --check \
            "$ROOT/patches/netherite/${NETHERITE_PATCHES[i]}.diff" \
            2>/dev/null; then
        first_missing=$((i + 1))
        break
    fi
done
for ((i=first_missing; i < ${#NETHERITE_PATCHES[@]}; i++)); do
    p="${NETHERITE_PATCHES[i]}"
    NETHERITE_PATCH="$ROOT/patches/netherite/$p.diff"
    if git -C "$NETHERITE_HOME" apply --check "$NETHERITE_PATCH" 2>/dev/null;
    then
        git -C "$NETHERITE_HOME" apply "$NETHERITE_PATCH"
    else
        # Neither already-applied nor cleanly applicable. The usual cause is a
        # checkout carrying an OLDER revision of this same patch, which the
        # forward apply cannot land because its context has moved. Refusing is
        # the same policy as sync_pinned_checkout above: this script never
        # discards work in a checkout it does not own.
        cat >&2 <<EOF

netherite checkout $NETHERITE_HOME already carries changes that conflict
with $NETHERITE_PATCH, most likely an earlier revision of it.

Nothing was changed. Choose one of these.
  restore just the files this patch owns, then run this again
    git -C $NETHERITE_HOME checkout -- \\
      \$(git -C $NETHERITE_HOME apply --numstat "$NETHERITE_PATCH" | cut -f3)
  point NETHERITE_HOME at a new directory and let this script clone it fresh

EOF
        exit 1
    fi
done
# Record what this engine was built with, IN the engine checkout, so
# that adding a diff to patches/netherite/ stops invalidating every
# already-built engine at once. bedrock_rl/env/engine.py::patch_status
# has the reasoning behind this file.
#
# Written on EVERY run, not only when something was applied, so
# re-running this on an engine that is already correct adopts it with no
# rebuild. The loop above leaves the full list applied, so the full list
# is what this records.
#
# It is CALLED below the build rather than here, and that ordering is
# what the file is worth. A manifest is an attestation about a BINARY,
# and the patches above are source. Written before the compiler runs, it
# says "this engine carries these diffs" at a moment when that is a claim
# about a source tree, and every provenance reader would then certify a binary
# nobody compiled from those diffs. An engine with NO manifest is merely
# unknown, which provenance checks handle by inferring and saying so; an
# engine with a WRONG one is worse than the staleness this file exists to
# catch, because now the staleness is attested as correct.
#
# The filename is bedrock_rl/env/engine.py::PATCH_MANIFEST_NAME,
# spelled out here for the same reason PINNED_SHA is catted directly:
# this is shell and that is the only python-free way to say it. There is
# a test.
#
# Via a temp file and a rename, and every step checked. A partial
# manifest is worse than none: it makes the engine claim a SHORTER stack
# than it has, and a missing digest silently drops a patch from the
# claim. A group redirection that fails does not trip errexit on bash
# 3.2, so the failure has to be handled here rather than assumed fatal.
brl_sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}
brl_write_manifest() {
    local tmp="$NETHERITE_HOME/.bedrock-rl-patches.tmp" p f sum
    : > "$tmp" || return 1
    {
        echo "# bedrock-rl engine patch manifest: the diffs this engine" \
             "was built with,"
        echo "# sha256 then filename. Written by bin/setup_deps.sh and" \
             "read by"
        echo "# bedrock_rl/env/engine.py::read_patch_manifest. Delete it" \
             "and the checks"
        echo "# fall back to inferring the stack from the tree."
    } >> "$tmp" || return 1
    for p in "${NETHERITE_PATCHES[@]}"; do
        f="$ROOT/patches/netherite/$p.diff"
        [ -f "$f" ] || return 1
        sum="$(brl_sha256 "$f")" || return 1
        [ -n "$sum" ] || return 1
        echo "$sum  $p.diff" >> "$tmp" || return 1
    done
    mv "$tmp" "$NETHERITE_HOME/.bedrock-rl-patches"
}
# Texture atlases are generated C headers. MC_JAR builds them from a vanilla
# 1.11.2 client jar you own. TEXTURE_PACK optionally overlays a resource-pack
# zip/directory before the engine's own bootstrap reads it. Without a jar,
# tools/stub_atlases.py
# draws a synthetic resource pack and runs that same bootstrap against it, so
# the headers keep the real struct shapes and symbol names and only the
# texels are procedural.
#
# Stubs are the default because a clone that cannot build at all is worse
# than one that looks different, and because nothing in the observation
# record samples a texel, so only the rasterised frame changes. They are
# generated deterministically, so two clones agree on every rendered input.
#
# water_frames.h is the sentinel because both paths write it last, and
# because the one header that does ship in the checkout (blockmodels.h)
# would otherwise read as a completed bootstrap and hand a broken build to
# make.
if [ -n "${TEXTURE_PACK:-}" ] && [ -z "${MC_JAR:-}" ]; then
    echo "TEXTURE_PACK needs MC_JAR pointing at a Minecraft 1.11.2 client" \
         "jar you own" >&2
    exit 2
fi
if [ -n "${MC_JAR:-}" ] || \
   [ ! -f "$NETHERITE_HOME/magma/assets/water_frames.h" ]; then
    if [ -n "${MC_JAR:-}" ]; then
        if [ ! -f "$MC_JAR" ]; then
            echo "MC_JAR=$MC_JAR is not a file" >&2
            exit 2
        fi
        brl_bootstrap_jar="$MC_JAR"
        brl_composite_jar=""
        if [ -n "${TEXTURE_PACK:-}" ]; then
            if [ ! -f "$TEXTURE_PACK" ] && [ ! -d "$TEXTURE_PACK" ]; then
                echo "TEXTURE_PACK=$TEXTURE_PACK is not a zip file or directory" >&2
                exit 2
            fi
            brl_composite_jar="$(mktemp "${TMPDIR:-/tmp}/netherite-pack.XXXXXX.jar")"
            uv run --no-project python "$ROOT/tools/compose_texture_pack.py" \
                --base "$MC_JAR" --pack "$TEXTURE_PACK" \
                --out "$brl_composite_jar"
            brl_bootstrap_jar="$brl_composite_jar"
        fi
        if ! (cd "$NETHERITE_HOME" && MC_JAR="$brl_bootstrap_jar" \
                bash scripts/bootstrap_assets.sh); then
            [ -z "$brl_composite_jar" ] || rm -f "$brl_composite_jar"
            exit 1
        fi
        [ -z "$brl_composite_jar" ] || rm -f "$brl_composite_jar"
        if command -v sha1sum >/dev/null 2>&1; then
            brl_asset_sha1="$(sha1sum "$MC_JAR" | cut -d' ' -f1)"
        else
            brl_asset_sha1="$(shasum -a 1 "$MC_JAR" | cut -d' ' -f1)"
        fi
        {
            echo "kind=jar"
            if [ -n "${TEXTURE_PACK:-}" ]; then
                echo "source=minecraft-client.jar+resource-pack"
            else
                echo "source=minecraft-client.jar"
            fi
            echo "sha1=$brl_asset_sha1"
            if [ -n "${TEXTURE_PACK:-}" ]; then
                echo "texture_pack=$(basename "$TEXTURE_PACK")"
                echo "texture_pack_sha256=$(uv run --no-project python \
                    "$ROOT/tools/compose_texture_pack.py" --digest \
                    --pack "$TEXTURE_PACK")"
            fi
        } > "$NETHERITE_HOME/.bedrock-rl-assets"
    else
        uv run --no-project --with pillow python \
            "$ROOT/tools/stub_atlases.py" --netherite-home "$NETHERITE_HOME"
        cat >&2 <<'EOF'

This engine carries STUB textures. Blocks, items and container art are
generated, not Mojang's. For comparison runs using real textures, supply a
1.11.2 client jar you own and build again in a fresh checkout:

  bedrock setup --mc-jar /path/to/minecraft-1.11.2.jar
  bedrock setup --mc-jar /path/to/minecraft-1.11.2.jar \
    --texture-pack /path/to/pack.zip

EOF
    fi
fi
# UNCONDITIONALLY, and not `if [ ! -f magma_game ]`. The patches above
# edit C source, and a binary that already exists is exactly the case
# where a new or edited diff has just landed in a tree that was built
# from the old one. Skipping the build there leaves a stale binary and
# the manifest below then attests the new stack for it. make is the tool
# whose entire job is deciding what actually needs recompiling, so asking
# it every time costs a few milliseconds on an up-to-date checkout and is
# the only thing that keeps "patched" and "built" the same fact.
if ! make -C "$NETHERITE_HOME/magma" game; then
    cat >&2 <<EOF

the engine at $NETHERITE_HOME did not build, so anything already in
$NETHERITE_HOME/magma/magma_game is from an earlier source tree and
nothing here can say which one. No patch manifest was written, so the
provenance checks will report this engine as unverified rather than
certify it.

EOF
    exit 1
fi
# Only now, because the binary is what a manifest is about; see the block
# that defines brl_write_manifest above.
if ! brl_write_manifest; then
    rm -f "$NETHERITE_HOME/.bedrock-rl-patches.tmp"
    cat >&2 <<EOF

warning: could not write $NETHERITE_HOME/.bedrock-rl-patches, so this
engine records nothing about the patches it was built with. Everything
still runs; the provenance checks fall back to inferring the stack from
the tree, which cannot tell a patch staged for the next build from one
that was reverted or edited. Re-run setup after changing patches.

EOF
fi

# Task authoring, synthetic generation, remote-API trials, and Modal launches
# need the patched Minecraft runtime but not a local Linux/CUDA trainer. Keep
# that common clone path small and portable: the complete setup below remains
# the default, while `bedrock setup --runtime-only` stops only after the
# engine, texture assets, and patch manifest are all ready.
if [ "${BRL_RUNTIME_ONLY:-0}" = 1 ]; then
    echo "runtime setup complete: NETHERITE_HOME=$NETHERITE_HOME"
    exit 0
fi

# ---- verl (trainer) ---------------------------------------------------------
if [ ! -d "$VERL_HOME/.git" ]; then
    git clone https://github.com/volcengine/verl.git "$VERL_HOME"
fi
sync_pinned_checkout "$VERL_HOME" "$VERL_SHA" verl

# Every diff in patches/verl/. Unlike the engine stack these are checked
# INDEPENDENTLY rather than by the prefix rule, which is valid because
# the three touch different files (vllm_rollout/utils.py,
# engine/fsdp/transformer_impl.py, and workers/utils/padding.py) and so
# never appear in each other's
# context. That is a property of this set,
# not a general one: a future diff editing a file another already touched
# has to move to the prefix rule the engine patches use above.
#
# A patch that neither reverse-checks nor applies is a HARD failure, not
# a warning. Printing to stderr and carrying on leaves a half-patched
# trainer behind one line in a setup log, and a full LoRA run on a tree
# with neither LoRA fix trains an adapter and serves a different one
# while every log looks normal. There is no useful thing this script can
# do with a half-patched trainer, so it stops and says which diff and
# what it costs.
if [ -d "$ROOT/patches/verl" ]; then
    verl_applied=0
    verl_total=0
    for VERL_PATCH in "$ROOT"/patches/verl/*.diff; do
        [ -e "$VERL_PATCH" ] || continue
        verl_total=$((verl_total + 1))
        if git -C "$VERL_HOME" apply --reverse --check "$VERL_PATCH" \
                2>/dev/null; then
            verl_applied=$((verl_applied + 1))
            continue                      # already applied
        fi
        if git -C "$VERL_HOME" apply --check "$VERL_PATCH" 2>/dev/null; then
            git -C "$VERL_HOME" apply "$VERL_PATCH"
            # the forward apply reporting success is not the same fact as
            # the change being in the tree
            if ! git -C "$VERL_HOME" apply --reverse --check "$VERL_PATCH" \
                    2>/dev/null; then
                echo "error: $(basename "$VERL_PATCH") applied to" \
                     "$VERL_HOME and then did not reverse-check, so it is" \
                     "not in that tree" >&2
                exit 1
            fi
            verl_applied=$((verl_applied + 1))
            echo "applied $(basename "$VERL_PATCH") to $VERL_HOME"
        else
            cat >&2 <<EOF

verl checkout $VERL_HOME cannot take
$VERL_PATCH.

It is neither already applied nor cleanly applicable, so that tree is
NOT the trainer this repo pins. 0001 and 0002 are on the LoRA
weight-sync path; 0003 is on Qwen-VL's actor input path. Without them a
run can silently serve stale adapter weights or fail after rollout when
it recomputes log probabilities.

Nothing further was changed. Choose one of these.
  restore just the files this patch owns, then run this again
    git -C $VERL_HOME checkout -- \\
      \$(git -C $VERL_HOME apply --numstat "$VERL_PATCH" | cut -f3)
  point VERL_HOME at a new directory and let this script clone it fresh

EOF
            exit 1
        fi
    done
    echo "verl at $VERL_SHA with $verl_applied/$verl_total patches applied"
fi

if [ ! -x "$VERL_ENV/bin/python" ]; then
    uv venv --python "$VERL_PYTHON" "$VERL_ENV"
fi
# known-good combo for verl 0.8 on H100/cu128 (see README). verl installs
# --no-deps, so its runtime deps are pinned explicitly here: torchdata
# carries StatefulDataLoader, qwen-vl-utils carries the VL path,
# transformers must stay pinned or the resolver moves it, and datasets
# needs the >=4 floor or fsspec constraints drag it down to a 2.x that
# crashes on modern pyarrow. rich is the console
# (bedrock_rl/reporting.py); the launchers run out of THIS venv on
# PYTHONPATH rather than installing this project into it, so the CPU
# project's dependency list does not reach here. Without it the console
# falls back to plain text, which is correct but is not what a terminal
# should get. verl's W&B and TensorBoard adapters import their client packages
# only when selected, but installing them here keeps a typed train.yaml from
# failing after GPUs have already been allocated.
# nvidia-cutlass-dsl is NOT a dependency this repo uses. It is pinned
# because vLLM 0.12 probes for a FlashAttention 4 kernel like this
# (vllm/vllm_flash_attn/flash_attn_interface.py):
#
#     try:
#         from flash_attn.cute.interface import _flash_attn_fwd
#         FA4_AVAILABLE = True
#     except ImportError:
#         FA4_AVAILABLE = False
#
# The guard catches ImportError and nothing else. flash-attn 2.8.3's
# `cute` layer is written against the cutlass DSL, and up to 4.5 a
# mismatch surfaces as ModuleNotFoundError, which IS an ImportError, so
# the probe fails quietly and the optional kernel is simply off. From
# 4.7 the DSL no longer carries `cutlass.cute.core.ThrMma`, so the same
# mismatch surfaces as an AttributeError, which the guard does not
# catch, and it escapes all the way out.
#
# What it takes down is not an optional kernel. verl imports its MoE
# weight-loader patch at module scope in
# workers/rollout/vllm_rollout/utils.py, that pulls in vllm's
# deepseek_v2 -> rotary_embedding -> vllm_flash_attn chain, so
# `import verl.workers.rollout.vllm_rollout` raises. EVERY vLLM rollout
# in the venv dies about thirty seconds in, dense models included,
# before a byte of GPU memory is allocated. It is not MoE-specific and
# it is not LoRA-specific; both were suspected first and both were
# wrong.
#
# Measured on this stack: 4.5.3 imports clean, 4.7.0 does not.
uv pip install --python "$VERL_ENV/bin/python" \
    --torch-backend cu128 \
    --requirement "$ROOT/requirements/verl-cu128.lock"
uv pip install --python "$VERL_ENV/bin/python" --no-deps -e "$VERL_HOME"

# verl 0.8 reaches for flash_attn.bert_padding when it unpads a batch to
# compute log probs, and on CUDA that import has no fallback, so training
# needs the package even when the model itself runs sdpa attention. No
# build picks up a matching prebuilt wheel when one exists for your python,
# torch, and CUDA, and compiles from source when one does not, which is slow;
# MAX_JOBS caps the parallel nvcc jobs. Set FLASH_ATTN=0 to skip it, which
# leaves data generation, the CPU path, and SFT working.
if [ "${FLASH_ATTN:-1}" != 0 ]; then
    if ! uv pip install --python "$VERL_ENV/bin/python" --no-build-isolation \
            "flash-attn==$FLASH_ATTN_VERSION"; then
        cat >&2 <<EOF

warning: flash-attn did not build, so bin/rl.sh will fail once it
computes log probs. Everything else in this setup is usable. Retry with:

  MAX_JOBS=8 uv pip install --python "$VERL_ENV/bin/python" \\
      --no-build-isolation "flash-attn==$FLASH_ATTN_VERSION"

EOF
    fi
fi

# DeepSpeed is optional and off by default, because only the SFT
# SFT trainer in bedrock_rl/train/sft.py can use it. Set
# INSTALL_DEEPSPEED to anything but 0 to install it. It does nothing for
# bin/rl.sh, since RL runs on verl, which shards with its own FSDP and
# ships no DeepSpeed backend.
# The name is INSTALL_DEEPSPEED and not DEEPSPEED because bin/sft.sh
# already reads DEEPSPEED, where it is a ZeRO preset NAME. One variable that
# means "install it" here and "shard this way" there is a variable that does
# the wrong thing when someone exports it once for a whole session.
if [ "${INSTALL_DEEPSPEED:-0}" != 0 ]; then
    uv pip install --python "$VERL_ENV/bin/python" \
        "deepspeed==$DEEPSPEED_VERSION"
fi

# importing the trainer entrypoint is the real check; a bare `import verl`
# passes even when a runtime dep is missing
uv run --no-project --python "$VERL_ENV/bin/python" \
    python -c "import verl, verl.trainer.main_ppo; print('verl', verl.__version__, 'ok')"

echo "setup complete: NETHERITE_HOME=$NETHERITE_HOME VERL_HOME=$VERL_HOME VERL_ENV=$VERL_ENV"
