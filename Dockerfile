# syntax=docker/dockerfile:1.7
#
# The environment, defined once. Two images come out of this file.
#
#   cpu   the engine at the pinned SHA with its patches, this package and
#         its CPU dependencies. What task/data generation and remote-API
#         trials need.
#   gpu   the cpu image plus the verl 0.8 training venv, torch and vLLM.
#         What a Slurm job or a rented GPU runs.
#
#   SHA=$(cat patches/netherite/PINNED_SHA)
#   docker build --target cpu --build-arg NETHERITE_SHA=$SHA \
#       -t bedrock-rl:cpu .
#   docker build --target gpu --build-arg NETHERITE_SHA=$SHA \
#       -t bedrock-rl:gpu .
#
# NETHERITE_SHA is not optional and has no default on purpose, so these
# lines carry it. A build without it stops at the engine clone rather
# than quietly building whatever HEAD the clone landed on.
#
# It reproduces bin/setup_deps.sh, and the pins are build args with the
# same values so a reader can compare the two files line by line. The
# engine pin is NOT defaulted here: it is passed in from
# patches/netherite/PINNED_SHA, which stays the one file that holds it.
#
# BOTH trees are pinned to a SHA and BOTH carry the diffs this repo
# ships: patches/netherite/ for the engine, patches/verl/ for the
# trainer. The gpu stage used to clone verl at tag v0.8.0 and apply
# nothing, so an image and a box set up by bin/setup_deps.sh were two
# different trainers, and the difference sat on the LoRA weight-sync
# path. `bedrock doctor` reports the same fact for whatever tree a box
# actually has, because the divergence was never only docker-vs-native.
#
ARG NETHERITE_REPO=https://github.com/Infatoshi/netherite.git
ARG VERL_REPO=https://github.com/volcengine/verl.git
# A SHA, not a tag, and the SAME one bin/setup_deps.sh pins. This used to
# be `ARG VERL_TAG=v0.8.0`, which is a different tree from the pin AND
# carried none of patches/verl/, so the container path could serve a LoRA
# adapter the trainer never updated while the docs claimed parity with a
# box set up by bin/setup_deps.sh: a working image that trains a LoRA run
# wrong. The trainer diffs in patches/verl/ are cut against this
# revision and do not apply to the tag; see the gpu stage. Written in
# full because the fetch below asks the remote for this object by name
# and a git server cannot resolve an abbreviation.
ARG VERL_SHA=7aed6b230776f963fa09509c10d9c3a767d1102c
ARG FLASH_ATTN_VERSION=2.8.3
ARG PYTHON_VERSION=3.12
ARG FINAL_STAGE=cpu

# ── toolchain ────────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS toolchain
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git libsdl2-dev pkg-config \
        xxd unzip \
    && rm -rf /var/lib/apt/lists/*
# uv system-wide: the image builds as root and runs as somebody else
RUN curl -LsSf https://astral.sh/uv/install.sh \
    | UV_INSTALL_DIR=/usr/local/bin sh

# ── the engine, at the pin, with every shipped patch ─────────────────────
FROM toolchain AS engine-src
ARG NETHERITE_REPO
ARG NETHERITE_SHA
COPY patches/netherite /opt/patches/netherite
# A single commit, and only the directories this build reads. Upstream
# carries 500 MB of verification corpora, optimiser run logs, docs and a
# java reference implementation that this runtime does not need.
# blob:none defers the file contents and the sparse cone then asks for
# the four directories the build and the provenance check touch, so the
# .git that arrives is metadata plus those blobs and nothing else.
#
# The image still carries a .git, deliberately. The engine check reads
# `git rev-parse HEAD` and reverse-applies the topmost patch, and that is
# the whole provenance guarantee; both work on a detached single-commit
# sparse checkout.
RUN test -n "$NETHERITE_SHA" \
    || (echo "NETHERITE_SHA is required; pass the contents of \
patches/netherite/PINNED_SHA" >&2; exit 1) \
 && git init -q /opt/netherite \
 && git -C /opt/netherite remote add origin "$NETHERITE_REPO" \
 && git -C /opt/netherite config core.sparseCheckout true \
 && git -C /opt/netherite sparse-checkout set --cone \
        magma blaze scripts renderkernels \
 && git -C /opt/netherite fetch -q --depth 1 --filter=blob:none \
        origin "$NETHERITE_SHA" \
 && git -C /opt/netherite checkout -q --detach FETCH_HEAD \
 && test "$(git -C /opt/netherite rev-parse HEAD)" = "$NETHERITE_SHA"
# set -e and a counted glob: an unmatched glob would hand `git apply` a
# literal path, and a failed apply on any but the last diff would
# otherwise build an engine that is patched halfway
RUN set -ex; set -- /opt/patches/netherite/*.diff; test "$#" -ge 1; \
    manifest=/opt/netherite/.bedrock-rl-patches.tmp; \
    printf '%s\n' '# bedrock-rl engine patch manifest' > "$manifest"; \
    for d; do \
        git -C /opt/netherite apply "$d"; \
        sum="$(sha256sum "$d" | cut -d' ' -f1)"; \
        printf '%s  %s\n' "$sum" "$(basename "$d")" >> "$manifest"; \
    done; \
    mv "$manifest" /opt/netherite/.bedrock-rl-patches

# ── the texture atlases ──────────────────────────────────────────────────
# The one part of the build that ever needed anything Mojang owns, and it
# no longer does by default.
#
#   ASSETS=stub   (default) tools/stub_atlases.py draws a synthetic
#                 resource pack and runs the engine's own bootstrap
#                 against it, so the headers keep the real struct shapes
#                 and symbol names and only the texels are procedural.
#                 Deterministic, so two builds agree on every pixel,
#                 so independent clones present the same pixels. An image built this
#                 way carries nothing anyone else owns and can be
#                 published.
#   ASSETS=jar    generate them from a vanilla 1.11.2 client jar, fetched
#                 from Mojang's CDN and sha1-verified. Use this when
#                 comparison runs require Mojang's textures.
#                 You must own Minecraft to use its assets, so an image
#                 built this way is yours and is not redistributable.
#   ASSETS=none   skip, for a caller who bind-mounts prebuilt headers.
#                 The build then fails at `make`, by design.
FROM engine-src AS assets
ARG ASSETS=stub
ARG MC_JAR_SHA1=db5aa600f0b0bf508aaf579509b345c4e34087be
ARG TEXTURE_PACK_URL
ARG TEXTURE_PACK_SHA256
ARG TEXTURE_PACK_NAME=resource-pack.zip
COPY tools/stub_atlases.py /opt/tools/stub_atlases.py
COPY tools/compose_texture_pack.py /opt/tools/compose_texture_pack.py
RUN set -e; \
    case "$ASSETS" in \
    stub) \
        uv run --no-project --with pillow python /opt/tools/stub_atlases.py \
            --netherite-home /opt/netherite; \
        test -f /opt/netherite/magma/assets/water_frames.h ;; \
    jar) \
        curl -fsSL -o /tmp/mc.jar \
            "https://piston-data.mojang.com/v1/objects/${MC_JAR_SHA1}/client.jar" \
        || curl -fsSL -o /tmp/mc.jar \
            "https://launcher.mojang.com/v1/objects/${MC_JAR_SHA1}/client.jar"; \
        echo "${MC_JAR_SHA1}  /tmp/mc.jar" | sha1sum -c -; \
        if [ -n "$TEXTURE_PACK_URL" ]; then \
            test -n "$TEXTURE_PACK_SHA256" || \
                (echo TEXTURE_PACK_SHA256 is required with TEXTURE_PACK_URL >&2; exit 1); \
            curl -fsSL -o /tmp/pack.zip "$TEXTURE_PACK_URL"; \
            echo "$TEXTURE_PACK_SHA256  /tmp/pack.zip" | sha256sum -c -; \
            uv run --no-project python /opt/tools/compose_texture_pack.py \
                --base /tmp/mc.jar --pack /tmp/pack.zip --out /tmp/composite.jar; \
            pack_digest="$(uv run --no-project python \
                /opt/tools/compose_texture_pack.py --digest --pack /tmp/pack.zip)"; \
            cd /opt/netherite && MC_JAR=/tmp/composite.jar bash scripts/bootstrap_assets.sh; \
            printf 'kind=jar\nsource=minecraft-client.jar+resource-pack\nsha1=%s\ntexture_pack=%s\ntexture_pack_sha256=%s\n' \
                "$MC_JAR_SHA1" "$TEXTURE_PACK_NAME" "$pack_digest" \
                > /opt/netherite/.bedrock-rl-assets; \
        else \
            cd /opt/netherite && MC_JAR=/tmp/mc.jar bash scripts/bootstrap_assets.sh; \
            printf 'kind=jar\nsource=minecraft-1.11.2-client.jar\nsha1=%s\n' \
                "$MC_JAR_SHA1" > /opt/netherite/.bedrock-rl-assets; \
        fi; \
        rm -f /tmp/mc.jar /tmp/pack.zip /tmp/composite.jar; \
        test -f /opt/netherite/magma/assets/water_frames.h ;; \
    none) echo "skipping atlases (ASSETS=none)" ;; \
    *) echo "ASSETS must be stub, jar or none" >&2; exit 1 ;; \
    esac

# ── build the sim ────────────────────────────────────────────────────────
FROM assets AS engine
# builder images export CC and CFLAGS, which override the Makefile's `?=`
# defaults and drop its -Icore -I. include paths
RUN env -u CC -u CFLAGS -u CXXFLAGS -u LDFLAGS \
        make -C /opt/netherite/magma game \
 && test -x /opt/netherite/magma/magma_game \
 && find /opt/netherite/magma -name '*.o' -delete
# The sparse checkout above never fetched the rest of the tree, so there
# is nothing left to prune here. What remains is the engine, its sources,
# and a .git that can answer for its own revision.

# ── cpu: the engine, the package, and nothing that only a trainer needs ──
FROM debian:bookworm-slim AS cpu
ENV DEBIAN_FRONTEND=noninteractive
# libsdl2-2.0-0 not libsdl2-dev: the engine links SDL2 and never builds
# again in this image. git stays, because the engine check asks the
# checkout what revision it is and that is the provenance guarantee.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git libsdl2-2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=toolchain /usr/local/bin/uv /usr/local/bin/uvx /usr/local/bin/
COPY --from=engine /opt/netherite /opt/netherite
# The image defaults to root, but batch systems commonly inject their own
# numeric uid. Git 2.35+ rejects a root-owned checkout for that uid unless the
# image declares the immutable checkout safe, which made `bedrock doctor`
# report a false provenance failure under Kubernetes/Slurm security contexts.
RUN git config --system --add safe.directory /opt/netherite
ARG PYTHON_VERSION
# One interpreter, at a fixed path, so every caller names the same one.
# Keep uv's managed interpreter outside its cache. A venv points at that
# interpreter, so deleting a cache-backed install can leave bin/python as a
# dangling symlink in an otherwise successful image.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN uv python install "$PYTHON_VERSION" \
 && uv venv --python "$PYTHON_VERSION" /opt/bedrock-env \
 && /opt/bedrock-env/bin/python -V
# UV_NO_CACHE, because a wheel cache inside the image is 11 GB of a GPU
# image that nobody will ever install from again. It was, measured.
ENV PATH=/opt/bedrock-env/bin:$PATH \
    VIRTUAL_ENV=/opt/bedrock-env \
    NETHERITE_HOME=/opt/netherite \
    BEDROCK_RL_HOME=/opt/bedrock-rl \
    BRL_CONTAINER_IMAGE=1 \
    SDL_VIDEODRIVER=dummy \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1
# dependencies first, off the metadata alone, so editing the package does
# not re-resolve the world
COPY pyproject.toml uv.lock README.md /opt/bedrock-rl/
RUN uv export --project /opt/bedrock-rl --frozen --no-dev \
        --extra netherite --extra data \
        --no-emit-project --output-file /tmp/brl-requirements.txt \
 && uv pip install --python /opt/bedrock-env/bin/python \
        --requirement /tmp/brl-requirements.txt \
 && rm /tmp/brl-requirements.txt
COPY . /opt/bedrock-rl
# editable, so bedrock_rl.resources.repo_root() finds a real checkout
# and examples/ and patches/ resolve. An installed wheel is not a working
# environment here, by design; see pyproject.toml.
RUN uv pip install --python /opt/bedrock-env/bin/python --no-deps \
        -e /opt/bedrock-rl
WORKDIR /opt/bedrock-rl
# the engine is at the pin with every patch, which is the one thing worth
# failing the build over
RUN python -c "from bedrock_rl.env.engine import engine_problems; \
p = engine_problems(); print(p or 'engine at the pin, patches applied'); \
raise SystemExit(1 if p else 0)"
CMD ["bedrock", "doctor", "--runtime-only"]

# ── gpu: the training stack on top ───────────────────────────────────────
FROM cpu AS gpu
ARG VERL_REPO
ARG VERL_SHA
ARG PYTHON_VERSION
ARG FLASH_ATTN_VERSION
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*
# The trainer, at the pin, with every shipped diff. Same shape as the
# engine clone above and for the same reason: a single commit fetched by
# name, then checked, so a redirected or shallow fetch that lands
# somewhere else stops the build instead of training on it.
COPY patches/verl /opt/patches/verl
COPY requirements /opt/bedrock-rl/requirements
RUN set -ex; \
    git init -q /opt/verl; \
    git -C /opt/verl remote add origin "$VERL_REPO"; \
    git -C /opt/verl fetch -q --depth 1 origin "$VERL_SHA"; \
    git -C /opt/verl checkout -q --detach FETCH_HEAD; \
    test "$(git -C /opt/verl rev-parse HEAD)" = "$VERL_SHA"; \
    git config --system --add safe.directory /opt/verl
# Every diff in patches/verl/, applied and then REVERSE-CHECKED.
#
# The forward apply already fails the build on a diff that does not fit.
# The reverse check is the other half: it is the only thing that can tell
# a diff that landed from one that was a no-op, and a silently unapplied
# LoRA diff is exactly the failure this stage exists to prevent. #6688
# and #7014 are both on the LoRA serving path, so without them a run
# starts, trains, logs a reward and serves an adapter that is not the one
# it trained. Nothing downstream notices.
#
# Each diff is checked on its own rather than only the topmost, which is
# valid HERE because the three touch different files
# (vllm_rollout/utils.py, engine/fsdp/transformer_impl.py, and
# workers/utils/padding.py) and so never
# appear in each other's context. bin/setup_deps.sh checks the same way.
# A future diff that edits a file another one already touched breaks that
# assumption and has to move to the prefix rule the engine patches use.
#
# The counted glob is not decoration: an unmatched glob would hand `git
# apply` a literal path, and a COPY that silently brought nothing would
# otherwise build an unpatched trainer that passes every later check.
RUN set -ex; set -- /opt/patches/verl/*.diff; test "$#" -ge 1; \
    for d; do git -C /opt/verl apply "$d"; done; \
    for d; do \
        git -C /opt/verl apply --reverse --check "$d" \
        || { echo "verl patch $d did not land in /opt/verl" >&2; exit 1; }; \
    done; \
    echo "verl at $VERL_SHA with $# patches applied"
# The training stack is CUDA-ABI-locked and lives in its OWN venv, exactly
# as on a box: bin/rl.sh runs out of VERL_ENV with this repo on
# PYTHONPATH. Same pins as bin/setup_deps.sh.
# The clone and the patch apply are the two RUNs above, and there is
# exactly one of each: the fetch-by-name form checks the object it got is
# the object it asked for, and the reverse check is the only thing that
# separates a diff that landed from one that was a no-op. The two LoRA
# fixes bin/setup_deps.sh calls required for a LoRA RL run to be CORRECT
# rather than merely to start (#6688 clones the adapter out of the reused
# IPC buffer before add_lora, #7014 syncs merged weights before the FSDP
# context exits) are the reason that matters: without them a run samples
# its rollouts from an adapter that is not the one being trained and says
# nothing at all about it, so the failure is a silently worse curve. This
# image applied neither, and the same defect was found live on two of
# four training boxes, which is why tests/test_dockerfile.py asserts the
# whole directory is applied rather than trusting a sentence.
#
# nvidia-cutlass-dsl<4.7 is here for the same reason it is in
# bin/setup_deps.sh, which has the long version: it is a dependency of
# nothing here, but from 4.7 the DSL drops `cutlass.cute.core.ThrMma`,
# vLLM 0.12's FlashAttention-4 probe catches only ImportError, and the
# resulting AttributeError escapes through verl's module-scope MoE
# weight-loader import. Every vLLM rollout in the venv then dies about
# thirty seconds in, dense models included. Measured on this stack: 4.5.3
# imports clean, 4.7.0 does not. It was missing here and present there,
# which made the image and the box two different environments.
RUN uv venv --python "$PYTHON_VERSION" /opt/verl-env \
 && uv pip install --python /opt/verl-env/bin/python \
        --torch-backend cu128 \
        --requirement /opt/bedrock-rl/requirements/verl-cu128.lock \
 && uv pip install --python /opt/verl-env/bin/python --no-deps -e /opt/verl
# verl reaches into flash_attn.bert_padding to compute log probs and that
# import has no fallback on CUDA, so the trainer needs the package even
# when the model itself runs sdpa. A prebuilt wheel matching torch 2.9 /
# cu12 / cp312 / cxx11abi TRUE, because compiling it here costs an hour.
RUN uv pip install --python /opt/verl-env/bin/python \
        "https://github.com/Dao-AILab/flash-attention/releases/download/\
v${FLASH_ATTN_VERSION}/flash_attn-${FLASH_ATTN_VERSION}%2Bcu12torch2.9cxx11abiTRUE\
-cp312-cp312-linux_x86_64.whl"
ENV PATH=/opt/verl-env/bin:$PATH \
    VERL_HOME=/opt/verl \
    VERL_ENV=/opt/verl-env \
    PYTHONPATH=/opt/bedrock-rl
RUN /opt/verl-env/bin/python -c \
    "import verl, verl.trainer.main_ppo; print('verl', verl.__version__)"
CMD ["bedrock", "doctor"]

# ── the stage a plain `docker build .` produces ──────────────────────────
# CPU stays the lightweight default. Ask for `--target gpu` explicitly
# when building a training image.
FROM ${FINAL_STAGE} AS default
