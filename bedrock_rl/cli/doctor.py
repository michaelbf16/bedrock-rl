"""Why the last command failed, before you run it.

Every check stands for a failure someone has already hit, and each names
the fix rather than the symptom. The checks cover the repo checkout and
the commit it is on, uv, the engine checkout and its pinned SHA, the
applied patches, whether the built binary predates its sources, the
texture atlases, the verl checkout, the diffs applied to it, the
training venv, verl itself, flash-attn, the model weights and free GPUs.

The verl patch row is the newest and it is here because the divergence
it reports was found the expensive way. patches/verl/ carries two fixes
on the LoRA weight-sync path and one for Qwen-VL ragged position IDs.
A box without them can silently serve stale adapter weights or fail only
after an expensive rollout. This box had exactly that state while two
others did not, so "which trainer is this" is a question every box has
to be able to answer about itself.

For multi-node runs the training venv, pinned engine, model weights path, and
repo commit must match on every node. The corresponding checks print values to
compare rather than verdicts, since one box cannot inspect another.

Required checks failing exits non-zero. Warnings do not, because a
warning is a thing that will bite later rather than now.
"""
import os
import shutil
import subprocess

from bedrock_rl import resources
from bedrock_rl.env import engine

# The trainer diffs, and the tree they belong to. Resolved once at module
# scope rather than inside the check, because `resources.data_path`
# answers differently in a checkout and in an installed wheel and every
# caller has to get the same answer.
VERL_PATCH_DIR = resources.data_path("patches", "verl")


class Check:
    """One row of the table. `state` is ok, fail, warn or skip."""

    def __init__(self, name, state, detail="", fix=""):
        self.name = name
        self.state = state
        self.detail = detail
        self.fix = fix

    @property
    def broken(self):
        return self.state == "fail"


class Timeout:
    """A run that did not finish. Distinct from a run that failed, because
    "verl does not import" and "verl took longer than two minutes to
    import" call for different next steps."""
    returncode = -1
    stdout = stderr = ""


def _run(argv, **kw):
    try:
        return subprocess.run(argv, capture_output=True, text=True,
                              timeout=120, **kw)
    except subprocess.TimeoutExpired:
        return Timeout()
    except (OSError, subprocess.SubprocessError):
        return None


def _ok(r):
    return r is not None and r.returncode == 0


def check_uv():
    path = shutil.which("uv")
    if not path:
        return Check("uv", "fail", "not on PATH",
                     "curl -LsSf https://astral.sh/uv/install.sh | sh")
    r = _run([path, "--version"])
    ver = r.stdout.strip() if _ok(r) else "unknown version"
    return Check("uv", "ok", f"{ver} at {path}")


def _newer_sources(magma, binary):
    """The name of one engine source newer than the built binary, or None.

    Object files are not consulted, only sources, because a stale .o is
    what make itself will notice and a stale binary is what it will not.
    """
    try:
        built = binary.stat().st_mtime
    except OSError:
        return None
    newest = None
    for pattern in ("**/*.c", "**/*.h", "**/*.cu"):
        for p in magma.glob(pattern):
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            if mtime > built and (newest is None or mtime > newest[1]):
                newest = (p.name, mtime)
    return None if newest is None else newest[0]


def engine_checks(netherite_home=None):
    root = engine.engine_home(netherite_home)
    setup = "bedrock setup"
    if not root.exists():
        return [Check("engine checkout", "fail", f"{root} does not exist",
                      setup),
                Check("engine patches", "skip", "no checkout to inspect"),
                Check("engine binary", "skip", "no checkout to inspect"),
                Check("texture atlases", "skip", "no checkout to inspect")]

    out = []
    want = engine.pinned_sha()
    head = engine.head_sha(netherite_home)
    if want is None:
        out.append(Check("engine checkout", "fail",
                         f"{engine.PINNED_SHA_FILE} is unreadable, so no "
                         "checkout can be verified"))
    elif head is None:
        out.append(Check("engine checkout", "fail",
                         f"{root} is not a git checkout", setup))
    elif head == want:
        out.append(Check("engine checkout", "ok",
                         f"{root} at the pin {want[:12]}"))
    else:
        out.append(Check("engine checkout", "fail",
                         f"{root} is at {head[:12]}, the pin is {want[:12]}",
                         setup))

    # The stack this engine CLAIMS, not the contents of
    # patches/netherite/. A diff nobody has built with is inert, so it is
    # reported in the detail and does not colour the row; asking the
    # directory instead turns this row red everywhere the moment a patch
    # is staged, while no engine has changed. engine.patch_status has the
    # reasoning.
    #
    # The counts are two different denominators on purpose. With a
    # manifest the engine's own claim is the fact and the directory is
    # merely context, and the two can legitimately differ in both
    # directions, so writing it as N/M invited "4/3 adopted".
    patches = engine.shipped_patches()
    if not patches:
        out.append(Check("engine patches", "fail",
                         f"no diffs under {engine.PATCH_DIR}"))
    elif head is None:
        out.append(Check("engine patches", "skip",
                         "not a git checkout, so patches cannot be checked"))
    else:
        st = engine.patch_status(netherite_home)
        if st["manifest"]:
            detail = "%d adopted of %d shipped" % (len(st["adopted"]),
                                                   len(patches))
        else:
            detail = ("%d of %d shipped, inferred from the tree and only %s "
                      "verified (this engine records no %s)"
                      % (len(st["adopted"]), len(patches),
                         st["verified"] or "nothing",
                         engine.PATCH_MANIFEST_NAME))
        # Notes ride along on a failing row too. They used to be dropped
        # there, which hid a staged patch at the one moment somebody is
        # reading this table to work out what happened.
        parts = ([("; ".join(st["problems"]))] if st["problems"] else [detail])
        out.append(Check("engine patches",
                         "fail" if st["problems"] else "ok",
                         "; ".join(parts + st["notes"]),
                         setup if st["problems"] else ""))

    magma = engine.binary_dir(netherite_home)
    binaries = [(magma / engine.binary_name(r), f"{r} raster")
                for r in engine.RASTER_BACKENDS]
    built = [f"{p.name} ({what})" for p, what in binaries if p.exists()]
    game = binaries[0][0]
    if not game.exists():
        out.append(Check("engine binary", "fail",
                         f"{game} is not built",
                         f"make -C {magma} game"))
    else:
        # A binary older than its own sources is the failure this check
        # exists for. bin/setup_deps.sh only builds when magma_game is
        # ABSENT, so applying a patch to a checkout that already has one
        # leaves a binary that predates the patch, and every check above
        # this line still passes. Nothing in the observation record says
        # which revision produced it.
        stale = _newer_sources(magma, game)
        if stale:
            out.append(Check("engine binary", "warn",
                             f"{game.name} is older than {stale}, so it "
                             "predates a source change in this checkout",
                             f"make -C {magma} game"))
        else:
            out.append(Check("engine binary", "ok", ", ".join(built)))

    # WHICH atlases, not merely whether there are any. A stub build and a
    # jar build render different pixels, so a rollout scored against one
    # cannot be replayed against the other, and this is the only place that
    # says which one this checkout is.
    atlas = magma / "assets" / "water_frames.h"
    marker = magma / "assets" / "STUB_ATLASES"
    if marker.exists():
        first = marker.read_text().splitlines()[0] if marker.read_text() else ""
        out.append(Check("texture atlases", "ok",
                         f"stub textures ({first}), no Minecraft jar used"))
    elif atlas.exists():
        out.append(Check("texture atlases", "ok",
                         "generated from a Minecraft jar"))
    else:
        out.append(Check("texture atlases", "fail",
                         f"{atlas} is missing, so the asset bootstrap has "
                         "never run in this checkout",
                         "bedrock setup"))
    return out


def _verl_root(verl_home=None):
    """The trainer checkout bin/rl.sh will cd into."""
    return os.path.expanduser(verl_home
                              or os.environ.get("VERL_HOME", "~/verl"))


def verl_home_check(verl_home=None):
    """bin/rl.sh does `cd $VERL_HOME`, so this is a real prerequisite and
    not the same fact as the training venv importing verl."""
    root = _verl_root(verl_home)
    if not os.path.isdir(os.path.join(root, "verl", "trainer")):
        return Check("verl checkout", "fail",
                     f"{root} does not hold a verl tree, and bin/rl.sh "
                     "changes into it before starting the trainer",
                     "bedrock setup")
    return Check("verl checkout", "ok", root)


def verl_patches(root, patches):
    """Split `patches` into the ones this tree carries and the ones it
    does not, by the reverse-check rule.

    Each diff is asked about INDEPENDENTLY rather than by the prefix rule
    the engine patches use. That is valid for this set because the three
    touch three different files and so never appear in each other's
    context; it is a property of the set and not a general one. The same
    reasoning, and the same set, is in bin/setup_deps.sh and the
    Dockerfile.
    """
    have, missing = [], []
    for p in patches:
        r = _run(["git", "-C", str(root), "apply", "--reverse", "--check",
                  str(p)])
        (have if _ok(r) else missing).append(p)
    return have, missing


def verl_patch_check(verl_home=None):
    """Which shipped diffs the trainer checkout actually carries.

    A row and not a footnote, because this is the check that would have
    caught a box training LoRA on a verl with neither LoRA fix. `git
    apply --check` succeeding FORWARD is the same evidence, from the
    other side: a diff that still applies is a diff that is not there.
    """
    root = _verl_root(verl_home)
    name = "verl patches"
    if not os.path.isdir(os.path.join(root, ".git")):
        return Check(name, "skip",
                     f"{root} is not a git checkout, so its diffs cannot "
                     "be checked")
    patches = sorted(VERL_PATCH_DIR.glob("*.diff"))
    if not patches:
        return Check(name, "fail", f"no diffs under {VERL_PATCH_DIR}")
    have, missing = verl_patches(root, patches)
    if not missing:
        return Check(name, "ok", f"{len(have)}/{len(patches)} applied "
                                 f"in {root}")
    return Check(name, "fail",
                 f"{len(have)}/{len(patches)} applied in {root}, missing "
                 + ", ".join(p.stem for p in missing)
                 + ("; 0001 and 0002 are the LoRA weight-sync fixes, so a "
                    "LoRA run here serves an adapter it did not train"
                    if any(p.stem.startswith(("0001", "0002"))
                           for p in missing) else ""),
                 "bedrock setup")


def _venv_python(verl_env=None):
    """The training venv's interpreter, or None when there is not one."""
    venv = os.path.expanduser(verl_env
                              or os.environ.get("VERL_ENV", "~/verl-env"))
    py = os.path.join(venv, "bin", "python")
    return py if os.access(py, os.X_OK) else None


def venv_checks(verl_env=None):
    venv = os.path.expanduser(verl_env
                              or os.environ.get("VERL_ENV", "~/verl-env"))
    py = os.path.join(venv, "bin", "python")
    setup = "bedrock setup"
    if not os.access(py, os.X_OK):
        return [Check("training venv", "fail", f"no interpreter at {py}",
                      setup),
                Check("verl", "skip", "no training venv"),
                Check("flash-attn", "skip", "no training venv"),
                Check("peft", "skip", "no training venv")]
    out = [Check("training venv", "ok", py)]
    # importing the trainer entrypoint is the real check; a bare
    # `import verl` passes even when a runtime dep is missing
    r = _run([py, "-c", "import verl, verl.trainer.main_ppo; "
                        "print(verl.__version__)"])
    if _ok(r):
        out.append(Check("verl", "ok", f"{r.stdout.strip()}, "
                                       "trainer entrypoint imports"))
    elif isinstance(r, Timeout):
        out.append(Check("verl", "warn",
                         "importing verl.trainer.main_ppo took longer than "
                         "120s, so it was not waited out. The venv is "
                         "probably fine and the box is probably loaded"))
    else:
        lines = r.stderr.strip().splitlines() if r else []
        tail = lines[-1] if lines else "no output"
        out.append(Check("verl", "fail",
                         f"verl.trainer.main_ppo does not import, {tail}",
                         setup))
    r = _run([py, "-c", "import flash_attn; print(flash_attn.__version__)"])
    if _ok(r):
        out.append(Check("flash-attn", "ok", r.stdout.strip()))
    else:
        out.append(Check("flash-attn", "warn",
                         "not importable, so RL dies at the first log-prob "
                         "step; data generation and SFT still run",
                         "MAX_JOBS=8 uv pip install --python " + py
                         + " --no-build-isolation flash-attn==2.8.3"))
    # A warning and not a failure, because a full fine-tune needs none of
    # this. It is the check that answers "can this box do LORA_RANK", and
    # every path that can, verl's included, goes through peft.
    r = _run([py, "-c", "import peft; print(peft.__version__)"])
    if _ok(r):
        out.append(Check("peft", "ok", f"{r.stdout.strip()}, so LORA_RANK "
                                       "works in every trainer"))
    else:
        out.append(Check("peft", "warn",
                         "not importable, so LORA_RANK fails at model load; "
                         "full fine-tuning is unaffected",
                         f"uv pip install --python {py} peft"))
    return out


def _compute_gpus():
    """Indices with a compute process on them, as strings.

    Memory alone lies in the direction that costs someone their run. A
    trainer that has just started, or one between phases, holds almost
    nothing and is still using the device. This is the authoritative
    answer; the memory numbers below are the detail.
    """
    r = _run(["nvidia-smi", "--query-compute-apps=gpu_uuid",
              "--format=csv,noheader"])
    if not _ok(r) or not r.stdout.strip():
        return set()
    uuids = {u.strip() for u in r.stdout.split() if u.strip()}
    r = _run(["nvidia-smi", "--query-gpu=index,uuid",
              "--format=csv,noheader"])
    if not _ok(r):
        return set()
    out = set()
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[1] in uuids:
            out.add(parts[0])
    return out


def gpu_check():
    if not shutil.which("nvidia-smi"):
        return Check("gpus", "skip", "no nvidia-smi on this box")
    r = _run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
              "--format=csv,noheader,nounits"])
    if not _ok(r):
        return Check("gpus", "warn", "nvidia-smi did not answer")
    working = _compute_gpus()
    free, busy, unknown = [], [], []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        idx = parts[0]
        try:
            used, total = int(parts[1]), int(parts[2])
        except ValueError:
            # MIG parents and some vGPU setups answer [N/A] here. One
            # device that cannot report its memory must not cost the
            # whole table, which is what an escaping ValueError did.
            unknown.append(f"{idx} ({parts[1]})")
            continue
        # a few hundred MiB with no process is a display server
        where = busy if (idx in working or used >= 1024) else free
        where.append(f"{idx} ({used}/{total} MiB)")
    detail = []
    if free:
        detail.append("free: " + ", ".join(free))
    if busy:
        detail.append("in use: " + ", ".join(busy))
    if unknown:
        detail.append("no memory reading: " + ", ".join(unknown))
    if not free:
        return Check("gpus", "warn", "; ".join(detail) or "no devices",
                     "CUDA_VISIBLE_DEVICES picks the devices a run may "
                     "touch")
    return Check("gpus", "ok", "; ".join(detail))


def repo_check():
    from bedrock_rl import resources
    root = resources.repo_root()
    if root is None:
        return Check("repo checkout", "warn",
                     "installed from a wheel; setup, actual training, eval, "
                     "rendering, and included example discovery need a clone "
                     "(library commands and config dry-runs still work)",
                     "git clone https://github.com/michaelbf16/bedrock-rl "
                     "bedrock-rl")
    return Check("repo checkout", "ok", str(root))


def commit_check():
    """Which commit of THIS repo is about to run.

    One of the four facts a multi-node run needs identical everywhere,
    and the one nothing could report. A skew here is a rollout loop, a
    reward function or a task file that differs between nodes, which
    produces a run that trains rather than one that fails.
    """
    from bedrock_rl import resources
    root = resources.repo_root()
    if root is None:
        return Check("repo commit", "skip", "installed from a wheel")
    r = _run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"])
    if not _ok(r):
        if os.environ.get("BRL_CONTAINER_IMAGE") == "1":
            return Check(
                "repo commit", "skip",
                "source copied into a container image; use the same image "
                "digest on every node")
        return Check("repo commit", "warn", f"{root} is not a git checkout",
                     "compare the trees on every node by hand")
    sha = r.stdout.strip()
    d = _run(["git", "-C", str(root), "status", "--porcelain"])
    dirty = bool(_ok(d) and d.stdout.strip())
    detail = sha + (", with uncommitted changes" if dirty else "")
    if dirty:
        return Check("repo commit", "warn",
                     detail + "; every node must be on the same tree",
                     "commit or stash, then sync every node")
    return Check("repo commit", "ok",
                 detail + "; every node must report this same commit")


def model_check(model=None):
    """The weights the next run would load, and whether they are here.

    A LOCAL path is the case that bites on more than one node: Ray ships
    tasks and not files, so a checkpoint directory that exists on the
    head and not on a worker fails there and nowhere else. A hub id is
    fine everywhere and is reported as such.
    """
    name = model or os.environ.get("MODEL") or "qwen3-vl"
    try:
        from bedrock_rl import models
        location = models.resolve_model_location(name)
        expanded = os.path.expanduser(location)
        explicit_path = (os.path.isabs(expanded)
                         or str(name).startswith(("~", "." + os.sep, "./")))
        if explicit_path and not os.path.isdir(expanded):
            return Check("model weights", "fail",
                         f"{name} resolves to {expanded}, which is not a "
                         f"directory here",
                         f"put the weights at {expanded} on every node")
        models.resolve(location)
        path = models.model_path(location)
    except Exception as e:                          # pragma: no cover
        return Check("model weights", "fail", f"{name}: {e}",
                     "bedrock models list")
    if os.path.isdir(os.path.expanduser(path)):
        full = os.path.abspath(os.path.expanduser(path))
        if not os.path.isdir(full):
            return Check("model weights", "fail",
                         f"{name} resolves to {full}, which is not a "
                         f"directory here",
                         f"put the weights at {full} on every node")
        return Check("model weights", "ok",
                     f"{full}; this exact path must exist on every node")
    return Check("model weights", "ok",
                 f"{path}, a hub id, so every node resolves it the same")


def ray_check(verl_env=None):
    """Whether this node can join a multi-node run.

    Only meaningful once NNODES or RAY_ADDRESS says the user means to,
    and it answers the question the other checks do not: this box is
    provisioned, but is it provisioned the SAME as the others. It reports
    ray's version and the address it would dial, because a version skew
    across nodes and a head nobody is listening on are the two ways a
    cluster fails that look identical from here.
    """
    nnodes = os.environ.get("NNODES", "1")
    address = os.environ.get("RAY_ADDRESS", "")
    if nnodes in ("", "1") and not address:
        return Check("ray cluster", "skip",
                     "single node (set NNODES to check)")
    py = _venv_python(verl_env)
    if py is None:
        return Check("ray cluster", "fail", "no training venv to ask",
                     "bash bin/setup_deps.sh")
    r = _run([py, "-c", "import ray; print(ray.__version__)"])
    if not _ok(r):
        return Check("ray cluster", "fail", "ray does not import",
                     f"{py} -m pip install 'ray[default]'")
    ver = r.stdout.strip()
    if not address:
        return Check("ray cluster", "fail",
                     f"ray {ver}, but RAY_ADDRESS is unset",
                     "ray start --head on one node, ray start --address "
                     "on the rest, then export RAY_ADDRESS")
    return Check("ray cluster", "ok",
                 f"ray {ver}, NNODES={nnodes}, RAY_ADDRESS={address}; "
                 f"every node needs this same version, the engine at the "
                 f"pinned SHA and the same weights path")


def run(netherite_home=None, verl_env=None, verl_home=None, model=None,
        runtime_only=False):
    """Every check, in the order a first run hits them."""
    checks = [check_uv(), repo_check(), commit_check()]
    checks += engine_checks(netherite_home)
    if runtime_only:
        return checks
    checks.append(verl_home_check(verl_home))
    checks.append(verl_patch_check(verl_home))
    checks += venv_checks(verl_env)
    checks.append(model_check(model))
    checks.append(gpu_check())
    checks.append(ray_check(verl_env))
    return checks
