"""Driver for one `magma_game --rl-bin` process (netherite's game).

Protocol: one JSON action line on stdin -> one packed RlBinObs record on
stdout per tick. `act` is the action side and `_read_obs` is the record
layout. Worldgen comes from --seed, so episodes need no snapshots;
--snapshot-in is supported for curriculum start states. --frames-out
additionally renders the real game view per tick for VL policies.
capture_every=N renders every N ticks for a demo recording; the default
renders only on "snap" ticks, which is what keeps rollouts affordable.
Either way the engine numbers the files it writes and `frame` reads that
number back off the directory, because the two cadences number
differently and a copy of the counter here was wrong under one of them.

The binary record's layout is frozen. The JSON protocol (--rl) carries
two fields it does not, "furnace" [burn, current_burn, cook, total_cook]
and "furnace_slots" [[item, count, meta] x3] for input, fuel and output,
both zero when no furnace is open, so reward code that needs smelt
progress reads the JSON protocol instead.

Every wait on that process is bounded and every wait knows whether the
process is still alive, because neither used to be true and two held-out
evaluations hung for ever on it. NETHERITE_ENGINE_TIMEOUT is the bound and
DEFAULT_ENGINE_TIMEOUT below has the measurements it was chosen from.

`raster` picks which binary drives the rendered view, "cpu" (magma_game,
the default) or "cuda" (magma_game_cuda --backend cuda). The two produce
bit-identical frames, so it is purely a performance choice. `gpu` pins a
CUDA environment process to a specific device so environment workers can
stay off the trainer's GPUs.
"""
import hashlib
import io
import json
import math
import os
import re
import select
import signal as _signal
import struct
import subprocess
import sys
import time
from pathlib import Path

from bedrock_rl.env import journal as _journal
from bedrock_rl.env.catalog import INV_ORDER
from bedrock_rl import resources

CAM_W, CAM_H = 64, 36
NPIX = CAM_W * CAM_H
N_BLOCKS, N_LOGS, N_COAL = 256, 64, 32
BIN_MAGIC = b"BOLR"
HEAD_FMT = "<qdddffi"          # tick,x,y,z,yaw,pitch,dead (after magic)
OBS_INT_COUNT = 9 + 9 + 9 + 27 + 27 + 27 + 1 + 1 + 9
BIN_HEAD = 4 + struct.calcsize(HEAD_FMT) + OBS_INT_COUNT * 4
BIN_SIZE = (BIN_HEAD + N_BLOCKS * 4 * 4 + N_LOGS * 3 * 4 + N_COAL * 3 * 4
            + NPIX * 2 + NPIX + NPIX)
# depth u8 = FLOOR(dist * 4), so a reading is the distance rounded DOWN
# to a quarter block and never up: 4.99 blocks reads as 4.75. 255 means
# the ray reached OC_FAR = 48 blocks (blaze/core/obs_camera.h) without
# hitting anything, which is the camera's range and not any interaction
# reach; this used to say "no hit within reach" and there is no reach
# involved.
DEPTH_SCALE = 4.0

# Rasterizer backend for the --frames-out view. "cpu" is magma_game (the
# CPU rasterizer, no CUDA runtime linked); "cuda" is magma_game_cuda run
# with --backend cuda. The semantic camera (cam/depth/edge in RlBinObs) is
# CPU raycast either way and is NOT affected by this choice.
#
# The CUDA rasterizer is faster once warm and still loses on a short episode,
# because a cold episode respawns the process to restore its snapshot and
# opening a CUDA context costs more boot than the raster saves.
# Break-even is around 11 rendered turns. CUDA also holds GPU memory per
# process, so environment concurrency has to fit the card.
RASTER_BACKENDS = ("cpu", "cuda")
DEFAULT_RASTER = os.environ.get("NETHERITE_RASTER", "cpu")


_BIN_FOR_RASTER = {"cpu": "magma_game", "cuda": "magma_game_cuda"}

# The engine names capture files frame_<n>.ppm and IT owns n. The digit
# count is not asserted here: the engine writes %06d and rolls to seven
# digits past a million frames, which a fixed-width reader would silently
# stop finding.
FRAME_FILE_RE = re.compile(r"^frame_(\d+)\.ppm$")

# Serial for capture subdirectories, process-wide and not per instance.
# frame() reads the engine's index off the directory, so two envs sharing
# one subdir would serve each other's frames. Two MagmaEnvs built in one
# process on the same seed and the same frames_dir used to land in the
# same subdir, because the counter was per instance.
_CAPTURE_SERIAL = 0

# Chunk radius the renderer builds meshes for, 1 to 8, engine default 8.
#
# It is NOT render-only, which this comment used to say. The engine sets
# `randtick_radius = view_distance` (magma/game/runtime.c), so it is also
# the radius over which random block ticks run, which is grass spread and
# leaf decay, and that is simulation and not drawing.
#
# The mechanism, exactly, because "simulation knob" on its own reads as a
# defect and this one has never produced an observable difference.
# `gm_randtick_pass` sweeps a CHEBYSHEV box of randtick_radius CHUNKS
# around the player's chunk, all sixteen sections of each, so the full
# 0..255 column. At the smallest setting the engine accepts, 1, that box
# is three chunks on a side, and wherever in their own chunk the player
# stands it reaches at least 16 blocks in every horizontal direction. The
# observation scan reads RL_OBS_R = 16 blocks (rl_mode.c), so every block
# a check can see is inside the random-tick box at every legal setting,
# with equality in the worst case. That is why no one has made it change
# an observation, and it is a coincidence of two numbers rather than a
# guarantee: raise RL_OBS_R above 16 and a low view distance starts
# freezing grass and leaves the observation reports.
# Both numbers are the engine's, restated here, and nothing compares the
# copies: raising RL_OBS_R in rl_mode.c without revisiting this comment
# turns a coincidence into a bug that only shows up as observations that
# stop updating at low view distance.
#
# What it does change either way is the volume being simulated, so a run
# at 4 and a run at 8 are not the same experiment even where their
# observations agree.
#
# The semantic camera is a separate raycast over a region tensor with its
# own 48-block reach and is not built from the renderer's chunk meshes,
# so the chunk radius does not reach the observation text through the
# camera; only the far terrain in the rendered frame changes.
#
# It is the largest single lever on data-generation wall clock, because the
# first render in a process builds those meshes and dominates a short episode.
#
# Default stays at the engine default so nothing changes silently. If you
# lower it to generate data faster you MUST lower it the same way for
# rollout and evaluation, because a policy trained on frames with terrain
# out to 4 chunks and deployed on frames with terrain out to 8 is being
# shown a different picture.
DEFAULT_VIEW_DISTANCE = int(os.environ.get("NETHERITE_VIEW_DISTANCE", 8))


def binary_name(raster="cpu"):
    """The file the engine build produces for a rasterizer."""
    if raster not in RASTER_BACKENDS:
        raise ValueError(f"raster must be one of {RASTER_BACKENDS}, "
                         f"got {raster!r}")
    return _BIN_FOR_RASTER[raster]


def binary_dir(netherite_home=None):
    """The directory this checkout keeps its engine build in.

    A checkout is laid out as magma/ or as c/magma/, and the one that
    counts is the one the binary is actually IN, which is the rule
    find_binary resolves by. Answering with the first directory that
    merely EXISTS reports a working c/magma/ engine as unbuilt, so this
    is the one place the layout is decided and the CLI doctor asks here
    rather than looking again. With nothing built it names the first
    candidate, which is where make would put it.
    """
    root = engine_home(netherite_home)
    dirs = [root / "magma", root / "c" / "magma"]
    for d in dirs:
        if any((d / _BIN_FOR_RASTER[r]).exists() for r in RASTER_BACKENDS):
            return d
    for d in dirs:
        if d.is_dir():
            return d
    return dirs[0]


def find_binary(netherite_home=None, raster="cpu"):
    name = binary_name(raster)
    root = engine_home(netherite_home)
    for p in (root / "magma" / name, root / "c" / "magma" / name):
        if p.exists():
            return p
    target = "game" if raster == "cpu" else "game-cuda"
    hint = ("" if raster == "cpu" else
            " NVFLAGS_GAME='-O2 --fmad=false -arch=sm_XX -Icore -I.'"
            " (sm_80 A100, sm_90 H100; rm cuda/*.o when changing arch)")
    raise FileNotFoundError(
        f"{name} not found under {root}; build with `make -C "
        f"{root}/magma {target}`{hint}")


# ── engine provenance ────────────────────────────────────────────────────
# A smoke test that passes against the wrong engine is the worst failure
# this repo has: every number it prints is about a binary nobody pinned.
# So every process that opens an env says so, once, on stderr.
#
# The pin lives in ONE file, patches/netherite/PINNED_SHA, and is read by
# ONE function. bin/setup_deps.sh cats it directly because it is
# shell; every python reader calls pinned_sha().
PATCH_DIR = resources.data_path("patches", "netherite")
PINNED_SHA_FILE = PATCH_DIR / "PINNED_SHA"
_ENGINE_CHECKED = set()

# What a built engine records about the patch stack it was built with,
# written into the engine checkout by bin/setup_deps.sh. One line per
# applied diff:
#
#     <sha256 of the .diff>  <filename>
#
# The name is spelled out again in bin/setup_deps.sh, which writes it
# directly because it is shell, exactly as it cats PINNED_SHA directly;
# every python reader goes through read_patch_manifest(). It is
# untracked and cannot make a clean engine look dirty, because
# sync_pinned_checkout in that script inspects the checkout with
# `git status --porcelain --untracked-files=no`. The dot buys nothing
# from git and only keeps it out of a bare `ls`.
PATCH_MANIFEST_NAME = ".bedrock-rl-patches"


def pinned_sha():
    """The netherite commit this repo's patches are generated against, or
    None when no copy of the pin file is reachable.
    """
    for path in (PINNED_SHA_FILE,):
        try:
            return path.read_text().split()[0]
        except (OSError, IndexError):
            continue
    return None


def engine_home(netherite_home=None):
    return Path(os.path.expanduser(
        netherite_home or os.environ.get("NETHERITE_HOME", "~/netherite")))


def _git(root, *args, timeout=30):
    try:
        return subprocess.run(("git", "-C", str(root)) + args,
                              capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None


def head_sha(netherite_home=None):
    """The engine checkout's HEAD, or None when git cannot say.

    One asker, because "is this the engine the repo pins" is one
    question: engine_problems answers it for every process that opens an
    env, and the CLI doctor shows it in a table.
    """
    r = _git(engine_home(netherite_home), "rev-parse", "HEAD")
    return r.stdout.strip() if r is not None and r.returncode == 0 else None


def patch_applied(netherite_home=None, patch=None):
    """Does this checkout carry `patch`, by the reverse-check rule."""
    r = _git(engine_home(netherite_home), "apply", "--reverse", "--check",
             str(patch))
    return r is not None and r.returncode == 0


def shipped_patches():
    """What THIS REPO ships, which is not what any engine carries.

    Apply order is the filename order, which is why they are numbered;
    every reader of this list has to remember that it is the repo's
    current opinion and an engine's manifest is the engine's.
    """
    return sorted(PATCH_DIR.glob("*.diff"))


def patch_digest(path):
    """sha256 of a diff file's bytes, which is what a manifest records.

    The content and not the name, because a diff can be edited in place
    and keep its number, and an engine built against the old text is
    then a different engine from the one the file now describes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def applied_prefix(netherite_home=None, patches=()):
    """How many of `patches` this checkout carries, by the prefix rule.

    A checkout can only hold a PREFIX of the stack, so the highest patch
    that reverse-checks clean says how many are there. Only that one is
    verified: the ones below it are inferred from the prefix property
    and cannot be tested individually, which is what patch_status has to
    qualify when it reports an inferred stack.
    """
    for i in range(len(patches) - 1, -1, -1):
        if patch_applied(netherite_home, patches[i]):
            return i + 1
    return 0


def patch_manifest_path(netherite_home=None):
    return engine_home(netherite_home) / PATCH_MANIFEST_NAME


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def read_patch_manifest(netherite_home=None):
    """{diff filename: recorded sha256} for a checkout, or None.

    None means this engine records nothing about how it was built, which
    is every engine built before the manifest existed; patch_status
    below says what happens then.

    A line that is neither blank, a comment, nor `<sha256> <name>` is
    raised rather than skipped. Skipping was the first version and it
    fails in the worst direction: a truncated write leaves the engine
    claiming a SHORTER stack, the reverse-check then lands on a lower
    patch and passes, and the checkout reports clean while nothing knows
    what it lost. UnicodeDecodeError is caught with OSError for the same
    reason it must not escape: this runs on every env open, and a
    mangled file has to degrade to "no manifest" rather than take the
    process down.
    """
    try:
        text = patch_manifest_path(netherite_home).read_text()
    except (OSError, ValueError):
        return None
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # split once, so a diff whose name has a space in it survives
        parts = line.split(None, 1)
        if len(parts) != 2 or not _SHA256_RE.match(parts[0]):
            raise ValueError(
                f"{patch_manifest_path(netherite_home)} has a line that is "
                f"not `<sha256> <filename>`: {line!r}. An engine whose "
                "manifest cannot be read has to be rebuilt or re-recorded "
                "with bin/setup_deps.sh, because a half-read one "
                "would claim a shorter stack than it has")
        out[parts[1].strip()] = parts[0]
    # A present but empty manifest is indistinguishable from a botched
    # write, and the no-manifest path below handles that checkout
    # correctly, so it is treated as absent rather than as an engine
    # claiming it was built with nothing.
    return out or None


# ── what a patch file is allowed to break ────────────────────────────────
# A patch is INERT until an engine claims it. That is the whole rule, and
# it exists because the check used to compare a checkout against whatever
# .diff files were in patches/netherite/ AT THAT MOMENT.
#
# Under that question, writing a new .diff into the directory fails
# preflight everywhere at once -- "<new>.diff is not applied to this
# checkout, so the shipped patch stack is not complete here" -- while no
# engine anywhere has changed. Staging a patch for the next build and
# invalidating every running job become the same act, which makes an
# engine patch unstageable. That is a property of the QUESTION and not of
# any engine: the engines are fine.
#
# So the question is asked of the engine's own manifest instead.
#
#   adopted, and the top of the stack still reverse-checks   nothing
#   adopted, but the diff's CONTENT has changed since        FAIL
#   adopted, but the diff is no longer shipped               FAIL
#   adopted, and the tree no longer carries the stack        FAIL
#   in the tree, in no manifest, numbered INSIDE the stack   FAIL
#   shipped and in no manifest, numbered above it            a NOTE
#
# Every failure there is drift that makes a measurement describe a binary
# nobody pinned, which is what this whole section is for. The note is a
# patch staged for the next build, which changes no engine.
def patch_status(netherite_home=None):
    """This checkout's patch stack against the one it claims.

    Returns {"manifest": bool, "adopted": [names], "problems": [str],
    "notes": [str]}. Problems are drift and fail a check; notes are
    reported and fail nothing.
    """
    shipped = shipped_patches()
    by_name = {p.name: p for p in shipped}
    out = {"manifest": False, "adopted": [], "verified": None,
           "problems": [], "notes": []}
    if not shipped:
        out["problems"].append(
            f"no patches found under {PATCH_DIR}, so the engine cannot be "
            "checked for them")
        return out

    try:
        manifest = read_patch_manifest(netherite_home)
    except ValueError as e:
        # A failure and not an exception: this runs inside every env
        # open, and a mangled manifest must stop the RESULT rather than
        # the process.
        out["problems"].append(str(e))
        return out
    if manifest is None:
        # An engine with no manifest is every engine that existed when
        # this was written, so "gracefully" cannot mean "fail" and it
        # cannot mean the old behaviour either, because the old
        # behaviour IS the bug: it tests the topmost SHIPPED diff and so
        # goes red the moment anyone stages one.
        #
        # It infers the stack instead, by the same prefix walk
        # bin/setup_deps.sh uses to decide what to apply, and treats
        # what it finds as adopted. The floor is kept: a checkout that
        # carries NO shipped patch is not a patched engine at all and
        # still fails, which is the failure this check was created for
        # (SFT trained against an unpinned checkout measured
        # 82.3% on that engine and 52.1% on the pinned one).
        #
        # What is lost is real and worth naming, in three parts. Without
        # a manifest this cannot tell "staged, never adopted" from "was
        # adopted and has since been reverted" or from "was edited in
        # place", so the missing tail is a note in all three cases where
        # the old check failed on all three. It hashes nothing, so a diff
        # edited in place is invisible here. And only ONE of the diffs it
        # reports is verified, the top; the rest come from the prefix
        # property, so a diff numbered into the middle of the stack is
        # claimed as adopted without ever being tested. `verified` is in
        # the result so that a provenance reader can say which is which
        # instead of printing a list that reads as
        # if every entry was checked.
        #
        # All three are reasons to have a manifest, and re-running
        # bin/setup_deps.sh on an existing engine writes one without
        # rebuilding anything.
        have = applied_prefix(netherite_home, shipped)
        if have == 0:
            out["problems"].append(
                f"no shipped patch reverse-checks against this checkout and "
                f"it records no {PATCH_MANIFEST_NAME}, so nothing here is "
                "the patched engine")
            return out
        out["adopted"] = [p.name for p in shipped[:have]]
        out["verified"] = shipped[have - 1].name
        missing = [p.name for p in shipped[have:]]
        if missing:
            out["notes"].append(
                f"{', '.join(missing)} is not applied to this checkout, and "
                f"the checkout records no {PATCH_MANIFEST_NAME}, so this "
                "cannot tell a patch staged for the next build from one "
                "that was reverted. Re-run bin/setup_deps.sh to adopt and "
                "record the stack")
        return out

    out["manifest"] = True
    out["adopted"] = sorted(manifest)
    trustworthy = []
    for name in out["adopted"]:
        p = by_name.get(name)
        if p is None:
            out["problems"].append(
                f"{name} is in this engine's {PATCH_MANIFEST_NAME} and is no "
                f"longer shipped under {PATCH_DIR}, so nothing can say what "
                "this engine was built with")
            continue
        now = patch_digest(p)
        if now != manifest[name]:
            out["problems"].append(
                f"{name} has changed since this engine adopted it (recorded "
                f"{manifest[name][:12]}, the file is now {now[:12]}), so the "
                "engine and the diff that names it describe different code")
            continue
        trustworthy.append(name)
    # One reverse-check, on the top of the stack the engine CLAIMS rather
    # than the top of the directory. Only the topmost is meaningful:
    # later patches edit lines that are context in earlier ones, so once
    # the top patch is in the tree, reverse-checking 0001 may fail even
    # though 0001 is plainly there, and testing each independently reported
    # failures on a correct checkout. A clean reverse-check of the top is
    # evidence the whole prefix is present and unmodified.
    #
    # Skipped once anything above has already been flagged. With the top
    # diff out of the trustworthy list, this would land on the one BELOW
    # it, whose reverse-check fails on a perfectly correct tree by the
    # very rule above, and print a second problem naming the wrong diff.
    if trustworthy and len(trustworthy) == len(out["adopted"]):
        top = trustworthy[-1]
        if patch_applied(netherite_home, by_name[top]):
            out["verified"] = top
        else:
            out["problems"].append(
                f"{top} is the top of the patch stack this engine's "
                f"{PATCH_MANIFEST_NAME} claims and it does not reverse-check "
                "here, so either a patch it was built with has been reverted "
                "or the tree carries changes above it that nothing recorded")
    for p in shipped:
        if p.name in manifest:
            continue
        # "Not adopted" is a claim about the TREE, so it is tested rather
        # than assumed: a diff that reverse-checks clean is IN this
        # checkout while the manifest says it is not.
        #
        # Whether that is a failure depends on WHERE it sorts, and the
        # distinction is the whole staging rule rather than a nicety. A
        # diff numbered ABOVE the stack this engine claims is a staged
        # one, and it can reverse-check clean for an innocent reason: it
        # can be a re-export or a renumbering of hunks the engine already
        # carries. Failing on that would put back the defect this whole
        # rule exists to remove, so it is reported rather than failed. A
        # diff numbered INSIDE the claimed prefix
        # cannot have arrived that way, because everything above it is
        # already in the tree, so somebody applied it by hand and nothing
        # recorded whether the binary was built from it.
        in_tree = patch_applied(netherite_home, p)
        below_top = bool(out["adopted"]) and p.name < out["adopted"][-1]
        if in_tree and below_top:
            out["problems"].append(
                f"{p.name} is applied to this checkout, is not in its "
                f"{PATCH_MANIFEST_NAME}, and sorts inside the stack that "
                "manifest claims, so it was applied by hand and nothing "
                "recorded whether the binary was built from it")
            continue
        note = (f"{p.name} is shipped but this engine has not adopted it, "
                "which is not a failure: it changes nothing until an engine "
                "is built with it")
        if in_tree:
            note += (", though the checkout already carries its changes, so "
                     "it is either a re-export of a diff this engine has or "
                     "a hand apply nobody recorded")
        elif below_top:
            note += (f", and it sorts below {out['adopted'][-1]}, so "
                     "adopting it needs a rebuild rather than a git apply")
        out["notes"].append(note)
    return out


def engine_report(netherite_home=None):
    """Everything the provenance checks have to say about a checkout.

    {"problems": [...], "notes": [...]}. One verdict for preflight and
    for every env open, so a note only one of them printed cannot exist.
    `bedrock doctor` deliberately does not use it: it reports the
    revision and the patch stack as separate rows and assembles them
    from head_sha and patch_status itself.
    """
    root = engine_home(netherite_home)
    want = pinned_sha()
    if want is None:
        # An unreadable pin file used to make every checkout look fine,
        # which is the opposite of what this function is for.
        return {"problems": [f"{PINNED_SHA_FILE} is unreadable, so no "
                             "checkout can be verified against the pin"],
                "notes": []}
    if not (root / ".git").exists():
        return {"problems": [f"{root} is not a git checkout, so its revision "
                             "cannot be verified against the pin"],
                "notes": []}

    problems = []
    head = head_sha(netherite_home)
    if head is None:
        problems.append("git could not report HEAD here, so the revision "
                        "cannot be compared with the pin")
    elif head != want:
        problems.append(f"HEAD is {head[:12]}, the pin is {want[:12]}")
    status = patch_status(netherite_home)
    return {"problems": problems + status["problems"],
            "notes": status["notes"]}


def engine_problems(netherite_home=None):
    """Reasons this checkout is not the engine the repo pins, as strings.

    Empty means the checkout sits at the pinned SHA and still carries the
    patch stack it was built with. It does NOT mean the checkout carries
    every diff in patches/netherite/, and that difference is the point:
    see patch_status. engine_report has the notes this drops.
    """
    return engine_report(netherite_home)["problems"]


def check_engine(netherite_home=None):
    """Warn once per checkout when the engine is not the pinned one."""
    if os.environ.get("NETHERITE_ENGINE_CHECK", "1") == "0":
        return []
    root = str(engine_home(netherite_home))
    if root in _ENGINE_CHECKED:
        return []
    _ENGINE_CHECKED.add(root)
    report = engine_report(netherite_home)
    problems = report["problems"]
    if problems:
        print(f"warning: netherite checkout {root} is not the engine this "
              "repo pins.", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print("  results from it describe a different binary. Run "
              "bin/setup_deps.sh, or set NETHERITE_ENGINE_CHECK=0 to "
              "silence this.", file=sys.stderr)
    # Printed even on a clean engine, and deliberately not as a warning.
    # A staged patch is the state that used to kill runs, so it has to be
    # visible, and it has to be visible as the non-event it is.
    for n in report["notes"]:
        print(f"note: netherite checkout {root}: {n}", file=sys.stderr)
    return problems


def engine_argv(binary, seed, frame_w, frame_h, view_distance,
                mobs=False, snapshot_in=None, protocol="--rl-bin",
                world_time=None):
    """The engine command line every process in this repo is started with.

    ONE place builds it. A second hand-written argv is how a process ends
    up on a different framebuffer or a different view distance from the
    episodes it is supposed to be comparable with, which is silent: the
    GUI lays its slots out from the framebuffer size (gui scale =
    max(1, h/240)), so a driver at another size hit-tests different slots,
    and the renderer's chunk radius changes what the camera sees.

    The framebuffer size is declared even with rendering off, so every
    environment process hit-tests the same slots instead of falling back to
    the engine's 854x480 default.

    protocol selects the observation wire format: "--rl-bin" is the packed
    record MagmaEnv reads, "--rl" the JSON one, which carries fields the
    binary record deliberately does not (furnace burn and cook progress).
    Pair it with engine_env(): the command line alone does not turn the
    inventory GUI on.
    """
    argv = [str(binary), protocol, "--pace", "unlimited",
            "--seed", str(int(seed)),
            "--mobs", "on" if mobs else "off",
            "--width", str(int(frame_w)), "--height", str(int(frame_h)),
            "--view-distance", str(int(view_distance))]
    if snapshot_in:
        argv += ["--snapshot-in", str(snapshot_in)]
    # NETHERITE_WORLD_TIME pins the world clock at boot (patch 0002). A fresh
    # world starts at tick 0, which is sunrise, and sky light only reaches
    # full daylight at tick ~730, so an unset clock lights every short
    # episode at about 0.6 of daylight. It goes through THIS function so
    # data-generation and rollout processes cannot start on different clocks.
    # Unset changes nothing.
    wt = (world_time if world_time is not None
          else os.environ.get("NETHERITE_WORLD_TIME"))
    if wt is not None and str(wt) != "":
        argv += ["--set", f"set_time={int(wt)}"]
    # NETHERITE_RENDER_SET carries extra registry knobs for recordings, as a
    # comma list of key=value (no_hand=1, hide_gui=1). Nothing in RlBinObs
    # samples them; they change rendered pixels only.
    for kv in os.environ.get("NETHERITE_RENDER_SET", "").split(","):
        if kv.strip():
            argv += ["--set", kv.strip()]
    return argv


# ── the event journal ────────────────────────────────────────────────────
# The engine patch emits a typed record for each state transition on its
# own file descriptor, named by MAGMA_JOURNAL_FD. It is what makes
# "crafted a table" different from "picked one up", and what makes
# ordering expressible at all; bedrock_rl/env/journal.py decodes it and
# turns each event type into a task check.
#
# It is ON by default. Its measured cost is a few percent of engine CPU on
# a workload that produces events and negligible on one that does not,
# which is small against the cost of rendering. A default of
# off would mean every journal-backed task carried a flag that scores
# zero when someone forgets it. Set NETHERITE_JOURNAL=0, or
# `journal: false` in a task, to
# turn it off for a run that uses no journal check; with no fd set the
# engine-side samplers return on one predicate and cost nothing.
DEFAULT_JOURNAL = os.environ.get("NETHERITE_JOURNAL", "1") == "1"


def engine_env(lazy_frames=True, gpu=None):
    """The environment every engine process is started with.

    MAGMA_GUI_RL is on by default in EVERY process that runs an episode
    (live tool, oracle, data generation, evaluation), so the action space
    cannot silently differ between them; export MAGMA_GUI_RL=0 to get
    bit-identical upstream RL behaviour back.
    """
    env = dict(os.environ)
    # lazy capture skips the per-tick bookkeeping, so only snap ticks
    # reach the capture layer. A recording wants every tick instead.
    env["MAGMA_FRAME_LAZY"] = "1" if lazy_frames else "0"
    if gpu is not None:
        # Pin this environment process away from trainer/rollout devices.
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["MAGMA_GUI_RL"] = os.environ.get("MAGMA_GUI_RL", "1")
    return env


# ── how long the wrapper waits for the engine ────────────────────────────
# Every read of an engine's stdout used to be unbounded, which hangs the
# caller for ever with no traceback and no exit. The signature is a
# process at 0.0% CPU blocked in read(), a GPU at 0% with its memory
# still allocated, a log that is byte-identical for as long as anyone
# watches it, and fewer live `magma_game` children than the run holds.
#
# An engine that DIES was already caught: its stdout closes, the read
# returns b"" and the loop below raises. What was not caught is an engine
# that is ALIVE and silent, and there are ordinary ways to get there on a
# loaded machine. The evaluation driver holds every episode process
# open for the whole evaluation, at about 500 MiB of host memory each, so a
# 96-episode run next to a vLLM engine is a
# machine in reclaim, and a process stalled in the kernel writes nothing
# while staying perfectly alive. So does one whose pipe is held open by
# something that inherited it, and so does one stopped by a signal.
#
# The bound is on ONE ENGINE INTERACTION -- one tick, one boot -- and not
# on a turn or an episode, because the interaction is the measured operation:
#
#   a frameless tick, wall clock                            1.6 ms
#   one rastered tick (a snap), later in a process     110 to 140 ms
#   process boot at view_distance 8               0.52 s / 724 ms
#   the FIRST render in a process, view distance 8    2.88 s / 3757 ms
#   a whole 12-turn episode                                 4.94 s
#
# The two-column rows are two machines; the larger is the A100. 120 s is
# 32x the largest single interaction ever measured here, the first
# chunk-mesh build, and 24x a whole twelve-turn episode. Nothing healthy
# has been within an order of magnitude of it, and it turns an unbounded
# hang into a two-minute error. Every figure above is a row of one of
# those two pages, deliberately: a bound justified by a number nobody can
# recompute is a bound nobody can argue with.
#
# It is a deadlock breaker and not a performance SLA. On a machine that
# pages, raise it; do not lower it to catch slow episodes, because a
# rollout killed for being slow is a rollout missing from a measurement
# that still prints a percentage.
DEFAULT_ENGINE_TIMEOUT = float(
    os.environ.get("NETHERITE_ENGINE_TIMEOUT", "120"))

# How often the wait below looks at the CHILD rather than at the pipe. A
# dead child usually closes its stdout and the read ends by itself; this
# is for when it does not, which is any process that inherited the write
# end holding it open, and then nothing would arrive for the whole
# deadline. Half a second costs two syscalls per second on a blocked env
# and one extra select() per tick on a busy one, against a 1.6 ms tick.
LIVENESS_POLL = 0.5

# How much non-record noise the resynchroniser will step over before it
# gives up. The loop tolerates stray text ahead of a record and used to
# tolerate an unbounded amount of it, so an engine printing to stdout
# forever was resynchronised forever, at full CPU.
#
# The deadline would end that too, eventually, because the wait below
# checks it before every byte. What the cap buys is the DIAGNOSIS: a
# stream that is not this protocol at all reads as "wrong binary, wrong
# flag" in a megabyte rather than as "your engine was slow" in two
# minutes, and those send a reader to different places. It is also the
# only bound on the descriptor-less fallback. A MiB is 71 records' worth,
# far more than any preamble and still a bound.
MAX_PRE_RECORD_JUNK = 1 << 20

# How long close() waits for an engine to go on its own after its stdin
# closes, before killing it. Named rather than inline so a caller tearing
# down hundreds of envs can shorten it in one place.
CLOSE_GRACE = 5.0


class EngineError(RuntimeError):
    """The engine could not be talked to."""


class EngineGone(EngineError):
    """The process is not there any more, or will not take another byte."""


class EngineTimeout(EngineError):
    """The process is there and said nothing inside the deadline."""


def proc_status(proc):
    """How a child ended, in words, or None while it is still running.

    The signal matters more than the number here. An engine the OOM killer
    took reads `killed by signal 9 (SIGKILL)` and an engine that aborted
    reads `killed by signal 6 (SIGABRT)`, and those are different bugs
    with different fixes; `died` on its own, which is all this used to
    say, is neither.
    """
    if proc is None:
        return None
    rc = proc.poll()
    if rc is None:
        return None
    if rc >= 0:
        return f"exited {rc}"
    try:
        name = _signal.Signals(-rc).name
    except ValueError:                 # a signal number python has no name for
        return f"killed by signal {-rc}"
    return f"killed by signal {-rc} ({name})"


def _readable_poller(fd):
    """A poller armed on one descriptor.

    poll() and NOT select(). select is built on `fd_set`, which is a
    bitmap of FD_SETSIZE bits, 1024 on both Linux and macOS, and python
    raises `ValueError: filedescriptor out of range in select()` for any
    descriptor at or above it. That is reachable here rather than
    theoretical: a MagmaEnv holds three descriptors (stdin, stdout, the
    journal pipe), a held-out evaluation holds one env per episode, and
    the same process is also holding a vLLM engine's own descriptors, so
    a few hundred episodes puts these pipes above 1024. The ValueError is
    not an EngineError, so nothing that handles a failed env would catch
    it, and it would break envs that were working perfectly. poll has no
    such ceiling.
    """
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    return poller


def _label(what):
    """The name of an engine, built only when something has to say it.

    MagmaEnv passes its `describe` METHOD and not the string it returns.
    The string interpolates five fields and is read only by a message
    that a healthy tick never prints, so building it eagerly charged
    every tick for a diagnostic almost none of them need. A plain string
    still works, which is
    what tests and direct protocol callers pass.
    """
    return what() if callable(what) else what


def _wait_readable(poller, proc, deadline, timeout, what):
    """Block until the descriptor has bytes to read, or raise.

    Two things end the wait besides data, and they are the two an
    unbounded read could not tell apart from a slow tick: the deadline,
    and the child no longer being there to answer.
    """
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            raise EngineTimeout(
                f"{_label(what)} has not answered in {timeout:g}s and is "
                f"{proc_status(proc) or 'still running'}. The read is "
                "bounded because an engine that is alive and silent used to "
                "block this process for ever with no traceback and no exit; "
                "raise NETHERITE_ENGINE_TIMEOUT if this machine is genuinely "
                "that slow.")
        # milliseconds, and any event counts: a write end that closed
        # arrives as POLLHUP and the read below turns it into the death
        # it is, rather than into another wait.
        if poller.poll(min(left, LIVENESS_POLL) * 1000.0):
            return
        # Readability FIRST, then death, and then readability again. A
        # child can write a whole record and exit in the same breath, and
        # those bytes are still in the pipe and still owed to the caller.
        status = proc_status(proc)
        if status is not None:
            if poller.poll(0):
                return
            raise EngineGone(
                f"{_label(what)} {status} with no record on the wire")


def read_record(stream, what="the engine", proc=None, timeout=None):
    """One RlBinObs record off a raw engine stdout, or an EngineError.

    The framing is resynchronisation on a four-byte magic followed by
    exactly BIN_SIZE bytes, and it is here rather than in MagmaEnv
    because the benchmarks and the wire-format check drive their own
    Popen and had each written the same loop out again. Three readers of
    a frozen binary layout is three places to get the length wrong.

    `timeout` bounds the WHOLE record, not each chunk, because the engine
    writes a record with one write() after the tick it describes: the
    bytes arrive in a burst or they do not arrive. A trickle that reset a
    per-chunk deadline on every byte would be unbounded again. `proc` is
    the child, and passing it is what makes a death visible before the
    deadline instead of after it -- see _wait_readable.

    Reads go to the file descriptor rather than through `stream`, so the
    same wait can see them arrive. That is sound because read_record is
    the ONLY reader of an engine's stdout in this repo (MagmaEnv._read_obs),
    so no buffered object is holding bytes
    this would skip. A stream with no descriptor -- a BytesIO in a test --
    falls back to plain blocking reads and gets no deadline, which is
    stated here because a silent fallback to the behaviour this function
    exists to remove would be worse than not having it.
    """
    timeout = DEFAULT_ENGINE_TIMEOUT if timeout is None else float(timeout)
    try:
        fd = stream.fileno()
    except (AttributeError, OSError, ValueError):
        fd = None
    deadline = time.monotonic() + timeout
    poller = None if fd is None else _readable_poller(fd)

    def exact(n):
        buf = b""
        while len(buf) < n:
            if poller is None:
                chunk = stream.read(n - len(buf))
            else:
                _wait_readable(poller, proc, deadline, timeout, what)
                try:
                    chunk = os.read(fd, n - len(buf))
                except OSError as e:
                    # poll() reports POLLERR and POLLNVAL as readable, so
                    # a descriptor that has been closed underneath this
                    # comes back ready and then fails the read. Left bare
                    # it is an OSError, which is not an EngineError, so
                    # callers expecting EngineError would not catch it.
                    raise EngineGone(
                        f"{_label(what)} could not be read ({e}) and it "
                        f"{proc_status(proc) or 'is still running'}") from e
            if not chunk:
                how = (proc_status(proc)
                       or "closed its stdout while still running")
                raise EngineGone(
                    f"{_label(what)} died mid-record and it {how}; argv "
                    "includes --rl-bin? binary built?")
            buf += chunk
        return buf

    win = exact(4)
    junk = 0
    while win != BIN_MAGIC:            # tolerate stray text pre-record
        win = win[1:] + exact(1)
        junk += 1
        if junk > MAX_PRE_RECORD_JUNK:
            raise EngineError(
                # junk + 4: the four bytes of the first window were read
                # before the count started and are just as much not a
                # record as the rest of them
                f"{_label(what)} wrote {junk + 4} bytes with no "
                f"{BIN_MAGIC!r} record header in them, so this is not the "
                "--rl-bin protocol at all. Built the right binary, and is "
                "--rl-bin on the argv? The deadline would end this too, "
                "eventually, but 'that stream is not this protocol' and "
                "'your engine was slow' send a reader to different "
                "places.")
    return win + exact(BIN_SIZE - 4)


def parse_obs(rec):
    """One RlBinObs wire record -> the observation dict.

    A module function so protocol parsing can be tested independently of
    process management.
    """
    t, x, y, z, yaw, pitch, dead = struct.unpack_from(HEAD_FMT, rec, 4)
    o = 4 + struct.calcsize(HEAD_FMT)
    ints = struct.unpack_from(f"<{OBS_INT_COUNT}i", rec, o)
    o = BIN_HEAD
    blocks = struct.unpack_from(f"<{N_BLOCKS * 4}i", rec, o)
    o += N_BLOCKS * 4 * 4
    logs = struct.unpack_from(f"<{N_LOGS * 3}i", rec, o)
    o += N_LOGS * 3 * 4
    coal = struct.unpack_from(f"<{N_COAL * 3}i", rec, o)
    o += N_COAL * 3 * 4
    cam = memoryview(rec)[o:o + NPIX * 2].cast("H")
    o += NPIX * 2
    depth = memoryview(rec)[o:o + NPIX]
    o += NPIX
    edge = memoryview(rec)[o:o + NPIX]
    return {
        "tick": t, "x": x, "y": y, "z": z, "yaw": yaw, "pitch": pitch,
        "dead": dead,
        "hotbar_ids": ints[0:9], "hotbar_counts": ints[9:18],
        "hotbar_meta": ints[18:27],
        "inventory_ids": ints[0:9] + ints[27:54],
        "inventory_counts": ints[9:18] + ints[54:81],
        "inventory_meta": ints[18:27] + ints[81:108],
        "hotbar_sel": ints[108], "container": ints[109],
        "inv_counts": dict(zip(INV_ORDER, ints[110:119])),
        "blocks": [blocks[i:i + 4] for i in range(0, N_BLOCKS * 4, 4)],
        "logs": [logs[i:i + 3] for i in range(0, N_LOGS * 3, 3)],
        "coal": [coal[i:i + 3] for i in range(0, N_COAL * 3, 3)],
        "cam": cam, "depth": depth, "edge": edge,
    }


# Where the player's eyes are above their feet, which is what every
# "how far away is that block" answer in this repo is measured from. It
# was a bare 1.62 in five places, three of which wrote the whole
# eye-to-block-centre expression out again.
#
# It is the STANDING eye height and the engine has three:
# psv_player_eye_height returns 1.62 - 0.08 while sneaking and 0.4 in
# elytra flight (blaze/core/player_survival.h). Neither is reachable from
# the model's action vocabulary, which carries no sneak key and no
# elytra, so a single number is correct here rather than merely
# convenient. That correctness is conditional on the vocabulary: add a
# sneak key to bedrock_rl/adapters/netherite/computer.py and this constant is wrong by
# 0.08 for every action taken while it is held, which moves every reach
# and eye-delta computation below it.
EYE_HEIGHT = 1.62

# The survival break/place ray length in blaze/core/player_survival.h.
BREAK_REACH = 5.0


def eye_delta(obs, wx, wy, wz):
    """(dx, dy, dz) from the player's EYE to a block cell's centre."""
    return (wx + 0.5 - obs["x"],
            wy + 0.5 - (obs["y"] + EYE_HEIGHT),
            wz + 0.5 - obs["z"])


def eye_dist2(obs, wx, wy, wz):
    """Squared eye-to-centre distance. Squared because the callers that
    only rank candidates should not pay for a square root per block."""
    dx, dy, dz = eye_delta(obs, wx, wy, wz)
    return dx * dx + dy * dy + dz * dz


def look_angles(obs, wx, wy, wz):
    """(dyaw, dpitch, dist) from the current view to a block centre.

    A module function on the obs dict, not only a method, because the
    demos drive the engine over a raw pipe and had written this out
    again against a dict they parsed themselves.
    """
    dx, dy, dz = eye_delta(obs, wx, wy, wz)
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    tgt_yaw = math.degrees(math.atan2(-dx, dz))
    tgt_pitch = -math.degrees(math.asin(dy / max(dist, 1e-9)))
    return wrap180(tgt_yaw - obs["yaw"]), tgt_pitch - obs["pitch"], dist


def miss_angle_deg(obs, wx, wy, wz):
    """ONE angle, in degrees, between where the view points and a block
    cell's centre. 0 is dead on it and 180 is behind the player.

    `look_angles` answers the same question in two numbers, and two
    numbers are not an angle: a target 10 degrees right and 10 down is
    14.1 degrees away and not 20, and near the poles the yaw component
    shrinks to nothing while the ray does not move. Anything that has to
    RANK how close two poses are needs the single angle, which is why
    this is here rather than a caller taking a hypotenuse of the pair.

    Derived from `look_angles`, not from a second raycast, so the two
    cannot disagree about which direction the cell is in. With p1 the
    view's pitch and p2 the cell's, the spherical law of cosines on the
    engine's own convention (dy = -sin pitch, azimuth difference dyaw)
    is cos t = cos p1 cos p2 cos dyaw + sin p1 sin p2.

    It is measured from the LOOK AXIS, which the semantic camera's
    crosshair pixel misses by about 1.11 degrees on
    each axis. That is a real 1.6-degree bias against `crosshair()` and
    it is left in: no sample is on the axis (see CROSSHAIR_PX), so the
    alternative is a second opinion about where the player is looking,
    and a shaping potential is allowed a bias a CHECK would not be.
    """
    dyaw, dpitch, _ = look_angles(obs, wx, wy, wz)
    p1 = math.radians(obs["pitch"])
    p2 = math.radians(obs["pitch"] + dpitch)
    c = (math.cos(p1) * math.cos(p2) * math.cos(math.radians(dyaw))
         + math.sin(p1) * math.sin(p2))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def hotbar_count(obs, item_ids):
    """How many of `item_ids` the nine hotbar slots hold, in total.

    The observation record carries the hotbar and nine inventory
    counters and nothing else, so for anything outside those counters
    this sum IS the inventory. The task's success check and the solver's
    progress check both ask it, and they must not answer differently:
    the solver stops when it believes the count is reached and the check
    decides whether it was.
    """
    ids = set(item_ids)
    return sum(n for i, n in zip(obs["hotbar_ids"], obs["hotbar_counts"])
               if i in ids)

# ── the crosshair pixel ──────────────────────────────────────────────────
# The camera is 64x36 and oc_pixel samples pixel CENTRES, so with both
# dimensions even the optical axis falls on the corner where four pixels
# meet and no pixel is on it. This is the one, and the offset below is
# what that costs.
#
# It is not a choice that can be made better. All four candidates are
# HALF A PIXEL from the axis in each direction, measured at 1.1143
# degrees, and the pixel is 1.601 by 1.944 degrees. Measured again
# through the exact inverse of the engine ray (env/projection.py) the
# two axes come apart in the fourth decimal, 1.114274 in yaw against
# 1.114064 in pitch, because the engine normalizes the whole ray rather
# than the two axes separately. Nothing above changes; the pair is below
# because a single number for both is a claim the engine does not make. So the direction the
# model aims, which is the frame centre and is exactly what
# adapters/netherite/computer.py::coord_to_angles(500, 500) returns as (0, 0), and the ray
# the reward reads here are different rays by construction.
#
# The lateral difference between the optical axis and sampled ray grows with
# distance. Moving the sample would only select a different equally offset
# center pixel while breaking the exact match described below.
# The engine journals its `aimed_at` events from this same index
# (rl_mode.c samples cam[(RL_CAM_H/2)*RL_CAM_W + RL_CAM_W/2] right before
# it writes the record), so the instantaneous crosshair check and the
# "was ever aimed at" check cannot disagree with each other. A fix is an
# engine change, an odd camera dimension or an axis-aligned sample, and
# it would change the observation contract.
CROSSHAIR_PX = (CAM_W // 2, CAM_H // 2)          # (32, 18)
CROSSHAIR_I = CROSSHAIR_PX[1] * CAM_W + CROSSHAIR_PX[0]


class ObsHelpers:
    """The read-only questions anyone asks of an env.

    Every one of them is a pure function of `self.obs`, so they belong to
    the observation surface rather than process-management code.
    """

    def crosshair(self):
        c = CROSSHAIR_I
        bid = int(self.obs["cam"][c])
        d = int(self.obs["depth"][c])
        return {"id": bid, "dist": None if (bid == 0 or d >= 255)
                else d / DEPTH_SCALE}

    def visible_count(self, ids):
        ids = set(ids)
        return sum(1 for v in self.obs["cam"] if v in ids)

    def rel_angles_to(self, wx, wy, wz):
        """(dyaw, dpitch, dist) from current view to block center."""
        return look_angles(self.obs, wx, wy, wz)

    def miss_angle_to(self, wx, wy, wz):
        """How far off the view is from a block centre, in degrees."""
        return miss_angle_deg(self.obs, wx, wy, wz)

    def eye_dist2(self, wx, wy, wz):
        """Squared eye-to-centre distance to a block cell."""
        return eye_dist2(self.obs, wx, wy, wz)

    def nearest(self, kind):
        """Nearest log/coal position or None. kind: 'logs' | 'coal'."""
        for (wx, wy, wz) in self.obs[kind]:
            if wx == 0 and wy == 0 and wz == 0:
                continue
            return (wx, wy, wz)
        return None

    def act_batch(self, items, observer=None, stop_factory=None):
        """[(action_keys, n_ticks), ...] as ONE unit of work.

        Episode.step calls this because nothing reads observations between
        scheduled actions and the complete turn needs one shared deadline.

        The WHOLE BATCH is bounded as well as each tick in it, and the two
        are not the same bound. A per-tick deadline cannot see an engine
        that answers every tick just inside it: at 200 ticks a turn, an
        engine returning a record one second before each deadline makes
        forward progress for six hours, and from the inside that is
        indistinguishable from a slow machine. A turn therefore gets one
        budget for all of its ticks, at the same setting, which is 150x
        the roughly 0.77 s that 200 simulation ticks take.

        The optional observer spends no tick.  A per-item stop callback may
        release that one scheduled hold early after a live transition (for
        example, when its exact mining target breaks); later schedule items
        still run.  Otherwise the batch finishes or raises loudly.
        """
        budget = getattr(self, "timeout", None)
        deadline = None if not budget else time.monotonic() + float(budget)
        for keys, n in items:
            # The callback is deliberately created immediately before THIS
            # schedule item. A preceding mouse pan can change the exact block
            # an attack will hit, so a turn-start target would describe the
            # stale ray. Releasing one hold must not skip later schedule items
            # such as the normal settle tick.
            stop = None if stop_factory is None else stop_factory(self, keys)
            for _ in range(int(n)):
                self.act(**keys)
                if observer is not None:
                    observer(self)
                if stop is not None and stop(self):
                    break
                if deadline is not None and time.monotonic() > deadline:
                    raise EngineTimeout(
                        f"{self.describe()} answered every tick but took "
                        f"longer than {float(budget):g}s over one turn, so "
                        "the turn is bounded as well as the tick: an engine "
                        "that keeps answering just inside the per-tick "
                        "deadline never trips it and never finishes either")
        return self.obs


class MagmaEnv(ObsHelpers):
    """One upstream Netherite game process running in RL mode."""

    def __init__(self, seed=0, netherite_home=None, mobs=False,
                 frames_dir=None, frame_w=854, frame_h=480,
                 snapshot_in=None, raster=None, gpu=None,
                 view_distance=None, capture_every=None, journal=None,
                 timeout=None, world_time=None):
        # raster: "cpu" | "cuda" (default from NETHERITE_RASTER). gpu is the
        # device list visible to a CUDA raster process and is ignored on CPU.
        self.netherite_home = netherite_home
        self.raster = (raster or DEFAULT_RASTER).lower()
        # capture_every=N renders and writes a PPM every N ticks to
        # frames_dir instead of only on demand. Off by default and it must
        # stay off for training and rollout, where render-on-demand is what
        # makes episodes affordable. It exists for recordings, where one
        # frame per turn is a slideshow of decision points and every step
        # between them exists in the simulation and is simply never drawn.
        # Frames are left on disk for the caller, so frame() does not
        # unlink under this setting whatever `keep` says.
        self.capture_every = None if not capture_every else int(capture_every)
        self.view_distance = int(view_distance or DEFAULT_VIEW_DISTANCE)
        if not 1 <= self.view_distance <= 8:
            raise ValueError("view_distance must be in 1..8, got "
                             f"{self.view_distance}")
        self.gpu = gpu if gpu is not None else os.environ.get(
            "NETHERITE_ENV_GPUS")
        # How long any one interaction with this process may take before
        # it is called a failure rather than a slow tick. Per env rather
        # than global so a caller that knows its box can say so; the
        # default and the measurements behind it are above.
        self.timeout = (DEFAULT_ENGINE_TIMEOUT if timeout is None
                        else float(timeout))
        # Set the first time this process fails to answer and never cleared.
        self.failed = False
        check_engine(netherite_home)
        self.binary = find_binary(netherite_home, self.raster)
        self.seed = int(seed)
        self.mobs = mobs
        self.world_time = world_time
        self.snapshot_in = snapshot_in
        self.frames_dir = None if frames_dir is None else Path(frames_dir)
        self.frame_w, self.frame_h = frame_w, frame_h
        # Event journal for this process-backed episode.
        self.journal_on = (DEFAULT_JOURNAL if journal is None
                           else bool(journal))
        self.journal = []
        self._jfd = None
        self._jreader = None
        self._frames_path = None
        # engine index of the last capture file served, -1 for none. It is
        # a HIGH-WATER MARK and not a counter: the engine decides what the
        # next index is, and frame() reads that decision off the directory.
        self._last_frame = -1
        self._spawn_n = 0
        self.proc = None
        self.obs = None
        # Initial-state snapshots are Overworld-only. Cross-dimension changes
        # arrive as typed journal events without changing the byte-frozen BOLR
        # observation record.
        self._dimension = 0
        self.ticks = 0
        self._spawn()

    def _spawn(self):
        argv = engine_argv(self.binary, self.seed, self.frame_w,
                           self.frame_h, self.view_distance, mobs=self.mobs,
                           snapshot_in=self.snapshot_in,
                           world_time=self.world_time)
        if self.frames_dir is None:
            argv += ["--render", "off"]
        else:
            # fresh subdir per spawn: a respawn restarts frame numbering at
            # 0 and stale PPMs from the previous process must not be served
            global _CAPTURE_SERIAL
            _CAPTURE_SERIAL += 1
            self._spawn_n = _CAPTURE_SERIAL
            self._frames_path = (self.frames_dir.resolve()
                                 / f"seed{self.seed}_{os.getpid()}"
                                   f"_{self._spawn_n}")
            self._frames_path.mkdir(parents=True, exist_ok=True)
            argv += ["--frames-out", str(self._frames_path),
                     # render ONLY on demand: the "snap":1 action key (an
                     # engine patch) forces a one-tick render; periodic
                     # rendering off. capture_every overrides that for
                     # recordings only.
                     "--frame-every", str(self.capture_every or 1000000000),
                     # rendering dense foliage can overflow the default
                     # draw_cutout cap and abort() mid-episode (the semantic
                     # camera never touches draw buffers); 4x headroom
                     "--set", "draw_cutout=1048576"]
            # GPU rasterizer only matters when a view is actually rendered;
            # a frameless env never opens a CUDA context even on the cuda
            # binary, so the flag is scoped to the frames-out branch.
            if self.raster == "cuda":
                # no_defer=1 is REQUIRED here, not a tuning knob. The CUDA
                # capture path normally arms a depth-1 deferred readback:
                # frame N's PPM is written on capture call N+1
                # (game/frame_capture.c finish_pending). frame() below polls
                # for the CURRENT index, so it would time out on every call.
                # An open GUI screen already forces the sync path
                # (force_sync in frame_capture.c), so this only costs the
                # non-GUI frames a few milliseconds each.
                argv += ["--backend", "cuda", "--set", "no_defer=1"]
        # a respawn gets a fresh subdir, so the high-water mark starts over
        self._last_frame = -1
        env = engine_env(
            lazy_frames=not self.capture_every,
            gpu=self.gpu if self.raster == "cuda" else None)
        # The journal gets its OWN pipe. It cannot share stdout: the
        # observation record is byte-frozen, and _read_obs resynchronises
        # on a four-byte magic, so a journal record that happened to
        # contain those bytes would be parsed as an observation.
        pass_fds = ()
        self._close_journal()
        if self.journal_on:
            r_fd, w_fd = os.pipe()
            env["MAGMA_JOURNAL_FD"] = str(w_fd)
            pass_fds = (w_fd,)
        else:
            env.pop("MAGMA_JOURNAL_FD", None)
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=env, pass_fds=pass_fds)
        if self.journal_on:
            os.close(w_fd)                 # the child owns the write end
            os.set_blocking(r_fd, False)
            self._jfd = r_fd
            self._jreader = _journal.JournalReader()
        self.journal = []
        self.obs = self._read_obs()
        self.ticks = 0

    def describe(self):
        """This process, in enough detail to find the episode it is.

        The failure this names is one env out of the dozens a held-out
        evaluation holds open at once, so `magma_game (seed=14)`, which is
        all it used to say, does not narrow it down: the snapshot path
        carries the task, the pid is what `ps` and `kill` take, and the
        tick is how far in it got.
        """
        where = (f" snapshot={self.snapshot_in}" if self.snapshot_in
                 else " (no snapshot)")
        pid = getattr(self.proc, "pid", None)
        return (f"magma_game seed={self.seed} pid={pid} "
                f"vd={self.view_distance} tick={self.ticks}{where}")

    def _send(self, line):
        """One protocol line to the engine, or a named EngineGone.

        A dead child turns a bare write into a BrokenPipeError from inside
        whichever of act or snapshot the caller happened to be
        in, naming neither the episode nor how the engine died. It spends
        no tick and reads nothing, so nothing about tick accounting passes
        through here.
        """
        if self.proc is None:
            self.failed = True
            raise EngineGone(f"{self.describe()} has already been closed, so "
                             "there is nothing to send this action to")
        try:
            self.proc.stdin.write(line)
            self.proc.stdin.flush()
        except (OSError, ValueError) as e:
            self.failed = True
            how = proc_status(self.proc) or "is still running"
            raise EngineGone(
                f"{self.describe()} would not take an action ({e}) and it "
                f"{how}") from e

    def _read_obs(self):
        try:
            # `self.describe` and not `self.describe()`: the label is
            # only ever read by a failure, and this is the per-tick path.
            rec = read_record(self.proc.stdout, self.describe,
                              proc=self.proc, timeout=self.timeout)
        except EngineError:
            # Marked here rather than by each caller: a wedged process must
            # never be treated as healthy after any read path fails.
            self.failed = True
            raise
        # Drain the journal AFTER the observation, never before. The
        # engine writes this tick's records and then the record above, so
        # having the observation is proof that the events are already in
        # the pipe. Draining first would race the engine for no reason.
        if self._jfd is not None:
            events = _journal.drain(self._jfd, self._jreader)
            self.journal.extend(events)
            for event in events:
                if event.name == "dimension_changed":
                    self._dimension = int(event["dimension"])
            # Anything the decoder could not turn into an event. Both
            # ways of losing one used to be silent, and both make a
            # journal-backed check answer False for a reason that has
            # nothing to do with the policy. Reported as a `[journal]`
            # diagnostic, which the verl console filter never drops.
            _journal.report_problems(self._jreader)
        obs = parse_obs(rec)
        obs["dimension"] = self._dimension
        return obs

    def act(self, **keys):
        """One tick. The full key list is magma/game/rl_mode.c.

        All keys are optional and default 0, EXCEPT hotbar and craft,
        which default to -1 meaning no change, so "hotbar":0 actively
        selects slot 0.

          forward, strafe, dyaw, dpitch, jump, sneak, sprint, attack,
          use, hotbar (0..8), craft (0..7), interact (0/1), smelt (0/1)

        plus the MAGMA_GUI_RL inventory-screen keys, on by default here
        (see _spawn): gui_open (0/1), gui_cursor_x and gui_cursor_y in
        framebuffer pixels, gui_click (1 left, 2 right).

        craft, interact and smelt are engine-internal and are not part of
        the model-facing action space. A right click aimed at a placed
        crafting table or furnace opens that container's screen, so obs
        "container" reads 0 player, 1 table, 2 furnace.

        dyaw and dpitch are DELTA DEGREES applied this tick. Positive
        pitch looks DOWN, and straight up is -90.
        """
        self._send((json.dumps(keys) + "\n").encode())
        self.obs = self._read_obs()
        self.ticks += 1
        return self.obs

    def snapshot(self, path, radius=32):
        """Export a .bsnp start-state (action-key protocol).

        It costs an engine TICK, which is counted here because the engine
        charges it either way. Measured, not assumed: the record that
        comes back carries a tick one higher than the one before it. The
        rule every method here follows is that a path spending an engine
        tick moves `self.ticks` in the same statement. This one used not
        to. A snapshot of a live episode would otherwise run on a budget
        one tick longer than the task
        set — an off-by-one that shows up only as an episode that solves
        just inside its budget where an identical one does not.
        """
        self._send((json.dumps({"snapshot": str(path), "snapshot_r": radius})
                    + "\n").encode())
        self.obs = self._read_obs()
        self.ticks += 1

    # ── pose / crosshair helpers ────────────────────────────────────────
    def reset_pose(self, yaw, pitch):
        """Scramble look in place (one unrewarded tick)."""
        cur = self.obs
        self.act(dyaw=wrap180(yaw - cur["yaw"]),
                 dpitch=float(pitch) - cur["pitch"])
        self.ticks = 0
        return self.obs

    # ── VL frame ────────────────────────────────────────────────────────
    def frame(self, fmt="PNG", keep=False, snap=True):
        """Rendered view as encoded image bytes, or None on a frameless
        process.

        Spends ONE tick with "snap":1 so that only frames actually
        consumed get rastered (~140 ms each). That tick is spent WHETHER
        OR NOT this process renders, and the early return for a frameless
        one comes after it. This keeps tick accounting independent of whether
        a harness requested pixels. A snap tick renders and mutates nothing,
        so paying it on the frameless path costs ~1.6 ms and buys that
        invariant.

        The obs record can arrive before the PPM is fully written; wait
        for the exact expected size.

        keep leaves the PPM on disk. It is forced on under capture_every,
        which exists to leave a recording behind, and unlinking the frames
        this method happens to read punched a hole in that recording once
        per turn.
        """
        if snap:
            self.act(snap=1)
        if self._frames_path is None:
            return None
        p = self._await_capture(new=bool(snap))
        if p is None:
            return None
        from PIL import Image
        with Image.open(p) as im:
            im.load()
            buf = io.BytesIO()
            im.save(buf, format=fmt)
        if not keep and not self.capture_every:
            try:
                p.unlink()
            except OSError:
                pass
        return buf.getvalue()

    def _await_capture(self, new):
        """Return the newest complete capture path, if one exists.

        Discover files from the capture directory because engine indices count
        actual capture writes: lazy mode advances on snap ticks, while
        capture-every mode advances each tick. A wrapper-side predicted counter
        would therefore drift between modes.

        new=True means a snap tick was just spent, so a file with a
        HIGHER index than the last one served is on its way and this
        waits for it. new=False asks for whatever the last capture was,
        and answers None when there has not been one.
        """
        header = f"P6\n{self.frame_w} {self.frame_h}\n255\n"
        want = len(header) + self.frame_w * self.frame_h * 3
        # The SAME bound as a record read, and not a second constant. This
        # used to be a hard-coded 20 s, which made it the tightest wait in
        # the class and one that NETHERITE_ENGINE_TIMEOUT did not reach:
        # on a machine slow enough to want the knob raised, raising it
        # moved every wait except this one. It is still the shortest wait
        # in practice, because the snap tick's record has already come
        # back by the time this is called, so the render is done and only
        # the PPM write is outstanding.
        deadline = time.monotonic() + self.timeout
        floor = self._last_frame if new else -1
        while True:
            for i in self._capture_indices(above=floor):
                p = self._frames_path / f"frame_{i:06d}.ppm"
                try:
                    if p.stat().st_size >= want:
                        self._last_frame = max(self._last_frame, i)
                        return p
                except OSError:
                    continue
            if not new:
                return None
            if time.monotonic() > deadline:
                self.failed = True
                raise EngineTimeout(
                    f"{self.describe()}: no capture file above index {floor} "
                    f"completed in {self._frames_path} within "
                    f"{self.timeout:g}s; the engine was asked to render a "
                    "snap tick and wrote nothing")
            time.sleep(0.002)

    def _capture_indices(self, above=-1):
        """Capture indices present, newest first.

        Newest first because that is the one wanted, and the loop above
        falls through to older ones only if the newest is still being
        written.
        """
        out = []
        try:
            entries = os.scandir(self._frames_path)
        except OSError:
            return out
        with entries:
            for e in entries:
                m = FRAME_FILE_RE.match(e.name)
                if m is None:
                    continue
                i = int(m.group(1))
                if i > above:
                    out.append(i)
        out.sort(reverse=True)
        return out

    def _close_journal(self):
        if self._jfd is not None:
            try:
                os.close(self._jfd)
            except OSError:
                pass
        self._jfd = None
        self._jreader = None

    def close(self):
        if self.proc is None:
            self._close_journal()
            return
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=CLOSE_GRACE)
        except Exception:
            self.proc.kill()
            # and REAP it. A killed child that nobody waits on stays a
            # zombie for as long as this process lives, and counting live
            # magma_game processes with `ps` is how both of the hangs this
            # module now bounds were diagnosed: a run that leaves 96
            # defunct entries behind makes that count a lie, and an
            # evaluation holding one env per episode leaves 96 of them.
            try:
                self.proc.wait(timeout=CLOSE_GRACE)
            except Exception:
                pass
        finally:
            self.proc = None
            # after the process is gone, so the engine never writes into
            # a closed pipe and takes a SIGPIPE for it
            self._close_journal()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def wrap180(a):
    return (float(a) + 180.0) % 360.0 - 180.0
