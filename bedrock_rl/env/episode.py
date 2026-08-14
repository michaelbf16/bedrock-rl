"""One deterministic Netherite episode shared by every harness and verifier.

`Episode` is the centre of the Netherite adapter. The verl tool, synthetic
generation, oracles, and demos all step it directly. Reward is read from
that same live instance rather than reconstructed from model text.

It has nothing to do with verl, so this module imports no trainer and
works in the plain CPU venv where verl is absent.

An episode is (task, seed, initial pose) plus a parser that turns the
model's tool-call payload into an action schedule. A pose that already
has the crosshair on the block the task's success check names is not a
legal start, and `Episode.draw` is the one place that says so; every
producer of a fresh episode goes through it and `Episode.from_spec`,
which replays a pose somebody already recorded, deliberately does not.
`adapters.netherite.computer.cu_parser` is the shipped parser. `step` never
inspects it, so a different action
language is a different parser and nothing else. It hands the parser the
inventory-screen latch and the camera pitch, because both change what a
mouse action means.

Reward is `final_reward()`, which is shaping plus the success reward,
minus the per-action decision cost, minus 0.2 per malformed turn. A task
that opts into the give-up defences adds three more terms, all of them
declared in its YAML and resolved in bedrock_rl/env/task.py: a
terminal `fail:` check for the wrong answer, a `no_commit:` charge for
ending without answering, and `converge:`, dense potential-based shaping
that is summed per turn rather than latched.

Task YAML extras read here, all optional:
  max_turns           assistant turns per episode
  max_ticks_per_turn  tick budget per tool call
  raster              frame rasterizer, cpu or cuda, defaulting to cpu
  view                semantic, netherite-procedural, minecraft-official,
                      or a custom import-path component
"""
import io
import os
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

from bedrock_rl.env.engine import (DEFAULT_RASTER, DEFAULT_VIEW_DISTANCE,
                                     EngineError, MagmaEnv)
from bedrock_rl.env.actions import ProgramError
from bedrock_rl.env.observation import render_obs
from bedrock_rl.env.views import capture_png, open_view

# half-res: 4x cheaper raster, plenty for VL. This is the size the POLICY
# sees and it is the default for everything. NETHERITE_FRAME_SCALE raises all
# frames in that process by an INTEGER factor, which is useful for a separate
# high-resolution recording run. The container panel is laid out at GUI scale
# max(1, h/240), so only multiples of 240 keep slot geometry self-consistent;
# gui_layout derives its table from the same dimensions. Leave it unset for
# the standard 428x240 training view.
_FRAME_SCALE = max(1, int(os.environ.get("NETHERITE_FRAME_SCALE", "1")))
FRAME_W, FRAME_H = 428 * _FRAME_SCALE, 240 * _FRAME_SCALE
# The normalized screen the model names points on, in both directions:
# adapters/netherite/computer.py maps a coordinate onto the framebuffer and
# env/gui_layout.py maps a slot centre back. Two files stating the size
# of one coordinate space is two chances for the round trip to stop
# being a round trip.
SCREEN = 1000.0
_FRAME_OWNER = ".bedrock-frame-owner"
_REAPED_FRAME_ROOTS = set()
_OWN_PROCESS_IDENTITY = None


def _process_alive(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


def _process_identities():
    """Best-effort PID -> process-start identity for PID-reuse detection."""
    try:
        output = subprocess.check_output(
            ["ps", "-axo", "pid=,lstart="], text=True,
            stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    identities = {}
    for line in output.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            identities[int(parts[0])] = parts[1]
        except ValueError:
            continue
    return identities


def _own_process_identity():
    global _OWN_PROCESS_IDENTITY
    pid = os.getpid()
    if (_OWN_PROCESS_IDENTITY is None
            or _OWN_PROCESS_IDENTITY[0] != pid):
        identities = _process_identities()
        _OWN_PROCESS_IDENTITY = (
            pid, "" if identities is None else identities.get(pid, ""))
    return _OWN_PROCESS_IDENTITY[1]


def _owned_frames_dir(root):
    """Create one process-owned frame dir and reap dead-process leftovers."""
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    key = os.fspath(root)
    should_reap = key not in _REAPED_FRAME_ROOTS
    _REAPED_FRAME_ROOTS.add(key)
    if should_reap:
        identities = None
        for candidate in root.iterdir():
            marker = candidate / _FRAME_OWNER
            if not candidate.is_dir() or not marker.is_file():
                continue
            try:
                owner_line, _, recorded_identity = (
                    marker.read_text().partition("\n"))
                owner = int(owner_line.strip())
            except (OSError, ValueError):
                continue
            stale = not _process_alive(owner)
            if not stale and recorded_identity:
                if identities is None:
                    identities = _process_identities()
                if (identities is not None
                        and identities.get(owner) != recorded_identity):
                    stale = True
            if stale:
                shutil.rmtree(candidate, ignore_errors=True)
    directory = root / f"{os.getpid()}-{uuid4().hex}"
    directory.mkdir()
    (directory / _FRAME_OWNER).write_text(
        f"{os.getpid()}\n{_own_process_identity()}")
    return os.fspath(directory)


def retain_frames_dir(directory, owner_pid=None):
    """Transfer scratch ownership to the process writing the artifact."""
    directory = Path(directory)
    owner = os.getpid() if owner_pid is None else int(owner_pid)
    identities = _process_identities()
    identity = "" if identities is None else identities.get(owner, "")
    (directory / _FRAME_OWNER).write_text(f"{owner}\n{identity}")
    return directory


def to_image(png):
    """Encoded frame bytes -> PIL image, passing None through."""
    if png is None:
        return None
    from PIL import Image
    img = Image.open(io.BytesIO(png))
    img.load()
    return img


# What a trajectory with no parseable tool call scores. It is the format
# floor and it is the LOWEST reward this repo hands out, which is what
# makes it format pressure rather than a discouragement from trying.
NO_TOOL_CALL_REWARD = -1.0

# A parsed tool call must always outrank refusing the tool format. Task
# authors may choose arbitrary fail, no-commit, convergence, and shaping
# values, so a cap on format tax alone cannot prove that ordering. Clamp the
# final task total at this boundary, after every component has been applied.
TOOL_USE_REWARD_FLOOR = -0.99

# The most an episode can be charged for how it behaved, which is the
# per-action decision cost plus 0.2 per malformed turn. Strictly smaller
# than the floor above, so the ordering NO_TOOL_CALL_REWARD < any
# tool-using trajectory holds by construction rather than by arithmetic
# that happens to work out on one task's budget.
MAX_FORMAT_TAX = 0.9


# Maximum start-pose attempts before treating a task's initial-state constraints
# as unsatisfiable for the selected seed.
MAX_INIT_DRAWS = 32


def _open_env(task, seed, **kw):
    """Open the local engine at the task's declared initial state."""
    try:
        kw = dict(kw)
        kw["snapshot_in"] = task.snapshot_for(
            int(seed), netherite_home=kw.get("netherite_home"))
        return MagmaEnv(seed=int(seed), **kw)
    except EngineError as exc:
        raise type(exc)(
            f"{exc}\n  while opening the episode for task {task.name} "
            f"({task.path}) on seed {int(seed)}") from exc


def episode_spec(task_yaml, seed, init_yaw, init_pitch):
    """The four coordinates that name one episode, in ONE shape.

    Every producer of a dataset row, a rollout spec or a reward
    extra_info builds it here, and Episode.from_spec is the only thing
    that turns it back into an episode. Five places used to build it by
    hand and two of them disagreed about whether the pose was the one
    asked for or the one the engine settled at.
    """
    return {"task_yaml": str(task_yaml), "seed": int(seed),
            "init_yaw": float(init_yaw), "init_pitch": float(init_pitch)}


def spec_values(spec):
    """A recorded spec read back, as (task_yaml, seed, yaw, pitch).

    The reading half of the pair above, and the only one. `from_spec`
    turns a spec into an episode and is the reason the shape is fixed,
    but it is not the only thing that ever has to know what a spec SAYS:
    anything that identifies an episode across a process boundary has to
    read the same four fields the same way, and a second reader that
    coerces them differently is a second opinion about which episode is
    which. Coercion matters here rather than being decoration -- a
    parquet row hands back numpy scalars, a live caller hands back
    Python floats, and str() of those two is not the same string.
    """
    return (str(spec["task_yaml"]), int(spec["seed"]),
            float(spec["init_yaw"]), float(spec["init_pitch"]))


class Episode:
    """One live deterministic episode, optionally with rendered frames."""

    def __init__(self, task, seed, init_yaw, init_pitch, parser,
                 netherite_home=None, frames_root=None,
                 raster=None, view_distance=None, capture_every=None,
                 capture_frames=True, tick_observer=None):
        self.task = task
        self.frame_width, self.frame_height = FRAME_W, FRAME_H
        self.capture_frames = bool(capture_frames)
        # Runtime-only sampling hook used by replay media. It observes a
        # transition after the engine has produced its new observation and
        # never adds, removes, or rearranges ticks.
        self._tick_observer = tick_observer
        self.model_view, selected_home, self.view_provenance = open_view(
            task.view, netherite_home)
        # NETHERITE_FRAME_SCALE changes the actual policy pixels (including
        # semantic-view GUI fallback). Record the effective value and size so
        # replay/evaluation provenance cannot certify mixed resolutions as the
        # same view merely because the renderer/asset name matches.
        self.view_provenance.update({
            "engine_frame_scale": _FRAME_SCALE,
            "engine_frame_size": [self.frame_width, self.frame_height],
        })
        # rasterizer for the per-turn frame: explicit arg, else the task
        # YAML's `raster:` key, else NETHERITE_RASTER, else cpu.
        raster = raster or task.raw.get("raster") or DEFAULT_RASTER
        # renderer chunk radius: explicit arg, else the task YAML, else
        # NETHERITE_VIEW_DISTANCE, else the engine default. Must be the
        # same at data generation and at rollout, see engine.py.
        vd = (view_distance or task.raw.get("view_distance")
              or DEFAULT_VIEW_DISTANCE)
        self._frames_dir = (_owned_frames_dir(frames_root)
                            if frames_root else None)
        kw = dict(netherite_home=str(selected_home),
                  frames_dir=self._frames_dir, frame_w=FRAME_W,
                  frame_h=FRAME_H, raster=raster,
                  view_distance=vd, capture_every=capture_every,
                  # the event journal, which the journal-backed checks
                  # read off env.journal; `journal: false` in a task
                  # turns it off, and Task refuses that combination when
                  # the task's own checks need it
                  journal=task.journal,
                  mobs=task.initial_state.world.get("mobs", False),
                  world_time=task.initial_state.world.get("time"))
        self.env = None
        try:
            self.env = _open_env(task, int(seed), **kw)
            self._initialize_state(
                task, seed, init_yaw, init_pitch, parser)
        except BaseException:
            try:
                self.close()
            except BaseException:
                # Preserve the initialization failure; cleanup is best effort
                # and close() has already removed owned frame scratch in its
                # finally block.
                pass
            raise

    def _initialize_state(self, task, seed, init_yaw, init_pitch, parser):
        """Finish initialization while the constructor owns cleanup."""
        self.env.reset_pose(float(init_yaw), float(init_pitch))
        o = self.env.obs
        # Record the requested pose as the canonical episode coordinate. The
        # engine may represent an equivalent settled yaw outside the task's
        # declared range because it accumulates rotations without wrapping.
        self.spec = episode_spec(task.path, seed, init_yaw, init_pitch)
        self.spec["view"] = dict(self.view_provenance)
        # How many poses were drawn to arrive at this one, which is 1 for
        # every episode built from a pose somebody already has: a replay,
        # a hand-written pose, a recorded spec. Only `draw` below can move
        # it, and it is how a data build reports how often the start-pose
        # screen fired.
        self.start_obs = {"x": o["x"], "y": o["y"], "z": o["z"],
                          "inv_counts": dict(o["inv_counts"]),
                          "hotbar_ids": tuple(o["hotbar_ids"]),
                          "hotbar_counts": tuple(o["hotbar_counts"]),
                          "hotbar_meta": tuple(o.get("hotbar_meta", (0,) * 9))}
        self.fired = [False] * len(task.shaping)
        self.shaped = 0.0
        self.penalty = 0.0               # accumulated malformed-turn cost
        self.nlines = 0
        self.turns = 0
        self.done = False
        self.success = False
        # The WRONG answer, which ends the episode the way a right one
        # does. None of it fires on a task with no `fail:` check.
        self.failed = False
        # Dense convergence shaping (bedrock_rl/env/task.py::Converge).
        # The target CELL is found once, here, and never again: the
        # potential has to keep being defined after the block in that cell
        # is broken, or a correct commit would hand back everything the
        # approach earned. `converged` is the running sum of the per-turn
        # phi deltas, which telescopes to phi(last) - phi(first).
        self.converge_cell = (task.converge.locate(self.env)
                              if task.converge else None)
        self.phi = (task.converge.phi(self.env, self.converge_cell)
                    if task.converge else 0.0)
        self.converged = 0.0
        # Inventory-screen latch: which surface the mouse acts on,
        # which the action compiler needs and the model is told about.
        # Read off the engine, in _sync_screen below. `_jseen` is how far
        # through the engine's event stream that read has got.
        #
        # It does NOT start closed by assertion. A .bsnp can carry an open
        # world container, so reading the engine here makes arbitrary input
        # snapshots safe rather than assuming how they were produced.
        self.gui_open = False
        self.container = 0
        self._jseen = 0
        self.max_turns = task.max_turns
        self.per_turn_ticks = task.max_ticks_per_turn
        # Action-language hook: custom harnesses can inject a parser without
        # changing the environment's transition or reward logic.
        self._parse = parser
        self.attack_stop_trace = None
        # The t0 screenshot, taken HERE and not by the caller. Every
        # producer of visual training data needs it and it costs one engine
        # tick. On a frameless episode it is None and the tick is still spent,
        # keeping transition timing independent of rendering.
        # A start state CAN carry an open container: --snapshot-in
        # restores r->container. The engine opens the screen over it on
        # the first tick it runs, which by here has already happened, so
        # the first turn compiles against the screen that is really up.
        self._sync_screen()
        self.t0_png = (capture_png(self.model_view, self)
                       if self.capture_frames else self._frameless_tick())
        self._sync_screen()
        # reset_pose and the t0 capture tick the live engine. Their events
        # describe episode setup, not model behavior, so journal-backed task
        # checks begin immediately after the first observation is complete.
        self.journal_start = len(self.env.journal)

    @property
    def init_pose(self):
        """(yaw, pitch) this episode actually started at.

        For a caller that wants to PRINT or record the pose rather than
        replay it, and it exists so that caller does not name the spec's
        keys itself. `Episode.draw` can reject a pose and draw another,
        so the interesting number is the one that was accepted, and one
        accessor for it is the same argument episode_spec makes about the
        four coordinates.
        """
        return (self.spec["init_yaw"], self.spec["init_pitch"])

    def _sync_screen(self):
        """Take the inventory-screen latch from the ENGINE's own state.

        The engine samples its screen once per tick and journals every
        change of it as `container_opened` / `container_closed`, carrying
        which container it is (0 the player's own, 1 a crafting table, 2
        a furnace, 3 a chest). See `rlj_sample_state` in
        magma/game/rl_mode.c: the value it diffs is the engine's own
        `gui_open`, so this is not an inference from the engine's state,
        it IS the engine's state.

        Nothing weaker is enough, and that is worth saying because
        weaker things look enough. `obs["container"]` alone cannot tell a
        player screen from no screen, since both are 0. Watching the
        container change ACROSS A TURN cannot see a screen that closed
        and reopened inside one, which a call that presses `key e` over
        an open table and then right-clicks it again does. And the
        `gui_open` schedule items alone cannot see a container the model
        never asked for. The event stream has all three, per tick and in
        order.

        With the stream off (`journal: false`, `NETHERITE_JOURNAL=0`)
        there is nothing to read and the latch falls back to the
        `gui_open` items the schedule executed, which is exact for an
        episode that never opens a world container and blind to one that
        does. A task routed through a container screen must leave the
        journal on.
        """
        events = self.env.journal
        for e in events[self._jseen:]:
            if e.name == "container_opened":
                self.gui_open, self.container = True, int(e["kind"])
            elif e.name == "container_closed":
                self.gui_open, self.container = False, 0
        self._jseen = len(events)

    @classmethod
    def from_spec(cls, spec, parser, task=None, **kw):
        """The episode a recorded spec names.

        The ONE thing that turns the four coordinates back into a world,
        reading them through `spec_values` so that a producer and a
        consumer cannot hold different opinions about what they mean.
        `parser` is the bound parse function, exactly as the constructor
        takes it. `task` is optional and only saves a YAML load; the
        spec's own task_yaml is what identifies the task, and it resolves
        on this box through bedrock_rl/env/task.py.
        """
        from bedrock_rl.env.task import Task
        task_yaml, seed, init_yaw, init_pitch = spec_values(spec)
        if task is None:
            task = Task.load(task_yaml)
        episode = cls(task, seed, init_yaw, init_pitch, parser=parser, **kw)
        recorded_view = spec.get("view")
        if recorded_view is not None and recorded_view != episode.spec["view"]:
            episode.close()
            raise RuntimeError(
                "recorded episode view does not match the renderer selected "
                f"now:\n  recorded: {recorded_view}\n  current:  "
                f"{episode.spec['view']}\nReplay with the same task view and "
                "engine asset checkout.")
        return episode

    @classmethod
    def draw(cls, task, seed, rng, parser, **kw):
        """Draw a reproducible unsolved episode for ``seed``.

        New poses may be rejected, but :meth:`from_spec` always replays a saved
        pose exactly. A pose is rejected when the action ray already targets a
        success block or when the success check already holds after reset.
        Rejected poses rebuild the episode because ``reset_pose`` applies a
        delta and repeated mutation would make the accepted coordinates
        non-reproducible.

        Returns an open episode owned by the caller.
        """
        blocked = frozenset(task.success.block_ids())
        rejected = []
        for _ in range(MAX_INIT_DRAWS):
            yaw, pitch = task.sample_init(rng)
            ep = cls(task, seed, yaw, pitch, parser=parser, **kw)
            # Distance is deliberately irrelevant: beginning on a target skips
            # the aiming skill even if the first click cannot yet reach it. Use
            # the action ray so screening agrees with execution.
            from bedrock_rl.env.perception import aim_target
            aimed_at_target = (bool(blocked)
                               and aim_target(ep.env)["id"] in blocked)
            state_already_solved = task.success.initially_solved(
                ep.env, ep.start_obs)
            if not aimed_at_target and not state_already_solved:
                return ep
            rejected.append(f"({yaw:+.1f}, {pitch:+.1f})")
            ep.close()
        named = task.success.spec.get("block", task.success.spec.get(
            "blocks", sorted(blocked)))
        rng_txt = task.init or "the default, yaw [-180, 180] and pitch 0"
        raise RuntimeError(
            f"task {task.name!r} ({task.path}): drew {MAX_INIT_DRAWS} "
            f"start poses on seed {seed} and every one of them began with "
            f"the crosshair already on {named}, the block its success "
            f"check names or already satisfied the state-only success "
            f"check at reset. The init range is {rng_txt}, and the poses "
            f"drawn were {', '.join(rejected[:8])}"
            + (" ..." if len(rejected) > 8 else "")
            + ". That indicates incompatible initial-state constraints rather "
            "than bad luck: move the start state or narrow `init:` so a "
            "drawn pose can begin unsolved.")

    def _score_transition(self, journal_start, step_r):
        """Score the complete transition, including its observation tick."""
        for i, chk in enumerate(self.task.shaping):
            if not self.fired[i] and chk.holds(
                    self.env, self.start_obs, journal_start):
                self.fired[i] = True
                self.shaped += chk.reward
                step_r += chk.reward
        # The dense term is deliberately NOT behind the latch above. A
        # milestone asks whether something ever happened and pays once. A
        # potential asks where the policy is now and must pay every turn.
        # Its deltas telescope, so a path from one state back to itself earns
        # exactly zero without needing a milestone latch.
        if self.task.converge is not None:
            phi = self.task.converge.phi(self.env, self.converge_cell)
            self.converged += phi - self.phi
            step_r += phi - self.phi
            self.phi = phi
        # Failure is tested first. A turn that commits to a wrong target and
        # then reaches the right one has already answered incorrectly; testing
        # success first would reward brute-force multi-action calls.
        if self.task.fail is not None and self.task.fail.holds(
                self.env, self.start_obs, journal_start):
            self.failed = True
            self.done = True
            step_r += self.task.fail.reward
        elif self.task.success.holds(
                self.env, self.start_obs, journal_start):
            self.success = True
            self.done = True
            step_r += self.task.success_reward
        elif (self.turns >= self.max_turns
              or self.env.ticks >= self.task.max_ticks):
            self.done = True
        return step_r

    def step(self, program_text, *, count_turn=True):
        """Run one tool call and optionally start a new assistant turn."""
        journal_start = getattr(self, "journal_start", 0)
        if count_turn:
            self.turns += 1
        step_r = 0.0
        # The screen can have moved since the last call returned: the
        # screenshot costs a tick, and a tick is enough for a player who
        # was still drifting to leave a container's reach. Ask before
        # compiling, not only after executing.
        self._sync_screen()
        try:
            # The camera pitch goes in with the payload because a screen
            # coordinate names a different turn at every pitch
            # (bedrock_rl/env/projection.py). It is read off the
            # engine's own observation, so every harness compiles the same
            # schedule from the same state.
            sched, nlines = self._parse(program_text, self.gui_open,
                                        self.env.obs["pitch"])
        except ProgramError as e:
            step_r -= 0.2                       # malformed turn, keep going
            self.penalty += 0.2
            frame = self.frame()
            step_r = self._score_transition(journal_start, step_r)
            return (PARSE_ERROR + str(e), frame,
                    step_r, self.done)
        self.nlines += nlines
        budget = min(self.per_turn_ticks,
                     self.task.max_ticks - self.env.ticks)
        # The whole turn as ONE batch. Truncation is by tick count and
        # nothing between the ticks reads the observation, so the items
        # that execute are known before any of them do. That makes this
        # the loop it replaces, tick for tick.
        items, used = [], 0
        for act, n_ticks in sched:
            if used >= budget:
                break
            take = min(int(n_ticks), budget - used)
            if take <= 0:
                break
            items.append((act, take))
            used += take
            if "gui_open" in act:
                # Only the fallback for an episode with no event stream;
                # _sync_screen overwrites this from the engine below when
                # there is one. Advanced from the items that ACTUALLY
                # execute, so a tick-budget truncation cannot leave it
                # describing a screen the engine never opened.
                self.gui_open = bool(act["gui_open"])
        if self.task.settle_ticks:
            items.append(({}, self.task.settle_ticks))
        if items:
            # A world mining hold is a maximum duration, not permission to
            # keep attacking the next voxel after the intended one breaks.
            # Capture the exact public click ray immediately before each
            # attack item (after any mouse pan earlier in the call), then stop
            # that hold on the first matching journal event. Ordinary
            # computer actions stay unchanged; this only gives their existing
            # per-tick batch loop the mouse-button release a human gets when
            # the target disappears.
            stop_factory = None
            if bool(getattr(self.env, "journal_on", self.task.journal)):
                def stop_factory(live_env, keys):
                    if "attack" not in keys:
                        return None
                    click = next((
                        event for event in reversed(live_env.journal)
                        if event.name == "click_target"), None)
                    if click is None:
                        return None
                    target = tuple(int(click[key]) for key in
                                   ("x", "y", "z"))
                    expected = int(click["block"])
                    if not expected:
                        return None
                    event_start = len(live_env.journal)
                    self.attack_stop_trace = {
                        "target": target, "block": expected,
                        "event_start": event_start, "stopped": False,
                        "last_events": (),
                    }

                    def stop(live_env):
                        recent = tuple(
                            (event.name,
                             tuple(int(event.get(key, -999)) for key in
                                   ("x", "y", "z")),
                             int(event.get("block", 0)))
                            for event in live_env.journal[event_start:])
                        matched = any(
                            event.name == "block_broken"
                            and tuple(int(event[key]) for key in
                                      ("x", "y", "z")) == target
                            and int(event["block"]) == expected
                            for event in live_env.journal[event_start:])
                        self.attack_stop_trace["last_events"] = recent[-6:]
                        self.attack_stop_trace["stopped"] = matched
                        return matched
                    return stop
            if self._tick_observer is None:
                self.env.act_batch(items, stop_factory=stop_factory)
            else:
                self.env.act_batch(
                    items, observer=self._tick_observer,
                    stop_factory=stop_factory)
        # The screen the NEXT call compiles against is whatever the engine
        # has up now, which is not always what the action stream asked for.
        self._sync_screen()
        # The capture is part of the transition and spends an engine tick.
        # Score only after it, so a pickup, terminal condition, or tick budget
        # reached on the exact observation shown to the policy cannot be left
        # pending until a later turn that may never happen.
        frame = self.frame()
        step_r = self._score_transition(journal_start, step_r)
        # `perception` is task configuration. Every field it carries is
        # derived from `env.obs`; nothing in it depends on rendered pixels.
        return (render_obs(self.env, gui_open=self.gui_open,
                           perception=self.task.perception), frame,
                step_r, self.done)

    def frame(self):
        """The current frame as a PIL image, or None when rendering is off.

        Costs one engine tick either way; see ``MagmaEnv.frame``.
        """
        image = (capture_png(self.model_view, self)
                 if self.capture_frames else self._frameless_tick())
        self._sync_screen()
        return to_image(image)

    def _frameless_tick(self):
        """Match a capture's one transition tick without rasterizing it."""
        self.env.act()
        if self._tick_observer is not None:
            self._tick_observer(self.env)
        return None

    def t0_image(self):
        """The t0 screenshot as a PIL image, or None when frameless."""
        return to_image(self.t0_png)

    def final_reward(self):
        """Compute terminal reward with a bounded tool-format tax.

        Decision cost and malformed-call penalties share a capped format-tax
        budget. Task components are intentionally configurable and can still
        be arbitrarily negative, so the complete total is clamped to
        TOOL_USE_REWARD_FLOOR after all components are applied. That final
        boundary, not an assumption about one task's arithmetic, proves that
        every parsed tool attempt outranks refusing the tool format.

        Successful episodes are not taxed: decision cost discourages spam on
        unsuccessful attempts without ranking shorter lucky successes above
        deliberate successful behavior.

        An episode ends having answered right, answered wrong, or not answered.
        ``no_commit`` is charged to the last case and
        clamped at load to at least what `fail` costs, so not answering is
        never safer than answering wrongly and the only way up is a right
        answer. A task naming neither key has `failed` False and
        `no_commit` 0.0 and lands on exactly the line it always did.

        The dense total rides on every branch INCLUDING the win, because
        it is a potential and not a prize. Withholding it from one branch
        is what would make it farmable: a policy could then bank progress
        by picking the branch that keeps it.
        """
        tax = min(self.task.decision_cost * self.nlines + self.penalty,
                  MAX_FORMAT_TAX)
        r = self.shaped + self.converged
        if self.success:
            raw = r + self.task.success_reward
        elif self.failed:
            raw = r + self.task.fail.reward - tax
        else:
            raw = r + self.task.no_commit - tax
        return max(raw, TOOL_USE_REWARD_FLOOR)

    def close(self):
        try:
            if self.env is not None:
                self.env.close()
        finally:
            self.env = None
            if self._frames_dir:
                shutil.rmtree(self._frames_dir, ignore_errors=True)
                self._frames_dir = None


# The first line of every tool response. It makes observations readable in
# decoded logs; structured trajectories use tool roles and tool_call_id.
OBS_HEADER = "=== NEW OBSERVATION ==="

PARSE_ERROR = "could not parse program: "


def tool_response_text(obs_text, done, next_msg):
    """The text half of one tool response.

    Single-sourced because live tools and synthetic data generation must
    produce the same bytes. An SFT row
    whose tool turns read differently from the ones the agent loop hands
    back is training on a prompt the model never sees.
    """
    return (OBS_HEADER + "\n" + obs_text
            + ("\n\nEPISODE COMPLETE." if done else "\n\n" + next_msg))
