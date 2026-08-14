"""Task spec (YAML): the checks, seeds, and episode budget.

  name, goal                   the task and its goal text
  goal_cu                      the computer-use goal shown to the model,
                               falling back to goal when absent
  seeds                        world seeds episodes draw from
  initial_state                inventory, vitals, spawn, and world settings
  snapshots_dir                optional cache location for initial_state
  snapshot_radius             captured block radius for initial_state
  init                         start pose, a range or a number, drawn
                               per episode
  max_lines                    actions per tool call
  max_ticks                    tick budget per episode
  max_turns                    assistant turns per episode
  max_ticks_per_turn           tick budget per tool call
  settle_ticks                 no-op ticks before the terminal check
  decision_cost                per-action reward cost
  success                      terminal check, worth `reward`
  shaping                      milestones, each firing at most once,
                               tested ONCE A TURN, after the whole batch
                               of schedule items and after settle_ticks.
                               A milestone that holds only in the middle
                               of a turn is never seen, which matters for
                               a snapshot check like crosshair_block; the
                               journal checks read what HAPPENED and do
                               not care
  fail                         the WRONG answer: a terminal check, worth
                               a negative `reward`, tested BEFORE success
  no_commit                    what ending without answering at all costs
  converge                     dense potential-based shaping on how close
                               the crosshair is, which is not a milestone
                               and is not behind the fired-latch

Score is the sum of fired shaping rewards, plus the success reward if the
terminal check holds after settle, plus the dense convergence total, plus
whichever of `fail`/`no_commit` the episode ended on, minus
decision_cost per action.

Both kinds of check are evaluated once per turn, on the state the turn
ended in. Nothing looks at the world between the actions inside a turn,
which is what lets a turn run as one batch. So a SNAPSHOT check that is
only true mid turn cannot pay:
`crosshair_block` on a block the agent aims at and then turns away from
in the same tool call sees the state after the turn and answers False. A
milestone like that has to be a JOURNAL check, which reads what happened
rather than what is true now.

Checks come in two kinds. Snapshot checks read the observation record and
say only what is true now. Journal checks read the engine's event stream
and say what HAPPENED, including in what order. Each kind is registered
from one table, CHECKS below and bedrock_rl/env/journal.py.

Every mapping in the file is CLOSED. TASK_KEYS below is the whole of the
top level and CHECK_KEYS is the whole of a snapshot check, so a key this
repo does not read stops the load instead of doing nothing quietly.
"""
import difflib
import os

import yaml

from bedrock_rl import resources
from bedrock_rl.env import journal
from bedrock_rl.env.catalog import INV_ORDER, resolve_blocks, resolve_items
from bedrock_rl.env.engine import hotbar_count


# ── unknown keys are typos ───────────────────────────────────────────────
# A YAML mapping is read with `.get(key, default)`, so a key nobody reads
# is a key that does nothing, and a MISSPELLED key is a key that does
# nothing while looking like it does. `max_tick: 2000` leaves the tick
# budget at its default, `shapings:` deletes every shaping reward, and
# both load, validate and train, so the run is scored against a task
# nobody wrote.
#
# So every mapping in a task file is closed: its keys are declared, and
# anything else stops the load.

def reject_unknown_keys(what, d, known, where=None):
    """Raise unless every key of `d` is in `known`.

    `what` names the mapping in the message and `where` the file it came from.
    """
    bad = sorted(set(d) - set(known))
    if not bad:
        return
    lines = []
    for k in bad:
        near = difflib.get_close_matches(k, sorted(known), n=1, cutoff=0.6)
        lines.append(f"  {k}" + (f"    did you mean {near[0]}?"
                                 if near else ""))
    raise ValueError(f"unknown {what} key{'' if len(bad) == 1 else 's'}"
                     + (f" in {where}" if where else "") + ":\n"
                     + "\n".join(lines)
                     + f"\nknown keys are {', '.join(sorted(known))}")


# ── the check registry ───────────────────────────────────────────────────
# `success.type` in a task YAML names one of these. It is a registry and
# not an if/elif chain, because the same strings are switched on again in
# expert.py, and a check type added to one switch and not the other is a
# silent gap. The oracles are the same shape, keyed the same way, in
# expert.py::ORACLES.
CHECKS = {}
# Per check type, the spec keys it reads. A snapshot check that is handed
# `counts: 4` instead of `count: 4` would otherwise silently want one
# item, so the registry carries the keys and Check refuses the rest.
# `type` and `reward` are on every check and are added below.
CHECK_KEYS = {}
# And the ones it cannot run without. A `crosshair_block` with no
# `blocks:` loads, never resolves Check.ids, and raises AttributeError on
# the first evaluation, inside the reward path, where a bare except turns
# it into the format floor. Every rollout then scores exactly -1.0, the
# GRPO group has no variance, and the run reads as a model that will not
# learn. Journal checks compile their names at load for this reason;
# these now do the same.
CHECK_REQUIRED = {}
# What a check pays when its spec names no `reward:`. Read here and
# nowhere else. The episode paid 0.1 for a shaping milestone with no
# `reward:` while `bedrock task show`, the documented way to see what a
# task pays, printed 0 for the same milestone; two files each holding
# their own opinion of one number is how that happens.
DEFAULT_SHAPING_REWARD = 0.1
DEFAULT_SUCCESS_REWARD = 1.0

# ── commitment incentives ────────────────────────────────────────────────
# ``fail``, ``no_commit``, and ``converge`` let tasks avoid an abstention
# optimum: not attempting the task should not be safer than a wrong attempt,
# while incremental progress can still receive dense feedback. These are the
# defaults used when a task names a key without an explicit number.
DEFAULT_FAIL_REWARD = -0.5
DEFAULT_CONVERGE_REWARD = 0.2
# Angular distance at which the default dense aiming potential reaches zero.
# Tasks can override this to match the separation between valid and invalid
# targets in their own geometry.
DEFAULT_CONVERGE_SPAN = 20.0


def check_type(name, keys=(), required=()):
    """Decorator: register a success/shaping check under `name`.

    The function takes (check, env, start_obs) and returns a bool. `check`
    carries the YAML spec plus any ids resolved at load time.

    `keys` is the spec keys the function reads and `required` the subset
    it cannot run without. A spec carrying anything else, or missing one
    of these, is refused at load, so a check body never has to defend
    itself and a mistake is a load error rather than an exception per
    rollout. Journal-backed checks pass keys=None, because their
    constraint names come from the event's own slots and
    bedrock_rl/env/journal.py::EventMatch already refuses the rest."""
    def deco(fn):
        CHECKS[name] = fn
        if keys is not None:
            assert set(required) <= set(keys), (name, required, keys)
            CHECK_KEYS[name] = frozenset(keys) | {"type", "reward"}
            CHECK_REQUIRED[name] = frozenset(required)
        return fn
    return deco


@check_type("pitch_below", ("value",), required=("value",))
def _pitch_below(c, env, start_obs):
    return env.obs["pitch"] <= float(c.spec["value"])


@check_type("pitch_above", ("value",), required=("value",))
def _pitch_above(c, env, start_obs):
    return env.obs["pitch"] >= float(c.spec["value"])


@check_type("crosshair_block", ("blocks", "max_dist"),
            required=("blocks",))
def _crosshair_block(c, env, start_obs):
    # Use the action ray rather than the camera pixel so this milestone agrees
    # with the block a click would affect.
    from bedrock_rl.env.perception import aim_target
    ch = aim_target(env)
    if ch["id"] not in c.ids:
        return False
    md = c.spec.get("max_dist")
    return md is None or (ch["dist"] is not None
                          and ch["dist"] <= float(md))


@check_type("blocks_visible", ("blocks", "min_pixels"),
            required=("blocks",))
def _blocks_visible(c, env, start_obs):
    return env.visible_count(c.ids) >= int(c.spec.get("min_pixels", 1))


@check_type("item_gained", ("item", "count"), required=("item",))
def _item_gained(c, env, start_obs):
    k = c.spec["item"]
    return (env.obs["inv_counts"][k] - start_obs["inv_counts"][k]
            >= int(c.spec.get("count", 1)))


@check_type("hotbar_gained", ("items", "count"), required=("items",))
def _hotbar_gained(c, env, start_obs):
    now = hotbar_count(env.obs, c.item_ids)
    was = hotbar_count(start_obs, c.item_ids)
    return now - was >= int(c.spec.get("count", 1))


@check_type("selected_item", ("items",), required=("items",))
def _selected_item(c, env, start_obs):
    """Whether the currently selected hotbar slot holds a target item."""
    del start_obs
    selected = int(env.obs["hotbar_sel"])
    hotbar = env.obs["hotbar_ids"]
    return (0 <= selected < len(hotbar)
            and hotbar[selected] in c.item_ids)


@check_type("moved_distance", ("value",), required=("value",))
def _moved_distance(c, env, start_obs):
    """Straight-line DISPLACEMENT from the start pose.

    Kept because tasks use it, but note it is not distance travelled: an
    agent that walks out and back scores zero here. The journal's
    `travelled` check is the path length and is what "blocks travelled"
    means."""
    dx = env.obs["x"] - start_obs["x"]
    dz = env.obs["z"] - start_obs["z"]
    return (dx * dx + dz * dz) ** 0.5 >= float(c.spec["value"])


@check_type("player_touching_block", ("blocks", "margin"),
            required=("blocks",))
def _player_touching_block(c, env, start_obs):
    """Whether the current sampled collision box overlaps a listed block.

    The player is 0.6 blocks wide and 1.8 blocks tall; ``margin`` optionally
    requires extra clearance. This reads the bounded nearby-block summary, so
    it is useful for authored obstacles but cannot prove contact never
    happened between observations or with an underground fluid omitted from
    that summary. Use the journaled ``player_fluid_state`` event when fluid
    contact itself must be terminal.
    """
    del start_obs
    margin = float(c.spec.get("margin", 0.0))
    if margin < 0:
        raise ValueError("player_touching_block.margin cannot be negative")
    x = float(env.obs["x"])
    y = float(env.obs["y"])
    z = float(env.obs["z"])
    low = (x - 0.3 - margin, y - margin, z - 0.3 - margin)
    high = (x + 0.3 + margin, y + 1.8 + margin, z + 0.3 + margin)
    for block, bx, by, bz in env.obs["blocks"]:
        if block not in c.ids:
            continue
        if (low[0] <= bx + 1 and high[0] >= bx
                and low[1] <= by + 1 and high[1] >= by
                and low[2] <= bz + 1 and high[2] >= bz):
            return True
    return False


# Every event the engine journals becomes a check here, plus `sequence`
# for ordering. One call, because the event table is the registry: see
# bedrock_rl/env/journal.py for how to add an event type.
journal.register_checks(check_type)


def resolve_path(path):
    """A task file path, resolved on THIS box.

    Every RL row pins the task it was generated from, and that used to be
    an absolute path. A data-generation box and a trainer box with
    different checkout paths therefore produced a dataset where every row
    named a file that was not there, and the reward turned each one into
    -1.0, which reads exactly like format collapse. Rows carry the path
    relative to the checkout now, and both forms resolve here.

    A path that resolves nowhere raises, and says which box it is on.
    """
    p = os.path.expanduser(str(path))
    if os.path.exists(p):
        return os.path.abspath(p)
    root = resources.repo_root()
    if root is not None and not os.path.isabs(p):
        cand = os.path.join(str(root), p)
        if os.path.exists(cand):
            return os.path.abspath(cand)
    import socket
    raise FileNotFoundError(
        f"task file {path!r} is not on this box ({socket.gethostname()}). "
        "A dataset row written on another machine carries an absolute "
        "path that is only right there. Regenerate the rows with "
        "`bedrock generate`, which writes the path relative to the "
        "checkout, or run from a checkout laid out the same way.")


def relative_path(path):
    """A task path as an RL row should carry it.

    Relative to the checkout when the task lives inside it, so the row
    can be read on any box with this repo, and absolute otherwise, since
    a task outside the checkout has no portable name.
    """
    p = os.path.abspath(os.path.expanduser(str(path)))
    root = resources.repo_root()
    if root is None:
        return p
    try:
        return os.path.relpath(p, str(root)) if os.path.commonpath(
            [p, str(root)]) == str(root) else p
    except ValueError:                       # different drives on Windows
        return p


# ── the task keys ────────────────────────────────────────────────────────
# Every top-level key a task file may carry, next to what reads it. Not
# every one is read HERE: several are read off `task.raw` by the episode,
# the prompt builder or the snapshot builder, and a key that has a reader
# anywhere is a key a task may set. Adding a `task.raw.get("...")`
# anywhere without adding the name here makes every task that sets it
# fail to load, which is the safe direction to be wrong in.
TASK_KEYS = frozenset({
    "name", "goal",                     # here
    "goal_cu",                          # adapters/netherite/computer.py, the model's goal
    "seeds", "init",                    # here
    "snapshots_dir", "snapshot_radius", # initial-state cache, resolved here
    "max_lines", "max_ticks",           # here
    "settle_ticks", "decision_cost",    # here
    "max_turns",                        # env/episode.py, train/rollout.py
    "max_ticks_per_turn",               # env/episode.py, adapters/netherite/computer.py
    "success", "shaping",               # here
    "fail", "no_commit", "converge",    # here, and env/episode.py
    "journal",                          # here, and env/episode.py
    "view_distance", "raster", "view",  # env/episode.py, env/views.py
    "perception",                       # here, and env/observation.py
    "system_hint", "system_hint_coarse", "finish_hint",   # sdg/guidance.py
    "turn_hint_provider",               # import-path dynamic feedback
    "initial_state",                    # env/initial_state.py
})


class Check:
    def __init__(self, spec, default_reward=DEFAULT_SHAPING_REWARD):
        self.type = spec["type"]
        self.spec = spec
        self.custom = None
        if self.type in CHECKS:
            self.fn = CHECKS[self.type]
        elif ":" in self.type:
            reject_unknown_keys(
                "custom check", spec, {"type", "config", "reward"})
            from bedrock_rl.core.components import ComponentSpec
            component = {"type": self.type}
            if "config" in spec:
                component["config"] = spec["config"]
            component = ComponentSpec.parse(component, what="custom check")
            target = component.resolve()
            # An unconfigured function is already the predicate. Classes and
            # configured factories are constructed like every other component.
            import inspect
            self.custom = (component.create()
                           if component.config or inspect.isclass(target)
                           else target)
            if not (callable(self.custom)
                    or callable(getattr(self.custom, "holds", None))):
                raise TypeError(
                    f"custom check {self.type!r} must be callable or expose "
                    "holds(env, start_obs)")
            self.fn = None
        else:
            raise KeyError(f"unknown check type {self.type!r}; registered "
                           f"types are {sorted(CHECKS)}. Use a "
                           "package.module:object import path for a custom "
                           "check.")
        if self.type in CHECK_KEYS:
            reject_unknown_keys(f"{self.type} check", spec,
                                CHECK_KEYS[self.type])
            missing = sorted(CHECK_REQUIRED[self.type] - set(spec))
            if missing:
                raise ValueError(
                    f"{self.type} check is missing {', '.join(missing)}, "
                    f"which it cannot run without. Its keys are "
                    f"{', '.join(sorted(CHECK_KEYS[self.type]))}.")
        if self.type == "item_gained" and spec["item"] not in INV_ORDER:
            raise ValueError(
                f"item_gained.item must be one of the aggregate inventory "
                f"counters {list(INV_ORDER)}, got {spec['item']!r}. Use "
                "hotbar_gained for arbitrary item ids.")
        if (self.type == "player_touching_block"
                and float(spec.get("margin", 0.0)) < 0):
            raise ValueError("player_touching_block.margin cannot be negative")
        # What this check pays when it fires, resolved ONCE, at load.
        # Everything that pays it or prints it reads this attribute.
        self.reward = float(spec.get("reward", default_reward))
        # Journal-backed checks compile their constraints HERE, at task
        # load, so a misspelled slot name is a load error naming the real
        # slots rather than a constraint that quietly matches everything.
        self.journal = journal.compile_check(spec)
        if "blocks" in spec:
            self.ids = resolve_blocks(spec["blocks"])
        if "items" in spec:
            self.item_ids = resolve_items(spec["items"])

    def block_ids(self):
        """The engine block ids this check TARGETS, or ().

        One question asked of both kinds of check, because the two spell
        it differently and nothing else should have to know that: a
        snapshot check names `blocks:` and resolved them into `self.ids`
        above, and a journal check names the event's own `block` slot and
        resolved it inside its EventMatch. Both answers come from
        catalog.resolve_blocks, so this is the check's own mapping and not
        a second reading of the YAML.

        () is the honest answer for a check with no block target at all,
        `item_gained` being the shipped one, and callers treat it as "this
        rule does not apply here" rather than as an error.

        For a `sequence` it is the LAST step's block, because that step is
        the one whose event ends the episode; the blocks the earlier steps
        name are waypoints an episode is meant to pass through. For `all`
        and `any`, it is the union: neither composition designates one event
        as the terminal target.
        """
        if "blocks" in self.spec:
            return tuple(self.ids)
        j = self.journal
        if isinstance(j, list):
            if self.type in ("all", "any"):
                return tuple(sorted({block for step in j
                                     for block in step.block_ids()}))
            j = j[-1] if j else None                 # a `sequence`
        return j.block_ids() if j is not None else ()

    def initially_solved(self, env, start_obs):
        """Whether a built-in state terminal already holds at reset.

        Journal checks describe events during the episode and an empty
        journal is never pre-solved. Custom checks may have side effects or
        depend on application state, so the generic draw path intentionally
        does not invoke them.
        """
        if self.custom is not None or self.journal is not None:
            return False
        return bool(self.fn(self, env, start_obs))

    def holds(self, env, start_obs, journal_start=0):
        if self.custom is not None:
            fn = getattr(self.custom, "holds", self.custom)
            return bool(fn(env, start_obs))
        if self.journal is not None:
            return journal.holds(
                self.type, self.journal, env.journal, journal_start)
        return self.fn(self, env, start_obs)


class Converge:
    """DENSE, potential-based shaping on how close the crosshair is.

    A shaping rung is a milestone: it answers "did this ever happen",
    which is true forever once it is true, and the fired-latch is what
    stops it being paid twice. That shape cannot carry "how close are you
    NOW", which is signed, which has to be paid on every turn, and which
    a latch would silence after the first one. So this is not a Check and
    it does not go in `shaping:`.

    What keeps it from being farmable is not a latch, it is the form.
    This is potential-based shaping in the Ng-Harada-Russell (1999)
    sense: the turn's reward is phi(after) - phi(before) for a phi that
    is a function of the STATE, so the sum over an episode TELESCOPES to
    phi(last) - phi(first) whatever happened in between. Panning onto the
    target and off it again nets exactly zero, twenty times over nets
    exactly zero, and there is nothing to farm. It is also why the term
    is bounded without a clip: phi is in [0, reward] by construction, so
    the whole episode's dense total is in [-reward, +reward] and a
    correct commit always dominates it.

      phi(state) = reward * (1 - min(miss, span_deg) / span_deg)

    with `miss` the angle from the view to the target cell's centre
    (bedrock_rl/env/engine.py::miss_angle_deg). Zero once the target is
    span_deg or more away, so a task with the target off screen gets a
    gradient only once it is roughly in view, and full value on the
    target.

    THE TARGET CELL IS LOCKED AT t0 and phi is measured against those
    coordinates for the rest of the episode, which is the one subtle
    thing here. Re-finding the block every turn would make phi jump to
    zero on the turn the target is BROKEN, so a correct commit would
    hand back everything the approach earned and the term would pay
    "get close and freeze" more than "get close and commit" -- the exact
    island the source repo had to gate its FOV bonus against. A cell is a
    coordinate and does not stop existing when the block in it does.

    Frameless-safe. `obs["blocks"]` (the 256 nearest non-air cells) and
    the pose are in the observation record, which a replay parses like
    any other, so nothing here needs a rendered frame.
    """

    KEYS = frozenset({"blocks", "span_deg", "reward"})

    def __init__(self, spec):
        reject_unknown_keys("converge", spec, self.KEYS)
        if "blocks" not in spec:
            raise ValueError(
                "converge is missing blocks, which names the cell the "
                "crosshair is converging ON and which it cannot run "
                f"without. Its keys are {', '.join(sorted(self.KEYS))}.")
        self.spec = spec
        self.ids = resolve_blocks(spec["blocks"])
        self.span_deg = float(spec.get("span_deg", DEFAULT_CONVERGE_SPAN))
        self.reward = float(spec.get("reward", DEFAULT_CONVERGE_REWARD))
        if self.span_deg <= 0.0:
            raise ValueError(f"converge.span_deg is {self.span_deg}, and a "
                             "span of zero degrees divides the potential "
                             "by zero; it is the angle at which the term "
                             "reads as no progress at all")
        if self.reward < 0.0:
            raise ValueError(
                f"converge.reward is {self.reward}. It is the SCALE of a "
                "potential and not a penalty: the term already pays "
                "negative when the crosshair moves away, and a negative "
                "scale would pay for moving away instead.")

    def locate(self, env):
        """The target cell this episode's potential is measured against,
        or None when the observation cannot see one.

        The nearest matching block by eye distance is locked once so the
        potential cannot jump to another instance mid-episode.
        """
        from bedrock_rl.env.engine import eye_dist2
        best, best_distance = None, float("inf")
        for block, x, y, z in env.obs["blocks"]:
            if block not in self.ids:
                continue
            distance = eye_dist2(env.obs, x, y, z)
            if distance < best_distance:
                best, best_distance = (x, y, z), distance
        return best

    def phi(self, env, cell):
        """The potential of the state the env is in NOW.

        0.0 for a cell of None, which is an episode whose target is not
        in the observation's range: the term is then inert for the whole
        episode rather than switching on halfway through, because a
        potential that appears mid-episode is a step of free reward.
        """
        if cell is None:
            return 0.0
        miss = env.miss_angle_to(*cell)
        return self.reward * (1.0 - min(miss, self.span_deg) / self.span_deg)


class Task:
    def __init__(self, d, path=None):
        reject_unknown_keys("task", d, TASK_KEYS, where=path)
        self.raw = dict(d)
        d = self.raw
        self.path = path
        self.name = d["name"]
        self.goal = d["goal"].strip()
        # An ABSENT seeds key still means seed 0, which is the documented
        # default for a hand-written task. An EMPTY one is different and is
        # not a default. Every consumer cycles
        # `seeds[i % len(seeds)]`, so an empty list used to surface as a
        # ZeroDivisionError inside the data generator, several files away
        # from the cause.
        self.seeds = list(d["seeds"] or []) if "seeds" in d else [0]
        if not self.seeds:
            raise ValueError(
                f"task {d['name']!r}"
                + (f" ({path})" if path else "")
                + " has an empty seed list, so no episode can be started."
                " Give it a non-empty `seeds:` list, or remove the key to"
                " use seed 0. Synthetic runs should select worlds through"
                " their sampler component.")
        from bedrock_rl.env.initial_state import InitialStateSpec
        self.initial_state = InitialStateSpec.parse(d.get("initial_state"))
        if not self.initial_state.enabled and "snapshot_radius" in d:
            raise ValueError(
                "snapshot_radius only controls initial_state snapshots; "
                "task needs initial_state to set it")
        radius = d.get("snapshot_radius", 32)
        # Ten chunks is the largest region the patched writer deliberately
        # emits.  A 320x128x320 snapshot is 13,107,200 cells, below the
        # engine loader's 2^24-cell safety cap.  Keep this boundary identical
        # to the native writer instead of accepting a value it would silently
        # clamp.
        if isinstance(radius, bool) or not isinstance(radius, int):
            raise TypeError("snapshot_radius must be an integer in 1..160")
        if not 1 <= radius <= 160:
            raise ValueError("snapshot_radius must be in 1..160")
        self.snapshot_radius = radius
        if self.initial_state.enabled:
            if not d.get("snapshots_dir"):
                base = (os.path.dirname(os.path.abspath(path)) if path
                        else os.getcwd())
                d["snapshots_dir"] = os.path.join(
                    base, ".bedrock", "snapshots", self.name)
        elif d.get("snapshots_dir"):
            raise ValueError(
                "snapshots_dir is only a cache location; task needs "
                "initial_state to build a snapshot")
        self.snapshots_dir = d.get("snapshots_dir")
        if self.snapshots_dir and path and not os.path.isabs(
                self.snapshots_dir):
            self.snapshots_dir = os.path.normpath(os.path.join(
                os.path.dirname(os.path.abspath(path)), self.snapshots_dir))
        self.init = d.get("init", {})
        if not isinstance(self.init, dict):
            raise TypeError("task init must be a mapping")
        reject_unknown_keys("task init", self.init, {"yaw", "pitch"})
        self.max_lines = int(d.get("max_lines", 20))
        self.max_ticks = int(d.get("max_ticks", 2000))
        # Parsed HERE like every other budget. It was the one documented
        # key with no attribute, so four consumers reached into `raw` and
        # each restated the default; two of them cap the same loop, so a
        # changed default would have moved one cap and not the other.
        self.max_turns = int(d.get("max_turns", 6))
        self.max_ticks_per_turn = int(d.get("max_ticks_per_turn", 200))
        self.settle_ticks = int(d.get("settle_ticks", 6))
        for label, value in (
                ("max_lines", self.max_lines),
                ("max_ticks", self.max_ticks),
                ("max_turns", self.max_turns),
                ("max_ticks_per_turn", self.max_ticks_per_turn)):
            if value < 1:
                raise ValueError(f"task {self.name!r} needs {label} >= 1")
        if self.settle_ticks < 0:
            raise ValueError(
                f"task {self.name!r} needs settle_ticks >= 0")
        self.decision_cost = float(d.get("decision_cost", 0.005))
        # Structured perception is task-scoped because it changes the prompt.
        # Keeping it in task config makes data generation, rollout, and
        # evaluation use the same observation contract.
        self.perception = bool(d.get("perception", False))
        from bedrock_rl.env.views import ViewSpec
        self.view = ViewSpec.parse(d.get("view"))
        self.success = Check(d["success"],
                             default_reward=DEFAULT_SUCCESS_REWARD)
        self.success_reward = self.success.reward
        self.shaping = [Check(s) for s in d.get("shaping", [])]
        # A wrong answer is terminal like success. Treating it as repeatable
        # shaping would let a policy brute-force invalid targets before the
        # valid one and makes reward balancing depend on the remaining budget.
        self.fail = (Check(d["fail"], default_reward=DEFAULT_FAIL_REWARD)
                     if "fail" in d else None)
        if self.fail is not None and self.fail.reward >= 0.0:
            raise ValueError(
                f"task {self.name!r} pays {self.fail.reward} for its "
                "`fail:` check, which is the WRONG answer ending the "
                "episode; it has to cost something. Give it a negative "
                "`reward:`.")
        # THE GIVE-UP DEFENCE. `no_commit:` is charged to an episode that
        # ended having neither succeeded nor tripped `fail:` -- it never
        # answered. Without it, not answering is a free floor and a policy
        # that is unsure is paid to sit still, which is the collapse
        # DEFAULT_FAIL_REWARD above describes.
        #
        # THE KNOB IS HARDENED, and the hardening is part of the design
        # rather than a nicety on top of it: the number is CLAMPED at load
        # to at least the worst commit penalty the task carries, so
        # `no_commit: -0.1`
        # beside a -0.6 wrong answer loads as -0.6 and `no_commit: 0` is
        # REFUSED. The defence must not be silently disableable by the one
        # file that most wants to disable it. A task that does not want it
        # leaves the key out; a task that names it gets at least as much
        # as its wrong answer costs, and may ask for more.
        self.no_commit = float(d.get("no_commit", 0.0))
        if "no_commit" in d:
            if self.no_commit >= 0.0:
                raise ValueError(
                    f"task {self.name!r} sets no_commit: {self.no_commit}. "
                    "It is what ending WITHOUT committing costs and it has "
                    "to be negative; 0 is the give-up hatch this key "
                    "exists to close. Remove the key to have no penalty "
                    "at all.")
            worst = min([0.0] + [c.reward for c in self.shaping
                                 if c.reward < 0.0]
                        + ([self.fail.reward] if self.fail else []))
            self.no_commit = min(self.no_commit, worst)
        # Dense convergence shaping. See Converge above for why it is not
        # a `shaping:` rung and cannot be one.
        self.converge = Converge(d["converge"]) if "converge" in d else None
        # `journal: false` turns the event stream off for a task that
        # does not read it. A task that DOES read it and also turns it
        # off would score zero on every journal check with no error, so
        # that combination is refused here instead.
        #
        # A task whose route runs through a WORLD CONTAINER also reads
        # the stream, without saying so in a check: the inventory-screen
        # latch is taken from it (Episode._sync_screen), and with it off
        # the harness cannot see a screen the engine opened. That cannot
        # be detected from the YAML.
        self.journal = d.get("journal")
        if self.journal is not None and not isinstance(self.journal, bool):
            raise TypeError("task journal must be true or false")
        needs_journal = any(c.journal is not None
                            for c in [self.success] + self.shaping
                            + ([self.fail] if self.fail else []))
        if self.journal is False and needs_journal:
            raise ValueError(
                f"task {self.name!r} sets journal: false but has checks "
                "that read the event journal, which would then match "
                "nothing. Remove the key, or use snapshot checks.")
        if needs_journal:
            # A task correctness requirement outranks the process-wide
            # performance switch. Otherwise NETHERITE_JOURNAL=0 overrides an
            # omitted task key after validation and makes every event-backed
            # check silently impossible.
            self.journal = True

    @classmethod
    def load(cls, path):
        real = resolve_path(path)
        with open(real) as f:
            return cls(yaml.safe_load(f), path=real)

    def snapshot_for(self, seed, netherite_home=None):
        """Path of this seed's start-state snapshot (built lazily on
        first use, with the given engine checkout), or None for tasks
        without one."""
        if not self.snapshots_dir:
            return None
        from bedrock_rl.env.snapshots import ensure_seed
        return ensure_seed(self, seed, netherite_home=netherite_home)

    def sample_init(self, rng):
        def draw(key, default):
            v = self.init.get(key, default)
            if isinstance(v, (list, tuple)):
                return rng.uniform(float(v[0]), float(v[1]))
            return float(v)
        return draw("yaw", [-180, 180]), draw("pitch", 0.0)
