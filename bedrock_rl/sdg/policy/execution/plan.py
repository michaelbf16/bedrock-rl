"""Closed-loop execution of immutable mining and navigation plans.

Planning and execution deliberately stay separate.  ``planning.py`` proves a
route against an authoritative initial snapshot; this module advances that
route one observed engine transition at a time.  It never invents a detour or
silently treats an absent block as success.  Every excavation and target
removal must have an exact live journal record before the player moves on.
"""

from __future__ import annotations

import math
from typing import Iterable

from bedrock_rl.sdg.policy import (
    MinecraftController,
    PolicyVeto,
)
from bedrock_rl.sdg.policy.planning import MiningTargetPlan
from bedrock_rl.sdg.policy.spatial import DEFAULT_BLOCK_RULES
from bedrock_rl.env.journal import FLUID_BODY


Cell = tuple[int, int, int]

# Public click rays select these replaceable plants even though collision and
# path planners correctly treat them as passable. Clearing the exact reported
# voxel is normal play and is safer than attacking through it. IDs are the
# pinned 1.11.2 registry: tall grass, dead bush, flowers/mushrooms, crops,
# stems, vines, lily pads, and the two crop variants.
_SOFT_RAY_OBSTRUCTIONS = frozenset({
    31, 32, 37, 38, 39, 40, 59, 104, 105, 106, 111, 115, 141, 142, 175,
})


def _cell(value) -> Cell:
    return tuple(int(item) for item in value)


def _event_cell(event) -> Cell:
    return tuple(int(event[key]) for key in ("x", "y", "z"))


def _live_required_occluder(controller, env, requested, required):
    """Return an exact public-ray blocker that is itself safe to clear.

    Falling blocks can refill a route-headroom voxel after its original
    ``block_broken`` proof.  A later aim then reports that same declared cell
    in front of the next target.  Historical proof must not make the executor
    pretend the refill is air, but only the public click ray may authorize a
    second break.
    """

    click_target = getattr(controller, "click_target", None)
    if click_target is None:
        return None
    _block, target = click_target(env)
    if target is None:
        return None
    target = _cell(target)
    return (target if target != _cell(requested)
            and target in frozenset(required) else None)


from bedrock_rl.sdg.policy.execution.contracts import (
    MiningExecution,
)

class PlanExecutor:
    """Execute ordered mining legs with live safety and outcome checks.

    The controller must already have been reset for the current episode.
    Call :meth:`reset` at the exact live state represented by the first
    route's first stand, then call :meth:`actions` once per observation until
    :attr:`done` is true.  Returned :class:`PolicyVeto` objects are ordinary
    expected case rejections in the synthetic-data harness.

    ``lighting`` is optional and intentionally duck-typed.  A callable is
    invoked as ``lighting(env)``.  An object may instead expose the reusable
    route-lighting hooks from ``WallTorchRouteMixin``: ``start_route_lighting``,
    ``observe_route_travel``, ``remember_route_position``, and ``light_route``.
    Lighting is offered only while the player is at a proved route stand.
    """

    def __init__(
        self,
        controller: MinecraftController,
        targets: Iterable[MiningTargetPlan | MiningExecution],
        *,
        target_block=None,
        target_tool=None,
        drop_item=None,
        drop_count: int = 0,
        tunnel_tool=None,
        mining_ticks: int = 80,
        tunnel_ticks: int | None = None,
        stand_tolerance: float = 0.7,
        max_clear_attempts: int = 8,
        max_mine_attempts: int = 8,
        max_move_attempts: int = 8,
        max_pickup_attempts: int = 4,
        health_loss_tolerance: float = 0.0,
        veto_fluids: bool = True,
        lighting=None,
        lighting_immediate: bool = False,
        lighting_route_kinds: Iterable[str] = ("tunnel",),
        item_recovery=None,
        break_declarer=None,
    ):
        self.controller = controller
        self.targets = tuple(
            item if isinstance(item, MiningExecution) else MiningExecution(
                item, target_block, target_tool, drop_item, drop_count)
            for item in targets)
        if not self.targets:
            raise ValueError("PlanExecutor needs at least one target")
        self.tunnel_tool = tunnel_tool
        self.mining_ticks = int(mining_ticks)
        self.tunnel_ticks = int(
            mining_ticks if tunnel_ticks is None else tunnel_ticks)
        self.stand_tolerance = float(stand_tolerance)
        self.max_clear_attempts = int(max_clear_attempts)
        self.max_mine_attempts = int(max_mine_attempts)
        self.max_move_attempts = int(max_move_attempts)
        self.max_pickup_attempts = int(max_pickup_attempts)
        self.health_loss_tolerance = float(health_loss_tolerance)
        self.veto_fluids = bool(veto_fluids)
        self.lighting = lighting
        self.lighting_immediate = bool(lighting_immediate)
        self.lighting_route_kinds = frozenset(
            str(value) for value in lighting_route_kinds)
        if item_recovery is not None and not callable(item_recovery):
            raise TypeError("item_recovery must be callable")
        self.item_recovery = item_recovery
        if break_declarer is not None and not callable(break_declarer):
            raise TypeError("break_declarer must be callable")
        self.break_declarer = break_declarer
        if self.mining_ticks < 1 or self.tunnel_ticks < 1:
            raise ValueError("mining tick counts must be positive")
        for value, label in (
            (self.max_clear_attempts, "max_clear_attempts"),
            (self.max_mine_attempts, "max_mine_attempts"),
            (self.max_move_attempts, "max_move_attempts"),
            (self.max_pickup_attempts, "max_pickup_attempts"),
        ):
            if value < 1:
                raise ValueError(f"{label} must be positive")
        if self.stand_tolerance <= 0:
            raise ValueError("stand_tolerance must be positive")
        if self.health_loss_tolerance < 0:
            raise ValueError("health_loss_tolerance cannot be negative")
        self._validate_plans()
        self._started = False
        self.done = False

    def _validate_plans(self):
        previous = None
        for execution in self.targets:
            plan = execution.plan
            stands = plan.route.stands
            if previous is not None and stands[0] != previous:
                raise ValueError(
                    "each target route must start at the preceding mining "
                    "stance")
            for left, right in zip(stands, stands[1:]):
                horizontal = (abs(left[0] - right[0])
                              + abs(left[2] - right[2]))
                if horizontal != 1 or abs(left[1] - right[1]) > 1:
                    raise ValueError(
                        "route stands must be cardinal one-block steps with "
                        "at most one block of vertical change")
                if plan.route.kind == "tunnel" and right[1] < left[1] and (
                        right[0], right[1] + 2, right[2]
                ) not in plan.route.reserved_cells:
                    raise ValueError(
                        "descending route steps need upper transition "
                        "clearance in reserved_cells")
                if plan.route.kind == "tunnel" and right[1] > left[1] and (
                        left[0], left[1] + 2, left[2]
                ) not in plan.route.reserved_cells:
                    raise ValueError(
                        "ascending route steps need origin jump headroom "
                        "in reserved_cells")
            mine_cells = frozenset(plan.route.mine_cells)
            if plan.route.kind == "surface" and mine_cells:
                raise ValueError("surface routes cannot contain mine_cells")
            if not mine_cells.issubset(plan.route.reserved_cells):
                raise ValueError(
                    "route mine_cells must belong to reserved body cells")
            previous = plan.mining_stance
        for index, execution in enumerate(self.targets):
            removed = {
                execution.plan.cell,
                *execution.plan.route.mine_cells,
            }
            future_supports = {
                _cell(cell)
                for future in self.targets[index:]
                for cell in future.plan.removal.protected_support_cells
            }
            conflict = removed.intersection(future_supports)
            if conflict:
                raise ValueError(
                    "a route/target removal conflicts with current or "
                    f"future protected support {sorted(conflict)[0]}")

    @property
    def target_index(self) -> int:
        return self._target_index

    @property
    def route_index(self) -> int:
        return self._route_index

    def diagnostics(self):
        """Return a compact, JSON-safe snapshot of live plan execution.

        Rejection ledgers must be useful without retaining a worker process or
        replaying a several-minute world.  Keep this deliberately bounded:
        the immutable full plans already live in provenance, while this view
        identifies the active leg and the last controls that affected it.
        """

        result = {
            "started": self._started,
            "done": self.done,
        }
        if not self._started:
            return result
        result.update({
            "target_index": self._target_index,
            "target_count": len(self.targets),
            "route_index": self._route_index,
            "move_attempts": self._move_attempts,
            "mine_attempts": self._mine_attempts,
            "mine_control_attempts": self._mine_control_attempts,
            "stance_recovery_attempts": self._stance_recovery_attempts,
            "soft_clear_attempts": {
                repr(cell): attempts
                for cell, attempts in self._soft_clear_attempts.items()
            },
            "recent_mine_control": tuple(self._mine_control_trace[-4:]),
        })
        if self.done or self._target_index >= len(self.targets):
            return result
        execution = self.targets[self._target_index]
        plan = execution.plan
        stands = plan.route.stands
        previous = (stands[self._route_index - 1]
                    if 0 < self._route_index <= len(stands) else None)
        destination = (stands[self._route_index]
                       if self._route_index < len(stands) else None)
        relevant_clear_cells = tuple(dict.fromkeys(
            cell
            for cell in plan.route.mine_cells
            if (destination is not None
                and (cell[0], cell[2]) ==
                    (destination[0], destination[2]))
        ))
        # Include all recent touched cells as well. Keys become strings so the
        # structure can pass directly through json.dumps and rejection JSONL.
        clear_trace = {
            repr(cell): tuple(entries[-4:])
            for cell, entries in self._clear_control_trace.items()
            if entries and (not relevant_clear_cells
                            or cell in relevant_clear_cells)
        }
        if not clear_trace:
            clear_trace = {
                repr(cell): tuple(entries[-4:])
                for cell, entries in list(
                    self._clear_control_trace.items())[-4:]
                if entries
            }
        result["active"] = {
            "target": plan.cell,
            "mining_stance": plan.mining_stance,
            "route_kind": plan.route.kind,
            "route_length": len(stands),
            "route_previous": previous,
            "route_destination": destination,
            "route_window": stands[max(0, self._route_index - 2):
                                   self._route_index + 3],
            "route_mine_cells": plan.route.mine_cells,
            "block": execution.block,
            "tool": execution.tool,
            "drop_item": execution.drop_item,
            "recent_clear_control": clear_trace,
        }
        return result

    def reset(self, env):
        """Bind the executor to the current live journal and player state."""

        self._started = True
        self.done = False
        self._event_start = len(env.journal)
        self._baseline_health = self._health(env)
        self._target_index = 0
        self._route_index = 0
        self._clear_attempts: dict[Cell, int] = {}
        self._clear_control_attempts: dict[Cell, int] = {}
        self._clear_control_trace: dict[Cell, list[dict[str, object]]] = {}
        self._mine_attempts = 0
        self._mine_control_attempts = 0
        self._mine_control_trace: list[dict[str, object]] = []
        self._move_attempts = 0
        self._stance_recovery_attempts = 0
        self._soft_clear_attempts: dict[Cell, int] = {}
        self._declared_extra_break_cells: set[Cell] = set()
        self._target_event_start = self._event_start
        self._target_item_baseline = 0
        self._pickup_attempts = 0
        self._pickup_navigation_attempts = 0
        self._last_pickup_target = None
        self._last_pickup_sneak = False
        self._target_entered = False
        self._seal_attempts: dict[Cell, int] = {}
        self._lighting_signature = None
        if self.lighting is not None:
            start = getattr(self.lighting, "start_route_lighting", None)
            if start is not None:
                start(env, immediate=self.lighting_immediate)
            remember = getattr(self.lighting, "remember_route_position", None)
            if remember is not None:
                remember(env)
        return self

    @staticmethod
    def _health(env):
        if "health" in env.obs:
            return float(env.obs["health"])
        for event in reversed(env.journal):
            if getattr(event, "name", None) == "vitals_changed":
                return float(event["health"])
        return None

    @staticmethod
    def _at(env, stand: Cell, tolerance: float) -> bool:
        return (math.hypot(
                    float(env.obs["x"]) - (stand[0] + 0.5),
                    float(env.obs["z"]) - (stand[2] + 0.5))
                <= tolerance
                and abs(float(env.obs["y"]) - stand[1]) <= 1.1)

    @staticmethod
    def _at_interaction_stance(
            env, stand: Cell, tolerance: float = 0.08) -> bool:
        """Require a centered, level pose before clearing the next leg.

        The ordinary stand envelope is intentionally wide enough to absorb
        walking collision jitter.  It is too wide for an irreversible click:
        residual movement can carry a player off the certified origin while a
        camera-only aiming action is being applied.  Mining starts only from
        this tighter envelope; safe on-leg poses are crouch-corrected below.
        """

        return (math.hypot(
                    float(env.obs["x"]) - (stand[0] + 0.5),
                    float(env.obs["z"]) - (stand[2] + 0.5))
                <= float(tolerance)
                and abs(float(env.obs["y"]) - stand[1]) <= 0.55)

    @staticmethod
    def _on_horizontal_leg(
            env, left: Cell, right: Cell, tolerance: float) -> bool:
        """Accept the x/z envelope of one certified cardinal route leg."""
        point = (float(env.obs["x"]), float(env.obs["z"]))
        start = (left[0] + 0.5, left[2] + 0.5)
        end = (right[0] + 0.5, right[2] + 0.5)
        dx, dz = end[0] - start[0], end[1] - start[1]
        length2 = dx * dx + dz * dz
        progress = ((point[0] - start[0]) * dx
                    + (point[1] - start[1]) * dz) / length2
        nearest = (start[0] + max(0.0, min(1.0, progress)) * dx,
                   start[1] + max(0.0, min(1.0, progress)) * dz)
        slack = float(tolerance) / math.sqrt(length2)
        return (-slack <= progress <= 1.0 + slack
                and math.dist(point, nearest) <= tolerance)

    @staticmethod
    def _on_leg(env, left: Cell, right: Cell, tolerance: float) -> bool:
        """Accept normal partial progress, but no unplanned lateral drift."""

        point = (float(env.obs["x"]), float(env.obs["z"]))
        start = (left[0] + 0.5, left[2] + 0.5)
        end = (right[0] + 0.5, right[2] + 0.5)
        dx, dz = end[0] - start[0], end[1] - start[1]
        length2 = dx * dx + dz * dz
        progress = ((point[0] - start[0]) * dx
                    + (point[1] - start[1]) * dz) / length2
        expected_y = left[1] + max(0.0, min(1.0, progress)) * (
            right[1] - left[1])
        return (PlanExecutor._on_horizontal_leg(
                    env, left, right, tolerance)
                and abs(float(env.obs["y"]) - expected_y) <= 1.25)

    @staticmethod
    def _at_route_destination(env, left: Cell, right: Cell,
                              tolerance: float) -> bool:
        """Require the destination level on vertical proof steps.

        The broad ordinary stand tolerance intentionally absorbs collision
        jitter, but it must not let a descending leg advance while the player
        remains exactly one block above its destination.
        """

        changing_level = left[1] != right[1]
        # A vertical leg is not complete merely because an observation caught
        # the player near the destination during a jump.  Advancing at
        # landing+0.2 lets the next action add a second jump and eventually
        # leaves the certified corridor.  Settled engine feet are integral to
        # floating-point noise, so require that level before advancing.
        vertical = 0.05 if changing_level else 1.1
        horizontal = min(tolerance, 0.4)
        return (math.hypot(
                    float(env.obs["x"]) - (right[0] + 0.5),
                    float(env.obs["z"]) - (right[2] + 0.5))
                <= horizontal
                and abs(float(env.obs["y"]) - right[1]) <= vertical)

    def _veto(self, code, reason, *, cells=(), target=None):
        if target is None:
            if self._started and not self.done:
                plan = self.targets[self._target_index].plan
                target = (plan.cell[0] + 0.5, plan.cell[2] + 0.5)
            else:
                target = (0.0, 0.0)
        return PolicyVeto(
            str(code), str(reason), tuple(map(float, target)),
            tuple(_cell(cell) for cell in cells))

    def _events(self, env, name, *, since=None):
        start = self._event_start if since is None else int(since)
        return [event for event in env.journal[start:]
                if getattr(event, "name", None) == name]

    def _broken_events(self, env, cell, *, since=None):
        cell = _cell(cell)
        return [event for event in self._events(
            env, "block_broken", since=since) if _event_cell(event) == cell]

    def _safety_veto(self, env):
        if bool(env.obs.get("dead", False)):
            return self._veto("player_died", "the live observation is dead")

        health = self._health(env)
        if (health is not None and self._baseline_health is not None
                and health < (self._baseline_health
                              - self.health_loss_tolerance)):
            return self._veto(
                "health_lost",
                f"health fell from {self._baseline_health:g} to {health:g}")
        for event in self._events(env, "vitals_changed"):
            if (float(event["health"])
                    < float(event["was_health"])
                    - self.health_loss_tolerance):
                return self._veto(
                    "health_lost",
                    "the engine journal records health loss during the plan")

        if self.veto_fluids:
            fluid_events = self._events(env, "player_fluid_state")
            for event in fluid_events:
                if (int(event.get("flags", 0)) & FLUID_BODY
                        and DEFAULT_BLOCK_RULES.is_fluid(
                            int(event.get("block", 0)))):
                    return self._veto(
                        "fluid_contact",
                        "the player collision body entered a fluid",
                        cells=(_event_cell(event),))
            # If execution begins while already submerged, its latest state
            # may precede ``_event_start``.  Check that current state too.
            for event in reversed(env.journal):
                if getattr(event, "name", None) != "player_fluid_state":
                    continue
                if (int(event.get("flags", 0)) & FLUID_BODY
                        and DEFAULT_BLOCK_RULES.is_fluid(
                            int(event.get("block", 0)))):
                    return self._veto(
                        "fluid_contact",
                        "the current player collision body is in a fluid",
                        cells=(_event_cell(event),))
                break

        x, y, z = (float(env.obs[key]) for key in ("x", "y", "z"))
        clearance = float(getattr(
            getattr(self.controller, "behaviors", None),
            "hazard_clearance_blocks", 0.0))
        hazard_ids = frozenset(getattr(self.controller, "hazard_ids", ()))
        for block, bx, by, bz in env.obs.get("blocks", ()):
            cell = int(bx), int(by), int(bz)
            if (int(block) in hazard_ids
                    and x - 0.3 - clearance <= cell[0] + 1
                    and x + 0.3 + clearance >= cell[0]
                    and y - clearance <= cell[1] + 1
                    and y + 1.8 + clearance >= cell[1]
                    and z - 0.3 - clearance <= cell[2] + 1
                    and z + 0.3 + clearance >= cell[2]):
                return self._veto(
                    "hazard_contact",
                    "a configured hazard intersects the player clearance",
                    cells=(cell,))
        return None

    def _support_veto(self, env):
        protected = {
            _cell(cell)
            for execution in self.targets[self._target_index:]
            for cell in execution.plan.removal.protected_support_cells
        }
        for event in self._events(env, "block_broken"):
            cell = _event_cell(event)
            if cell in protected:
                owners = tuple(
                    index for index, execution in enumerate(self.targets)
                    if cell in {
                        _cell(value) for value in
                        execution.plan.removal.protected_support_cells})
                return self._veto(
                    "protected_support_removed",
                    f"route support {cell} was removed during execution; "
                    f"current target index={self._target_index}, "
                    f"support owners={owners}, current target="
                    f"{self.targets[self._target_index].plan.cell}",
                    cells=(cell,))
        return None

    def _exact_break_veto(self, env):
        """Reject collateral mining outside the immutable world plan."""

        allowed = {
            _cell(cell)
            for execution in self.targets
            for cell in (
                execution.plan.cell, *execution.plan.route.mine_cells)
        }
        allowed.update(self._declared_extra_break_cells)
        for event in self._events(env, "block_broken"):
            cell = _event_cell(event)
            if cell not in allowed:
                active_target = (
                    self.targets[self._target_index].plan.cell
                    if self._target_index < len(self.targets) else None)
                broken_context = tuple(
                    (int(candidate.get("tick", -1)),
                     _event_cell(candidate),
                     int(candidate.get("block", 0)),
                     int(candidate.get("drop", 0)))
                    for candidate in self._events(env, "block_broken")[-8:])
                clear_control = {
                    known: tuple(entries[-2:])
                    for known, entries in self._clear_control_trace.items()
                    if entries
                }
                episode = getattr(self.controller, "_episode", None)
                return self._veto(
                    "unplanned_block_removed",
                    f"engine journal records collateral block removal at "
                    f"{cell}; it is not a declared tunnel or target cell; "
                    f"active_target={active_target}; "
                    f"broken_context={broken_context!r}; "
                    f"recent_target_control="
                    f"{self._mine_control_trace[-4:]!r}; "
                    f"recent_clear_control={clear_control!r}; "
                    f"attack_stop_trace="
                    f"{getattr(episode, 'attack_stop_trace', None)!r}",
                    cells=(cell,))
        return None

    def _live_removal_veto(self, env, execution: MiningExecution):
        target = execution.plan.cell
        for cell in execution.plan.removal.inspected_cells:
            if cell == target:
                continue
            if cell in execution.plan.removal.fluid_seal_cells:
                if self.controller.placed(
                        env, cell, execution.fluid_seal_item):
                    continue
                return self._veto(
                    "removal_fluid_seal_missing",
                    f"planned fluid seal at {cell} is not live-proven",
                    cells=(cell,))
            block = self.controller.observed_block(env, cell)
            if (block is not None
                    and (DEFAULT_BLOCK_RULES.is_fluid(block)
                         or DEFAULT_BLOCK_RULES.is_hazard(block))):
                return self._veto(
                    "removal_hazard_diverged",
                    f"live hazard/fluid at {cell} invalidates the snapshot "
                    "removal proof",
                    cells=(cell,))
        return None

    def _seal_fluids(self, env, execution):
        """Displace planner-approved water faces before breaking the ore."""

        target = execution.plan.cell
        for cell in execution.plan.removal.fluid_seal_cells:
            if self.controller.placed(
                    env, cell, execution.fluid_seal_item):
                continue
            attempts = self._seal_attempts.get(cell, 0)
            if attempts >= self.max_clear_attempts:
                return self._veto(
                    "fluid_seal_unverified",
                    f"no exact placement proved fluid seal {cell}",
                    cells=(cell,))
            result = self.controller.place(
                env, execution.fluid_seal_item, cell, target,
                block=execution.fluid_seal_item)
            if isinstance(result, PolicyVeto):
                return result
            if not result:
                return self._veto(
                    "fluid_seal_item_unavailable",
                    f"no {execution.fluid_seal_item!r} is visible for the "
                    f"planned fluid seal at {cell}", cells=(cell,))
            self._seal_attempts[cell] = attempts + 1
            return result
        return None

    def _observe_lighting(self, env):
        if self.lighting is None:
            return
        signature = (
            env.obs.get("tick"), len(env.journal),
            float(env.obs["x"]), float(env.obs["y"]), float(env.obs["z"]))
        if signature == self._lighting_signature:
            return
        self._lighting_signature = signature
        observe = getattr(self.lighting, "observe_route_travel", None)
        if observe is not None:
            observe(env)
        remember = getattr(self.lighting, "remember_route_position", None)
        if remember is not None:
            remember(env)

    def _lighting_actions(self, env, kind):
        if self.lighting is None or kind not in self.lighting_route_kinds:
            return None
        self._observe_lighting(env)
        hook = getattr(self.lighting, "light_route", None)
        if hook is None and callable(self.lighting):
            hook = self.lighting
        if hook is None:
            raise TypeError(
                "lighting must be callable or expose light_route(env)")
        result = hook(env)
        if result is not None:
            return result
        overdue = getattr(self.lighting, "route_light_overdue", None)
        if overdue is not None and overdue():
            debug_hook = getattr(self.lighting, "route_light_debug", None)
            debug = None if debug_hook is None else debug_hook(env)
            return self._veto(
                "route_light_overdue",
                "no verified wall light was placed before the configured "
                f"route distance; lighting={debug!r}")
        return None

    def _movement_lookahead(self, stands, route):
        """Coalesce safe straight proof cells into one VLM-sized move."""

        first = self._route_index
        if first >= len(stands):
            return first
        origin = stands[first - 1]
        first_cell = stands[first]
        direction = (first_cell[0] - origin[0],
                     first_cell[1] - origin[1],
                     first_cell[2] - origin[2])
        if direction[1] != 0:
            return first
        # The pinned survival physics retains forward momentum after a held
        # key is released. Its tick-to-distance calibration is reliable for
        # one-cell closed-loop legs, but a nominal two-block hold can coast
        # beyond two later certified stands before the next observation.
        # Keep route synchronization one cell at a time. The action itself
        # still bundles turning, movement, and its normal pause in one turn.
        maximum = min(1.0, float(getattr(
            getattr(self.controller, "movement", None), "max_blocks", 1.0)))
        mine_cells = frozenset(route.mine_cells)
        selected = first
        for index in range(first + 1, len(stands)):
            cell = stands[index]
            previous = stands[index - 1]
            delta = (cell[0] - previous[0], cell[1] - previous[1],
                     cell[2] - previous[2])
            if delta != direction:
                break
            if math.dist(origin, cell) > maximum + 1e-9:
                break
            body = (cell, (cell[0], cell[1] + 1, cell[2]))
            if any(item in mine_cells for item in body):
                break
            selected = index
        return selected

    def _proved_later_stand(self, env, stands, route):
        """Resynchronize after momentum lands on a later proved route cell."""

        mine_cells = frozenset(route.mine_cells)
        for index in range(
                self._route_index + 1,
                min(len(stands), self._route_index + 3)):
            if not self._at_route_destination(
                    env, stands[index - 1], stands[index],
                    min(0.4, self.stand_tolerance)):
                continue
            crossed = stands[self._route_index:index + 1]
            required = set()
            for offset, stand in enumerate(crossed):
                body = (stand, (stand[0], stand[1] + 1, stand[2]))
                if (offset > 0
                        and stand[1] < crossed[offset - 1][1]):
                    body += ((stand[0], stand[1] + 2, stand[2]),)
                required.update(cell for cell in body if cell in mine_cells)
            if all(self._broken_events(env, cell) for cell in required):
                return index
        return None

    def _enter_target(self, env, execution):
        if self._target_entered:
            return
        self._target_entered = True
        self._target_event_start = len(env.journal)
        exact_inventory = hasattr(
            self.controller, "inventory_item_count") and all(
            env.obs.get(name) is not None
            and len(env.obs[name]) == 36
            for name in ("inventory_ids", "inventory_counts",
                         "inventory_meta"))
        self._target_inventory_exact = exact_inventory
        if not execution.drop_count:
            # Controllers for no-drop plans need not expose inventory
            # accounting.  Resolve that optional protocol only when the plan
            # actually requires collection proof.
            self._target_item_baseline = 0
        else:
            counter = (self.controller.inventory_item_count
                       if exact_inventory
                       else self.controller.item_count)
            self._target_item_baseline = counter(env, execution.drop_item)
        self._pickup_attempts = 0
        self._pickup_navigation_attempts = 0

    def _recover_item(self, env, item):
        if item is None or self.item_recovery is None:
            return None
        return self.item_recovery(env, item)

    def _target_break(self, env, execution):
        events = self._broken_events(
            env, execution.plan.cell, since=self._target_event_start)
        if not events:
            return None
        if execution.block is not None:
            expected = frozenset(self.controller._block_ids(execution.block))
            if not any(int(event.get("block", 0)) in expected
                       for event in events):
                return self._veto(
                    "target_block_mismatch",
                    f"the journal reports a different block removed at "
                    f"{execution.plan.cell}",
                    cells=(execution.plan.cell,))
        if execution.block_metadata is not None:
            expected_metadata = frozenset(execution.block_metadata)
            if not any(int(event.get("meta", 0)) in expected_metadata
                       for event in events):
                return self._veto(
                    "target_block_state_mismatch",
                    "the journal reports unexpected metadata for the "
                    f"destroyed block at {execution.plan.cell}; expected="
                    f"{tuple(sorted(expected_metadata))}, observed="
                    f"{tuple(int(event.get('meta', 0)) for event in events)}",
                    cells=(execution.plan.cell,))
        if execution.drop_count:
            expected_drop = frozenset(
                self.controller._item_ids(execution.drop_item))
            if not any(int(event.get("drop", 0)) in expected_drop
                       for event in events):
                reported = tuple({
                    "tick": int(getattr(
                        event, "tick", event.get("tick", -1))),
                    "block": int(event.get("block", 0)),
                    "meta": int(event.get("meta", 0)),
                    "drop": int(event.get("drop", 0)),
                    "tool": int(event.get("tool", 0)),
                } for event in events)
                selected_slot = int(env.obs.get("hotbar_sel", -1))
                hotbar = env.obs.get("hotbar_ids", ())
                selected_item = (
                    int(hotbar[selected_slot])
                    if 0 <= selected_slot < len(hotbar) else None)
                planned_tool_ids = (
                    self.controller._item_ids(execution.tool)
                    if execution.tool is not None else ())
                item_events = tuple(
                    {
                        "event": getattr(event, "name", "unknown"),
                        "tick": int(getattr(
                            event, "tick", event.get("tick", -1))),
                        "item": int(event.get("item", 0)),
                        "count": int(event.get("count", 0)),
                        "meta": int(event.get("meta", 0)),
                        "cause": event.get("cause"),
                    }
                    for event in env.journal[self._target_event_start:]
                    if getattr(event, "name", None) in (
                        "item_dropped", "item_picked_up")
                )
                hotbar_counts = tuple(int(value) for value in env.obs.get(
                    "hotbar_counts", ()))
                hotbar_meta = tuple(int(value) for value in env.obs.get(
                    "hotbar_meta", (0,) * len(hotbar)))
                hotbar_state = tuple(
                    (index, int(item),
                     hotbar_counts[index]
                     if index < len(hotbar_counts) else 0,
                     hotbar_meta[index] if index < len(hotbar_meta) else 0)
                    for index, item in enumerate(hotbar)
                    if int(item) != 0
                )
                inventory_counts = {
                    str(name): int(count)
                    for name, count in env.obs.get(
                        "inv_counts", {}).items() if int(count)
                }
                return self._veto(
                    "target_drop_mismatch",
                    "the exact block break did not produce the expected "
                    "drop (usually a missing or wrong tool); "
                    f"expected_drop_ids={tuple(sorted(expected_drop))}, "
                    f"events={reported!r}, selected="
                    f"({selected_slot}, {selected_item}), "
                    f"planned_tool={execution.tool!r}, "
                    f"planned_tool_ids={planned_tool_ids}, "
                    f"item_events={item_events!r}, "
                    f"hotbar={hotbar_state!r}, "
                    f"inventory={inventory_counts!r}",
                    cells=(execution.plan.cell,))
        return True

    def _clear_soft_target_occluder(self, env, execution):
        """Clear one exact selectable plant from the proved target ray."""

        if self.break_declarer is None:
            return None
        found, target = self.controller.click_target(env)
        if target is None or int(found) not in _SOFT_RAY_OBSTRUCTIONS:
            return None
        target = _cell(target)
        plan = execution.plan
        if (target == plan.cell
                or target in frozenset(plan.removal.protected_support_cells)
                or math.dist(target, plan.cell) > 2.5):
            return None
        attempts = self._soft_clear_attempts.get(target, 0)
        if attempts >= self.max_clear_attempts:
            return self._veto(
                "target_soft_occluder_clear_exhausted",
                f"replaceable ray obstruction {target} did not clear",
                cells=(target, plan.cell))
        self._declared_extra_break_cells.add(target)
        self.break_declarer((target,))
        self._soft_clear_attempts[target] = attempts + 1
        return [{"action": "left_click_hold", "ticks": 2}]

    def _clear_soft_route_occluder(self, env, body, requested):
        """Clear selectable vegetation from one certified route body."""

        if self.break_declarer is None:
            return None
        found, target = self.controller.click_target(env)
        if target is None or int(found) not in _SOFT_RAY_OBSTRUCTIONS:
            return None
        target = _cell(target)
        if target == _cell(requested) or target not in frozenset(body):
            return None
        attempts = self._soft_clear_attempts.get(target, 0)
        if attempts >= self.max_clear_attempts:
            return self._veto(
                "route_soft_occluder_clear_exhausted",
                f"replaceable route obstruction {target} did not clear",
                cells=(target, requested))
        self._declared_extra_break_cells.add(target)
        self.break_declarer((target,))
        self._soft_clear_attempts[target] = attempts + 1
        return [{"action": "left_click_hold", "ticks": 2}]

    def _drop_collected(self, env, execution):
        if not execution.drop_count:
            return True
        wanted = frozenset(self.controller._item_ids(execution.drop_item))
        picked_up = sum(
            int(event.get("count", 1))
            for event in self._events(
                env, "item_picked_up", since=self._target_event_start)
            if int(event.get("item", 0)) in wanted)
        if getattr(self, "_target_inventory_exact", False):
            current = self.controller.inventory_item_count(
                env, execution.drop_item)
            return (current - self._target_item_baseline
                    >= execution.drop_count)
        current = self.controller.item_count(env, execution.drop_item)
        return (picked_up >= execution.drop_count
                or current - self._target_item_baseline
                >= execution.drop_count)

    def _advance_target(self):
        self._target_index += 1
        self._route_index = 0
        self._clear_attempts.clear()
        self._clear_control_attempts.clear()
        self._clear_control_trace.clear()
        self._mine_attempts = 0
        self._mine_control_attempts = 0
        self._mine_control_trace.clear()
        self._move_attempts = 0
        self._stance_recovery_attempts = 0
        self._target_entered = False

    def actions(self, env):
        """Return the next bounded action list, veto, or ``None`` when done."""

        if not self._started:
            raise RuntimeError("PlanExecutor.reset(env) must be called first")
        if self.done:
            return None

        veto = (self._safety_veto(env) or self._support_veto(env)
                or self._exact_break_veto(env))
        if veto is not None:
            return veto

        while self._target_index < len(self.targets):
            execution = self.targets[self._target_index]
            plan = execution.plan
            route = plan.route
            stands = route.stands
            self._enter_target(env, execution)

            if self._route_index == 0:
                if not self._at_interaction_stance(env, stands[0]):
                    recover = getattr(
                        self.controller, "recover_route_origin", None)
                    if recover is not None:
                        result = recover(env, stands[0])
                        if result is not None:
                            if self._move_attempts >= self.max_move_attempts:
                                return self._veto(
                                    "route_origin_recovery_exhausted",
                                    "bounded recovery did not reach exact "
                                    f"route origin {stands[0]}",
                                    cells=(stands[0],))
                            self._move_attempts += 1
                            return result
                    return self._veto(
                        "route_origin_diverged",
                        f"live player is not at proved route origin "
                        f"{stands[0]}", cells=(stands[0],))
                self._route_index = 1
                self._move_attempts = 0

            if self._route_index < len(stands):
                previous = stands[self._route_index - 1]
                destination_index = self._movement_lookahead(stands, route)
                destination = stands[destination_index]
                if self._at_route_destination(
                        env, previous, destination, self.stand_tolerance):
                    self._route_index = destination_index + 1
                    self._move_attempts = 0
                    continue
                later = self._proved_later_stand(env, stands, route)
                if later is not None:
                    self._route_index = later + 1
                    self._move_attempts = 0
                    continue
                body = (
                    destination,
                    (destination[0], destination[1] + 1,
                     destination[2]),
                )
                if destination[1] < previous[1]:
                    body += ((destination[0], destination[1] + 2,
                              destination[2]),)
                elif destination[1] > previous[1]:
                    body += ((previous[0], previous[1] + 2,
                              previous[2]),)
                # Clear the first selectable blocker before farther cells at
                # the same height. In an ascending transition the origin jump
                # headroom can occlude the destination head cell; a lexical
                # tie-break repeatedly aimed through the nearer solid voxel.
                eye = (previous[0] + 0.5, previous[1] + 1.62,
                       previous[2] + 0.5)
                required = tuple(sorted((
                    cell for cell in body
                    if cell in frozenset(route.mine_cells)),
                    key=lambda cell: (
                        round(math.dist(
                            eye, (cell[0] + 0.5, cell[1] + 0.5,
                                  cell[2] + 0.5)), 12),
                        -cell[1], cell)))

                # A one-block jump can be horizontally centered over its
                # destination while the observation still catches the player
                # near the 1.25-block apex. That pose is outside the ordinary
                # interpolated leg envelope, but it has not left the certified
                # transition: the destination body and the origin's jump
                # headroom are all part of ``body`` above. Wait without adding
                # momentum only when the player is centered over the proved
                # landing, is at a plausible vanilla jump height, and every
                # snapshot-solid clearance voxel has an exact break event.
                horizontal_error = math.hypot(
                    float(env.obs["x"]) - (destination[0] + 0.5),
                    float(env.obs["z"]) - (destination[2] + 0.5),
                )
                horizontal_arrived = (
                    horizontal_error <= min(0.4, self.stand_tolerance))
                height_above_landing = (
                    float(env.obs["y"]) - destination[1])
                ascending_clear = (
                    destination[1] > previous[1]
                    and horizontal_arrived
                    and all(self._broken_events(env, cell)
                            for cell in required)
                )
                # A raised block immediately beside the destination can catch
                # the edge of the player's collision body after a jump.  The
                # live pose then has the exact next-block y level but remains
                # visibly off-center over the certified landing (for example,
                # y=landing+1 and 0.38 blocks from center).  Waiting cannot
                # resolve that stable perch.  Recenter *within the same proved
                # destination cell*, without jumping, so gravity can complete
                # the landing.  This is deliberately distinct from accepting
                # the pose: the route advances only after a later observation
                # proves the destination level.
                ascending_perched = (
                    ascending_clear
                    and 0.95 <= height_above_landing <= 1.05
                    and horizontal_error > min(0.08, self.stand_tolerance)
                )
                if ascending_perched:
                    if self._move_attempts >= self.max_move_attempts:
                        return self._veto(
                            "route_vertical_settle_exhausted",
                            f"player did not settle onto ascending stand "
                            f"{destination}; live position is "
                            f"({float(env.obs['x']):.3f}, "
                            f"{float(env.obs['y']):.3f}, "
                            f"{float(env.obs['z']):.3f})",
                            cells=(destination,))
                    result = self.controller.face_and_move_to(
                        env, destination[0] + 0.5, destination[2] + 0.5,
                        tolerance=min(0.05, self.stand_tolerance),
                        sequential_hop=True,
                        step_up=False,
                        allow_jump=False,
                        movement_cap_blocks=0.3)
                    if isinstance(result, PolicyVeto):
                        return result
                    if not result:
                        return self._veto(
                            "route_move_unavailable",
                            f"controller produced no recenter move over "
                            f"planned ascending stand {destination}",
                            cells=(destination,))
                    self._move_attempts += 1
                    return result

                ascending_airborne = (
                    ascending_clear
                    and 0.05 < height_above_landing <= 1.35
                )
                if ascending_airborne:
                    if self._move_attempts >= self.max_move_attempts:
                        return self._veto(
                            "route_vertical_settle_exhausted",
                            f"player did not settle onto ascending stand "
                            f"{destination}; live position is "
                            f"({float(env.obs['x']):.3f}, "
                            f"{float(env.obs['y']):.3f}, "
                            f"{float(env.obs['z']):.3f})",
                            cells=(destination,))
                    self._move_attempts += 1
                    return [{"action": "wait", "ticks": 2}]

                vertical_landed_offcenter = (
                    destination[1] != previous[1]
                    and abs(float(env.obs["y"]) - destination[1]) <= 0.05
                    and horizontal_error > min(0.05, self.stand_tolerance)
                    and self._on_horizontal_leg(
                        env, previous, destination,
                        self.stand_tolerance))
                if vertical_landed_offcenter:
                    if self._move_attempts >= self.max_move_attempts:
                        return self._veto(
                            "route_vertical_recenter_exhausted",
                            f"player landed outside the center of vertical "
                            f"stand {destination}", cells=(destination,))
                    result = self.controller.face_and_move_to(
                        env, destination[0] + 0.5,
                        destination[2] + 0.5,
                        tolerance=0.05, sequential_hop=True,
                        step_up=False, allow_jump=False,
                        movement_cap_blocks=0.3)
                    if isinstance(result, PolicyVeto):
                        return result
                    if not result:
                        return self._veto(
                            "route_vertical_recenter_unavailable",
                            f"controller produced no recenter move over "
                            f"vertical stand {destination}",
                            cells=(destination,))
                    self._move_attempts += 1
                    return result

                flat_height = float(env.obs["y"]) - previous[1]
                flat_airborne = (
                    destination[1] == previous[1]
                    and 0.05 < flat_height <= 1.35
                    and self._on_horizontal_leg(
                        env, previous, destination,
                        self.stand_tolerance)
                )
                if flat_airborne:
                    if self._move_attempts >= self.max_move_attempts:
                        return self._veto(
                            "route_vertical_settle_exhausted",
                            f"player did not settle onto level route leg "
                            f"{previous} -> {destination}",
                            cells=(previous, destination))
                    self._move_attempts += 1
                    return [{"action": "wait", "ticks": 2}]

                if not self._on_leg(
                        env, previous, destination, self.stand_tolerance):
                    return self._veto(
                        "route_position_diverged",
                        f"live player left planned leg {previous} -> "
                        f"{destination} from "
                        f"({float(env.obs['x']):.3f}, "
                        f"{float(env.obs['y']):.3f}, "
                        f"{float(env.obs['z']):.3f})",
                        cells=(previous, destination))
                drop_is_safe = getattr(
                    self.controller, "drop_is_safe", None)
                if (drop_is_safe is not None
                        and not drop_is_safe(previous[1], destination[1])):
                    return self._veto(
                        "route_drop_rejected",
                        f"planned drop {previous[1] - destination[1]:g} "
                        "exceeds configured safe descent",
                        cells=(previous, destination))

                for cell in required:
                    if self._broken_events(env, cell):
                        continue
                    attempts = self._clear_attempts.get(cell, 0)
                    if attempts >= self.max_clear_attempts:
                        trace = self._clear_control_trace.get(cell, [])[-4:]
                        return self._veto(
                            "tunnel_clear_unverified",
                            f"no exact block_broken event proved clearance "
                            f"of tunnel cell {cell}; "
                            f"recent_control={trace!r}", cells=(cell,))
                    controls = self._clear_control_attempts.get(cell, 0)
                    if controls >= self.max_clear_attempts * 4:
                        trace = self._clear_control_trace.get(cell, [])[-4:]
                        return self._veto(
                            "tunnel_clear_control_unverified",
                            f"bounded aim/recenter actions never produced "
                            f"an exact attack on tunnel cell {cell}; "
                            f"recent_control={trace!r}",
                            cells=(cell,))
                    if not self._at_interaction_stance(env, previous):
                        trace = self._clear_control_trace.get(cell, [])[-4:]
                        # A bounded route move can reach a stand with residual
                        # momentum.  The next camera-only aim tick may then
                        # carry the player slightly into the still-solid
                        # destination column.  That pose is recoverable only
                        # while it remains on the same certified leg and level;
                        # crouch-correct back to the exact mining origin before
                        # selecting, aiming at, or attacking any block.
                        recoverable = (
                            self._on_leg(
                                env, previous, destination,
                                self.stand_tolerance)
                            and abs(float(env.obs["y"])
                                    - previous[1]) <= 0.55
                        )
                        recenter = getattr(
                            self.controller,
                            "recenter_interaction_stance", None)
                        if recoverable and recenter is not None:
                            result = recenter(
                                env, previous[0] + 0.5,
                                previous[2] + 0.5, tolerance=0.05)
                            if result:
                                self._clear_control_trace.setdefault(
                                    cell, []).append({
                                        "pose": tuple(round(
                                            float(env.obs[key]), 3)
                                            for key in ("x", "y", "z")),
                                        "recenter": previous,
                                        "actions": tuple(
                                            action.get("action")
                                            for action in result),
                                    })
                                self._clear_control_attempts[cell] = (
                                    controls + 1)
                                return result
                        return self._veto(
                            "tunnel_clear_stance_diverged",
                            f"cannot clear {cell} after leaving proved "
                            f"stance {previous}; live position is "
                            f"({float(env.obs['x']):.3f}, "
                            f"{float(env.obs['y']):.3f}, "
                            f"{float(env.obs['z']):.3f}); "
                            f"recent_control={trace!r}",
                            cells=(previous, cell))
                    light = self._lighting_actions(env, route.kind)
                    if light is not None:
                        return light
                    recovered = self._recover_item(env, self.tunnel_tool)
                    if recovered:
                        return recovered
                    hold_ticks = self.tunnel_ticks
                    resolver = getattr(
                        self.controller, "mining_hold_ticks", None)
                    if resolver is not None:
                        hold_ticks = resolver(
                            env, cell, tool=self.tunnel_tool,
                            fallback=self.tunnel_ticks)
                    result = self.controller.mine_exact(
                        env, cell, tool=self.tunnel_tool,
                        ticks=hold_ticks,
                        stand=(previous[0] + 0.5,
                               previous[2] + 0.5))
                    if isinstance(result, PolicyVeto):
                        if result.code == "mine_target_unproven":
                            soft_clear = self._clear_soft_route_occluder(
                                env, body, cell)
                            if soft_clear is not None:
                                return soft_clear
                        occluder = _live_required_occluder(
                            self.controller, env, cell, required)
                        if (result.code == "mine_target_unproven"
                                and occluder is not None):
                            occluder_attempts = self._clear_attempts.get(
                                occluder, 0)
                            if occluder_attempts < self.max_clear_attempts:
                                occluder_ticks = self.tunnel_ticks
                                if resolver is not None:
                                    occluder_ticks = resolver(
                                        env, occluder,
                                        tool=self.tunnel_tool,
                                        fallback=self.tunnel_ticks)
                                self._clear_attempts[occluder] = (
                                    occluder_attempts + 1)
                                self._clear_control_trace.setdefault(
                                    occluder, []).append({
                                        "pose": tuple(round(
                                            float(env.obs[key]), 3)
                                            for key in ("x", "y", "z")),
                                        "reoccupied": occluder,
                                        "requested": cell,
                                        "actions": ("left_click_hold",),
                                    })
                                return [{
                                    "action": "left_click_hold",
                                    "ticks": int(occluder_ticks),
                                }]
                        return result
                    if not result:
                        selected_slot = int(env.obs.get("hotbar_sel", -1))
                        hotbar = env.obs.get("hotbar_ids", ())
                        selected_item = (
                            int(hotbar[selected_slot])
                            if 0 <= selected_slot < len(hotbar) else None)
                        return self._veto(
                            "tunnel_clear_unavailable",
                            f"controller could not clear required tunnel "
                            f"cell {cell}; observed_block="
                            f"{self.controller.observed_block(env, cell)!r}, "
                            f"selected=({selected_slot}, {selected_item}), "
                            f"tool={self.tunnel_tool!r}, "
                            f"break_events={len(self._broken_events(env, cell))}",
                            cells=(cell,))
                    click_target = getattr(
                        self.controller, "click_target", None)
                    found, click_cell = ((None, None)
                                         if click_target is None
                                         else click_target(env))
                    selected_slot = int(env.obs.get("hotbar_sel", 0))
                    hotbar = env.obs.get("hotbar_ids", ())
                    selected_item = (int(hotbar[selected_slot])
                                     if 0 <= selected_slot < len(hotbar)
                                     else None)
                    self._clear_control_trace.setdefault(cell, []).append({
                        "pose": tuple(round(float(env.obs[key]), 3)
                                      for key in ("x", "y", "z")),
                        "yaw": round(float(env.obs.get("yaw", 0.0)), 3),
                        "pitch": round(float(
                            env.obs.get("pitch", 0.0)), 3),
                        "click": (found, click_cell),
                        "observed_block": self.controller.observed_block(
                            env, cell),
                        "tool": self.tunnel_tool,
                        "tool_ids": self.controller._item_ids(
                            self.tunnel_tool) if self.tunnel_tool else (),
                        "selected": (selected_slot, selected_item),
                        "actions": tuple(action.get("action")
                                         for action in result),
                    })
                    self._clear_control_attempts[cell] = controls + 1
                    if any(action.get("action") == "left_click_hold"
                           for action in result):
                        self._clear_attempts[cell] = attempts + 1
                    return result

                # Only wait for gravity after every snapshot-proved solid in
                # the destination's body column has an exact break event.
                # Otherwise a descending tunnel leg can walk onto the top of
                # its still-solid feet block and wait forever one level high.
                horizontal_arrived = math.hypot(
                    float(env.obs["x"]) - (destination[0] + 0.5),
                    float(env.obs["z"]) - (destination[2] + 0.5),
                ) <= min(0.08, self.stand_tolerance)
                if (horizontal_arrived
                        and destination[1] < previous[1]
                        and float(env.obs["y"]) > destination[1] + 0.05):
                    if self._move_attempts >= self.max_move_attempts:
                        return self._veto(
                            "route_vertical_settle_exhausted",
                            f"player did not settle onto descending stand "
                            f"{destination}; live position is "
                            f"({float(env.obs['x']):.3f}, "
                            f"{float(env.obs['y']):.3f}, "
                            f"{float(env.obs['z']):.3f})",
                            cells=(destination,))
                    self._move_attempts += 1
                    return [{"action": "wait", "ticks": 2}]

                light = self._lighting_actions(env, route.kind)
                if light is not None:
                    return light
                if self._move_attempts >= self.max_move_attempts:
                    return self._veto(
                        "route_move_exhausted",
                        f"player did not reach planned stand {destination}; "
                        f"live position is ({float(env.obs['x']):.3f}, "
                        f"{float(env.obs['y']):.3f}, "
                        f"{float(env.obs['z']):.3f})",
                        cells=(destination,))
                result = self.controller.face_and_move_to(
                    env, destination[0] + 0.5, destination[2] + 0.5,
                    # A level route may stop near the cell center. Vertical
                    # steps must cross the block boundary far enough for
                    # collision/gravity to move the player to the proved
                    # destination level; a 0.55 horizontal tolerance can
                    # leave them balanced on the preceding block.
                    tolerance=min(
                        0.05 if destination[1] != previous[1] else 0.4,
                        self.stand_tolerance),
                    sequential_hop=True,
                    step_up=destination[1] > previous[1],
                    # A generic stuck-recovery jump is useful on flat and
                    # ascending terrain. On a proved descending tunnel step
                    # it can repeatedly bounce the player on the old ledge,
                    # preventing the cleared drop from ever settling.
                    allow_jump=destination[1] >= previous[1],
                    # A normal stride can carry the player through a
                    # one-block descent or jump past the top of an ascent.
                    # Keep every vertical transition deliberately
                    # fine-grained so the executor observes and certifies its
                    # landing before continuing. Horizontal legs still use
                    # the configured VLM-scale movement profile and lookahead
                    # above.
                    movement_cap_blocks=(
                        0.3 if destination[1] != previous[1] else None))
                if isinstance(result, PolicyVeto):
                    return result
                if not result:
                    return self._veto(
                        "route_move_unavailable",
                        f"controller produced no move toward planned stand "
                        f"{destination}", cells=(destination,))
                self._move_attempts += 1
                return result

            broken = self._target_break(env, execution)
            if isinstance(broken, PolicyVeto):
                return broken
            if not broken:
                if not self._at_interaction_stance(
                        env, plan.mining_stance):
                    live_cell = (
                        math.floor(float(env.obs["x"])),
                        math.floor(float(env.obs["y"]) + 0.05),
                        math.floor(float(env.obs["z"])),
                    )
                    same_body_column = (
                        (live_cell[0], live_cell[2]) ==
                        (plan.mining_stance[0], plan.mining_stance[2])
                        and abs(float(env.obs["y"])
                                - plan.mining_stance[1]) <= 0.55)
                    horizontal = math.hypot(
                        float(env.obs["x"])
                        - (plan.mining_stance[0] + 0.5),
                        float(env.obs["z"])
                        - (plan.mining_stance[2] + 0.5))
                    adjacent_boundary = (
                        not same_body_column
                        and abs(float(env.obs["y"])
                                - plan.mining_stance[1]) <= 0.55
                        and horizontal <= 0.65
                        and max(abs(live_cell[0]
                                    - plan.mining_stance[0]),
                                abs(live_cell[2]
                                    - plan.mining_stance[2])) <= 1)
                    if adjacent_boundary:
                        if (self._stance_recovery_attempts
                                >= self.max_move_attempts):
                            return self._veto(
                                "mining_stance_recovery_exhausted",
                                "bounded boundary correction did not center "
                                f"the proved mining stance "
                                f"{plan.mining_stance}",
                                cells=(plan.mining_stance,))
                        result = self.controller.face_and_move_to(
                            env, plan.mining_stance[0] + 0.5,
                            plan.mining_stance[2] + 0.5,
                            tolerance=0.05, allow_jump=False,
                            movement_cap_blocks=0.3)
                        if result:
                            self._stance_recovery_attempts += 1
                            return result
                    recenter = getattr(
                        self.controller,
                        "recenter_interaction_stance", None)
                    if same_body_column and recenter is not None:
                        if (self._stance_recovery_attempts
                                >= self.max_move_attempts):
                            return self._veto(
                                "mining_stance_recovery_exhausted",
                                "bounded correction did not center the "
                                f"proved mining stance {plan.mining_stance}",
                                cells=(plan.mining_stance,))
                        result = recenter(
                            env, plan.mining_stance[0] + 0.5,
                            plan.mining_stance[2] + 0.5, tolerance=0.05)
                        if result:
                            self._stance_recovery_attempts += 1
                            return result
                    return self._veto(
                        "mining_stance_diverged",
                        f"live player is not at proved mining stance "
                        f"{plan.mining_stance}",
                        cells=(plan.mining_stance,))
                seal = self._seal_fluids(env, execution)
                if seal is not None:
                    return seal
                live_veto = self._live_removal_veto(env, execution)
                if live_veto is not None:
                    return live_veto
                light = self._lighting_actions(env, route.kind)
                if light is not None:
                    return light
                recovered = self._recover_item(env, execution.tool)
                if recovered:
                    return recovered
                if self._mine_attempts >= self.max_mine_attempts:
                    return self._veto(
                        "target_break_unverified",
                        f"no exact block_broken event proved removal of "
                        f"target {plan.cell}", cells=(plan.cell,))
                if self._mine_control_attempts >= self.max_mine_attempts * 4:
                    return self._veto(
                        "target_mine_control_unverified",
                        "bounded aim/recenter actions never produced an "
                        f"exact attack on target {plan.cell}; "
                        f"recent_control="
                        f"{self._mine_control_trace[-4:]!r}",
                        cells=(plan.cell,))
                result = self.controller.mine_exact(
                    env, plan.cell, tool=execution.tool,
                    ticks=self.mining_ticks,
                    block=execution.block,
                    stand=(plan.mining_stance[0] + 0.5,
                           plan.mining_stance[2] + 0.5))
                if isinstance(result, PolicyVeto):
                    if result.code == "mine_target_unproven":
                        clear = self._clear_soft_target_occluder(
                            env, execution)
                        if clear is not None:
                            return clear
                    return result
                if not result:
                    return self._veto(
                        "target_mine_unavailable",
                        f"controller could not mine planned target "
                        f"{plan.cell}", cells=(plan.cell,))
                click_target = getattr(
                    self.controller, "click_target", None)
                found, click_cell = ((None, None)
                                     if click_target is None
                                     else click_target(env))
                selected_slot = int(env.obs.get("hotbar_sel", 0))
                hotbar = env.obs.get("hotbar_ids", ())
                selected_item = (int(hotbar[selected_slot])
                                 if 0 <= selected_slot < len(hotbar)
                                 else None)
                self._mine_control_trace.append({
                    "pose": tuple(round(float(env.obs[key]), 3)
                                  for key in ("x", "y", "z")),
                    "yaw": round(float(env.obs.get("yaw", 0.0)), 3),
                    "pitch": round(float(env.obs.get("pitch", 0.0)), 3),
                    "click": (found, click_cell),
                    "observed_block": self.controller.observed_block(
                        env, plan.cell),
                    "expected_blocks": tuple(sorted(
                        self.controller._block_ids(execution.block)))
                    if execution.block is not None else (),
                    "tool": execution.tool,
                    "tool_ids": self.controller._item_ids(execution.tool)
                    if execution.tool else (),
                    "selected": (selected_slot, selected_item),
                    "actions": tuple(action.get("action")
                                     for action in result),
                })
                self._mine_control_attempts += 1
                if any(action.get("action") == "left_click_hold"
                       for action in result):
                    self._mine_attempts += 1
                return result

            if not self._drop_collected(env, execution):
                if self._pickup_attempts >= self.max_pickup_attempts:
                    position = tuple(
                        round(float(env.obs[key]), 3)
                        for key in ("x", "y", "z"))
                    pickup_geometry = getattr(
                        self.controller, "_drop_pickup_trace", None)
                    pickup_history = getattr(
                        self.controller, "_drop_pickup_history", ())[-8:]
                    return self._veto(
                        "target_drop_uncollected",
                        f"journal/inventory never verified collection of "
                        f"{execution.drop_count} {execution.drop_item!r}; "
                        f"position={position}, last pickup target="
                        f"{self._last_pickup_target}, sneak="
                        f"{self._last_pickup_sneak}; pickup_geometry="
                        f"{pickup_geometry!r}; pickup_history="
                        f"{pickup_history!r}",
                        cells=(plan.cell,))
                max_navigation_attempts = self.max_move_attempts * 4
                if self._pickup_navigation_attempts >= max_navigation_attempts:
                    position = tuple(
                        round(float(env.obs[key]), 3)
                        for key in ("x", "y", "z"))
                    pickup_geometry = getattr(
                        self.controller, "_drop_pickup_trace", None)
                    pickup_history = getattr(
                        self.controller, "_drop_pickup_history", ())[-8:]
                    return self._veto(
                        "target_drop_approach_exhausted",
                        "bounded safe navigation never reached a "
                        f"pickup-radius stance after "
                        f"{max_navigation_attempts} controls; "
                        f"position={position}; pickup_geometry="
                        f"{pickup_geometry!r}; pickup_history="
                        f"{pickup_history!r}",
                        cells=(plan.cell,))
                pickup_target = (
                    plan.cell[0] + 0.5, plan.cell[2] + 0.5)
                pickup_sneak = False
                pickup_allow_jump = (
                    plan.cell[1] < math.floor(float(env.obs["y"])))
                pickup_site = getattr(
                    self.controller, "drop_pickup_site", None)
                if pickup_site is not None:
                    pickup_target, pickup_sneak = pickup_site(env, plan)
                    pickup_allow_jump = bool(getattr(
                        self.controller, "_drop_pickup_allow_jump",
                        pickup_allow_jump))
                self._last_pickup_target = tuple(
                    float(value) for value in pickup_target)
                self._last_pickup_sneak = bool(pickup_sneak)
                result = self.controller.pickup(
                    env, execution.drop_item,
                    minimum=self._target_item_baseline
                    + execution.drop_count,
                    attempt=self._pickup_attempts,
                    target=pickup_target, sneak=pickup_sneak,
                    allow_jump=pickup_allow_jump)
                if isinstance(result, PolicyVeto):
                    return result
                if not result:
                    return self._veto(
                        "target_drop_pickup_no_progress",
                        "pickup controller returned no action before exact "
                        "drop collection was verified",
                        cells=(plan.cell,))
                # The configured pickup bound describes bounded collection
                # waits once the player is in range.  A planner-owned safe
                # approach may legitimately take several route controls and
                # must not consume that retry budget before reaching the item.
                # Navigation remains independently bounded above.
                if all(action.get("action") == "wait" for action in result):
                    self._pickup_attempts += 1
                else:
                    self._pickup_navigation_attempts += 1
                return result

            self._advance_target()

        self.done = True
        return None




__all__ = ("MiningExecution", "PlanExecutor")
