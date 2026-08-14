"""Closed-loop execution of deterministic fluid-cast plans."""

from __future__ import annotations

import math
from typing import Iterable

from bedrock_rl.sdg.policy import (
    MinecraftController,
    PolicyVeto,
)
from bedrock_rl.sdg.policy.execution.contracts import (
    FluidCastExecution,
    FluidSourceExecution,
)
from bedrock_rl.sdg.policy.execution.plan import (
    PlanExecutor,
    _cell,
    _event_cell,
    _live_required_occluder,
)
from bedrock_rl.sdg.policy.spatial import (
    DEFAULT_BLOCK_RULES,
)
from bedrock_rl.env.journal import FLUID_BODY


class FluidCastExecutor:
    """Execute a planner-owned water-first cast using one live bucket.

    The executor owns no source search and no world-dependent geometry.  It
    follows the immutable routes in :class:`FluidCastExecution`, and every
    scoop, placement, reaction, recovery, mine, and pickup advances only from
    public observation plus exact journal evidence.
    """

    def __init__(
        self,
        controller: MinecraftController,
        execution: FluidCastExecution,
        *,
        stand_tolerance: float = 0.7,
        max_move_attempts: int = 8,
        max_clear_attempts: int = 8,
        max_scoop_attempts: int = 4,
        max_place_attempts: int = 4,
        max_reaction_attempts: int = 4,
        max_mine_attempts: int = 8,
        max_pickup_attempts: int = 4,
        max_cleanup_attempts: int = 4,
        health_loss_tolerance: float = 0.0,
        lighting=None,
        lighting_immediate: bool = False,
        lighting_route_kinds: Iterable[str] = ("surface", "tunnel"),
        item_recovery=None,
        result_recovery=None,
        final_stand_tolerance: float = 0.08,
    ):
        if not isinstance(execution, FluidCastExecution):
            raise TypeError("execution must be FluidCastExecution")
        self.controller = controller
        self.execution = execution
        self.plan = execution.plan
        self.stand_tolerance = float(stand_tolerance)
        self.max_move_attempts = int(max_move_attempts)
        self.max_clear_attempts = int(max_clear_attempts)
        self.max_scoop_attempts = int(max_scoop_attempts)
        self.max_place_attempts = int(max_place_attempts)
        self.max_reaction_attempts = int(max_reaction_attempts)
        self.max_mine_attempts = int(max_mine_attempts)
        self.max_pickup_attempts = int(max_pickup_attempts)
        self.max_cleanup_attempts = int(max_cleanup_attempts)
        self.health_loss_tolerance = float(health_loss_tolerance)
        self.lighting = lighting
        self.lighting_immediate = bool(lighting_immediate)
        self.lighting_route_kinds = frozenset(
            str(value) for value in lighting_route_kinds)
        self.item_recovery = item_recovery
        self.result_recovery = result_recovery
        self.final_stand_tolerance = float(final_stand_tolerance)
        if self.stand_tolerance <= 0:
            raise ValueError("stand_tolerance must be positive")
        if self.final_stand_tolerance <= 0:
            raise ValueError("final_stand_tolerance must be positive")
        if item_recovery is not None and not callable(item_recovery):
            raise TypeError("item_recovery must be callable")
        if result_recovery is not None and not callable(result_recovery):
            raise TypeError("result_recovery must be callable")
        if self.health_loss_tolerance < 0:
            raise ValueError("health_loss_tolerance cannot be negative")
        for label in (
                "max_move_attempts", "max_clear_attempts",
                "max_scoop_attempts", "max_place_attempts",
                "max_reaction_attempts", "max_mine_attempts",
                "max_pickup_attempts", "max_cleanup_attempts"):
            if getattr(self, label) < 1:
                raise ValueError(f"{label} must be positive")
        self._validate_routes()
        self._started = False
        self.done = False

    def _validate_routes(self):
        for source in self.execution.sources:
            for route in (source.outbound_route, source.return_route):
                stands = route.stands
                for left, right in zip(stands, stands[1:]):
                    horizontal = (abs(left[0] - right[0])
                                  + abs(left[2] - right[2]))
                    if horizontal != 1 or abs(left[1] - right[1]) > 1:
                        raise ValueError(
                            "source routes require cardinal one-block steps")

    def diagnostics(self):
        """Return bounded, JSON-safe state for rejection ledgers."""

        result = {
            "started": self._started,
            "done": self.done,
        }
        if not self._started:
            return result
        recent = {
            repr(cell): tuple(entries[-4:])
            for cell, entries in list(
                self._route_clear_control_trace.items())[-4:]
            if entries
        }
        result.update({
            "phase": self.phase,
            "cast_index": self._cast_index,
            "route_index": self._route_index,
            "route_move_attempts": self._route_move_attempts,
            "route_clear_attempts": {
                repr(cell): attempts
                for cell, attempts in self._route_clear_attempts.items()
            },
            "route_clear_control_attempts": {
                repr(cell): attempts for cell, attempts in
                self._route_clear_control_attempts.items()
            },
            "recent_route_clear_control": recent,
            "recent_phase_transitions": tuple(self._phase_trace[-8:]),
            "phase_attempts": self._phase_attempts,
            "control_attempts": self._control_attempts,
        })
        latest_fluid = next((
            event for event in reversed(getattr(self, "_diagnostic_journal", ()))
            if getattr(event, "name", None) == "player_fluid_state"), None)
        if latest_fluid is not None:
            result["latest_fluid"] = {
                "block": int(latest_fluid.get("block", 0)),
                "flags": int(latest_fluid.get("flags", 0)),
                "cell": _event_cell(latest_fluid),
            }
        return result

    @staticmethod
    def _health(env):
        if "health" in env.obs:
            return float(env.obs["health"])
        return None

    @staticmethod
    def _at(env, stand, tolerance):
        return (math.hypot(
            float(env.obs["x"]) - (stand[0] + 0.5),
            float(env.obs["z"]) - (stand[2] + 0.5)) <= tolerance
            and abs(float(env.obs["y"]) - stand[1]) <= 1.1)

    @staticmethod
    def _centered_at(env, stand, tolerance):
        return (math.hypot(
            float(env.obs["x"]) - (stand[0] + 0.5),
            float(env.obs["z"]) - (stand[2] + 0.5)) <= tolerance
            and abs(float(env.obs["y"]) - stand[1]) <= 0.2)

    def _events(self, env, name, *, since=None):
        start = self._event_start if since is None else int(since)
        return tuple(event for event in env.journal[start:]
                     if getattr(event, "name", None) == name)

    def _veto(self, code, reason, *, cells=(), target=None):
        if target is None:
            target = (self.plan.work_stance[0] + 0.5,
                      self.plan.work_stance[2] + 0.5)
        return PolicyVeto(
            f"fluid_cast_{code}", str(reason), tuple(map(float, target)),
            tuple(_cell(cell) for cell in cells))

    def _set_phase(self, env, phase):
        prior = getattr(self, "phase", None)
        click_target = getattr(self.controller, "click_target", None)
        click = ((None, None) if click_target is None
                 else click_target(env))
        self._phase_trace.append({
            "from": prior,
            "to": str(phase),
            "tick": env.obs.get("tick"),
            "pose": tuple(round(float(env.obs[key]), 3)
                          for key in ("x", "y", "z")),
            "catalyst_block": self.controller.observed_block(
                env, self.plan.catalyst_cell),
            "result_block": self.controller.observed_block(
                env, self.plan.result_cell),
            "click": click,
        })
        self._phase_trace = self._phase_trace[-16:]
        self.phase = str(phase)
        self._phase_event_start = len(env.journal)
        self._phase_attempts = 0
        self._control_attempts = 0
        self._pending_slot = None
        self._route_index = 0
        self._route_move_attempts = 0
        self._route_clear_attempts = {}
        self._route_clear_control_attempts = {}
        self._route_clear_control_trace = {}
        self._route_settle_pending = False

    def reset(self, env):
        self._started = True
        self.done = False
        self._event_start = len(env.journal)
        self._baseline_health = self._health(env)
        self._setup_index = 0
        self._setup_placement_index = 0
        self._cleanup_placement_index = 0
        self._cast_index = 0
        self._mine_event_start = None
        self._result_item_baseline = 0
        self._consumed_sources = set()
        self._lighting_signature = None
        self._phase_trace = []
        if self.controller.item_count(
                env, self.execution.empty_bucket_item) != 1:
            raise ValueError(
                "fluid cast reset needs exactly one visible empty bucket")
        filled = {
            source.filled_bucket_item for source in self.execution.sources}
        if any(self.controller.item_count(env, item) for item in filled):
            raise ValueError(
                "fluid cast reset cannot begin with a filled bucket")
        if self.lighting is not None:
            start = getattr(self.lighting, "start_route_lighting", None)
            if start is not None:
                start(env, immediate=self.lighting_immediate)
            remember = getattr(
                self.lighting, "remember_route_position", None)
            if remember is not None:
                remember(env)
        self._set_phase(env, "approach")
        return self

    def _safety_veto(self, env):
        if bool(env.obs.get("dead", False)):
            return self._veto("player_died", "the live observation is dead")
        health = self._health(env)
        if (health is not None and self._baseline_health is not None
                and health < self._baseline_health
                - self.health_loss_tolerance):
            return self._veto(
                "health_lost", "health fell during contained casting")
        latest_fluid = next((
            event for event in reversed(env.journal)
            if getattr(event, "name", None) == "player_fluid_state"), None)
        dangerous_fluids = frozenset(
            block_id for source in self.execution.sources
            if source.role == "casting"
            for block_id in self._source_fluid_ids(source))
        if (latest_fluid is not None
                and int(latest_fluid.get("flags", 0)) & FLUID_BODY
                and int(latest_fluid.get("block", 0)) in dangerous_fluids):
            return self._veto(
                "fluid_contact",
                "the player collision body entered casting fluid",
                cells=(_event_cell(latest_fluid),))
        allowed = set(self.plan.access_clear_cells)
        allowed.update(self.plan.excavation_cells)
        allowed.add(self.plan.result_cell)
        allowed.update(source.source_cell
                       for source in self.execution.sources)
        allowed.update(
            cell for source in self.execution.sources
            for route in (source.outbound_route, source.return_route)
            for cell in route.mine_cells)
        for event in self._events(env, "block_broken"):
            cell = _event_cell(event)
            if cell not in allowed:
                return self._veto(
                    "unplanned_block_removed",
                    f"block {cell} is outside the immutable cast execution",
                    cells=(cell,))
        bucket_items = {
            self.execution.empty_bucket_item,
            *(source.filled_bucket_item
              for source in self.execution.sources),
        }
        if sum(self.controller.item_count(env, item)
               for item in bucket_items) != 1:
            return self._veto(
                "bucket_not_singleton",
                "live inventory does not expose exactly one reusable bucket")
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

    def _in_coolant(self, env):
        """Whether the current collision body is in the safe cast fluid."""

        latest = next((
            event for event in reversed(env.journal)
            if getattr(event, "name", None) == "player_fluid_state"), None)
        if latest is None or not int(latest.get("flags", 0)) & FLUID_BODY:
            return False
        coolant_ids = frozenset(
            block_id for source in self.execution.sources
            if source.role == "coolant"
            for block_id in self._source_fluid_ids(source))
        return int(latest.get("block", 0)) in coolant_ids

    def _source_fluid_ids(self, source):
        """Expand still/source fluid names to their flowing block family."""

        ids = set(self.controller._block_ids(source.fluid))
        water = set(self.controller._block_ids(
            ("flowing_water", "water")))
        lava = set(self.controller._block_ids(
            ("flowing_lava", "lava")))
        if ids & water:
            ids.update(water)
        if ids & lava:
            ids.update(lava)
        return frozenset(ids)

    def _follow_stands(self, env, stands, *, route=None,
                       lighting_kind=None):
        stands = tuple(stands)
        if self._route_index == 0:
            if not self._at(env, stands[0], self.stand_tolerance):
                return self._veto(
                    "route_origin_diverged",
                    f"player is not at route origin {stands[0]}",
                    cells=(stands[0],))
            self._route_index = 1
        kind = route.kind if route is not None else lighting_kind
        if kind is not None:
            light = self._lighting_actions(env, kind)
            if light is not None:
                return light
        if self._route_index >= len(stands):
            final = stands[-1]
            if self._centered_at(
                    env, final, self.final_stand_tolerance):
                # Route movement can still carry a little momentum on the
                # first centered observation. Settle once, then re-check the
                # exact stance before an irreversible bucket interaction.
                if self._route_settle_pending:
                    self._route_settle_pending = False
                    return [{"action": "wait", "ticks": 2}]
                return None
            if not self._at(env, final, self.stand_tolerance):
                return self._veto(
                    "route_final_stance_diverged",
                    f"player left final route stance {final}",
                    cells=(final,))
            if self._route_move_attempts >= self.max_move_attempts:
                return self._veto(
                    "route_final_recenter_exhausted",
                    f"player did not center on interaction stance {final}",
                    cells=(final,))
            live_y = float(env.obs["y"])
            if abs(live_y - final[1]) > 0.05:
                self._route_move_attempts += 1
                return [{"action": "wait", "ticks": 2}]
            micro_recenter = getattr(
                self.controller, "recenter_interaction_stance", None)
            if micro_recenter is not None and abs(live_y - final[1]) <= 0.05:
                result = micro_recenter(
                    env, final[0] + 0.5, final[2] + 0.5,
                    tolerance=min(0.02, self.final_stand_tolerance))
            else:
                result = self.controller.face_and_move_to(
                    env, final[0] + 0.5, final[2] + 0.5,
                    tolerance=min(0.05, self.final_stand_tolerance),
                    sequential_hop=True, step_up=False, allow_jump=False,
                    movement_cap_blocks=0.3)
            if isinstance(result, PolicyVeto):
                return self._veto(
                    "route_final_recenter_unproven", result.reason,
                    cells=(final,))
            if not result:
                return self._veto(
                    "route_final_recenter_unavailable",
                    f"controller produced no exact recenter for {final}",
                    cells=(final,))
            self._route_move_attempts += 1
            return result
        previous = stands[self._route_index - 1]
        destination = stands[self._route_index]
        if PlanExecutor._at_route_destination(
                env, previous, destination, self.stand_tolerance):
            self._route_index += 1
            self._route_move_attempts = 0
            if self._route_index >= len(stands):
                self._route_settle_pending = True
            return self._follow_stands(
                env, stands, route=route, lighting_kind=lighting_kind)
        horizontal_error = math.hypot(
            float(env.obs["x"]) - (destination[0] + 0.5),
            float(env.obs["z"]) - (destination[2] + 0.5),
        )
        horizontal_arrived = horizontal_error <= min(
            0.4, self.stand_tolerance)
        live_y = float(env.obs["y"])
        ascending_airborne = (
            destination[1] > previous[1]
            and horizontal_arrived
            and 0.05 < live_y - destination[1] <= 1.35)
        descending_airborne = (
            destination[1] < previous[1]
            and math.hypot(
                float(env.obs["x"]) - (destination[0] + 0.5),
                float(env.obs["z"]) - (destination[2] + 0.5),
            ) <= min(0.08, self.stand_tolerance)
            and destination[1] + 0.05 < live_y
            <= previous[1] + 1.35)
        flat_airborne = (
            destination[1] == previous[1]
            and 0.05 < live_y - previous[1] <= 1.35
            and PlanExecutor._on_horizontal_leg(
                env, previous, destination, self.stand_tolerance))
        if ascending_airborne or descending_airborne or flat_airborne:
            if self._route_move_attempts >= self.max_move_attempts:
                return self._veto(
                    "route_vertical_settle_exhausted",
                    f"player did not settle onto route leg "
                    f"{previous} -> {destination}",
                    cells=(previous, destination))
            self._route_move_attempts += 1
            return [{"action": "wait", "ticks": 2}]
        vertical_landed_offcenter = (
            destination[1] != previous[1]
            and abs(live_y - destination[1]) <= 0.05
            and horizontal_error > min(0.05, self.stand_tolerance)
            and PlanExecutor._on_horizontal_leg(
                env, previous, destination, self.stand_tolerance))
        if vertical_landed_offcenter:
            if self._route_move_attempts >= self.max_move_attempts:
                return self._veto(
                    "route_vertical_recenter_exhausted",
                    f"player landed outside the center of vertical stand "
                    f"{destination}", cells=(destination,))
            result = self.controller.face_and_move_to(
                env, destination[0] + 0.5, destination[2] + 0.5,
                tolerance=0.05, sequential_hop=True,
                step_up=False, allow_jump=False,
                movement_cap_blocks=0.3)
            if isinstance(result, PolicyVeto):
                return self._veto(
                    "route_move_unproven", result.reason,
                    cells=(destination,))
            if not result:
                return self._veto(
                    "route_vertical_recenter_unavailable",
                    f"controller produced no recenter move over "
                    f"vertical stand {destination}", cells=(destination,))
            self._route_move_attempts += 1
            return result
        if not PlanExecutor._on_leg(
                env, previous, destination, self.stand_tolerance):
            swimming = self._in_coolant(env)
            previous_distance = math.hypot(
                float(env.obs["x"]) - (previous[0] + 0.5),
                float(env.obs["z"]) - (previous[2] + 0.5))
            if (swimming and previous_distance <= 1.5
                    and abs(float(env.obs["y"]) - previous[1]) <= 1.5):
                if self._route_move_attempts >= self.max_move_attempts * 4:
                    return self._veto(
                        "route_current_recovery_exhausted",
                        "safe coolant current kept the player outside the "
                        f"certified leg {previous} -> {destination}",
                        cells=(previous, destination))
                result = self.controller.face_and_move_to(
                    env, previous[0] + 0.5, previous[2] + 0.5,
                    tolerance=0.2, sequential_hop=True,
                    step_up=float(env.obs["y"]) < previous[1] - 0.05,
                    allow_jump=True, swim=True,
                    movement_cap_blocks=1.5)
                if isinstance(result, PolicyVeto):
                    return self._veto(
                        "route_current_recovery_unproven", result.reason,
                        cells=(previous, destination))
                if result:
                    self._route_move_attempts += 1
                    return result
            return self._veto(
                "route_position_diverged",
                f"player left planned leg {previous} -> {destination}",
                cells=(previous, destination))
        drop_is_safe = getattr(self.controller, "drop_is_safe", None)
        if (drop_is_safe is not None
                and not drop_is_safe(previous[1], destination[1])):
            return self._veto(
                "route_drop_rejected",
                f"planned descent to {destination} exceeds policy bounds",
                cells=(previous, destination))
        mine_cells = () if route is None else route.mine_cells
        body = {
            destination,
            (destination[0], destination[1] + 1, destination[2]),
        }
        if destination[1] < previous[1]:
            body.add((destination[0], destination[1] + 2, destination[2]))
        elif destination[1] > previous[1]:
            body.add((previous[0], previous[1] + 2, previous[2]))
        required = tuple(cell for cell in mine_cells if cell in body)
        for cell in sorted(required, key=lambda value: -value[1]):
            if any(_event_cell(event) == cell
                   for event in self._events(env, "block_broken")):
                continue
            recovered = self._recover_item(
                env, self.execution.excavation_tool)
            if recovered is not None:
                return recovered
            attempts = self._route_clear_attempts.get(cell, 0)
            if attempts >= self.max_clear_attempts:
                trace = self._route_clear_control_trace.get(cell, [])[-4:]
                return self._veto(
                    "route_clear_exhausted",
                    f"no exact break proved route clearance {cell}; "
                    f"recent_control={trace!r}",
                    cells=(cell,))
            occluder = _live_required_occluder(
                self.controller, env, cell, required)
            if occluder is not None:
                occluder_attempts = self._route_clear_attempts.get(
                    occluder, 0)
                if occluder_attempts >= self.max_clear_attempts:
                    return self._veto(
                        "route_occluder_clear_exhausted",
                        f"journal-proven route cell {occluder} reoccupied "
                        f"the ray to {cell}", cells=(occluder, cell))
                selected = self.controller.select(
                    env, self.execution.excavation_tool)
                if selected is None:
                    return self._veto(
                        "route_occluder_tool_missing",
                        "no visible tool can clear the reoccupied route "
                        f"cell {occluder}", cells=(occluder,))
                if selected:
                    self._trace_route_clear(
                        env, occluder, selected, requested=cell,
                        reoccupied=True)
                    return selected
                occluder_ticks = self._mining_ticks(
                    env, occluder, self.execution.excavation_tool,
                    self.execution.excavation_ticks)
                self._route_clear_attempts[occluder] = (
                    occluder_attempts + 1)
                actions = [{
                    "action": "left_click_hold",
                    "ticks": int(occluder_ticks),
                }]
                self._trace_route_clear(
                    env, occluder, actions, requested=cell,
                    reoccupied=True)
                return actions
            control_attempts = self._route_clear_control_attempts.get(
                cell, 0)
            if control_attempts >= self.max_clear_attempts * 4:
                trace = self._route_clear_control_trace.get(cell, [])[-4:]
                return self._veto(
                    "route_clear_control_exhausted",
                    f"camera/stance controls never proved route cell "
                    f"{cell}; recent_control={trace!r}", cells=(cell,))
            ticks = self._mining_ticks(
                env, cell, self.execution.excavation_tool,
                self.execution.excavation_ticks)
            result = self.controller.mine_exact(
                env, cell, tool=self.execution.excavation_tool,
                ticks=ticks,
                stand=(previous[0] + 0.5, previous[2] + 0.5))
            if isinstance(result, PolicyVeto):
                occluder = _live_required_occluder(
                    self.controller, env, cell, required)
                if (result.code == "mine_target_unproven"
                        and occluder is not None):
                    occluder_attempts = self._route_clear_attempts.get(
                        occluder, 0)
                    if occluder_attempts < self.max_clear_attempts:
                        occluder_ticks = self._mining_ticks(
                            env, occluder,
                            self.execution.excavation_tool,
                            self.execution.excavation_ticks)
                        self._route_clear_attempts[occluder] = (
                            occluder_attempts + 1)
                        actions = [{
                            "action": "left_click_hold",
                            "ticks": int(occluder_ticks),
                        }]
                        self._trace_route_clear(
                            env, occluder, actions, requested=cell,
                            reoccupied=True)
                        return actions
                return self._veto(
                    "route_clear_unproven",
                    f"{result.reason}; recent_control="
                    f"{self._route_clear_control_trace.get(cell, [])[-4:]!r}",
                    cells=(cell,))
            if not result:
                return self._veto(
                    "route_clear_unavailable",
                    f"controller could not clear route cell {cell}; "
                    f"recent_control="
                    f"{self._route_clear_control_trace.get(cell, [])[-4:]!r}",
                    cells=(cell,))
            self._trace_route_clear(env, cell, result)
            if any(action.get("action") == "left_click_hold"
                   for action in result):
                self._route_clear_attempts[cell] = attempts + 1
            else:
                self._route_clear_control_attempts[cell] = (
                    control_attempts + 1)
            return result
        swimming = self._in_coolant(env)
        move_limit = self.max_move_attempts * (4 if swimming else 1)
        if self._route_move_attempts >= move_limit:
            return self._veto(
                "route_move_exhausted",
                f"player did not reach route stand {destination}",
                cells=(destination,))
        result = self.controller.face_and_move_to(
            env, destination[0] + 0.5, destination[2] + 0.5,
            tolerance=(0.05 if destination[1] != previous[1] else 0.4),
            sequential_hop=True,
            step_up=destination[1] > previous[1],
            allow_jump=destination[1] >= previous[1], swim=swimming,
            movement_cap_blocks=(
                1.5 if swimming else
                0.3 if destination[1] != previous[1] else None))
        if isinstance(result, PolicyVeto):
            return self._veto(
                "route_move_unproven", result.reason,
                cells=(destination,))
        if not result:
            return self._veto(
                "route_move_unavailable",
                f"controller produced no move to {destination}",
                cells=(destination,))
        self._route_move_attempts += 1
        return result

    def _slot_transition(self, env, slot, before, after):
        if slot is None:
            return False
        before_ids = frozenset(self.controller._item_ids(before))
        after_ids = frozenset(self.controller._item_ids(after))
        return any(
            int(event.get("slot", -1)) == int(slot)
            and int(event.get("was_item", 0)) in before_ids
            and int(event.get("item", 0)) in after_ids
            and int(event.get("count", 0)) == 1
            for event in self._events(
                env, "slot_changed", since=self._phase_event_start))

    def _mining_ticks(self, env, cell, tool, fallback):
        """Resolve a bounded hold from the live block when supported."""

        resolver = getattr(self.controller, "mining_hold_ticks", None)
        if resolver is None:
            return int(fallback)
        ticks = resolver(env, cell, tool=tool, fallback=int(fallback))
        if (not isinstance(ticks, int) or isinstance(ticks, bool)
                or ticks < 1):
            raise ValueError(
                "controller.mining_hold_ticks must return a positive integer")
        return ticks

    def _trace_route_clear(
            self, env, cell, actions, *, requested=None,
            reoccupied=False):
        """Remember the public evidence behind one bounded route control."""

        click_target = getattr(self.controller, "click_target", None)
        found, click_cell = ((None, None) if click_target is None
                             else click_target(env))
        selected_slot = int(env.obs.get("hotbar_sel", -1))
        hotbar = env.obs.get("hotbar_ids", ())
        selected_item = (int(hotbar[selected_slot])
                         if 0 <= selected_slot < len(hotbar) else None)
        entry = {
            "pose": tuple(round(float(env.obs[key]), 3)
                          for key in ("x", "y", "z")),
            "yaw": round(float(env.obs.get("yaw", 0.0)), 3),
            "pitch": round(float(env.obs.get("pitch", 0.0)), 3),
            "requested": _cell(cell if requested is None else requested),
            "click": (found, click_cell),
            "observed_block": self.controller.observed_block(env, cell),
            "tool": self.execution.excavation_tool,
            "tool_ids": tuple(self.controller._item_ids(
                self.execution.excavation_tool)),
            "selected": (selected_slot, selected_item),
            "actions": tuple(action.get("action") for action in actions),
        }
        if reoccupied:
            entry["reoccupied"] = _cell(cell)
        self._route_clear_control_trace.setdefault(
            _cell(cell), []).append(entry)

    def _recover_item(self, env, item, minimum=1):
        if item is None or self.item_recovery is None:
            return None
        if self.controller.item_count(env, item) >= int(minimum):
            return None
        return self.item_recovery(env, item, int(minimum))

    def _broken_proof(self, env, cell, block, *, since=None):
        wanted = frozenset(self.controller._block_ids(block))
        return any(
            _event_cell(event) == cell
            and int(event.get("block", 0)) in wanted
            for event in self._events(env, "block_broken", since=since))

    def _placed_proof(self, env, step, item, *, since=None, slot=None):
        blocks = frozenset(self.controller._block_ids(step.material))
        items = frozenset(self.controller._item_ids(item))
        return any(
            _event_cell(event) == step.cell
            and int(event.get("block", 0)) in blocks
            and int(event.get("item", 0)) in items
            and (slot is None
                 or int(event.get("slot", -1)) == int(slot))
            for event in self._events(env, "block_placed", since=since))

    def _bucket_target(self, env):
        return next((event for event in reversed(env.journal)
                     if getattr(event, "name", None) == "bucket_target"),
                    None)

    def _scoop(self, env, source):
        reusable_catalyst = source.source_cell == self.plan.catalyst_cell
        if (not reusable_catalyst
                and source.source_cell in self._consumed_sources):
            return self._veto(
                "source_reused",
                f"source cell {source.source_cell} was already consumed",
                cells=(source.source_cell,))
        if (self._slot_transition(
                env, self._pending_slot,
                self.execution.empty_bucket_item,
                source.filled_bucket_item)
                and self._broken_proof(
                    env, source.source_cell, source.fluid,
                    since=self._phase_event_start)):
            if not reusable_catalyst:
                self._consumed_sources.add(source.source_cell)
            return True
        selected = self.controller.select(
            env, self.execution.empty_bucket_item)
        if selected is None:
            return self._veto(
                "empty_bucket_missing",
                "the singleton reusable empty bucket is not visible")
        if selected:
            return selected
        target = self._bucket_target(env)
        wanted = frozenset(self.controller._block_ids(source.fluid))
        slot = int(env.obs.get("hotbar_sel", -1))
        exact = (target is not None
                 and _event_cell(target) == source.source_cell
                 and int(target.get("block", 0)) in wanted
                 and int(target.get("meta", -1)) == source.source_meta
                 and int(target.get("slot", -2)) == slot)
        if not exact:
            aim = self.controller.aim(env, source.source_cell)
            self._control_attempts += 1
            if self._control_attempts <= self.max_scoop_attempts * 4:
                # A zero-length aim correction means the camera angles are
                # already within tolerance, not that the bucket ray from this
                # newly selected slot has been observed.  Give the public
                # bucket-target probe one engine tick to refresh; continue to
                # enforce the same finite control budget.
                return aim or [{"action": "wait", "ticks": 1}]
            return self._veto(
                "bucket_target_unproven",
                f"empty-bucket ray does not prove source "
                f"{source.source_cell} meta {source.source_meta}",
                cells=(source.source_cell,))
        if self._phase_attempts >= self.max_scoop_attempts:
            return self._veto(
                "scoop_exhausted", "bucket scoop transition was not proven",
                cells=(source.source_cell,))
        self._pending_slot = slot
        self._phase_attempts += 1
        return [{"action": "right_click"}]

    def _place_fluid(self, env, step, filled_item):
        if (self._placed_proof(
                env, step, filled_item, since=self._phase_event_start,
                slot=self._pending_slot)
                and self._slot_transition(
                    env, self._pending_slot, filled_item,
                    self.execution.empty_bucket_item)):
            return True
        selected = self.controller.select(env, filled_item)
        if selected is None:
            return self._veto(
                "filled_bucket_missing",
                f"no {filled_item!r} is visible for {step.role}",
                cells=(step.cell,))
        if selected:
            return selected
        _block, target = self.controller.click_target(env)
        destination = self.controller.click_destination(env)
        if target != step.support or destination != step.cell:
            from bedrock_rl.sdg.policy.spatial import (
                placement_face_point,
            )
            point = tuple(value - 0.5 for value in placement_face_point(
                step.support, step.cell,
                observer=(float(env.obs["x"]),
                          float(env.obs["y"]) + 1.62,
                          float(env.obs["z"]))))
            aim = self.controller.aim_point(env, point)
            if aim:
                self._control_attempts += 1
                if self._control_attempts <= self.max_place_attempts * 4:
                    return aim
            return self._veto(
                "placement_ray_unproven",
                f"click ray does not prove {step.support} -> {step.cell}",
                cells=(step.support, step.cell))
        if self._phase_attempts >= self.max_place_attempts:
            return self._veto(
                "placement_exhausted",
                f"no exact placement proved {step.role}",
                cells=(step.cell,))
        self._pending_slot = int(env.obs.get("hotbar_sel", -1))
        self._phase_attempts += 1
        return [{"action": "right_click"}]

    def _setup_actions(self, env):
        cells = self.plan.access_clear_cells + self.plan.excavation_cells
        if self._setup_index < len(cells):
            cell = cells[self._setup_index]
            if any(_event_cell(event) == cell for event in self._events(
                    env, "block_broken", since=self._phase_event_start)):
                self._setup_index += 1
                self._set_phase(env, "setup_excavate")
                return self._setup_actions(env)
            if self._phase_attempts >= self.max_clear_attempts:
                return self._veto(
                    "setup_excavate_exhausted",
                    f"cast-site excavation {cell} was not journal-proven",
                    cells=(cell,))
            recovered = self._recover_item(
                env, self.execution.excavation_tool)
            if recovered is not None:
                return recovered
            ticks = self._mining_ticks(
                env, cell, self.execution.excavation_tool,
                self.execution.excavation_ticks)
            result = self.controller.mine_exact(
                env, cell, tool=self.execution.excavation_tool,
                ticks=ticks,
                stand=(self.plan.work_stance[0] + 0.5,
                       self.plan.work_stance[2] + 0.5))
            if isinstance(result, PolicyVeto):
                return self._veto(
                    "setup_excavate_unproven", result.reason, cells=(cell,))
            if not result:
                return self._veto(
                    "setup_excavate_unavailable",
                    f"controller could not excavate cast-site cell {cell}",
                    cells=(cell,))
            if any(action.get("action") == "left_click_hold"
                   for action in result):
                self._phase_attempts += 1
            return result
        placements = self.plan.setup_placements
        if self._setup_placement_index < len(placements):
            step = placements[self._setup_placement_index]
            if self._placed_proof(
                    env, step, step.material,
                    since=self._phase_event_start):
                self._setup_placement_index += 1
                self._set_phase(env, "setup_place")
                return self._setup_actions(env)
            recovered = self._recover_item(env, step.material)
            if recovered is not None:
                return recovered
            result = self.controller.place(
                env, step.material, step.cell, step.support,
                block=step.material,
                stand=(step.stance[0] + 0.5, step.stance[2] + 0.5))
            if isinstance(result, PolicyVeto):
                return self._veto(
                    "setup_place_unproven", result.reason,
                    cells=(step.cell,))
            if not result:
                return self._veto(
                    "setup_place_unavailable",
                    f"controller could not place setup cell {step.cell}",
                    cells=(step.cell,))
            self._phase_attempts += 1
            if self._phase_attempts > self.max_place_attempts:
                return self._veto(
                    "setup_place_exhausted",
                    f"setup placement {step.cell} was not proven",
                    cells=(step.cell,))
            return result
        return None

    def _result_live(self, env):
        wanted = frozenset(self.controller._block_ids(self.plan.result_block))
        observed = self.controller.observed_block(env, self.plan.result_cell)
        found, target = self.controller.click_target(env)
        return (observed in wanted
                or (target == self.plan.result_cell and found in wanted))

    def _result_collected(self, env):
        wanted = frozenset(self.controller._item_ids(
            self.execution.result_item))
        picked = sum(
            int(event.get("count", 0)) for event in self._events(
                env, "item_picked_up", since=self._mine_event_start)
            if int(event.get("item", 0)) in wanted)
        return (picked >= self.execution.drop_count
                and self.controller.item_count(
                    env, self.execution.result_item)
                >= self._result_item_baseline
                + self.execution.drop_count)

    def _result_pickup_proven(self, env):
        wanted = frozenset(self.controller._item_ids(
            self.execution.result_item))
        return sum(
            int(event.get("count", 0)) for event in self._events(
                env, "item_picked_up", since=self._mine_event_start)
            if int(event.get("item", 0)) in wanted
        ) >= self.execution.drop_count

    def _result_break_proof(self, env):
        blocks = frozenset(self.controller._block_ids(
            self.plan.result_block))
        tools = frozenset(self.controller._item_ids(
            self.execution.mining_tool))
        drops = frozenset(self.controller._item_ids(
            self.execution.result_item))
        return any(
            _event_cell(event) == self.plan.result_cell
            and int(event.get("block", 0)) in blocks
            and int(event.get("tool", 0)) in tools
            and int(event.get("drop", 0)) in drops
            for event in self._events(
                env, "block_broken", since=self._mine_event_start))

    def actions(self, env):
        if not self._started:
            raise RuntimeError("FluidCastExecutor.reset(env) is required")
        if self.done:
            return None
        self._diagnostic_journal = env.journal
        safety = self._safety_veto(env)
        if safety is not None:
            return safety
        while not self.done:
            if self.phase == "approach":
                result = self._follow_stands(
                    env, self.plan.approach_stands,
                    lighting_kind="tunnel")
                if result is not None:
                    return result
                self._set_phase(env, "setup_excavate")
                continue
            if self.phase in ("setup_excavate", "setup_place"):
                result = self._setup_actions(env)
                if result is not None:
                    return result
                self._set_phase(env, "coolant_route_out")
                continue
            if self.phase == "coolant_route_out":
                source = self.execution.sources[0]
                result = self._follow_stands(
                    env, source.outbound_route.stands,
                    route=source.outbound_route)
                if result is not None:
                    return result
                self._set_phase(env, "coolant_scoop")
                continue
            if self.phase == "coolant_scoop":
                source = self.execution.sources[0]
                result = self._scoop(env, source)
                if result is not True:
                    return result
                self._set_phase(env, "coolant_route_back")
                continue
            if self.phase == "coolant_route_back":
                source = self.execution.sources[0]
                result = self._follow_stands(
                    env, source.return_route.stands,
                    route=source.return_route)
                if result is not None:
                    return result
                self._set_phase(env, "place_catalyst")
                continue
            if self.phase == "place_catalyst":
                result = self._place_fluid(
                    env, self.plan.catalyst_placement,
                    self.execution.sources[0].filled_bucket_item)
                if result is not True:
                    return result
                self._set_phase(env, "casting_route_out")
                continue
            if self.phase == "casting_route_out":
                source = self.execution.sources[self._cast_index + 1]
                result = self._follow_stands(
                    env, source.outbound_route.stands,
                    route=source.outbound_route)
                if result is not None:
                    return result
                self._set_phase(env, "casting_scoop")
                continue
            if self.phase == "casting_scoop":
                source = self.execution.sources[self._cast_index + 1]
                result = self._scoop(env, source)
                if result is not True:
                    return result
                self._set_phase(env, "casting_route_back")
                continue
            if self.phase == "casting_route_back":
                source = self.execution.sources[self._cast_index + 1]
                result = self._follow_stands(
                    env, source.return_route.stands,
                    route=source.return_route)
                if result is not None:
                    return result
                self._set_phase(env, "place_reactant")
                continue
            if self.phase == "place_reactant":
                source = self.execution.sources[self._cast_index + 1]
                result = self._place_fluid(
                    env, self.plan.reactant_placement,
                    source.filled_bucket_item)
                if result is not True:
                    return result
                self._set_phase(env, "verify_reaction")
                continue
            if self.phase == "verify_reaction":
                if self._result_live(env):
                    # Keep the water in the mold while the result is mined.
                    # Besides being the safer, conventional cast loop, this
                    # lets every later lava source reuse the same catalyst
                    # without two extra bucket operations per block.
                    self._set_phase(env, "mine_result")
                    self._mine_event_start = len(env.journal)
                    self._result_item_baseline = self.controller.item_count(
                        env, self.execution.result_item)
                    continue
                if self._phase_attempts >= self.max_reaction_attempts:
                    return self._veto(
                        "reaction_unverified",
                        f"live state never proved {self.plan.result_block!r} "
                        f"at {self.plan.result_cell}",
                        cells=(self.plan.result_cell,))
                self._phase_attempts += 1
                return [{
                    "action": "wait",
                    "ticks": self.execution.casting_wait_ticks,
                }]
            if self.phase == "recover_catalyst":
                source = FluidSourceExecution(
                    "coolant", self.plan.catalyst_cell,
                    self.execution.sources[0].outbound_route,
                    self.execution.sources[0].return_route,
                    self.plan.catalyst_fluid,
                    self.execution.sources[0].filled_bucket_item,
                    self.execution.sources[0].source_meta)
                result = self._scoop(env, source)
                if result is not True:
                    return result
                self._set_phase(env, "cleanup")
                continue
            if self.phase == "mine_result":
                if self._result_break_proof(env):
                    self._set_phase(env, "pickup_result")
                    continue
                if not self._result_live(env):
                    return self._veto(
                        "dry_result_missing",
                        "no exact reaction result block is live",
                        cells=(self.plan.result_cell,))
                if self._phase_attempts >= self.max_mine_attempts:
                    return self._veto(
                        "result_mine_exhausted",
                        "exact result break was not journal-proven",
                        cells=(self.plan.result_cell,))
                recovered = self._recover_item(
                    env, self.execution.mining_tool)
                if recovered is not None:
                    return recovered
                ticks = self._mining_ticks(
                    env, self.plan.result_cell,
                    self.execution.mining_tool,
                    self.execution.mining_ticks)
                result = self.controller.mine_exact(
                    env, self.plan.result_cell,
                    tool=self.execution.mining_tool,
                    ticks=ticks,
                    block=self.plan.result_block,
                    since=self._mine_event_start,
                    stand=(self.plan.result_mining_stance[0] + 0.5,
                           self.plan.result_mining_stance[2] + 0.5))
                if isinstance(result, PolicyVeto):
                    return self._veto(
                        "result_mine_unproven", result.reason,
                        cells=(self.plan.result_cell,))
                if not result:
                    return self._veto(
                        "result_mine_unavailable",
                        "controller could not mine the dry result",
                        cells=(self.plan.result_cell,))
                if any(action.get("action") == "left_click_hold"
                       for action in result):
                    self._phase_attempts += 1
                return result
            if self.phase == "pickup_result":
                if self._result_collected(env):
                    self._cast_index += 1
                    self._set_phase(env, "post_pickup_recenter")
                    continue
                if (self._result_pickup_proven(env)
                        and self.result_recovery is not None):
                    recovered = self.result_recovery(
                        env, self.execution.result_item,
                        self._result_item_baseline
                        + self.execution.drop_count)
                    if recovered is not None:
                        return recovered
                if self._phase_attempts >= self.max_pickup_attempts:
                    return self._veto(
                        "result_pickup_exhausted",
                        "result pickup was not inventory+journal proven",
                        cells=(self.plan.result_cell,))
                result = self.controller.pickup(
                    env, self.execution.result_item,
                    minimum=(self._result_item_baseline
                             + self.execution.drop_count),
                    attempt=self._phase_attempts,
                    target=(self.plan.result_cell[0] + 0.5,
                            self.plan.result_cell[2] + 0.5),
                    allow_jump=False)
                if isinstance(result, PolicyVeto):
                    return self._veto(
                        "result_pickup_unproven", result.reason,
                        cells=(self.plan.result_cell,))
                self._phase_attempts += 1
                if result:
                    return result
                continue
            if self.phase == "post_pickup_recenter":
                if self._centered_at(
                        env, self.plan.work_stance,
                        self.final_stand_tolerance):
                    if self._cast_index < self.plan.casts_required:
                        self._set_phase(env, "casting_route_out")
                    else:
                        self._set_phase(env, "recover_catalyst")
                    continue
                if self._phase_attempts >= self.max_move_attempts * 3:
                    return self._veto(
                        "post_pickup_recenter_exhausted",
                        "player did not return to the certified work stance "
                        "after collecting the reaction result",
                        cells=(self.plan.work_stance,))
                result = self.controller.face_and_move_to(
                    env, self.plan.work_stance[0] + 0.5,
                    self.plan.work_stance[2] + 0.5,
                    tolerance=min(0.05, self.final_stand_tolerance),
                    sequential_hop=True,
                    step_up=(float(env.obs["y"])
                             < self.plan.work_stance[1] - 0.05),
                    allow_jump=True, movement_cap_blocks=0.3)
                if isinstance(result, PolicyVeto):
                    return self._veto(
                        "post_pickup_recenter_unproven", result.reason,
                        cells=(self.plan.work_stance,))
                if not result:
                    return self._veto(
                        "post_pickup_recenter_unavailable",
                        "controller produced no return move to the "
                        f"certified work stance {self.plan.work_stance}",
                        cells=(self.plan.work_stance,))
                self._phase_attempts += 1
                return result
            if self.phase == "cleanup":
                fluid_cells = tuple(
                    cell for cell in self.plan.cleanup_cells
                    if DEFAULT_BLOCK_RULES.is_fluid(
                        self.controller.observed_block(env, cell) or 0))
                if fluid_cells:
                    if self._phase_attempts >= self.max_cleanup_attempts:
                        return self._veto(
                            "cleanup_exhausted",
                            "fluid remains in the reusable mold",
                            cells=fluid_cells)
                    self._phase_attempts += 1
                    return [{"action": "wait", "ticks": 1}]
                placements = self.execution.cleanup_placements
                if self._cleanup_placement_index < len(placements):
                    step = placements[self._cleanup_placement_index]
                    if self._placed_proof(
                            env, step, step.material,
                            since=self._phase_event_start):
                        self._cleanup_placement_index += 1
                        self._set_phase(env, "cleanup")
                        continue
                    if self._phase_attempts >= self.max_cleanup_attempts:
                        return self._veto(
                            "cleanup_place_exhausted",
                            f"cleanup placement {step.cell} was not proven",
                            cells=(step.cell,))
                    recovered = self._recover_item(env, step.material)
                    if recovered is not None:
                        return recovered
                    result = self.controller.place(
                        env, step.material, step.cell, step.support,
                        block=step.material,
                        stand=(step.stance[0] + 0.5,
                               step.stance[2] + 0.5))
                    if isinstance(result, PolicyVeto):
                        return self._veto(
                            "cleanup_place_unproven", result.reason,
                            cells=(step.cell,))
                    if not result:
                        return self._veto(
                            "cleanup_place_unavailable",
                            f"controller could not restore {step.cell}",
                            cells=(step.cell,))
                    self._phase_attempts += 1
                    return result
                if self.controller.item_count(
                        env, self.execution.sources[0].filled_bucket_item
                ) != 1:
                    return self._veto(
                        "coolant_recovery_missing",
                        "final cleanup did not retain the recovered coolant")
                self._set_phase(env, "final_recenter")
                continue
            if self.phase == "final_recenter":
                if self._centered_at(
                        env, self.plan.work_stance,
                        self.final_stand_tolerance):
                    self.phase = "complete"
                    self.done = True
                    return None
                if self._phase_attempts >= self.max_move_attempts:
                    return self._veto(
                        "final_recenter_exhausted",
                        "player did not return to the certified work stance",
                        cells=(self.plan.work_stance,))
                result = self.controller.face_and_move_to(
                    env,
                    self.plan.work_stance[0] + 0.5,
                    self.plan.work_stance[2] + 0.5,
                    tolerance=min(0.05, self.final_stand_tolerance),
                    sequential_hop=True,
                    step_up=(self.plan.work_stance[1]
                             > math.floor(float(env.obs["y"]) + 0.05)),
                    allow_jump=True,
                    movement_cap_blocks=0.3)
                if isinstance(result, PolicyVeto):
                    return self._veto(
                        "final_recenter_unproven", result.reason,
                        cells=(self.plan.work_stance,))
                if not result:
                    return self._veto(
                        "final_recenter_unavailable",
                        "controller produced no move to the certified work "
                        "stance",
                        cells=(self.plan.work_stance,))
                self._phase_attempts += 1
                return result
            raise RuntimeError(f"unknown fluid cast phase {self.phase!r}")
        return None


__all__ = ("FluidCastExecutor",)
