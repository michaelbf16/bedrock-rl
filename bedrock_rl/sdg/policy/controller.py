"""Reusable policy controls for Minecraft demonstrations.

This module contains only objective-independent behavior.  Tasks and example
policies decide where to go and what to build; these controls bound ordinary
movement and refuse a move when the public observation already shows danger
on its path.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Any, Mapping

from bedrock_rl.sdg.policy.lighting import (
    WallTorchRouteMixin as WallTorchRouteMixin,
)
from bedrock_rl.sdg.policy.types import PolicyVeto
from bedrock_rl.sdg.policy.mining import (
    dry_mining_ticks, submerged_mining_ticks,
)
from bedrock_rl.env.journal import FLUID_VIEWPOINT


@dataclass
class _FurnaceSession:
    """Live, journal-verifiable state for one furnace transaction."""

    signature: tuple[Any, ...]
    event_start: int
    baseline_output: int
    phase: str = "load"
    load_start: int = 0
    input_loaded: int = 0
    fuel_loaded: int = 0
    pending_transfers: tuple[Any, ...] = ()
    output_start: int = 0
    output_slot: int | None = None
    output_slot_before: tuple[int, int] = (0, 0)
    waits: int = 0
    verify_attempts: int = 0


@dataclass(frozen=True)
class MovementProfile:
    """Human/VLM-like spacing for one movement decision.

    A policy asks for one move and receives one bounded key hold plus an
    optional pause. Long travel therefore stays closed-loop: observe, move a
    few blocks, observe again. The defaults model the common 2–3 block VLM
    step instead of letting a script cross a world in one perfect action.
    """

    min_blocks: float = 2.0
    max_blocks: float = 2.75
    ticks_per_block: float = 4.0 / 0.65
    min_pause_ticks: int = 0
    max_pause_ticks: int = 2

    def __post_init__(self):
        from bedrock_rl.adapters.netherite.computer import MAX_MOVE_HOLD
        if not 0 < self.min_blocks <= self.max_blocks:
            raise ValueError("movement needs 0 < min_blocks <= max_blocks")
        if self.ticks_per_block <= 0:
            raise ValueError("movement ticks_per_block must be positive")
        if not 0 <= self.min_pause_ticks <= self.max_pause_ticks:
            raise ValueError("movement pause ticks need 0 <= min <= max")
        if round(self.max_blocks * self.ticks_per_block) > MAX_MOVE_HOLD:
            raise ValueError(
                "movement max_blocks exceeds the computer tool's per-action "
                "movement limit")

    @classmethod
    def parse(cls, value=None):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("movement profile must be a mapping")
        return cls(**dict(value))

    def vary(self, values):
        """Apply optional case values without coupling sampler to policy."""
        values = dict(values or {})
        updates = {}
        for field, key in (
            ("min_blocks", "movement_min_blocks"),
            ("max_blocks", "movement_max_blocks"),
            ("ticks_per_block", "movement_ticks_per_block"),
            ("min_pause_ticks", "movement_min_pause_ticks"),
            ("max_pause_ticks", "movement_max_pause_ticks"),
        ):
            if key in values:
                updates[field] = values[key]
        return replace(self, **updates) if updates else self

    def actions(self, rng, *, key="w", blocks=None):
        if blocks is None:
            distance = rng.uniform(self.min_blocks, self.max_blocks)
        else:
            # min_blocks describes ordinary sampled strides, not a ban on
            # fine positioning. A policy approaching a furnace face or portal
            # must be able to request its remaining sub-block correction.
            distance = max(0.05, min(float(blocks), self.max_blocks))
        ticks = max(1, round(distance * self.ticks_per_block))
        out = [{"action": "key", "k": key, "ticks": ticks}]
        pause = rng.randint(self.min_pause_ticks, self.max_pause_ticks)
        if pause:
            out.append({"action": "wait", "ticks": pause})
        return out


@dataclass(frozen=True)
class SurvivalBehaviors:
    """Reusable guardrails for closed-loop Minecraft policies."""

    avoid_hazards: bool = True
    hazard_blocks: tuple[str, ...] = (
        "flowing_lava", "lava", "fire", "cactus", "magma")
    hazard_clearance_blocks: float = 0.55
    safe_descent: bool = True
    max_drop_blocks: float = 1.0
    collect_drops: bool = True
    pickup_attempts: int = 4
    pickup_wait_ticks: int = 12
    pickup_step_blocks: float = 0.5
    lighting: bool = True
    light_every_blocks: int = 10
    light_grace_blocks: float = 2.0
    light_overshoot_blocks: float = 2.0
    light_item: str = "torch"
    light_pause_ticks: int = 8

    def __post_init__(self):
        if self.hazard_clearance_blocks < 0:
            raise ValueError("hazard clearance cannot be negative")
        if self.max_drop_blocks < 0:
            raise ValueError("max_drop_blocks cannot be negative")
        if self.pickup_attempts < 0 or self.pickup_wait_ticks < 0:
            raise ValueError(
                "pickup attempts and wait ticks cannot be negative")
        if self.pickup_step_blocks <= 0:
            raise ValueError("pickup step_blocks must be positive")
        if self.light_every_blocks < 1:
            raise ValueError("light_every_blocks must be positive")
        if self.light_grace_blocks < 0:
            raise ValueError("lighting grace_blocks cannot be negative")
        if self.light_overshoot_blocks < 0:
            raise ValueError("lighting overshoot_blocks cannot be negative")
        if not self.light_item:
            raise ValueError("light_item cannot be empty")
        if self.light_pause_ticks < 0:
            raise ValueError("lighting pause_ticks cannot be negative")

    @classmethod
    def parse(cls, value=None):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("survival behaviors must be a mapping")
        value = dict(value)
        unknown = set(value) - {"hazards", "descent", "pickups", "lighting"}
        if unknown:
            raise TypeError(
                "unknown survival behavior section(s): "
                + ", ".join(sorted(unknown)))

        updates = {}
        sections = {
            "hazards": {
                "enabled": "avoid_hazards", "blocks": "hazard_blocks",
                "clearance_blocks": "hazard_clearance_blocks",
            },
            "descent": {
                "enabled": "safe_descent",
                "max_drop_blocks": "max_drop_blocks",
            },
            "pickups": {
                "enabled": "collect_drops", "attempts": "pickup_attempts",
                "wait_ticks": "pickup_wait_ticks",
                "step_blocks": "pickup_step_blocks",
            },
            "lighting": {
                "enabled": "lighting", "every_blocks": "light_every_blocks",
                "grace_blocks": "light_grace_blocks",
                "overshoot_blocks": "light_overshoot_blocks",
                "item": "light_item", "pause_ticks": "light_pause_ticks",
            },
        }
        for section, names in sections.items():
            settings = value.get(section, {})
            if isinstance(settings, bool):
                settings = {"enabled": settings}
            if not isinstance(settings, Mapping):
                raise TypeError(
                    f"survival {section} must be a mapping or bool")
            extra = set(settings) - set(names)
            if extra:
                raise TypeError(
                    f"unknown survival {section} setting(s): "
                    + ", ".join(sorted(extra)))
            for key, raw in settings.items():
                updates[names[key]] = raw
        if "hazard_blocks" in updates:
            blocks = updates["hazard_blocks"]
            if (isinstance(blocks, str)
                    or not isinstance(blocks, (list, tuple))):
                raise TypeError("survival hazards.blocks must be a list")
            updates["hazard_blocks"] = tuple(str(block) for block in blocks)
        return cls(**updates)

    def vary(self, values):
        """Apply optional per-case behavior variance independently of seeds."""
        values = dict(values or {})
        updates = {}
        for field, key in (
            ("hazard_clearance_blocks", "hazard_clearance_blocks"),
            ("max_drop_blocks", "max_drop_blocks"),
            ("pickup_attempts", "pickup_attempts"),
            ("pickup_wait_ticks", "pickup_wait_ticks"),
            ("pickup_step_blocks", "pickup_step_blocks"),
            ("light_every_blocks", "lighting_every_blocks"),
            ("light_grace_blocks", "lighting_grace_blocks"),
            ("light_overshoot_blocks", "lighting_overshoot_blocks"),
            ("light_pause_ticks", "lighting_pause_ticks"),
        ):
            if key in values:
                updates[field] = values[key]
        return replace(self, **updates) if updates else self


def _path_hazards(env, x, z, hazard_ids, clearance):
    """Visible hazard cells intersecting the current straight-line segment."""
    x0, z0 = float(env.obs["x"]), float(env.obs["z"])
    dx, dz = float(x) - x0, float(z) - z0
    length2 = dx * dx + dz * dz
    if length2 == 0:
        return ()
    feet_y = float(env.obs["y"])
    hazards = []
    for block, bx, by, bz in env.obs.get("blocks", ()):
        if int(block) not in hazard_ids:
            continue
        cell = (int(bx), int(by), int(bz))
        hx, hy, hz = cell[0] + 0.5, cell[1], cell[2] + 0.5
        if not feet_y - 1.25 <= hy <= feet_y + 1.0:
            continue
        along = max(0.0, min(1.0,
            ((hx - x0) * dx + (hz - z0) * dz) / length2))
        px, pz = x0 + along * dx, z0 + along * dz
        if math.hypot(hx - px, hz - pz) <= clearance:
            hazards.append(cell)
    return tuple(hazards)


def movement_actions(env, x, z, rng, *, profile=None, hazard_ids=(),
                     clearance_blocks=0.55, key="w", blocks=None,
                     sneak=False):
    """Emit one bounded safe movement decision toward ``(x, z)``.

    The caller remains responsible for facing the target. This controller
    handles the part that should not be optional: it checks the intended
    straight path against hazards already present in the public block scan.
    It returns a :class:`PolicyVeto` instead of executable actions when that
    segment is unsafe, and ``None`` when the target is already reached.

    ``sneak=True`` combines the movement key with Minecraft's crouch key for
    edge-safe movement. The stride and pause distribution remain those of
    :class:`MovementProfile`, capped by the remaining target distance.
    """
    profile = MovementProfile.parse(profile)
    target = (float(x), float(z))
    remaining = math.hypot(
        target[0] - float(env.obs["x"]),
        target[1] - float(env.obs["z"]))
    if remaining <= 0.05:
        return None
    hazards = _path_hazards(
        env, *target, frozenset(int(item) for item in hazard_ids),
        float(clearance_blocks))
    if hazards:
        return PolicyVeto(
            "observed_hazard",
            "visible hazard intersects the requested movement path",
            target, hazards)

    if blocks is None:
        stride = rng.uniform(profile.min_blocks, profile.max_blocks)
    else:
        stride = float(blocks)
    stride = min(remaining, stride)
    move_key: Any = key
    if sneak:
        keys = [key] if isinstance(key, str) else list(key)
        if "sneak" not in keys:
            keys.append("sneak")
        move_key = keys
    return profile.actions(rng, key=move_key, blocks=stride)


class SurvivalPolicy:
    """Objective-independent controls for ordinary survival policies."""

    def __init__(self, movement=None, behaviors=None):
        self.base_movement = MovementProfile.parse(movement)
        self.base_behaviors = SurvivalBehaviors.parse(behaviors)

    def reset(self, task, episode, case):
        del task, episode
        self.rng = random.Random(case.decision_seed)
        self.movement = self.base_movement.vary(case.values)
        self.behaviors = self.base_behaviors.vary(case.values)
        from bedrock_rl.env.catalog import resolve_blocks
        self.hazard_ids = (frozenset(resolve_blocks(
            self.behaviors.hazard_blocks))
            if self.behaviors.avoid_hazards else frozenset())
        self.lit_distances = set()
        self.lighting_distance = 0.0
        self.route_distance_total = 0.0
        self.verified_light_odometer = []
        self._lighting_position = None
        self._lighting_cell = None
        self._traversed_route_edges = set()

    def is_hazard(self, block):
        return int(block) in self.hazard_ids

    def observed_hazards(self, env, max_distance=None):
        """Hazard cells visible in the same sparse scan a model receives."""
        cells = []
        for block, x, y, z in env.obs.get("blocks", ()):
            if not self.is_hazard(block):
                continue
            cell = (int(x), int(y), int(z))
            _yaw, _pitch, distance = env.rel_angles_to(*cell)
            if max_distance is None or distance <= max_distance:
                cells.append((distance, cell))
        return [cell for _distance, cell in sorted(cells)]

    def path_is_safe(self, env, x, z):
        """Whether the observed straight path is clear of configured danger."""
        if not self.behaviors.avoid_hazards:
            return True
        return not _path_hazards(
            env, x, z, self.hazard_ids,
            self.behaviors.hazard_clearance_blocks)

    def movement_actions(self, env, x, z, rng=None, **kwargs):
        """Use this policy's movement profile and mandatory hazard check."""
        return movement_actions(
            env, x, z, self.rng if rng is None else rng,
            profile=self.movement, hazard_ids=self.hazard_ids,
            clearance_blocks=self.behaviors.hazard_clearance_blocks,
            **kwargs)

    def drop_is_safe(self, current_y, landing_y):
        return (not self.behaviors.safe_descent
                or float(current_y) - float(landing_y)
                <= self.behaviors.max_drop_blocks)

    def lighting_due(self, distance):
        distance = int(distance)
        return (self.behaviors.lighting
                and distance % self.behaviors.light_every_blocks == 0
                and distance not in self.lit_distances)

    def mark_lit(self, distance):
        self.lit_distances.add(int(distance))

    def start_route_lighting(self, env, *, immediate=True):
        """Start measuring novel route progress for an underground route."""
        self._lighting_position = tuple(
            float(env.obs[key]) for key in ("x", "y", "z"))
        self._lighting_cell = self._route_travel_cell(
            self._lighting_position)
        self._traversed_route_edges = set()
        self.lighting_distance = (
            float(self.behaviors.light_every_blocks) if immediate else 0.0)

    @staticmethod
    def _route_travel_cell(position):
        return (math.floor(float(position[0])),
                math.floor(float(position[1]) + 0.05),
                math.floor(float(position[2])))

    @staticmethod
    def _route_edge(first, second):
        """Canonical undirected route edge, so retracing is never progress."""
        return tuple(sorted((tuple(first), tuple(second))))

    def observe_route_travel(self, env):
        """Accumulate only newly explored route edges since the last light.

        Continuous controller corrections, stair settling, and backtracking
        may cross the same edge many times. They are not new tunnel progress
        and therefore must not manufacture an overdue light. Only a newly
        journal-verified placement resets interval pressure; proximity to an
        older light cannot silently stretch the promised maximum gap.
        """
        position = tuple(float(env.obs[key]) for key in ("x", "y", "z"))
        previous = self._lighting_position
        previous_cell = self._lighting_cell
        self._lighting_position = position
        cell = self._route_travel_cell(position)
        self._lighting_cell = cell
        if previous is None:
            return
        distance = math.dist(previous, position)
        # A transition, respawn, or correction is not route travel.
        plausible = max(6.0, self.base_movement.max_blocks * 3.0)
        if distance > plausible or previous_cell is None or cell == previous_cell:
            return
        edge = self._route_edge(previous_cell, cell)
        novel = edge not in self._traversed_route_edges
        self._traversed_route_edges.add(edge)
        if novel:
            route_distance = math.dist(previous_cell, cell)
            self.lighting_distance += route_distance
            self.route_distance_total += route_distance

    def route_light_due(self):
        lead = max(0.0, float(self.behaviors.light_grace_blocks))
        threshold = max(
            0.0, float(self.behaviors.light_every_blocks) - lead)
        return (self.behaviors.lighting
                and self._lighting_position is not None
                and self.lighting_distance >= threshold)

    def route_light_overdue(self):
        # Route edges are discrete (and stair/diagonal edges are longer than
        # one block), so an exact hard cutoff can reject the observation that
        # first crosses the interval. Lead/grace controls when attempts begin;
        # overshoot is a separate bounded window for finding a certified face.
        maximum = (float(self.behaviors.light_every_blocks)
                   + max(0.0, float(
                       self.behaviors.light_overshoot_blocks)))
        return self.route_light_due() and self.lighting_distance > maximum

    def route_light_placed(self):
        """Acknowledge one verified placement and restart the interval."""
        self.verified_light_odometer.append(self.route_distance_total)
        self.lighting_distance = 0.0

    def pickup_wait(self, attempt):
        if (not self.behaviors.collect_drops
                or int(attempt) >= self.behaviors.pickup_attempts):
            return None
        return [{"action": "wait",
                 "ticks": self.behaviors.pickup_wait_ticks}]

    @staticmethod
    def observed_inventory_slot(env, item_ids):
        """Latest public-journal slot that still contains a wanted item."""
        wanted = {int(item) for item in item_ids}
        observation = getattr(env, "obs", {})
        ids = observation.get("inventory_ids")
        counts = observation.get("inventory_counts")
        if ids is not None and counts is not None:
            if len(ids) != 36 or len(counts) != 36:
                raise ValueError(
                    "full inventory observation must contain 36 slots")
            return next((slot for slot in range(9, 36)
                         if int(ids[slot]) in wanted
                         and int(counts[slot]) > 0), None)
        settled = set()
        for event in reversed(env.journal):
            if event.name != "slot_changed":
                continue
            slot = int(event["slot"])
            if slot in settled or not 0 <= slot < 36:
                continue
            settled.add(slot)
            if (int(event["item"]) in wanted
                    and int(event["count"]) > 0):
                return slot
        return None

    @staticmethod
    def observed_inventory_stack(env, item_ids):
        """Latest public backpack stack as ``(slot, count, meta)``.

        Hotbar slots are already present in every observation. This helper is
        deliberately restricted to main-inventory slots 9..35 so a generic
        recipe preparer can recover a displaced ingredient without guessing
        hidden inventory state.
        """
        wanted = {int(item) for item in item_ids}
        observation = getattr(env, "obs", {})
        ids = observation.get("inventory_ids")
        counts = observation.get("inventory_counts")
        metas = observation.get("inventory_meta")
        if ids is not None and counts is not None and metas is not None:
            if not (len(ids) == len(counts) == len(metas) == 36):
                raise ValueError(
                    "full inventory observation must contain 36 slots")
            slot = next((index for index in range(9, 36)
                         if int(ids[index]) in wanted
                         and int(counts[index]) > 0), None)
            return (None if slot is None else
                    (slot, int(counts[slot]), int(metas[slot])))
        settled = set()
        for event in reversed(env.journal):
            if event.name != "slot_changed":
                continue
            slot = int(event["slot"])
            if slot in settled or not 9 <= slot < 36:
                continue
            settled.add(slot)
            if (int(event["item"]) in wanted
                    and int(event["count"]) > 0):
                return (slot, int(event["count"]),
                        int(event.get("meta", 0)))
        return None




class MinecraftController(SurvivalPolicy):
    """Objective-independent closed-loop Minecraft computer controls.

    Every method reads only the public observation and journal and returns
    ordinary ``computer`` actions, :class:`PolicyVeto`, or ``None``. Resource
    coordinates, progression order, and success conditions belong to the
    task's planner, not this controller.
    """

    def reset(self, task, episode, case):
        super().reset(task, episode, case)
        self._episode = episode
        self._walk_state = None
        self._workstation_attempts = {}
        self._furnace_sessions = {}
        self.computer_max_actions = int(
            getattr(task, "max_lines", 12) if task is not None else 12)

    @staticmethod
    def _item_ids(item):
        from bedrock_rl.env.catalog import resolve_items
        if isinstance(item, str):
            return tuple(resolve_items([item]))
        if isinstance(item, int) and not isinstance(item, bool):
            return (int(item),)
        values = []
        for value in item:
            values.extend(MinecraftController._item_ids(value))
        return tuple(dict.fromkeys(values))

    @classmethod
    def mining_hold_ticks(cls, env, cell, *, tool=None, fallback=80):
        """Return a conservative hold for the exact live-observed block.

        Tunnel routes may cross dirt, stone, ore, or obsidian while a stage
        supplies one fallback duration. Reusing an obsidian-length hold on
        dirt can break through into an undeclared voxel; reusing a soft-block
        hold on stone never clears the route. Keep this compact calibration
        in the objective-independent controller and fall back to the task's
        explicit duration for blocks it does not know.
        """

        block = cls.observed_block(env, cell)
        if block is None:
            block, target = cls.click_target(env)
            if target != tuple(int(value) for value in cell):
                return int(fallback)
        block = int(block)
        ticks = dry_mining_ticks(block, tool, fallback)

        latest_fluid = next((
            event for event in reversed(env.journal)
            if getattr(event, "name", None) == "player_fluid_state"), None)
        submerged = (latest_fluid is not None
                     and int(latest_fluid.get("flags", 0))
                     & FLUID_VIEWPOINT)
        if not submerged:
            return ticks

        return submerged_mining_ticks(block, tool, ticks)

    @staticmethod
    def _block_ids(block):
        from bedrock_rl.env.catalog import resolve_blocks
        if isinstance(block, str):
            return tuple(resolve_blocks([block]))
        if isinstance(block, int) and not isinstance(block, bool):
            return (int(block),)
        values = []
        for value in block:
            values.extend(MinecraftController._block_ids(value))
        return tuple(dict.fromkeys(values))

    @staticmethod
    def events(env, name, **values):
        """Public journal events matching a name and exact field values."""
        return [event for event in env.journal if event.name == name
                and all(event.get(key) == value
                        for key, value in values.items())]

    @classmethod
    def event_count(cls, env, name, **values):
        return len(cls.events(env, name, **values))

    @staticmethod
    def _container_kind(container):
        from bedrock_rl.env.journal import CONTAINER_KINDS
        if isinstance(container, str):
            try:
                return int(CONTAINER_KINDS[container])
            except KeyError as exc:
                raise ValueError(
                    f"unknown container kind {container!r}; expected one of "
                    f"{sorted(CONTAINER_KINDS)}") from exc
        value = int(container)
        if value not in frozenset(CONTAINER_KINDS.values()):
            raise ValueError(f"unknown container kind id {value}")
        return value

    @staticmethod
    def current_container_event(env):
        """Latest still-open container as ``(kind, cell)``, if proven.

        ``obs.container`` distinguishes the screen type but not its world
        coordinate. The event boundary supplies that missing identity, so a
        policy never treats a different furnace of the same type as its
        planned workstation.
        """
        for event in reversed(env.journal):
            if event.name == "container_closed":
                return None
            if event.name == "container_opened":
                return int(event["kind"]), (
                    int(event["x"]), int(event["y"]), int(event["z"]))
        return None

    @staticmethod
    def cursor_state(env):
        """Latest public cursor stack as ``(item, count, meta)``."""
        for event in reversed(env.journal):
            if event.name == "cursor_changed":
                return (int(event["item"]), int(event["count"]),
                        int(event.get("meta", 0)))
        return 0, 0, 0

    @staticmethod
    def observed_slot_state(env, slot, *, since=0):
        """Latest journal-proven state for an inventory/container slot."""
        for event in reversed(env.journal[int(since):]):
            if (event.name == "slot_changed"
                    and int(event["slot"]) == int(slot)):
                return (int(event["item"]), int(event["count"]),
                        int(event.get("meta", 0)))
        return None

    @classmethod
    def observed_block(cls, env, cell):
        """Block id for an exact cell in the public sparse scan, or None."""
        cell = tuple(int(value) for value in cell)
        for block, x, y, z in env.obs.get("blocks", ()):
            if (int(x), int(y), int(z)) == cell:
                return int(block)
        return None

    @classmethod
    def item_count(cls, env, item):
        """Count matching items in the publicly visible hotbar."""
        wanted = frozenset(cls._item_ids(item))
        return sum(int(count) for found, count in zip(
            env.obs["hotbar_ids"], env.obs["hotbar_counts"])
            if int(found) in wanted)

    @classmethod
    def inventory_item_count(cls, env, item):
        """Count an item across the exact public 36-slot observation."""

        wanted = frozenset(cls._item_ids(item))
        ids, counts, _metas = cls._observed_inventory(env)
        return sum(int(count) for found, count in zip(ids, counts)
                   if int(found) in wanted)

    @classmethod
    def broken(cls, env, cell, block=None, *, since=0):
        values = {"x": int(cell[0]), "y": int(cell[1]), "z": int(cell[2])}
        events = [event for event in env.journal[int(since):]
                  if event.name == "block_broken"
                  and all(event.get(key) == value
                          for key, value in values.items())]
        if block is None:
            return bool(events)
        wanted = frozenset(cls._block_ids(block))
        return any(int(event.get("block", 0)) in wanted for event in events)

    @classmethod
    def placed(cls, env, cell, block, *, since=0):
        wanted = frozenset(cls._block_ids(block))
        return any(int(event.get("block", 0)) in wanted for event in
                   env.journal[int(since):]
                   if event.name == "block_placed"
                   and int(event.get("x", 0)) == int(cell[0])
                   and int(event.get("y", 0)) == int(cell[1])
                   and int(event.get("z", 0)) == int(cell[2]))

    @staticmethod
    def click_target(env):
        """Latest exact public click ray as ``(block_id, cell)``."""
        for event in reversed(env.journal):
            if event.name == "click_target":
                return int(event["block"]), (
                    int(event["x"]), int(event["y"]), int(event["z"]))
        return 0, None

    @staticmethod
    def click_destination(env):
        """Replaceable cell selected by the latest public use ray."""
        from bedrock_rl.env.journal import CLICK_FACES
        for event in reversed(env.journal):
            if event.name != "click_target":
                continue
            offset = CLICK_FACES.get(int(event.get("face", 0)))
            if offset is None:
                return None
            return (int(event["x"]) + offset[0],
                    int(event["y"]) + offset[1],
                    int(event["z"]) + offset[2])
        return None

    @classmethod
    def nearby_cells(cls, env, block, max_distance):
        """Matching cells from the same sparse scan the model receives."""
        wanted = frozenset(cls._block_ids(block))
        cells = []
        for found, x, y, z in env.obs.get("blocks", ()):
            if int(found) not in wanted:
                continue
            cell = (int(x), int(y), int(z))
            _yaw, _pitch, distance = env.rel_angles_to(*cell)
            if distance <= float(max_distance):
                cells.append((distance, cell))
        return [cell for _distance, cell in sorted(cells)]

    @classmethod
    def select(cls, env, item, count=1, metas=None):
        """Select a visible hotbar stack; ``None`` means unavailable."""
        from bedrock_rl.env import gui_layout
        slot = gui_layout.find_stack(
            env, cls._item_ids(item), int(count), metas)
        if slot is None:
            return None
        return ([] if int(env.obs["hotbar_sel"]) == slot
                else gui_layout.select_hotbar(slot))

    def recover_hotbar(self, env, item, replace=(), *, minimum=1):
        """Make at least ``minimum`` items visible in the hotbar.

        Unstackable recipe outputs can be split between several hotbar and
        backpack slots.  Treating one visible copy as complete caused a
        multi-tool craft to repeat indefinitely while more outputs accumulated
        in the backpack.  Recovery therefore compares the requested aggregate
        and selects only a journal-proven main-inventory slot.
        """
        from bedrock_rl.env import gui_layout
        wanted = self._item_ids(item)
        minimum = int(minimum)
        if minimum < 1:
            raise ValueError("hotbar recovery minimum must be positive")
        if self.item_count(env, wanted) >= minimum:
            return []
        if self.current_container_event(env) is not None:
            # Slot geometry depends on the active screen. Close first; the
            # next closed-loop call can open the player inventory safely.
            return gui_layout.close_screen()
        stack = self.observed_inventory_stack(env, wanted)
        if stack is None:
            return None
        source, _count, source_meta = stack
        hotbar_meta = tuple(int(value) for value in env.obs.get(
            "hotbar_meta", (0,) * len(env.obs["hotbar_ids"])))
        destination = next((
            slot for slot, (found, count) in enumerate(zip(
                env.obs["hotbar_ids"], env.obs["hotbar_counts"]))
            if (int(found) in wanted and int(count) < 64
                and hotbar_meta[slot] == int(source_meta))), None)
        if destination is None:
            destination = next((slot for slot, found in enumerate(
            env.obs["hotbar_ids"]) if int(found) == 0), None)
        if destination is None:
            replacement_ids = self._item_ids(replace) if replace else ()
            destination = gui_layout.find_stack(env, replacement_ids, 1)
        if destination is None or source == destination:
            return None
        return (gui_layout.open_screen()
                + gui_layout.click_slot(source, "left", "player")
                + gui_layout.click_slot(destination, "left", "player")
                + gui_layout.close_screen())

    def recover_inventory_slot(self, env, source, *, replace=()):
        """Move one exact journal-proven backpack slot into the hotbar."""
        from bedrock_rl.env import gui_layout

        source = int(source)
        if not 9 <= source < 36:
            raise ValueError("inventory recovery source must be slot 9..35")
        if self.current_container_event(env) is not None:
            return gui_layout.close_screen()
        destination = next((slot for slot, found in enumerate(
            env.obs["hotbar_ids"]) if int(found) == 0), None)
        if destination is None:
            replacement_ids = self._item_ids(replace) if replace else ()
            destination = gui_layout.find_stack(env, replacement_ids, 1)
        if destination is None:
            return None
        return (gui_layout.open_screen()
                + gui_layout.click_slot(source, "left", "player")
                + gui_layout.click_slot(destination, "left", "player")
                + gui_layout.close_screen())

    @staticmethod
    def _observed_inventory(env):
        """Public 36-slot projection: live hotbar plus journaled backpack."""
        observed_ids = env.obs.get("inventory_ids")
        observed_counts = env.obs.get("inventory_counts")
        observed_metas = env.obs.get("inventory_meta")
        if (observed_ids is not None and observed_counts is not None
                and observed_metas is not None):
            if not (len(observed_ids) == len(observed_counts)
                    == len(observed_metas) == 36):
                raise ValueError(
                    "full inventory observation must contain 36 slots")
            return (list(map(int, observed_ids)),
                    list(map(int, observed_counts)),
                    list(map(int, observed_metas)))
        ids = [int(value) for value in env.obs["hotbar_ids"]] + [0] * 27
        counts = [int(value) for value in env.obs["hotbar_counts"]] + [0] * 27
        metas = [int(value) for value in env.obs.get(
            "hotbar_meta", [0] * 9)] + [0] * 27
        settled = set()
        for event in reversed(env.journal):
            if event.name != "slot_changed":
                continue
            slot = int(event["slot"])
            if slot in settled or not 9 <= slot < 36:
                continue
            settled.add(slot)
            ids[slot] = int(event["item"])
            counts[slot] = int(event["count"])
            metas[slot] = int(event.get("meta", 0))
        return ids, counts, metas

    def prepare_recipe_ingredients(self, env, item, grid_size, *, replace=()):
        """Recover one missing recipe ingredient using public slot events.

        Returns ``[]`` when some recipe variant is already hotbar-visible,
        recovery actions for one backpack ingredient, or ``None`` when no
        complete variant is publicly known. Repeated closed-loop calls stage
        recipes with several displaced ingredients without hidden state.
        """
        from bedrock_rl.sdg.policy.recipes import (
            WILDCARD, choose_recipe, ingredient_cells,
        )
        from bedrock_rl.env import gui_layout

        ids, counts, metas = self._observed_inventory(env)
        recipe = choose_recipe(item, ids, counts, grid_size, metas)
        if recipe is None:
            return None
        if choose_recipe(
                item, env.obs["hotbar_ids"], env.obs["hotbar_counts"],
                grid_size, env.obs.get("hotbar_meta")) is not None:
            return []
        if self.current_container_event(env) is not None:
            return gui_layout.close_screen()
        for item_id, meta, cells in ingredient_cells(recipe, grid_size):
            wanted_meta = None if meta == WILDCARD else int(meta)
            visible = sum(
                int(count) for found, count, found_meta in zip(
                    env.obs["hotbar_ids"], env.obs["hotbar_counts"],
                    env.obs.get("hotbar_meta", [0] * 9))
                if int(found) == int(item_id)
                and (wanted_meta is None
                     or int(found_meta) == wanted_meta))
            if visible >= len(cells):
                continue
            source = next((slot for slot in range(9, 36)
                           if ids[slot] == int(item_id)
                           and counts[slot] > 0
                           and (wanted_meta is None
                                or metas[slot] == wanted_meta)), None)
            if source is None:
                return None
            return self.recover_inventory_slot(
                env, source, replace=replace)
        return None

    def recover_cursor(self, env):
        """Put a journal-observed GUI cursor stack safely into inventory.

        Container clicks are deliberately multi-action transactions. If a
        run is interrupted between them, this method gives a policy one
        reusable recovery path instead of letting the next recipe click with
        an unknown carried stack. Closing through the engine's normal path is
        the final fallback because it returns the cursor to inventory.
        """
        from bedrock_rl.env import gui_layout

        _item, count, _meta = self.cursor_state(env)
        if count <= 0:
            return []
        opened = self.current_container_event(env)
        if opened is None:
            return PolicyVeto(
                "cursor_recovery_unavailable",
                "the journal reports a carried stack but no open GUI",
                (float(env.obs["x"]), float(env.obs["z"])))
        screen = {
            gui_layout.CONTAINER_PLAYER: "player",
            gui_layout.CONTAINER_TABLE: "table",
            gui_layout.CONTAINER_FURNACE: "furnace",
            gui_layout.CONTAINER_CHEST: "player",
        }[opened[0]]
        empty = next((slot for slot, found in enumerate(
            env.obs["hotbar_ids"]) if int(found) == 0), None)
        actions = [] if empty is None else gui_layout.click_slot(
            empty, "left", screen)
        return actions + gui_layout.close_screen()

    def _bounded_workstation_action(self, key, actions, *, max_attempts,
                                    code, reason, target, observed=()):
        max_attempts = int(max_attempts)
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        attempts = int(self._workstation_attempts.get(key, 0))
        if attempts >= max_attempts:
            return PolicyVeto(code, reason, target, tuple(observed))
        self._workstation_attempts[key] = attempts + 1
        return actions

    def _workstation_ready(self, env, cell, block):
        cell = tuple(int(value) for value in cell)
        wanted = frozenset(self._block_ids(block))
        observed = self.observed_block(env, cell)
        return (observed in wanted
                or any(int(event.get("block", 0)) in wanted for event in
                       self.events(env, "block_placed", x=cell[0],
                                   y=cell[1], z=cell[2])))

    def place_workstation(self, env, item, cell, support, *, block=None,
                          sneak=False, stand=None, replace=(),
                          max_attempts=4):
        """Place and live-verify one workstation at an exact destination.

        An observed conflicting block is a hard rejection. A click is only
        emitted when the public ray proves both the support and adjacent
        destination, and a finite retry budget prevents a bad face from
        spinning forever.
        """
        cell = tuple(int(value) for value in cell)
        support = tuple(int(value) for value in support)
        expected = item if block is None else block
        key = ("place", cell)
        if self._workstation_ready(env, cell, expected):
            self._workstation_attempts.pop(key, None)
            return None
        if self.current_container_event(env) is not None:
            from bedrock_rl.env import gui_layout
            return self._bounded_workstation_action(
                key, gui_layout.close_screen(), max_attempts=max_attempts,
                code="workstation_close_exhausted",
                reason="an open GUI could not be closed before placement",
                target=(cell[0] + 0.5, cell[2] + 0.5))
        observed = self.observed_block(env, cell)
        if observed not in (None, 0):
            return PolicyVeto(
                "workstation_destination_blocked",
                f"workstation destination {cell} contains block {observed}",
                (cell[0] + 0.5, cell[2] + 0.5), (cell,))
        if self.item_count(env, item) < 1:
            recovered = self.recover_hotbar(env, item, replace=replace)
            if recovered:
                return recovered
            return PolicyVeto(
                "workstation_item_unavailable",
                f"no publicly recoverable {item!r} is available to place",
                (cell[0] + 0.5, cell[2] + 0.5), (cell,))
        actions = self.place(
            env, item, cell, support, block=expected, sneak=sneak,
            stand=stand)
        if isinstance(actions, PolicyVeto):
            return actions
        if not actions:
            return PolicyVeto(
                "workstation_place_unavailable",
                f"could not produce a placement action for {cell}",
                (cell[0] + 0.5, cell[2] + 0.5), (cell,))
        if not any(action.get("action") == "right_click"
                   for action in actions):
            # Selection and camera alignment are preparation, not failed
            # placement attempts. Count only clicks that can mutate state;
            # otherwise a distant but valid face can exhaust the budget on
            # the exact observation where it first becomes clickable.
            return actions
        return self._bounded_workstation_action(
            key, actions, max_attempts=max_attempts,
            code="workstation_place_exhausted",
            reason=(f"workstation at {cell} was not verified after "
                    f"{int(max_attempts)} attempts; position="
                    f"({float(env.obs['x']):.3f}, "
                    f"{float(env.obs['y']):.3f}, "
                    f"{float(env.obs['z']):.3f}), "
                    f"click_target={self.click_target(env)[1]}, "
                    f"click_destination={self.click_destination(env)}, "
                    f"selected_slot={env.obs.get('hotbar_sel')}"),
            target=(cell[0] + 0.5, cell[2] + 0.5), observed=(cell,))

    def open_workstation(self, env, cell, *, block, container, stand=None,
                         reach=4.0, max_attempts=4):
        """Open one exact workstation and verify its type and coordinates."""
        from bedrock_rl.env import gui_layout

        cell = tuple(int(value) for value in cell)
        kind = self._container_kind(container)
        key = ("open", kind, cell)
        current = int(env.obs.get("container", gui_layout.CONTAINER_PLAYER))
        opened = self.current_container_event(env)
        if current == kind and opened == (kind, cell):
            self._workstation_attempts.pop(key, None)
            return None
        if current == kind and opened != (kind, cell):
            return PolicyVeto(
                "workstation_origin_unproven",
                f"container {kind} is open but not proven at {cell}",
                (cell[0] + 0.5, cell[2] + 0.5),
                (() if opened is None else (opened[1],)))
        if current != gui_layout.CONTAINER_PLAYER:
            return self._bounded_workstation_action(
                key, gui_layout.close_screen(), max_attempts=max_attempts,
                code="workstation_close_exhausted",
                reason="a different container could not be closed",
                target=(cell[0] + 0.5, cell[2] + 0.5))
        if opened is not None:
            return self._bounded_workstation_action(
                key, gui_layout.close_screen(), max_attempts=max_attempts,
                code="workstation_close_exhausted",
                reason="the player inventory could not be closed",
                target=(cell[0] + 0.5, cell[2] + 0.5))

        found, target = self.click_target(env)
        wanted = frozenset(self._block_ids(block))
        if target == cell and found not in wanted:
            return PolicyVeto(
                "workstation_block_mismatch",
                f"expected {block!r} at {cell}, public ray reports {found}",
                (cell[0] + 0.5, cell[2] + 0.5), (cell,))
        actions = self.use_block(
            env, cell, stand=stand, reach=reach, sneak=False)
        if isinstance(actions, PolicyVeto):
            return actions
        if not actions:
            return PolicyVeto(
                "workstation_open_unavailable",
                f"could not produce an open action for {cell}",
                (cell[0] + 0.5, cell[2] + 0.5), (cell,))
        # Long approaches remain planner-controlled and may need arbitrarily
        # many 2-3 block strides. Bound only the near-field aim/click loop.
        navigating = any(
            action.get("action") == "key"
            and (action.get("k") == "w"
                 or "w" in (action.get("k") or []))
            for action in actions)
        if navigating:
            return actions
        return self._bounded_workstation_action(
            key, actions, max_attempts=max_attempts,
            code="workstation_open_exhausted",
            reason=(f"workstation at {cell} did not open after "
                    f"{int(max_attempts)} attempts"),
            target=(cell[0] + 0.5, cell[2] + 0.5), observed=(cell,))

    def ensure_workstation(self, env, item, cell, support, *, block=None,
                           container, stand=None, sneak=False, replace=(),
                           reach=4.0, max_attempts=4):
        """Place a workstation if absent, then open that exact block."""
        expected = item if block is None else block
        if not self._workstation_ready(env, cell, expected):
            return self.place_workstation(
                env, item, cell, support, block=expected, sneak=sneak,
                replace=replace, max_attempts=max_attempts)
        return self.open_workstation(
            env, cell, block=expected, container=container, stand=stand,
            reach=reach, max_attempts=max_attempts)

    @staticmethod
    def _smelted_since(env, since, output_ids, input_ids):
        return sum(int(event.get("count", 1)) for event in
                   env.journal[int(since):]
                   if event.name == "item_smelted"
                   and int(event.get("item", 0)) in output_ids
                   and int(event.get("input", 0)) in input_ids)

    def clear_furnace_session(self, furnace):
        """Forget a completed transaction so the same recipe can run again."""
        cell = tuple(int(value) for value in furnace)
        session = self._furnace_sessions.get(cell)
        if session is not None and session.phase != "done":
            raise RuntimeError("cannot clear an active furnace transaction")
        self._furnace_sessions.pop(cell, None)

    def furnace_session_complete(self, furnace):
        """Whether the current transaction at ``furnace`` is verified done."""
        cell = tuple(int(value) for value in furnace)
        session = self._furnace_sessions.get(cell)
        return bool(session is not None and session.phase == "done")

    def smelt(self, env, furnace, input_item, output_item, *, count=1,
              fuel_item="coal", fuel_count=1, stand=None,
              output_replace=(), wait_ticks=200, max_waits=None,
              max_attempts=4, require_fresh=True, max_actions=None):
        """Run one exact, closed-loop furnace transaction.

        The method may be called once per policy turn. It opens only the
        requested furnace, loads exactly ``count`` inputs and ``fuel_count``
        fuel items through real GUI clicks, waits for journal-proven smelts,
        and moves the exact produced stack into a visible hotbar slot. Every
        irreversible phase is verified before advancing; uncertain GUI state
        is rejected rather than replayed and accidentally duplicating input.

        The furnace must already exist. Use :meth:`ensure_workstation` when a
        task also needs to place it. ``None`` means the transaction is done;
        callers can also query :meth:`furnace_session_complete`.
        """
        from bedrock_rl.env import gui_layout

        furnace = tuple(int(value) for value in furnace)
        count, fuel_count = int(count), int(fuel_count)
        wait_ticks, max_attempts = int(wait_ticks), int(max_attempts)
        max_actions = int(self.computer_max_actions if max_actions is None
                          else max_actions)
        if not 1 <= count <= 64:
            raise ValueError("furnace count must be between 1 and 64")
        if not 1 <= fuel_count <= 64:
            raise ValueError("furnace fuel_count must be between 1 and 64")
        if wait_ticks < 1:
            raise ValueError("furnace wait_ticks must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if max_actions < 3:
            raise ValueError("furnace max_actions must be at least 3")
        if max_waits is None:
            # Vanilla smelting is 200 ticks per item. Two intervals cover
            # scheduling/GUI latency without allowing an infinite stall.
            max_waits = math.ceil(count * 200 / wait_ticks) + 2
        max_waits = int(max_waits)
        if max_waits < 1:
            raise ValueError("max_waits must be positive")

        input_ids = frozenset(self._item_ids(input_item))
        output_ids = frozenset(self._item_ids(output_item))
        fuel_ids = frozenset(self._item_ids(fuel_item))
        replace_ids = (frozenset(self._item_ids(output_replace))
                       if output_replace else frozenset())
        signature = (
            tuple(sorted(input_ids)), tuple(sorted(output_ids)), count,
            tuple(sorted(fuel_ids)), fuel_count, bool(require_fresh))
        if require_fresh and not self.placed(env, furnace, "furnace"):
            return PolicyVeto(
                "furnace_freshness_unproven",
                "an exact transaction requires a furnace placed during "
                "this episode, whose initial slots are known empty",
                (furnace[0] + 0.5, furnace[2] + 0.5), (furnace,))
        session = self._furnace_sessions.get(furnace)
        if session is not None and session.signature != signature:
            return PolicyVeto(
                "furnace_session_conflict",
                "a different furnace transaction is already active at "
                f"{furnace}",
                (furnace[0] + 0.5, furnace[2] + 0.5), (furnace,))
        if session is not None and session.phase == "done":
            return None

        # Verify a completed output move before trying to reopen the GUI.
        if session is not None and session.phase == "verify_output":
            carried = self.cursor_state(env)
            if carried[1] > 0:
                recovered = self.recover_cursor(env)
                if recovered:
                    return recovered
                if isinstance(recovered, PolicyVeto):
                    return recovered
            slot = int(session.output_slot)
            before_item, before_count = session.output_slot_before
            expected_count = (before_count + count
                              if before_item in output_ids else count)
            actual_item = int(env.obs["hotbar_ids"][slot])
            actual_count = int(env.obs["hotbar_counts"][slot])
            slot_proven = any(
                event.name == "slot_changed"
                and int(event.get("slot", -1)) == slot
                and int(event.get("item", 0)) in output_ids
                and int(event.get("count", 0)) == expected_count
                for event in env.journal[session.output_start:])
            if (slot_proven
                    and actual_item in output_ids
                    and actual_count == expected_count
                    and self.item_count(env, output_ids)
                    >= session.baseline_output + count
                    and int(env.obs.get("container", 0))
                    == gui_layout.CONTAINER_PLAYER):
                session.phase = "done"
                return None
            session.verify_attempts += 1
            if session.verify_attempts >= max_attempts:
                return PolicyVeto(
                    "furnace_output_unverified",
                    f"exact {output_item!r} output did not arrive in hotbar "
                    f"slot {slot}",
                    (furnace[0] + 0.5, furnace[2] + 0.5), (furnace,))
            return [{"action": "wait", "ticks": 1}]

        # Backpack recovery must happen with no workstation GUI open. Do it
        # before open_workstation, otherwise a close/reopen loop can never
        # reach the player-inventory clicks that make the stack visible.
        if session is None or session.phase == "load":
            input_remaining = count - (
                0 if session is None else session.input_loaded)
            fuel_remaining = fuel_count - (
                0 if session is None else session.fuel_loaded)
            for item, item_ids, minimum, code, label in (
                    (input_item, input_ids, input_remaining,
                     "furnace_input_unavailable", "input"),
                    (fuel_item, fuel_ids, fuel_remaining,
                     "furnace_fuel_unavailable", "fuel")):
                if minimum <= 0:
                    continue
                if self.select(env, item_ids, minimum) is not None:
                    continue
                recovered = self.recover_hotbar(
                    env, item_ids, replace=output_replace,
                    minimum=minimum)
                if recovered:
                    return recovered
                return PolicyVeto(
                    code,
                    f"no publicly recoverable {label} stack contains "
                    f"{minimum} {item!r}",
                    (furnace[0] + 0.5, furnace[2] + 0.5), (furnace,))
            if input_remaining > 0 and fuel_remaining > 0:
                input_source = gui_layout.find_stack(
                    env, input_ids, input_remaining)
                fuel_source = gui_layout.find_stack(
                    env, fuel_ids, fuel_remaining)
                if input_source == fuel_source:
                    available = int(env.obs["hotbar_counts"][input_source])
                    if available < input_remaining + fuel_remaining:
                        return PolicyVeto(
                            "furnace_shared_stack_too_small",
                            "one stack supplies input and fuel but does not "
                            "hold their combined exact count",
                            (furnace[0] + 0.5, furnace[2] + 0.5),
                            (furnace,))

        opened = self.open_workstation(
            env, furnace, block="furnace", container="furnace",
            stand=stand, max_attempts=max_attempts)
        if opened is not None:
            return opened
        if session is None:
            session = _FurnaceSession(
                signature=signature, event_start=len(env.journal),
                baseline_output=self.item_count(env, output_ids))
            self._furnace_sessions[furnace] = session

        carried = self.cursor_state(env)
        if carried[1] > 0:
            return self.recover_cursor(env)

        if session.phase == "load":
            actions, transfers, used_slots = [], [], set()
            for kind, ids, remaining, destination in (
                    ("input", input_ids,
                     count - session.input_loaded, 46),
                    ("fuel", fuel_ids,
                     fuel_count - session.fuel_loaded, 47)):
                if remaining <= 0:
                    continue
                source = gui_layout.find_stack(env, ids, remaining)
                if source is None:
                    return PolicyVeto(
                        "furnace_hotbar_changed",
                        f"required {kind} changed while opening furnace",
                        (furnace[0] + 0.5, furnace[2] + 0.5), (furnace,))
                if source in used_slots:
                    # The first transfer changes this stack. Verify it before
                    # planning another exact reduction from the new count.
                    break
                available_actions = max_actions - len(actions)
                before = int(env.obs["hotbar_counts"][source])
                if before == remaining and available_actions >= 2:
                    removed = remaining
                    transfer = (gui_layout.click_slot(
                                    source, "left", "furnace")
                                + gui_layout.click_slot(
                                    destination, "left", "furnace"))
                else:
                    removed = min(remaining, available_actions - 2)
                    if removed < 1:
                        break
                    transfer = gui_layout.fill_cells(
                        source, (destination,) * removed, "furnace")
                actions.extend(transfer)
                transfers.append((kind, int(source), before, int(removed),
                                  tuple(sorted(ids))))
                used_slots.add(source)
            if not actions:
                return PolicyVeto(
                    "furnace_action_budget_too_small",
                    "no exact furnace transfer fits the action budget",
                    (furnace[0] + 0.5, furnace[2] + 0.5), (furnace,))
            session.load_start = len(env.journal)
            session.pending_transfers = tuple(transfers)
            session.phase = "verify_load"
            return actions

        if session.phase == "verify_load":
            events = env.journal[session.load_start:]
            load_proven = all(
                any(
                    event.name == "slot_changed"
                    and int(event.get("slot", -1)) == slot
                    and int(event.get("count", -1)) == before - removed
                    and (((before - removed) == 0
                          and int(event.get("item", -1)) == 0)
                         or ((before - removed) > 0
                             and int(event.get("item", 0)) in ids))
                    for event in events)
                for _kind, slot, before, removed, ids
                in session.pending_transfers)
            if not load_proven:
                session.verify_attempts += 1
                if session.verify_attempts >= max_attempts:
                    return PolicyVeto(
                        "furnace_load_unverified",
                        "furnace input/fuel clicks were not proven by slot "
                        "events; refusing to replay an uncertain load",
                        (furnace[0] + 0.5, furnace[2] + 0.5), (furnace,))
                return [{"action": "wait", "ticks": 1}]
            for kind, _slot, _before, removed, _ids in (
                    session.pending_transfers):
                if kind == "input":
                    session.input_loaded += removed
                else:
                    session.fuel_loaded += removed
            session.pending_transfers = ()
            session.verify_attempts = 0
            if (session.input_loaded < count
                    or session.fuel_loaded < fuel_count):
                session.phase = "load"
                return [{"action": "wait", "ticks": 1}]
            session.phase = "wait"

        if session.phase == "wait":
            smelted = self._smelted_since(
                env, session.event_start, output_ids, input_ids)
            if smelted > count:
                return PolicyVeto(
                    "furnace_output_exceeded",
                    f"journal reports {smelted} outputs for an exact "
                    f"{count}-item transaction",
                    (furnace[0] + 0.5, furnace[2] + 0.5), (furnace,))
            if smelted < count:
                if session.waits >= max_waits:
                    return PolicyVeto(
                        "furnace_smelt_stalled",
                        f"only {smelted}/{count} outputs were journaled "
                        f"after {session.waits} bounded waits",
                        (furnace[0] + 0.5, furnace[2] + 0.5), (furnace,))
                session.waits += 1
                return [{"action": "wait", "ticks": wait_ticks}]
            session.phase = "take"

        if session.phase == "take":
            destination = next((slot for slot, found in enumerate(
                env.obs["hotbar_ids"]) if int(found) == 0), None)
            if destination is None and replace_ids:
                destination = gui_layout.find_stack(env, replace_ids, 1)
            if destination is None:
                return PolicyVeto(
                    "furnace_output_hotbar_unavailable",
                    "exact furnace output needs an empty or explicitly "
                    "replaceable hotbar slot",
                    (furnace[0] + 0.5, furnace[2] + 0.5), (furnace,))
            session.output_slot = int(destination)
            session.output_slot_before = (
                int(env.obs["hotbar_ids"][destination]),
                int(env.obs["hotbar_counts"][destination]))
            session.output_start = len(env.journal)
            session.phase = "verify_output"
            session.verify_attempts = 0
            return (gui_layout.click_slot(48, "left", "furnace")
                    + gui_layout.click_slot(
                        destination, "left", "furnace")
                    + gui_layout.close_screen())

        raise RuntimeError(f"unknown furnace phase {session.phase!r}")

    def face_and_move_to(self, env, x, z, tolerance=0.55, *,
                         sequential_hop=False, step_up=False, sneak=False,
                         allow_jump=True, swim=False,
                         movement_cap_blocks=None):
        """Face and take one hazard-checked, bounded step toward a point."""
        from bedrock_rl.adapters.netherite.computer import moves_for_angles
        from bedrock_rl.env.engine import wrap180

        x, z = float(x), float(z)
        dx, dz = x - float(env.obs["x"]), z - float(env.obs["z"])
        distance = math.hypot(dx, dz)
        if distance <= float(tolerance):
            self._walk_state = None
            return None
        previous = self._walk_state
        stuck = bool(previous and abs(previous[0] - x) < 0.01
                     and abs(previous[1] - z) < 0.01
                     and math.hypot(previous[2] - float(env.obs["x"]),
                                    previous[3] - float(env.obs["z"])) < 0.04)
        self._walk_state = (x, z, float(env.obs["x"]), float(env.obs["z"]))
        target_yaw = math.degrees(math.atan2(-dx, dz))
        turn = moves_for_angles(
            wrap180(target_yaw - float(env.obs["yaw"])),
            -float(env.obs["pitch"]), float(env.obs["pitch"]))
        travel = max(0.05, distance - float(tolerance))
        if movement_cap_blocks is not None:
            cap = float(movement_cap_blocks)
            if cap <= 0:
                raise ValueError("movement_cap_blocks must be positive")
            travel = min(travel, cap)
        key = (["w", "space", "sprint"] if swim else
               ["w", "space"] if allow_jump and (step_up or (
                    stuck and not sequential_hop)) else "w")
        move = self.movement_actions(
            env, x, z, key=key, blocks=travel, sneak=sneak)
        if isinstance(move, PolicyVeto):
            return move
        if move is None:
            return turn or None
        hop = ([{"action": "key", "k": "space", "ticks": 2}]
               if allow_jump and not swim and stuck and sequential_hop
               else [])
        return turn + hop + move

    def recenter_interaction_stance(self, env, x, z, *, tolerance=0.02):
        """Micro-correct a stance without turning the camera away.

        Ordinary route movement faces the destination first. That is exactly
        wrong while preparing a click: alternating around a cell center can
        rotate 180 degrees every attempt and erase the target aim. Choose the
        closest of forward/back/strafe in the player's current frame and use
        one crouched tick, preserving the camera for the subsequent exact aim
        proof.  On the pinned engine a settled ordinary tick travels about
        0.210 blocks, while a crouched tick travels about 0.059; the latter
        turns the observed 0.081--0.087 stance errors into <=0.028 rather than
        overshooting into the same oscillation from the other side.
        """

        from bedrock_rl.env.engine import wrap180

        dx = float(x) - float(env.obs["x"])
        dz = float(z) - float(env.obs["z"])
        distance = math.hypot(dx, dz)
        if distance <= float(tolerance):
            return None
        target_yaw = math.degrees(math.atan2(-dx, dz))
        relative = wrap180(target_yaw - float(env.obs["yaw"]))
        candidates = (("w", 0.0), ("d", 90.0),
                      ("s", 180.0), ("a", -90.0))
        key, _angle = min(
            candidates,
            key=lambda item: abs(wrap180(relative - item[1])))
        return [{"action": "key", "k": [key, "sneak"], "ticks": 1}]

    @staticmethod
    def aim(env, cell, *, tolerance=0.12, max_moves=1):
        """Aim at a block cell without clicking it."""
        from bedrock_rl.adapters.netherite.computer import moves_for_angles
        yaw, pitch, _distance = env.rel_angles_to(*cell)
        return moves_for_angles(
            yaw, pitch, env.obs["pitch"], tol=float(tolerance))[
                :int(max_moves)]

    @staticmethod
    def aim_point(env, point, *, tolerance=0.12, max_moves=1):
        """Aim using the environment's block-coordinate convention."""
        from bedrock_rl.adapters.netherite.computer import moves_for_angles
        yaw, pitch, _distance = env.rel_angles_to(*point)
        return moves_for_angles(
            yaw, pitch, env.obs["pitch"], tol=float(tolerance))[
                :int(max_moves)]

    def use_block(self, env, cell, *, stand=None, reach=4.0, sneak=False):
        """Right-click only after the public ray proves the exact block."""
        cell = tuple(int(value) for value in cell)
        _yaw, _pitch, distance = env.rel_angles_to(*cell)
        if distance > float(reach):
            target = (cell[0] + 0.5, cell[2] + 0.5)
            if stand is not None:
                target = (float(stand[0]), float(stand[1]))
            return self.face_and_move_to(
                env, *target, tolerance=max(0.55, float(reach) - 1.0))
        _block, target = self.click_target(env)
        if target == cell:
            action = {"action": "right_click"}
            if sneak:
                action["sneak"] = True
            return [action]
        aim = self.aim(env, cell)
        if aim:
            return aim
        return PolicyVeto(
            "use_target_unproven",
            f"refusing to use {cell}; public click target is {target}",
            (cell[0] + 0.5, cell[2] + 0.5),
            (() if target is None else (target,)))

    def mine_exact(self, env, cell, *, tool=None, ticks=80, block=None,
                   stand=None, reach=None, since=0):
        """Mine only after the live public click ray proves the exact voxel."""
        from bedrock_rl.sdg.policy.spatial import MINING_REACH
        reach = MINING_REACH if reach is None else float(reach)
        cell = tuple(int(value) for value in cell)
        if self.broken(env, cell, block, since=since):
            return None
        selected = [] if tool is None else self.select(env, tool)
        if selected is None:
            return None
        # Selecting a hotbar slot and beginning an irreversible mining hold
        # in one computer call can race the engine's held-item transition.
        # The target may then break under the previously selected item and
        # legitimately emit no drop. Observe the selected tool on the next
        # turn before aiming or attacking, just as exact placement already
        # does for its material stack.
        if selected:
            return selected
        _yaw, _pitch, distance = env.rel_angles_to(*cell)
        if distance > float(reach):
            if stand is not None:
                return self.face_and_move_to(env, stand[0], stand[1])
            return PolicyVeto(
                "mine_target_out_of_reach",
                f"exact mine target {cell} is {distance:.2f} blocks away",
                (cell[0] + 0.5, cell[2] + 0.5), (cell,))
        found, target = self.click_target(env)
        expected = (None if block is None
                    else frozenset(self._block_ids(block)))
        if target == cell and (expected is None or found in expected):
            return [{
                "action": "left_click_hold", "ticks": int(ticks)}]
        # Correct camera error before optional stance micro-correction. A
        # valid planned stand can be a few centimeters off center while the
        # camera is still facing a workstation behind the player; moving
        # around that center first only oscillates without ever addressing
        # the wrong ray. Exact click proof above remains the sole attack gate.
        aim = self.aim(env, cell)
        if aim:
            return aim
        if (stand is not None and math.hypot(
                float(env.obs["x"]) - float(stand[0]),
                float(env.obs["z"]) - float(stand[1])) > 0.03):
            # The planned center ray can be visible while a tolerance-edge
            # pose clips the neighboring voxel. Recenter inside the already
            # proved stance, then let the next observation prove the click;
            # never attack the adjacent block reported by the stale ray.
            move = self.recenter_interaction_stance(
                env, float(stand[0]), float(stand[1]), tolerance=0.02)
            if move is not None:
                return move
        return PolicyVeto(
            "mine_target_unproven",
            f"refusing to mine {cell}; public click target is "
            f"({found}, {target}); "
            f"pose=({float(env.obs['x']):.3f}, "
            f"{float(env.obs['y']):.3f}, {float(env.obs['z']):.3f}, "
            f"yaw={float(env.obs['yaw']):.3f}, "
            f"pitch={float(env.obs['pitch']):.3f}); "
            f"planned_stand={stand}",
            (cell[0] + 0.5, cell[2] + 0.5),
            (() if target is None else (target,)))

    def craft_2x2(self, env, item, *, output_replace=()):
        """Perform one public inventory-GUI 2x2 craft."""
        from bedrock_rl.sdg.policy.oracle import gui_craft_2x2
        return gui_craft_2x2(
            env, item, output_replace=output_replace) or None

    def craft_3x3(self, env, item, *, table=None, gui_open=False,
                  output_replace=()):
        """Advance one closed-loop 3x3 craft using an existing table."""
        from bedrock_rl.sdg.policy.oracle import gui_craft_3x3
        from bedrock_rl.env import gui_layout

        if (gui_open and int(env.obs["container"])
                != gui_layout.CONTAINER_TABLE):
            return gui_layout.close_screen()
        if int(env.obs["container"]) == gui_layout.CONTAINER_TABLE:
            if table is not None:
                expected = (
                    gui_layout.CONTAINER_TABLE,
                    tuple(int(value) for value in table),
                )
                opened = self.current_container_event(env)
                if opened != expected:
                    return PolicyVeto(
                        "crafting_table_origin_unproven",
                        "a crafting screen is open but its exact planned "
                        f"table {expected[1]} is not journal-proven",
                        (expected[1][0] + 0.5, expected[1][2] + 0.5),
                        (() if opened is None else (opened[1],)))
            return gui_craft_3x3(
                env, item, self.rng,
                output_replace=output_replace) or None
        tables = ([tuple(int(value) for value in table)] if table is not None
                  else self.nearby_cells(
                      env, gui_layout.CRAFTING_TABLE_BLOCK, 16.0))
        if not tables:
            return PolicyVeto(
                "crafting_table_unavailable",
                "3x3 crafting requires an observed or supplied table",
                (float(env.obs["x"]), float(env.obs["z"])))
        target = tables[0]
        _yaw, _pitch, distance = env.rel_angles_to(*target)
        if distance > 4.0:
            return self.face_and_move_to(
                env, target[0] + 0.5, target[2] + 0.5, tolerance=3.0)
        return self.use_block(env, target, reach=4.0)

    def place(self, env, item, cell, support, *, block=None, sneak=False,
              stand=None, jump_from_y=None, since=0):
        """Place only when the public ray proves the exact destination."""
        cell = tuple(int(value) for value in cell)
        support = tuple(int(value) for value in support)
        expected = item if block is None else block
        if self.placed(env, cell, expected, since=since):
            return None
        selected = self.select(env, item)
        if selected is None:
            return None
        # Selecting a hotbar slot and using it in the same computer call can
        # race the engine's held-item update. Observe the selected stack on
        # the next turn before any irreversible placement click.
        if selected:
            return selected
        delta = tuple(cell[index] - support[index] for index in range(3))
        if sum(abs(value) for value in delta) != 1:
            raise ValueError("place cell must be face-adjacent to support")
        _found, target = self.click_target(env)
        if target == support and self.click_destination(env) == cell:
            if (stand is not None and math.hypot(
                    float(env.obs["x"]) - float(stand[0]),
                    float(env.obs["z"]) - float(stand[1])) > 0.05):
                # The click ray can be exact while the player's collision
                # box still overlaps an adjacent destination.  Minecraft
                # silently ignores that placement, so center within the
                # already certified stance before the irreversible click.
                move = self.recenter_interaction_stance(
                    env, float(stand[0]), float(stand[1]), tolerance=0.02)
                if move is not None:
                    return selected + move
            action = {"action": "right_click"}
            if sneak:
                action["sneak"] = True
            return selected + [action]
        from bedrock_rl.sdg.policy.spatial import (
            placement_face_point,
        )
        # rel_angles_to adds 0.5 to each argument internally. Convert the
        # shared physical face point back to that block-coordinate convention
        # so live aim and preflight certify the same support face.
        observer = (
            float(env.obs["x"]),
            float(env.obs["y"]) + 1.62,
            float(env.obs["z"]),
        )
        point = tuple(value - 0.5 for value in placement_face_point(
            support, cell, observer=observer))
        if (jump_from_y is not None and delta == (0, 1, 0)
                and abs(float(env.obs["y"])
                        - float(jump_from_y)) <= 0.2):
            # Aim for the same physical face from the measured two-tick jump
            # apex before leaving the ground.  Mouse corrections consume
            # simulation ticks, so aiming only after takeoff lands the player
            # before a tall top face can become the public click target.
            from bedrock_rl.adapters.netherite.computer import (
                moves_for_angles,
            )
            from bedrock_rl.env.engine import look_angles

            apex = dict(env.obs)
            apex["y"] = float(jump_from_y) + 1.166
            dyaw, dpitch, _distance = look_angles(apex, *point)
            preaim = moves_for_angles(
                dyaw, dpitch, float(env.obs["pitch"]))
            return selected + preaim + [{
                "action": "key", "k": "space", "ticks": 2}]
        aim = self.aim_point(env, point)
        if aim:
            return selected + aim
        if (stand is not None and math.hypot(
                float(env.obs["x"]) - float(stand[0]),
                float(env.obs["z"]) - float(stand[1])) > 0.03):
            move = self.recenter_interaction_stance(
                env, float(stand[0]), float(stand[1]), tolerance=0.02)
            if move is not None:
                return selected + move
        return PolicyVeto(
            "place_destination_unproven",
            f"refusing to place at {cell}; public destination is "
            f"{self.click_destination(env)}; pose="
            f"({float(env.obs['x']):.3f}, "
            f"{float(env.obs['y']):.3f}, {float(env.obs['z']):.3f}, "
            f"yaw={float(env.obs['yaw']):.3f}, "
            f"pitch={float(env.obs['pitch']):.3f}); "
            f"planned_stand={stand}",
            (cell[0] + 0.5, cell[2] + 0.5),
            (() if target is None else (target,)))

    def pickup(self, env, item, *, minimum=1, attempt=0, target=None,
               replace=(), sneak=False, allow_jump=False):
        """Recover or approach a drop, then use the bounded pickup wait."""
        if self.item_count(env, item) >= int(minimum):
            return None
        recovered = self.recover_hotbar(env, item, replace=replace)
        if recovered:
            return recovered
        if target is not None:
            previous = self._walk_state
            edge_stuck = bool(
                previous
                and abs(previous[0] - float(target[0])) < 0.01
                and abs(previous[1] - float(target[1])) < 0.01
                and math.hypot(
                    previous[2] - float(env.obs["x"]),
                    previous[3] - float(env.obs["z"])) < 0.04)
            if edge_stuck and bool(sneak):
                # A planned ledge approach intentionally stops at its proved
                # support edge. Give the nearby entity time to enter pickup
                # range instead of issuing the same blocked hold forever.
                return self.pickup_wait(attempt)
            # Preserve the planner's edge semantic. An explicit ledge
            # approach crouches and advances one observed input tick at a
            # time. A multi-tick hold can cross the item column before the
            # next frame and then repeat from stale edge geometry, producing
            # several blocks of drift. Safe neighboring/descent cells still
            # use the configured pickup stride so a proved one-block descent
            # is not starved under the attempt limit.
            movement_cap = self.behaviors.pickup_step_blocks
            if sneak:
                movement_cap = min(
                    movement_cap, 1.0 / self.movement.ticks_per_block)
            move = self.face_and_move_to(
                env, float(target[0]), float(target[1]), tolerance=0.1,
                sneak=bool(sneak), allow_jump=bool(allow_jump),
                movement_cap_blocks=movement_cap)
            if move is not None:
                return move
        return self.pickup_wait(attempt)
