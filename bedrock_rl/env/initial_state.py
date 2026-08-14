"""Declarative Netherite initial state and capability validation."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Mapping

from bedrock_rl.env.catalog import resolve_blocks, resolve_items


TOP_KEYS = frozenset(("inventory", "player", "spawn", "world", "blocks"))
PLAYER_KEYS = frozenset(("health", "food", "saturation", "exhaustion",
                         "selected_item"))
WORLD_KEYS = frozenset(("mode", "difficulty", "weather", "time", "mobs"))
SPAWN_KEYS = frozenset(("points", "constraints", "hazard_clearance"))
ITEM_KEYS = frozenset(("item", "count", "slot", "meta"))
POINT_KEYS = frozenset(("x", "y", "z", "yaw", "pitch"))
CONSTRAINT_KEYS = frozenset(("block", "min_distance", "max_distance"))
BLOCK_KEYS = frozenset(("block", "x", "y", "z", "meta"))


def _mapping(value, what):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{what} must be a mapping")
    return dict(value)


def _closed(value, keys, what):
    unknown = sorted(set(value) - keys)
    if unknown:
        raise ValueError(f"unknown {what} keys: {', '.join(unknown)}")


@dataclass(frozen=True)
class InventoryStack:
    item: str
    count: int = 1
    slot: int | str = "random_hotbar"
    meta: int = 0

    @classmethod
    def parse(cls, value):
        raw = _mapping(value, "initial_state.inventory item")
        _closed(raw, ITEM_KEYS, "initial_state.inventory item")
        item = str(raw.get("item") or "").strip()
        if not item:
            raise ValueError("initial inventory item needs item")
        resolve_items([item])
        count = int(raw.get("count", 1))
        if not 1 <= count <= 64:
            raise ValueError("initial inventory count must be in 1..64")
        slot = raw.get("slot", "random_hotbar")
        if isinstance(slot, str):
            if slot not in ("random_hotbar", "random_inventory"):
                raise ValueError("inventory slot must be 0..36, "
                                 "random_hotbar, or random_inventory")
        elif not 0 <= int(slot) <= 36:
            raise ValueError("inventory slot must be in 0..36")
        else:
            slot = int(slot)
        return cls(item, count, slot, int(raw.get("meta", 0)))


@dataclass(frozen=True)
class SpawnPoint:
    x: float
    y: float
    z: float
    yaw: float | None = None
    pitch: float | None = None

    @classmethod
    def parse(cls, value):
        raw = _mapping(value, "initial_state.spawn point")
        _closed(raw, POINT_KEYS, "initial_state.spawn point")
        missing = [key for key in ("x", "y", "z") if key not in raw]
        if missing:
            raise ValueError("spawn point needs " + ", ".join(missing))
        return cls(float(raw["x"]), float(raw["y"]), float(raw["z"]),
                   None if raw.get("yaw") is None else float(raw["yaw"]),
                   None if raw.get("pitch") is None else float(raw["pitch"]))


@dataclass(frozen=True)
class BlockConstraint:
    block: str
    min_distance: float = 0.0
    max_distance: float = 16.0
    ids: tuple[int, ...] = field(default=(), repr=False)

    @classmethod
    def parse(cls, value):
        raw = _mapping(value, "initial_state.spawn constraint")
        _closed(raw, CONSTRAINT_KEYS, "initial_state.spawn constraint")
        block = str(raw.get("block") or "").strip()
        if not block:
            raise ValueError("spawn constraint needs block")
        low = float(raw.get("min_distance", 0.0))
        high = float(raw.get("max_distance", 16.0))
        if low < 0 or high < low:
            raise ValueError("spawn distance needs 0 <= min <= max")
        return cls(block, low, high, tuple(resolve_blocks([block])))


def _axis(value, what):
    """One coordinate or an inclusive ``[low, high]`` range."""
    if isinstance(value, bool):
        raise TypeError(f"{what} must be an integer or [low, high]")
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"{what} range must contain [low, high]")
        low, high = (int(part) for part in value)
    else:
        low = high = int(value)
    if high < low:
        raise ValueError(f"{what} range needs low <= high")
    return low, high


@dataclass(frozen=True)
class BlockVolume:
    """An ordered block or cuboid edit applied to the initial snapshot.

    Later entries win, which lets a task clear an arena, lay a floor, then
    place resources without a task-specific setup script.
    """

    block: str
    x: tuple[int, int]
    y: tuple[int, int]
    z: tuple[int, int]
    meta: int = 0
    block_id: int = field(default=0, repr=False)

    @classmethod
    def parse(cls, value):
        raw = _mapping(value, "initial_state block")
        _closed(raw, BLOCK_KEYS, "initial_state block")
        missing = [key for key in ("block", "x", "y", "z")
                   if key not in raw]
        if missing:
            raise ValueError("initial_state block needs " + ", ".join(missing))
        block = str(raw["block"]).strip()
        ids = resolve_blocks([block])
        if len(ids) != 1:
            raise ValueError(
                f"initial_state block {block!r} resolves to {ids}; choose a "
                "single block-state name")
        meta = int(raw.get("meta", 0))
        if not 0 <= meta <= 15:
            raise ValueError("initial_state block meta must be in 0..15")
        return cls(block, _axis(raw["x"], "block.x"),
                   _axis(raw["y"], "block.y"),
                   _axis(raw["z"], "block.z"), meta, ids[0])

    def cells(self):
        for x in range(self.x[0], self.x[1] + 1):
            for y in range(self.y[0], self.y[1] + 1):
                for z in range(self.z[0], self.z[1] + 1):
                    yield x, y, z, self.block_id, self.meta

    def to_dict(self):
        def axis(value):
            return value[0] if value[0] == value[1] else list(value)
        return {"block": self.block, "x": axis(self.x), "y": axis(self.y),
                "z": axis(self.z), "meta": self.meta}


@dataclass(frozen=True)
class InitialStateSpec:
    inventory: tuple[InventoryStack, ...] = ()
    player: dict[str, Any] = field(default_factory=dict)
    points: tuple[SpawnPoint, ...] = ()
    constraints: tuple[BlockConstraint, ...] = ()
    spawn_hazard_clearance: int = 0
    world: dict[str, Any] = field(default_factory=dict)
    blocks: tuple[BlockVolume, ...] = ()

    @classmethod
    def parse(cls, value):
        raw = _mapping(value, "initial_state")
        _closed(raw, TOP_KEYS, "initial_state")
        if not raw:
            return cls()
        inventory = raw.get("inventory") or []
        if not isinstance(inventory, list):
            raise TypeError("initial_state.inventory must be a list")
        stacks = tuple(InventoryStack.parse(item) for item in inventory)

        blocks = raw.get("blocks") or []
        if not isinstance(blocks, list):
            raise TypeError("initial_state.blocks must be a list")
        blocks = tuple(BlockVolume.parse(item) for item in blocks)

        player = _mapping(raw.get("player"), "initial_state.player")
        _closed(player, PLAYER_KEYS, "initial_state.player")
        selected_item = player.pop("selected_item", None)
        player = {key: float(value) for key, value in player.items()}
        if selected_item is not None:
            selected_item = str(selected_item).strip()
            if not selected_item:
                raise ValueError("initial player selected_item cannot be empty")
            resolve_items([selected_item])
            player["selected_item"] = selected_item
        for key in ("health", "food"):
            if key in player and not 0 <= player[key] <= 20:
                raise ValueError(f"initial player {key} must be in 0..20")
        for key in ("saturation", "exhaustion"):
            if key in player and player[key] < 0:
                raise ValueError(f"initial player {key} cannot be negative")

        spawn = _mapping(raw.get("spawn"), "initial_state.spawn")
        _closed(spawn, SPAWN_KEYS, "initial_state.spawn")
        points = spawn.get("points") or []
        constraints = spawn.get("constraints") or []
        if not isinstance(points, list) or not isinstance(constraints, list):
            raise TypeError("spawn.points and spawn.constraints must be lists")
        hazard_clearance = spawn.get("hazard_clearance", 0)
        if isinstance(hazard_clearance, bool):
            raise TypeError("spawn.hazard_clearance must be an integer")
        hazard_clearance = int(hazard_clearance)
        if hazard_clearance < 0:
            raise ValueError("spawn.hazard_clearance cannot be negative")

        world = _mapping(raw.get("world"), "initial_state.world")
        _closed(world, WORLD_KEYS, "initial_state.world")
        mode = str(world.get("mode", "survival")).lower()
        difficulty = str(world.get("difficulty", "normal")).lower()
        weather = str(world.get("weather", "clear")).lower()
        if mode != "survival":
            raise ValueError("Netherite currently supports mode: survival only")
        if difficulty != "normal":
            raise ValueError("Netherite currently supports difficulty: normal only")
        if weather != "clear":
            raise ValueError("Netherite currently supports weather: clear only")
        if "mode" in world:
            world["mode"] = mode
        if "difficulty" in world:
            world["difficulty"] = difficulty
        if "weather" in world:
            world["weather"] = weather
        if "mobs" in world:
            if not isinstance(world["mobs"], bool):
                raise TypeError("initial world mobs must be true or false")
        if world.get("time") is not None:
            if isinstance(world["time"], bool):
                raise TypeError("initial world time must be an integer")
            world["time"] = int(world["time"])
            if world["time"] < 0:
                raise ValueError("initial world time cannot be negative")
        return cls(
            inventory=stacks, player=player,
            points=tuple(SpawnPoint.parse(point) for point in points),
            constraints=tuple(
                BlockConstraint.parse(item) for item in constraints),
            spawn_hazard_clearance=hazard_clearance,
            world=world, blocks=blocks)

    @property
    def enabled(self):
        return bool(self.inventory or self.player or self.points
                    or self.constraints or self.spawn_hazard_clearance
                    or self.world or self.blocks)

    @property
    def cache_key(self):
        body = json.dumps(self.to_dict(), sort_keys=True,
                          separators=(",", ":"))
        return hashlib.sha256(body.encode()).hexdigest()[:12]

    def to_dict(self):
        return {
            "inventory": [dict(item=item.item, count=item.count,
                               slot=item.slot, meta=item.meta)
                          for item in self.inventory],
            "player": dict(self.player),
            "spawn": {
                "points": [dict(x=p.x, y=p.y, z=p.z, yaw=p.yaw,
                                pitch=p.pitch) for p in self.points],
                "constraints": [dict(block=c.block,
                                     min_distance=c.min_distance,
                                     max_distance=c.max_distance)
                                for c in self.constraints],
                "hazard_clearance": self.spawn_hazard_clearance,
            },
            "world": dict(self.world),
            "blocks": [block.to_dict() for block in self.blocks],
        }

    def point_for(self, seed):
        if not self.points:
            return None
        return self.points[random.Random(int(seed) ^ 0x51A7E).randrange(
            len(self.points))]

    def inventory_slots(self, seed):
        rng = random.Random(int(seed) ^ 0x1A2B3C)
        explicit = {}
        for stack in self.inventory:
            if not isinstance(stack.slot, int):
                continue
            if stack.slot in explicit:
                raise ValueError(
                    f"two initial inventory stacks use slot {stack.slot}")
            explicit[stack.slot] = stack
        # Fixed slots are reserved before randomized entries are placed. A
        # fixed entry later in YAML must not make task validity depend on the
        # seed used for an earlier random entry.
        available_hotbar = [slot for slot in range(9)
                            if slot not in explicit]
        available_all = [slot for slot in range(37)
                         if slot not in explicit]
        slots = {}
        for stack in self.inventory:
            if isinstance(stack.slot, int):
                slot = stack.slot
            else:
                choices = (available_hotbar if stack.slot == "random_hotbar"
                           else available_all)
                choices = [value for value in choices if value not in slots]
                if not choices:
                    raise ValueError("not enough free inventory slots")
                slot = rng.choice(choices)
            slots[slot] = (resolve_items([stack.item])[0], stack.count,
                           stack.meta)
        return slots

    def validate_constraints(self, observation):
        for constraint in self.constraints:
            distances = []
            for block, x, y, z in observation["blocks"]:
                if block not in constraint.ids:
                    continue
                distances.append(math.sqrt(
                    (x + 0.5 - observation["x"]) ** 2
                    + (y + 0.5 - observation["y"]) ** 2
                    + (z + 0.5 - observation["z"]) ** 2))
            nearest = min(distances, default=math.inf)
            if not (constraint.min_distance <= nearest
                    <= constraint.max_distance):
                raise ValueError(
                    f"spawn's nearest {constraint.block} is not between "
                    f"{constraint.min_distance:g} and "
                    f"{constraint.max_distance:g} blocks")
