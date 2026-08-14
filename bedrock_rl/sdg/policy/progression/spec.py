"""Declarative contracts and planning helpers for survival progression.

The policy in this module contains no world coordinates and no preferred
world seed.  A snapshot planner chooses every natural resource and route;
the live executor proves every break, pickup, craft, smelt, placement, and
dimension change before the state machine advances.  Worlds for which that
proof cannot be constructed are ordinary synthetic-data rejections.

``SurvivalProgressionSpec`` is deliberately data-only.  The shipped Nether
portal recipe is one preset, not framework logic: callers can replace the
resource stages and their milestone operations without subclassing the
controller or planner.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, replace
from typing import Iterable, Mapping, TypeAlias

from bedrock_rl.sdg.policy.mining import (
    LOG_TOOL_TICKS, OBSIDIAN_DIAMOND_TICKS, PICK_TOOL_TICKS,
    SOFT_BLOCK_TICKS, UNKNOWN_TUNNEL_TICKS,
)
from bedrock_rl.sdg.policy.planning import (
    GoalSpec,
    ResourceRequirement,
    WorldPlan,
)
from bedrock_rl.sdg.policy.spatial import (
    DEFAULT_BLOCK_RULES,
    PlanRejected,
    clear_placement_ray,
    clear_voxel_ray,
    is_surface_stand,
    plan_safe_removal,
    plan_surface_path,
)
from bedrock_rl.sdg.policy.structures import (
    StructurePlan,
)
from bedrock_rl.sdg.generation import CaseRejected
from bedrock_rl.env.catalog import resolve_blocks


Cell: TypeAlias = tuple[int, int, int]


# Usable block breaks in the pinned Minecraft 1.11 runtime.  Durability is an
# engine rule, not objective policy, so every data-only progression shares the
# same table.  Custom tools must provide a recognized vanilla material name
# rather than silently running past an unproved capacity.
_VANILLA_TOOL_BREAKS = {
    "wooden": 59,
    "golden": 32,
    "stone": 131,
    "iron": 250,
    "diamond": 1561,
}


def _tool_break_capacity(tool):
    if tool is None:
        return None
    name = str(tool)
    material = name.split("_", 1)[0]
    return _VANILLA_TOOL_BREAKS.get(material)


def _cell(value: Iterable[int], label: str = "cell") -> Cell:
    result = tuple(value)
    if len(result) != 3 or any(
            not isinstance(item, int) or isinstance(item, bool)
            for item in result):
        raise TypeError(f"{label} must contain exactly three integers")
    return result


def _mapping(value, label):
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return dict(value)


def _closed_mapping(value, label, keys):
    result = _mapping(value, label)
    unknown = sorted(set(result) - set(keys))
    if unknown:
        raise ValueError(
            f"unknown {label} key{'' if len(unknown) == 1 else 's'}: "
            + ", ".join(unknown))
    return result


def _boolean(value, label):
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be true or false")
    return value


@dataclass(frozen=True, slots=True)
class CraftOperation:
    """Craft until at least ``minimum`` output items are live-observable."""

    item: str
    minimum: int
    grid: int
    table: str | None = None

    def __post_init__(self):
        if not self.item.strip():
            raise ValueError("craft item cannot be empty")
        if self.minimum < 1:
            raise ValueError("craft minimum must be positive")
        if self.grid not in (2, 3):
            raise ValueError("craft grid must be 2 or 3")
        if self.grid == 3 and not self.table:
            raise ValueError("3x3 craft needs a workstation key")
        if self.grid == 2 and self.table is not None:
            raise ValueError("2x2 craft cannot name a workstation")

    @classmethod
    def parse(cls, value):
        value = _closed_mapping(value, "craft operation", {
            "type", "item", "minimum", "grid", "table"})
        return cls(
            str(value["item"]), int(value.get("minimum", 1)),
            int(value.get("grid", 3)), value.get("table"))

    def to_dict(self):
        result = {"type": "craft", "item": self.item,
                  "minimum": self.minimum, "grid": self.grid}
        if self.table is not None:
            result["table"] = self.table
        return result


@dataclass(frozen=True, slots=True)
class WorkstationOperation:
    """Place one exact workstation at a planner-selected nearby site."""

    key: str
    item: str
    block: str
    container: str

    def __post_init__(self):
        if not all(value.strip() for value in (
                self.key, self.item, self.block, self.container)):
            raise ValueError("workstation fields cannot be empty")

    @classmethod
    def parse(cls, value):
        value = _closed_mapping(value, "workstation operation", {
            "type", "key", "item", "block", "container"})
        item = str(value["item"])
        return cls(str(value["key"]), item,
                   str(value.get("block", item)),
                   str(value.get("container", item)))

    def to_dict(self):
        return {"type": "workstation", "key": self.key,
                "item": self.item, "block": self.block,
                "container": self.container}


@dataclass(frozen=True, slots=True)
class SmeltOperation:
    """Run one exact transaction in a freshly placed furnace."""

    furnace: str
    input_item: str
    output_item: str
    count: int
    fuel_item: str = "coal"
    fuel_count: int = 1

    def __post_init__(self):
        if not all(value.strip() for value in (
                self.furnace, self.input_item, self.output_item,
                self.fuel_item)):
            raise ValueError("smelt fields cannot be empty")
        if self.count < 1 or self.fuel_count < 1:
            raise ValueError("smelt counts must be positive")

    @classmethod
    def parse(cls, value):
        value = _closed_mapping(value, "smelt operation", {
            "type", "furnace", "input", "output", "count", "fuel",
            "fuel_count"})
        return cls(
            str(value["furnace"]), str(value["input"]),
            str(value["output"]), int(value.get("count", 1)),
            str(value.get("fuel", "coal")),
            int(value.get("fuel_count", 1)))

    def to_dict(self):
        return {"type": "smelt", "furnace": self.furnace,
                "input": self.input_item, "output": self.output_item,
                "count": self.count, "fuel": self.fuel_item,
                "fuel_count": self.fuel_count}


@dataclass(frozen=True, slots=True)
class RequireItemOperation:
    """Reject a run unless a stochastic/branching drop was obtained."""

    item: str
    minimum: int = 1

    def __post_init__(self):
        if not self.item.strip() or self.minimum < 1:
            raise ValueError("required item and minimum must be positive")

    @classmethod
    def parse(cls, value):
        value = _closed_mapping(value, "require operation", {
            "type", "item", "minimum"})
        return cls(str(value["item"]), int(value.get("minimum", 1)))

    def to_dict(self):
        return {"type": "require", "item": self.item,
                "minimum": self.minimum}


@dataclass(frozen=True, slots=True)
class BeginLightingOperation:
    """Enable route-distance lighting and place the first light now."""

    support: str

    def __post_init__(self):
        if not self.support.strip():
            raise ValueError("lighting support key cannot be empty")

    @classmethod
    def parse(cls, value):
        value = _closed_mapping(value, "begin_lighting operation", {
            "type", "support"})
        return cls(str(value["support"]))

    def to_dict(self):
        return {"type": "begin_lighting", "support": self.support}


@dataclass(frozen=True, slots=True)
class RepeatDropOperation:
    """Replace and re-mine one drop until its stochastic output appears."""

    resource: str
    source_item: str
    target_item: str
    block: str
    mining_ticks: int = SOFT_BLOCK_TICKS
    max_attempts: int = 32

    def __post_init__(self):
        if not all(value.strip() for value in (
                self.resource, self.source_item, self.target_item,
                self.block)):
            raise ValueError("repeat-drop fields cannot be empty")
        if self.mining_ticks < 1 or self.max_attempts < 1:
            raise ValueError("repeat-drop bounds must be positive")

    @classmethod
    def parse(cls, value):
        value = _closed_mapping(value, "repeat_drop operation", {
            "type", "resource", "source", "target", "block",
            "mining_ticks", "max_attempts"})
        return cls(
            resource=str(value["resource"]),
            source_item=str(value["source"]),
            target_item=str(value["target"]),
            block=str(value.get("block", value["source"])),
            mining_ticks=int(value.get("mining_ticks", SOFT_BLOCK_TICKS)),
            max_attempts=int(value.get("max_attempts", 32)),
        )

    def to_dict(self):
        return {
            "type": "repeat_drop", "resource": self.resource,
            "source": self.source_item, "target": self.target_item,
            "block": self.block, "mining_ticks": self.mining_ticks,
            "max_attempts": self.max_attempts,
        }


@dataclass(frozen=True, slots=True)
class BucketCastMineOperation:
    """Create fluid-reaction blocks, then mine their verified dry result.

    Geometry and fluid-source routes belong to the snapshot planner.  This
    operation only declares the inventory milestone that the live executor
    must prove.  The corresponding source fluids, reusable bucket, reaction
    block, and bounded search settings live on ``PortalSettings`` so changing
    the construction method never introduces coordinates into a progression.
    """

    item: str
    minimum: int
    tool: str
    mining_ticks: int = OBSIDIAN_DIAMOND_TICKS

    def __post_init__(self):
        if not self.item.strip() or not self.tool.strip():
            raise ValueError("bucket-cast item and tool cannot be empty")
        if self.minimum < 1 or self.mining_ticks < 1:
            raise ValueError("bucket-cast counts and ticks must be positive")

    @classmethod
    def parse(cls, value):
        value = _closed_mapping(value, "bucket_cast_mine operation", {
            "type", "item", "minimum", "tool", "mining_ticks"})
        return cls(
            item=str(value.get("item", "obsidian")),
            minimum=int(value.get("minimum", 10)),
            tool=str(value.get("tool", "diamond_pickaxe")),
            mining_ticks=int(value.get(
                "mining_ticks", OBSIDIAN_DIAMOND_TICKS)),
        )

    def to_dict(self):
        return {
            "type": "bucket_cast_mine", "item": self.item,
            "minimum": self.minimum, "tool": self.tool,
            "mining_ticks": self.mining_ticks,
        }


@dataclass(frozen=True, slots=True)
class NetherPortalOperation:
    """Build, ignite, and enter the portal planned for this progression.

    Geometry remains in the snapshot plan and live execution remains in the
    controller, but the endgame is an explicit terminal graph operation. A
    different final objective can therefore add its own operation instead of
    growing another implicit tail in ``SurvivalProgressionPolicy.actions``.
    """

    @classmethod
    def parse(cls, value):
        _closed_mapping(value, "nether_portal operation", {"type"})
        return cls()

    def to_dict(self):
        return {"type": "nether_portal"}


@dataclass(frozen=True, slots=True)
class FluidSourceRequirement:
    """One source-only fluid search contract derived from the progression."""

    role: str
    fluid: str
    count: int
    metadata: int
    unique: bool
    filled_bucket_item: str
    recover: bool = False

    def __post_init__(self):
        if self.role not in ("coolant", "casting"):
            raise ValueError("fluid source role must be coolant or casting")
        if not self.fluid.strip() or not self.filled_bucket_item.strip():
            raise ValueError("fluid source names cannot be empty")
        if self.count < 1 or self.metadata < 0:
            raise ValueError("fluid source count and metadata are invalid")
        if not isinstance(self.unique, bool) or not isinstance(
                self.recover, bool):
            raise TypeError("fluid source switches must be true or false")

    def to_dict(self):
        return {
            "role": self.role, "fluid": self.fluid,
            "count": self.count, "metadata": self.metadata,
            "source_only": True, "unique": self.unique,
            "filled_bucket_item": self.filled_bucket_item,
            "recover": self.recover,
        }


Operation: TypeAlias = (
    CraftOperation | WorkstationOperation | SmeltOperation
    | RequireItemOperation | BeginLightingOperation | RepeatDropOperation
    | BucketCastMineOperation | NetherPortalOperation
)


@dataclass(frozen=True, slots=True)
class HotbarPolicy:
    """Ordered item stacks that may be swapped into the backpack.

    Recovering a journal-observed item from the backpack needs a visible
    destination slot.  The swap is lossless, but choosing that slot is policy,
    not a property of crafting, smelting, or any other generic operation.
    """

    swap_priority: tuple[str | int, ...] = ()

    def __post_init__(self):
        for item in self.swap_priority:
            if isinstance(item, bool) or not isinstance(item, (str, int)):
                raise TypeError(
                    "hotbar swap_priority entries must be item names or ids")
            if isinstance(item, str) and not item.strip():
                raise ValueError("hotbar swap_priority entries cannot be empty")

    @classmethod
    def parse(cls, value=None):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        value = _closed_mapping(value, "hotbar policy", {"swap_priority"})
        items = value.get("swap_priority", ())
        if isinstance(items, (str, int)) and not isinstance(items, bool):
            items = (items,)
        return cls(tuple(items))

    def to_dict(self):
        return {"swap_priority": list(self.swap_priority)}


_OPERATION_TYPES = {
    "craft": CraftOperation,
    "workstation": WorkstationOperation,
    "smelt": SmeltOperation,
    "require": RequireItemOperation,
    "begin_lighting": BeginLightingOperation,
    "repeat_drop": RepeatDropOperation,
    "bucket_cast_mine": BucketCastMineOperation,
    "nether_portal": NetherPortalOperation,
}


def _operation(value) -> Operation:
    if isinstance(value, tuple(_OPERATION_TYPES.values())):
        return value
    value = _mapping(value, "progression operation")
    kind = str(value.get("type", ""))
    try:
        return _OPERATION_TYPES[kind].parse(value)
    except KeyError as exc:
        raise ValueError(
            f"unknown progression operation {kind!r}; expected one of "
            f"{sorted(_OPERATION_TYPES)}") from exc


@dataclass(frozen=True, slots=True)
class ResourceStage:
    """One natural resource leg followed by declarative milestone work."""

    name: str
    blocks: tuple[str | int, ...]
    count: int
    drop_item: str | tuple[str, ...]
    tool: str | None = None
    mining_ticks: int = 80
    tunnel_tool: str | None = None
    tunnel_ticks: int | None = None
    route_kinds: tuple[str, ...] = ("surface", "tunnel")
    prefer_exposed: bool = False
    hazard_clearance: int | None = None
    sealable_fluids: tuple[str | int, ...] = ()
    fluid_seal_item: str | None = None
    max_drop_fall: int | None = None
    required_drop: str | int | None = None
    after: tuple[Operation, ...] = ()
    natural_only: bool = False
    metadata: tuple[int, ...] | None = None

    def __post_init__(self):
        if not self.name.strip() or not self.blocks or self.count < 1:
            raise ValueError("resource stage needs name, blocks, and count")
        if self.mining_ticks < 1:
            raise ValueError("mining_ticks must be positive")
        if self.tunnel_ticks is not None and self.tunnel_ticks < 1:
            raise ValueError("tunnel_ticks must be positive when supplied")
        if not self.route_kinds or any(
                item not in ("surface", "tunnel")
                for item in self.route_kinds):
            raise ValueError("route_kinds must use surface and/or tunnel")
        if isinstance(self.drop_item, tuple) and not self.drop_item:
            raise ValueError("drop_item alternatives cannot be empty")
        if not isinstance(self.prefer_exposed, bool):
            raise TypeError("prefer_exposed must be boolean")
        if not isinstance(self.natural_only, bool):
            raise TypeError("natural_only must be boolean")
        if self.hazard_clearance is not None and self.hazard_clearance < 0:
            raise ValueError("stage hazard_clearance cannot be negative")
        if self.sealable_fluids and not self.fluid_seal_item:
            raise ValueError("sealable_fluids needs fluid_seal_item")
        if self.fluid_seal_item and not self.sealable_fluids:
            raise ValueError("fluid_seal_item needs sealable_fluids")
        if self.max_drop_fall is not None and self.max_drop_fall < 0:
            raise ValueError("max_drop_fall cannot be negative")
        if self.metadata is not None:
            metadata = tuple(self.metadata)
            if (not metadata or len(metadata) != len(set(metadata))
                    or any(not isinstance(value, int)
                           or isinstance(value, bool)
                           or not 0 <= value <= 15
                           for value in metadata)):
                raise ValueError(
                    "resource stage metadata must be unique integers "
                    "in 0..15")
            object.__setattr__(self, "metadata", metadata)

    @classmethod
    def parse(cls, value):
        value = _closed_mapping(value, "resource stage", {
            "name", "blocks", "block", "count", "drop", "tool",
            "mining_ticks", "tunnel_tool", "tunnel_ticks", "route_kinds",
            "prefer_exposed", "natural_only", "hazard_clearance",
            "sealable_fluids",
            "fluid_seal_item", "max_drop_fall", "required_drop",
            "metadata", "after"})
        # ``required_drop`` is a planner constraint, separate from the item
        # count that the live executor verifies after mining.
        blocks = value.get("blocks", value.get("block", value["name"]))
        if isinstance(blocks, (str, int)):
            blocks = (blocks,)
        drop = value.get("drop", value["name"])
        if isinstance(drop, list):
            drop = tuple(str(item) for item in drop)
        metadata = value.get("metadata")
        if isinstance(metadata, int) and not isinstance(metadata, bool):
            metadata = (metadata,)
        elif metadata is not None:
            metadata = tuple(metadata)
        return cls(
            name=str(value["name"]), blocks=tuple(blocks),
            count=int(value.get("count", 1)), drop_item=drop,
            tool=value.get("tool"),
            mining_ticks=int(value.get("mining_ticks", 80)),
            tunnel_tool=value.get("tunnel_tool", value.get("tool")),
            tunnel_ticks=(None if value.get("tunnel_ticks") is None
                          else int(value["tunnel_ticks"])),
            route_kinds=tuple(value.get(
                "route_kinds", ("surface", "tunnel"))),
            prefer_exposed=_boolean(
                value.get("prefer_exposed", False),
                "resource stage prefer_exposed"),
            natural_only=_boolean(
                value.get("natural_only", False),
                "resource stage natural_only"),
            hazard_clearance=(None if value.get("hazard_clearance") is None
                              else int(value["hazard_clearance"])),
            sealable_fluids=tuple(value.get("sealable_fluids", ())),
            fluid_seal_item=value.get("fluid_seal_item"),
            max_drop_fall=(None if value.get("max_drop_fall") is None else
                           int(value["max_drop_fall"])),
            required_drop=value.get("required_drop"),
            after=tuple(_operation(item)
                        for item in value.get("after", ())),
            metadata=metadata,
        )

    @property
    def requirement(self):
        return ResourceRequirement(
            self.name, self.blocks, self.count, self.route_kinds,
            self.prefer_exposed, self.sealable_fluids,
            self.max_drop_fall, self.required_drop, self.metadata)

    def to_dict(self):
        return {
            "name": self.name, "blocks": list(self.blocks),
            "count": self.count,
            "drop": (list(self.drop_item)
                     if isinstance(self.drop_item, tuple)
                     else self.drop_item),
            "tool": self.tool, "mining_ticks": self.mining_ticks,
            "tunnel_tool": self.tunnel_tool,
            "tunnel_ticks": self.tunnel_ticks,
            "route_kinds": list(self.route_kinds),
            "prefer_exposed": self.prefer_exposed,
            "natural_only": self.natural_only,
            "hazard_clearance": self.hazard_clearance,
            "sealable_fluids": list(self.sealable_fluids),
            "fluid_seal_item": self.fluid_seal_item,
            "max_drop_fall": self.max_drop_fall,
            "required_drop": self.required_drop,
            "metadata": (None if self.metadata is None
                         else list(self.metadata)),
            "after": [item.to_dict() for item in self.after],
        }


@dataclass(frozen=True, slots=True)
class PortalSettings:
    """Final structure and entry settings for a portal progression."""

    enabled: bool = False
    method: str = "placed_blocks"
    strategy: str = "initial"
    scaffold_item: str = "cobblestone"
    search_radius: int = 12
    vertical_radius: int = 3
    hazard_clearance: int = 1
    max_entry_attempts: int = 12
    return_tool: str = "diamond_pickaxe"
    return_ticks: int = 80
    active_block: str = "portal"
    igniter_item: str = "flint_and_steel"
    entry_wait_ticks: int = 82
    bucket_item: str = "bucket"
    casting_bucket_item: str = "lava_bucket"
    coolant_bucket_item: str = "water_bucket"
    casting_fluid: str = "lava"
    coolant_fluid: str = "water"
    cast_block: str = "obsidian"
    casting_backing_item: str = "cobblestone"
    source_meta: int = 0
    fluid_search_radius: int = 128
    casting_wait_ticks: int = 1
    unique_casting_sources: bool = True
    recover_coolant: bool = True

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise TypeError("portal enabled must be true or false")
        for value, label in (
                (self.source_meta, "source_meta"),
                (self.fluid_search_radius, "fluid_search_radius"),
                (self.casting_wait_ticks, "casting_wait_ticks")):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"portal {label} must be an integer")
        if self.method not in ("placed_blocks", "bucket_cast_mined"):
            raise ValueError(
                "portal method must be 'placed_blocks' or "
                "'bucket_cast_mined'")
        if self.strategy not in (
                "initial", "surface", "completion_chamber"):
            raise ValueError(
                "portal strategy must be 'initial', 'surface', or "
                "'completion_chamber'")
        if not all(value.strip() for value in (
                self.scaffold_item, self.return_tool,
                self.active_block, self.igniter_item, self.bucket_item,
                self.casting_bucket_item, self.coolant_bucket_item,
                self.casting_fluid, self.coolant_fluid, self.cast_block,
                self.casting_backing_item)):
            raise ValueError(
                "portal item, block, and tool fields cannot be empty")
        if min(self.search_radius, self.vertical_radius,
               self.hazard_clearance, self.source_meta,
               self.fluid_search_radius) < 0:
            raise ValueError("portal search radii cannot be negative")
        if self.max_entry_attempts < 1:
            raise ValueError("portal max_entry_attempts must be positive")
        if self.entry_wait_ticks < 1:
            raise ValueError("portal entry_wait_ticks must be positive")
        if self.return_ticks < 1:
            raise ValueError("portal return_ticks must be positive")
        if self.casting_wait_ticks < 1:
            raise ValueError("portal casting_wait_ticks must be positive")
        if self.casting_fluid == self.coolant_fluid:
            raise ValueError(
                "portal casting_fluid and coolant_fluid must differ")
        if not isinstance(self.unique_casting_sources, bool):
            raise TypeError(
                "portal unique_casting_sources must be true or false")
        if not isinstance(self.recover_coolant, bool):
            raise TypeError("portal recover_coolant must be true or false")

    @classmethod
    def parse(cls, value=None):
        if value is None:
            return cls()
        if isinstance(value, bool):
            return cls(enabled=value)
        if isinstance(value, cls):
            return value
        value = _closed_mapping(value, "portal settings", {
            "enabled", "method", "strategy", "scaffold_item",
            "search_radius",
            "vertical_radius",
            "hazard_clearance", "max_entry_attempts", "entry_wait_ticks",
            "return_tool", "return_ticks", "active_block", "igniter_item",
            "bucket_item", "casting_bucket_item", "coolant_bucket_item",
            "casting_fluid", "coolant_fluid", "cast_block",
            "casting_backing_item", "source_meta", "fluid_search_radius",
            "casting_wait_ticks", "unique_casting_sources",
            "recover_coolant"})
        if "enabled" in value:
            value["enabled"] = _boolean(
                value["enabled"], "portal enabled")
        for key in ("unique_casting_sources", "recover_coolant"):
            if key in value:
                value[key] = _boolean(value[key], f"portal {key}")
        return cls(**value)

    def to_dict(self):
        return {
            "enabled": self.enabled,
            "method": self.method,
            "strategy": self.strategy,
            "scaffold_item": self.scaffold_item,
            "search_radius": self.search_radius,
            "vertical_radius": self.vertical_radius,
            "hazard_clearance": self.hazard_clearance,
            "max_entry_attempts": self.max_entry_attempts,
            "entry_wait_ticks": self.entry_wait_ticks,
            "return_tool": self.return_tool,
            "return_ticks": self.return_ticks,
            "active_block": self.active_block,
            "igniter_item": self.igniter_item,
            "bucket_item": self.bucket_item,
            "casting_bucket_item": self.casting_bucket_item,
            "coolant_bucket_item": self.coolant_bucket_item,
            "casting_fluid": self.casting_fluid,
            "coolant_fluid": self.coolant_fluid,
            "cast_block": self.cast_block,
            "casting_backing_item": self.casting_backing_item,
            "source_meta": self.source_meta,
            "fluid_search_radius": self.fluid_search_radius,
            "casting_wait_ticks": self.casting_wait_ticks,
            "unique_casting_sources": self.unique_casting_sources,
            "recover_coolant": self.recover_coolant,
        }


@dataclass(frozen=True, slots=True)
class SurvivalProgressionSpec:
    """A serializable progression graph with no seed-specific geometry."""

    stages: tuple[ResourceStage, ...]
    portal: PortalSettings | bool = PortalSettings()
    hazard_clearance: int = 1
    # This budget applies to each bounded A* alternative.  A progression can
    # test many resource cells, so the much larger generic planner ceiling is
    # inappropriate here: one hostile world must be rejected in seconds,
    # not monopolize a synthetic-data worker for minutes.
    node_cap: int = 256
    route_search_cap: int = 64
    cluster_sample_limit: int = 256
    health_loss_tolerance: float = 0.0
    veto_fluids: bool = True
    hotbar: HotbarPolicy | Mapping = HotbarPolicy()

    def __post_init__(self):
        if isinstance(self.portal, bool):
            object.__setattr__(self, "portal", PortalSettings(self.portal))
        elif not isinstance(self.portal, PortalSettings):
            raise TypeError("portal must be settings or a boolean")
        object.__setattr__(self, "hotbar", HotbarPolicy.parse(self.hotbar))
        if not self.stages:
            raise ValueError("progression needs at least one resource stage")
        names = tuple(stage.name for stage in self.stages)
        if len(names) != len(set(names)):
            raise ValueError("progression stage names must be unique")
        if (self.hazard_clearance < 0 or self.node_cap < 1
                or self.route_search_cap < 1):
            raise ValueError("progression planner bounds are invalid")
        if self.cluster_sample_limit < 1:
            raise ValueError("cluster_sample_limit must be positive")
        if self.health_loss_tolerance < 0:
            raise ValueError("health_loss_tolerance cannot be negative")
        portal_operations = [
            (stage_index, operation_index)
            for stage_index, stage in enumerate(self.stages)
            for operation_index, operation in enumerate(stage.after)
            if isinstance(operation, NetherPortalOperation)
        ]
        # Backward compatibility for user-authored specs from before the
        # endgame became a first-class operation. Serialization always emits
        # the upgraded graph, so this compatibility path does not perpetuate
        # the implicit tail.
        if self.portal.enabled and not portal_operations:
            stages = list(self.stages)
            stages[-1] = replace(
                stages[-1],
                after=stages[-1].after + (NetherPortalOperation(),))
            object.__setattr__(self, "stages", tuple(stages))
            portal_operations = [(
                len(self.stages) - 1,
                len(self.stages[-1].after) - 1,
            )]
        if not self.portal.enabled and portal_operations:
            raise ValueError(
                "nether_portal operation needs enabled portal settings")
        if len(portal_operations) > 1:
            raise ValueError("progression may contain only one portal endgame")
        if portal_operations and portal_operations[0] != (
                len(self.stages) - 1, len(self.stages[-1].after) - 1):
            raise ValueError(
                "nether_portal must be the final operation of the final stage")

        workstations: set[str] = set()
        resources: set[str] = set()
        bucket_casts: list[tuple[int, int, BucketCastMineOperation]] = []
        for stage_index, stage in enumerate(self.stages):
            resources.add(stage.name)
            for operation_index, operation in enumerate(stage.after):
                if isinstance(operation, WorkstationOperation):
                    if operation.key in workstations:
                        raise ValueError(
                            f"workstation key {operation.key!r} is duplicated")
                    workstations.add(operation.key)
                    continue
                reference = None
                if isinstance(operation, CraftOperation):
                    reference = operation.table
                elif isinstance(operation, SmeltOperation):
                    reference = operation.furnace
                elif isinstance(operation, BeginLightingOperation):
                    reference = operation.support
                elif (isinstance(operation, RepeatDropOperation)
                      and operation.resource not in resources):
                    raise ValueError(
                        f"repeat-drop references resource "
                        f"{operation.resource!r} before it is planned")
                elif isinstance(operation, BucketCastMineOperation):
                    bucket_casts.append((
                        stage_index, operation_index, operation))
                    if operation.item != self.portal.cast_block:
                        raise ValueError(
                            "bucket-cast operation item must match portal "
                            "cast_block")
                if reference is not None and reference not in workstations:
                    raise ValueError(
                        f"operation references workstation {reference!r} "
                        "before it is declared")
        if self.portal.enabled and self.portal.method == "bucket_cast_mined":
            if len(bucket_casts) != 1:
                raise ValueError(
                    "bucket_cast_mined portal needs exactly one "
                    "bucket_cast_mine operation")
            cast_stage, cast_operation, _cast = bucket_casts[0]
            expected_operation = (
                len(self.stages[-1].after) - 2
                if portal_operations else len(self.stages[-1].after) - 1)
            if (cast_stage != len(self.stages) - 1
                    or cast_operation != expected_operation):
                raise ValueError(
                    "bucket_cast_mine must immediately precede the terminal "
                    "portal operation so preflight and live execution share "
                    "one world-state order")
        elif bucket_casts:
            raise ValueError(
                "bucket_cast_mine operation needs an enabled "
                "bucket_cast_mined portal")

    @classmethod
    def parse(cls, value=None):
        if value is None:
            return cls.nether_portal()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            if value == "nether_portal":
                return cls.nether_portal()
            raise ValueError(
                f"unknown survival progression preset {value!r}; "
                "expected 'nether_portal'")
        value = _closed_mapping(value, "survival progression", {
            "stages", "portal", "hotbar", "hazard_clearance", "node_cap",
            "route_search_cap", "cluster_sample_limit",
            "health_loss_tolerance", "veto_fluids"})
        return cls(
            stages=tuple(ResourceStage.parse(item)
                         for item in value["stages"]),
            portal=PortalSettings.parse(value.get("portal", False)),
            hotbar=HotbarPolicy.parse(value.get("hotbar")),
            hazard_clearance=int(value.get("hazard_clearance", 1)),
            node_cap=int(value.get("node_cap", 256)),
            route_search_cap=int(value.get("route_search_cap", 64)),
            cluster_sample_limit=int(value.get(
                "cluster_sample_limit", 256)),
            health_loss_tolerance=float(value.get(
                "health_loss_tolerance", 0.0)),
            veto_fluids=_boolean(
                value.get("veto_fluids", True),
                "survival progression veto_fluids"),
        )

    @classmethod
    def nether_portal(cls):
        """Vanilla 1.11.2: empty inventory to a bucket-cast portal.

        Twelve logs cover the recipe chain, spare tools, five waypoint
        crafting tables. Twenty-seven stone cover three pickaxes, one furnace,
        and the reusable contained casting mold. Eighteen coal cover sixty-four
        route lights plus two furnace fuel units. Ten iron become two iron pickaxes,
        an igniter, and one reusable bucket. The second pickaxe raises the
        certified route budget from 250 to 500 breaks. The planner certifies
        one source-only water cell and ten distinct source-only lava cells;
        each reaction result is mined under the reusable water flow before
        the ordinary block-placement portal builder runs.
        """

        def table(key):
            return WorkstationOperation(
                key, "crafting_table", "crafting_table", "crafting_table")
        return cls(stages=(
            ResourceStage(
                # One uninterrupted hand-mining hold, short enough not to
                # continue through a vertical trunk into the next log.
                "logs", ("log",), 12, "log", mining_ticks=LOG_TOOL_TICKS,
                route_kinds=("surface",), after=(
                    CraftOperation("planks", 48, 2),
                    CraftOperation("stick", 32, 2),
                    CraftOperation("crafting_table", 5, 2),
                    table("table_logs"),
                    CraftOperation("wooden_pickaxe", 2, 3, "table_logs"),
                )),
            ResourceStage(
                "stone", ("stone",), 27, "cobblestone",
                tool="wooden_pickaxe", mining_ticks=PICK_TOOL_TICKS,
                route_kinds=("surface", "tunnel"), prefer_exposed=False,
                max_drop_fall=1, required_drop="cobblestone",
                metadata=(0,),
                after=(
                    table("table_stone"),
                    CraftOperation(
                        "stone_pickaxe", 3, 3, "table_stone"),
                    CraftOperation("furnace", 1, 3, "table_stone"),
                )),
            ResourceStage(
                "coal", ("coal_ore",), 18, "coal",
                tool="stone_pickaxe", mining_ticks=PICK_TOOL_TICKS,
                route_kinds=("tunnel",), after=(
                    table("table_coal"),
                    CraftOperation("torch", 64, 3, "table_coal"),
                    BeginLightingOperation("table_coal"),
                )),
            ResourceStage(
                "gravel", ("gravel",), 1, "flint",
                mining_ticks=SOFT_BLOCK_TICKS, tunnel_tool="stone_pickaxe",
                tunnel_ticks=UNKNOWN_TUNNEL_TICKS, route_kinds=("tunnel",),
                required_drop="flint"),
            ResourceStage(
                "iron", ("iron_ore",), 10, "iron_ore",
                tool="stone_pickaxe", mining_ticks=PICK_TOOL_TICKS,
                route_kinds=("tunnel",), after=(
                    WorkstationOperation(
                        "furnace_iron", "furnace", "furnace", "furnace"),
                    SmeltOperation(
                        "furnace_iron", "iron_ore", "iron_ingot", 10,
                        fuel_count=2),
                    table("table_iron"),
                    CraftOperation(
                        "iron_pickaxe", 2, 3, "table_iron"),
                    CraftOperation("flint_and_steel", 1, 2),
                    CraftOperation("bucket", 1, 3, "table_iron"),
                )),
            ResourceStage(
                "diamonds", ("diamond_ore",), 3, "diamond",
                tool="iron_pickaxe", mining_ticks=PICK_TOOL_TICKS,
                route_kinds=("tunnel",), after=(
                    table("table_diamonds"),
                    CraftOperation(
                        "diamond_pickaxe", 1, 3, "table_diamonds"),
                    BucketCastMineOperation(
                        "obsidian", 10, "diamond_pickaxe",
                        OBSIDIAN_DIAMOND_TICKS),
                    NetherPortalOperation(),
                )),
        ), portal=PortalSettings(
            enabled=True, method="bucket_cast_mined",
            strategy="completion_chamber", fluid_search_radius=160,
            return_ticks=OBSIDIAN_DIAMOND_TICKS), route_search_cap=512,
            veto_fluids=False,
            hotbar=HotbarPolicy(swap_priority=(
                "dirt", "planks", "stick", "crafting_table",
                "wooden_pickaxe", "stone_pickaxe", "cobblestone",
                "gravel", "torch")))

    @property
    def goal(self):
        routes = tuple(dict.fromkeys(
            route for stage in self.stages for route in stage.route_kinds))
        return GoalSpec(
            resources=tuple(stage.requirement for stage in self.stages),
            natural_obsidian=any(stage.natural_only for stage in self.stages),
            hazard_clearance=self.hazard_clearance,
            node_cap=self.node_cap,
            route_search_cap=self.route_search_cap,
            routes=routes,
        )

    @property
    def fluid_source_requirements(self):
        """Ordered, source-only fluid inputs for snapshot preflight.

        Coolant is intentionally first: the live route establishes a
        contained water mold before any lava can enter the work site.
        """

        cast = next((
            operation
            for stage in self.stages for operation in stage.after
            if isinstance(operation, BucketCastMineOperation)
        ), None)
        if cast is None:
            return ()
        portal = self.portal
        return (
            FluidSourceRequirement(
                "coolant", portal.coolant_fluid, 1, portal.source_meta,
                True, portal.coolant_bucket_item, portal.recover_coolant),
            FluidSourceRequirement(
                "casting", portal.casting_fluid, cast.minimum,
                portal.source_meta, portal.unique_casting_sources,
                portal.casting_bucket_item),
        )

    def to_dict(self):
        return {
            "stages": [stage.to_dict() for stage in self.stages],
            "portal": self.portal.to_dict(),
            "hotbar": self.hotbar.to_dict(),
            "hazard_clearance": self.hazard_clearance,
            "node_cap": self.node_cap,
            "route_search_cap": self.route_search_cap,
            "cluster_sample_limit": self.cluster_sample_limit,
            "health_loss_tolerance": self.health_loss_tolerance,
            "veto_fluids": self.veto_fluids,
        }


@dataclass(frozen=True, slots=True)
class WorkstationSite:
    key: str
    cell: Cell
    support: Cell
    stand: Cell
    light_cells: tuple[Cell, ...] = ()
    approach_stands: tuple[Cell, ...] = ()
    clear_cells: tuple[Cell, ...] = ()

    def to_dict(self):
        return {"key": self.key, "cell": list(self.cell),
                "support": list(self.support), "stand": list(self.stand),
                "light_cells": [list(cell) for cell in self.light_cells],
                "approach_stands": [list(cell)
                                     for cell in self.approach_stands],
                "clear_cells": [list(cell) for cell in self.clear_cells]}


class _ChangedWorld:
    """Read-only snapshot overlay after planned mining and placements."""

    def __init__(
        self,
        world,
        *,
        opened=(),
        removed=(),
        placed=None,
    ):
        self.world = world
        self.air = frozenset(_cell(item) for item in (*opened, *removed))
        self.placed = {_cell(key): int(value)
                       for key, value in dict(placed or {}).items()}
        self.seed = world.seed
        self.snapshot_sha256 = world.snapshot_sha256
        self.player = world.player

    @staticmethod
    def _coordinates(coordinates):
        return (_cell(coordinates[0]) if len(coordinates) == 1
                else _cell(coordinates))

    def in_bounds(self, *coordinates):
        cell = self._coordinates(coordinates)
        return self.world.in_bounds(*cell)

    def block(self, *coordinates):
        cell = self._coordinates(coordinates)
        if cell in self.placed:
            return self.placed[cell]
        if cell in self.air:
            return 0
        return self.world.block(*cell)


def _targets(plan: WorldPlan):
    return tuple(
        target for resource in plan.resources for cluster in resource.clusters
        for target in cluster.targets)


def _resource_targets(plan: WorldPlan, name: str):
    resource = next((item for item in plan.resources
                     if item.name == name), None)
    if resource is None:
        raise ValueError(f"world plan has no resource {name!r}")
    return tuple(target for cluster in resource.clusters
                 for target in cluster.targets)


def _portal_protected_cells(plan: StructurePlan) -> frozenset[Cell]:
    """Cells that must remain valid for the complete portal certificate.

    The structure cells alone are insufficient: an earlier resource plan can
    mine the floor beneath an approach or use a future player body cell as a
    workstation.  Surface stands have already-existing support, so preserve
    their feet, headroom, and support.  The elevated entry is different: its
    support is a future frame block, already protected by ``occupied_cells``;
    do not invent an additional pre-build surface requirement for it.
    """

    protected = set(
        plan.occupied_cells + plan.open_cells + plan.temporary_cells
        + tuple(getattr(plan, "clear_cells", ())))
    surface_stands = tuple(dict.fromkeys((
        *plan.approach_stands,
        plan.work_stance,
        plan.interaction_stance,
        plan.exit_stance,
    )))
    for x, y, z in surface_stands:
        protected.update(((x, y - 1, z), (x, y, z), (x, y + 1, z)))
    x, y, z = plan.entry_stance
    protected.update(((x, y, z), (x, y + 1, z)))
    return frozenset(protected)


def _workstation_protected_cells(
        sites: Mapping[str, WorkstationSite]) -> frozenset[Cell]:
    """Preserve placed workstations and the terrain supporting them.

    Interaction stands intentionally remain traversable: a workstation may
    be placed at the endpoint from which cast planning begins, and prior
    tunnel excavation can legitimately include that feet/head volume.
    ``light_cells`` are optional empty destinations around the workstation,
    not structure. A later route may legitimately excavate one; the live
    lighting controller rechecks them and falls back to another wall.
    """

    return frozenset(
        cell
        for site in sites.values()
        for cell in (site.cell, site.support)
    )


def _route_preservation_cells(route) -> tuple[frozenset[Cell],
                                                frozenset[Cell]]:
    """Return immutable player volume and support for a certified route.

    ``RoutePlan.reserved_cells`` includes tunnel transition headroom that
    cannot be reconstructed from stands alone.  The explicit stand-derived
    cells keep this helper compatible with small custom/test route objects and
    make the public invariant clear: every breadcrumb preserves feet, head,
    and floor, even for an implementation that omits the richer fields.
    """

    stands = tuple(_cell(cell) for cell in getattr(route, "stands", ()))
    passage = {
        _cell(cell) for cell in getattr(route, "reserved_cells", ())}
    passage.update(
        cell
        for x, y, z in stands
        for cell in ((x, y, z), (x, y + 1, z)))
    supports = {
        _cell(cell) for cell in getattr(route, "support_cells", ())}
    supports.update((x, y - 1, z) for x, y, z in stands)
    return frozenset(passage), frozenset(supports)


class _ReturnBreadcrumb:
    """Loop-spliced certified walking spine from spawn to the live stage.

    Resource routes may revisit an earlier corridor after exploring a branch.
    Only the simple path from spawn to the current endpoint is required for
    the final return; retaining abandoned loops overprotects unrelated world
    cells and rejects otherwise valid seeds.  Every surviving edge came from
    a certified route, including the extra headroom swept by vertical steps.
    """

    def __init__(self, start: Cell):
        start = _cell(start, "breadcrumb start")
        self._stands = [start]
        self._positions = {start: 0}
        self.broken = False

    @property
    def stands(self) -> tuple[Cell, ...]:
        return tuple(self._stands)

    def extend(self, stands: Iterable[Cell]) -> None:
        if self.broken:
            return
        incoming = tuple(
            _cell(cell, "breadcrumb stand") for cell in stands)
        if not incoming:
            return
        if incoming[0] != self._stands[-1]:
            self._break()
            return
        for cell in incoming[1:]:
            previous = self._stands[-1]
            if (abs(previous[0] - cell[0])
                    + abs(previous[2] - cell[2]) != 1
                    or abs(previous[1] - cell[1]) > 1):
                self._break()
                return
            prior = self._positions.get(cell)
            if prior is not None:
                for dropped in self._stands[prior + 1:]:
                    self._positions.pop(dropped, None)
                del self._stands[prior + 1:]
                continue
            self._positions[cell] = len(self._stands)
            self._stands.append(cell)
        if self.passage_cells.intersection(self.support_cells):
            self._break()

    def _break(self) -> None:
        self.broken = True
        self._stands.clear()
        self._positions.clear()

    @property
    def passage_cells(self) -> frozenset[Cell]:
        passage = {
            cell
            for x, y, z in self._stands
            for cell in ((x, y, z), (x, y + 1, z))
        }
        for left, right in zip(self._stands, self._stands[1:]):
            if right[1] < left[1]:
                passage.add((right[0], right[1] + 2, right[2]))
            elif right[1] > left[1]:
                passage.add((left[0], left[1] + 2, left[2]))
        return frozenset(passage)

    @property
    def support_cells(self) -> frozenset[Cell]:
        return frozenset(
            (x, y - 1, z) for x, y, z in self._stands)


def _workstation_sites(
    world, plan, spec, decision_seed, *, stages=None,
    opened=(), removed=(), placed=None, protected=(),
    preserve_return_route=False,
):
    """Choose exact nearby placements without owning any fixed coordinate."""

    targets = _targets(plan)
    opened = set(opened) | {
        cell for target in targets for cell in target.route.mine_cells}
    removed = set(removed) | {target.cell for target in targets}
    changed = _ChangedWorld(
        world, opened=opened, removed=removed, placed=placed)
    # Workstations are placed only after this stage is complete. Its old
    # route cells are therefore legitimate sites and become part of the
    # persistent overlay for the next JIT plan. Keep only structure cells,
    # the live stand, and actual removed targets unavailable.
    forbidden = set(protected)
    sites: dict[str, WorkstationSite] = {}
    stages = tuple(spec.stages if stages is None else stages)

    for stage in stages:
        clearance = (spec.hazard_clearance
                     if stage.hazard_clearance is None
                     else stage.hazard_clearance)
        stage_targets = _resource_targets(plan, stage.name)
        stage_passage = set()
        for target in stage_targets:
            route_passage, _route_supports = _route_preservation_cells(
                target.route)
            stage_passage.update(route_passage)
        if preserve_return_route:
            # Workstations may share the corridor's existing solid floor but
            # cannot occupy or excavate its player/head/transition volume.
            forbidden.update(stage_passage)
        history = []
        for target in stage_targets:
            for route_stand in target.route.stands:
                if not history or history[-1] != route_stand:
                    history.append(route_stand)
        history_cells = frozenset(history)
        final_stand = stage_targets[-1].mining_stance
        forbidden.add(final_stand)
        operations = tuple(item for item in stage.after
                           if isinstance(item, WorkstationOperation))
        for operation_index, operation in enumerate(operations):
            digest = hashlib.sha256(
                f"{decision_seed}:workstation:{operation.key}".encode()
            ).digest()
            rng = random.Random(int.from_bytes(digest[:8], "big"))
            offsets = [
                (dx, dz) for radius in (1, 2, 3)
                for dx in range(-radius, radius + 1)
                for dz in range(-radius, radius + 1)
                if max(abs(dx), abs(dz)) == radius
            ]
            rng.shuffle(offsets)
            selected = None
            for history_index in range(len(history) - 1, -1, -1):
                stand = history[history_index]
                approach = tuple(reversed(history[history_index:]))
                if not all(is_surface_stand(
                        changed, route_stand, hazard_clearance=clearance)
                        for route_stand in approach):
                    continue
                for dx, dz in offsets:
                    cell = stand[0] + dx, stand[1], stand[2] + dz
                    support = cell[0], cell[1] - 1, cell[2]
                    if (cell in history_cells
                            or cell in forbidden or cell == stand):
                        continue
                    if (not changed.in_bounds(cell)
                            or not changed.in_bounds(support)):
                        continue
                    cell_block = changed.block(cell)
                    if int(getattr(
                            cell_block, "block_id", cell_block)) != 0:
                        continue
                    support_block = changed.block(support)
                    if (not DEFAULT_BLOCK_RULES.is_solid(support_block)
                            or DEFAULT_BLOCK_RULES.is_hazard(support_block)
                            or DEFAULT_BLOCK_RULES.is_fluid(support_block)):
                        continue
                    eye = (stand[0] + 0.5, stand[1] + 1.62,
                           stand[2] + 0.5)
                    face = cell[0] + 0.5, cell[1], cell[2] + 0.5
                    if math.dist(eye, face) > 4.0:
                        continue
                    if not clear_placement_ray(
                            changed, stand, cell, support,
                            max_reach=4.0, origin_margin=0.04):
                        continue
                    light_cells = tuple(
                        candidate
                        for ldx, ldz in (
                            (1, 0), (-1, 0), (0, 1), (0, -1))
                        if (candidate := (
                            cell[0] + ldx, cell[1], cell[2] + ldz))
                        not in forbidden
                        and candidate != stand
                        and changed.in_bounds(candidate)
                        and DEFAULT_BLOCK_RULES.is_passable(
                            changed.block(candidate))
                    )
                    selected = WorkstationSite(
                        operation.key, cell, support, stand, light_cells,
                        approach)
                    break
                if selected is not None:
                    break
            if selected is None and not preserve_return_route:
                # When several workstations share one narrow tunnel, allocate
                # old route cells from farthest to nearest in operation order.
                # A near-first placement would wall the player off from every
                # later site and force a brittle head-height fallback.
                remaining = len(operations) - operation_index - 1
                start_index = len(history) - 2 - remaining
                for index in range(start_index, -1, -1):
                    cell = history[index]
                    stand = history[index + 1]
                    support = cell[0], cell[1] - 1, cell[2]
                    approach = tuple(reversed(history[index + 1:]))
                    if (cell[1] != stand[1]
                            or cell in forbidden or cell == stand):
                        continue
                    if not all(is_surface_stand(
                            changed, route_stand,
                            hazard_clearance=clearance)
                            for route_stand in approach):
                        continue
                    cell_block = changed.block(cell)
                    if int(getattr(
                            cell_block, "block_id", cell_block)) != 0:
                        continue
                    support_block = changed.block(support)
                    if (not DEFAULT_BLOCK_RULES.is_solid(support_block)
                            or DEFAULT_BLOCK_RULES.is_hazard(support_block)
                            or DEFAULT_BLOCK_RULES.is_fluid(support_block)):
                        continue
                    if not clear_placement_ray(
                            changed, stand, cell, support,
                            max_reach=4.0, origin_margin=0.04):
                        continue
                    light_cells = tuple(
                        candidate for dx, dz in (
                            (1, 0), (-1, 0), (0, 1), (0, -1))
                        if (candidate := (
                            cell[0] + dx, cell[1], cell[2] + dz))
                        not in forbidden and candidate != stand
                        and changed.in_bounds(candidate)
                        and DEFAULT_BLOCK_RULES.is_passable(
                            changed.block(candidate)))
                    selected = WorkstationSite(
                        operation.key, cell, support, stand, light_cells,
                        approach)
                    break
            if selected is None:
                # A tight tunnel may have no untouched side-floor alcove,
                # while the stage itself has just opened several ore cells.
                # Replan over that changed world rather than assuming the
                # mining stance remains the only valid interaction stance.
                # Both the workstation cell and the stand stay feet-level;
                # elevated shelves are brittle under live player collision.
                target_cells = [
                    target.cell for target in stage_targets
                    if (not preserve_return_route
                        or target.cell not in stage_passage)]
                rng.shuffle(target_cells)
                for cell in target_cells:
                    support = cell[0], cell[1] - 1, cell[2]
                    if cell in forbidden:
                        continue
                    if (not changed.in_bounds(cell)
                            or not changed.in_bounds(support)
                            or cell not in changed.air):
                        continue
                    support_block = changed.block(support)
                    if (not DEFAULT_BLOCK_RULES.is_solid(support_block)
                            or DEFAULT_BLOCK_RULES.is_hazard(support_block)
                            or DEFAULT_BLOCK_RULES.is_fluid(support_block)):
                        continue
                    stand_candidates = [
                        (cell[0] + dx, cell[1], cell[2] + dz)
                        for dx, dz in (
                            (1, 0), (-1, 0), (0, 1), (0, -1))]
                    rng.shuffle(stand_candidates)
                    for stand in stand_candidates:
                        if (stand in forbidden or not is_surface_stand(
                                changed, stand,
                                hazard_clearance=clearance)):
                            continue
                        try:
                            approach = tuple(plan_surface_path(
                                changed, final_stand, stand,
                                hazard_clearance=clearance,
                                node_cap=spec.node_cap,
                                decision_rng=rng))
                        except PlanRejected:
                            continue
                        if not clear_placement_ray(
                                changed, stand, cell, support,
                                max_reach=4.0, origin_margin=0.04):
                            continue
                        light_cells = tuple(
                            candidate for dx, dz in (
                                (1, 0), (-1, 0), (0, 1), (0, -1))
                            if (candidate := (
                                cell[0] + dx, cell[1], cell[2] + dz))
                            not in forbidden and candidate != stand
                            and changed.in_bounds(candidate)
                            and DEFAULT_BLOCK_RULES.is_passable(
                                changed.block(candidate)))
                        selected = WorkstationSite(
                            operation.key, cell, support, stand,
                            light_cells, approach)
                        break
                    if selected is not None:
                        break
            if selected is None:
                # If no already-open floor site exists, certify one bounded
                # side alcove. Search backward along the certified route: a
                # natural mine may be tight at its deepest target yet have a
                # perfectly good wall one or two bends behind it. The route
                # prefix becomes an explicit workstation approach.
                alcoves = []
                for history_index in range(len(history) - 1, -1, -1):
                    stand = history[history_index]
                    approach = tuple(reversed(history[history_index:]))
                    candidates = [
                        (stand[0] + dx, stand[1], stand[2] + dz)
                        for dx, dz in (
                            (1, 0), (-1, 0), (0, 1), (0, -1))]
                    rng.shuffle(candidates)
                    alcoves.extend(
                        (stand, approach, cell) for cell in candidates)
                for stand, approach, cell in alcoves:
                    support = cell[0], cell[1] - 1, cell[2]
                    if (cell in forbidden
                            or (preserve_return_route
                                and cell in stage_passage)
                            or cell in changed.air
                            or not changed.in_bounds(cell)
                            or not changed.in_bounds(support)):
                        continue
                    block = changed.block(cell)
                    support_block = changed.block(support)
                    if (not DEFAULT_BLOCK_RULES.is_excavatable(block)
                            or not DEFAULT_BLOCK_RULES.is_solid(
                                support_block)
                            or DEFAULT_BLOCK_RULES.is_hazard(support_block)
                            or DEFAULT_BLOCK_RULES.is_fluid(support_block)):
                        continue
                    try:
                        plan_safe_removal(
                            changed, cell, route_stands=(stand,))
                    except PlanRejected:
                        continue
                    eye = (stand[0] + 0.5, stand[1] + 1.62,
                           stand[2] + 0.5)
                    point = (cell[0] + 0.5, cell[1] + 0.5,
                             cell[2] + 0.5)
                    if not clear_voxel_ray(
                            changed, eye, point, target=cell,
                            origin_margin=0.04):
                        continue
                    opened_world = _ChangedWorld(changed, opened=(cell,))
                    if not clear_placement_ray(
                            opened_world, stand, cell, support,
                            max_reach=4.0, origin_margin=0.04):
                        continue
                    light_cells = tuple(
                        candidate for dx, dz in (
                            (1, 0), (-1, 0), (0, 1), (0, -1))
                        if (candidate := (
                            cell[0] + dx, cell[1], cell[2] + dz))
                        not in forbidden and candidate != stand
                        and opened_world.in_bounds(candidate)
                        and DEFAULT_BLOCK_RULES.is_passable(
                            opened_world.block(candidate)))
                    selected = WorkstationSite(
                        operation.key, cell, support, stand, light_cells,
                        approach, (cell,))
                    break
            if selected is None:
                raise CaseRejected(
                    f"no exact workstation placement near {stage.name!r}",
                    code="progression_workstation_site_missing")
            sites[operation.key] = selected
            forbidden.add(selected.cell)
            changed.placed[selected.cell] = resolve_blocks(
                [operation.block])[0]
    return sites, changed


class _LightingHooks:
    """PlanExecutor hooks that preserve one route-distance clock."""

    def __init__(self, policy):
        self.policy = policy

    def observe_route_travel(self, env):
        self.policy.observe_route_travel(env)

    def remember_route_position(self, env):
        self.policy.remember_route_position(env)

    def light_route(self, env):
        return self.policy.light_route(env)

    def route_light_overdue(self):
        return self.policy.route_light_overdue()

    def route_light_debug(self, env):
        return self.policy.route_light_debug(env)




__all__ = [
    "BeginLightingOperation", "BucketCastMineOperation", "CraftOperation",
    "FluidSourceRequirement", "HotbarPolicy",
    "NetherPortalOperation", "PortalSettings",
    "RepeatDropOperation", "RequireItemOperation", "ResourceStage",
    "SmeltOperation",
    "SurvivalProgressionSpec",
    "WorkstationOperation", "WorkstationSite",
]
