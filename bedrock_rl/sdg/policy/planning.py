"""Declarative, action-free resource planning over a world snapshot."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, TypeAlias

from bedrock_rl.sdg.policy.spatial import (
    MINING_REACH,
    DEFAULT_BLOCK_RULES,
    ExcavationPlan,
    PlanRejected,
    RejectReason,
    RemovalProof,
    clear_voxel_ray,
    is_excavation_stand,
    is_surface_stand,
    plan_safe_removal,
    plan_staircase_tunnel,
    plan_surface_path,
    safe_mining_stances,
    voxel_ray,
)
from bedrock_rl.sdg.policy.world import (
    ResourceCluster,
    SnapshotWorldMap,
)
from bedrock_rl.sdg.generation import CaseRejected
from bedrock_rl.env.catalog import resolve_blocks, resolve_items


Cell: TypeAlias = tuple[int, int, int]
BlockRef: TypeAlias = int | str
ToolRef: TypeAlias = int | str


def natural_harvest_drop(
    block_id: int,
    cell: Cell,
    *,
    block_meta: int = 0,
) -> int | None:
    """Netherite's deterministic survival drop for a natural block cell.

    The engine intentionally makes coordinate-dependent drops reproducible.
    Preflight must use that same public rule when a task requires a specific
    drop (not merely a source block), otherwise a planner can select gravel
    that can never yield flint no matter how often it is replaced.
    """

    block_id = int(block_id)
    block_meta = int(block_meta)
    if block_id == 13:
        x, y, z = (int(value) & 0xFFFFFFFF for value in cell)
        h = ((x * 73428767) & 0xFFFFFFFF
             ^ (y * 912931) & 0xFFFFFFFF
             ^ (z * 19349663) & 0xFFFFFFFF)
        h ^= h >> 13
        h = (h * 0x85EBCA6B) & 0xFFFFFFFF
        h ^= h >> 16
        return 318 if h % 10 == 0 else 13
    if block_id == 1:
        # Vanilla BlockStone: plain stone drops cobblestone, while the
        # granite/diorite/andesite metadata variants retain stone item id 1.
        # A block-id-only preflight otherwise certifies a target that can
        # never satisfy a cobblestone recipe even with the right pickaxe.
        return 1 if 1 <= block_meta <= 6 else 4
    return {
        2: 3, 3: 3, 4: 4, 12: 12, 14: 14, 15: 15,
        16: 263, 17: 17, 49: 49, 54: 54, 56: 264, 162: 162,
    }.get(block_id)


def _closed_mapping(value, *, what: str, keys: Iterable[str]):
    if not isinstance(value, Mapping):
        raise TypeError(f"{what} must be a mapping")
    result = dict(value)
    unknown = sorted(set(result) - set(keys))
    if unknown:
        raise ValueError(
            f"unknown {what} key{'' if len(unknown) == 1 else 's'}: "
            + ", ".join(unknown))
    return result


def _boolean(value, *, what: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{what} must be true or false")
    return value


def _cells(values: Iterable[Cell]) -> tuple[Cell, ...]:
    return tuple(tuple(int(value) for value in cell) for cell in values)


def _json(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    return value


def _drop_landing_within_reach(
    world, target: Cell, stance: Cell, max_fall: int | None = None,
) -> bool:
    """Prove a mined item cannot fall below the collection stance.

    Drops inherit gravity. A perfectly reachable exposed ore can overhang a
    cave and send its item several blocks below the player, making the action
    trace unable to verify collection. Conservatively require the first solid
    support below the removed block to leave the item within one vertical
    block of the proved mining stance. Fluid/hazard landing columns are never
    accepted.
    """

    if math.hypot(
            stance[0] - target[0], stance[2] - target[2]) > 1.25:
        # Block interaction reach is substantially larger than item pickup
        # reach.  A surface stance two cells away can mine successfully while
        # leaving the drop in an unstandable wall pocket.  Require the same
        # cardinal local geometry used by the live collection controller.
        return False

    x, y, z = target
    while y > -64:
        y -= 1
        cell = x, y, z
        if not world.in_bounds(cell):
            return False
        block = world.block(cell)
        if (DEFAULT_BLOCK_RULES.is_fluid(block)
                or DEFAULT_BLOCK_RULES.is_hazard(block)):
            return False
        if not DEFAULT_BLOCK_RULES.is_passable(block):
            landing = y + 1
            if landing < stance[1]:
                # Entering a one-block-lower drop column transiently occupies
                # one voxel above the eventual two-block standing volume.
                # Without that upper transition clearance the player's body
                # collision-clamps at the ledge even though the landing itself
                # is a valid surface stand. Evaluate the post-removal overlay
                # handed in by the caller, so prior planned clears count.
                transition = tuple(
                    (target[0], landing + offset, target[2])
                    for offset in (0, 1, 2))
                if any(
                        not world.in_bounds(cell)
                        or not DEFAULT_BLOCK_RULES.is_passable(
                            world.block(cell))
                        for cell in transition):
                    return False
            if max_fall is not None:
                return target[1] - landing <= max_fall
            return abs(landing - stance[1]) <= 1
    return False


def _drop_fall_within(world, target: Cell, maximum: int) -> bool:
    """Cheaply reject deep item falls before spending any route search."""

    x, y, z = target
    # The target itself is removed. A solid directly below it means zero
    # blocks of item fall; ``maximum=1`` also permits one intervening air
    # cell before support.
    for offset in range(1, int(maximum) + 2):
        cell = x, y - offset, z
        if not world.in_bounds(cell):
            return False
        block = world.block(cell)
        if (DEFAULT_BLOCK_RULES.is_fluid(block)
                or DEFAULT_BLOCK_RULES.is_hazard(block)):
            return False
        if not DEFAULT_BLOCK_RULES.is_passable(block):
            return offset - 1 <= maximum
    return False


def _excavation_clear_rays(
    world, excavation: ExcavationPlan, *, protected: Iterable[Cell] = (),
) -> bool:
    """Keep protected future targets out of earlier clear click rays."""

    mine_cells = frozenset(excavation.mine_cells)
    protected = frozenset(protected)
    del world  # The staircase planner already proves ordinary excavation.
    for previous, destination in zip(
            excavation.stands, excavation.stands[1:]):
        body = [
            destination,
            (destination[0], destination[1] + 1, destination[2]),
        ]
        if destination[1] < previous[1]:
            body.append((destination[0], destination[1] + 2,
                         destination[2]))
        required = sorted(
            (cell for cell in body if cell in mine_cells),
            key=lambda cell: (-cell[1], cell[0], cell[2]))
        eye = previous[0] + 0.5, previous[1] + 1.62, previous[2] + 0.5
        for target in required:
            point = target[0] + 0.5, target[1] + 0.5, target[2] + 0.5
            if any(cell in protected
                   for cell in voxel_ray(eye, point)[:-1]):
                return False
    return True


def _excavation_falling_blocks_stable(
    world, excavation: ExcavationPlan, *, falling=(12, 13),
) -> bool:
    """Reject clears that let sand/gravel collapse into the route body."""

    removed = frozenset(excavation.mine_cells)
    # A gravity block declared as route clearance is not stable merely
    # because it is absent in the final overlay: removing a lower cell can
    # move an upper sand/gravel voxel before the executor reaches its own
    # declared coordinate. Keep falling materials as resource targets, never
    # intermediate passage excavation.
    for cell in removed:
        if not world.in_bounds(cell):
            return False
        block = world.block(cell)
        block_id = int(getattr(block, "block_id", block))
        if block_id in falling:
            return False
    columns = {(cell[0], cell[2]) for cell in excavation.reserved_cells}
    for x, z in columns:
        body_y = [cell[1] for cell in excavation.reserved_cells
                  if cell[0] == x and cell[2] == z]
        if not body_y:
            continue
        y = min(body_y)
        # Find the first block that remains after all declared route clears.
        # If it is a gravity block, live ticks move it into the proved body
        # volume and invalidate both movement and click-ray certificates.
        for _ in range(64):
            cell = (x, y, z)
            if not world.in_bounds(cell):
                break
            block = world.block(cell)
            if cell in removed or DEFAULT_BLOCK_RULES.is_passable(block):
                y += 1
                continue
            block_id = int(getattr(block, "block_id", block))
            if block_id in falling:
                return False
            break
    return True


def _target_falling_blocks_stable(
    world, target: Cell, *, falling=(12, 13),
) -> bool:
    """Whether removing ``target`` leaves no gravity block above it.

    Route excavation and the final resource break are separate mutations.
    The route stability proof above protects player body cells, while this
    proof protects the resource cell after the final break.  A sand/gravel
    block separated from the target by air (or by cells the route already
    opened) will fall into that cell on the next live tick and invalidate the
    removal and pickup certificate.
    """

    x, y, z = target
    y += 1
    while world.in_bounds((x, y, z)):
        block = world.block((x, y, z))
        if DEFAULT_BLOCK_RULES.is_passable(block):
            y += 1
            continue
        block_id = int(getattr(block, "block_id", block))
        return block_id not in falling
    return True


@dataclass(frozen=True, slots=True)
class ResourceRequirement:
    """A minimum count across one or more equivalent block ids/names."""

    name: str
    blocks: tuple[BlockRef, ...]
    minimum: int
    routes: tuple[str, ...] | None = None
    prefer_exposed: bool = False
    sealable_fluids: tuple[BlockRef, ...] = ()
    max_drop_fall: int | None = None
    required_drop: BlockRef | None = None
    metadata: tuple[int, ...] | None = None

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("resource requirement name cannot be empty")
        if not self.blocks:
            raise ValueError(
                f"resource {self.name!r} needs at least one block")
        if not isinstance(self.minimum, int) \
                or isinstance(self.minimum, bool) or self.minimum < 1:
            raise ValueError(
                f"resource {self.name!r} minimum must be positive")
        if self.routes is not None:
            if (not self.routes or any(
                    route not in ("surface", "tunnel")
                    for route in self.routes)):
                raise ValueError(
                    "resource routes must use surface and/or tunnel")
            if len(set(self.routes)) != len(self.routes):
                raise ValueError("resource routes cannot contain duplicates")
        if not isinstance(self.prefer_exposed, bool):
            raise TypeError("prefer_exposed must be boolean")
        resolve_blocks(self.blocks)
        resolve_blocks(self.sealable_fluids)
        if self.required_drop is not None:
            resolve_items([self.required_drop])
        if self.metadata is not None:
            metadata = tuple(self.metadata)
            if (not metadata or len(metadata) != len(set(metadata))
                    or any(not isinstance(value, int)
                           or isinstance(value, bool)
                           or not 0 <= value <= 15
                           for value in metadata)):
                raise ValueError(
                    "resource metadata must be unique integers in 0..15")
            object.__setattr__(self, "metadata", metadata)
        if self.max_drop_fall is not None and self.max_drop_fall < 0:
            raise ValueError("resource max_drop_fall cannot be negative")

    @classmethod
    def parse(cls, value, *, name: str | None = None):
        if isinstance(value, cls):
            return value
        if name is not None and isinstance(value, int) \
                and not isinstance(value, bool):
            return cls(str(name), (str(name),), value)
        value = _closed_mapping(
            value,
            what="resource requirement",
            keys={
                "name", "blocks", "block", "minimum", "count", "routes",
                "prefer_exposed", "sealable_fluids", "max_drop_fall",
                "required_drop", "metadata",
            },
        )
        blocks = value.get("blocks", value.get("block", name))
        if isinstance(blocks, (str, int)):
            blocks = (blocks,)
        if not isinstance(blocks, (list, tuple)) or not blocks:
            raise TypeError(
                "resource blocks must be a name/id or nonempty list")
        minimum = value.get("minimum", value.get("count"))
        if minimum is None:
            raise ValueError("resource requirement needs minimum or count")
        label = str(value.get("name", name or blocks[0]))
        routes = value.get("routes")
        prefer_exposed = _boolean(
            value.get("prefer_exposed", False),
            what="resource prefer_exposed",
        )
        metadata = value.get("metadata")
        if isinstance(metadata, int) and not isinstance(metadata, bool):
            metadata = (metadata,)
        elif metadata is not None:
            metadata = tuple(metadata)
        return cls(label, tuple(blocks), int(minimum),
                   None if routes is None else tuple(routes),
                   prefer_exposed, tuple(value.get("sealable_fluids", ())),
                   (None if value.get("max_drop_fall") is None else
                    int(value["max_drop_fall"])),
                   value.get("required_drop"), metadata)

    @property
    def block_ids(self) -> tuple[int, ...]:
        return resolve_blocks(self.blocks)

    @property
    def sealable_fluid_ids(self) -> tuple[int, ...]:
        return resolve_blocks(self.sealable_fluids)

    @property
    def required_drop_ids(self) -> tuple[int, ...]:
        return (() if self.required_drop is None
                else resolve_items([self.required_drop]))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "blocks": list(self.blocks),
            "block_ids": list(self.block_ids),
            "minimum": self.minimum,
            "routes": (None if self.routes is None
                       else list(self.routes)),
            "prefer_exposed": self.prefer_exposed,
            "sealable_fluids": list(self.sealable_fluids),
            "max_drop_fall": self.max_drop_fall,
            "required_drop": self.required_drop,
            "metadata": (None if self.metadata is None
                         else list(self.metadata)),
        }


@dataclass(frozen=True, slots=True)
class ProximityConstraint:
    """Require the nearest matching block to lie within a distance band."""

    name: str
    blocks: tuple[BlockRef, ...]
    minimum: float = 0.0
    maximum: float = math.inf

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("proximity constraint name cannot be empty")
        if not self.blocks:
            raise ValueError(f"proximity {self.name!r} has no blocks")
        if self.minimum < 0 or self.maximum < self.minimum:
            raise ValueError(
                f"proximity {self.name!r} needs 0 <= min <= max")
        resolve_blocks(self.blocks)

    @classmethod
    def parse(cls, value, *, name: str | None = None):
        if isinstance(value, cls):
            return value
        if isinstance(value, (list, tuple)) and name is not None:
            if len(value) != 2:
                raise ValueError("proximity distance range needs [min, max]")
            return cls(str(name), (str(name),),
                       float(value[0]), float(value[1]))
        value = _closed_mapping(
            value,
            what="proximity constraint",
            keys={
                "name", "blocks", "block", "distance", "minimum", "min",
                "maximum", "max",
            },
        )
        blocks = value.get("blocks", value.get("block", name))
        if isinstance(blocks, (str, int)):
            blocks = (blocks,)
        if not isinstance(blocks, (list, tuple)) or not blocks:
            raise TypeError("proximity blocks must be a name/id or list")
        distance = value.get("distance")
        if distance is not None:
            if not isinstance(distance, (list, tuple)) or len(distance) != 2:
                raise ValueError("proximity distance needs [min, max]")
            low, high = distance
        else:
            low = value.get("minimum", value.get("min", 0.0))
            high = value.get("maximum", value.get("max", math.inf))
        label = str(value.get("name", name or blocks[0]))
        return cls(label, tuple(blocks), float(low), float(high))

    @property
    def block_ids(self) -> tuple[int, ...]:
        return resolve_blocks(self.blocks)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "blocks": list(self.blocks),
            "block_ids": list(self.block_ids),
            "distance": [self.minimum,
                         None if math.isinf(self.maximum) else self.maximum],
        }


@dataclass(frozen=True, slots=True)
class GoalSpec:
    """Data-only world feasibility requirements for a task family."""

    resources: tuple[ResourceRequirement, ...]
    proximity: tuple[ProximityConstraint, ...] = ()
    natural_obsidian: bool = False
    hazard_clearance: int = 0
    node_cap: int = 20_000
    route_search_cap: int = 512
    routes: tuple[str, ...] = ("surface", "tunnel")

    def __post_init__(self):
        if not self.resources:
            raise ValueError("a world goal needs at least one resource")
        names = [item.name for item in self.resources]
        if len(set(names)) != len(names):
            raise ValueError("resource requirement names must be unique")
        if self.hazard_clearance < 0:
            raise ValueError("hazard_clearance cannot be negative")
        if self.node_cap < 1:
            raise ValueError("node_cap must be positive")
        if self.route_search_cap < 1:
            raise ValueError("route_search_cap must be positive")
        if not self.routes or any(
                route not in ("surface", "tunnel")
                for route in self.routes):
            raise ValueError("routes must contain surface and/or tunnel")
        if len(set(self.routes)) != len(self.routes):
            raise ValueError("routes cannot contain duplicates")
        if self.natural_obsidian and not any(
                49 in item.block_ids for item in self.resources):
            raise ValueError(
                "natural_obsidian needs an obsidian resource requirement")

    @classmethod
    def parse(cls, value):
        if isinstance(value, cls):
            return value
        value = _closed_mapping(
            value,
            what="world goal",
            keys={
                "requirements", "resources", "proximity",
                "natural_obsidian", "hazard_clearance", "node_cap",
                "route_search_cap", "routes",
            },
        )
        raw_resources = value.get("requirements", value.get("resources"))
        if isinstance(raw_resources, Mapping):
            resources = tuple(
                ResourceRequirement.parse(item, name=str(name))
                for name, item in raw_resources.items())
        elif isinstance(raw_resources, Sequence) \
                and not isinstance(raw_resources, (str, bytes)):
            resources = tuple(ResourceRequirement.parse(item)
                              for item in raw_resources)
        else:
            raise TypeError("goal requirements must be a mapping or list")

        raw_proximity = value.get("proximity", ())
        if isinstance(raw_proximity, Mapping):
            proximity = tuple(
                ProximityConstraint.parse(item, name=str(name))
                for name, item in raw_proximity.items())
        elif isinstance(raw_proximity, Sequence) \
                and not isinstance(raw_proximity, (str, bytes)):
            proximity = tuple(ProximityConstraint.parse(item)
                              for item in raw_proximity)
        else:
            raise TypeError("goal proximity must be a mapping or list")
        natural_obsidian = _boolean(
            value.get("natural_obsidian", False),
            what="goal natural_obsidian",
        )
        return cls(
            resources=resources,
            proximity=proximity,
            natural_obsidian=natural_obsidian,
            hazard_clearance=int(value.get("hazard_clearance", 0)),
            node_cap=int(value.get("node_cap", 20_000)),
            route_search_cap=int(value.get("route_search_cap", 512)),
            routes=tuple(value.get("routes", ("surface", "tunnel"))),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "requirements": [item.to_dict() for item in self.resources],
            "proximity": [item.to_dict() for item in self.proximity],
            "natural_obsidian": self.natural_obsidian,
            "hazard_clearance": self.hazard_clearance,
            "node_cap": self.node_cap,
            "route_search_cap": self.route_search_cap,
            "routes": list(self.routes),
        }


class WorldPlanningRejected(CaseRejected):
    """Expected seed rejection with machine-readable preflight context."""

    def __init__(self, reason: str, *, code: str, stage: str,
                 resource: str | None = None, details=None):
        super().__init__(reason, code=code)
        self.stage = stage
        self.resource = resource
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "reason": str(self),
            "stage": self.stage,
            "resource": self.resource,
            "details": _json(self.details),
        }


@dataclass(frozen=True, slots=True)
class RoutePlan:
    kind: str
    stands: tuple[Cell, ...]
    reserved_cells: tuple[Cell, ...]
    support_cells: tuple[Cell, ...]
    mine_cells: tuple[Cell, ...]

    @classmethod
    def surface(cls, stands: Iterable[Cell]):
        stands = _cells(stands)
        occupied = tuple(dict.fromkeys(
            cell for feet in stands
            for cell in (feet, (feet[0], feet[1] + 1, feet[2]))))
        supports = tuple(dict.fromkeys(
            (feet[0], feet[1] - 1, feet[2]) for feet in stands))
        return cls("surface", stands, occupied, supports, ())

    @classmethod
    def tunnel(cls, plan: ExcavationPlan):
        return cls(
            "tunnel", _cells(plan.stands), _cells(plan.reserved_cells),
            _cells(plan.support_cells), _cells(plan.mine_cells))

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "stands": _json(self.stands),
            "reserved_cells": _json(self.reserved_cells),
            "support_cells": _json(self.support_cells),
            "mine_cells": _json(self.mine_cells),
        }


@dataclass(frozen=True, slots=True)
class MiningTargetPlan:
    """One ordered mining leg with its own reachability and safety proof."""

    cell: Cell
    mining_stance: Cell
    route: RoutePlan
    removal: RemovalProof

    def __post_init__(self):
        if not self.route.stands:
            raise ValueError("a mining target route cannot be empty")
        if self.route.stands[-1] != self.mining_stance:
            raise ValueError("target route must end at its mining stance")
        if self.removal.target != self.cell:
            raise ValueError("removal proof must describe the target cell")

    def to_dict(self) -> dict[str, object]:
        return _json({
            "cell": self.cell,
            "mining_stance": self.mining_stance,
            "route": self.route,
            "removal": self.removal,
        })

    @property
    def fluid_seal_cells(self) -> tuple[Cell, ...]:
        """Unique solid placements required before this target is mined."""

        return tuple(dict.fromkeys(self.removal.fluid_seal_cells))


@dataclass(frozen=True, slots=True)
class ClusterPlan:
    """Certified mining targets from one snapshot candidate group.

    For ``source=initial_snapshot``, ``cluster_size`` is the exact connected
    component size.  For ``source=initial_snapshot_target_pool``, it is the
    exact matching population size; the source explicitly records that the
    planner avoided computing irrelevant component boundaries.
    """

    anchor: Cell
    cluster_size: int
    selected_count: int
    block_ids: tuple[int, ...]
    targets: tuple[MiningTargetPlan, ...]
    source: str = "initial_snapshot"

    def __post_init__(self):
        if not self.targets:
            raise ValueError("a cluster plan needs at least one target")
        if self.selected_count != len(self.targets):
            raise ValueError(
                "cluster selected_count must match its ordered targets")

    @property
    def target_cells(self) -> tuple[Cell, ...]:
        """Return the cells from all ordered target plans."""

        return tuple(target.cell for target in self.targets)

    @property
    def mining_stance(self) -> Cell:
        """Return the first target's mining stance."""

        return self.targets[0].mining_stance

    @property
    def route(self) -> RoutePlan:
        """First target route retained for older single-target callers."""

        return self.targets[0].route

    @property
    def fluid_seal_cells(self) -> tuple[Cell, ...]:
        """Unique fluid seals required across all ordered cluster targets."""

        return tuple(dict.fromkeys(
            cell for target in self.targets
            for cell in target.fluid_seal_cells))

    def to_dict(self) -> dict[str, object]:
        return _json({
            "anchor": self.anchor,
            "cluster_size": self.cluster_size,
            "selected_count": self.selected_count,
            "block_ids": self.block_ids,
            "target_cells": self.target_cells,
            "mining_stance": self.mining_stance,
            "route": self.route,
            "targets": self.targets,
            "fluid_seal_cells": self.fluid_seal_cells,
            "fluid_seal_count": len(self.fluid_seal_cells),
            "source": self.source,
        })


@dataclass(frozen=True, slots=True)
class ResourcePlan:
    name: str
    blocks: tuple[BlockRef, ...]
    minimum: int
    available_count: int
    selected_count: int
    clusters: tuple[ClusterPlan, ...]
    natural: bool = False

    @property
    def fluid_seal_cells(self) -> tuple[Cell, ...]:
        """Unique fluid seals required across all selected clusters."""

        return tuple(dict.fromkeys(
            cell for cluster in self.clusters
            for cell in cluster.fluid_seal_cells))

    def to_dict(self) -> dict[str, object]:
        return _json({
            "name": self.name,
            "blocks": self.blocks,
            "minimum": self.minimum,
            "available_count": self.available_count,
            "selected_count": self.selected_count,
            "natural": self.natural,
            "clusters": self.clusters,
            "fluid_seal_cells": self.fluid_seal_cells,
            "fluid_seal_count": len(self.fluid_seal_cells),
        })


@dataclass(frozen=True, slots=True)
class ProximityResult:
    name: str
    blocks: tuple[BlockRef, ...]
    nearest_cell: Cell
    distance: float
    minimum: float
    maximum: float

    def to_dict(self) -> dict[str, object]:
        return _json({
            "name": self.name,
            "blocks": self.blocks,
            "nearest_cell": self.nearest_cell,
            "distance": self.distance,
            "minimum": self.minimum,
            "maximum": None if math.isinf(self.maximum) else self.maximum,
        })


@dataclass(frozen=True, slots=True)
class WorldPlan:
    """Immutable, JSON-ready preflight plan for one snapshot and decision."""

    snapshot_sha256: str
    world_seed: int
    decision_seed: int
    start: Cell
    goal: GoalSpec
    proximity: tuple[ProximityResult, ...]
    resources: tuple[ResourcePlan, ...]
    safe_cells: tuple[Cell, ...]

    @property
    def fluid_seal_cells(self) -> tuple[Cell, ...]:
        """Unique fluid seals required by the complete immutable plan."""

        return tuple(dict.fromkeys(
            cell for resource in self.resources
            for cell in resource.fluid_seal_cells))

    def to_dict(self) -> dict[str, object]:
        return _json({
            "schema": "bedrock.world_plan.v1",
            "snapshot_sha256": self.snapshot_sha256,
            "world_seed": self.world_seed,
            "decision_seed": self.decision_seed,
            "start": self.start,
            "goal": self.goal,
            "proximity": self.proximity,
            "resources": self.resources,
            "safe_cells": self.safe_cells,
            "fluid_seal_cells": self.fluid_seal_cells,
            "fluid_seal_count": len(self.fluid_seal_cells),
        })

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)


@dataclass(frozen=True, slots=True)
class FluidSealLedger:
    """Material accounting for the unique seals required by one plan.

    Multiple target proofs may name the same water cell.  Charging the raw
    per-target count would reject a plan that reuses an already placed cap;
    this ledger therefore accounts for unique coordinates only.  ``reserve``
    is inventory that must remain untouched for later operations, such as
    the three temporary supports in a minimal Nether portal frame.
    """

    cells: tuple[Cell, ...]
    available: int
    reserve: int = 0

    def __post_init__(self):
        cells = tuple(dict.fromkeys(_cells(self.cells)))
        object.__setattr__(self, "cells", cells)
        for value, label in (
            (self.available, "available"), (self.reserve, "reserve"),
        ):
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value < 0):
                raise ValueError(
                    f"fluid seal {label} must be a non-negative integer")
        if self.reserve > self.available:
            raise ValueError(
                "fluid seal reserve cannot exceed available material")

    @property
    def required(self) -> int:
        return len(self.cells)

    @property
    def usable(self) -> int:
        return self.available - self.reserve

    @property
    def shortfall(self) -> int:
        return max(0, self.required - self.usable)

    @property
    def within_budget(self) -> bool:
        return self.shortfall == 0

    def to_dict(self) -> dict[str, object]:
        return _json({
            "cells": self.cells,
            "required": self.required,
            "available": self.available,
            "reserve": self.reserve,
            "usable": self.usable,
            "shortfall": self.shortfall,
            "within_budget": self.within_budget,
        })


def fluid_seal_ledger(
    plan: WorldPlan | ResourcePlan | ClusterPlan,
    *,
    available: int,
    reserve: int = 0,
) -> FluidSealLedger:
    """Account for aggregate unique seal placements in ``plan``."""

    return FluidSealLedger(
        tuple(plan.fluid_seal_cells), available, reserve)


def require_fluid_seal_budget(
    plan: WorldPlan | ResourcePlan | ClusterPlan,
    *,
    available: int,
    reserve: int = 0,
    resource: str | None = None,
) -> FluidSealLedger:
    """Return the seal ledger or reject before any irreversible action."""

    ledger = fluid_seal_ledger(
        plan, available=available, reserve=reserve)
    if not ledger.within_budget:
        raise WorldPlanningRejected(
            "planned fluid seals exceed available material after reserve",
            code="world_plan_fluid_seal_budget",
            stage="resources",
            resource=resource,
            details=ledger.to_dict(),
        )
    return ledger


@dataclass(frozen=True, slots=True)
class ToolDurabilityLedger:
    """Usable block breaks charged to one kind of tool.

    ``capacity_per_copy`` is deliberately supplied by the caller.  It means
    actual usable block breaks, not an item's maximum-damage metadata; that
    keeps engine-version durability semantics out of the generic planner.
    ``planned_cells`` are unique and retain first-use order.  Operations that
    intentionally break a replaced cell again, or are outside a resource
    plan (for example workstation and return-tunnel clears), belong in
    ``extra_breaks``.
    """

    tool: ToolRef
    capacity_per_copy: int
    copies: int
    planned_cells: tuple[Cell, ...] = ()
    extra_breaks: int = 0

    def __post_init__(self):
        if isinstance(self.tool, bool) or not isinstance(
                self.tool, (str, int)):
            raise TypeError("durability tool must be an item name or id")
        if isinstance(self.tool, str) and not self.tool.strip():
            raise ValueError("durability tool name cannot be empty")
        cells = tuple(dict.fromkeys(_cells(self.planned_cells)))
        object.__setattr__(self, "planned_cells", cells)
        for value, label, minimum in (
            (self.capacity_per_copy, "capacity_per_copy", 1),
            (self.copies, "copies", 0),
            (self.extra_breaks, "extra_breaks", 0),
        ):
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value < minimum):
                qualifier = "positive" if minimum else "non-negative"
                raise ValueError(
                    f"tool durability {label} must be a {qualifier} integer")

    @property
    def planned_breaks(self) -> int:
        return len(self.planned_cells)

    @property
    def required(self) -> int:
        return self.planned_breaks + self.extra_breaks

    @property
    def available(self) -> int:
        return self.capacity_per_copy * self.copies

    @property
    def shortfall(self) -> int:
        return max(0, self.required - self.available)

    @property
    def within_budget(self) -> bool:
        return self.shortfall == 0

    def to_dict(self) -> dict[str, object]:
        return _json({
            "tool": self.tool,
            "capacity_per_copy": self.capacity_per_copy,
            "copies": self.copies,
            "planned_cells": self.planned_cells,
            "planned_breaks": self.planned_breaks,
            "extra_breaks": self.extra_breaks,
            "required": self.required,
            "available": self.available,
            "shortfall": self.shortfall,
            "within_budget": self.within_budget,
        })


def tool_durability_ledgers(
    plan: WorldPlan,
    *,
    capacities: Mapping[ToolRef, int],
    route_tools: Mapping[str, ToolRef | None],
    target_tools: Mapping[str, ToolRef | None] | None = None,
    copies: Mapping[ToolRef, int] | None = None,
    extra_breaks: Mapping[ToolRef, int] | None = None,
) -> tuple[ToolDurabilityLedger, ...]:
    """Account for ordered unique resource-plan breaks by selected tool.

    Route and target tools are separate because a progression may excavate a
    route with one tool and harvest the target by hand or with another.  A
    missing/``None`` assignment means that break does not consume a tracked
    tool.  Cells are de-duplicated globally in plan order: an already opened
    tunnel cell cannot consume durability again merely because a later route
    contains it.  Deliberate re-mining is represented by ``extra_breaks``.
    """

    capacities = dict(capacities)
    route_tools = dict(route_tools)
    target_tools = (route_tools if target_tools is None
                    else dict(target_tools))
    copies = dict(copies or {})
    extra_breaks = dict(extra_breaks or {})
    referenced = {
        tool for mapping in (route_tools, target_tools)
        for tool in mapping.values() if tool is not None
    } | set(copies) | set(extra_breaks)
    missing = referenced - set(capacities)
    if missing:
        raise ValueError(
            "tool durability capacity missing for: "
            + ", ".join(sorted(map(str, missing))))

    planned: dict[ToolRef, list[Cell]] = {
        tool: [] for tool in capacities
    }
    seen: set[Cell] = set()
    for resource in plan.resources:
        route_tool = route_tools.get(resource.name)
        target_tool = target_tools.get(resource.name)
        for cluster in resource.clusters:
            for target in cluster.targets:
                for cell in target.route.mine_cells:
                    if cell in seen:
                        continue
                    seen.add(cell)
                    if route_tool is not None:
                        planned[route_tool].append(cell)
                if target.cell in seen:
                    continue
                seen.add(target.cell)
                if target_tool is not None:
                    planned[target_tool].append(target.cell)

    return tuple(ToolDurabilityLedger(
        tool=tool,
        capacity_per_copy=capacity,
        copies=copies.get(tool, 1),
        planned_cells=tuple(planned[tool]),
        extra_breaks=extra_breaks.get(tool, 0),
    ) for tool, capacity in capacities.items())


def require_tool_durability_budget(
    plan: WorldPlan,
    **kwargs,
) -> tuple[ToolDurabilityLedger, ...]:
    """Return tool ledgers or reject an under-provisioned plan up front."""

    ledgers = tool_durability_ledgers(plan, **kwargs)
    failures = tuple(ledger for ledger in ledgers
                     if not ledger.within_budget)
    if failures:
        raise WorldPlanningRejected(
            "planned block breaks exceed crafted tool durability",
            code="world_plan_tool_durability_budget",
            stage="resources",
            details={"tools": tuple(
                ledger.to_dict() for ledger in failures)},
        )
    return ledgers


class _PlanningWorld:
    """Snapshot overlay containing only mutations guaranteed by prior legs."""

    def __init__(
        self,
        world: SnapshotWorldMap,
        *,
        opened: Iterable[Cell] = (),
        removed: Iterable[Cell] = (),
        placed: Mapping[Cell, BlockRef] | None = None,
    ):
        self.world = world
        self.opened = frozenset(opened)
        self.removed = frozenset(removed)
        self.placed = {tuple(map(int, cell)): block
                       for cell, block in dict(placed or {}).items()}

    def in_bounds(self, cell: Cell) -> bool:
        return self.world.in_bounds(*cell)

    def block(self, cell: Cell):
        if cell in self.placed:
            return self.placed[cell]
        if cell in self.opened or cell in self.removed:
            return "air"
        return self.world.block(*cell)


class _ProtectedWorld:
    """Read-only view that prevents a tunnel from mining protected solids."""

    def __init__(self, world, protected: Iterable[Cell],
                 allowed_passage: Iterable[Cell] = ()):
        self.world = world
        self.protected = frozenset(protected)
        self.allowed_passage = frozenset(allowed_passage)

    def in_bounds(self, cell: Cell) -> bool:
        return self.world.in_bounds(cell)

    def block(self, cell: Cell):
        block = self.world.block(cell)
        if cell in self.protected and cell not in self.allowed_passage \
                and DEFAULT_BLOCK_RULES.is_solid(block) \
                and not DEFAULT_BLOCK_RULES.is_fluid(block) \
                and not DEFAULT_BLOCK_RULES.is_hazard(block):
            return _PROTECTED_SOLID
        if cell in self.protected and cell not in self.allowed_passage:
            return _PROTECTED_VOID
        return block


_PROTECTED_SOLID = "__netherite_protected_solid__"
_PROTECTED_VOID = "__netherite_protected_void__"


class _ProtectedRules:
    """Make preserved solid immutable while retaining valid floor semantics."""

    def __init__(self, rules=DEFAULT_BLOCK_RULES):
        self.rules = rules

    def name(self, value):
        return self.rules.name(value)

    def is_passable(self, value):
        return False if value in (_PROTECTED_SOLID, _PROTECTED_VOID) \
            else self.rules.is_passable(value)

    def is_fluid(self, value):
        return False if value in (_PROTECTED_SOLID, _PROTECTED_VOID) \
            else self.rules.is_fluid(value)

    def is_hazard(self, value):
        return False if value in (_PROTECTED_SOLID, _PROTECTED_VOID) \
            else self.rules.is_hazard(value)

    def is_unbreakable(self, value):
        # Tunnel stands reject unbreakable floors. The sentinel is immutable
        # because it is not excavatable, while remaining a legal dry support.
        return False if value in (_PROTECTED_SOLID, _PROTECTED_VOID) \
            else self.rules.is_unbreakable(value)

    def is_solid(self, value):
        if value == _PROTECTED_SOLID:
            return True
        if value == _PROTECTED_VOID:
            return False
        return self.rules.is_solid(value)

    def is_excavatable(self, value):
        return False if value in (_PROTECTED_SOLID, _PROTECTED_VOID) \
            else self.rules.is_excavatable(value)


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.dist(tuple(left), tuple(right))


def _cluster_distance(
    cluster: ResourceCluster, origin: Sequence[float],
) -> float:
    bounds = cluster.bounds
    nearest = (
        min(max(origin[0], bounds.min_x), bounds.max_x),
        min(max(origin[1], bounds.min_y), bounds.max_y),
        min(max(origin[2], bounds.min_z), bounds.max_z),
    )
    return _distance(origin, nearest)


@dataclass(frozen=True, slots=True)
class _PlannedCluster:
    plan: ClusterPlan
    current_start: Cell
    opened: frozenset[Cell]
    removed: frozenset[Cell]
    traversed_passage: frozenset[Cell]
    traversed_supports: frozenset[Cell]


class WorldPlanBuilder:
    """Select safe resource approaches without executing engine actions."""

    def __init__(
        self,
        world: SnapshotWorldMap,
        goal: GoalSpec | Mapping,
        *,
        decision_seed: int,
        cluster_sample_limit: int = 256,
        start: Cell | None = None,
        opened: Iterable[Cell] = (),
        removed: Iterable[Cell] = (),
        protected: Iterable[Cell] = (),
        placed: Mapping[Cell, BlockRef] | None = None,
        preserve_traversed_route: bool = False,
        traversed_passage: Iterable[Cell] = (),
        traversed_supports: Iterable[Cell] = (),
    ):
        if not isinstance(world, SnapshotWorldMap):
            raise TypeError("WorldPlanBuilder needs an open SnapshotWorldMap")
        if world.closed:
            raise ValueError("WorldPlanBuilder cannot use a closed snapshot")
        if cluster_sample_limit < 1:
            raise ValueError("cluster_sample_limit must be positive")
        self.world = world
        self.goal = GoalSpec.parse(goal)
        self.decision_seed = int(decision_seed)
        self.cluster_sample_limit = int(cluster_sample_limit)
        player = world.player
        self.origin = (player.x, player.y, player.z)
        self.start = ((math.floor(player.x), math.floor(player.y),
                       math.floor(player.z))
                      if start is None else tuple(map(int, start)))
        if not world.in_bounds(*self.start):
            raise ValueError("planner start is outside snapshot bounds")
        self._current_start = self.start
        self._opened: set[Cell] = {tuple(map(int, cell))
                                  for cell in opened}
        self._removed: set[Cell] = {tuple(map(int, cell))
                                   for cell in removed}
        self._protected: frozenset[Cell] = frozenset(
            tuple(map(int, cell)) for cell in protected)
        if not isinstance(preserve_traversed_route, bool):
            raise TypeError("preserve_traversed_route must be true or false")
        self.preserve_traversed_route = preserve_traversed_route
        self._traversed_passage: set[Cell] = {
            tuple(map(int, cell)) for cell in traversed_passage}
        self._traversed_supports: set[Cell] = {
            tuple(map(int, cell)) for cell in traversed_supports}
        if self._traversed_passage.intersection(
                self._traversed_supports):
            raise ValueError(
                "traversed passage and support cells must be disjoint")
        if (self._traversed_passage or self._traversed_supports) \
                and not self.preserve_traversed_route:
            raise ValueError(
                "traversed route cells require preserve_traversed_route")
        self._placed = {tuple(map(int, cell)): block
                        for cell, block in dict(placed or {}).items()}
        self._route_searches = 0
        self._active_resource = None
        self._active_routes = self.goal.routes
        self._active_sealable_fluids = ()
        self._active_max_drop_fall = None

    def _rng(self, label: str) -> random.Random:
        digest = hashlib.sha256(
            f"{self.decision_seed}:{label}".encode()).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def _reject(self, reason, *, code, stage, resource=None, details=None):
        raise WorldPlanningRejected(
            reason, code=code, stage=stage,
            resource=resource, details=details)

    def _consume_route_search(self):
        """Bound total A* calls so one hostile seed rejects promptly."""
        if self._route_searches >= self.goal.route_search_cap:
            self._reject(
                "world planning exhausted its aggregate route-search budget",
                code="world_plan_search_budget",
                stage="routes", resource=self._active_resource,
                details={"attempted": self._route_searches,
                         "limit": self.goal.route_search_cap},
            )
        self._route_searches += 1

    def _proximity(self) -> tuple[ProximityResult, ...]:
        results = []
        for item in self.goal.proximity:
            nearest = self.world.nearest(
                item.block_ids, self.origin, limit=1)
            if not nearest:
                self._reject(
                    f"no block matches proximity constraint {item.name!r}",
                    code="world_plan_proximity_missing",
                    stage="proximity", resource=item.name,
                    details={"constraint": item.to_dict()},
                )
            cell = nearest[0]
            position = cell.position
            distance = _distance(
                self.origin,
                (position[0] + 0.5, position[1] + 0.5,
                 position[2] + 0.5),
            )
            if distance < item.minimum or distance > item.maximum:
                self._reject(
                    f"nearest {item.name!r} is {distance:.3f} blocks away",
                    code="world_plan_proximity_out_of_range",
                    stage="proximity", resource=item.name,
                    details={
                        "constraint": item.to_dict(),
                        "nearest_cell": position,
                        "actual_distance": distance,
                    },
                )
            results.append(ProximityResult(
                item.name, item.blocks, position, distance,
                item.minimum, item.maximum))
        return tuple(results)

    def _surface_approach(
        self, world, start: Cell, target: Cell, rng: random.Random,
        *, search_deadline: int,
    ) -> tuple[Cell, RoutePlan] | None:
        try:
            stances = safe_mining_stances(
                world, target,
                hazard_clearance=self.goal.hazard_clearance,
                decision_rng=rng,
            )
        except PlanRejected:
            return None
        # Each route search is bounded, and so is the number of alternatives.
        # This keeps one hostile seed from monopolizing a parallel batch.
        pickup_stances = tuple(
            stance for stance in stances
            if math.hypot(
                stance.feet[0] - target[0],
                stance.feet[2] - target[2]) <= 1.25)
        for stance in pickup_stances[:4]:
            if self._route_searches >= search_deadline:
                break
            try:
                self._consume_route_search()
                path = plan_surface_path(
                    world, start, stance.feet,
                    hazard_clearance=self.goal.hazard_clearance,
                    node_cap=self.goal.node_cap,
                    decision_rng=rng,
                )
            except PlanRejected:
                continue
            route = RoutePlan.surface(path)
            if target in route.support_cells:
                continue
            return stance.feet, route
        return None

    def _current_stance_approach(self, world, current, target):
        """Mine another visible vein block without a redundant global A*."""

        eye = current[0] + 0.5, current[1] + 1.62, current[2] + 0.5
        point = target[0] + 0.5, target[1] + 0.5, target[2] + 0.5
        if math.dist(eye, point) > MINING_REACH:
            return None
        if math.hypot(
                current[0] - target[0], current[2] - target[2]) > 1.25:
            # Reach alone is not a collection proof. A block mined two or
            # three horizontal cells away can drop into an inaccessible wall
            # pocket. Force a closer planned stance so exact mining and exact
            # pickup share the same traversable local geometry.
            return None
        if not is_surface_stand(
                world, current, hazard_clearance=self.goal.hazard_clearance):
            return None
        if not clear_voxel_ray(world, eye, point, target=target):
            return None
        if "tunnel" in self._active_routes:
            route = RoutePlan.tunnel(ExcavationPlan(
                stands=(current,),
                reserved_cells=(current,
                                (current[0], current[1] + 1, current[2])),
                support_cells=((current[0], current[1] - 1, current[2]),),
                mine_cells=(),
            ))
        elif "surface" in self._active_routes:
            route = RoutePlan.surface((current,))
        else:
            return None
        if target in route.support_cells:
            return None
        return current, route

    def _tunnel_approach(
        self,
        planning_world,
        start: Cell,
        target: Cell,
        protected: frozenset[Cell],
        allowed_passage: frozenset[Cell],
        rng: random.Random,
        *,
        search_deadline: int,
    ) -> tuple[Cell, RoutePlan] | None:
        world = _ProtectedWorld(
            planning_world, protected,
            allowed_passage=allowed_passage)
        protected_rules = _ProtectedRules()
        offsets = [
            (dx, dy, dz)
            for radius in (1, 2, 3)
            for dy in (0, -1, 1)
            for dx, dz in ((radius, 0), (0, radius),
                           (-radius, 0), (0, -radius))
        ]
        # Prefer stances whose final two-block excavation itself exposes the
        # target ray. A buried ore below the player is often reachable from a
        # one-level-lower side stance, while a geometrically closer stance
        # leaves solid stone between the eye and ore. Ranking the cheap ray
        # lower bound first avoids spending every bounded A* call on routes
        # that can never pass the later visibility proof.
        variance = {offset: rng.random() for offset in offsets}

        def ray_occlusions(offset):
            feet = (target[0] + offset[0], target[1] + offset[1],
                    target[2] + offset[2])
            body = {feet, (feet[0], feet[1] + 1, feet[2])}
            eye = feet[0] + 0.5, feet[1] + 1.62, feet[2] + 0.5
            point = target[0] + 0.5, target[1] + 0.5, target[2] + 0.5
            return sum(
                cell != target and cell not in body
                and (not planning_world.in_bounds(cell)
                     or not DEFAULT_BLOCK_RULES.is_passable(
                         planning_world.block(cell)))
                for cell in voxel_ray(eye, point))

        offsets.sort(key=lambda offset: (
            ray_occlusions(offset),
            round(_distance(start, (
                target[0] + offset[0], target[1] + offset[1],
                target[2] + offset[2])), 12),
            variance[offset], offset,
        ))
        target_point = (
            target[0] + 0.5, target[1] + 0.5, target[2] + 0.5)
        for dx, dy, dz in offsets:
            if self._route_searches >= search_deadline:
                break
            if abs(dx) + abs(dz) != 1:
                # Reach is not collection. A drop in a low or walled target
                # cavity cannot be recovered from a stance two or three
                # horizontal cells away, even though the block itself is
                # clickable. Require a cardinally adjacent final stance.
                continue
            feet = target[0] + dx, target[1] + dy, target[2] + dz
            head = feet[0], feet[1] + 1, feet[2]
            # A resource can be visible from directly below, but that is not
            # a stand the player can occupy before the target is mined. The
            # execution order is route first, target second, so feet/head
            # must exclude the protected target itself.
            if target in (feet, head):
                continue
            if (feet in protected and feet not in allowed_passage) \
                    or (head in protected
                        and head not in allowed_passage):
                continue
            # A zero-clearance stage may work beside a fluid, but never with
            # fluid in the player's feet/head cells: is_excavation_stand()
            # checks those cells directly. This is required for dry side
            # galleries beside a water-covered obsidian shelf; forcing one
            # block of fluid clearance makes its closest visible stance
            # impossible even though the player never enters the water.
            clearance = self.goal.hazard_clearance
            if not is_excavation_stand(
                    world, feet, rules=protected_rules,
                    hazard_clearance=clearance):
                continue
            try:
                self._consume_route_search()
                excavation = plan_staircase_tunnel(
                    world, start, feet,
                    rules=protected_rules,
                    hazard_clearance=clearance,
                    node_cap=self.goal.node_cap,
                    decision_rng=rng,
                )
            except PlanRejected:
                continue
            if any(target in (
                    stand,
                    (stand[0], stand[1] + 1, stand[2]))
                    for stand in excavation.stands):
                # The target is protected from route mining, but that alone
                # does not make a route executable: a staircase cannot pass
                # with the resource occupying any intermediate body voxel.
                continue
            if not _excavation_clear_rays(
                    planning_world, excavation, protected=(target,)):
                # A body voxel can be geometrically excavatable yet hidden
                # behind a protected resource or another uncleared block.
                # The live executor clears in this exact top-down order.
                continue
            if not _excavation_falling_blocks_stable(
                    planning_world, excavation):
                continue
            eye = feet[0] + 0.5, feet[1] + 1.62, feet[2] + 0.5
            if _distance(eye, target_point) > MINING_REACH:
                continue
            opened = set(excavation.reserved_cells)
            ray = voxel_ray(eye, target_point)
            visible = all(
                cell == target or cell in opened
                or (planning_world.in_bounds(cell)
                    and DEFAULT_BLOCK_RULES.is_passable(
                        planning_world.block(cell)))
                for cell in ray
            )
            if not visible:
                continue
            return feet, RoutePlan.tunnel(excavation)
        return None

    def _cluster_plan(
        self,
        cluster: ResourceCluster,
        selected_count: int,
        rng: random.Random,
        *,
        start: Cell,
        opened: frozenset[Cell],
        removed: frozenset[Cell],
        source: str = "initial_snapshot",
    ) -> _PlannedCluster | tuple[PlanRejected, ...]:
        cells = cluster.cells
        if not cells:
            return (PlanRejected(
                RejectReason.UNREACHABLE_TARGET,
                goal=cluster.anchor,
                detail="cluster sampling retained no target cells"),)

        current = start
        local_opened = set(opened)
        local_removed = set(removed)
        local_passage = set(self._traversed_passage)
        local_supports = set(self._traversed_supports)
        unavailable = (
            self._protected | frozenset(local_passage | local_supports))
        remaining = set(cells).difference(local_removed, unavailable)
        targets: list[MiningTargetPlan] = []
        failures: list[PlanRejected] = []

        while len(targets) < selected_count and remaining:
            # Re-rank every leg from the last accepted stance.  The random
            # tiebreak remains deterministic for a decision seed.  Consume
            # RNG in sorted order so translating coordinates cannot change
            # set iteration order and therefore the selected geometry.
            variance = {
                cell: rng.random() for cell in sorted(remaining)}
            ranking_world = _PlanningWorld(
                self.world, opened=local_opened, removed=local_removed,
                placed=self._placed)

            def exposure(cell):
                adjacent = (
                    (cell[0] + 1, cell[1], cell[2]),
                    (cell[0] - 1, cell[1], cell[2]),
                    (cell[0], cell[1] + 1, cell[2]),
                    (cell[0], cell[1] - 1, cell[2]),
                    (cell[0], cell[1], cell[2] + 1),
                    (cell[0], cell[1], cell[2] - 1),
                )
                return sum(
                    ranking_world.in_bounds(other)
                    and DEFAULT_BLOCK_RULES.is_passable(
                        ranking_world.block(other))
                    for other in adjacent)

            ranked = sorted(
                remaining,
                key=lambda cell: ((
                    0 if _drop_fall_within(
                        ranking_world, cell,
                        self._active_max_drop_fall) else 1,
                    round(_distance(current, cell), 12),
                    -exposure(cell), variance[cell], cell,
                ) if self._active_max_drop_fall is not None else (
                    -exposure(cell),
                    round(_distance(current, cell), 12),
                    variance[cell], cell,
                )),
            )
            accepted = False
            candidates = ranked[:min(8, len(ranked))]
            for target_index, target in enumerate(candidates):
                # A previously selected target is guaranteed to disappear.
                # It cannot therefore support this target's route or mining
                # stance, even though the immutable source snapshot still
                # labels it solid. Reject such candidates before planning so
                # the live executor never depends on already-mined terrain.
                planning_overlay = _PlanningWorld(
                    self.world, opened=local_opened,
                    removed=local_removed, placed=self._placed)
                # Divide the remaining aggregate budget across the remaining
                # ranked blocks. One awkward ore must not spend every A* call
                # proving dozens of stance offsets while equally good blocks
                # are never attempted. A large generic budget still permits
                # deep per-target exploration; the tight progression preset
                # gets eight alternatives per target at its default 64/8.
                budget_left = (
                    self.goal.route_search_cap - self._route_searches)
                if budget_left <= 0:
                    self._consume_route_search()
                # Preserve enough aggregate searches for every resource leg
                # still owed, not merely for the alternatives in this one
                # eight-cell window. Otherwise an early awkward block in a
                # long quarry may consume the entire budget before the later
                # sequential targets get their one required route search.
                targets_left = max(
                    len(candidates) - target_index,
                    selected_count - len(targets),
                )
                target_budget = max(1, budget_left // targets_left)
                search_deadline = min(
                    self.goal.route_search_cap,
                    self._route_searches + target_budget)
                planning_world = planning_overlay
                if (self._active_max_drop_fall is not None
                        and not _drop_fall_within(
                            planning_world, target,
                            self._active_max_drop_fall)):
                    failures.append(PlanRejected(
                        RejectReason.UNSAFE_REMOVAL,
                        start=current, goal=target,
                        detail="mined drop exceeds configured fall bound"))
                    continue
                # Preserve the chosen target and only the number of future
                # sampled blocks this cluster still owes. Protecting every
                # cell of an incomplete giant cluster (stone is the common
                # case) turns the whole mountain into artificial bedrock and
                # makes a legitimate quarry/tunnel impossible. Keep the
                # future reserve far from the current leg so the planner may
                # excavate ordinary same-resource blocks on its way in.
                future_needed = selected_count - len(targets) - 1
                future = sorted(
                    remaining.difference((target,)),
                    key=lambda cell: (
                        -round(_distance(current, cell), 12), cell),
                )[:future_needed]
                protected = (
                    frozenset((target, *future))
                    | self._protected
                    | frozenset(local_passage | local_supports)
                )
                approach = self._current_stance_approach(
                    planning_world, current, target)
                if approach is None and "surface" in self._active_routes:
                    approach = self._surface_approach(
                        planning_world, current, target, rng,
                        search_deadline=search_deadline)
                if approach is None and "tunnel" in self._active_routes:
                    approach = self._tunnel_approach(
                        planning_world, current, target,
                        protected, frozenset(local_passage), rng,
                        search_deadline=search_deadline)
                if approach is None:
                    failures.append(PlanRejected(
                        RejectReason.UNREACHABLE_TARGET,
                        start=current, goal=target,
                        detail="no safe surface or tunnel approach"))
                    continue
                stance, route = approach
                stale_supports = set(route.support_cells).intersection(
                    local_removed | set(route.mine_cells))
                if stale_supports:
                    failures.append(PlanRejected(
                        RejectReason.UNSAFE_REMOVAL,
                        start=current, goal=target,
                        detail="route depends on a removed/mined support "
                        f"cell {sorted(stale_supports)[0]}"))
                    continue
                try:
                    removal = plan_safe_removal(
                        planning_world,
                        target,
                        route_stands=route.stands,
                        protected_support_cells=route.support_cells,
                        sealable_fluids=self._active_sealable_fluids,
                    )
                except PlanRejected as error:
                    failures.append(error)
                    continue
                mutations = {
                    target, *route.mine_cells, *removal.fluid_seal_cells}
                conflict = mutations.intersection(
                    local_passage | local_supports)
                if conflict:
                    failures.append(PlanRejected(
                        RejectReason.UNSAFE_REMOVAL,
                        start=current, goal=target,
                        detail="later target would damage preserved route "
                        f"cell {sorted(conflict)[0]}"))
                    continue
                collection_world = _PlanningWorld(
                    self.world,
                    opened=local_opened | set(route.mine_cells),
                    removed=local_removed | {target},
                    placed=self._placed)
                if not _target_falling_blocks_stable(
                        collection_world, target):
                    failures.append(PlanRejected(
                        RejectReason.UNSAFE_REMOVAL,
                        start=stance, goal=target,
                        detail="target removal would release an overhead "
                        "falling block"))
                    continue
                if not _drop_landing_within_reach(
                        collection_world, target, stance,
                        self._active_max_drop_fall):
                    failures.append(PlanRejected(
                        RejectReason.UNSAFE_REMOVAL,
                        start=stance, goal=target,
                        detail="mined drop would fall below the proved "
                        "collection stance"))
                    continue

                targets.append(MiningTargetPlan(
                    cell=target,
                    mining_stance=stance,
                    route=route,
                    removal=removal,
                ))
                local_opened.update(route.mine_cells)
                local_removed.add(target)
                remaining.difference_update(route.mine_cells)
                remaining.remove(target)
                if self.preserve_traversed_route:
                    local_passage.update(route.reserved_cells)
                    local_passage.update(
                        cell for x, y, z in route.stands
                        for cell in ((x, y, z), (x, y + 1, z)))
                    local_supports.update(route.support_cells)
                    local_supports.update(
                        (x, y - 1, z) for x, y, z in route.stands)
                    remaining.difference_update(
                        local_passage | local_supports)
                current = stance
                accepted = True
                break
            if not accepted:
                break

        if len(targets) < selected_count:
            if not failures:
                failures.append(PlanRejected(
                    RejectReason.UNREACHABLE_TARGET,
                    start=current, goal=cluster.anchor,
                    detail="cluster contains too few safe ordered targets"))
            return tuple(failures)

        return _PlannedCluster(
            plan=ClusterPlan(
                anchor=cluster.anchor,
                cluster_size=cluster.size,
                selected_count=len(targets),
                block_ids=cluster.block_ids,
                targets=tuple(targets),
                source=source,
            ),
            current_start=current,
            opened=frozenset(local_opened),
            removed=frozenset(local_removed),
            traversed_passage=frozenset(local_passage),
            traversed_supports=frozenset(local_supports),
        )

    def _attempt_resource_clusters(
        self,
        item: ResourceRequirement,
        clusters: Iterable[ResourceCluster],
        *,
        source: str,
    ) -> tuple[tuple[ClusterPlan, ...], int, list[dict[str, object]]]:
        """Try one immutable target set, leaving rejection to the caller."""

        rng = self._rng(f"resource:{item.name}")
        ranked = sorted(
            clusters,
            key=lambda cluster: (
                round(_cluster_distance(cluster, self._current_start), 12),
                rng.random(), cluster.anchor),
        )
        chosen: list[ClusterPlan] = []
        failures: list[dict[str, object]] = []
        selected = 0
        for cluster in ranked:
            needed = item.minimum - selected
            if needed <= 0:
                break
            unavailable = self._protected | frozenset(
                self._traversed_passage | self._traversed_supports)
            if unavailable:
                usable = tuple(cell for cell in cluster.cells
                               if cell not in unavailable)
                if not usable:
                    continue
                cluster = ResourceCluster(
                    cluster.block_ids, cluster.size, cluster.bounds,
                    cluster.anchor, usable, cluster.cells_complete)
            # ``cluster.size`` describes the natural source cluster, whereas
            # ``cluster.cells`` is the actual sampled/usable target set.  A
            # protected cell must not inflate the requested take: asking the
            # cluster planner for more cells than remain makes it discard the
            # usable prefix and can incorrectly reject a world whose later
            # clusters satisfy the balance.
            take = min(len(cluster.cells), needed)
            planned = self._cluster_plan(
                cluster, take, rng,
                start=self._current_start,
                opened=frozenset(self._opened),
                removed=frozenset(self._removed),
                source=source,
            )
            if isinstance(planned, _PlannedCluster):
                chosen.append(planned.plan)
                selected += planned.plan.selected_count
                self._current_start = planned.current_start
                self._opened = set(planned.opened)
                self._removed = set(planned.removed)
                self._traversed_passage = set(
                    planned.traversed_passage)
                self._traversed_supports = set(
                    planned.traversed_supports)
            else:
                failures.extend(error.as_dict() for error in planned[:3])
        return tuple(chosen), selected, failures

    def _resource(self, item: ResourceRequirement) -> ResourcePlan:
        self._active_resource = item.name
        self._active_routes = item.routes or self.goal.routes
        self._active_sealable_fluids = item.sealable_fluid_ids
        self._active_max_drop_fall = item.max_drop_fall
        sample_limit = max(self.cluster_sample_limit, item.minimum)
        required_drops = frozenset(item.required_drop_ids)

        # A required drop normally needs a coordinate-by-coordinate check:
        # gravel's deterministic flint rule is deliberately position based.
        # Explicit metadata makes ordinary state-only drops safe to index,
        # however. Metadata filters can distinguish plain stone from its
        # variants without turning an abundant query into a full component
        # flood.
        indexed_drop_filter = bool(
            required_drops
            and item.metadata is not None
            and 13 not in item.block_ids
            and all(
                natural_harvest_drop(
                    block_id, (0, 0, 0), block_meta=metadata,
                ) in required_drops
                for block_id in item.block_ids
                for metadata in item.metadata
            )
        )

        # A giant stone mass can occupy most of a radius-128 snapshot.  The
        # public component query must flood every one of those cells to report
        # exact cluster boundaries even though this planner needs only 27
        # certified targets.  Use the one-scan population query first when its
        # bounded sample is incomplete.  If that optimization cannot produce
        # a complete safe plan, restore all planner state and fall back to the
        # exact component path, preserving the planner's acceptance criteria.
        pool = None
        initial_state = (
            self._current_start, frozenset(self._opened),
            frozenset(self._removed), self._route_searches,
            frozenset(self._traversed_passage),
            frozenset(self._traversed_supports),
        )
        if not required_drops or indexed_drop_filter:
            pool = self.world.resource_target_pool(
                item.block_ids,
                sample_limit=sample_limit,
                sample_origin=self._current_start,
                prefer_exposed=item.prefer_exposed,
                metadata=item.metadata,
            )
            if pool.available_count < item.minimum:
                self._reject(
                    f"world has {pool.available_count}/{item.minimum} "
                    f"{item.name}",
                    code="world_plan_resource_missing",
                    stage="resources", resource=item.name,
                    details={
                        "required": item.minimum,
                        "available": pool.available_count,
                        "block_ids": item.block_ids,
                    },
                )
            if not pool.cells_complete:
                candidate_pool = ResourceCluster(
                    pool.block_ids, pool.available_count, pool.bounds,
                    pool.anchor, pool.cells, False,
                )
                try:
                    (chosen, selected,
                     _failures) = self._attempt_resource_clusters(
                        item, (candidate_pool,),
                        source="initial_snapshot_target_pool",
                    )
                except WorldPlanningRejected as error:
                    if error.code != "world_plan_search_budget":
                        raise
                    chosen, selected = (), 0
                if selected >= item.minimum:
                    natural = (
                        self.goal.natural_obsidian and 49 in item.block_ids)
                    return ResourcePlan(
                        item.name, item.blocks, item.minimum,
                        pool.available_count, selected, chosen,
                        natural=natural,
                    )
                (self._current_start, opened, removed,
                 self._route_searches, passage, supports) = initial_state
                self._opened = set(opened)
                self._removed = set(removed)
                self._traversed_passage = set(passage)
                self._traversed_supports = set(supports)

        clusters = list(self.world.resource_clusters(
            item.block_ids,
            sample_limit=sample_limit,
            sample_origin=self._current_start,
            prefer_exposed=item.prefer_exposed,
            metadata=item.metadata))
        if required_drops or item.metadata is not None:
            filtered = []
            for cluster in clusters:
                cells = []
                for cell in cluster.cells:
                    block = self.world.block(*cell)
                    block_id = int(getattr(block, "block_id", block))
                    block_meta = int(getattr(block, "meta", 0))
                    if (item.metadata is not None
                            and block_meta not in item.metadata):
                        continue
                    if (required_drops
                            and natural_harvest_drop(
                                block_id, cell, block_meta=block_meta,
                            ) not in required_drops):
                        continue
                    cells.append(cell)
                cells = tuple(cells)
                if not cells:
                    continue
                bounds = type(cluster.bounds)(
                    min(cell[0] for cell in cells),
                    min(cell[1] for cell in cells),
                    min(cell[2] for cell in cells),
                    max(cell[0] for cell in cells),
                    max(cell[1] for cell in cells),
                    max(cell[2] for cell in cells),
                )
                filtered.append(ResourceCluster(
                    cluster.block_ids, len(cells), bounds, cells[0], cells,
                    True))
            clusters = filtered
        available = sum(cluster.size for cluster in clusters)
        if available < item.minimum:
            self._reject(
                f"world has {available}/{item.minimum} {item.name}",
                code="world_plan_resource_missing",
                stage="resources", resource=item.name,
                details={
                    "required": item.minimum,
                    "available": available,
                    "block_ids": item.block_ids,
                },
            )
        chosen, selected, failures = self._attempt_resource_clusters(
            item, clusters, source="initial_snapshot")
        if selected < item.minimum:
            self._reject(
                f"only {selected}/{item.minimum} {item.name} is reachable",
                code="world_plan_resource_unreachable",
                stage="resources", resource=item.name,
                details={
                    "required": item.minimum,
                    "available": available,
                    "reachable": selected,
                    "spatial_rejections": failures,
                },
            )
        natural = self.goal.natural_obsidian and 49 in item.block_ids
        return ResourcePlan(
            item.name, item.blocks, item.minimum, available,
            selected, chosen, natural=natural)

    def build(self) -> WorldPlan:
        initial_world = _PlanningWorld(
            self.world, opened=self._opened, removed=self._removed,
            placed=self._placed)
        if not is_surface_stand(
                initial_world, self.start,
                hazard_clearance=self.goal.hazard_clearance):
            self._reject(
                "snapshot player pose is not a safe planning start",
                code="world_plan_invalid_start",
                stage="start",
                details={"start": self.start},
            )
        proximity = self._proximity()
        resources = tuple(self._resource(item)
                          for item in self.goal.resources)
        safe_cells = tuple(dict.fromkeys(
            [self.start]
            + [cell for resource in resources
               for cluster in resource.clusters
               for target in cluster.targets
               for cell in target.route.stands]
        ))
        return WorldPlan(
            snapshot_sha256=self.world.snapshot_sha256,
            world_seed=self.world.seed,
            decision_seed=self.decision_seed,
            start=self.start,
            goal=self.goal,
            proximity=proximity,
            resources=resources,
            safe_cells=safe_cells,
        )


def build_world_plan(
    world: SnapshotWorldMap,
    goal: GoalSpec | Mapping,
    *,
    decision_seed: int,
    **kwargs,
) -> WorldPlan:
    """Convenience wrapper for declarative preflight callers."""

    return WorldPlanBuilder(
        world, goal, decision_seed=decision_seed, **kwargs).build()


__all__ = [
    "ClusterPlan", "GoalSpec", "MiningTargetPlan", "ProximityConstraint",
    "ProximityResult", "ResourcePlan", "ResourceRequirement", "RoutePlan",
    "WorldPlan", "WorldPlanBuilder", "WorldPlanningRejected",
    "build_world_plan", "natural_harvest_drop",
]
