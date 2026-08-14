"""Deterministic source-fluid plans for bucket-driven construction.

This module owns source selection and travel, not structure geometry or live
actions.  A plan names exact metadata-zero fluid cells, dry bucket stances,
clear interaction rays, bounded outbound/return routes, and every tunnel cell
that must be opened.  Callers still verify those facts against the live engine
before each irreversible action.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import random
from dataclasses import dataclass
from typing import Iterable, Mapping, TypeAlias

from bedrock_rl.sdg.policy.planning import RoutePlan
from bedrock_rl.sdg.policy.spatial import (
    DEFAULT_BLOCK_RULES,
    MINING_REACH,
    BlockRules,
    PlanRejected,
    RejectReason,
    clear_selection_ray,
    is_excavation_stand,
    is_surface_stand,
    plan_staircase_tunnel,
    plan_surface_path,
    voxel_ray,
)
from bedrock_rl.sdg.policy.world import (
    BlockCell,
    SnapshotWorldMap,
)


Cell: TypeAlias = tuple[int, int, int]
Point: TypeAlias = tuple[float, float, float]

_PROTECTED_SOLID = "__netherite_protected_solid__"
_PROTECTED_VOID = "__netherite_protected_void__"


class _ProtectedRules:
    """Delegate block rules while preserving protected support topology."""

    def __init__(self, rules: BlockRules):
        self.rules = rules

    def name(self, value):
        return self.rules.name(value)

    def is_passable(self, value):
        if value in (_PROTECTED_SOLID, _PROTECTED_VOID):
            return False
        return self.rules.is_passable(value)

    def is_fluid(self, value):
        if value in (_PROTECTED_SOLID, _PROTECTED_VOID):
            return False
        return self.rules.is_fluid(value)

    def is_hazard(self, value):
        if value in (_PROTECTED_SOLID, _PROTECTED_VOID):
            return False
        return self.rules.is_hazard(value)

    def is_unbreakable(self, value):
        if value in (_PROTECTED_SOLID, _PROTECTED_VOID):
            return False
        return self.rules.is_unbreakable(value)

    def is_solid(self, value):
        if value == _PROTECTED_SOLID:
            return True
        if value == _PROTECTED_VOID:
            return False
        return self.rules.is_solid(value)

    def is_excavatable(self, value):
        if value in (_PROTECTED_SOLID, _PROTECTED_VOID):
            return False
        return self.rules.is_excavatable(value)


def _cell(value: Iterable[int], *, label: str) -> Cell:
    values = tuple(value)
    if len(values) != 3 or any(
            not isinstance(item, int) or isinstance(item, bool)
            for item in values):
        raise TypeError(f"{label} must contain exactly three integers")
    return values


def _block(world, cell: Cell):
    try:
        return world.block(*cell)
    except TypeError:
        return world.block(cell)


def _in_bounds(world, cell: Cell) -> bool:
    try:
        return bool(world.in_bounds(*cell))
    except TypeError:
        return bool(world.in_bounds(cell))


def _cast_prior_air_passage(cast_plan, initial_opened) -> frozenset[Cell]:
    """Return known-air intermediate cast-ray cells, never structure.

    Cast selection runs over the post-resource overlay, so a valid
    interaction ray can sight through a cell that a prior stage removed.
    Such cells are immutable cast geometry *and* real air by the time source
    travel begins.  Admit only that narrow intersection as passage.  Ray
    targets, the mold, its support/backing, and setup blocks stay strict so a
    stale plan cannot turn protected air into imaginary support.
    """

    opened = {_cell(cell, label="opened cell")
              for cell in initial_opened}
    structural = {
        _cell(cell, label="cast structural cell")
        for name in (
            "access_clear_cells", "excavation_cells", "support_cells",
            "backing_cells",
            "temporary_cells",
        )
        for cell in getattr(cast_plan, name, ())
    }
    for step in getattr(cast_plan, "setup_placements", ()):
        structural.add(_cell(step.cell, label="cast setup cell"))
        structural.add(_cell(step.support, label="cast setup support"))
    transit = {
        _cell(cell, label="cast interaction ray cell")
        for interaction in getattr(cast_plan, "interactions", ())
        for cell in tuple(getattr(interaction, "ray_cells", ()))[:-1]
    }
    return frozenset((transit - structural).intersection(opened))


class _FluidPlanningWorld:
    """Sequential snapshot overlay with protected cells made immutable."""

    def __init__(
        self,
        world: SnapshotWorldMap,
        *,
        opened: Iterable[Cell] = (),
        placed: Mapping[Cell, int | str] | None = None,
        protected: Iterable[Cell] = (),
        allowed_passage: Iterable[Cell] = (),
        rules: BlockRules = DEFAULT_BLOCK_RULES,
    ):
        self.world = world
        self.opened = frozenset(_cell(cell, label="opened cell")
                                for cell in opened)
        self.placed = {
            _cell(cell, label="placed cell"): block
            for cell, block in dict(placed or {}).items()
        }
        self.protected = frozenset(_cell(cell, label="protected cell")
                                   for cell in protected)
        self.allowed_passage = frozenset(
            _cell(cell, label="allowed passage cell")
            for cell in allowed_passage)
        self.rules = rules
        overlap = (self.opened.intersection(self.protected)
                   - self.allowed_passage)
        if overlap:
            raise ValueError(
                "opened cells overlap protected cells outside an explicit "
                f"passage: {sorted(overlap)[:4]}")

    def in_bounds(self, *coordinates):
        cell = coordinates[0] if len(coordinates) == 1 else coordinates
        return _in_bounds(self.world, tuple(cell))

    def block(self, *coordinates):
        cell = coordinates[0] if len(coordinates) == 1 else coordinates
        cell = tuple(cell)
        if cell in self.opened:
            block = "air"
        elif cell in self.placed:
            block = self.placed[cell]
        else:
            block = _block(self.world, cell)
        if cell in self.protected and cell not in self.allowed_passage:
            # Preserve real support semantics. Mapping every protected cell to
            # bedrock turns protected air into a fake bridge that passes route
            # preflight but cannot exist in the live engine. Real solids may be
            # made unbreakable; non-solids use water as a conservative
            # non-passable, non-supporting sentinel.
            if self.rules.is_solid(block):
                return _PROTECTED_SOLID
            if self.rules.is_fluid(block) or self.rules.is_hazard(block):
                return block
            return _PROTECTED_VOID
        return block


@dataclass(frozen=True, slots=True)
class FluidSourceStep:
    """One exact source pickup bracketed by proved travel to the mold."""

    order: int
    kind: str
    block_name: str
    block_id: int
    meta: int
    source_cell: Cell
    pickup_stance: Cell
    pickup_ray_cells: tuple[Cell, ...]
    outbound: RoutePlan
    return_route: RoutePlan

    def __post_init__(self):
        if not isinstance(self.order, int) or isinstance(
                self.order, bool) or self.order < 0:
            raise ValueError("fluid source order must be non-negative")
        if self.kind not in ("water", "lava"):
            raise ValueError("fluid source kind must be water or lava")
        if not self.block_name.strip():
            raise ValueError("fluid source block_name cannot be empty")
        if self.block_id < 0:
            raise ValueError("fluid source block_id cannot be negative")
        if self.meta != 0:
            raise ValueError("fluid source metadata must be exactly zero")
        source = _cell(self.source_cell, label="source cell")
        stance = _cell(self.pickup_stance, label="pickup stance")
        ray = tuple(_cell(cell, label="pickup ray cell")
                    for cell in self.pickup_ray_cells)
        if not ray or ray[-1] != source:
            raise ValueError("pickup ray must terminate at the exact source")
        if not self.outbound.stands \
                or self.outbound.stands[-1] != stance:
            raise ValueError("outbound route must end at pickup stance")
        if not self.return_route.stands \
                or self.return_route.stands[0] != stance:
            raise ValueError("return route must start at pickup stance")
        object.__setattr__(self, "source_cell", source)
        object.__setattr__(self, "pickup_stance", stance)
        object.__setattr__(self, "pickup_ray_cells", ray)

    def to_dict(self) -> dict[str, object]:
        return {
            "order": self.order,
            "kind": self.kind,
            "block": {
                "name": self.block_name,
                "id": self.block_id,
                "meta": self.meta,
                "source_only": True,
            },
            "source_cell": list(self.source_cell),
            "pickup_stance": list(self.pickup_stance),
            "pickup_ray_cells": [list(cell)
                                 for cell in self.pickup_ray_cells],
            "outbound": self.outbound.to_dict(),
            "return": self.return_route.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class FluidSourcePlan:
    """One water source and N unique lava sources for a cast plan."""

    snapshot_sha256: str
    world_seed: int
    decision_seed: int
    work_stance: Cell
    steps: tuple[FluidSourceStep, ...]
    protected_cells: tuple[Cell, ...]
    opened_cells: tuple[Cell, ...]
    node_cap: int
    route_search_cap: int
    candidate_limit: int
    max_reach: float
    initial_opened_cells: tuple[Cell, ...] = ()
    newly_opened_cells: tuple[Cell, ...] = ()
    placed_blocks: tuple[tuple[Cell, int | str], ...] = ()
    search_radius: float | None = None
    source_meta: int = 0
    unique_casting_sources: bool = True

    def __post_init__(self):
        work = _cell(self.work_stance, label="work stance")
        protected = tuple(_cell(cell, label="protected cell")
                          for cell in self.protected_cells)
        opened = tuple(_cell(cell, label="opened cell")
                       for cell in self.opened_cells)
        initial_opened = tuple(_cell(cell, label="initial opened cell")
                               for cell in self.initial_opened_cells)
        newly_opened = tuple(_cell(cell, label="newly opened cell")
                             for cell in self.newly_opened_cells)
        placed = tuple(
            (_cell(cell, label="placed cell"), block)
            for cell, block in self.placed_blocks)
        if len(protected) != len(set(protected)):
            raise ValueError("protected_cells cannot contain duplicates")
        if len(opened) != len(set(opened)):
            raise ValueError("opened_cells cannot contain duplicates")
        if len(initial_opened) != len(set(initial_opened)):
            raise ValueError("initial_opened_cells cannot contain duplicates")
        if len(newly_opened) != len(set(newly_opened)):
            raise ValueError("newly_opened_cells cannot contain duplicates")
        if len(placed) != len({cell for cell, _block in placed}):
            raise ValueError("placed_blocks cannot contain duplicate cells")
        # Older direct constructors supplied only opened_cells. Treat those
        # cells as work introduced by this plan, preserving their accounting.
        if opened and not initial_opened and not newly_opened:
            newly_opened = opened
        if set(initial_opened).intersection(newly_opened):
            raise ValueError(
                "initial and newly opened cells must be disjoint")
        if set(opened) != set(initial_opened).union(newly_opened):
            raise ValueError(
                "opened_cells must equal initial plus newly opened cells")
        if set(protected).intersection(newly_opened):
            raise ValueError(
                "newly opened cells cannot overlap protected cells")
        if not self.steps:
            raise ValueError("fluid source plan cannot be empty")
        if tuple(step.order for step in self.steps) != tuple(
                range(len(self.steps))):
            raise ValueError("fluid source orders must be contiguous")
        if self.steps[0].kind != "water" \
                or any(step.kind != "lava" for step in self.steps[1:]):
            raise ValueError("fluid source plan must contain water then lava")
        sources = tuple(step.source_cell for step in self.steps)
        if len(sources) != len(set(sources)):
            raise ValueError("fluid source cells must be unique")
        if not set(sources).issubset(opened):
            raise ValueError("every collected source must be in opened_cells")
        for step in self.steps:
            if step.outbound.stands[0] != work:
                raise ValueError("every source trip must leave the work stance")
            if step.return_route.stands[-1] != work:
                raise ValueError("every source trip must return to work")
        if (self.node_cap < 1 or self.route_search_cap < 1
                or self.candidate_limit < 1 or self.max_reach <= 0):
            raise ValueError("fluid source planner bounds must be positive")
        if self.search_radius is not None and self.search_radius < 0:
            raise ValueError("fluid source search_radius cannot be negative")
        if self.source_meta != 0:
            raise ValueError("fluid source plan supports metadata zero only")
        if not self.unique_casting_sources:
            raise ValueError("fluid source plan requires unique casting sources")
        object.__setattr__(self, "work_stance", work)
        object.__setattr__(self, "protected_cells", protected)
        object.__setattr__(self, "opened_cells", opened)
        object.__setattr__(self, "initial_opened_cells", initial_opened)
        object.__setattr__(self, "newly_opened_cells", newly_opened)
        object.__setattr__(self, "placed_blocks", placed)

    @property
    def placed_cells(self) -> tuple[Cell, ...]:
        """Cells occupied in the prior-stage overlay at planning time."""

        return tuple(cell for cell, _block in self.placed_blocks)

    @property
    def water_source(self) -> FluidSourceStep:
        return self.steps[0]

    @property
    def lava_sources(self) -> tuple[FluidSourceStep, ...]:
        return self.steps[1:]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "bedrock.fluid_source_plan.v1",
            "snapshot_sha256": self.snapshot_sha256,
            "world_seed": self.world_seed,
            "decision_seed": self.decision_seed,
            "work_stance": list(self.work_stance),
            "steps": [step.to_dict() for step in self.steps],
            "water_source": self.water_source.to_dict(),
            "lava_sources": [step.to_dict() for step in self.lava_sources],
            "protected_cells": [list(cell)
                                for cell in self.protected_cells],
            "opened_cells": [list(cell) for cell in self.opened_cells],
            "initial_opened_cells": [list(cell)
                                     for cell in self.initial_opened_cells],
            "newly_opened_cells": [list(cell)
                                   for cell in self.newly_opened_cells],
            "placed_blocks": [
                {"cell": list(cell), "block": block}
                for cell, block in self.placed_blocks
            ],
            "bounds": {
                "node_cap": self.node_cap,
                "route_search_cap": self.route_search_cap,
                "candidate_limit": self.candidate_limit,
                "max_reach": self.max_reach,
                "search_radius": self.search_radius,
                "source_meta": self.source_meta,
                "unique_casting_sources": self.unique_casting_sources,
            },
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)


def _fluid_ray(
    world,
    stance: Cell,
    source: Cell,
    *,
    rules: BlockRules,
    max_reach: float,
) -> tuple[Cell, ...] | None:
    """Return a clear exact-source ray, treating fluid as non-solid."""

    eye = stance[0] + 0.5, stance[1] + 1.62, stance[2] + 0.5
    target = tuple(value + 0.5 for value in source)
    if math.dist(eye, target) > max_reach + 1e-9:
        return None
    # The live controller accepts a small stance-centering tolerance. Prove
    # the use ray across that whole envelope so a mathematically exact
    # corner ray cannot become an adjacent-stone click after a few
    # hundredths of a block of ordinary movement drift.
    if not clear_selection_ray(
            world, eye, target, rules=rules, target=source,
            origin_margin=0.08):
        return None
    ray = voxel_ray(eye, target)
    if ray[-1] != source:
        return None
    for cell in ray[:-1]:
        if not _in_bounds(world, cell):
            return None
        block = _block(world, cell)
        if not (rules.is_passable(block) or rules.is_fluid(block)):
            return None
    return ray


def _route(
    world,
    start: Cell,
    goal: Cell,
    *,
    rules: BlockRules,
    hazard_clearance: int,
    node_cap: int,
    rng: random.Random,
    consume_search,
) -> RoutePlan | None:
    consume_search()
    try:
        return RoutePlan.surface(plan_surface_path(
            world, start, goal, rules=rules,
            hazard_clearance=hazard_clearance,
            node_cap=node_cap,
            decision_rng=random.Random(rng.getrandbits(64))))
    except PlanRejected:
        pass
    consume_search()
    try:
        return RoutePlan.tunnel(plan_staircase_tunnel(
            world, start, goal, rules=rules,
            hazard_clearance=hazard_clearance,
            node_cap=node_cap,
            decision_rng=random.Random(rng.getrandbits(64))))
    except PlanRejected:
        return None


def _route_breaches_fluid(world, mine_cells, *, rules) -> bool:
    """Whether excavation lets static fluid flow into the new passage.

    Fluid can enter a newly opened voxel from the same level or above, but it
    does not climb into a voxel above its surface. Source routes must reach a
    naturally exposed pickup ray instead of relying on snapshot-dry tunnel
    cells that live ticks will flood immediately after mining.
    """

    for cell in mine_cells:
        for dx, dy, dz in (
                (-1, 0, 0), (1, 0, 0), (0, -1, 0),
                (0, 1, 0), (0, 0, -1), (0, 0, 1)):
            neighbor = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
            if (neighbor[1] >= cell[1]
                    and _in_bounds(world, neighbor)
                    and rules.is_fluid(_block(world, neighbor))):
                return True
    return False


def _candidate_stances(
    source: Cell,
    work: Cell,
    *,
    max_reach: float,
    rng: random.Random,
) -> tuple[Cell, ...]:
    radius = math.ceil(max_reach)
    values = []
    for x in range(source[0] - radius, source[0] + radius + 1):
        for y in range(source[1] - radius, source[1] + radius + 1):
            for z in range(source[2] - radius, source[2] + radius + 1):
                stance = x, y, z
                eye = x + 0.5, y + 1.62, z + 0.5
                distance = math.dist(
                    eye, tuple(value + 0.5 for value in source))
                if distance > max_reach + 1e-9:
                    continue
                values.append((
                    math.dist(work, stance), distance,
                    rng.random(), stance))
    values.sort()
    return tuple(item[-1] for item in values)


def plan_fluid_sources(
    world: SnapshotWorldMap,
    cast_plan,
    *,
    decision_seed: int,
    lava_sources: int | None = None,
    water_blocks: int | str | Iterable[int | str] = "water",
    lava_blocks: int | str | Iterable[int | str] = "lava",
    opened: Iterable[Cell] = (),
    removed: Iterable[Cell] = (),
    placed: Mapping[Cell, int | str] | None = None,
    protected: Iterable[Cell] = (),
    allowed_passage: Iterable[Cell] = (),
    search_radius: float | None = None,
    source_meta: int = 0,
    unique_casting_sources: bool = True,
    hazard_clearance: int = 0,
    node_cap: int = 256,
    route_search_cap: int = 256,
    candidate_limit: int = 256,
    max_reach: float = MINING_REACH,
    rules: BlockRules = DEFAULT_BLOCK_RULES,
) -> FluidSourcePlan:
    """Plan one exact water source and N exact, unique lava sources.

    Every trip starts and ends at ``cast_plan.work_stance``. ``opened`` (route
    excavation), ``removed`` (collected targets), and ``placed`` describe the
    final, journal-proven overlay left by prior stages; a placement wins if a
    caller also lists its cell as previously opened/removed. Source planning
    only needs their union as prior air, so the returned
    ``initial_opened_cells`` intentionally combines both classes. Selected
    source cells and tunnel excavations are then committed before planning the
    next trip, so later routes never rely on a collected fluid or an unrecorded
    world mutation. Structure/resource protected cells are excluded from
    sources and made unbreakable to both route planners. ``allowed_passage``
    names protected, already-open player-volume cells which routes may
    traverse but never excavate; it is intended for a cumulative progression
    breadcrumb.
    """

    if not isinstance(world, SnapshotWorldMap):
        raise TypeError("fluid source planning needs an open SnapshotWorldMap")
    if world.closed:
        raise ValueError("fluid source planning needs an open snapshot")
    if lava_sources is None:
        lava_sources = int(getattr(cast_plan, "casts_required"))
    if not isinstance(lava_sources, int) or isinstance(
            lava_sources, bool) or lava_sources < 1:
        raise ValueError("lava_sources must be a positive integer")
    for value, label in (
        (hazard_clearance, "hazard_clearance"),
        (node_cap, "node_cap"),
        (route_search_cap, "route_search_cap"),
        (candidate_limit, "candidate_limit"),
    ):
        if not isinstance(value, int) or isinstance(value, bool) \
                or value < (0 if label == "hazard_clearance" else 1):
            raise ValueError(f"{label} is invalid")
    if max_reach <= 0:
        raise ValueError("max_reach must be positive")
    if search_radius is not None and float(search_radius) < 0:
        raise ValueError("search_radius cannot be negative")
    if not isinstance(source_meta, int) or isinstance(source_meta, bool):
        raise TypeError("source_meta must be an integer")
    if source_meta != 0:
        raise ValueError(
            "bucket collection supports source_meta=0 only")
    if not isinstance(unique_casting_sources, bool):
        raise TypeError("unique_casting_sources must be true or false")
    if not unique_casting_sources:
        raise ValueError(
            "consumable casting fluids require unique_casting_sources=true")
    if int(getattr(cast_plan, "world_seed")) != world.seed \
            or str(getattr(cast_plan, "snapshot_sha256")) \
            != world.snapshot_sha256:
        raise ValueError("cast plan provenance does not match snapshot")

    work = _cell(getattr(cast_plan, "work_stance"), label="work stance")
    planning_rules = _ProtectedRules(rules)
    placed_blocks = {
        _cell(cell, label="placed cell"): block
        for cell, block in dict(placed or {}).items()
    }
    # Cast setup is executed before the first source trip.  In a constructed
    # mold, one of those placements may deliberately restore a cell opened by
    # an earlier resource route.  Source planning therefore has to see the
    # post-setup world, not only the pre-cast progression overlay.  Otherwise
    # the same coordinate is simultaneously "opened" and protected structure,
    # which turns a valid constructed mold into a fatal overlay-contract error.
    for index, step in enumerate(getattr(cast_plan, "setup_placements", ())):
        try:
            cell = _cell(step.cell, label="cast setup cell")
            material = step.material
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"cast setup placement {index} is incomplete") from exc
        if cell in placed_blocks and placed_blocks[cell] != material:
            raise ValueError(
                f"cast setup placement {cell} conflicts with the prior "
                "placed-block overlay")
        placed_blocks[cell] = material
    initial_opened = {
        _cell(cell, label="opened cell") for cell in (*tuple(opened),
                                                        *tuple(removed))
    }
    # This API describes state immediately before source travel, not event
    # chronology. Therefore a prior or cast-setup replacement block is live
    # even if the cell was broken first.
    initial_opened.difference_update(placed_blocks)
    cast_protected = tuple(getattr(cast_plan, "protected_cells"))
    protected_cells = frozenset(
        _cell(cell, label="protected cell")
        for cell in (*cast_protected, *tuple(protected)))
    explicit_passage = frozenset(
        _cell(cell, label="allowed passage cell")
        for cell in allowed_passage)
    if not explicit_passage.issubset(protected_cells):
        raise ValueError("allowed_passage must be a subset of protected")
    passage_stands = tuple(dict.fromkeys((
        *tuple(getattr(cast_plan, "approach_stands", ())),
        work,
        getattr(cast_plan, "catalyst_recovery_stance", work),
        getattr(cast_plan, "result_mining_stance", work),
    )))
    # A prior tunnel may have excavated a protected feet/head voxel. Supports
    # are intentionally excluded: opening one would invalidate the stand.
    allowed_passage_cells = (
        explicit_passage
        | _cast_prior_air_passage(cast_plan, initial_opened)
        | frozenset(
        cell
        for x, y, z in passage_stands
        for cell in ((x, y, z), (x, y + 1, z))))
    # Validate the initial overlay once, before spending source/route search.
    _FluidPlanningWorld(
        world, opened=initial_opened, placed=placed_blocks,
        protected=protected_cells, allowed_passage=allowed_passage_cells,
        rules=rules)
    seed_bytes = hashlib.sha256(
        f"{world.seed}:{decision_seed}:fluid_sources".encode()).digest()
    rng = random.Random(int.from_bytes(seed_bytes[:8], "big"))
    requirements = (
        ("water", "water", water_blocks, 1),
        ("lava", "lava", lava_blocks, lava_sources),
    )
    source_groups: dict[str, tuple[BlockCell, ...]] = {}
    for kind, _name, blocks, count in requirements:
        # Snapshot iteration is x/y/z storage order, not spatial relevance.
        # Scan the vectorized source stream once while retaining only the
        # bounded nearest set; otherwise a far min-x lava lake can crowd a
        # nearby executable source out of ``candidate_limit``.
        cells = tuple(heapq.nsmallest(
            candidate_limit,
            (cell for cell in world.find_source_fluids(blocks)
             if (cell.meta == 0
                 and cell.position not in protected_cells
                 and cell.position not in initial_opened
                 and cell.position not in placed_blocks
                 and (search_radius is None
                      or math.dist(work, cell.position)
                      <= float(search_radius) + 1e-9))),
            key=lambda cell: (
                math.dist(work, cell.position),
                cell.position, cell.block_id),
        ))
        if len(cells) < count:
            raise PlanRejected(
                RejectReason.UNREACHABLE_TARGET,
                start=work,
                examined=len(cells),
                detail=(f"snapshot has {len(cells)}/{count} unprotected "
                        f"metadata-zero {kind} sources"),
            )
        source_groups[kind] = cells

    opened: set[Cell] = set(initial_opened)
    used: set[Cell] = set()
    steps: list[FluidSourceStep] = []
    searches = 0

    def consume_search():
        nonlocal searches
        if searches >= route_search_cap:
            raise PlanRejected(
                RejectReason.NODE_CAP,
                start=work,
                examined=searches,
                detail=("fluid planning exhausted aggregate route-search "
                        f"budget {route_search_cap}"),
            )
        searches += 1

    for kind, block_name, _blocks, count in requirements:
        for _ in range(count):
            candidates = [cell for cell in source_groups[kind]
                          if cell.position not in used]
            tie = {cell.position: rng.random() for cell in candidates}
            candidates.sort(key=lambda cell: (
                math.dist(work, cell.position),
                tie[cell.position], cell.position))
            selected = None
            for source in candidates:
                source_cell = source.position
                base_overlay = _FluidPlanningWorld(
                    world, opened=opened, placed=placed_blocks,
                    protected=protected_cells,
                    allowed_passage=allowed_passage_cells, rules=rules)
                stance_rng = random.Random(rng.getrandbits(64))
                for stance in _candidate_stances(
                        source_cell, work, max_reach=max_reach,
                        rng=stance_rng):
                    if stance in protected_cells:
                        continue
                    if not _in_bounds(base_overlay, stance):
                        continue
                    if not (
                            is_surface_stand(
                                base_overlay, stance, rules=planning_rules,
                                hazard_clearance=hazard_clearance)
                            or is_excavation_stand(
                                base_overlay, stance, rules=planning_rules,
                                hazard_clearance=hazard_clearance)):
                        continue
                    # Reject geometrically blind stances before spending a
                    # route-search call. For a tunnel candidate, only its
                    # own feet/head are hypothetically opened here; the
                    # chosen route must later prove and record those clears.
                    prospective_opened = set(opened)
                    for body in (
                            stance,
                            (stance[0], stance[1] + 1, stance[2])):
                        block = _block(base_overlay, body)
                        if planning_rules.is_excavatable(block):
                            prospective_opened.add(body)
                    prospective = _FluidPlanningWorld(
                        world, opened=prospective_opened,
                        placed=placed_blocks,
                        protected=protected_cells,
                        allowed_passage=allowed_passage_cells, rules=rules)
                    if _fluid_ray(
                            prospective, stance, source_cell,
                            rules=planning_rules,
                            max_reach=max_reach) is None:
                        continue
                    route_rng = random.Random(rng.getrandbits(64))
                    outbound = _route(
                        base_overlay, work, stance, rules=planning_rules,
                        hazard_clearance=hazard_clearance,
                        node_cap=node_cap, rng=route_rng,
                        consume_search=consume_search)
                    if outbound is None:
                        continue
                    if _route_breaches_fluid(
                            base_overlay, outbound.mine_cells,
                            rules=planning_rules):
                        continue
                    after_outbound = set(opened)
                    after_outbound.update(outbound.mine_cells)
                    if (after_outbound - initial_opened).intersection(
                            protected_cells):
                        continue
                    reached = _FluidPlanningWorld(
                        world, opened=after_outbound,
                        placed=placed_blocks,
                        protected=protected_cells,
                        allowed_passage=allowed_passage_cells, rules=rules)
                    ray = _fluid_ray(
                        reached, stance, source_cell, rules=planning_rules,
                        max_reach=max_reach)
                    if ray is None:
                        continue
                    # A natural source can refill immediately from adjacent
                    # fluid. Never certify its old voxel as route air. The
                    # pickup stance and click ray already prove collection
                    # without requiring the player to occupy that cell.
                    after_pickup = set(after_outbound)
                    returned_world = _FluidPlanningWorld(
                        world, opened=after_pickup,
                        placed=placed_blocks,
                        protected=protected_cells,
                        allowed_passage=allowed_passage_cells, rules=rules)
                    return_route = _route(
                        returned_world, stance, work, rules=planning_rules,
                        hazard_clearance=hazard_clearance,
                        node_cap=node_cap, rng=route_rng,
                        consume_search=consume_search)
                    if return_route is None:
                        continue
                    if _route_breaches_fluid(
                            returned_world, return_route.mine_cells,
                            rules=planning_rules):
                        continue
                    after_pickup.update(return_route.mine_cells)
                    if (after_pickup - initial_opened).intersection(
                            protected_cells):
                        continue
                    selected = (
                        source, stance, ray, outbound,
                        return_route, after_pickup)
                    break
                if selected is not None:
                    break
            if selected is None:
                raise PlanRejected(
                    RejectReason.UNREACHABLE_TARGET,
                    start=work,
                    examined=searches,
                    detail=(f"no exact dry bucket route for remaining "
                            f"metadata-zero {kind} source"),
                )
            source, stance, ray, outbound, return_route, committed = selected
            steps.append(FluidSourceStep(
                order=len(steps), kind=kind, block_name=block_name,
                block_id=source.block_id, meta=source.meta,
                source_cell=source.position, pickup_stance=stance,
                pickup_ray_cells=ray, outbound=outbound,
                return_route=return_route))
            used.add(source.position)
            opened = set(committed)

    # Collected source coordinates are mutations/provenance, but are not safe
    # route air because Minecraft may refill them from adjacent fluid.
    source_cells = {step.source_cell for step in steps}
    opened.update(source_cells)

    return FluidSourcePlan(
        snapshot_sha256=world.snapshot_sha256,
        world_seed=world.seed,
        decision_seed=int(decision_seed),
        work_stance=work,
        steps=tuple(steps),
        protected_cells=tuple(sorted(protected_cells)),
        opened_cells=tuple(sorted(opened)),
        node_cap=node_cap,
        route_search_cap=route_search_cap,
        candidate_limit=candidate_limit,
        max_reach=float(max_reach),
        initial_opened_cells=tuple(sorted(initial_opened)),
        newly_opened_cells=tuple(sorted(opened - initial_opened)),
        placed_blocks=tuple(sorted(placed_blocks.items())),
        search_radius=(None if search_radius is None
                       else float(search_radius)),
        source_meta=source_meta,
        unique_casting_sources=unique_casting_sources,
    )
