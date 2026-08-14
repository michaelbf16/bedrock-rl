"""Small, deterministic spatial planners for scripted SDG policies.

The planners depend on two read-only world operations only.  They can be used
with an engine snapshot, a synthetic test map, or a task-specific map without
adapting that object to a framework base class.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Protocol, Sequence, TypeAlias

from bedrock_rl.env.block_rules import BlockRules, DEFAULT_BLOCK_RULES


Cell: TypeAlias = tuple[int, int, int]
Point: TypeAlias = tuple[float, float, float]
BlockValue: TypeAlias = object


class WorldMap(Protocol):
    """Minimum world interface consumed by the planners."""

    def block(self, *coordinates: object) -> BlockValue:
        """Return a block name, id, or state for one cell."""

    def in_bounds(self, *coordinates: object) -> bool:
        """Return whether the map has authoritative data for one cell."""


def _in_bounds(world: WorldMap, cell: Cell) -> bool:
    """Support both ``method(cell)`` and ``method(x, y, z)`` maps."""

    try:
        return bool(world.in_bounds(cell))
    except TypeError:
        return bool(world.in_bounds(*cell))


def _block(world: WorldMap, cell: Cell) -> BlockValue:
    try:
        return world.block(cell)
    except TypeError:
        return world.block(*cell)


class RejectReason(str, Enum):
    OUT_OF_BOUNDS = "out_of_bounds"
    INVALID_START = "invalid_start"
    INVALID_GOAL = "invalid_goal"
    NODE_CAP = "node_cap"
    NO_PATH = "no_path"
    UNSAFE_CELL = "unsafe_cell"
    UNREACHABLE_TARGET = "unreachable_target"
    UNSAFE_REMOVAL = "unsafe_removal"


class PlanRejected(ValueError):
    """A normal, structured refusal to produce an unsafe spatial plan."""

    def __init__(
        self,
        reason: RejectReason,
        *,
        start: Cell | None = None,
        goal: Cell | None = None,
        examined: int = 0,
        detail: str = "",
    ):
        self.reason = reason
        self.start = start
        self.goal = goal
        self.examined = examined
        self.detail = detail
        message = reason.value
        if detail:
            message += f": {detail}"
        super().__init__(message)

    def as_dict(self) -> dict[str, object]:
        return {
            "reason": self.reason.value,
            "start": self.start,
            "goal": self.goal,
            "examined": self.examined,
            "detail": self.detail,
        }


# The pinned engine's reliable block-interaction bound. Planners and the live
# controller share this one value so a preflight-certified ray cannot later be
# refused merely because the executor used a shorter reach.
MINING_REACH = 4.7


def _add(cell: Cell, dx: int, dy: int, dz: int) -> Cell:
    return cell[0] + dx, cell[1] + dy, cell[2] + dz


def _known(world: WorldMap, cell: Cell) -> bool:
    return _in_bounds(world, cell) and _block(world, cell) is not None


def _hazard_clear(
    world: WorldMap,
    cells: Iterable[Cell],
    rules: BlockRules,
    clearance: int,
) -> bool:
    clearance = max(0, int(clearance))
    seen: set[Cell] = set()
    for origin in cells:
        for dx in range(-clearance, clearance + 1):
            for dy in range(-clearance, clearance + 1):
                for dz in range(-clearance, clearance + 1):
                    cell = _add(origin, dx, dy, dz)
                    if cell in seen:
                        continue
                    seen.add(cell)
                    if not _known(world, cell):
                        return False
                    block = _block(world, cell)
                    if rules.is_fluid(block) or rules.is_hazard(block):
                        return False
    return True


def is_surface_stand(
    world: WorldMap,
    feet: Cell,
    *,
    rules: BlockRules = DEFAULT_BLOCK_RULES,
    hazard_clearance: int = 0,
) -> bool:
    """Return whether a player can safely stand at ``feet``.

    A stand has known, passable feet and head cells and known solid support.
    Hazard clearance is measured in whole voxels around the occupied cells.
    """

    head = _add(feet, 0, 1, 0)
    support = _add(feet, 0, -1, 0)
    if not all(_known(world, cell) for cell in (feet, head, support)):
        return False
    if not rules.is_passable(_block(world, feet)):
        return False
    if not rules.is_passable(_block(world, head)):
        return False
    if not rules.is_solid(_block(world, support)):
        return False
    if rules.is_hazard(_block(world, support)):
        return False
    return _hazard_clear(
        world, (feet, head), rules, hazard_clearance)


def _is_surface_step(
    world: WorldMap,
    current: Cell,
    candidate: Cell,
    rules: BlockRules,
    hazard_clearance: int,
) -> bool:
    if not is_surface_stand(
            world, candidate, rules=rules,
            hazard_clearance=hazard_clearance):
        return False
    if candidate[1] == current[1]:
        return True
    # A vertical step needs one transition voxel beyond the ordinary two-high
    # destination: the destination column on descent, or the origin column
    # while the player's head rises during an ascending jump.
    transition = (_add(candidate, 0, 2, 0)
                  if candidate[1] < current[1]
                  else _add(current, 0, 2, 0))
    return (_known(world, transition)
            and rules.is_passable(_block(world, transition))
            and _hazard_clear(
                world, (transition,), rules, hazard_clearance))


def _heuristic(left: Cell, right: Cell) -> int:
    horizontal = abs(left[0] - right[0]) + abs(left[2] - right[2])
    # One cardinal move may also change elevation by one, so 3D Manhattan
    # would overestimate.  This lower bound keeps A* admissible.
    return max(horizontal, abs(left[1] - right[1]))


def _directional_tie(left: Cell, right: Cell) -> int:
    """Prefer simultaneous horizontal progress when vertical distance wins.

    The admissible lower bound intentionally uses ``max`` because one move
    may change both elevation and one horizontal axis. In a deep staircase,
    that gives every horizontal direction the same score and can produce a
    random descending zigzag. This secondary (non-priority) key steers equal
    lower-bound states toward the target's x/z without changing boundedness.
    """

    return sum(abs(left[index] - right[index]) for index in range(3))


def _ordered(
    values: Sequence[Cell], decision_rng: random.Random | None,
) -> list[Cell]:
    result = list(values)
    if decision_rng is not None:
        decision_rng.shuffle(result)
    return result


def _reconstruct(parent: dict[Cell, Cell], goal: Cell) -> tuple[Cell, ...]:
    path = [goal]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    path.reverse()
    return tuple(path)


def _astar(
    *,
    start: Cell,
    goal: Cell,
    neighbors,
    node_cap: int,
    decision_rng: random.Random | None,
    heuristic_weight: float = 1.0,
) -> tuple[Cell, ...]:
    if node_cap < 1:
        raise ValueError("node_cap must be positive")
    if heuristic_weight < 1.0:
        raise ValueError("heuristic_weight must be at least one")
    lower_bound = _heuristic(start, goal)
    if lower_bound >= node_cap:
        raise PlanRejected(
            RejectReason.NODE_CAP,
            start=start,
            goal=goal,
            examined=0,
            detail=(f"shortest possible route needs {lower_bound} steps, "
                    f"outside the {node_cap}-node bound"),
        )
    queue: list[tuple[float, float, float, int, Cell]] = []
    sequence = 0
    initial_tie = decision_rng.random() if decision_rng is not None else 0.0
    heapq.heappush(
        queue, (lower_bound * heuristic_weight,
                _directional_tie(start, goal),
                initial_tie, sequence, start))
    costs = {start: 0}
    parent: dict[Cell, Cell] = {}
    examined = 0

    while queue:
        _, _, _, _, current = heapq.heappop(queue)
        if current == goal:
            return _reconstruct(parent, goal)
        examined += 1
        if examined >= node_cap:
            raise PlanRejected(
                RejectReason.NODE_CAP,
                start=start,
                goal=goal,
                examined=examined,
                detail=f"search reached its {node_cap}-node bound",
            )
        next_cost = costs[current] + 1
        for candidate in neighbors(current):
            if next_cost >= costs.get(candidate, math.inf):
                continue
            costs[candidate] = next_cost
            parent[candidate] = current
            sequence += 1
            tie = (decision_rng.random()
                   if decision_rng is not None else 0.0)
            heuristic = _heuristic(candidate, goal)
            priority = next_cost + heuristic_weight * heuristic
            heapq.heappush(
                queue, (priority, _directional_tie(candidate, goal), tie,
                        sequence, candidate))

    raise PlanRejected(
        RejectReason.NO_PATH,
        start=start,
        goal=goal,
        examined=examined,
        detail="bounded known space contains no safe route",
    )


_CARDINAL = ((1, 0), (0, 1), (-1, 0), (0, -1))


def plan_surface_path(
    world: WorldMap,
    start: Cell,
    goal: Cell,
    *,
    rules: BlockRules = DEFAULT_BLOCK_RULES,
    hazard_clearance: int = 0,
    node_cap: int = 10_000,
    decision_rng: random.Random | None = None,
    forbidden_cells: Iterable[Cell] = (),
) -> tuple[Cell, ...]:
    """Plan a safe 2.5D walking path using cardinal one-block steps.

    ``forbidden_cells`` applies to the complete player column, including its
    supporting block.  Teaching the search about immutable structure and
    breadcrumb cells is both stricter and substantially cheaper than finding
    a path through them and rejecting that path afterward.

    Stand and transition proofs are memoized for this immutable planning
    call.  A* commonly considers the same voxel from several parents; the
    cache avoids repeating the corresponding block and hazard-neighborhood
    reads without persisting assumptions across world mutations.
    """

    forbidden = frozenset(tuple(cell) for cell in forbidden_cells)
    stand_cache: dict[Cell, bool] = {}
    transition_cache: dict[Cell, bool] = {}

    def route_cell_allowed(feet: Cell) -> bool:
        x, y, z = feet
        return forbidden.isdisjoint(
            ((x, y - 1, z), feet, (x, y + 1, z)))

    def safe_stand(feet: Cell) -> bool:
        if feet not in stand_cache:
            stand_cache[feet] = route_cell_allowed(feet) and is_surface_stand(
                world, feet, rules=rules,
                hazard_clearance=hazard_clearance)
        return stand_cache[feet]

    def safe_step(current: Cell, candidate: Cell) -> bool:
        if not safe_stand(candidate):
            return False
        if candidate[1] == current[1]:
            return True
        transition = (_add(candidate, 0, 2, 0)
                      if candidate[1] < current[1]
                      else _add(current, 0, 2, 0))
        if transition not in transition_cache:
            transition_cache[transition] = (
                transition not in forbidden
                and _known(world, transition)
                and rules.is_passable(_block(world, transition))
                and _hazard_clear(
                    world, (transition,), rules, hazard_clearance))
        return transition_cache[transition]

    for cell, reason, label in (
        (start, RejectReason.INVALID_START, "start"),
        (goal, RejectReason.INVALID_GOAL, "goal"),
    ):
        if not _in_bounds(world, cell):
            raise PlanRejected(
                RejectReason.OUT_OF_BOUNDS,
                start=start,
                goal=goal,
                detail=f"{label} is outside known bounds",
            )
        if not safe_stand(cell):
            raise PlanRejected(
                reason, start=start, goal=goal,
                detail=f"{label} is not a safe two-block stand",
            )

    def neighbors(current: Cell) -> Iterable[Cell]:
        directions = _ordered(
            tuple((dx, 0, dz) for dx, dz in _CARDINAL), decision_rng)
        for dx, _, dz in directions:
            # Flat travel comes first. Step-up/down are bounded to one block.
            for dy in (0, 1, -1):
                candidate = _add(current, dx, dy, dz)
                if safe_step(current, candidate):
                    yield candidate

    return _astar(
        start=start,
        goal=goal,
        neighbors=neighbors,
        node_cap=node_cap,
        decision_rng=decision_rng,
    )


@dataclass(frozen=True)
class ExcavationPlan:
    """A staircase/tunnel route and the cells its executor must preserve."""

    stands: tuple[Cell, ...]
    reserved_cells: tuple[Cell, ...]
    support_cells: tuple[Cell, ...]
    mine_cells: tuple[Cell, ...]


def _is_excavation_stand(
    world: WorldMap,
    feet: Cell,
    rules: BlockRules,
    hazard_clearance: int,
) -> bool:
    head = _add(feet, 0, 1, 0)
    support = _add(feet, 0, -1, 0)
    if not all(_known(world, cell) for cell in (feet, head, support)):
        return False
    for cell in (feet, head):
        block = _block(world, cell)
        if not (rules.is_passable(block) or rules.is_excavatable(block)):
            return False
    support_block = _block(world, support)
    if not rules.is_solid(support_block):
        return False
    if rules.is_hazard(support_block) or rules.is_unbreakable(support_block):
        return False
    return _hazard_clear(
        world, (feet, head), rules, hazard_clearance)


def _transition_headroom(left: Cell, right: Cell) -> Cell | None:
    """Return the extra voxel crossed by a one-block vertical step."""

    if right[1] < left[1]:
        return _add(right, 0, 2, 0)
    if right[1] > left[1]:
        return _add(left, 0, 2, 0)
    return None


def _is_excavation_transition(
    world: WorldMap,
    left: Cell,
    right: Cell,
    rules: BlockRules,
    hazard_clearance: int,
) -> bool:
    """Prove the extra head sweep for an excavation route edge.

    Stand validity covers feet and ordinary head cells.  A vertical edge
    additionally sweeps the voxel two blocks above its lower stance.  That
    voxel must participate in A* edge validity; discovering an unbreakable
    block only while materializing the finished plan would reject a world
    even when another safe staircase exists.
    """

    headroom = _transition_headroom(left, right)
    if headroom is None:
        return True
    if not _known(world, headroom):
        return False
    block = _block(world, headroom)
    if not (rules.is_passable(block) or rules.is_excavatable(block)):
        return False
    return _hazard_clear(world, (headroom,), rules, hazard_clearance)


def is_excavation_stand(
    world: WorldMap,
    feet: Cell,
    *,
    rules: BlockRules = DEFAULT_BLOCK_RULES,
    hazard_clearance: int = 1,
) -> bool:
    """Return whether ``feet`` is a valid conservative tunnel stance.

    This inexpensive predicate lets higher-level planners discard impossible
    goals before spending one of their bounded A* route searches.
    """

    return _is_excavation_stand(world, feet, rules, hazard_clearance)


def _cardinal_segment(left: Cell, right: Cell, axis: int) -> tuple[Cell, ...]:
    """Horizontal cells after ``left`` through ``right`` on one axis."""

    if left[1] != right[1]:
        raise ValueError("cardinal segment endpoints must share y")
    other = 2 if axis == 0 else 0
    if left[other] != right[other]:
        raise ValueError("cardinal segment may change only one axis")
    delta = right[axis] - left[axis]
    step = 1 if delta > 0 else -1
    out = []
    for amount in range(1, abs(delta) + 1):
        cell = list(left)
        cell[axis] += amount * step
        out.append(tuple(cell))
    return tuple(out)


def _staircase_fast_paths(start: Cell, goal: Cell) -> tuple[tuple[Cell, ...], ...]:
    """Generate non-self-crossing horizontal runs long enough for the slope."""

    flat_start = start[0], 0, start[2]
    flat_goal = goal[0], 0, goal[2]
    horizontal = abs(goal[0] - start[0]) + abs(goal[2] - start[2])
    vertical = abs(goal[1] - start[1])
    candidates = []

    def add_flat(waypoints):
        cells = [flat_start]
        for waypoint in waypoints:
            left = cells[-1]
            if left == waypoint:
                continue
            axes = [axis for axis in (0, 2)
                    if left[axis] != waypoint[axis]]
            if len(axes) != 1:
                return
            cells.extend(_cardinal_segment(left, waypoint, axes[0]))
        if cells[-1] != flat_goal or len(cells) - 1 < vertical:
            return
        columns = [(cell[0], cell[2]) for cell in cells]
        if len(columns) != len(set(columns)):
            return
        steps = len(cells) - 1
        stands = tuple((
            cell[0],
            start[1] + round(index * (goal[1] - start[1]) / steps),
            cell[2],
        ) for index, cell in enumerate(cells))
        if stands not in candidates:
            candidates.append(stands)

    if horizontal >= vertical:
        corner_x = goal[0], 0, start[2]
        corner_z = start[0], 0, goal[2]
        if corner_x != flat_start:
            add_flat((corner_x, flat_goal))
        if corner_z != flat_start:
            add_flat((corner_z, flat_goal))
        if flat_start == flat_goal:
            horizontal = 0

    # When vertical change exceeds direct horizontal distance, extend a
    # U-shaped ramp away from the goal and return on a parallel run. Extra
    # detour choices give fluids/bedrock several bounded alternatives before
    # the general A* fallback.
    required = max(1, math.ceil(max(0, vertical - horizontal) / 2))
    for axis in (0, 2):
        other = 2 if axis == 0 else 0
        goal_delta = goal[axis] - start[axis]
        preferred = -1 if goal_delta >= 0 else 1
        for sign in (preferred, -preferred):
            for distance in range(required, required + 7):
                offset = list(flat_start)
                offset[axis] += sign * distance
                cross = list(offset)
                cross[other] = flat_goal[other]
                add_flat((tuple(offset), tuple(cross), flat_goal))
    return tuple(candidates)


def _excavation_plan(world, stands, rules):
    transition_headroom = [
        cell for left, right in zip(stands, stands[1:])
        if (cell := _transition_headroom(left, right)) is not None
    ]
    body = tuple(
        cell for index, feet in enumerate(stands)
        for cell in (
            feet,
            _add(feet, 0, 1, 0),
        ))
    reserved = tuple(dict.fromkeys(body + tuple(transition_headroom)))
    supports = tuple(dict.fromkeys(
        _add(feet, 0, -1, 0) for feet in stands))
    overlap = set(reserved).intersection(supports)
    if overlap:
        raise PlanRejected(
            RejectReason.UNSAFE_CELL,
            start=stands[0],
            goal=stands[-1],
            detail="route would mine a support cell reserved by another step",
        )
    unsafe = tuple(
        cell for cell in reserved
        if not (rules.is_passable(_block(world, cell))
                or rules.is_excavatable(_block(world, cell))))
    if unsafe:
        raise PlanRejected(
            RejectReason.UNSAFE_CELL,
            start=stands[0],
            goal=stands[-1],
            detail=f"route transition clearance is unsafe at {unsafe[0]}",
        )
    mine = tuple(
        cell for cell in reserved
        if not rules.is_passable(_block(world, cell)))
    return ExcavationPlan(tuple(stands), reserved, supports, mine)


def plan_staircase_tunnel(
    world: WorldMap,
    start: Cell,
    goal: Cell,
    *,
    rules: BlockRules = DEFAULT_BLOCK_RULES,
    hazard_clearance: int = 1,
    node_cap: int = 20_000,
    decision_rng: random.Random | None = None,
) -> ExcavationPlan:
    """Plan a conservative cardinal tunnel with at most one level per step.

    Feet and head voxels may be air or mineable solid blocks.  Each stance
    keeps a solid, non-bedrock support block.  Fluids, unbreakable blocks,
    hazards, unknown cells, and hazard-adjacent cells are never excavated.
    """

    for cell, reason, label in (
        (start, RejectReason.INVALID_START, "start"),
        (goal, RejectReason.INVALID_GOAL, "goal"),
    ):
        if not _in_bounds(world, cell):
            raise PlanRejected(
                RejectReason.OUT_OF_BOUNDS,
                start=start,
                goal=goal,
                detail=f"{label} is outside known bounds",
            )
        if not _is_excavation_stand(
                world, cell, rules, hazard_clearance):
            raise PlanRejected(
                reason, start=start, goal=goal,
                detail=f"{label} cannot be safely excavated",
            )

    fast_paths = list(_staircase_fast_paths(start, goal))
    if decision_rng is not None:
        tie = {path: decision_rng.random() for path in fast_paths}
        fast_paths.sort(key=lambda path: (len(path), tie[path]))
    else:
        fast_paths.sort(key=len)
    for stands in fast_paths:
        if (all(_is_excavation_stand(
                world, cell, rules, hazard_clearance) for cell in stands)
                and all(_is_excavation_transition(
                    world, left, right, rules, hazard_clearance)
                    for left, right in zip(stands, stands[1:]))):
            try:
                return _excavation_plan(world, stands, rules)
            except PlanRejected as error:
                # A geometrically valid U-ramp may cross above/below itself,
                # making an earlier support part of a later body tunnel.
                # Try the remaining deterministic ramps before the bounded
                # A* fallback instead of rejecting the entire world.
                if error.reason is not RejectReason.UNSAFE_CELL:
                    raise

    forbidden_stands: set[Cell] = set()

    def neighbors(current: Cell) -> Iterable[Cell]:
        directions = _ordered(
            tuple((dx, dy, dz) for dx, dz in _CARDINAL
                  for dy in (0, 1, -1)),
            decision_rng,
        )
        for dx, dy, dz in directions:
            candidate = _add(current, dx, dy, dz)
            if candidate not in forbidden_stands and _is_excavation_stand(
                    world, candidate, rules, hazard_clearance) and (
                    _is_excavation_transition(
                        world, current, candidate, rules,
                        hazard_clearance)):
                yield candidate

    # A position-only A* can revisit a horizontal column at another height.
    # The route is locally walkable but may then use an earlier body voxel as
    # a later support. Retry with one conflicting intermediate stance banned;
    # this preserves the conservative support proof without path-history in
    # every A* node. The retry count is small and deterministic.
    for _ in range(8):
        stands = _astar(
            start=start,
            goal=goal,
            neighbors=neighbors,
            node_cap=node_cap,
            decision_rng=decision_rng,
            # Excavation only needs a safe bounded route, not the
            # mathematically shortest tunnel. A mildly weighted heuristic
            # avoids exploding the search across equivalent 3-D staircases.
            heuristic_weight=1.5,
        )
        try:
            return _excavation_plan(world, stands, rules)
        except PlanRejected as error:
            if error.reason is not RejectReason.UNSAFE_CELL:
                raise
            reserved = {
                cell for feet in stands
                for cell in (feet, _add(feet, 0, 1, 0))}
            supports = {_add(feet, 0, -1, 0) for feet in stands}
            overlap = reserved.intersection(supports)
            banned = None
            for cell in sorted(overlap):
                body_owners = [
                    feet for feet in stands[1:-1]
                    if feet == cell or _add(feet, 0, 1, 0) == cell]
                support_owner = _add(cell, 0, 1, 0)
                candidates = body_owners or (
                    [support_owner]
                    if support_owner in stands[1:-1] else [])
                if candidates:
                    banned = candidates[0]
                    break
            if banned is None:
                raise
            forbidden_stands.add(banned)
    raise PlanRejected(
        RejectReason.UNSAFE_CELL,
        start=start,
        goal=goal,
        detail="bounded retries could not avoid a route support conflict",
    )


def voxel_ray(start: Point, end: Point) -> tuple[Cell, ...]:
    """Return voxels crossed by the finite segment using 3D DDA."""

    current = tuple(math.floor(value) for value in start)
    final = tuple(math.floor(value) for value in end)
    cells = [current]
    if current == final:
        return tuple(cells)

    delta = tuple(end[index] - start[index] for index in range(3))
    step = tuple(1 if value > 0 else -1 if value < 0 else 0
                 for value in delta)
    t_delta = tuple(
        abs(1.0 / value) if value else math.inf for value in delta)
    t_max = []
    for index in range(3):
        if not step[index]:
            t_max.append(math.inf)
        elif step[index] > 0:
            boundary = current[index] + 1.0
            t_max.append((boundary - start[index]) / delta[index])
        else:
            boundary = float(current[index])
            t_max.append((boundary - start[index]) / delta[index])

    # A segment cannot cross more than its Manhattan voxel displacement plus
    # the starting voxel.  The small margin only protects float edge cases.
    limit = sum(abs(final[i] - current[i]) for i in range(3)) + 4
    for _ in range(limit):
        threshold = min(t_max)
        next_cell = list(current)
        for axis in range(3):
            if math.isclose(t_max[axis], threshold, abs_tol=1e-12):
                next_cell[axis] += step[axis]
                t_max[axis] += t_delta[axis]
        current = tuple(next_cell)
        cells.append(current)
        if current == final:
            return tuple(cells)
    raise RuntimeError("DDA did not reach its finite endpoint")


def clear_voxel_ray(
    world: WorldMap,
    start: Point,
    end: Point,
    *,
    rules: BlockRules = DEFAULT_BLOCK_RULES,
    target: Cell | None = None,
    origin_margin: float = 0.0,
) -> bool:
    """Return whether every voxel before ``target`` is known and passable.

    ``origin_margin`` proves the ray from a small horizontal envelope around
    ``start``. This matters for executable mining plans: a player accepted
    near a stand center can otherwise see a different foreground voxel than
    the mathematical center ray, especially where the ray crosses a block
    corner. The default remains the exact single ray for general queries.
    """

    margin = float(origin_margin)
    if margin < 0:
        raise ValueError("ray origin_margin cannot be negative")
    if margin:
        return all(clear_voxel_ray(
            world,
            (start[0] + dx, start[1], start[2] + dz),
            end,
            rules=rules,
            target=target,
        ) for dx in (-margin, 0.0, margin)
          for dz in (-margin, 0.0, margin))

    ray = voxel_ray(start, end)
    target = target if target is not None else ray[-1]
    for cell in ray:
        if cell == target:
            return _in_bounds(world, cell)
        if not _known(world, cell):
            return False
        if not rules.is_passable(_block(world, cell)):
            return False
    return ray[-1] == target


def clear_selection_ray(
    world: WorldMap,
    start: Point,
    end: Point,
    *,
    rules: BlockRules = DEFAULT_BLOCK_RULES,
    target: Cell | None = None,
    origin_margin: float = 0.0,
) -> bool:
    """Prove the exact vanilla attack/use ray has no selectable blocker.

    Movement-passable plants are still selectable in Minecraft. Treating
    every passable voxel as ray-transparent lets a planner certify a steep
    mining ray that actually hits tall grass in the player's feet cell. The
    pinned selection implementation passes only air, fire, and ordinary
    fluids, so mirror that narrower rule here.
    """

    margin = float(origin_margin)
    if margin < 0:
        raise ValueError("ray origin_margin cannot be negative")
    if margin:
        return all(clear_selection_ray(
            world,
            (start[0] + dx, start[1], start[2] + dz),
            end, rules=rules, target=target,
        ) for dx in (-margin, 0.0, margin)
          for dz in (-margin, 0.0, margin))
    ray = voxel_ray(start, end)
    target = target if target is not None else ray[-1]
    for cell in ray:
        if cell == target:
            return _in_bounds(world, cell)
        if not _known(world, cell):
            return False
        block = _block(world, cell)
        if not _selection_transparent(block, rules):
            return False
    return ray[-1] == target


def _selection_transparent(block, rules: BlockRules) -> bool:
    """Return whether the pinned engine's selection ray passes a block."""

    block_id = (block if isinstance(block, int) and not isinstance(block, bool)
                else getattr(block, "block_id", None))
    return (block_id in (0, 51)
            or rules.name(block) in ("air", "cave_air", "void_air", "fire")
            or rules.is_fluid(block))


def _clear_player_body_selection_ray(
    world: WorldMap,
    start: Point,
    end: Point,
    feet: Cell,
    *,
    rules: BlockRules,
    target: Cell,
    origin_margin: float,
) -> bool:
    """Reject selectable passable blocks in the player's own column.

    The voxel map does not carry the collision boxes needed to reproduce a
    vanilla selection ray for every partial block.  Treating a whole plant,
    rail, or torch voxel as opaque is therefore too conservative for distant
    ray cells.  The important executable edge case is narrower: a steep
    attack/use ray can select tall grass (or another selectable passable
    block) occupying the player's feet/head cell instead of the planned
    target.  Requiring the two body voxels to be selection-transparent is a
    small, executable stance invariant and avoids pretending that voxel DDA
    alone models a partial block's selection box.
    """

    del start, end, target, origin_margin
    body = (tuple(feet), (feet[0], feet[1] + 1, feet[2]))
    return all(_selection_transparent(_block(world, cell), rules)
               for cell in body)


def placement_face_point(
    support: Cell,
    destination: Cell,
    *,
    inset: float = 0.01,
    observer: Point | None = None,
) -> Point:
    """Return a point just inside the exact support face to click.

    The point is expressed in ordinary world coordinates (unlike
    ``Environment.rel_angles_to``, whose arguments are block coordinates and
    are centered internally).  Keeping this geometry in the shared spatial
    layer lets both preflight and live control use the same face convention.
    """

    support = tuple(int(value) for value in support)
    destination = tuple(int(value) for value in destination)
    delta = tuple(destination[index] - support[index]
                  for index in range(3))
    if sum(abs(value) for value in delta) != 1:
        raise ValueError("placement destination must be face-adjacent")
    inset = float(inset)
    if not 0.0 < inset < 0.5:
        raise ValueError("placement face inset must be between 0 and 0.5")
    point = [
        support[index] + 0.5 + delta[index] * (0.5 - inset)
        for index in range(3)
    ]
    if observer is not None:
        observer = tuple(float(value) for value in observer)
        if len(observer) != 3:
            raise ValueError("placement observer must have three values")
        # Aim at the visible region of the requested face, biased toward the
        # observer rather than always at its center.  Center-only rays can be
        # hidden by a neighboring block even though a human can plainly
        # click the exposed edge.  Keep a generous margin from cube edges so
        # the pinned engine cannot quantize the click onto an adjacent face.
        edge_inset = max(0.12, inset)
        for axis, direction in enumerate(delta):
            if direction == 0:
                point[axis] = min(
                    support[axis] + 1.0 - edge_inset,
                    max(support[axis] + edge_inset, observer[axis]),
                )
    return tuple(point)


def _placement_entry_face(
    support: Cell,
    observer: Point,
    point: Point,
) -> tuple[int, int, int] | None:
    """Return the unique cube face entered by ``observer -> point``."""

    entries = []
    for axis in range(3):
        low, high = float(support[axis]), float(support[axis] + 1)
        origin, endpoint = float(observer[axis]), float(point[axis])
        if origin < low:
            entries.append(((low - origin) / (endpoint - origin), axis, -1))
        elif origin > high:
            entries.append(((high - origin) / (endpoint - origin), axis, 1))
    if not entries:
        return None
    entries.sort(reverse=True)
    if (len(entries) > 1
            and abs(entries[0][0] - entries[1][0]) <= 1e-9):
        return None
    _distance, axis, direction = entries[0]
    face = [0, 0, 0]
    face[axis] = direction
    return tuple(face)


def clear_placement_ray(
    world: WorldMap,
    stand: Cell,
    destination: Cell,
    support: Cell,
    *,
    rules: BlockRules = DEFAULT_BLOCK_RULES,
    max_reach: float = 5.0,
    origin_margin: float = 0.0,
) -> bool:
    """Prove that a stance can click one exact placement support face.

    Placement is stricter than merely reaching an empty destination: the ray
    must terminate on the adjacent solid support without a foreground voxel
    intercepting it.  This certificate is objective-independent and works
    for floor, wall, and ceiling placements.
    """

    if max_reach <= 0:
        raise ValueError("placement max_reach must be positive")
    stand = tuple(int(value) for value in stand)
    destination = tuple(int(value) for value in destination)
    support = tuple(int(value) for value in support)
    eye = stand[0] + 0.5, stand[1] + 1.62, stand[2] + 0.5
    point = placement_face_point(
        support, destination, observer=eye)
    if not all(_known(world, cell) for cell in (destination, support)):
        return False
    if not rules.is_passable(_block(world, destination)):
        return False
    support_block = _block(world, support)
    if (not rules.is_solid(support_block)
            or rules.is_hazard(support_block)
            or rules.is_fluid(support_block)):
        return False
    if math.dist(eye, point) > float(max_reach) + 1e-9:
        return False
    expected_face = tuple(destination[index] - support[index]
                          for index in range(3))
    if _placement_entry_face(support, eye, point) != expected_face:
        return False
    return (clear_selection_ray(
                world, eye, point, rules=rules, target=support,
                origin_margin=float(origin_margin))
            and _clear_player_body_selection_ray(
                world, eye, point, stand, rules=rules, target=support,
                origin_margin=float(origin_margin)))


@dataclass(frozen=True)
class MiningStance:
    feet: Cell
    eye: Point
    distance: float


@dataclass(frozen=True)
class RemovalProof:
    """Snapshot evidence that one block can be removed conservatively.

    Successful construction proves that the target is excavatable, every
    face-adjacent cell is known and free of fluids/hazards, and the target is
    not support required by the route used to reach it.  It deliberately does
    not try to predict falling blocks, fluid updates beyond the six exposed
    faces, or mutations made outside the supplied snapshot.
    """

    target: Cell
    inspected_cells: tuple[Cell, ...]
    protected_support_cells: tuple[Cell, ...]
    rule: str = "known_faces_no_fluid_or_hazard_v1"
    fluid_seal_cells: tuple[Cell, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "target": list(self.target),
            "inspected_cells": [list(cell) for cell in self.inspected_cells],
            "protected_support_cells": [
                list(cell) for cell in self.protected_support_cells],
            "rule": self.rule,
            "fluid_seal_cells": [list(cell)
                                 for cell in self.fluid_seal_cells],
        }


_FACES = (
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
)


def plan_safe_removal(
    world: WorldMap,
    target: Cell,
    *,
    route_stands: Iterable[Cell] = (),
    protected_support_cells: Iterable[Cell] = (),
    sealable_fluids: Iterable[str | int] = (),
    rules: BlockRules = DEFAULT_BLOCK_RULES,
) -> RemovalProof:
    """Prove that mining ``target`` will not expose a known local hazard.

    ``route_stands`` makes the safety relationship explicit: the block below
    every stand is protected, including the target stance and all approach
    legs.  Callers may add supports needed by later legs through
    ``protected_support_cells``.

    Unknown face neighbors are rejected rather than guessed safe.  This is a
    preflight proof over one snapshot, not a replacement for live hazard
    checks while executing the plan.
    """

    target = tuple(int(value) for value in target)
    if not _known(world, target):
        raise PlanRejected(
            RejectReason.OUT_OF_BOUNDS,
            goal=target,
            detail="removal target is outside authoritative snapshot bounds",
        )
    block = _block(world, target)
    if rules.is_passable(block) or not rules.is_excavatable(block):
        raise PlanRejected(
            RejectReason.UNSAFE_REMOVAL,
            goal=target,
            detail="removal target is not an excavatable solid block",
        )

    supports = set(tuple(int(value) for value in cell)
                   for cell in protected_support_cells)
    supports.update(
        (int(feet[0]), int(feet[1]) - 1, int(feet[2]))
        for feet in route_stands
    )
    ordered_supports = tuple(sorted(supports))
    if target in supports:
        raise PlanRejected(
            RejectReason.UNSAFE_REMOVAL,
            goal=target,
            detail="target supports a required route stand",
        )

    sealable = {
        str(getattr(value, "block_id", value)).lower().split(":")[-1]
        for value in sealable_fluids
    }
    adjacent = tuple(_add(target, *offset) for offset in _FACES)
    seals = []
    for cell in adjacent:
        if not _known(world, cell):
            raise PlanRejected(
                RejectReason.UNSAFE_REMOVAL,
                goal=target,
                detail=f"face neighbor {cell} is outside known snapshot data",
            )
        neighbor = _block(world, cell)
        key = rules.name(neighbor)
        if rules.is_fluid(neighbor) and key in sealable:
            seals.append(cell)
            continue
        if rules.is_fluid(neighbor) or rules.is_hazard(neighbor):
            raise PlanRejected(
                RejectReason.UNSAFE_REMOVAL,
                goal=target,
                detail=f"removal would expose fluid or hazard at {cell}",
            )

    return RemovalProof(
        target=target,
        inspected_cells=(target,) + adjacent,
        protected_support_cells=ordered_supports,
        rule=("known_faces_sealable_fluid_v1" if seals
              else "known_faces_no_fluid_or_hazard_v1"),
        fluid_seal_cells=tuple(seals),
    )


def safe_mining_stances(
    world: WorldMap,
    target: Cell,
    *,
    rules: BlockRules = DEFAULT_BLOCK_RULES,
    max_reach: float = MINING_REACH,
    hazard_clearance: int = 1,
    decision_rng: random.Random | None = None,
) -> tuple[MiningStance, ...]:
    """Find surface stands with a clear eye ray to a mineable target."""

    if max_reach <= 0:
        raise ValueError("max_reach must be positive")
    if not _known(world, target):
        raise PlanRejected(
            RejectReason.OUT_OF_BOUNDS,
            goal=target,
            detail="mining target is outside known bounds",
        )
    target_block = _block(world, target)
    if rules.is_passable(target_block) \
            or not rules.is_excavatable(target_block):
        raise PlanRejected(
            RejectReason.UNSAFE_CELL,
            goal=target,
            detail="mining target is not a safe solid block",
        )

    target_point = (
        target[0] + 0.5, target[1] + 0.5, target[2] + 0.5)
    radius = math.ceil(max_reach)
    candidates: list[tuple[float, float, Cell, Point]] = []
    for x in range(target[0] - radius, target[0] + radius + 1):
        for y in range(target[1] - radius, target[1] + radius + 1):
            for z in range(target[2] - radius, target[2] + radius + 1):
                feet = x, y, z
                if not is_surface_stand(
                        world, feet, rules=rules,
                        hazard_clearance=hazard_clearance):
                    continue
                # Mining the block holding the player is technically in
                # reach, but it is not a safe stance after the block breaks.
                if (x, y - 1, z) == target:
                    continue
                eye = x + 0.5, y + 1.62, z + 0.5
                distance = math.dist(eye, target_point)
                if distance > max_reach + 1e-9:
                    continue
                if not clear_voxel_ray(
                        world, eye, target_point,
                        rules=rules, target=target,
                        origin_margin=0.04):
                    continue
                if not _clear_player_body_selection_ray(
                        world, eye, target_point, feet,
                        rules=rules, target=target,
                        origin_margin=0.04):
                    continue
                variance = (decision_rng.random()
                            if decision_rng is not None else 0.0)
                candidates.append((distance, variance, feet, eye))
    candidates.sort(key=lambda item: (round(item[0], 12), item[1], item[2]))
    return tuple(MiningStance(feet, eye, distance)
                 for distance, _, feet, eye in candidates)


def choose_mining_stance(
    world: WorldMap,
    target: Cell,
    **kwargs,
) -> MiningStance:
    """Return the best safe mining stance or a structured refusal."""

    candidates = safe_mining_stances(world, target, **kwargs)
    if not candidates:
        raise PlanRejected(
            RejectReason.UNREACHABLE_TARGET,
            goal=target,
            detail="no safe stand has a clear ray within reach",
        )
    return candidates[0]
