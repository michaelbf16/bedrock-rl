"""Deterministic plans for player-built structures.

This module plans *where* and *in what order* blocks can be placed.  It does
not own inventory, crafting, camera control, or engine actions.  Executors
must still prove each named support face from the live observation before
placing a block.

The first concrete selector is a minimal ten-obsidian Nether portal.  The
primitives are deliberately structure-shaped rather than portal-policy-
shaped so other builders can emit the same :class:`PlaceStep` contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Iterable, Literal, TypeAlias

from bedrock_rl.sdg.policy.spatial import (
    DEFAULT_BLOCK_RULES,
    BlockRules,
    PlanRejected,
    RejectReason,
    WorldMap,
    clear_placement_ray,
    clear_selection_ray,
    clear_voxel_ray,
    is_surface_stand,
    placement_face_point,
    plan_surface_path,
    voxel_ray,
)


Cell: TypeAlias = tuple[int, int, int]
Axis: TypeAlias = Literal["x", "z"]


_FACE_BY_OFFSET = {
    (-1, 0, 0): "west",
    (1, 0, 0): "east",
    (0, -1, 0): "down",
    (0, 1, 0): "up",
    (0, 0, -1): "north",
    (0, 0, 1): "south",
}
_FACE_IDS = {
    "west": 1,
    "east": 2,
    "down": 3,
    "up": 4,
    "north": 5,
    "south": 6,
}


def _cell(value: Iterable[int], *, label: str) -> Cell:
    values = tuple(value)
    if len(values) != 3 or any(
            not isinstance(item, int) or isinstance(item, bool)
            for item in values):
        raise TypeError(f"{label} must contain exactly three integers")
    return values


def _offset(left: Cell, right: Cell) -> Cell:
    return tuple(left[index] - right[index] for index in range(3))


def _face(cell: Cell, support: Cell) -> str:
    try:
        return _FACE_BY_OFFSET[_offset(cell, support)]
    except KeyError as exc:
        raise ValueError(
            f"placement {cell} is not face-adjacent to support {support}"
        ) from exc


def _in_bounds(world: WorldMap, cell: Cell) -> bool:
    try:
        return bool(world.in_bounds(cell))
    except TypeError:
        return bool(world.in_bounds(*cell))


def _block(world: WorldMap, cell: Cell):
    try:
        return world.block(cell)
    except TypeError:
        return world.block(*cell)


def _known(world: WorldMap, cell: Cell) -> bool:
    return _in_bounds(world, cell) and _block(world, cell) is not None


def _is_air(value, rules: BlockRules) -> bool:
    # Structure placement is intentionally stricter than navigation.  Rails,
    # vines, fire, and water may all be collision-free while still making a
    # placement or portal activation ambiguous.
    return rules.name(value) in {"0", "air", "cave_air", "void_air"}


def _translated(cell: Cell, offset: Cell) -> Cell:
    return tuple(cell[index] + offset[index] for index in range(3))


@dataclass(frozen=True, slots=True)
class PlaceStep:
    """One support-proven block placement in a structure build order."""

    cell: Cell
    support: Cell
    face: str
    order: int
    material: str
    role: str = "structure"
    stance: Cell | None = None
    jump: bool = False

    def __post_init__(self):
        cell = _cell(self.cell, label="cell")
        support = _cell(self.support, label="support")
        object.__setattr__(self, "cell", cell)
        object.__setattr__(self, "support", support)
        if not isinstance(self.order, int) or isinstance(self.order, bool) \
                or self.order < 0:
            raise ValueError("placement order must be a non-negative integer")
        expected = _face(cell, support)
        if self.face != expected:
            raise ValueError(
                f"placement face {self.face!r} does not match {expected!r}")
        if not self.material.strip() or not self.role.strip():
            raise ValueError("placement material and role cannot be empty")
        if self.stance is not None:
            object.__setattr__(
                self, "stance", _cell(self.stance, label="stance"))
        if not isinstance(self.jump, bool):
            raise TypeError("placement jump must be a boolean")

    @classmethod
    def on(
        cls,
        cell: Cell,
        support: Cell,
        *,
        order: int,
        material: str,
        role: str = "structure",
        stance: Cell | None = None,
        jump: bool = False,
    ) -> PlaceStep:
        """Create a step and derive the clicked support face."""

        cell = _cell(cell, label="cell")
        support = _cell(support, label="support")
        return cls(
            cell, support, _face(cell, support), order,
            material, role, stance, jump)

    @property
    def face_id(self) -> int:
        """Netherite journal face id for this cardinal face."""

        return _FACE_IDS[self.face]

    def to_dict(self) -> dict[str, object]:
        return {
            "cell": list(self.cell),
            "support": list(self.support),
            "face": self.face,
            "face_id": self.face_id,
            "order": self.order,
            "material": self.material,
            "role": self.role,
            "stance": None if self.stance is None else list(self.stance),
            "jump": self.jump,
        }


@dataclass(frozen=True, slots=True)
class StructurePlan:
    """Immutable placement plan with world and decision provenance."""

    kind: str
    snapshot_sha256: str
    world_seed: int
    decision_seed: int
    orientation: str
    anchor: Cell
    occupied_cells: tuple[Cell, ...]
    open_cells: tuple[Cell, ...]
    temporary_cells: tuple[Cell, ...]
    placements: tuple[PlaceStep, ...]
    approach_stands: tuple[Cell, ...]
    work_stance: Cell
    interaction_stance: Cell
    interaction_cell: Cell
    interaction_support: Cell
    interaction_face: str
    entry_stance: Cell
    exit_stance: Cell
    hazard_clearance: int
    clear_cells: tuple[Cell, ...] = ()

    def __post_init__(self):
        if not self.kind.strip():
            raise ValueError("structure kind cannot be empty")
        if self.hazard_clearance < 0:
            raise ValueError("hazard clearance cannot be negative")
        if len(self.clear_cells) != len(set(self.clear_cells)):
            raise ValueError("structure clear cells must be unique")
        cells = self.occupied_cells + self.open_cells + self.temporary_cells
        if len(cells) != len(set(cells)):
            raise ValueError("occupied, open, and temporary cells must differ")
        if tuple(step.order for step in self.placements) \
                != tuple(range(len(self.placements))):
            raise ValueError("placement orders must be contiguous from zero")
        planned_cells = tuple(step.cell for step in self.placements)
        if len(planned_cells) != len(set(planned_cells)) \
                or set(planned_cells) != set(
                    self.temporary_cells + self.occupied_cells):
            raise ValueError(
                "placements must name every temporary and occupied cell once")
        if self.interaction_face != _face(
                self.interaction_cell, self.interaction_support):
            raise ValueError("interaction face does not match its support")

    @property
    def frame_cells(self) -> tuple[Cell, ...]:
        """Alias used by portal executors and artifact validators."""

        return self.occupied_cells

    @property
    def ignition_stance(self) -> Cell:
        return self.interaction_stance

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "bedrock.structure_plan.v1",
            "kind": self.kind,
            "snapshot_sha256": self.snapshot_sha256,
            "world_seed": self.world_seed,
            "decision_seed": self.decision_seed,
            "orientation": self.orientation,
            "anchor": list(self.anchor),
            "occupied_cells": [list(cell) for cell in self.occupied_cells],
            "open_cells": [list(cell) for cell in self.open_cells],
            "temporary_cells": [list(cell)
                                for cell in self.temporary_cells],
            "placements": [step.to_dict() for step in self.placements],
            "approach_stands": [list(cell)
                                for cell in self.approach_stands],
            "work_stance": list(self.work_stance),
            "interaction": {
                "stance": list(self.interaction_stance),
                "cell": list(self.interaction_cell),
                "support": list(self.interaction_support),
                "face": self.interaction_face,
                "face_id": _FACE_IDS[self.interaction_face],
            },
            "entry_stance": list(self.entry_stance),
            "exit_stance": list(self.exit_stance),
            "hazard_clearance": self.hazard_clearance,
            "clear_cells": [list(cell) for cell in self.clear_cells],
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)


@dataclass(frozen=True, slots=True)
class InteractionRay:
    """One immutable, preflight-clear player interaction ray."""

    action: str
    stance: Cell
    target: Cell
    ray_cells: tuple[Cell, ...]

    def __post_init__(self):
        if not self.action.strip():
            raise ValueError("interaction action cannot be empty")
        object.__setattr__(self, "stance", _cell(
            self.stance, label="interaction stance"))
        target = _cell(self.target, label="interaction target")
        object.__setattr__(self, "target", target)
        ray = tuple(_cell(cell, label="interaction ray cell")
                    for cell in self.ray_cells)
        if not ray or ray[-1] != target:
            raise ValueError("interaction ray must terminate at its target")
        object.__setattr__(self, "ray_cells", ray)

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "stance": list(self.stance),
            "target": list(self.target),
            "ray_cells": [list(cell) for cell in self.ray_cells],
        }


@dataclass(frozen=True, slots=True)
class FluidCastPlan:
    """A reusable, contained fluid mold for making collectible blocks.

    Source-fluid discovery and routing deliberately live outside this plan.
    The plan starts at a dry work site, clears any explicitly proved access
    cells, excavates two bounded mold cells,
    places a catalyst fluid, places a reactant beside it, recovers the
    catalyst, and mines the dry result. Repeating that sequence reuses the
    same geometry without baking a final structure into fluid behavior.
    """

    kind: str
    snapshot_sha256: str
    world_seed: int
    decision_seed: int
    orientation: str
    anchor: Cell
    casts_required: int
    catalyst_fluid: str
    reactant_fluid: str
    result_block: str
    catalyst_cell: Cell
    result_cell: Cell
    access_clear_cells: tuple[Cell, ...]
    excavation_cells: tuple[Cell, ...]
    support_cells: tuple[Cell, ...]
    backing_cells: tuple[Cell, ...]
    fluid_envelope: tuple[Cell, ...]
    temporary_cells: tuple[Cell, ...]
    # Cells that must contain no fluid when execution finishes.  The default
    # selector leaves these excavated mold cells open; restoration is an
    # explicit executor-owned placement plan, never an implicit promise here.
    cleanup_cells: tuple[Cell, ...]
    setup_placements: tuple[PlaceStep, ...]
    catalyst_placement: PlaceStep
    reactant_placement: PlaceStep
    approach_stands: tuple[Cell, ...]
    work_stance: Cell
    catalyst_recovery_stance: Cell
    result_mining_stance: Cell
    interactions: tuple[InteractionRay, ...]
    hazard_clearance: int

    def __post_init__(self):
        if not self.kind.strip():
            raise ValueError("cast kind cannot be empty")
        if self.casts_required < 1:
            raise ValueError("casts_required must be positive")
        if self.hazard_clearance < 0:
            raise ValueError("hazard clearance cannot be negative")
        for name in ("catalyst_fluid", "reactant_fluid", "result_block"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} cannot be empty")
        cell_fields = (
            "anchor", "catalyst_cell", "result_cell", "work_stance",
            "catalyst_recovery_stance", "result_mining_stance",
        )
        for name in cell_fields:
            object.__setattr__(self, name, _cell(
                getattr(self, name), label=name))
        tuple_fields = (
            "access_clear_cells", "excavation_cells", "support_cells",
            "backing_cells",
            "fluid_envelope", "temporary_cells", "cleanup_cells",
            "approach_stands",
        )
        for name in tuple_fields:
            cells = tuple(_cell(cell, label=name)
                          for cell in getattr(self, name))
            if len(cells) != len(set(cells)):
                raise ValueError(f"{name} cannot contain duplicate cells")
            object.__setattr__(self, name, cells)
        if set(self.excavation_cells) != {
                self.catalyst_cell, self.result_cell}:
            raise ValueError(
                "cast excavation must contain catalyst and result cells")
        if set(self.fluid_envelope) != set(self.excavation_cells):
            raise ValueError(
                "contained cast envelope must equal the two-cell mold")
        if set(self.access_clear_cells).intersection(
                self.excavation_cells):
            raise ValueError(
                "cast access clears cannot overlap the two-cell mold")
        if self.catalyst_placement.cell != self.catalyst_cell:
            raise ValueError("catalyst placement targets the wrong cell")
        if self.reactant_placement.cell != self.result_cell:
            raise ValueError("reactant placement targets the wrong cell")
        if (self.catalyst_placement.order,
                self.reactant_placement.order) != (0, 1):
            raise ValueError("cast fluid placement orders must be 0 then 1")
        if (self.catalyst_placement.material != self.catalyst_fluid
                or self.reactant_placement.material
                != self.reactant_fluid):
            raise ValueError("cast placement materials do not match fluids")
        if (self.catalyst_placement.stance != self.work_stance
                or self.reactant_placement.stance != self.work_stance):
            raise ValueError("cast fluids must be placed from the work stance")
        if (self.catalyst_placement.support not in self.backing_cells
                or self.reactant_placement.support
                not in self.backing_cells):
            raise ValueError("cast fluid placements need declared backing")
        if tuple(step.order for step in self.setup_placements) != tuple(
                range(len(self.setup_placements))):
            raise ValueError("setup placement orders must be contiguous")
        if not self.approach_stands \
                or self.approach_stands[-1] != self.work_stance:
            raise ValueError("cast approach must end at the work stance")
        actions = tuple(item.action for item in self.interactions)
        if actions != (
                "place_catalyst", "place_reactant",
                "recover_catalyst", "mine_result"):
            raise ValueError("cast interactions are incomplete or unordered")
        expected_targets = (
            self.catalyst_placement.support,
            self.reactant_placement.support,
            self.catalyst_cell,
            self.result_cell,
        )
        if tuple(item.target for item in self.interactions) \
                != expected_targets:
            raise ValueError("cast interaction rays target the wrong cells")
        expected_stances = (
            self.work_stance, self.work_stance,
            self.catalyst_recovery_stance,
            self.result_mining_stance,
        )
        if tuple(item.stance for item in self.interactions) \
                != expected_stances:
            raise ValueError("cast interaction rays use the wrong stances")

    @property
    def protected_cells(self) -> tuple[Cell, ...]:
        """All mold, access, interaction, body, head, and support cells."""

        cells = set(
            self.access_clear_cells + self.excavation_cells
            + self.support_cells + self.backing_cells + self.temporary_cells)
        for x, y, z in self.approach_stands:
            cells.update(((x, y - 1, z), (x, y, z), (x, y + 1, z)))
        for stance in (
                self.work_stance, self.catalyst_recovery_stance,
                self.result_mining_stance):
            x, y, z = stance
            cells.update(((x, y - 1, z), (x, y, z), (x, y + 1, z)))
        for step in self.setup_placements:
            cells.update((step.cell, step.support))
            if step.stance is not None:
                x, y, z = step.stance
                cells.update(((x, y - 1, z), (x, y, z), (x, y + 1, z)))
        for interaction in self.interactions:
            cells.update(interaction.ray_cells)
        return tuple(sorted(cells))

    @property
    def final_open_cells(self) -> tuple[Cell, ...]:
        """All explicitly cleared cells left open after cast execution."""

        return self.access_clear_cells + self.cleanup_cells

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "bedrock.fluid_cast_plan.v3",
            "kind": self.kind,
            "snapshot_sha256": self.snapshot_sha256,
            "world_seed": self.world_seed,
            "decision_seed": self.decision_seed,
            "orientation": self.orientation,
            "anchor": list(self.anchor),
            "casts_required": self.casts_required,
            "materials": {
                "catalyst_fluid": self.catalyst_fluid,
                "reactant_fluid": self.reactant_fluid,
                "result_block": self.result_block,
            },
            "catalyst_cell": list(self.catalyst_cell),
            "result_cell": list(self.result_cell),
            "access_clear_cells": [list(cell)
                                   for cell in self.access_clear_cells],
            "excavation_cells": [list(cell)
                                 for cell in self.excavation_cells],
            "support_cells": [list(cell) for cell in self.support_cells],
            "backing_cells": [list(cell) for cell in self.backing_cells],
            "fluid_envelope": [list(cell) for cell in self.fluid_envelope],
            "temporary_cells": [list(cell)
                                for cell in self.temporary_cells],
            "final_state": {
                "default": "open_dry_mold",
                "final_open_cells": [list(cell)
                                     for cell in self.final_open_cells],
                "restoration_placements": [],
            },
            "setup_placements": [step.to_dict()
                                 for step in self.setup_placements],
            "catalyst_placement": self.catalyst_placement.to_dict(),
            "reactant_placement": self.reactant_placement.to_dict(),
            "approach_stands": [list(cell)
                                for cell in self.approach_stands],
            "work_stance": list(self.work_stance),
            "catalyst_recovery_stance": list(
                self.catalyst_recovery_stance),
            "result_mining_stance": list(self.result_mining_stance),
            "interactions": [item.to_dict() for item in self.interactions],
            "protected_cells": [list(cell)
                                for cell in self.protected_cells],
            "hazard_clearance": self.hazard_clearance,
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)


def validate_support_order(
    world: WorldMap,
    placements: Iterable[PlaceStep],
    *,
    rules: BlockRules = DEFAULT_BLOCK_RULES,
) -> None:
    """Prove that every placement clicks existing or earlier support.

    This validates topology, not visibility.  The executor remains
    responsible for live ray and destination checks because earlier player
    actions may have changed the snapshot.
    """

    placed: set[Cell] = set()
    steps = tuple(placements)
    if tuple(step.order for step in steps) != tuple(range(len(steps))):
        raise ValueError("placement orders must be contiguous from zero")
    for step in steps:
        if step.cell in placed:
            raise ValueError(f"placement cell {step.cell} is duplicated")
        if step.support not in placed:
            if not _known(world, step.support):
                raise PlanRejected(
                    RejectReason.OUT_OF_BOUNDS,
                    goal=step.cell,
                    detail=f"placement support {step.support} is unknown",
                )
            value = _block(world, step.support)
            if not rules.is_solid(value) or rules.is_hazard(value):
                raise PlanRejected(
                    RejectReason.UNSAFE_CELL,
                    goal=step.cell,
                    detail=f"placement support {step.support} is not safe",
                )
        placed.add(step.cell)


def _portal_geometry(
    anchor: Cell,
    axis: Axis,
) -> tuple[tuple[Cell, ...], tuple[Cell, ...], tuple[Cell, ...]]:
    """Return minimal frame, open interior, and scaffold cells.

    ``anchor`` is the lower-left omitted corner as viewed from either side.
    The permanent frame is four blocks wide, five high, and contains exactly
    ten obsidian blocks.  Three non-frame scaffold cells make that minimal
    frame placeable without assuming a pre-built wall or floating blocks.
    """

    if axis not in ("x", "z"):
        raise ValueError("portal axis must be 'x' or 'z'")
    horizontal = (1, 0, 0) if axis == "x" else (0, 0, 1)
    up = (0, 1, 0)

    def at(horizontal_offset: int, vertical_offset: int) -> Cell:
        return _translated(
            anchor,
            tuple(horizontal[index] * horizontal_offset
                  + up[index] * vertical_offset for index in range(3)),
        )

    frame = (
        at(1, 0), at(2, 0),
        at(0, 1), at(3, 1),
        at(0, 2), at(3, 2),
        at(0, 3), at(3, 3),
        at(1, 4), at(2, 4),
    )
    interior = tuple(at(column, level)
                     for level in (1, 2, 3) for column in (1, 2))
    scaffold = (at(0, 0), at(3, 0), at(0, 4), at(3, 4))
    return frame, interior, scaffold


def _portal_view_volume(
    anchor: Cell,
    axis: Axis,
    ignition: Cell,
) -> tuple[Cell, ...]:
    """Bound the empty near-side chamber needed for frame interaction rays."""

    horizontal = (1, 0, 0) if axis == "x" else (0, 0, 1)
    normal = ((0, 0, anchor[2] - ignition[2]) if axis == "x"
              else (anchor[0] - ignition[0], 0, 0))
    return tuple(dict.fromkeys(
        (
            anchor[0] + horizontal[0] * across - normal[0] * depth,
            anchor[1] + height,
            anchor[2] + horizontal[2] * across - normal[2] * depth,
        )
        for depth in range(1, 4)
        for height in range(6)
        for across in range(4)
    ))


def _placement_steps(
    anchor: Cell,
    axis: Axis,
    *,
    stance: Cell,
    ignition_stance: Cell,
) -> tuple[PlaceStep, ...]:
    horizontal = (1, 0, 0) if axis == "x" else (0, 0, 1)

    def at(horizontal_offset: int, vertical_offset: int) -> Cell:
        return (
            anchor[0] + horizontal[0] * horizontal_offset,
            anchor[1] + vertical_offset,
            anchor[2] + horizontal[2] * horizontal_offset,
        )

    outer = (
        2 * stance[0] - ignition_stance[0],
        stance[1],
        2 * stance[2] - ignition_stance[2],
    )
    raised_work = (stance[0], stance[1] + 1, stance[2])
    raised_outer = (outer[0], outer[1] + 2, outer[2])

    raw = [
        # Lower omitted corners are ordinary scaffolds so the first side
        # obsidian never floats.
        (at(0, 0), (at(0, 0)[0], at(0, 0)[1] - 1, at(0, 0)[2]),
         "scaffold", "scaffold"),
        (at(3, 0), (at(3, 0)[0], at(3, 0)[1] - 1, at(3, 0)[2]),
         "scaffold", "scaffold"),
        (at(1, 0), (at(1, 0)[0], at(1, 0)[1] - 1, at(1, 0)[2]),
         "obsidian", "frame"),
        (at(2, 0), (at(2, 0)[0], at(2, 0)[1] - 1, at(2, 0)[2]),
         "obsidian", "frame"),
        (at(0, 1), at(0, 0), "obsidian", "frame"),
        (at(3, 1), at(3, 0), "obsidian", "frame"),
        # Build a three-block stair behind the work position.  It provides
        # an ordinary, safely descendable high stance for the upper half of
        # the frame instead of assuming a player on the floor can click the
        # top of a five-high pillar.
        (outer, (outer[0], outer[1] - 1, outer[2]),
         "scaffold", "scaffold", ignition_stance),
        (stance, (stance[0], stance[1] - 1, stance[2]),
         "scaffold", "scaffold", ignition_stance),
        ((outer[0], outer[1] + 1, outer[2]), outer,
         "scaffold", "scaffold", raised_work),
        (at(0, 2), at(0, 1), "obsidian", "frame"),
        (at(3, 2), at(3, 1), "obsidian", "frame"),
        (at(0, 3), at(0, 2), "obsidian", "frame"),
        (at(3, 3), at(3, 2), "obsidian", "frame"),
        # Both upper corner scaffolds are placed only after the side pillars
        # exist.  Building the top row inward from its visible outer faces
        # prevents the first obsidian from occluding the second support ray.
        (at(0, 4), at(0, 3), "scaffold", "scaffold",
         raised_outer, True),
        (at(3, 4), at(3, 3), "scaffold", "scaffold",
         raised_outer, True),
        (at(1, 4), at(0, 4), "obsidian", "frame"),
        (at(2, 4), at(3, 4), "obsidian", "frame"),
    ]
    steps = []
    for order, values in enumerate(raw):
        cell, support, material, role = values[:4]
        explicit_stance = values[4] if len(values) >= 5 else None
        jump = values[5] if len(values) >= 6 else False
        steps.append(PlaceStep.on(
            cell, support, order=order, material=material, role=role,
            stance=(explicit_stance if explicit_stance is not None
                    else (raised_outer if order >= 9 else stance)),
            jump=jump))
    return tuple(steps)


def _hazard_clear(
    world: WorldMap,
    cells: Iterable[Cell],
    *,
    rules: BlockRules,
    clearance: int,
) -> bool:
    checked: set[Cell] = set()
    for origin in cells:
        for dx in range(-clearance, clearance + 1):
            for dy in range(-clearance, clearance + 1):
                for dz in range(-clearance, clearance + 1):
                    cell = _translated(origin, (dx, dy, dz))
                    if cell in checked:
                        continue
                    checked.add(cell)
                    if not _known(world, cell):
                        return False
                    value = _block(world, cell)
                    if rules.is_fluid(value) or rules.is_hazard(value):
                        return False
    return True


def _reach(stance: Cell, support: Cell) -> float:
    eye = stance[0] + 0.5, stance[1] + 1.62, stance[2] + 0.5
    center = tuple(value + 0.5 for value in support)
    return math.dist(eye, center)


def _candidate_anchor(
    ignition: Cell,
    axis: Axis,
    normal: int,
) -> tuple[Cell, Cell, Cell, Cell, Cell]:
    """Return anchor, work, entry, exit, and portal normal."""

    if axis == "x":
        vector = (0, 0, normal)
        anchor = ignition[0] - 1, ignition[1], ignition[2] + normal
        entry = ignition[0], ignition[1] + 1, ignition[2] + normal
    else:
        vector = (normal, 0, 0)
        anchor = ignition[0] + normal, ignition[1], ignition[2] - 1
        entry = ignition[0] + normal, ignition[1] + 1, ignition[2]
    work = _translated(ignition, tuple(-value for value in vector))
    exit_stance = _translated(ignition, tuple(2 * value for value in vector))
    return anchor, work, entry, exit_stance, vector


def _site_plan(
    world: WorldMap,
    *,
    start: Cell,
    ignition: Cell,
    axis: Axis,
    normal: int,
    decision_seed: int,
    world_seed: int,
    snapshot_sha256: str,
    hazard_clearance: int,
    node_cap: int,
    max_reach: float,
    rules: BlockRules,
    route_rng: random.Random,
) -> StructurePlan | None:
    anchor, work, entry, exit_stance, _vector = _candidate_anchor(
        ignition, axis, normal)
    frame, interior, scaffold = _portal_geometry(anchor, axis)
    view_volume = _portal_view_volume(anchor, axis, ignition)

    # Both construction positions and the far-side exit must already be safe
    # surface stands.  The elevated entry becomes standable once its bottom
    # obsidian support is placed; its feet and head are verified as air here.
    outer = (2 * work[0] - ignition[0], work[1],
             2 * work[2] - ignition[2])
    construction_stair = (
        work, outer, (outer[0], outer[1] + 1, outer[2]))
    for stand in (work, ignition, outer, exit_stance):
        if not is_surface_stand(
                world, stand, rules=rules,
                hazard_clearance=hazard_clearance):
            return None
    if not all(_known(world, cell) and _is_air(_block(world, cell), rules)
               for cell in frame + interior + scaffold
               + construction_stair + view_volume):
        return None
    if entry not in interior or not _is_air(_block(world, entry), rules):
        return None
    entry_head = entry[0], entry[1] + 1, entry[2]
    if entry_head not in interior or not _is_air(
            _block(world, entry_head), rules):
        return None

    targets = (frame + interior + scaffold + construction_stair + view_volume
               + (work, ignition, outer, exit_stance))
    if not _hazard_clear(
            world, targets, rules=rules, clearance=hazard_clearance):
        return None

    placements = _placement_steps(
        anchor, axis, stance=work, ignition_stance=ignition)
    try:
        validate_support_order(world, placements, rules=rules)
    except PlanRejected:
        return None
    if not _placement_rays_clear(
            world, placements, rules=rules, max_reach=max_reach):
        return None
    if any(_reach(step.stance, step.support) > max_reach + 1e-9
           for step in placements):
        return None
    try:
        approach = plan_surface_path(
            world, start, work, rules=rules,
            hazard_clearance=hazard_clearance,
            node_cap=node_cap, decision_rng=route_rng)
    except PlanRejected:
        return None

    temporary_steps = tuple(step for step in placements
                            if step.role == "scaffold")
    frame_steps = tuple(step for step in placements if step.role == "frame")
    validate_support_order(world, placements, rules=rules)
    occupied = tuple(step.cell for step in frame_steps)
    temporary = tuple(step.cell for step in temporary_steps)
    interaction_support = occupied[0]
    interaction_cell = (
        interaction_support[0], interaction_support[1] + 1,
        interaction_support[2])
    return StructurePlan(
        kind="nether_portal",
        snapshot_sha256=snapshot_sha256,
        world_seed=world_seed,
        decision_seed=decision_seed,
        orientation=f"{axis}:{'positive' if normal > 0 else 'negative'}",
        anchor=anchor,
        occupied_cells=occupied,
        open_cells=interior,
        temporary_cells=temporary,
        placements=placements,
        approach_stands=approach,
        work_stance=work,
        interaction_stance=ignition,
        interaction_cell=interaction_cell,
        interaction_support=interaction_support,
        interaction_face=_face(interaction_cell, interaction_support),
        entry_stance=entry,
        exit_stance=exit_stance,
        hazard_clearance=hazard_clearance,
    )


def select_nether_portal_site(
    world: WorldMap,
    *,
    decision_seed: int,
    start: Cell | None = None,
    search_radius: int = 12,
    vertical_radius: int = 3,
    hazard_clearance: int = 1,
    node_cap: int = 20_000,
    max_reach: float = 5.0,
    rules: BlockRules = DEFAULT_BLOCK_RULES,
) -> StructurePlan:
    """Select and prove a buildable minimal Nether portal site.

    Candidate positions come only from the supplied world and decision seed.
    No world coordinate or seed-specific route is embedded in the selector.
    The resulting frame has ten permanent obsidian placements, a clear 2x3
    interior, support-ordered scaffold placements, a safe approach, a safe
    ignition stance, an elevated supported entry, and a clear far-side exit.
    """

    for value, label in (
        (search_radius, "search_radius"),
        (vertical_radius, "vertical_radius"),
        (hazard_clearance, "hazard_clearance"),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{label} must be a non-negative integer")
    if node_cap < 1:
        raise ValueError("node_cap must be positive")
    if max_reach <= 0:
        raise ValueError("max_reach must be positive")

    if start is None:
        player = getattr(world, "player", None)
        if player is None:
            raise TypeError("start is required when the world has no player")
        start = math.floor(player.x), math.floor(player.y), math.floor(player.z)
    start = _cell(start, label="start")
    if not is_surface_stand(
            world, start, rules=rules,
            hazard_clearance=hazard_clearance):
        raise PlanRejected(
            RejectReason.INVALID_START,
            start=start,
            detail="portal planning start is not a safe two-block stand",
        )

    try:
        world_seed = int(getattr(world, "seed"))
        snapshot_sha256 = str(getattr(world, "snapshot_sha256"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(
            "world must expose seed and snapshot_sha256 provenance") from exc

    seed_bytes = hashlib.sha256(
        f"{world_seed}:{decision_seed}:nether_portal".encode()).digest()
    rng = random.Random(int.from_bytes(seed_bytes[:8], "big"))
    y_values = [start[1]]
    for amount in range(1, vertical_radius + 1):
        y_values.extend((start[1] + amount, start[1] - amount))
    ignitions: list[tuple[int, float, Cell]] = []
    for x in range(start[0] - search_radius, start[0] + search_radius + 1):
        for z in range(start[2] - search_radius, start[2] + search_radius + 1):
            distance = abs(x - start[0]) + abs(z - start[2])
            if distance > search_radius:
                continue
            for y in y_values:
                cell = x, y, z
                if is_surface_stand(
                        world, cell, rules=rules,
                        hazard_clearance=hazard_clearance):
                    ignitions.append((distance, rng.random(), cell))
    ignitions.sort(key=lambda item: (item[0], item[1], item[2]))

    attempts = 0
    orientations = [(axis, normal)
                    for axis in ("x", "z") for normal in (-1, 1)]
    for _distance_from_start, _tie, ignition in ignitions:
        choices = list(orientations)
        rng.shuffle(choices)
        for axis, normal in choices:
            attempts += 1
            route_seed = rng.getrandbits(64)
            plan = _site_plan(
                world,
                start=start,
                ignition=ignition,
                axis=axis,
                normal=normal,
                decision_seed=int(decision_seed),
                world_seed=world_seed,
                snapshot_sha256=snapshot_sha256,
                hazard_clearance=hazard_clearance,
                node_cap=node_cap,
                max_reach=max_reach,
                rules=rules,
                route_rng=random.Random(route_seed),
            )
            if plan is not None:
                return plan

    raise PlanRejected(
        RejectReason.NO_PATH,
        start=start,
        examined=attempts,
        detail=(
            "known search region has no hazard-clear, support-ordered "
            "ten-obsidian portal site with safe build and entry stances"
        ),
    )


class _ExcavatedWorld:
    """Read-only view of a prospective two-cell dry cast mold."""

    def __init__(self, world: WorldMap, excavated: Iterable[Cell]):
        self.world = world
        self.excavated = frozenset(excavated)

    def in_bounds(self, *coordinates):
        cell = coordinates[0] if len(coordinates) == 1 else coordinates
        return _in_bounds(self.world, tuple(cell))

    def block(self, *coordinates):
        cell = coordinates[0] if len(coordinates) == 1 else coordinates
        cell = tuple(cell)
        return "air" if cell in self.excavated else _block(self.world, cell)


class _PreparedWorld(_ExcavatedWorld):
    """Prospective cast site after exact excavation and setup placements."""

    def __init__(self, world, excavated, placed, material):
        super().__init__(world, excavated)
        self.placed = frozenset(placed)
        self.material = material

    def block(self, *coordinates):
        cell = coordinates[0] if len(coordinates) == 1 else coordinates
        cell = tuple(cell)
        if cell in self.placed:
            return self.material
        return super().block(cell)


def _placement_rays_clear(
    world: WorldMap,
    placements: Iterable[PlaceStep],
    *,
    rules: BlockRules,
    max_reach: float,
) -> bool:
    """Certify each support ray against the sequential build overlay."""

    placed: set[Cell] = set()
    for step in placements:
        if step.stance is None:
            return False
        staged = _PreparedWorld(
            world, excavated=(), placed=placed, material="stone")
        stances = (step.stance,)
        if step.jump:
            # A conventional jump-place exposes the top face for the two
            # five-high corner scaffolds.  The irreversible click is still
            # gated on the next live public ray while airborne.
            stances += ((step.stance[0], step.stance[1] + 1,
                         step.stance[2]),)
        if not any(clear_placement_ray(
                staged, candidate, step.cell, step.support,
                rules=rules, max_reach=max_reach, origin_margin=0.04)
                for candidate in stances):
            return False
        placed.add(step.cell)
    return True


def _visible_excavation_order(
    world: WorldMap,
    cells: Iterable[Cell],
    *,
    stance: Cell,
    rules: BlockRules,
    max_reach: float,
) -> tuple[Cell, ...] | None:
    """Order a chamber so every next mine target is publicly selectable."""

    eye = stance[0] + 0.5, stance[1] + 1.62, stance[2] + 0.5
    remaining = set(cells)
    removed: set[Cell] = set()
    ordered = []
    while remaining:
        staged = _ExcavatedWorld(world, removed)
        visible = []
        for cell in remaining:
            center = tuple(value + 0.5 for value in cell)
            distance = math.dist(eye, center)
            if distance > max_reach + 1e-9:
                continue
            if clear_selection_ray(
                    staged, eye, center, rules=rules, target=cell,
                    origin_margin=0.04):
                visible.append((distance, -cell[1], cell))
        if not visible:
            return None
        _distance, _height, chosen = min(visible)
        ordered.append(chosen)
        removed.add(chosen)
        remaining.remove(chosen)
    return tuple(ordered)


def _completion_chamber_plan(
    world: WorldMap,
    *,
    start: Cell,
    ignition: Cell,
    axis: Axis,
    normal: int,
    decision_seed: int,
    world_seed: int,
    snapshot_sha256: str,
    hazard_clearance: int,
    node_cap: int,
    max_reach: float,
    rules: BlockRules,
    route_rng: random.Random,
) -> StructurePlan | None:
    """Prove a portal chamber that may be excavated from known solid rock."""

    anchor, work, entry, exit_stance, _vector = _candidate_anchor(
        ignition, axis, normal)
    frame, interior, scaffold = _portal_geometry(anchor, axis)
    view_volume = _portal_view_volume(anchor, axis, ignition)
    exit_body = (exit_stance,
                 (exit_stance[0], exit_stance[1] + 1, exit_stance[2]))

    # The near side is the already-walkable construction area. Everything
    # beyond it may be air or ordinary mineable terrain; fluids, hazards,
    # bedrock, and unknown snapshot edges reject the candidate.
    outer = (2 * work[0] - ignition[0], work[1],
             2 * work[2] - ignition[2])
    construction_stair = (
        work, outer, (outer[0], outer[1] + 1, outer[2]))
    for stand in (work, ignition, outer):
        if not is_surface_stand(
                world, stand, rules=rules,
                hazard_clearance=hazard_clearance):
            return None
    chamber = tuple(dict.fromkeys(
        (*frame, *interior, *scaffold, *construction_stair,
         *view_volume, *exit_body)))
    if not all(
            _known(world, cell)
            and (_is_air(_block(world, cell), rules)
                 or rules.is_excavatable(_block(world, cell)))
            for cell in chamber):
        return None
    if not _hazard_clear(
            world, (*chamber, work, ignition),
            rules=rules, clearance=hazard_clearance):
        return None

    unordered_clear = tuple(
        cell for cell in chamber
        if not _is_air(_block(world, cell), rules))
    clear_cells = _visible_excavation_order(
        world, unordered_clear, stance=work, rules=rules,
        max_reach=max_reach)
    if clear_cells is None:
        return None
    prepared = _ExcavatedWorld(world, clear_cells)
    if not is_surface_stand(
            prepared, exit_stance, rules=rules,
            hazard_clearance=hazard_clearance):
        return None
    if entry not in interior:
        return None

    placements = _placement_steps(
        anchor, axis, stance=work, ignition_stance=ignition)
    try:
        validate_support_order(prepared, placements, rules=rules)
        approach = plan_surface_path(
            world, start, work, rules=rules,
            hazard_clearance=hazard_clearance,
            node_cap=node_cap, decision_rng=route_rng)
    except PlanRejected:
        return None
    if not _placement_rays_clear(
            prepared, placements, rules=rules, max_reach=max_reach):
        return None
    if any(_reach(step.stance, step.support) > max_reach + 1e-9
           for step in placements):
        return None

    occupied = tuple(step.cell for step in placements
                     if step.role == "frame")
    temporary = tuple(step.cell for step in placements
                      if step.role == "scaffold")
    interaction_support = occupied[0]
    interaction_cell = (interaction_support[0],
                        interaction_support[1] + 1,
                        interaction_support[2])
    return StructurePlan(
        kind="nether_portal_completion_chamber",
        snapshot_sha256=snapshot_sha256,
        world_seed=world_seed,
        decision_seed=decision_seed,
        orientation=f"{axis}:{'positive' if normal > 0 else 'negative'}",
        anchor=anchor,
        occupied_cells=occupied,
        open_cells=interior,
        temporary_cells=temporary,
        placements=placements,
        approach_stands=approach,
        work_stance=work,
        interaction_stance=ignition,
        interaction_cell=interaction_cell,
        interaction_support=interaction_support,
        interaction_face=_face(interaction_cell, interaction_support),
        entry_stance=entry,
        exit_stance=exit_stance,
        hazard_clearance=hazard_clearance,
        clear_cells=clear_cells,
    )


def select_nether_portal_completion_site(
    world: WorldMap,
    *,
    decision_seed: int,
    start: Cell | None = None,
    search_radius: int = 12,
    vertical_radius: int = 3,
    hazard_clearance: int = 1,
    node_cap: int = 20_000,
    max_reach: float = 5.0,
    rules: BlockRules = DEFAULT_BLOCK_RULES,
) -> StructurePlan:
    """Select a bounded nearby portal chamber and its exact excavation.

    Unlike :func:`select_nether_portal_site`, this does not require a naturally
    empty 4x5 build volume or far-side exit. It starts from known walkable
    terrain and proves a finite set of ordinary blocks that the policy can
    mine before placing the same support-ordered ten-obsidian frame.
    """

    if start is None:
        player = getattr(world, "player", None)
        if player is None:
            raise TypeError("start is required when the world has no player")
        start = math.floor(player.x), math.floor(player.y), math.floor(player.z)
    start = _cell(start, label="start")
    if not is_surface_stand(
            world, start, rules=rules,
            hazard_clearance=hazard_clearance):
        raise PlanRejected(
            RejectReason.INVALID_START, start=start,
            detail="portal completion start is not a safe stand")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
           for value in (search_radius, vertical_radius, hazard_clearance)):
        raise ValueError("portal search bounds must be non-negative integers")
    if node_cap < 1 or max_reach <= 0:
        raise ValueError("portal node cap and reach must be positive")

    world_seed = int(getattr(world, "seed"))
    snapshot_sha256 = str(getattr(world, "snapshot_sha256"))
    seed_bytes = hashlib.sha256(
        f"{world_seed}:{decision_seed}:completion_portal".encode()).digest()
    rng = random.Random(int.from_bytes(seed_bytes[:8], "big"))
    y_values = [start[1]]
    for amount in range(1, vertical_radius + 1):
        y_values.extend((start[1] + amount, start[1] - amount))
    candidates = []
    for x in range(start[0] - search_radius, start[0] + search_radius + 1):
        for z in range(start[2] - search_radius, start[2] + search_radius + 1):
            distance = abs(x - start[0]) + abs(z - start[2])
            if distance > search_radius:
                continue
            for y in y_values:
                cell = x, y, z
                if is_surface_stand(
                        world, cell, rules=rules,
                        hazard_clearance=hazard_clearance):
                    candidates.append((distance, rng.random(), cell))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    attempts = 0
    orientations = [(axis, normal)
                    for axis in ("x", "z") for normal in (-1, 1)]
    for _distance, _tie, ignition in candidates:
        choices = list(orientations)
        rng.shuffle(choices)
        for axis, normal in choices:
            attempts += 1
            plan = _completion_chamber_plan(
                world, start=start, ignition=ignition,
                axis=axis, normal=normal,
                decision_seed=decision_seed,
                world_seed=world_seed,
                snapshot_sha256=snapshot_sha256,
                hazard_clearance=hazard_clearance,
                node_cap=node_cap, max_reach=max_reach, rules=rules,
                route_rng=random.Random(rng.getrandbits(64)))
            if plan is not None:
                return plan
    raise PlanRejected(
        RejectReason.NO_PATH, start=start, examined=attempts,
        detail=("known bounded region has no constructible hazard-clear "
                "portal chamber"))


def _eye(stance: Cell) -> tuple[float, float, float]:
    return stance[0] + 0.5, stance[1] + 1.62, stance[2] + 0.5


def _cell_interaction(
    action: str,
    world: WorldMap,
    stance: Cell,
    target: Cell,
    *,
    rules: BlockRules,
) -> InteractionRay | None:
    eye = _eye(stance)
    center = tuple(value + 0.5 for value in target)
    if not clear_voxel_ray(
            world, eye, center, rules=rules, target=target,
            origin_margin=0.04):
        return None
    return InteractionRay(action, stance, target, voxel_ray(eye, center))


def _placement_interaction(
    action: str,
    world: WorldMap,
    step: PlaceStep,
    *,
    rules: BlockRules,
    max_reach: float,
) -> InteractionRay | None:
    if step.stance is None:
        raise ValueError("cast placement needs an explicit stance")
    if not clear_placement_ray(
            world, step.stance, step.cell, step.support,
            rules=rules, max_reach=max_reach, origin_margin=0.04):
        return None
    eye = _eye(step.stance)
    point = placement_face_point(
        step.support, step.cell, observer=eye)
    return InteractionRay(
        action, step.stance, step.support, voxel_ray(eye, point))


def _cast_setup_placements(
    world: WorldMap,
    *,
    excavation: tuple[Cell, ...],
    required_solids: tuple[Cell, ...],
    work: Cell,
    material: str,
    max_reach: float,
    rules: BlockRules,
    protected: frozenset[Cell],
) -> tuple[tuple[PlaceStep, ...], _PreparedWorld] | None:
    """Construct missing dry mold boundary from supported exact placements.

    Existing safe solids are reused. Only known air may be filled, every
    placement clicks a natural or earlier-placed support, and each ray is
    proved against the exact prospective world state after excavation.
    """

    excavated = frozenset(excavation)
    missing = []
    for cell in dict.fromkeys(required_solids):
        if not _known(world, cell):
            return None
        value = _block(world, cell)
        if (rules.is_solid(value) and not rules.is_fluid(value)
                and not rules.is_hazard(value)):
            continue
        if not _is_air(value, rules) or cell in protected:
            return None
        missing.append(cell)
    if not missing:
        return (), _PreparedWorld(world, excavated, (), material)

    placed: list[Cell] = []
    steps: list[PlaceStep] = []
    pending = set(missing)
    directions = (
        (0, -1, 0), (1, 0, 0), (-1, 0, 0),
        (0, 0, 1), (0, 0, -1), (0, 1, 0),
    )
    while pending:
        prepared = _PreparedWorld(world, excavated, placed, material)
        choices = []
        for destination in sorted(
                pending, key=lambda cell: (math.dist(work, cell), cell)):
            supports = []
            for direction in directions:
                support = _translated(destination, direction)
                if support in excavated or not _known(prepared, support):
                    continue
                value = _block(prepared, support)
                if (not rules.is_solid(value) or rules.is_fluid(value)
                        or rules.is_hazard(value)):
                    continue
                step = PlaceStep.on(
                    destination, support, order=len(steps),
                    material=material, role="cast_setup", stance=work)
                if clear_placement_ray(
                        prepared, work, destination, support,
                        rules=rules, max_reach=max_reach,
                        origin_margin=0.04):
                    supports.append((math.dist(work, support), support, step))
            if supports:
                choices.append((
                    math.dist(work, destination), destination,
                    min(supports)[2],
                ))
        if not choices:
            return None
        _distance, destination, step = min(choices)
        steps.append(step)
        placed.append(destination)
        pending.remove(destination)

    prepared = _PreparedWorld(world, excavated, placed, material)
    try:
        validate_support_order(
            _ExcavatedWorld(world, excavated), steps, rules=rules)
    except PlanRejected:
        return None
    return tuple(steps), prepared


def _fluid_cast_site_plan(
    world: WorldMap,
    *,
    start: Cell,
    work: Cell,
    forward: Cell,
    side: int,
    decision_seed: int,
    world_seed: int,
    snapshot_sha256: str,
    casts_required: int,
    catalyst_fluid: str,
    reactant_fluid: str,
    result_block: str,
    setup_material: str,
    hazard_clearance: int,
    node_cap: int,
    max_reach: float,
    rules: BlockRules,
    route_rng: random.Random,
    protected: frozenset[Cell],
    allowed_passage: frozenset[Cell],
    route_cache: dict[Cell, tuple[Cell, ...] | None],
    debug_counts: dict[str, int] | None = None,
    debug_examples: dict[str, list[object]] | None = None,
) -> FluidCastPlan | None:
    """Prove one two-cell, ground-cut mold beside a dry work stand."""

    def rejected(reason, example=None):
        if debug_counts is not None:
            debug_counts[reason] = debug_counts.get(reason, 0) + 1
        if (example is not None and debug_examples is not None
                and len(debug_examples.setdefault(reason, [])) < 3):
            debug_examples[reason].append(example)
        return None

    perpendicular = (-forward[2] * side, 0, forward[0] * side)
    catalyst = (
        work[0] + forward[0], work[1] - 1,
        work[2] + forward[2])
    result = _translated(catalyst, perpendicular)
    # The catalyst cell is directly in front of the work stance. Its adjacent
    # result cell completes the strictly two-cell fluid envelope.
    excavation = catalyst, result
    supports = tuple((cell[0], cell[1] - 1, cell[2])
                     for cell in excavation)
    directions = ((1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1))
    backing = tuple(sorted({
        _translated(cell, direction)
        for cell in excavation for direction in directions
        if _translated(cell, direction) not in excavation
    }))

    # A protected coordinate may not be modified. An explicit, already-open
    # breadcrumb may still be occupied, and any protected solid may be reused
    # unchanged as the mold's floor/backing or as a clicked placement face.
    # ``protected`` is a mutation contract, not an exclusion bubble: treating
    # immutable backing or a harmless click ray as a write makes a long
    # preserved return breadcrumb poison every nearby mold candidate.
    # Route cells are checked once their exact geometry has materialized.
    work_body = (
        (work[0], work[1] - 1, work[2]),
        work,
        (work[0], work[1] + 1, work[2]),
    )
    passage_supports = {
        (x, y - 1, z)
        for x, y, z in allowed_passage
        if (x, y + 1, z) in allowed_passage
    }
    allowed_geometry = allowed_passage | passage_supports
    if protected.intersection(excavation):
        return rejected("protected_excavation")
    if protected.intersection(work_body) - allowed_geometry:
        return rejected("protected_work_body")

    for cell in excavation:
        if not _known(world, cell):
            return rejected("unknown_excavation")
        block = _block(world, cell)
        # The live executor must break both mold cells before it can place
        # either fluid.  ``is_excavatable`` deliberately answers whether a
        # known value is safe to remove and therefore also accepts air; a
        # mold site is stricter and needs two real solid blocks whose break
        # events can be observed.  Requiring both predicates prevents a
        # prior resource cavity from being certified as fresh ground.
        if not rules.is_solid(block) or not rules.is_excavatable(block):
            return rejected("not_fresh_excavatable_ground")

    # A full-height wall immediately above a valid ground mold blocks the
    # player's click ray even though the two mold cells themselves are safe.
    # Treat only those exact overhead voxels as constructible access clears.
    # Air/passable cells need no mutation; unknown, protected, fluid, hazard,
    # or non-excavatable cells reject the candidate before any live action.
    access_candidates = tuple(dict.fromkeys(
        (cell[0], cell[1] + 1, cell[2]) for cell in excavation))
    access_clear = []
    for cell in access_candidates:
        if not _known(world, cell):
            return rejected("unknown_access_clear")
        block = _block(world, cell)
        if rules.is_passable(block):
            continue
        if (cell in protected or rules.is_fluid(block)
                or rules.is_hazard(block)
                or not rules.is_solid(block)
                or not rules.is_excavatable(block)):
            return rejected("unsafe_access_clear", {
                "cell": cell, "block": block,
                "protected": cell in protected,
            })
        access_clear.append(cell)
    access_clear = tuple(access_clear)
    if not _hazard_clear(
            world, access_clear + excavation + supports + backing + (work,),
            rules=rules, clearance=hazard_clearance):
        return rejected("hazard_clearance")

    eye = _eye(work)
    removed: set[Cell] = set()

    # At most two overhead blocks exist. Prove one deterministic executable
    # order rather than assuming the declaration order is visible: clearing
    # either near block may reveal the other.
    clear_orders = (access_clear,)
    if len(access_clear) == 2:
        clear_orders = (access_clear, tuple(reversed(access_clear)))
    proved_access = None
    for order in clear_orders:
        trial_removed: set[Cell] = set()
        for target in order:
            overlay = _ExcavatedWorld(world, trial_removed)
            center = tuple(value + 0.5 for value in target)
            if (math.dist(eye, center) > max_reach + 1e-9
                    or not clear_voxel_ray(
                        overlay, eye, center, rules=rules, target=target,
                        origin_margin=0.04)):
                break
            trial_removed.add(target)
        else:
            proved_access = order
            removed = trial_removed
            break
    if proved_access is None:
        return rejected("access_clear_ray", {
            "work": work, "access_clear": access_clear,
        })
    access_clear = proved_access

    for target in excavation:
        overlay = _ExcavatedWorld(world, removed)
        center = tuple(value + 0.5 for value in target)
        if not clear_voxel_ray(
                overlay, eye, center, rules=rules, target=target,
                origin_margin=0.04):
            ray = voxel_ray(eye, center)
            return rejected("excavation_ray", {
                "work": work,
                "excavation": excavation,
                "cleared": tuple(sorted(removed)),
                "target": target,
                "ray": tuple((cell, _block(overlay, cell))
                             for cell in ray),
            })
        if math.dist(eye, center) > max_reach + 1e-9:
            return rejected("excavation_reach")
        removed.add(target)
    setup = _cast_setup_placements(
        world, excavation=access_clear + excavation,
        required_solids=supports + backing, work=work,
        material=setup_material, max_reach=max_reach, rules=rules,
        protected=protected)
    if setup is None:
        return rejected("setup_geometry")
    setup_placements, mold = setup

    def fluid_step(cell, *, order, material, role):
        candidates = sorted(
            (support for support in backing
             if sum(abs(cell[index] - support[index])
                    for index in range(3)) == 1),
            key=lambda support: (math.dist(work, support), support),
        )
        for support in candidates:
            step = PlaceStep.on(
                cell, support, order=order, material=material,
                role=role, stance=work)
            if clear_placement_ray(
                    mold, work, cell, support, rules=rules,
                    max_reach=max_reach, origin_margin=0.04):
                return step
        return None

    catalyst_step = fluid_step(
        catalyst, order=0, material=catalyst_fluid,
        role="cast_catalyst")
    reactant_step = fluid_step(
        result, order=1, material=reactant_fluid,
        role="cast_reactant")
    if catalyst_step is None or reactant_step is None:
        return rejected("fluid_placement_ray")
    catalyst_place = _placement_interaction(
        "place_catalyst", mold, catalyst_step,
        rules=rules, max_reach=max_reach)
    reactant_place = _placement_interaction(
        "place_reactant", mold, reactant_step,
        rules=rules, max_reach=max_reach)
    recover = _cell_interaction(
        "recover_catalyst", mold, work, catalyst, rules=rules)
    mine = _cell_interaction(
        "mine_result", mold, work, result, rules=rules)
    if None in (catalyst_place, reactant_place, recover, mine):
        return rejected("interaction_ray")
    interactions = catalyst_place, reactant_place, recover, mine
    if work in route_cache:
        approach = route_cache[work]
        if approach is None:
            return rejected("cached_route")
    else:
        try:
            approach = plan_surface_path(
                world, start, work, rules=rules,
                hazard_clearance=hazard_clearance,
                node_cap=node_cap, decision_rng=route_rng,
                forbidden_cells=protected - allowed_geometry)
        except PlanRejected as exc:
            # Every orientation at one work stance has the same route
            # contract. Cache only exhaustive failures; a bounded A* may find
            # a path under a different deterministic tie order.
            if exc.reason in {
                    RejectReason.NO_PATH, RejectReason.INVALID_GOAL,
                    RejectReason.OUT_OF_BOUNDS}:
                route_cache[work] = None
            return rejected("route")
        route_cache[work] = approach
    route_cells = {
        body
        for x, y, z in approach
        for body in ((x, y - 1, z), (x, y, z), (x, y + 1, z))
    }
    if protected.intersection(route_cells) - allowed_geometry:
        return rejected("protected_route")

    return FluidCastPlan(
        kind=("contained_fluid_cast" if not setup_placements
              else "constructed_contained_fluid_cast"),
        snapshot_sha256=snapshot_sha256,
        world_seed=world_seed,
        decision_seed=decision_seed,
        orientation=(
            f"forward:{forward[0]},{forward[2]};"
            f"side:{'left' if side < 0 else 'right'}"),
        anchor=catalyst,
        casts_required=casts_required,
        catalyst_fluid=catalyst_fluid,
        reactant_fluid=reactant_fluid,
        result_block=result_block,
        catalyst_cell=catalyst,
        result_cell=result,
        access_clear_cells=access_clear,
        excavation_cells=excavation,
        support_cells=supports,
        backing_cells=backing,
        fluid_envelope=excavation,
        temporary_cells=excavation,
        # Execution must leave both cells fluid-free. By default they remain
        # open; any restoration is an explicit FluidCastExecution placement.
        cleanup_cells=excavation,
        setup_placements=setup_placements,
        catalyst_placement=catalyst_step,
        reactant_placement=reactant_step,
        approach_stands=approach,
        work_stance=work,
        catalyst_recovery_stance=work,
        result_mining_stance=work,
        interactions=interactions,
        hazard_clearance=hazard_clearance,
    )


def select_fluid_cast_site(
    world: WorldMap,
    *,
    decision_seed: int,
    start: Cell | None = None,
    casts_required: int = 10,
    catalyst_fluid: str = "water",
    reactant_fluid: str = "lava",
    result_block: str = "obsidian",
    setup_material: str = "cobblestone",
    search_radius: int = 12,
    vertical_radius: int = 3,
    hazard_clearance: int = 1,
    node_cap: int = 20_000,
    max_reach: float = 5.0,
    protected: Iterable[Cell] = (),
    allowed_passage: Iterable[Cell] = (),
    rules: BlockRules = DEFAULT_BLOCK_RULES,
) -> FluidCastPlan:
    """Select a reusable water-first mold without selecting fluid sources.

    The mold is a two-cell cut into known solid ground. Existing safe floor
    and boundary blocks are reused; missing known-air boundary cells are
    filled by exact support-ordered ``setup_material`` placements. The final
    floor and horizontal boundary contain the catalyst/reactant envelope;
    every excavation, bucket placement, recovery, and result-mining ray is
    clear from one dry, safely routed work stance. The same result cell can
    therefore be used ``casts_required`` times with distinct source fluids.
    ``protected`` cells are never excavated or placed into. They may be
    reused unchanged as solid floor/backing or clicked as a placement face;
    neither operation mutates them. Cells also named by ``allowed_passage``
    are a previously certified dry route that the mold approach may occupy.
    Every allowed passage cell must therefore also be protected.
    Without an explicit executor cleanup plan the final mold is open and dry.
    """

    for value, label in (
        (search_radius, "search_radius"),
        (vertical_radius, "vertical_radius"),
        (hazard_clearance, "hazard_clearance"),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{label} must be a non-negative integer")
    if not isinstance(casts_required, int) or isinstance(
            casts_required, bool) or casts_required < 1:
        raise ValueError("casts_required must be a positive integer")
    if not isinstance(setup_material, str) or not setup_material.strip():
        raise ValueError("setup_material must be a nonempty block name")
    if node_cap < 1:
        raise ValueError("node_cap must be positive")
    if max_reach <= 0:
        raise ValueError("max_reach must be positive")
    protected_cells = frozenset(
        _cell(cell, label="protected cell") for cell in protected)
    allowed_passage_cells = frozenset(
        _cell(cell, label="allowed passage cell")
        for cell in allowed_passage)
    if not allowed_passage_cells.issubset(protected_cells):
        raise ValueError("allowed_passage must be a subset of protected")
    if start is None:
        player = getattr(world, "player", None)
        if player is None:
            raise TypeError("start is required when the world has no player")
        start = math.floor(player.x), math.floor(player.y), math.floor(player.z)
    start = _cell(start, label="start")
    if not is_surface_stand(
            world, start, rules=rules,
            hazard_clearance=hazard_clearance):
        raise PlanRejected(
            RejectReason.INVALID_START, start=start,
            detail="fluid-cast planning start is not a safe stand")
    try:
        world_seed = int(getattr(world, "seed"))
        snapshot_sha256 = str(getattr(world, "snapshot_sha256"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(
            "world must expose seed and snapshot_sha256 provenance") from exc

    seed_bytes = hashlib.sha256(
        f"{world_seed}:{decision_seed}:fluid_cast".encode()).digest()
    rng = random.Random(int.from_bytes(seed_bytes[:8], "big"))
    y_values = [start[1]]
    for amount in range(1, vertical_radius + 1):
        y_values.extend((start[1] + amount, start[1] - amount))
    stands = []
    for x in range(start[0] - search_radius, start[0] + search_radius + 1):
        for z in range(start[2] - search_radius, start[2] + search_radius + 1):
            distance = abs(x - start[0]) + abs(z - start[2])
            if distance > search_radius:
                continue
            for y in y_values:
                stand = x, y, z
                if is_surface_stand(
                        world, stand, rules=rules,
                        hazard_clearance=hazard_clearance):
                    stands.append((distance, rng.random(), stand))
    stands.sort(key=lambda item: (item[0], item[1], item[2]))
    directions = [(1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)]
    route_cache: dict[Cell, tuple[Cell, ...] | None] = {}
    debug_counts = ({}
                    if os.environ.get("BRL_DEBUG_CAST_SITE") == "1"
                    else None)
    debug_examples = ({}
                      if debug_counts is not None else None)
    attempts = 0
    for _distance, _tie, work in stands:
        choices = [(forward, side)
                   for forward in directions for side in (-1, 1)]
        rng.shuffle(choices)
        for forward, side in choices:
            attempts += 1
            plan = _fluid_cast_site_plan(
                world, start=start, work=work, forward=forward, side=side,
                decision_seed=int(decision_seed), world_seed=world_seed,
                snapshot_sha256=snapshot_sha256,
                casts_required=casts_required,
                catalyst_fluid=catalyst_fluid,
                reactant_fluid=reactant_fluid,
                result_block=result_block,
                setup_material=setup_material,
                hazard_clearance=hazard_clearance, node_cap=node_cap,
                max_reach=max_reach, rules=rules,
                route_rng=random.Random(rng.getrandbits(64)),
                protected=protected_cells,
                allowed_passage=allowed_passage_cells,
                route_cache=route_cache,
                debug_counts=debug_counts,
                debug_examples=debug_examples,
            )
            if plan is not None:
                return plan
    if debug_counts is not None:
        print(
            "bedrock cast-site rejections "
            + " ".join(
                f"{name}={count}"
                for name, count in sorted(
                    debug_counts.items(), key=lambda item: (-item[1], item[0]))
            ),
            file=sys.stderr,
            flush=True,
        )
        for reason, examples in sorted(debug_examples.items()):
            print(
                f"bedrock cast-site examples {reason}={examples!r}",
                file=sys.stderr,
                flush=True,
            )
    raise PlanRejected(
        RejectReason.NO_PATH,
        start=start,
        examined=attempts,
        detail=(
            "known search region has no dry, hazard-clear, two-cell "
            "contained fluid mold with executable access rays"),
    )
