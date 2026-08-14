"""Immutable inputs consumed by scripted-policy plan executors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bedrock_rl.sdg.policy.planning import (
    MiningTargetPlan,
    RoutePlan,
)
from bedrock_rl.sdg.policy.structures import (
    FluidCastPlan,
    PlaceStep,
)


Cell = tuple[int, int, int]


def _cell(value) -> Cell:
    return tuple(int(item) for item in value)


@dataclass(frozen=True, slots=True)
class MiningExecution:
    """Live expectations attached to one immutable mining target plan."""

    plan: MiningTargetPlan
    block: Any = None
    tool: Any = None
    drop_item: Any = None
    drop_count: int = 0
    fluid_seal_item: Any = None
    block_metadata: tuple[int, ...] | None = None

    def __post_init__(self):
        if not isinstance(self.plan, MiningTargetPlan):
            raise TypeError("MiningExecution.plan must be MiningTargetPlan")
        if (
            not isinstance(self.drop_count, int)
            or isinstance(self.drop_count, bool)
            or self.drop_count < 0
        ):
            raise ValueError("drop_count must be a nonnegative integer")
        if self.drop_count and self.drop_item is None:
            raise ValueError("drop_count needs drop_item")
        if self.plan.removal.fluid_seal_cells and self.fluid_seal_item is None:
            raise ValueError("fluid seal cells need fluid_seal_item")
        if self.block_metadata is not None:
            metadata = tuple(self.block_metadata)
            if (
                not metadata
                or len(metadata) != len(set(metadata))
                or any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or not 0 <= value <= 15
                    for value in metadata
                )
            ):
                raise ValueError(
                    "block_metadata must be unique integers in 0..15"
                )
            object.__setattr__(self, "block_metadata", metadata)


@dataclass(frozen=True, slots=True)
class FluidSourceExecution:
    """One source fluid with separately proved outbound and return legs."""

    role: str
    source_cell: Cell
    outbound_route: RoutePlan
    return_route: RoutePlan
    fluid: Any
    filled_bucket_item: Any
    source_meta: int = 0

    def __post_init__(self):
        if self.role not in ("coolant", "casting"):
            raise ValueError("fluid source role must be coolant or casting")
        object.__setattr__(self, "source_cell", _cell(self.source_cell))
        for route, label in (
            (self.outbound_route, "outbound_route"),
            (self.return_route, "return_route"),
        ):
            if not isinstance(route, RoutePlan):
                raise TypeError(f"fluid source {label} must be RoutePlan")
            if not route.stands:
                raise ValueError(f"fluid source {label} cannot be empty")
        if self.outbound_route.stands[-1] != self.return_route.stands[0]:
            raise ValueError(
                "fluid source outbound and return routes must meet at the "
                "same dry pickup stance"
            )
        if (
            not isinstance(self.source_meta, int)
            or isinstance(self.source_meta, bool)
            or self.source_meta < 0
        ):
            raise ValueError("fluid source metadata must be non-negative")
        if self.outbound_route.stands[-1] == self.source_cell:
            raise ValueError("fluid source routes must meet at a dry stance")


@dataclass(frozen=True, slots=True)
class FluidCastExecution:
    """Inventory and source-route inputs for one contained fluid cast."""

    plan: FluidCastPlan
    sources: tuple[FluidSourceExecution, ...]
    empty_bucket_item: Any = "bucket"
    result_item: Any = "obsidian"
    mining_tool: Any = "diamond_pickaxe"
    mining_ticks: int = 188
    excavation_tool: Any = "diamond_pickaxe"
    excavation_ticks: int = 80
    drop_count: int = 1
    cleanup_placements: tuple[PlaceStep, ...] = ()
    casting_wait_ticks: int = 1
    recover_coolant: bool = True

    def __post_init__(self):
        if not isinstance(self.plan, FluidCastPlan):
            raise TypeError("FluidCastExecution.plan must be FluidCastPlan")
        sources = tuple(self.sources)
        object.__setattr__(self, "sources", sources)
        cleanup = tuple(self.cleanup_placements)
        object.__setattr__(self, "cleanup_placements", cleanup)
        if any(not isinstance(step, PlaceStep) for step in cleanup):
            raise TypeError("cleanup_placements must contain PlaceStep values")
        if tuple(step.order for step in cleanup) != tuple(range(len(cleanup))):
            raise ValueError("cleanup placement orders must be contiguous")
        if (
            len({step.cell for step in cleanup}) != len(cleanup)
            or not {step.cell for step in cleanup}.issubset(
                set(self.plan.cleanup_cells)
            )
        ):
            raise ValueError(
                "cleanup placements must uniquely target cleanup_cells"
            )
        expected = self.plan.casts_required + 1
        if len(sources) != expected:
            raise ValueError(
                "fluid cast needs one coolant and "
                f"{self.plan.casts_required} casting sources"
            )
        if sources[0].role != "coolant" or any(
            source.role != "casting" for source in sources[1:]
        ):
            raise ValueError(
                "fluid sources must be coolant first, then casting sources"
            )
        casting_cells = tuple(source.source_cell for source in sources[1:])
        if len(casting_cells) != len(set(casting_cells)):
            raise ValueError("casting source cells must be unique")
        if len({source.source_cell for source in sources}) != len(sources):
            raise ValueError("coolant and casting source cells must differ")
        if any(
            source.outbound_route.stands[0] != self.plan.work_stance
            for source in sources
        ):
            raise ValueError(
                "every source outbound_route must start at work_stance"
            )
        if any(
            source.return_route.stands[-1] != self.plan.work_stance
            for source in sources
        ):
            raise ValueError(
                "every source return_route must end at work_stance"
            )
        if sources[0].fluid != self.plan.catalyst_fluid or any(
            source.fluid != self.plan.reactant_fluid for source in sources[1:]
        ):
            raise ValueError("source fluids do not match the cast plan")
        for value, label in (
            (self.mining_ticks, "mining_ticks"),
            (self.excavation_ticks, "excavation_ticks"),
            (self.drop_count, "drop_count"),
            (self.casting_wait_ticks, "casting_wait_ticks"),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(f"fluid cast {label} must be positive")
        if not isinstance(self.recover_coolant, bool):
            raise TypeError("fluid cast recover_coolant must be true or false")
        if not self.recover_coolant:
            raise ValueError(
                "open-dry fluid cast requires recover_coolant=True; leaving "
                "the catalyst fluid in the mold is unsupported"
            )


__all__ = ("FluidCastExecution", "FluidSourceExecution", "MiningExecution")
