"""Live orchestration for a declarative survival progression."""

from __future__ import annotations

import hashlib
import math
import os
import random
import sys
import time
from types import SimpleNamespace
from typing import Mapping

from bedrock_rl.sdg.policy import (
    MinecraftController,
    PolicyVeto,
    WallTorchRouteMixin,
)
from bedrock_rl.sdg.policy.execution import (
    FluidCastExecution,
    FluidCastExecutor,
    FluidSourceExecution,
    MiningExecution,
    PlanExecutor,
)
from bedrock_rl.sdg.policy.fluids import plan_fluid_sources
from bedrock_rl.sdg.policy.planning import (
    GoalSpec,
    ToolDurabilityLedger,
    WorldPlan,
    WorldPlanBuilder,
    WorldPlanningRejected,
    require_fluid_seal_budget,
)
from bedrock_rl.sdg.policy.progression.portal import (
    PORTAL_STAGES,
    NetherPortalExecutor,
)
from bedrock_rl.sdg.policy.progression.spec import (
    BeginLightingOperation,
    BucketCastMineOperation,
    Cell,
    CraftOperation,
    NetherPortalOperation,
    RepeatDropOperation,
    RequireItemOperation,
    SmeltOperation,
    SurvivalProgressionSpec,
    WorkstationOperation,
    WorkstationSite,
    _cell,
    _ChangedWorld,
    _LightingHooks,
    _portal_protected_cells,
    _resource_targets,
    _ReturnBreadcrumb,
    _route_preservation_cells,
    _targets,
    _tool_break_capacity,
    _workstation_protected_cells,
    _workstation_sites,
)
from bedrock_rl.sdg.policy.spatial import (
    DEFAULT_BLOCK_RULES,
    MINING_REACH,
    PlanRejected,
    clear_placement_ray,
    clear_voxel_ray,
    is_excavation_stand,
    is_surface_stand,
    placement_face_point,
    plan_staircase_tunnel,
    plan_surface_path,
)
from bedrock_rl.sdg.policy.structures import (
    FluidCastPlan,
    StructurePlan,
    select_fluid_cast_site,
    select_nether_portal_completion_site,
    select_nether_portal_site,
)
from bedrock_rl.sdg.policy.world import SnapshotWorldMap
from bedrock_rl.sdg.generation import CaseRejected
from bedrock_rl.env.catalog import resolve_items
from bedrock_rl.env.journal import FLUID_BODY


class SurvivalProgressionPolicy(WallTorchRouteMixin, MinecraftController):
    """Execute a planned survival recipe from an empty natural spawn.

    The object is directly usable as a ``ScriptedEpisodeGenerator`` policy.
    It remains intentionally strict: hidden inventory guesses, unproved
    clicks, health loss, fluid contact, missing stochastic drops, route
    divergence, and an impossible portal site reject the case instead of
    being papered over with a seed-specific detour.
    """

    def __init__(
        self,
        spec=None,
        movement=None,
        behaviors=None,
        *,
        world_factory=SnapshotWorldMap,
        plan_builder=WorldPlanBuilder,
        portal_selector=select_nether_portal_site,
        completion_portal_selector=select_nether_portal_completion_site,
        cast_selector=select_fluid_cast_site,
        fluid_source_planner=plan_fluid_sources,
    ):
        super().__init__(movement=movement, behaviors=behaviors)
        self.spec = SurvivalProgressionSpec.parse(spec)
        self.world_factory = world_factory
        self.plan_builder = plan_builder
        self.portal_selector = portal_selector
        self.completion_portal_selector = completion_portal_selector
        self.cast_selector = cast_selector
        self.fluid_source_planner = fluid_source_planner
        self._portal_executor = NetherPortalExecutor(self)
        self.plan: WorldPlan | None = None
        self.portal_plan: StructurePlan | None = None
        self.cast_plan: FluidCastPlan | None = None
        self.fluid_source_plan = None
        self._cast_executor = None
        self.workstation_sites: dict[str, WorkstationSite] = {}
        self._protected_cells = frozenset()
        self._guard_event_index = 0
        self._baseline_health = None
        self._repeat_states = {}
        self._allowed_break_cells = frozenset()
        self._lighting_initial_origin = None
        self._lighting_initial_vantage = None
        self._lighting_initial_returning = False
        self._origin_settle_goal = None
        self._origin_settle_attempts = 0
        self._drop_pickup_history_target = None
        self._drop_pickup_history = []
        self._portal_clear_event_start = 0
        self._portal_place_control_attempts = {}
        self._portal_placement_route_index = None
        self._portal_placement_event_start = 0
        self._portal_ignite_control_attempts = 0
        self.stage = "not_reset"
        self._world = None

    def close(self):
        world, self._world = self._world, None
        if world is not None:
            world.close()

    def reset(self, task, episode, case):
        super().reset(task, episode, case)
        self.reset_route_lighting()
        self._baseline_health = self._health(episode.env)
        self._event_start = len(episode.env.journal)
        self._guard_event_index = self._event_start
        self._lighting_active = False
        self._lighting_initial_site = None
        self._lighting_initial_origin = None
        self._lighting_initial_vantage = None
        self._lighting_initial_returning = False
        self._stage_index = 0
        self._operation_index = 0
        self._executor = None
        self._cast_executor = None
        self._route = ()
        self._route_index = 0
        self._route_attempts = 0
        self._placement_index = 0
        self._portal_ignite_attempts = 0
        self._portal_ignite_control_attempts = 0
        self._portal_entry_attempts = 0
        self._portal_clear_attempts = {}
        self._portal_clear_control_attempts = {}
        self._portal_clear_event_start = self._event_start
        self._portal_place_control_attempts = {}
        self._portal_placement_route_index = None
        self._portal_placement_event_start = self._event_start
        self._repeat_states = {}
        self.portal_plan = None
        self.cast_plan = None
        self.fluid_source_plan = None
        self.workstation_sites = {}
        self.plan = None
        self._stage_plans = []
        self._provenance_workstations = []
        self._opened = set()
        self._removed = set()
        self._placed = {}
        self._protected_cells = frozenset()
        self._return_passage_cells = set()
        self._return_support_cells = set()
        self._return_breadcrumb = None
        self._decision_seed = int(case.decision_seed)
        self._route_mine_cells = frozenset()
        self._route_clear_attempts = {}
        self._workstation_route_keys = set()
        self._portal_return_certificate = None
        self._origin_settle_goal = None
        self._origin_settle_attempts = 0
        self._drop_pickup_history_target = None
        self._drop_pickup_history = []
        self._allowed_break_cells = frozenset()
        self._costs = None
        self._preflight_timing_seconds = {}
        debug_preflight = os.environ.get("BRL_DEBUG_PREFLIGHT") == "1"

        def timed(label, function, *args, **kwargs):
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = round(time.perf_counter() - started, 6)
                self._preflight_timing_seconds[label] = elapsed
                if debug_preflight:
                    print(
                        "bedrock preflight "
                        f"stage={label} seconds={elapsed:.6f}",
                        file=sys.stderr,
                        flush=True,
                    )

        snapshot = getattr(episode.env, "snapshot_in", None)
        if not snapshot:
            raise CaseRejected(
                "survival progression requires an initial-state snapshot",
                code="progression_snapshot_missing")
        self.close()
        self._world = self.world_factory(snapshot)
        world = self._world
        start = (math.floor(float(episode.env.obs["x"])),
                 math.floor(float(episode.env.obs["y"])),
                 math.floor(float(episode.env.obs["z"])))
        # Preflight every mandatory natural resource before the first action.
        # Routes are built as independent stage plans in progression order,
        # with the exact simulated endpoint and mutation overlay handed to
        # the next stage. This is deliberately up-front: hostile seeds must
        # reject before producing a partial demonstration.
        requirements = {stage.name: stage.requirement.block_ids
                        for stage in self.spec.stages}
        minima = {stage.name: stage.count for stage in self.spec.stages}
        counts = timed(
            "resource_counts", world.count_blocks_many,
            requirements, limits=minima)
        for stage in self.spec.stages:
            available = counts[stage.name]
            if available < stage.count:
                raise CaseRejected(
                    f"world has {available}/{stage.count} {stage.name}",
                    code="progression_resource_missing")
        if (self.spec.portal.enabled
                and self.spec.portal.strategy == "initial"):
            try:
                self.portal_plan = timed(
                    "portal_site", self.portal_selector,
                    world, decision_seed=case.decision_seed,
                    start=start,
                    search_radius=self.spec.portal.search_radius,
                    vertical_radius=self.spec.portal.vertical_radius,
                    hazard_clearance=self.spec.portal.hazard_clearance,
                    node_cap=self.spec.node_cap)
            except PlanRejected as exc:
                raise CaseRejected(
                    f"no safe player-built portal site: {exc}",
                    code="progression_portal_site_missing",
                    details={"preflight_timing_seconds": dict(
                        self._preflight_timing_seconds)}) from exc
            self._portal_cells = _portal_protected_cells(self.portal_plan)
        else:
            self._portal_cells = frozenset()
        if (self.spec.portal.enabled
                and self.spec.portal.strategy == "initial"):
            self._return_breadcrumb = _ReturnBreadcrumb(start)
            self._refresh_return_breadcrumb_cells()
        self._stage_plan_objects = []
        simulated = SimpleNamespace(obs=dict(episode.env.obs))
        for stage in self.spec.stages:
            try:
                plan = timed(
                    f"resource_stage:{stage.name}",
                    self._plan_stage, simulated, stage)
            except CaseRejected as exc:
                raise CaseRejected(
                    str(exc), code=exc.code,
                    details={
                        **dict(exc.details),
                        "preflight_timing_seconds": dict(
                            self._preflight_timing_seconds),
                    }) from exc
            self._stage_plan_objects.append(plan)
            self._commit_stage_plan(stage)
            operations = [operation for operation in stage.after
                          if isinstance(operation, WorkstationOperation)]
            if operations:
                for operation in operations:
                    site = self.workstation_sites[operation.key]
                    self._placed[site.cell] = self._block_ids(
                        operation.block)[0]
                end = self.workstation_sites[operations[-1].key].stand
            else:
                end = _resource_targets(
                    plan, stage.name)[-1].mining_stance
            simulated.obs.update(
                x=end[0] + 0.5, y=float(end[1]), z=end[2] + 0.5)
        cast = self._bucket_cast_operation()
        if cast is not None:
            if self.spec.portal.source_meta != 0:
                raise CaseRejected(
                    "bucket collection requires metadata-zero source fluids",
                    code="progression_fluid_source_meta_unsupported")
            if not self.spec.portal.unique_casting_sources:
                raise CaseRejected(
                    "consumable casting fluid needs one unique source per "
                    "cast",
                    code="progression_fluid_source_reuse_unsupported")
            if not self.spec.portal.recover_coolant:
                raise CaseRejected(
                    "the one-bucket dry-mold workflow must recover its "
                    "coolant",
                    code="progression_coolant_recovery_required")
            protected = (
                set(self._portal_cells)
                | set(_workstation_protected_cells(
                    self.workstation_sites))
                | set(self._return_passage_cells)
                | set(self._return_support_cells)
            )
            changed = _ChangedWorld(
                world, opened=self._opened, removed=self._removed,
                placed=self._placed)
            try:
                self.cast_plan = timed(
                    "fluid_cast_site", self.cast_selector, changed,
                    decision_seed=case.decision_seed,
                    start=self._current_cell(simulated),
                    casts_required=cast.minimum,
                    catalyst_fluid=self.spec.portal.coolant_fluid,
                    reactant_fluid=self.spec.portal.casting_fluid,
                    result_block=self.spec.portal.cast_block,
                    setup_material=self.spec.portal.casting_backing_item,
                    search_radius=self.spec.portal.search_radius,
                    vertical_radius=self.spec.portal.vertical_radius,
                    hazard_clearance=self.spec.portal.hazard_clearance,
                    node_cap=self.spec.node_cap,
                    max_reach=MINING_REACH,
                    protected=protected,
                    allowed_passage=self._return_passage_cells,
                )
            except PlanRejected as exc:
                raise CaseRejected(
                    f"no safe contained fluid-cast site: {exc}",
                    code="progression_fluid_cast_site_missing",
                    details={"preflight_timing_seconds": dict(
                        self._preflight_timing_seconds)}) from exc
            # Keep the contract true for injected/custom cast selectors too.
            # The executor emits one exact block-break action and requires a
            # matching engine journal event for every mold cell, so terrain
            # already opened by an earlier resource leg is not a valid mold.
            prior_air = (
                set(self._opened) | set(self._removed)
            ).difference(self._placed)
            stale_mold = prior_air.intersection(
                _cell(cell) for cell in self.cast_plan.excavation_cells)
            if stale_mold:
                raise CaseRejected(
                    "contained fluid-cast mold must cut fresh solid ground; "
                    f"already-open cells: {sorted(stale_mold)[:4]}",
                    code="progression_fluid_cast_site_invalid",
                    details={
                        "cells": [list(cell)
                                  for cell in sorted(stale_mold)[:4]],
                        "preflight_timing_seconds": dict(
                            self._preflight_timing_seconds),
                    })
            self._extend_return_breadcrumb(
                self.cast_plan.approach_stands,
                source="fluid-cast approach")
            protected = (
                set(self._portal_cells)
                | set(_workstation_protected_cells(
                    self.workstation_sites))
                | set(self._return_passage_cells)
                | set(self._return_support_cells)
            )
            try:
                self.fluid_source_plan = timed(
                    "fluid_sources", self.fluid_source_planner, world,
                    self.cast_plan,
                    decision_seed=case.decision_seed,
                    lava_sources=cast.minimum,
                    water_blocks=self.spec.portal.coolant_fluid,
                    lava_blocks=self.spec.portal.casting_fluid,
                    opened=self._opened,
                    removed=self._removed,
                    placed=self._placed,
                    protected=protected,
                    allowed_passage=self._return_passage_cells,
                    search_radius=self.spec.portal.fluid_search_radius,
                    source_meta=self.spec.portal.source_meta,
                    unique_casting_sources=(
                        self.spec.portal.unique_casting_sources),
                    hazard_clearance=0,
                    node_cap=self.spec.node_cap,
                    route_search_cap=max(
                        self.spec.route_search_cap,
                        self.spec.portal.fluid_search_radius),
                    candidate_limit=self.spec.cluster_sample_limit,
                    max_reach=MINING_REACH,
                )
            except PlanRejected as exc:
                raise CaseRejected(
                    f"no exact source-fluid itinerary: {exc}",
                    code="progression_fluid_source_missing",
                    details={"preflight_timing_seconds": dict(
                        self._preflight_timing_seconds)}) from exc

            self._validate_fluid_source_plan(protected)

            # Simulate the exact final state left by the immutable cast
            # execution. Sources, route excavation, the two-cell mold, and
            # every mined result are air; the one recovered coolant bucket is
            # inventory state and therefore needs no world overlay entry.
            self._opened.update(
                cell for cell in self.fluid_source_plan.newly_opened_cells
                if cell not in {
                    step.source_cell
                    for step in self.fluid_source_plan.steps
                })
            self._opened.update(self.cast_plan.final_open_cells)
            for step in getattr(self.cast_plan, "setup_placements", ()):
                self._placed[_cell(step.cell)] = self._block_ids(
                    step.material)[0]
            end = self.cast_plan.work_stance
            simulated.obs.update(
                x=end[0] + 0.5, y=float(end[1]), z=end[2] + 0.5)
        if (self.spec.portal.enabled
                and self.spec.portal.strategy != "initial"):
            try:
                selector = (self.completion_portal_selector
                            if self.spec.portal.strategy
                            == "completion_chamber"
                            else self.portal_selector)
                self.portal_plan = timed(
                    "portal_site", selector,
                    self._changed_world(),
                    decision_seed=case.decision_seed,
                    start=self._current_cell(simulated),
                    search_radius=self.spec.portal.search_radius,
                    vertical_radius=self.spec.portal.vertical_radius,
                    hazard_clearance=self.spec.portal.hazard_clearance,
                    node_cap=self.spec.node_cap)
            except PlanRejected as exc:
                raise CaseRejected(
                    f"no safe player-built portal site at completion: {exc}",
                    code="progression_portal_site_missing",
                    details={"preflight_timing_seconds": dict(
                        self._preflight_timing_seconds)}) from exc
            self._portal_cells = _portal_protected_cells(self.portal_plan)
        if (self.spec.portal.enabled
                and self.spec.portal.strategy == "initial"):
            route, mine_cells = timed(
                "portal_return", self._portal_return_plan,
                self._current_cell(simulated))
            self._portal_return_certificate = {
                "start": list(self._current_cell(simulated)),
                "goal": list(self.portal_plan.approach_stands[0]),
                "stands": [list(cell) for cell in route],
                "mine_cells": [list(cell) for cell in mine_cells],
                "source": "simulated_final_snapshot_overlay",
            }
        self._allowed_break_cells = self._planned_break_cells()
        self._costs = timed(
            "cost_certificate", self._build_cost_certificate)
        self._require_tool_durability_budget()
        self._require_light_budget()
        self.plan = None
        self._protected_cells = frozenset()
        self.stage = f"mine:{self.spec.stages[0].name}"

    def _stage_seed(self, stage):
        digest = hashlib.sha256(
            f"{self._decision_seed}:stage:{stage.name}".encode()).digest()
        return int.from_bytes(digest[:8], "big")

    def _current_cell(self, env):
        return (math.floor(float(env.obs["x"])),
                math.floor(float(env.obs["y"]) + 0.05),
                math.floor(float(env.obs["z"])))

    def _plan_stage(self, env, stage):
        """Plan only the next resource from the exact live route root."""

        preserve_return = (
            self.spec.portal.enabled
            and self.spec.portal.strategy == "initial"
        )
        persistent_protected = (
            set(self._portal_cells)
            | set(self._placed)
            | set(_workstation_protected_cells(self.workstation_sites))
            | set(getattr(self, "_return_passage_cells", ()))
            | set(getattr(self, "_return_support_cells", ()))
        )
        goal = GoalSpec(
            resources=(stage.requirement,),
            natural_obsidian=stage.natural_only,
            hazard_clearance=(self.spec.hazard_clearance
                              if stage.hazard_clearance is None
                              else stage.hazard_clearance),
            node_cap=self.spec.node_cap,
            route_search_cap=self.spec.route_search_cap,
            routes=stage.route_kinds,
        )
        plan = self.plan_builder(
            self._world, goal, decision_seed=self._stage_seed(stage),
            cluster_sample_limit=self.spec.cluster_sample_limit,
            start=self._current_cell(env), opened=self._opened,
            removed=self._removed, placed=self._placed,
            protected=persistent_protected,
            preserve_traversed_route=preserve_return,
            traversed_passage=getattr(
                self, "_return_passage_cells", ()),
            traversed_supports=getattr(
                self, "_return_support_cells", ())).build()
        targets = tuple(_targets(plan))
        invalid_routes = tuple(
            target.route.kind for target in targets
            if target.route.kind not in stage.route_kinds)
        if invalid_routes:
            raise CaseRejected(
                f"{stage.name!r} plan uses disallowed route kinds "
                f"{invalid_routes!r}",
                code="progression_route_kind_rejected")
        overlap = tuple(sorted(
            target.cell
            for index, target in enumerate(targets)
            if target.cell in {
                cell for later in targets[index:]
                for cell in (*later.route.support_cells,
                             *later.removal.protected_support_cells)
            }))
        if overlap:
            raise CaseRejected(
                f"{stage.name!r} mining would remove route support "
                f"{overlap[:4]}",
                code="progression_route_support_conflict")
        sites, _changed = _workstation_sites(
            self._world, plan, self.spec, self._decision_seed,
            stages=(stage,), opened=self._opened, removed=self._removed,
            placed=self._placed, protected=persistent_protected,
            preserve_return_route=preserve_return)
        if preserve_return:
            self._validate_stage_breadcrumb(
                stage, plan, sites,
                prior_passage=getattr(
                    self, "_return_passage_cells", ()),
                prior_supports=getattr(
                    self, "_return_support_cells", ()))
        self.workstation_sites.update(sites)
        self._provenance_workstations.extend(
            sites[key].to_dict() for key in sorted(sites))
        self.plan = plan
        self._stage_plans.append(plan.to_dict())
        self._protected_cells = frozenset(
            cell for target in _targets(plan)
            for cell in target.removal.protected_support_cells)
        self._preserve_stage_breadcrumb(stage, plan)
        return plan

    @staticmethod
    def _validate_stage_breadcrumb(
            stage, plan, sites, *, prior_passage=(), prior_supports=()):
        """Reject any later mutation that damages this stage's return spine."""

        passage: set[Cell] = {_cell(cell) for cell in prior_passage}
        supports: set[Cell] = {_cell(cell) for cell in prior_supports}
        for target in _targets(plan):
            mutations = {
                _cell(target.cell),
                *(_cell(cell) for cell in target.route.mine_cells),
            }
            overlap = mutations.intersection(passage | supports)
            if overlap:
                raise CaseRejected(
                    f"{stage.name!r} later mining damages an earlier "
                    f"return leg at {sorted(overlap)[:4]}",
                    code="progression_route_support_conflict")
            route_passage, route_supports = _route_preservation_cells(
                target.route)
            passage.update(route_passage)
            supports.update(route_supports)
        seal_overlap = set(getattr(plan, "fluid_seal_cells", ())).intersection(
            passage)
        if seal_overlap:
            raise CaseRejected(
                f"{stage.name!r} fluid seals block its return leg at "
                f"{sorted(seal_overlap)[:4]}",
                code="progression_route_support_conflict")
        workstation_mutations = {
            _cell(cell)
            for site in sites.values()
            for cell in (site.cell, *site.clear_cells)
        }
        overlap = workstation_mutations.intersection(passage | supports)
        if overlap:
            raise CaseRejected(
                f"{stage.name!r} workstation blocks its return leg at "
                f"{sorted(overlap)[:4]}",
                code="progression_route_support_conflict")

    def _preserve_stage_breadcrumb(self, stage, plan):
        """Advance and loop-splice the spawn-to-current return certificate.

        Later resource routes may traverse already-open breadcrumb air, but
        they may not mine its body/support, select it as a resource, or place
        a workstation into it.  The final workstation approach is part of the
        spine because stage execution ends there; earlier workstations remain
        protected as structures but are not required return waypoints.
        """

        for target in _targets(plan):
            self._extend_return_breadcrumb(
                target.route.stands, source=f"{stage.name} resource route")
        operations = tuple(
            operation for operation in stage.after
            if isinstance(operation, WorkstationOperation))
        if operations:
            approach = self.workstation_sites[
                operations[-1].key].approach_stands
            self._extend_return_breadcrumb(
                approach, source=f"{stage.name} workstation approach")

    def _extend_return_breadcrumb(self, stands, *, source="planned route"):
        breadcrumb = getattr(self, "_return_breadcrumb", None)
        if breadcrumb is None:
            return
        breadcrumb.extend(stands)
        self._refresh_return_breadcrumb_cells()
        if breadcrumb.broken:
            raise CaseRejected(
                f"{source} is disconnected or non-cardinal",
                code="progression_return_breadcrumb_invalid")

    def _validate_fluid_source_plan(self, protected):
        """Postvalidate injected source planners before any live actions."""

        plan = self.fluid_source_plan
        work = _cell(self.cast_plan.work_stance)
        declared_work = getattr(plan, "work_stance", work)
        if _cell(declared_work) != work:
            raise CaseRejected(
                "fluid-source plan uses a different cast work stance",
                code="progression_fluid_source_plan_invalid")
        source_cells = set()
        for index, step in enumerate(getattr(plan, "steps", ())):
            try:
                outbound = tuple(_cell(cell) for cell in step.outbound.stands)
                returned = tuple(
                    _cell(cell) for cell in step.return_route.stands)
                source_cells.add(_cell(step.source_cell))
            except (AttributeError, TypeError, ValueError) as exc:
                raise CaseRejected(
                    f"fluid-source step {index} has incomplete route data",
                    code="progression_fluid_source_plan_invalid") from exc
            if (not outbound or outbound[0] != work
                    or not returned or returned[-1] != work):
                raise CaseRejected(
                    f"fluid-source step {index} does not make a round trip "
                    "from the cast work stance",
                    code="progression_fluid_source_plan_invalid")
        newly_opened = {
            _cell(cell)
            for cell in getattr(plan, "newly_opened_cells", ())
        }
        conflict = newly_opened.intersection(
            {_cell(cell) for cell in protected})
        if conflict:
            raise CaseRejected(
                "fluid-source plan mutates protected geometry at "
                f"{sorted(conflict)[:4]}",
                code="progression_fluid_source_plan_invalid")

    def _refresh_return_breadcrumb_cells(self):
        breadcrumb = getattr(self, "_return_breadcrumb", None)
        if breadcrumb is None or breadcrumb.broken:
            self._return_passage_cells = set()
            self._return_support_cells = set()
            return
        self._return_passage_cells = set(breadcrumb.passage_cells)
        self._return_support_cells = set(breadcrumb.support_cells)

    def _commit_stage_plan(self, stage):
        if self.plan is None:
            return
        targets = _targets(self.plan)
        self._opened.update(
            cell for target in targets for cell in target.route.mine_cells)
        self._removed.update(target.cell for target in targets)
        seal_cells = tuple(getattr(self.plan, "fluid_seal_cells", ()))
        if seal_cells:
            if not stage.fluid_seal_item:
                raise CaseRejected(
                    f"{stage.name!r} plan requires fluid seals without a "
                    "declared placement item",
                    code="progression_fluid_seal_item_missing")
            seal_block = self._block_ids(stage.fluid_seal_item)[0]
            self._placed.update(
                {_cell(cell): seal_block for cell in seal_cells})

    def _planned_break_cells(self):
        """Every coordinate the immutable progression may deliberately mine."""

        cells = {
            _cell(cell)
            for plan in self._stage_plan_objects
            for target in _targets(plan)
            for cell in (target.cell, *target.route.mine_cells)
        }
        # Repeat-drop operations re-place and re-mine an already declared
        # resource target. Resolve those references explicitly so a custom
        # data-only progression cannot silently introduce an undeclared cell.
        repeated = {
            operation.resource
            for stage in self.spec.stages
            for operation in stage.after
            if isinstance(operation, RepeatDropOperation)
        }
        for resource_name in repeated:
            targets = tuple(
                target
                for plan in self._stage_plan_objects
                for resource in plan.resources
                if resource.name == resource_name
                for cluster in resource.clusters
                for target in cluster.targets
            )
            if not targets:
                raise ValueError(
                    f"repeat-drop resource {resource_name!r} has no plan")
            cells.add(_cell(targets[-1].cell))
        if self._portal_return_certificate is not None:
            cells.update(
                _cell(cell)
                for cell in self._portal_return_certificate["mine_cells"])
        if self.portal_plan is not None:
            cells.update(
                _cell(cell)
                for cell in getattr(self.portal_plan, "clear_cells", ()))
        if self.cast_plan is not None:
            cells.update(_cell(cell)
                         for cell in self.cast_plan.access_clear_cells)
            cells.update(_cell(cell)
                         for cell in self.cast_plan.excavation_cells)
            cells.add(_cell(self.cast_plan.result_cell))
        if self.fluid_source_plan is not None:
            cells.update(_cell(cell) for cell in
                         self.fluid_source_plan.newly_opened_cells)
        cells.update(
            cell for site in self.workstation_sites.values()
            for cell in site.clear_cells)
        return frozenset(cells)

    @staticmethod
    def _route_edges(cells):
        compact = []
        for value in cells:
            cell = _cell(value)
            if not compact or compact[-1] != cell:
                compact.append(cell)
        return tuple(
            tuple(sorted((left, right)))
            for left, right in zip(compact, compact[1:])
            if left != right)

    def _planned_routes(self):
        """Ordered route sequences materialized by the complete preflight."""

        routes = []
        lighting_started = False
        for stage, plan in zip(self.spec.stages, self._stage_plan_objects):
            for target in _targets(plan):
                routes.append((stage.name, target.route.kind,
                               tuple(target.route.stands), lighting_started))
            operations = tuple(stage.after)
            for operation in operations:
                if isinstance(operation, WorkstationOperation):
                    site = self.workstation_sites[operation.key]
                    routes.append((stage.name, "workstation",
                                   tuple(site.approach_stands),
                                   lighting_started))
                if isinstance(operation, BeginLightingOperation):
                    lighting_started = True
        if self.cast_plan is not None:
            # Mold approach is executed before FluidCastExecutor begins
            # route-distance lighting. Every source leg delegates lighting
            # through the same policy hooks once casting is under way.
            routes.append((
                "casting", "approach",
                tuple(self.cast_plan.approach_stands), False))
        if self.fluid_source_plan is not None:
            for step in self.fluid_source_plan.steps:
                routes.append((
                    "casting", step.outbound.kind,
                    tuple(step.outbound.stands), lighting_started))
                routes.append((
                    "casting", step.return_route.kind,
                    tuple(step.return_route.stands), lighting_started))
        if self._portal_return_certificate is not None:
            routes.append((
                "portal", "return",
                tuple(_cell(cell) for cell in
                      self._portal_return_certificate["stands"]),
                lighting_started))
        if self.portal_plan is not None:
            routes.append(("portal", "approach",
                           tuple(self.portal_plan.approach_stands),
                           lighting_started))
        return tuple(routes)

    def _crafted_minimums(self):
        result = {}
        for stage in self.spec.stages:
            for operation in stage.after:
                if isinstance(operation, CraftOperation):
                    result[operation.item] = max(
                        result.get(operation.item, 0), operation.minimum)
        return result

    def _bucket_cast_operation(self):
        return next((
            operation
            for stage in self.spec.stages for operation in stage.after
            if isinstance(operation, BucketCastMineOperation)
        ), None)

    @staticmethod
    def _recipe_fits(recipe, grid):
        if recipe["shaped"]:
            shape = recipe["shape"]
            return len(shape) <= grid and len(shape[0]) <= grid
        return len(recipe["ingredients"]) <= grid * grid

    @classmethod
    def _recipe_consumption(cls, operation, item_ids):
        """Guaranteed ingredient use for the GUI's first fitting recipe."""
        from bedrock_rl.sdg.policy.recipes import (
            recipe_variants,
        )

        recipe = next((row for row in recipe_variants(operation.item)
                       if cls._recipe_fits(row, operation.grid)), None)
        if recipe is None:
            raise CaseRejected(
                f"no {operation.grid}x{operation.grid} recipe for "
                f"{operation.item!r}",
                code="progression_recipe_preflight_missing")
        ingredients = (
            [value for row in recipe["shape"] for value in row
             if value is not None]
            if recipe["shaped"] else list(recipe["ingredients"]))
        per_craft = sum(int(value[0]) in item_ids for value in ingredients)
        output_count = int(recipe["output"][1])
        crafts = math.ceil(operation.minimum / output_count)
        return per_craft * crafts

    def _material_budget(self, item):
        """Data-driven production/consumption budget for one item name."""
        item_ids = frozenset(resolve_items([item]))
        produced = 0
        consumed = 0
        for stage in self.spec.stages:
            drop = ((stage.drop_item,) if isinstance(stage.drop_item, str)
                    else tuple(stage.drop_item))
            if drop and all(
                    frozenset(resolve_items([value])) & item_ids
                    for value in drop):
                produced += stage.count
            for operation in stage.after:
                if isinstance(operation, CraftOperation):
                    if frozenset(resolve_items([operation.item])) & item_ids:
                        produced += operation.minimum
                    consumed += self._recipe_consumption(
                        operation, item_ids)
                elif isinstance(operation, WorkstationOperation):
                    if frozenset(resolve_items([operation.item])) & item_ids:
                        consumed += 1
                elif isinstance(operation, SmeltOperation):
                    if frozenset(resolve_items(
                            [operation.input_item])) & item_ids:
                        consumed += operation.count
                    if frozenset(resolve_items(
                            [operation.fuel_item])) & item_ids:
                        consumed += operation.fuel_count
                elif isinstance(operation, BucketCastMineOperation):
                    if frozenset(resolve_items([operation.item])) & item_ids:
                        produced += operation.minimum
        return {
            "item": item,
            "produced": produced,
            "consumed_before_structure": consumed,
            "available": max(0, produced - consumed),
        }

    def _fluid_seal_budgets(self):
        """Aggregate unique fluid seals for each declared placement item."""
        result = []
        items = tuple(dict.fromkeys(
            stage.fluid_seal_item for stage in self.spec.stages
            if stage.fluid_seal_item))
        for item in items:
            cells = tuple(dict.fromkeys(
                cell
                for stage, plan in zip(
                    self.spec.stages, self._stage_plan_objects)
                if stage.fluid_seal_item == item
                for cell in plan.fluid_seal_cells))
            material = self._material_budget(item)
            reserve = (0 if self.portal_plan is None
                       or not (frozenset(resolve_items(
                           [self.spec.portal.scaffold_item]))
                               & frozenset(resolve_items([item]))) else sum(
                           step.role == "scaffold"
                           for step in self.portal_plan.placements))
            try:
                ledger = require_fluid_seal_budget(
                    SimpleNamespace(fluid_seal_cells=cells),
                    available=material["available"], reserve=reserve)
            except WorldPlanningRejected as exc:
                details = {**material, **exc.details}
                raise CaseRejected(
                    "planned fluid seals exceed available material after "
                    f"structure reserve; seals={details!r}",
                    code="progression_fluid_seal_budget",
                    details=details) from exc
            result.append({**material, **ledger.to_dict()})
        return tuple(result)

    def _tool_ledgers(self):
        """Charge every unique planned break to its declared live tool."""

        cells_by_tool = {}
        extra_breaks_by_tool = {}
        seen = set()

        def charge(tool, cells):
            if tool is None:
                return
            bucket = cells_by_tool.setdefault(str(tool), [])
            for value in cells:
                cell = _cell(value)
                if cell in seen:
                    continue
                seen.add(cell)
                bucket.append(cell)

        for stage, plan in zip(self.spec.stages, self._stage_plan_objects):
            route_tool = (stage.tool if stage.tunnel_tool is None
                          else stage.tunnel_tool)
            for target in _targets(plan):
                charge(route_tool, target.route.mine_cells)
                charge(stage.tool, (target.cell,))
            charge(stage.tool or stage.tunnel_tool, (
                cell for operation in stage.after
                if isinstance(operation, WorkstationOperation)
                for cell in self.workstation_sites[operation.key].clear_cells
            ))
        if self._portal_return_certificate is not None:
            charge(self.spec.portal.return_tool, (
                _cell(cell) for cell in
                self._portal_return_certificate["mine_cells"]))
        if self.portal_plan is not None:
            charge(self.spec.portal.return_tool,
                   getattr(self.portal_plan, "clear_cells", ()))
        cast = self._bucket_cast_operation()
        if cast is not None:
            if self.cast_plan is not None:
                charge(cast.tool, self.cast_plan.access_clear_cells)
                charge(cast.tool, self.cast_plan.excavation_cells)
            if self.fluid_source_plan is not None:
                charge(cast.tool, (
                    cell
                    for step in self.fluid_source_plan.steps
                    for route in (step.outbound, step.return_route)
                    for cell in route.mine_cells
                ))
            # The contained mold deliberately reuses one result cell. Tool
            # durability therefore needs repeated-break accounting rather
            # than the unique-coordinate ledger used for planned terrain.
            extra_breaks_by_tool[cast.tool] = (
                extra_breaks_by_tool.get(cast.tool, 0) + cast.minimum)

        crafted = self._crafted_minimums()
        ledgers = []
        for tool in dict.fromkeys((*cells_by_tool, *extra_breaks_by_tool)):
            cells = cells_by_tool.get(tool, ())
            capacity = _tool_break_capacity(tool)
            if capacity is None:
                raise CaseRejected(
                    f"no durability capacity is registered for {tool!r}",
                    code="progression_tool_capacity_unknown")
            ledgers.append(ToolDurabilityLedger(
                tool=tool, capacity_per_copy=capacity,
                copies=int(crafted.get(tool, 0)),
                planned_cells=tuple(cells),
                extra_breaks=extra_breaks_by_tool.get(tool, 0)))
        return tuple(ledgers)

    def _lighting_budget(self):
        light_item = self.behaviors.light_item
        available = int(self._crafted_minimums().get(light_item, 0))
        edges = set()
        distance = 0.0
        for _stage, _kind, route, enabled in self._planned_routes():
            if not enabled:
                continue
            for edge in self._route_edges(route):
                if edge in edges:
                    continue
                edges.add(edge)
                distance += math.dist(*edge)
        begins = sum(
            isinstance(operation, BeginLightingOperation)
            for stage in self.spec.stages for operation in stage.after)
        interval = float(self.behaviors.light_every_blocks)
        lead = max(0.0, float(self.behaviors.light_grace_blocks))
        due_at = max(0.0, interval - lead)
        # No light is required for a partial tail below the controller's due
        # threshold. At each threshold crossing a verified placement resets
        # the interval, so floor division is the exact conservative count
        # before considering reductions from existing-light coverage.
        route_placements = (math.floor(distance / due_at + 1e-9)
                            if due_at > 0 else 0)
        required = begins + route_placements
        return {
            "item": light_item,
            "available": available,
            "required": int(required),
            "initial_placements": int(begins),
            "unique_route_edges": len(edges),
            "unique_route_distance": round(distance, 6),
            "interval": int(self.behaviors.light_every_blocks),
            "lead": lead,
            "due_at": due_at,
            "within_budget": available >= required,
        }

    def _build_cost_certificate(self):
        stages = []
        for stage, plan in zip(self.spec.stages, self._stage_plan_objects):
            targets = tuple(_targets(plan))
            route_stands = tuple(
                cell for target in targets for cell in target.route.stands)
            route_mines = tuple(dict.fromkeys(
                cell for target in targets
                for cell in target.route.mine_cells))
            workstation_keys = tuple(
                operation.key for operation in stage.after
                if isinstance(operation, WorkstationOperation))
            clear_cells = tuple(dict.fromkeys(
                cell for key in workstation_keys
                for cell in self.workstation_sites[key].clear_cells))
            stages.append({
                "resource": stage.name,
                "targets": len(targets),
                "route_stands": len(route_stands),
                "route_mine_cells": len(route_mines),
                "target_breaks": len(targets),
                "workstations": len(workstation_keys),
                "workstation_clear_breaks": len(clear_cells),
            })
        cast = self._bucket_cast_operation()
        cast_plan = self.cast_plan
        source_plan = self.fluid_source_plan
        casting = {
            "enabled": cast is not None,
            "method": self.spec.portal.method,
            "bucket_item": self.spec.portal.bucket_item,
            "casting_bucket_item": self.spec.portal.casting_bucket_item,
            "coolant_bucket_item": self.spec.portal.coolant_bucket_item,
            "reusable_buckets": 1 if cast is not None else 0,
            "casting_fluid": self.spec.portal.casting_fluid,
            "casting_sources": 0 if cast is None else cast.minimum,
            "casting_sources_unique": (
                self.spec.portal.unique_casting_sources),
            "coolant_fluid": self.spec.portal.coolant_fluid,
            "coolant_sources": 0 if cast is None else 1,
            "recover_coolant": self.spec.portal.recover_coolant,
            "source_meta": self.spec.portal.source_meta,
            "source_search_radius": self.spec.portal.fluid_search_radius,
            "result_block": self.spec.portal.cast_block,
            "result_count": 0 if cast is None else cast.minimum,
            "result_breaks": 0 if cast is None else cast.minimum,
            "result_tool": None if cast is None else cast.tool,
            "result_mining_ticks": (
                0 if cast is None else cast.mining_ticks),
            "backing_item": self.spec.portal.casting_backing_item,
            "reaction_wait_ticks": self.spec.portal.casting_wait_ticks,
            "cast_plan": (None if cast_plan is None else {
                "schema": cast_plan.to_dict()["schema"],
                "approach_stands": len(cast_plan.approach_stands),
                "access_clear_cells": len(cast_plan.access_clear_cells),
                "mold_excavation_cells": len(cast_plan.excavation_cells),
                "setup_placements": len(cast_plan.setup_placements),
                "result_cell": list(cast_plan.result_cell),
                "work_stance": list(cast_plan.work_stance),
            }),
            "source_plan": (None if source_plan is None else {
                "schema": source_plan.to_dict()["schema"],
                "steps": len(source_plan.steps),
                "newly_opened_cells": len(
                    source_plan.newly_opened_cells),
                "route_stands": sum(
                    len(route.stands)
                    for step in source_plan.steps
                    for route in (step.outbound, step.return_route)),
                "route_mine_cells": len({
                    cell
                    for step in source_plan.steps
                    for route in (step.outbound, step.return_route)
                    for cell in route.mine_cells
                }),
                "source_cells": [list(step.source_cell)
                                 for step in source_plan.steps],
            }),
        }
        portal = {
            "enabled": self.portal_plan is not None,
            "method": self.spec.portal.method,
            "casting": casting,
            "placements": (0 if self.portal_plan is None else
                           len(self.portal_plan.placements)),
            "frame_placements": (0 if self.portal_plan is None else sum(
                step.role != "scaffold"
                for step in self.portal_plan.placements)),
            "scaffold_placements": (0 if self.portal_plan is None else sum(
                step.role == "scaffold"
                for step in self.portal_plan.placements)),
            "clear_breaks": (0 if self.portal_plan is None else
                             len(getattr(
                                 self.portal_plan, "clear_cells", ()))),
            "return_stands": (0 if self._portal_return_certificate is None
                              else len(self._portal_return_certificate[
                                  "stands"])),
            "return_mine_cells": (
                0 if self._portal_return_certificate is None else
                len(self._portal_return_certificate["mine_cells"])),
        }
        tools = []
        for ledger in self._tool_ledgers():
            summary = ledger.to_dict()
            summary.pop("planned_cells")
            summary["planned_cells_sha256"] = hashlib.sha256(b"".join(
                int(coordinate).to_bytes(8, "big", signed=True)
                for cell in ledger.planned_cells for coordinate in cell
            )).hexdigest()
            tools.append(summary)
        return {
            "schema": "bedrock.progression_costs.v1",
            "stages": stages,
            "portal": portal,
            "lighting": self._lighting_budget(),
            "fluid_seals": self._fluid_seal_budgets(),
            "tools": tools,
            "totals": {
                "targets": sum(stage["targets"] for stage in stages),
                "route_stands": sum(
                    stage["route_stands"] for stage in stages),
                "route_mine_cells": sum(
                    stage["route_mine_cells"] for stage in stages),
                "workstations": sum(
                    stage["workstations"] for stage in stages),
            },
        }

    def cost_certificate(self):
        """JSON-ready costs for the currently certified world plan."""

        return self._costs

    def _require_tool_durability_budget(self):
        failures = tuple(
            item for item in self._costs["tools"]
            if not item["within_budget"])
        if failures:
            raise CaseRejected(
                "planned block breaks exceed crafted tool durability; "
                f"tools={failures!r}",
                code="progression_tool_durability_budget",
                details={"tools": failures})

    def _require_light_budget(self):
        lighting = self._costs["lighting"]
        if lighting["required"] and not lighting["within_budget"]:
            raise CaseRejected(
                "planned route needs more verified lights than the recipe "
                f"guarantees; lighting={lighting!r}",
                code="progression_light_budget", details=lighting)

    def _declare_break_cells(self, cells):
        """Extend the ledger only from a newly materialized generic plan."""

        self._allowed_break_cells = frozenset({
            *self._allowed_break_cells,
            *(_cell(cell) for cell in cells),
        })

    def _changed_world(self):
        return _ChangedWorld(
            self._world, opened=self._opened, removed=self._removed,
            placed=self._placed)

    def _live_world(self, env):
        """Snapshot overlaid only with journal-proven live mutations."""

        air = set()
        placed = {}
        for event in env.journal:
            if event.name not in ("block_broken", "block_placed"):
                continue
            cell = tuple(int(event[key]) for key in ("x", "y", "z"))
            if event.name == "block_broken":
                air.add(cell)
                placed.pop(cell, None)
            else:
                air.discard(cell)
                placed[cell] = int(event["block"])
        return _ChangedWorld(
            self._world, removed=air, placed=placed)

    def _live_surface_route(self, env, goal, label):
        start = self._current_cell(env)
        digest = hashlib.sha256(
            f"{self._decision_seed}:live:{label}".encode()).digest()
        try:
            return plan_surface_path(
                self._live_world(env), start, goal,
                hazard_clearance=self.spec.hazard_clearance,
                node_cap=self.spec.node_cap,
                decision_rng=random.Random(
                    int.from_bytes(digest[:8], "big")))
        except PlanRejected as exc:
            executor = getattr(self, "_executor", None)
            execution = None
            if executor is not None and not executor.done:
                execution = executor.targets[executor.target_index]
            execution_context = None if execution is None else {
                "target_index": executor.target_index,
                "target": execution.plan.cell,
                "mining_stance": execution.plan.mining_stance,
                "route_origin": execution.plan.route.stands[0],
            }
            raise CaseRejected(
                f"no safe live route from {start} to {goal} "
                f"for {label}: {exc}; pose="
                f"({float(env.obs['x']):.3f}, "
                f"{float(env.obs['y']):.3f}, "
                f"{float(env.obs['z']):.3f}); "
                f"execution={execution_context!r}",
                code="progression_live_route_missing") from exc

    def _recover_live_planning_start(self, env):
        """Micro-step from a collision-edge voxel onto a proved live stand.

        After pickup/jump settling, ``floor(position)`` can name a snapshot
        cell that is not itself a legal A* start while the player overlaps a
        safe adjacent stand. Only bridge one local cell here; once centered,
        ordinary live route planning regains full authority.
        """

        world = self._live_world(env)
        current = self._current_cell(env)
        if is_surface_stand(
                world, current,
                hazard_clearance=self.spec.hazard_clearance):
            return None
        position = (float(env.obs["x"]), float(env.obs["y"]),
                    float(env.obs["z"]))
        candidates = []
        for y in range(current[1] - 1, current[1] + 2):
            for x in range(current[0] - 1, current[0] + 2):
                for z in range(current[2] - 1, current[2] + 2):
                    candidate = (x, y, z)
                    if (candidate == current
                            or not is_surface_stand(
                                world, candidate,
                                hazard_clearance=(
                                    self.spec.hazard_clearance))
                            or not self.drop_is_safe(
                                position[1], candidate[1])):
                        continue
                    horizontal = math.hypot(
                        candidate[0] + 0.5 - position[0],
                        candidate[2] + 0.5 - position[2])
                    if horizontal <= 1.25:
                        candidates.append((
                            horizontal, abs(candidate[1] - position[1]),
                            candidate))
        if not candidates:
            return PolicyVeto(
                "progression_live_start_unrecoverable",
                f"live pose {tuple(round(value, 3) for value in position)} "
                f"has no adjacent safe planning stand around {current}",
                (position[0], position[2]), (current,))
        destination = min(candidates)[2]
        return self.face_and_move_to(
            env, destination[0] + 0.5, destination[2] + 0.5,
            tolerance=0.05, sequential_hop=True,
            step_up=destination[1] > position[1] + 0.55,
            allow_jump=destination[1] >= position[1] - 0.55,
            movement_cap_blocks=0.3)

    def recover_route_origin(self, env, goal):
        """Return to a plan anchor after journal-verified drop collection."""

        goal = _cell(goal)
        horizontal = math.hypot(
            float(env.obs["x"]) - (goal[0] + 0.5),
            float(env.obs["z"]) - (goal[2] + 0.5))
        vertical = float(env.obs["y"]) - goal[1]
        live_cell = self._current_cell(env)
        same_column = (live_cell[0], live_cell[2]) == (goal[0], goal[2])
        if same_column:
            if abs(vertical) <= 0.55 and horizontal <= 0.08:
                self._origin_settle_goal = None
                self._origin_settle_attempts = 0
                return None
            if 0.55 < vertical <= 2.25:
                # A pickup jump or short descent can leave the observation one
                # block above the certified origin while gravity is still
                # settling. That transient cell has no solid support and is
                # therefore an invalid surface-planner start. Wait without
                # adding momentum, but retain a strict bound in case live
                # geometry really does prevent the expected landing.
                if self._origin_settle_goal != goal:
                    self._origin_settle_goal = goal
                    self._origin_settle_attempts = 0
                if self._origin_settle_attempts >= 8:
                    return PolicyVeto(
                        "progression_origin_settle_exhausted",
                        f"player did not settle from y={float(env.obs['y']):.3f} "
                        f"onto route origin {goal}",
                        (goal[0] + 0.5, goal[2] + 0.5), (goal,))
                self._origin_settle_attempts += 1
                if self._origin_settle_attempts <= 2:
                    return [{"action": "wait", "ticks": 2}]
                # If two physics intervals leave the player stably one block
                # high, they are perched on an adjacent block edge rather
                # than airborne. Center over the certified origin so gravity
                # can complete the descent.
                result = self.face_and_move_to(
                    env, goal[0] + 0.5, goal[2] + 0.5,
                    tolerance=0.05, allow_jump=False,
                    movement_cap_blocks=0.3)
                return (result if result is not None
                        else [{"action": "wait", "ticks": 2}])
            if abs(vertical) <= 0.55:
                # The player has landed in the certified origin cell but can
                # still be outside its center envelope after a pickup stride.
                # Recenter inside that same proved body column rather than ask
                # A* for a meaningless route from the cell to itself.
                if horizontal <= 0.2:
                    result = self.recenter_interaction_stance(
                        env, goal[0] + 0.5, goal[2] + 0.5,
                        tolerance=0.05)
                else:
                    result = self.face_and_move_to(
                        env, goal[0] + 0.5, goal[2] + 0.5,
                        tolerance=0.05, allow_jump=False,
                        movement_cap_blocks=0.3)
                if result is not None:
                    return result
        self._origin_settle_goal = None
        self._origin_settle_attempts = 0
        # Pickup motion can stop a few millimeters across a voxel boundary:
        # floor(position) then names an adjacent snapshot cell that is not a
        # valid stand even though the player is still within one cautious
        # half-step of the certified origin. Recenter directly into that
        # proved column before asking A* to validate a misleading start cell.
        if (abs(vertical) <= 0.55
                and horizontal <= 0.65
                and max(abs(live_cell[0] - goal[0]),
                        abs(live_cell[2] - goal[2])) <= 1):
            result = self.face_and_move_to(
                env, goal[0] + 0.5, goal[2] + 0.5,
                tolerance=0.05, allow_jump=False,
                movement_cap_blocks=0.3)
            if result is not None:
                return result
        route = self._live_surface_route(env, goal, "mining_route_origin")
        if len(route) < 2:
            return PolicyVeto(
                "progression_origin_recovery_empty",
                f"live origin recovery did not reach {goal}",
                (goal[0] + 0.5, goal[2] + 0.5), (goal,))
        current, destination = route[0], route[1]
        if not self.drop_is_safe(current[1], destination[1]):
            return PolicyVeto(
                "progression_origin_recovery_drop_rejected",
                "live origin recovery exceeds configured safe descent",
                (goal[0] + 0.5, goal[2] + 0.5),
                (current, destination))
        return self.face_and_move_to(
            env, destination[0] + 0.5, destination[2] + 0.5,
            tolerance=(0.05 if destination[1] != current[1] else 0.4),
            sequential_hop=True,
            step_up=destination[1] > current[1],
            allow_jump=destination[1] >= current[1])

    def drop_pickup_site(self, env, plan):
        """Return the next proved stand approaching a mined item's column.

        A target above the mining stance leaves its solid support in place.
        Walking at that target's x/z without jumping only pushes into the
        ledge, while jumping during drop collection can leave a proved route.
        Prefer a same-level safe stand beside the item column and route to it
        one certified cell at a time.  The old sneak-to-edge fallback remains
        for geometry where no such stand is snapshot-proved.
        """

        target = _cell(plan.cell)
        current = self._current_cell(env)
        world = self._live_world(env)
        direct_stand = is_surface_stand(
            world, target,
            hazard_clearance=self.spec.hazard_clearance)
        body_safe_stand = direct_stand or is_surface_stand(
            world, target, hazard_clearance=0)
        pickup_trace = {
            "target": target, "current": current,
            "mining_stance": getattr(plan, "mining_stance", None),
            "direct_stand": direct_stand,
            "body_safe_stand": body_safe_stand,
            "supports": [],
        }
        self._drop_pickup_trace = pickup_trace
        if self._drop_pickup_history_target != target:
            self._drop_pickup_history_target = target
            self._drop_pickup_history = []

        def approach(point, sneak, *, step_y):
            # The controller's generic stuck recovery needs the next certified
            # route edge, not the mined block's elevation. Permit an ordinary
            # jump on level/ascending approach steps; never jump during a
            # descending or explicit sneak-to-edge collection move.
            self._drop_pickup_allow_jump = bool(
                not sneak and int(step_y) >= current[1])
            pickup_trace["next_step_y"] = int(step_y)
            pickup_trace["allow_jump"] = self._drop_pickup_allow_jump
            pickup_trace["point"] = tuple(float(value) for value in point)
            pickup_trace["sneak"] = bool(sneak)
            self._drop_pickup_history.append(dict(pickup_trace))
            del self._drop_pickup_history[:-16]
            return tuple(float(value) for value in point), bool(sneak)

        if (target[1] <= current[1]
                and self.drop_is_safe(float(env.obs["y"]), target[1])
                and body_safe_stand):
            # A removed same-level or lower block can itself become the
            # safest collection stand. A strict planning buffer can reject
            # it merely for an adjacent hazard even though the occupied body
            # is dry; zero-buffer validity plus movement's live 0.55-block
            # hazard path check retains the collision invariant while
            # allowing the player to enter the exact drop column.
            return approach(
                (target[0] + 0.5, target[2] + 0.5), False,
                step_y=target[1])
        # The immutable planner already bounds resource-drop fall, but the
        # live pickup controller still needs the landing cell rather than the
        # removed block's old y. Prove the first support below this column,
        # require a dry two-block landing within the configured descent, then
        # walk into the column normally so gravity reaches the item. Sneak
        # would intentionally stop at the upper ledge.
        landing = None
        # ``max_drop_blocks`` bounds how far the *player* may descend in one
        # transition; it is not an item-entity fall bound.  In particular,
        # mining a tree from the ground can remove an upper log after the
        # lower trunk is already gone, leaving several air cells under the
        # drop.  The immutable plan proves that its eventual landing is at
        # most one block from ``mining_stance``.  Scan exactly that certified
        # vertical envelope instead of truncating the search at the player's
        # one-step descent limit.
        mining_stance = _cell(getattr(plan, "mining_stance", current))
        minimum_support_y = mining_stance[1] - 2
        maximum_offset = max(
            1, target[1] - minimum_support_y)
        for offset in range(1, maximum_offset + 1):
            support = (target[0], target[1] - offset, target[2])
            block = world.block(support)
            shown = getattr(block, "block_id", block)
            try:
                shown = int(shown)
            except (TypeError, ValueError):
                shown = type(block).__name__
            pickup_trace["supports"].append((support, shown))
            if (DEFAULT_BLOCK_RULES.is_fluid(block)
                    or DEFAULT_BLOCK_RULES.is_hazard(block)):
                break
            if not DEFAULT_BLOCK_RULES.is_passable(block):
                landing = (support[0], support[1] + 1, support[2])
                break
        pickup_trace["landing"] = landing
        pickup_trace["landing_drop_safe"] = (
            landing is not None
            and self.drop_is_safe(float(env.obs["y"]), landing[1]))
        pickup_trace["landing_surface"] = (
            landing is not None
            and is_surface_stand(world, landing, hazard_clearance=0))
        if (landing is not None
                and landing[1] < current[1]
                and pickup_trace["landing_drop_safe"]
                and pickup_trace["landing_surface"]):
            return approach(
                (target[0] + 0.5, target[2] + 0.5), False,
                step_y=landing[1])
        # A broken lower trunk can leave its old column blocked by the next
        # log while the item falls onto ordinary ground below. Searching only
        # at the player's current tree-canopy level finds no valid neighbor
        # and turns the old fallback into a blind walk toward an unstandable
        # column. Include the already-proved landing elevation and route to a
        # dry adjacent stand one certified step at a time.
        candidate_levels = [current[1]]
        if landing is not None and landing[1] not in candidate_levels:
            candidate_levels.append(landing[1])
        pickup_trace["candidate_levels"] = tuple(candidate_levels)
        candidates = []
        for candidate_y in candidate_levels:
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    candidate = (
                        target[0] + dx, candidate_y, target[2] + dz)
                    if not is_surface_stand(
                            world, candidate,
                            hazard_clearance=self.spec.hazard_clearance):
                        continue
                    candidates.append((
                        math.hypot(dx, dz),
                        math.dist(current, candidate),
                        candidate,
                    ))
        for _drop_distance, _travel, candidate in sorted(candidates):
            try:
                route = plan_surface_path(
                    world, current, candidate,
                    hazard_clearance=self.spec.hazard_clearance,
                    node_cap=self.spec.node_cap)
            except PlanRejected:
                continue
            if len(route) == 1 and _drop_distance > 0.01:
                # Do not issue an edge-sneak from a support whose boundary is
                # still more than pickup radius from the item column. It will
                # correctly stop at the ledge forever. Prefer another proved
                # candidate; if none exists, the final explicit fallback
                # below can reject this geometry instead of pretending it was
                # collected.
                edge_distance = max(0.0, _drop_distance - 0.5)
                if edge_distance > 0.9:
                    continue
                return approach(
                    (target[0] + 0.5, target[2] + 0.5), True,
                    step_y=current[1])
            destination = route[1] if len(route) > 1 else route[0]
            return approach(
                (destination[0] + 0.5, destination[2] + 0.5), False,
                step_y=destination[1])
        return approach(
            (target[0] + 0.5, target[2] + 0.5), True,
            step_y=current[1])

    def _portal_return_plan(self, start):
        """Return over certified breadcrumbs, with bounded search fallback."""

        goal = self.portal_plan.approach_stands[0]
        world = self._changed_world()
        breadcrumb = self._portal_breadcrumb_return(world, start, goal)
        if breadcrumb is not None:
            return breadcrumb, frozenset()
        digest = hashlib.sha256(
            f"{self._decision_seed}:portal_return".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        clearance = self.spec.hazard_clearance
        # The last obsidian stance may be dry but directly beside the water
        # cap that made its removal safe. Permit adjacency for the return
        # route only when that exact starting body/head/support is itself a
        # valid dry tunnel stance. Zero clearance never permits occupying a
        # fluid or hazard; it only stops rejecting a neighboring voxel.
        if not is_excavation_stand(
                world, start, hazard_clearance=clearance):
            if is_excavation_stand(world, start, hazard_clearance=0):
                clearance = 0
        route = None
        try:
            route = plan_surface_path(
                world, start, goal,
                hazard_clearance=clearance,
                node_cap=self.spec.node_cap, decision_rng=rng)
            mine_cells = frozenset()
        except PlanRejected:
            try:
                excavation = plan_staircase_tunnel(
                    world, start, goal,
                    hazard_clearance=clearance,
                    node_cap=self.spec.node_cap, decision_rng=rng)
            except PlanRejected as exc:
                raise CaseRejected(
                    f"no safe live return to portal site: {exc}",
                    code="progression_portal_return_missing") from exc
            route = excavation.stands
            mine_cells = frozenset(excavation.mine_cells)
            overlap = mine_cells.intersection(self._portal_cells)
            if overlap:
                raise CaseRejected(
                    "fallback return would excavate protected portal "
                    f"geometry {sorted(overlap)[:4]}",
                    code="progression_portal_return_conflict")
        return tuple(route), mine_cells

    def _portal_breadcrumb_return(self, world, start, goal):
        """Reverse the loop-spliced spine when it survives final state."""

        breadcrumb = getattr(self, "_return_breadcrumb", None)
        if breadcrumb is None or breadcrumb.broken:
            return None
        forward = breadcrumb.stands
        if not forward or forward[0] != goal or forward[-1] != start:
            return None

        route = tuple(reversed(forward))
        for cell in route:
            if not is_surface_stand(world, cell, hazard_clearance=0):
                return None
        for left, right in zip(route, route[1:]):
            if (abs(left[0] - right[0]) + abs(left[2] - right[2]) != 1
                    or abs(left[1] - right[1]) > 1):
                return None
            transition = None
            if right[1] < left[1]:
                transition = (right[0], right[1] + 2, right[2])
            elif right[1] > left[1]:
                transition = (left[0], left[1] + 2, left[2])
            if transition is not None:
                if not world.in_bounds(transition):
                    return None
                block = world.block(transition)
                if (not DEFAULT_BLOCK_RULES.is_passable(block)
                        or DEFAULT_BLOCK_RULES.is_fluid(block)
                        or DEFAULT_BLOCK_RULES.is_hazard(block)):
                    return None
        return route

    def _plan_portal_return(self, env):
        """Use the preflight return certificate from its exact live origin."""

        if self.spec.portal.strategy != "initial":
            self._route_mine_cells = frozenset()
            self._route_clear_attempts = {}
            self._start_route(
                self.portal_plan.approach_stands, "portal_approach")
            return
        current = self._current_cell(env)
        certificate = getattr(self, "_portal_return_certificate", None)
        if (certificate is not None
                and current == _cell(certificate["start"])):
            route = tuple(_cell(cell) for cell in certificate["stands"])
            mine_cells = frozenset(
                _cell(cell) for cell in certificate["mine_cells"])
        else:
            route, mine_cells = self._portal_return_plan(current)
        self._declare_break_cells(mine_cells)
        self._route_mine_cells = mine_cells
        self._route_clear_attempts = {}
        self._start_route(route, "return_to_surface")

    def _clear_return_leg(self, env):
        if self._route_index >= len(self._route):
            return None
        destination = self._route[self._route_index]
        previous = self._route[self._route_index - 1]
        body = (
            destination,
            (destination[0], destination[1] + 1, destination[2]),
        )
        if destination[1] < previous[1]:
            body += ((destination[0], destination[1] + 2,
                      destination[2]),)
        elif destination[1] > previous[1]:
            body += ((previous[0], previous[1] + 2,
                      previous[2]),)
        for cell in sorted(body, key=lambda value: -value[1]):
            if cell not in self._route_mine_cells:
                continue
            if self.events(env, "block_broken", x=cell[0], y=cell[1],
                           z=cell[2]):
                continue
            attempts = self._route_clear_attempts.get(cell, 0)
            if attempts >= 8:
                return PolicyVeto(
                    "progression_portal_return_clear_exhausted",
                    f"return tunnel cell {cell} did not break",
                    (cell[0] + 0.5, cell[2] + 0.5), (cell,))
            result = self.mine_exact(
                env, cell, tool=self.spec.portal.return_tool,
                ticks=self.spec.portal.return_ticks,
                stand=(previous[0] + 0.5, previous[2] + 0.5))
            if result:
                self._route_clear_attempts[cell] = attempts + 1
            return result
        return None

    @staticmethod
    def _health(env):
        if "health" in env.obs:
            return float(env.obs["health"])
        for event in reversed(env.journal):
            if event.name == "vitals_changed":
                return float(event["health"])
        return None

    def _available_initial_light_sites(self, support_key):
        site = self.workstation_sites[support_key]
        occupied = {other.cell for other in self.workstation_sites.values()}
        protected = set(self.route_light_protected_cells())
        return tuple(
            cell for cell in site.light_cells
            if cell not in occupied and cell not in protected
            and cell != site.stand)

    def _guard(self, env):
        if bool(env.obs.get("dead", False)):
            return PolicyVeto(
                "player_died", "player died during survival progression",
                (float(env.obs["x"]), float(env.obs["z"])))
        health = self._health(env)
        if (health is not None and self._baseline_health is not None
                and health < self._baseline_health
                - self.spec.health_loss_tolerance):
            return PolicyVeto(
                "health_lost",
                f"health fell from {self._baseline_health:g} to {health:g}",
                (float(env.obs["x"]), float(env.obs["z"])))
        for event in env.journal[self._guard_event_index:]:
            if event.name == "block_broken":
                cell = tuple(int(event[key]) for key in ("x", "y", "z"))
                if cell not in self._allowed_break_cells:
                    return PolicyVeto(
                        "unplanned_block_removed",
                        "engine journal records collateral block removal at "
                        f"{cell}; it is not in the episode-global declared "
                        "break ledger",
                        (float(env.obs["x"]), float(env.obs["z"])),
                        (cell,))
            if (self.spec.veto_fluids
                    and event.name == "player_fluid_state"
                    and int(event.get("flags", 0)) & FLUID_BODY
                    and DEFAULT_BLOCK_RULES.is_fluid(
                        int(event.get("block", 0)))):
                cell = tuple(int(event[key]) for key in ("x", "y", "z"))
                return PolicyVeto(
                    "fluid_contact",
                    "player collision body entered a fluid",
                    (float(env.obs["x"]), float(env.obs["z"])),
                    (cell,))
        if self.spec.veto_fluids:
            # The relevant transition can predate a milestone/executor
            # boundary.  The newest fluid record is the authoritative live
            # state; an exit record has flags=0 and clears an older contact.
            for event in reversed(env.journal):
                if event.name != "player_fluid_state":
                    continue
                if (int(event.get("flags", 0)) & FLUID_BODY
                        and DEFAULT_BLOCK_RULES.is_fluid(
                            int(event.get("block", 0)))):
                    cell = tuple(int(event[key])
                                 for key in ("x", "y", "z"))
                    return PolicyVeto(
                        "fluid_contact",
                        "current player collision body is in a fluid",
                        (float(env.obs["x"]), float(env.obs["z"])),
                        (cell,))
                break
        self._guard_event_index = len(env.journal)
        return None

    def route_light_protected_cells(self):
        cells = (set() if self.plan is None else
                 {target.cell for target in _targets(self.plan)})
        cells.update(site.cell for site in self.workstation_sites.values())
        if self.portal_plan is not None:
            cells.update(self._portal_cells)
        if self.cast_plan is not None:
            cells.update(self.cast_plan.protected_cells)
        if self.fluid_source_plan is not None:
            for step in self.fluid_source_plan.steps:
                cells.add(step.source_cell)
                cells.update(getattr(step, "pickup_ray_cells", ()))
                for route in (step.outbound, step.return_route):
                    cells.update(route.reserved_cells)
                    cells.update(route.support_cells)
                    cells.update(route.mine_cells)
        # The active workstation support is deliberately allowed: the first
        # torch is attached to it immediately after crafting.
        if self._lighting_initial_site is not None:
            cells.discard(self._lighting_initial_site[1])
        return cells

    def route_light_forbidden_support_cells(self):
        """Future-mutated cells, excluding immutable route wall/supports."""

        cells = (set() if self.plan is None else
                 {target.cell for target in _targets(self.plan)})
        if self.portal_plan is not None:
            cells.update(self._portal_cells)
        if self.cast_plan is not None:
            cells.update(self.cast_plan.protected_cells)
        if self.fluid_source_plan is not None:
            for step in self.fluid_source_plan.steps:
                cells.add(step.source_cell)
                for route in (step.outbound, step.return_route):
                    cells.update(route.mine_cells)
        if self._lighting_initial_site is not None:
            cells.discard(self._lighting_initial_site[1])
        return cells

    def route_light_forbidden_destination_cells(self):
        """Future interaction/mutation cells, not ordinary route air."""

        cells = (set() if self.plan is None else
                 {target.cell for target in _targets(self.plan)})
        cells.update(site.cell for site in self.workstation_sites.values())
        if self.portal_plan is not None:
            cells.update(self._portal_cells)
        if self.cast_plan is not None:
            cells.update(self.cast_plan.protected_cells)
        if self.fluid_source_plan is not None:
            for step in self.fluid_source_plan.steps:
                cells.add(step.source_cell)
                cells.update(getattr(step, "pickup_ray_cells", ()))
                for route in (step.outbound, step.return_route):
                    cells.update(route.mine_cells)
        return cells

    def route_light_initial_site(self, env):
        del env
        return self._lighting_initial_site

    def route_light_sneak_supports(self):
        return tuple(site.cell for site in self.workstation_sites.values())

    def route_light_hotbar_swaps(self, env, item):
        """Keep mandatory lighting live when torches move to the backpack."""

        return self._hotbar_swaps(env, wanted=(item,))

    def choose_route_light_site(self, env):
        """Choose a snapshot-certified wall behind the live route.

        The model-facing nearby-block summary is intentionally bounded and
        may omit an ordinary tunnel wall. Progression has a complete
        authoritative snapshot overlay, so use it to select geometry while
        retaining the public click ray as the irreversible-action gate.
        """

        if not self.torch_events_seen:
            initial = self.route_light_initial_site(env)
            if initial is not None:
                return initial
        world = self._live_world(env)
        current = self._current_cell(env)
        eye = (float(env.obs["x"]), float(env.obs["y"]) + 1.62,
               float(env.obs["z"]))
        protected = {tuple(map(int, cell)) for cell in
                     self.route_light_forbidden_destination_cells()}
        forbidden_supports = {tuple(map(int, cell)) for cell in
                              self.route_light_forbidden_support_cells()}
        # A tunnel route reserves its body before execution, so those cells
        # appear in the objective-wide protection set. Once an exact journal
        # event proves one was excavated and traversed, a passable torch may
        # safely occupy that old body cell (especially as an ordinary floor
        # torch in a staircase). Keep actual structure, workstation, fluid
        # source, and interaction-ray geometry immutable even if another
        # operation happened to remove a block there.
        released = {
            tuple(int(event[key]) for key in ("x", "y", "z"))
            for event in env.journal
            if event.name == "block_broken"
        }
        immutable = set(site.cell for site in self.workstation_sites.values())
        if self.portal_plan is not None:
            immutable.update(self._portal_cells)
        if self.cast_plan is not None:
            immutable.update(self.cast_plan.protected_cells)
        if self.fluid_source_plan is not None:
            for step in self.fluid_source_plan.steps:
                immutable.add(step.source_cell)
                immutable.update(getattr(step, "pickup_ray_cells", ()))
        protected.difference_update(released - immutable)
        minimum_spacing = max(
            2.0, float(self.behaviors.light_every_blocks) / 2.0)
        candidates = []
        rejected = {
            "height": 0, "occupied": 0, "protected": 0, "spacing": 0,
            "support": 0, "ray": 0, "reach": 0,
        }
        for route_index, stand in enumerate(reversed(
                self.recent_route_cells)):
            if abs(stand[1] - current[1]) > 1:
                rejected["height"] += 1
                continue
            levels = (stand[1] + 1,) if stand == current else (
                stand[1] + 1, stand[1])
            for level_priority, level in enumerate(levels):
                destination = (stand[0], level, stand[2])
                if (not world.in_bounds(destination)
                        or not DEFAULT_BLOCK_RULES.is_passable(
                            world.block(destination))):
                    rejected["occupied"] += 1
                    continue
                if (destination in protected
                        or destination in self.shown_lights):
                    rejected["protected"] += 1
                    continue
                too_close = any(
                    math.dist(destination, shown) < minimum_spacing
                    for shown in self.shown_lights)
                if too_close:
                    rejected["spacing"] += 1
                distance = math.hypot(
                    destination[0] + 0.5 - float(env.obs["x"]),
                    destination[2] + 0.5 - float(env.obs["z"]))
                for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    support = (destination[0] + dx, destination[1],
                               destination[2] + dz)
                    if (support in forbidden_supports
                            or support in self.rejected_light_supports):
                        rejected["protected"] += 1
                        continue
                    if not world.in_bounds(support):
                        # A route may legitimately run along the captured
                        # snapshot perimeter.  The adjacent wall candidate
                        # is then unknown, not a planner crash and certainly
                        # not safe placement geometry.
                        rejected["support"] += 1
                        continue
                    support_block = world.block(support)
                    if (not DEFAULT_BLOCK_RULES.is_solid(support_block)
                            or DEFAULT_BLOCK_RULES.is_hazard(support_block)
                            or DEFAULT_BLOCK_RULES.is_fluid(support_block)):
                        rejected["support"] += 1
                        continue
                    physical = placement_face_point(
                        support, destination, observer=eye)
                    if math.dist(eye, physical) > 4.0:
                        rejected["reach"] += 1
                        continue
                    if not clear_placement_ray(
                            world, current, destination, support,
                            max_reach=4.0):
                        rejected["ray"] += 1
                        continue
                    point = tuple(value - 0.5 for value in physical)
                    candidates.append((
                        ((1 if too_close else 0), 0, level_priority,
                         abs(distance - 1.5), route_index, support),
                        destination, support, point, "wall"))
            # Prefer a wall when one exists, then use an already traversed
            # floor cell. Torches are passable in the pinned engine, and the
            # separate destination contract above keeps them out of future
            # mining, fluid, workstation, and portal interaction cells.
            destination = stand
            support = (stand[0], stand[1] - 1, stand[2])
            distance = math.hypot(
                destination[0] + 0.5 - float(env.obs["x"]),
                destination[2] + 0.5 - float(env.obs["z"]))
            too_close = any(
                math.dist(destination, shown) < minimum_spacing
                for shown in self.shown_lights)
            if (stand != current
                    and world.in_bounds(destination)
                    and world.in_bounds(support)
                    and DEFAULT_BLOCK_RULES.is_passable(
                        world.block(destination))
                    and destination not in protected
                    and destination not in self.shown_lights
                    and support not in forbidden_supports
                    and support not in self.rejected_light_supports):
                support_block = world.block(support)
                if (DEFAULT_BLOCK_RULES.is_solid(support_block)
                        and not DEFAULT_BLOCK_RULES.is_hazard(support_block)
                        and not DEFAULT_BLOCK_RULES.is_fluid(support_block)):
                    physical = placement_face_point(
                        support, destination, observer=eye)
                    if (math.dist(eye, physical) <= 4.0
                            and clear_placement_ray(
                                world, current, destination, support,
                                max_reach=4.0)):
                        point = tuple(value - 0.5 for value in physical)
                        candidates.append((
                            ((1 if too_close else 0), 1, 0,
                             abs(distance - 1.5), route_index, support),
                            destination, support, point, "floor"))
        self._route_light_candidate_debug = {
            "considered": len(self.recent_route_cells),
            "accepted": len(candidates),
            "rejected": rejected,
            "current": current,
        }
        if candidates:
            _score, destination, support, point, kind = min(candidates)
            return destination, support, point, kind
        return super().choose_route_light_site(env)

    def route_light_debug(self, env):
        slot = int(env.obs.get("hotbar_sel", 0))
        hotbar = tuple(int(value) for value in env.obs.get(
            "hotbar_ids", ()))
        return {
            "distance": round(float(self.lighting_distance), 3),
            "total": round(float(self.route_distance_total), 3),
            "every": int(self.behaviors.light_every_blocks),
            "lead": float(self.behaviors.light_grace_blocks),
            "overshoot": float(self.behaviors.light_overshoot_blocks),
            "pending": self.pending_light,
            "recent_route": tuple(self.recent_route_cells[-8:]),
            "rejected_supports": tuple(sorted(
                self.rejected_light_supports)[-8:]),
            "torch_count": self._route_light_item_count(
                env, self.behaviors.light_item),
            "selected": (slot, hotbar[slot]
                         if 0 <= slot < len(hotbar) else None),
            "click": self._route_light_click(env),
            "candidates": getattr(
                self, "_route_light_candidate_debug", None),
        }

    def provenance(self):
        return {
            "schema": "bedrock.survival_progression.v2",
            "spec": self.spec.to_dict(),
            # This list is append-only and intentionally shared with the
            # trajectory recorder's shallow provenance copy. Each JIT plan
            # therefore remains visible in the final saved artifact.
            "stage_plans": self._stage_plans,
            "workstations": self._provenance_workstations,
            "structure_plan": (None if self.portal_plan is None
                               else self.portal_plan.to_dict()),
            "fluid_cast_plan": (None if self.cast_plan is None
                                else self.cast_plan.to_dict()),
            "fluid_source_plan": (
                None if self.fluid_source_plan is None
                else self.fluid_source_plan.to_dict()),
            "portal_return_certificate": self._portal_return_certificate,
            "preflight_timing_seconds": dict(
                self._preflight_timing_seconds),
            "teacher_knowledge": {
                "student_visible": False,
                "source": "initial_state_snapshot",
                "snapshot_sha256": (None if self._world is None
                                    else self._world.snapshot_sha256),
            },
        }

    def demonstration_grounding(self):
        return {
            "eligible": False,
            "target_selection": "privileged initial-state voxel snapshot",
            "execution": "public observation and journal events",
            "reason": (
                "underground route choices are not recoverable from the "
                "model-visible frame and observation text"),
        }

    def turn_grounding(self):
        """Declare only decisions recoverable from the model's inputs."""
        stage = str(self.stage)
        grounded = stage in {"mine:logs", "mine:coal"}
        return {
            "eligible": grounded,
            "target_selection": (
                "model-visible bearing and frame" if grounded
                else "privileged initial-state voxel snapshot"),
            "execution": "public observation and journal events",
            "reason": (None if grounded else
                       "this stage's target or route is not exposed by the "
                       "model-visible observation"),
        }

    def diagnostics(self, env):
        plan_execution = (None if self._executor is None else
                          self._executor.diagnostics())
        operation = None
        if 0 <= self._stage_index < len(self.spec.stages):
            operations = self.spec.stages[self._stage_index].after
            if 0 <= self._operation_index < len(operations):
                operation = operations[self._operation_index].to_dict()
        route_window = self._route[max(0, self._route_index - 2):
                                   self._route_index + 3]
        hotbar_ids = tuple(int(value) for value in env.obs.get(
            "hotbar_ids", ()))
        hotbar_counts = tuple(int(value) for value in env.obs.get(
            "hotbar_counts", ()))
        hotbar_meta = tuple(int(value) for value in env.obs.get(
            "hotbar_meta", (0,) * len(hotbar_ids)))
        inventory_ids = tuple(int(value) for value in env.obs.get(
            "inventory_ids", ()))
        inventory_counts = tuple(int(value) for value in env.obs.get(
            "inventory_counts", ()))
        inventory_meta = tuple(int(value) for value in env.obs.get(
            "inventory_meta", (0,) * len(inventory_ids)))
        return {
            "stage": self.stage,
            "resource_index": self._stage_index,
            "operation_index": self._operation_index,
            "route_index": self._route_index,
            "placement_index": self._placement_index,
            "position": [round(float(env.obs[key]), 3)
                         for key in ("x", "y", "z")],
            "inventory_ui": {
                "container": int(env.obs.get("container", 0)),
                "opened": self.current_container_event(env),
                "cursor": self.cursor_state(env),
                "hotbar": tuple(
                    (slot, item, hotbar_counts[slot], hotbar_meta[slot])
                    for slot, item in enumerate(hotbar_ids)
                    if item or hotbar_counts[slot]),
                "backpack": tuple(
                    (slot, inventory_ids[slot], inventory_counts[slot],
                     inventory_meta[slot])
                    for slot in range(9, len(inventory_ids))
                    if inventory_ids[slot] or inventory_counts[slot]),
                "counts": {
                    str(name): int(count)
                    for name, count in env.obs.get(
                        "inv_counts", {}).items() if int(count)
                },
            },
            "dimension": int(env.obs.get("dimension", 0)),
            "health": self._health(env),
            "lighting_active": self._lighting_active,
            "lighting_distance": round(self.lighting_distance, 3),
            "fluid_cast": {
                "planned": self.cast_plan is not None,
                "sources_planned": (
                    0 if self.fluid_source_plan is None
                    else len(self.fluid_source_plan.steps)),
                "phase": (None if self._cast_executor is None
                          else self._cast_executor.phase),
                "cast_index": (None if self._cast_executor is None
                               else self._cast_executor._cast_index),
                "execution": (
                    None if self._cast_executor is None
                    else self._cast_executor.diagnostics()),
            },
            "verified_light_odometer": [round(value, 3) for value in
                                         self.verified_light_odometer],
            "plan_execution": plan_execution,
            "operation": operation,
            "live_route": {
                "length": len(self._route),
                "index": self._route_index,
                "start": (None if not self._route else self._route[0]),
                "end": (None if not self._route else self._route[-1]),
                "window": route_window,
                "attempts": self._route_attempts,
            },
            "workstation_route_keys": tuple(sorted(
                self._workstation_route_keys)),
            "active_workstation": (
                None if not isinstance(operation, Mapping)
                or operation.get("type") != "workstation"
                else self.workstation_sites[operation["key"]].to_dict()),
        }

    def _stage_executor(self, env, stage):
        targets = _resource_targets(self.plan, stage.name)
        executions = tuple(MiningExecution(
            target, block=stage.blocks, tool=stage.tool,
            drop_item=stage.drop_item, drop_count=1,
            fluid_seal_item=stage.fluid_seal_item,
            block_metadata=stage.metadata)
            for target in targets)
        hooks = (_LightingHooks(self) if self._lighting_active else None)

        def recover_item(current_env, item, minimum=1):
            return self.recover_hotbar(
                current_env, item,
                replace=self._hotbar_swaps(current_env, item),
                minimum=minimum)

        executor = PlanExecutor(
            self, executions,
            tunnel_tool=(stage.tool if stage.tunnel_tool is None
                         else stage.tunnel_tool),
            mining_ticks=stage.mining_ticks,
            tunnel_ticks=stage.tunnel_ticks,
            max_pickup_attempts=self.behaviors.pickup_attempts,
            health_loss_tolerance=self.spec.health_loss_tolerance,
            veto_fluids=self.spec.veto_fluids,
            lighting=hooks, lighting_immediate=False,
            lighting_route_kinds=("tunnel",),
            item_recovery=recover_item,
            break_declarer=self._declare_break_cells)
        executor.reset(env)
        return executor

    def _initial_light_site(self, support_key, env):
        site = self.workstation_sites[support_key]
        for cell in self._available_initial_light_sites(support_key):
            observed = self.observed_block(env, cell)
            if observed not in (None, 0):
                continue
            dx, dz = cell[0] - site.cell[0], cell[2] - site.cell[2]
            point = (round(site.cell[0] + dx * 0.49, 6),
                     site.cell[1],
                     round(site.cell[2] + dz * 0.49, 6))
            return cell, site.cell, point, "wall"
        # Tight two-block tunnels may not leave a free side around the table.
        # Seed the generic wall-light behavior with the route just traversed;
        # it will select and live-prove the nearest wall behind the player.
        self.recent_route_cells = [
            cell for target in _targets(self.plan)
            for cell in target.route.stands][-48:]
        return None

    def _initial_light_vantage_route(self, env, light_site):
        """Find a safe stance that sees the selected workstation face.

        A workstation's interaction stance is necessarily adjacent to the
        block, and may be around a corner from the only free torch face.
        Clicking from there hits the nearer top/side face.  Route briefly to
        the outside of the intended destination, place against the exact
        face, then return before the next preplanned resource leg starts.
        """

        destination, support, _point, _kind = light_site
        delta = tuple(destination[index] - support[index]
                      for index in range(3))
        if delta[1] != 0 or abs(delta[0]) + abs(delta[2]) != 1:
            return ()
        world = self._live_world(env)
        start = self._current_cell(env)
        # The public angle helper interprets these as block coordinates and
        # adds 0.5 on every axis. Spatial ray proofs use physical points.
        point = tuple(float(value) + 0.5 for value in light_site[2])
        candidates = []
        # Search a small, fully proved local viewing half-space instead of
        # assuming level ground extends one more block beyond the free torch
        # cell. Natural slopes and tunnel mouths frequently put that cell at
        # an edge while leaving a reachable stance one level above/below.
        for radius in range(1, 5):
            for y in range(support[1] - 1, support[1] + 2):
                for x in range(support[0] - radius,
                               support[0] + radius + 1):
                    for z in range(support[2] - radius,
                                   support[2] + radius + 1):
                        if max(abs(x - support[0]),
                               abs(z - support[2])) != radius:
                            continue
                        candidate = (x, y, z)
                        outward = ((candidate[0] + 0.5
                                    - (support[0] + 0.5)) * delta[0]
                                   + (candidate[2] + 0.5
                                      - (support[2] + 0.5)) * delta[2])
                        eye = (candidate[0] + 0.5, candidate[1] + 1.62,
                               candidate[2] + 0.5)
                        if (outward < 1.0
                                or math.dist(eye, point) > 4.0
                                or not is_surface_stand(
                                    world, candidate,
                                    hazard_clearance=(
                                        self.spec.hazard_clearance))
                                or not clear_voxel_ray(
                                    world, eye, point, target=support)):
                            continue
                        candidates.append(candidate)
        routes = []
        for candidate in candidates:
            if (candidate == destination
                    or not is_surface_stand(
                        world, candidate,
                        hazard_clearance=self.spec.hazard_clearance)):
                continue
            digest = hashlib.sha256(
                f"{self._decision_seed}:initial-light:{candidate}".encode()
            ).digest()
            try:
                route = plan_surface_path(
                    world, start, candidate,
                    hazard_clearance=self.spec.hazard_clearance,
                    node_cap=self.spec.node_cap,
                    decision_rng=random.Random(
                        int.from_bytes(digest[:8], "big")))
            except PlanRejected:
                continue
            routes.append(tuple(route))
        if not routes:
            return ()
        return min(routes, key=lambda route: (len(route), route[-1]))

    def _hotbar_swaps(self, env, wanted=(), *, extra=()):
        """Choose the first configured, visible, non-wanted swap stack."""

        wanted_ids = frozenset(self._item_ids(wanted)) if wanted else set()
        seen = set()
        for item in (*tuple(extra), *self.spec.hotbar.swap_priority):
            item_ids = frozenset(self._item_ids(item))
            if not item_ids or item_ids & wanted_ids or item_ids <= seen:
                continue
            seen.update(item_ids)
            if self.item_count(env, item_ids) > 0:
                return (item,)
        return ()

    def _craft_hotbar_swaps(self, env, operation):
        """Choose a swap that cannot evict the active recipe ingredients."""

        from bedrock_rl.sdg.policy.recipes import (
            choose_recipe, ingredient_cells,
        )

        ids, counts, metas = self._observed_inventory(env)
        recipe = choose_recipe(
            operation.item, ids, counts, operation.grid, metas)
        protected = [operation.item]
        if recipe is not None:
            protected.extend(
                item_id for item_id, _meta, _cells in
                ingredient_cells(recipe, operation.grid))
        protected_ids = frozenset(self._item_ids(tuple(protected)))
        preferred = self._hotbar_swaps(env, tuple(protected))
        if preferred:
            return preferred
        if any(int(found) == 0 or int(count) <= 0 for found, count in zip(
                env.obs["hotbar_ids"], env.obs["hotbar_counts"])):
            return ()

        # A completely full hotbar is common late in progression.  The
        # configured priority is a preference, not an exhaustive allowlist:
        # when none of those stacks is visible, evict the first exact live
        # stack that is not the output or an ingredient of the active recipe.
        # Full-inventory observations let the displaced stack be recovered
        # later without guessing hidden state.
        for found, count in zip(
                env.obs["hotbar_ids"], env.obs["hotbar_counts"]):
            found = int(found)
            if (found != 0 and int(count) > 0
                    and found not in protected_ids):
                return (found,)
        return ()

    def _fluid_cast_executor(self, env, operation):
        if self.cast_plan is None or self.fluid_source_plan is None:
            raise RuntimeError(
                "bucket-cast operation has no immutable preflight plan")
        sources = tuple(FluidSourceExecution(
            role=("coolant" if step.kind == "water" else "casting"),
            source_cell=step.source_cell,
            outbound_route=step.outbound,
            return_route=step.return_route,
            fluid=step.block_name,
            filled_bucket_item=(
                self.spec.portal.coolant_bucket_item
                if step.kind == "water"
                else self.spec.portal.casting_bucket_item),
            source_meta=step.meta,
        ) for step in self.fluid_source_plan.steps)
        execution = FluidCastExecution(
            plan=self.cast_plan,
            sources=sources,
            empty_bucket_item=self.spec.portal.bucket_item,
            result_item=operation.item,
            mining_tool=operation.tool,
            mining_ticks=operation.mining_ticks,
            excavation_tool=operation.tool,
            excavation_ticks=self.spec.portal.return_ticks,
            drop_count=1,
            casting_wait_ticks=self.spec.portal.casting_wait_ticks,
            recover_coolant=self.spec.portal.recover_coolant,
        )

        def recover_result(current_env, item, minimum):
            if self.item_count(current_env, item) >= minimum:
                return None
            return self.recover_hotbar(
                current_env, item,
                replace=self._hotbar_swaps(current_env, item),
                minimum=minimum)

        def recover_item(current_env, item, minimum):
            return self.recover_hotbar(
                current_env, item,
                replace=self._hotbar_swaps(current_env, item),
                minimum=minimum)

        executor = FluidCastExecutor(
            self,
            execution,
            max_pickup_attempts=self.behaviors.pickup_attempts,
            health_loss_tolerance=self.spec.health_loss_tolerance,
            lighting=(_LightingHooks(self)
                      if self._lighting_active else None),
            lighting_immediate=False,
            lighting_route_kinds=("surface", "tunnel"),
            item_recovery=recover_item,
            result_recovery=recover_result,
        )
        executor.reset(env)
        return executor

    def _bucket_cast_actions(self, env, operation):
        count = self.item_count(env, operation.item)
        if self._cast_executor is None:
            # Inventory completion can legitimately predate this operation
            # (for example in a custom progression), but once the immutable
            # cast has started it owns the complete lifecycle.  In
            # particular the final drop enters inventory before coolant
            # recovery, mold cleanup, and the certified final recenter.  Do
            # not let the objective count bypass those safety phases.
            if count >= operation.minimum:
                return None
            bucket = self.spec.portal.bucket_item
            if self.item_count(env, bucket) != 1:
                recovered = self.recover_hotbar(
                    env, bucket,
                    replace=self._hotbar_swaps(env, bucket))
                if recovered:
                    return recovered
                return PolicyVeto(
                    "progression_cast_bucket_missing",
                    "contained casting requires exactly one visible reusable "
                    "empty bucket",
                    (float(env.obs["x"]), float(env.obs["z"])))
            self._cast_executor = self._fluid_cast_executor(env, operation)
            self.stage = "fluid_cast"
        if not self._cast_executor.done:
            result = self._cast_executor.actions(env)
            if result is not None:
                return result
        if self.item_count(env, operation.item) >= operation.minimum:
            return None
        recovered = self.recover_hotbar(
            env, operation.item,
            replace=self._hotbar_swaps(env, operation.item))
        if recovered:
            return recovered
        return PolicyVeto(
            "progression_cast_result_missing",
            f"contained casting completed without {operation.minimum} "
            f"visible {operation.item!r}",
            (self.cast_plan.result_cell[0] + 0.5,
             self.cast_plan.result_cell[2] + 0.5),
            (self.cast_plan.result_cell,))

    def _operation_actions(self, env, operation):
        if isinstance(operation, NetherPortalOperation):
            if self.stage not in PORTAL_STAGES:
                self._plan_portal_return(env)
            return self._portal_executor.actions(env)

        if isinstance(operation, BucketCastMineOperation):
            return self._bucket_cast_actions(env, operation)

        if isinstance(operation, WorkstationOperation):
            site = self.workstation_sites[operation.key]
            if not self._at(env, site.stand):
                if operation.key not in self._workstation_route_keys:
                    recovered = self._recover_live_planning_start(env)
                    if recovered is not None:
                        return recovered
                    route = site.approach_stands
                    if (not route
                            or self._current_cell(env) != route[0]):
                        route = self._live_surface_route(
                            env, site.stand,
                            f"workstation:{operation.key}")
                    self._start_route(
                        route,
                        f"workstation:{operation.key}")
                    self._workstation_route_keys.add(operation.key)
                result = self._follow_route(env)
                if result is not None:
                    return result
                if not self._at(env, site.stand):
                    position = tuple(round(float(env.obs[key]), 3)
                                     for key in ("x", "y", "z"))
                    return PolicyVeto(
                        "progression_workstation_route_diverged",
                        f"could not reach workstation {operation.key!r}; "
                        f"position={position}, route={self._route!r}, "
                        f"route_index={self._route_index}",
                        (site.stand[0] + 0.5, site.stand[2] + 0.5),
                        (site.stand,))
            # Route acceptance allows ordinary cell-level tolerance. Exact
            # workstation placement does not: a player near the shared face
            # can overlap the adjacent destination AABB even while the click
            # ray correctly names it, making vanilla reject every click.
            stand_x, stand_z = site.stand[0] + 0.5, site.stand[2] + 0.5
            if math.hypot(
                    float(env.obs["x"]) - stand_x,
                    float(env.obs["z"]) - stand_z) > 0.08:
                recenter = self.recenter_interaction_stance(
                    env, stand_x, stand_z, tolerance=0.05)
                if recenter is not None:
                    return recenter
            stage = (self.spec.stages[self._stage_index]
                     if site.clear_cells else None)
            for cell in site.clear_cells:
                key = ("clear_workstation", cell)
                if self.events(
                        env, "block_broken", x=cell[0], y=cell[1],
                        z=cell[2]):
                    self._workstation_attempts.pop(key, None)
                    self._opened.add(cell)
                    continue
                tool = stage.tool or stage.tunnel_tool
                ticks = self.mining_hold_ticks(
                    env, cell, tool=tool,
                    fallback=(stage.tunnel_ticks or stage.mining_ticks))
                result = self.mine_exact(
                    env, cell, tool=tool, ticks=ticks,
                    stand=(site.stand[0] + 0.5,
                           site.stand[2] + 0.5))
                if isinstance(result, PolicyVeto):
                    return result
                if not result:
                    return PolicyVeto(
                        "workstation_clear_unavailable",
                        f"could not clear planned workstation cell {cell}",
                        (cell[0] + 0.5, cell[2] + 0.5), (cell,))
                if not any(action.get("action") == "left_click_hold"
                           for action in result):
                    return result
                return self._bounded_workstation_action(
                    key, result, max_attempts=4,
                    code="workstation_clear_exhausted",
                    reason=(f"planned workstation cell {cell} was not "
                            "journal-verified after 4 attempts"),
                    target=(cell[0] + 0.5, cell[2] + 0.5),
                    observed=(cell,))
            interactive_supports = {
                other.cell for other in self.workstation_sites.values()
                if other.cell != site.cell
            }
            result = self.place_workstation(
                env, operation.item, site.cell, site.support,
                block=operation.block,
                sneak=site.support in interactive_supports,
                stand=(site.stand[0] + 0.5, site.stand[2] + 0.5),
                replace=self._hotbar_swaps(env, operation.item))
            if result is None and not self._workstation_ready(
                    env, site.cell, operation.block):
                return PolicyVeto(
                    "progression_workstation_unverified",
                    f"workstation {operation.key!r} was not live-verified",
                    (site.cell[0] + 0.5, site.cell[2] + 0.5), (site.cell,))
            if result is None:
                self._placed[site.cell] = self._block_ids(
                    operation.block)[0]
            return result

        if isinstance(operation, CraftOperation):
            craft_key = (
                "craft", getattr(self, "_stage_index", -1),
                getattr(self, "_operation_index", -1),
                operation.item)
            if self.item_count(env, operation.item) >= operation.minimum:
                self._workstation_attempts.pop(craft_key, None)
                return None
            replacement = self._craft_hotbar_swaps(env, operation)

            def bounded(actions):
                if not actions or isinstance(actions, PolicyVeto):
                    return actions
                return self._bounded_workstation_action(
                    craft_key, actions, max_attempts=64,
                    code="progression_craft_exhausted",
                    reason=(f"crafting {operation.minimum} "
                            f"{operation.item!r} made no observable "
                            "progress after 64 bounded controls"),
                    target=(float(env.obs["x"]),
                            float(env.obs["z"])))
            # A full hotbar may make vanilla insert the preceding craft into
            # the backpack. Recover that journal-proven result before ever
            # considering a second irreversible recipe click.
            if self.observed_inventory_stack(
                    env, self._item_ids(operation.item)) is not None:
                recovered = self.recover_hotbar(
                    env, operation.item,
                    replace=replacement,
                    minimum=operation.minimum)
                if recovered:
                    return bounded(recovered)
            prepared = self.prepare_recipe_ingredients(
                env, operation.item, operation.grid,
                replace=replacement)
            if prepared:
                return bounded(prepared)
            if operation.grid == 2:
                result = self.craft_2x2(
                    env, operation.item,
                    output_replace=replacement)
            else:
                site = self.workstation_sites[operation.table]
                if not self._workstation_ready(
                        env, site.cell, "crafting_table"):
                    return PolicyVeto(
                        "progression_crafting_table_missing",
                        f"crafting table {operation.table!r} is absent",
                        (site.cell[0] + 0.5, site.cell[2] + 0.5),
                        (site.cell,))
                result = self.craft_3x3(
                    env, operation.item, table=site.cell,
                    output_replace=replacement)
            if result is None:
                ids, counts, metas = self._observed_inventory(env)
                slots = tuple(
                    (slot, ids[slot], counts[slot], metas[slot])
                    for slot in range(36) if counts[slot] > 0)
                return PolicyVeto(
                    "progression_recipe_unavailable",
                    f"cannot craft {operation.item!r} from visible stacks; "
                    f"container={int(env.obs.get('container', 0))}, "
                    f"inventory={dict(env.obs.get('inv_counts', {}))!r}, "
                    f"observed_slots={slots!r}",
                    (float(env.obs["x"]), float(env.obs["z"])))
            return bounded(result)

        if isinstance(operation, SmeltOperation):
            site = self.workstation_sites[operation.furnace]
            return self.smelt(
                env, site.cell, operation.input_item,
                operation.output_item, count=operation.count,
                fuel_item=operation.fuel_item,
                fuel_count=operation.fuel_count,
                stand=(site.stand[0] + 0.5, site.stand[2] + 0.5),
                output_replace=self._hotbar_swaps(
                    env, (operation.input_item, operation.output_item,
                          operation.fuel_item)))

        if isinstance(operation, RequireItemOperation):
            if self.item_count(env, operation.item) >= operation.minimum:
                return None
            recovered = self.recover_hotbar(
                env, operation.item,
                replace=self._hotbar_swaps(env, operation.item))
            if recovered:
                return recovered
            return PolicyVeto(
                "progression_required_drop_missing",
                f"natural mining produced no recoverable "
                f"{operation.item!r}",
                (float(env.obs["x"]), float(env.obs["z"])))

        if isinstance(operation, RepeatDropOperation):
            return self._repeat_drop_actions(env, operation)

        if isinstance(operation, BeginLightingOperation):
            if not self._lighting_active:
                if self._lighting_initial_origin is None:
                    self._lighting_initial_origin = self._current_cell(env)
                    self._lighting_initial_site = self._initial_light_site(
                        operation.support, env)
                    if self._lighting_initial_site is not None:
                        route = self._initial_light_vantage_route(
                            env, self._lighting_initial_site)
                        if not route:
                            return PolicyVeto(
                                "progression_initial_light_vantage_missing",
                                "no safe live route sees the declared first "
                                "wall-light face",
                                (float(env.obs["x"]),
                                 float(env.obs["z"])))
                        self._lighting_initial_vantage = route[-1]
                        self._start_route(route, "initial_light_vantage")
                if (self._lighting_initial_vantage is not None
                        and not self._at(
                            env, self._lighting_initial_vantage)):
                    result = self._follow_route(env)
                    if result is not None:
                        return result
                self.start_route_lighting(env, immediate=True)
                self.remember_route_position(env)
                self._lighting_active = True
            # Let WallTorchRouteMixin consume and validate the placement
            # event (including wall metadata) before this milestone advances.
            if self.torch_events_seen < 1:
                result = self.light_route(env)
                if result is None:
                    site = self.workstation_sites[operation.support]
                    observed = {(int(x), int(y), int(z)): int(block)
                                for block, x, y, z in
                                env.obs.get("blocks", ())}
                    light_cells = tuple(
                        (cell, observed.get(cell))
                        for cell in site.light_cells)
                    clicks = tuple(
                        event.as_dict() for event in env.journal[-32:]
                        if event.name == "click_target")[-6:]
                    return PolicyVeto(
                        "progression_initial_light_unavailable",
                        "torches were crafted but the immediate first wall "
                        "light could not be placed; "
                        f"workstation={site.to_dict()}, "
                        f"light_cells={light_cells}, "
                        f"initial={self._lighting_initial_site}, "
                        f"pending={self.pending_light}, "
                        f"rejected={tuple(sorted(self.rejected_light_supports))}, "
                        f"recent_route={tuple(self.recent_route_cells[-12:])}, "
                        f"clicks={clicks}",
                        (float(env.obs["x"]), float(env.obs["z"])))
                return result
            if (self._lighting_initial_origin is not None
                    and not self._at(env, self._lighting_initial_origin)):
                if not self._lighting_initial_returning:
                    route = self._live_surface_route(
                        env, self._lighting_initial_origin,
                        "initial_light_return")
                    self._start_route(route, "initial_light_return")
                    self._lighting_initial_returning = True
                result = self._follow_route(env)
                if result is not None:
                    return result
            self._lighting_initial_site = None
            self._lighting_initial_origin = None
            self._lighting_initial_vantage = None
            self._lighting_initial_returning = False
            return None

        raise TypeError(f"unsupported progression operation {operation!r}")

    @staticmethod
    def _cell_events(env, cell, *, since):
        cell = _cell(cell)
        return [event for event in env.journal[int(since):]
                if event.name in ("block_placed", "block_broken")
                and tuple(int(event[key]) for key in ("x", "y", "z"))
                == cell]

    def _place_repeated(self, env, operation, cell, support, stand):
        """Exact placement that scopes history to the repeat operation."""

        selected = self.select(env, operation.source_item)
        if selected is None:
            return None
        # Keep the irreversible use on the observation after the held stack
        # changed. This is the same engine contract as generic ``place``;
        # combining selection and use can race the live held-item update.
        if selected:
            return selected
        if math.hypot(
                float(env.obs["x"]) - (stand[0] + 0.5),
                float(env.obs["z"]) - (stand[2] + 0.5)) > 0.03:
            move = self.recenter_interaction_stance(
                env, stand[0] + 0.5, stand[2] + 0.5,
                tolerance=0.02)
            if move is not None:
                return move
        delta = tuple(cell[index] - support[index] for index in range(3))
        if sum(abs(value) for value in delta) != 1:
            raise ValueError("repeat-drop support must be face-adjacent")
        target = self.click_target(env)[1]
        if target == support and self.click_destination(env) == cell:
            return selected + [{"action": "right_click"}]
        point = tuple(support[index] + delta[index] * 0.49
                      for index in range(3))
        aim = self.aim_point(env, point)
        if aim:
            return selected + aim
        return PolicyVeto(
            "progression_repeat_place_ray_unproven",
            f"public ray does not prove repeated placement at {cell}",
            (cell[0] + 0.5, cell[2] + 0.5),
            (() if target is None else (target,)))

    def _mine_repeated(self, env, operation, cell, stand):
        """Exact mining without treating an earlier break as current state."""

        _yaw, _pitch, distance = env.rel_angles_to(*cell)
        if distance > MINING_REACH:
            move = self.face_and_move_to(
                env, stand[0] + 0.5, stand[2] + 0.5,
                tolerance=0.4)
            return move or PolicyVeto(
                "progression_repeat_mine_out_of_reach",
                f"planned repeated target {cell} is out of reach",
                (cell[0] + 0.5, cell[2] + 0.5), (cell,))
        found, target = self.click_target(env)
        expected = frozenset(self._block_ids(operation.block))
        if target == cell and found in expected:
            return [{"action": "left_click_hold",
                     "ticks": operation.mining_ticks}]
        aim = self.aim(env, cell)
        if aim:
            return aim
        return PolicyVeto(
            "progression_repeat_mine_ray_unproven",
            f"public ray does not prove repeated mining at {cell}",
            (cell[0] + 0.5, cell[2] + 0.5),
            (() if target is None else (target,)))

    def _repeat_drop_actions(self, env, operation):
        """Closed-loop replace/break/pickup cycle for gravel-like drops."""

        if self.item_count(env, operation.target_item) >= 1:
            return None
        target_slot = self.observed_inventory_slot(
            env, self._item_ids(operation.target_item))
        if target_slot is not None:
            recovered = self.recover_hotbar(
                env, operation.target_item,
                replace=self._hotbar_swaps(
                    env, operation.target_item,
                    extra=(operation.source_item,)))
            if recovered:
                return recovered

        state = self._repeat_states.setdefault(operation, {
            "event_start": len(env.journal), "events_seen": 0,
            "pickup_attempts": 0, "place_attempts": 0,
            "mine_attempts": 0,
        })
        target_plan = _resource_targets(
            self.plan, operation.resource)[-1]
        cell = target_plan.cell
        stand = target_plan.mining_stance
        support = cell[0], cell[1] - 1, cell[2]
        events = self._cell_events(
            env, cell, since=state["event_start"])
        if len(events) > state["events_seen"]:
            state["events_seen"] = len(events)
            if events[-1].name == "block_placed":
                state["mine_attempts"] = 0
            else:
                state["place_attempts"] = 0
                state["pickup_attempts"] = 0
        breaks = sum(event.name == "block_broken" for event in events)
        if breaks >= operation.max_attempts:
            return PolicyVeto(
                "progression_repeat_drop_exhausted",
                f"{operation.target_item!r} did not drop after "
                f"{operation.max_attempts} verified retries",
                (cell[0] + 0.5, cell[2] + 0.5), (cell,))

        if events and events[-1].name == "block_placed":
            if state["mine_attempts"] >= 8:
                return PolicyVeto(
                    "progression_repeat_mine_exhausted",
                    f"replaced {operation.block!r} did not break",
                    (cell[0] + 0.5, cell[2] + 0.5), (cell,))
            result = self._mine_repeated(env, operation, cell, stand)
            if result is None:
                return PolicyVeto(
                    "progression_repeat_mine_unavailable",
                    f"could not mine replaced {operation.block!r}",
                    (cell[0] + 0.5, cell[2] + 0.5), (cell,))
            if result and not isinstance(result, PolicyVeto) and any(
                    action.get("action") == "left_click_hold"
                    for action in result):
                state["mine_attempts"] += 1
            return result

        if self.item_count(env, operation.source_item) >= 1:
            state["pickup_attempts"] = 0
            if state["place_attempts"] >= 4:
                recent = tuple(
                    event.as_dict() for event in env.journal[-48:]
                    if event.name in (
                        "click_target", "block_placed", "item_dropped"))[-12:]
                return PolicyVeto(
                    "progression_repeat_place_exhausted",
                    f"could not replace {operation.source_item!r} at "
                    f"{cell}; position="
                    f"({float(env.obs['x']):.3f}, "
                    f"{float(env.obs['y']):.3f}, "
                    f"{float(env.obs['z']):.3f}), "
                    f"selected_slot={env.obs.get('hotbar_sel')}, "
                    f"click_target={self.click_target(env)[1]}, "
                    f"click_destination={self.click_destination(env)}, "
                    f"recent={recent}",
                    (cell[0] + 0.5, cell[2] + 0.5), (cell,))
            result = self._place_repeated(
                env, operation, cell, support, stand)
            if result is None:
                return PolicyVeto(
                    "progression_repeat_source_unavailable",
                    f"no visible {operation.source_item!r} for retry",
                    (cell[0] + 0.5, cell[2] + 0.5), (cell,))
            if not isinstance(result, PolicyVeto) and any(
                    action.get("action") == "right_click"
                    for action in result):
                state["place_attempts"] += 1
            return result

        recovered = self.recover_hotbar(
            env, (operation.source_item, operation.target_item),
            replace=self._hotbar_swaps(
                env, (operation.source_item, operation.target_item)))
        if recovered:
            return recovered
        if state["pickup_attempts"] >= self.behaviors.pickup_attempts:
            return PolicyVeto(
                "progression_repeat_drop_uncollected",
                "repeated natural drop was not publicly recoverable; "
                f"position=({float(env.obs['x']):.3f}, "
                f"{float(env.obs['y']):.3f}, "
                f"{float(env.obs['z']):.3f}), "
                f"pickup_geometry="
                f"{getattr(self, '_drop_pickup_trace', None)!r}",
                (cell[0] + 0.5, cell[2] + 0.5), (cell,))
        pickup_target, pickup_sneak = self.drop_pickup_site(
            env, target_plan)
        result = self.pickup(
            env, (operation.source_item, operation.target_item),
            minimum=1, attempt=state["pickup_attempts"],
            target=pickup_target,
            replace=self._hotbar_swaps(
                env, (operation.source_item, operation.target_item)),
            sneak=pickup_sneak,
            allow_jump=(cell[1]
                        < math.floor(float(env.obs["y"]))))
        state["pickup_attempts"] += 1
        return result or [{"action": "wait", "ticks": 1}]

    @staticmethod
    def _at(env, cell, tolerance=0.45):
        return (math.hypot(
                    float(env.obs["x"]) - (cell[0] + 0.5),
                    float(env.obs["z"]) - (cell[2] + 0.5))
                <= float(tolerance)
                and abs(float(env.obs["y"]) - cell[1]) <= 1.1)

    @staticmethod
    def _on_leg(env, left, right, tolerance=0.7):
        point = float(env.obs["x"]), float(env.obs["z"])
        start = left[0] + 0.5, left[2] + 0.5
        end = right[0] + 0.5, right[2] + 0.5
        dx, dz = end[0] - start[0], end[1] - start[1]
        length2 = dx * dx + dz * dz
        if length2 == 0:
            return False
        progress = ((point[0] - start[0]) * dx
                    + (point[1] - start[1]) * dz) / length2
        bounded = max(0.0, min(1.0, progress))
        nearest = start[0] + bounded * dx, start[1] + bounded * dz
        expected_y = left[1] + bounded * (right[1] - left[1])
        slack = float(tolerance) / math.sqrt(length2)
        return (-slack <= progress <= 1.0 + slack
                and math.dist(point, nearest) <= float(tolerance)
                and abs(float(env.obs["y"]) - expected_y) <= 1.25)

    def _start_route(self, cells, stage):
        compact = []
        for cell in cells:
            if not compact or compact[-1] != cell:
                compact.append(cell)
        self._route = tuple(compact)
        self._route_index = 1 if self._route else 0
        self._route_attempts = 0
        self.stage = stage

    def _follow_route(self, env):
        if not self._route:
            return None
        if len(self._route) == 1:
            destination = self._route[0]
            if self._at(env, destination):
                return None
            if self._current_cell(env) != destination:
                return PolicyVeto(
                    "progression_singleton_route_diverged",
                    f"player left singleton route cell {destination}",
                    (destination[0] + 0.5, destination[2] + 0.5),
                    (destination,))
            if self._route_attempts >= 8:
                return PolicyVeto(
                    "progression_singleton_route_center_exhausted",
                    f"player did not center within route cell {destination}",
                    (destination[0] + 0.5, destination[2] + 0.5),
                    (destination,))
            self._route_attempts += 1
            return self.face_and_move_to(
                env, destination[0] + 0.5, destination[2] + 0.5,
                tolerance=0.4, allow_jump=False,
                movement_cap_blocks=0.3)
        while self._route_index < len(self._route):
            previous = self._route[self._route_index - 1]
            destination_index = self._route_index
            direction = (
                self._route[self._route_index][0] - previous[0],
                self._route[self._route_index][1] - previous[1],
                self._route[self._route_index][2] - previous[2])
            if direction[1] == 0:
                for index in range(
                        self._route_index + 1, len(self._route)):
                    cell, prior = self._route[index], self._route[index - 1]
                    if ((cell[0] - prior[0], cell[1] - prior[1],
                         cell[2] - prior[2]) != direction
                            or math.dist(previous, cell)
                            > min(1.0, self.movement.max_blocks) + 1e-9
                            or any(candidate in getattr(
                                       self, "_route_mine_cells", ())
                                   for candidate in (
                                       cell,
                                       (cell[0], cell[1] + 1, cell[2])))):
                        break
                    destination_index = index
            destination = self._route[destination_index]
            vertical_tolerance = (
                0.05 if destination[1] != previous[1] else 1.1)
            horizontal_tolerance = 0.4
            if (math.hypot(
                    float(env.obs["x"]) - (destination[0] + 0.5),
                    float(env.obs["z"]) - (destination[2] + 0.5))
                    <= horizontal_tolerance
                    and abs(float(env.obs["y"]) - destination[1])
                    <= vertical_tolerance):
                self._route_index = destination_index + 1
                self._route_attempts = 0
                continue
            if not self.drop_is_safe(previous[1], destination[1]):
                return PolicyVeto(
                    "progression_route_drop_rejected",
                    f"planned drop {previous[1] - destination[1]:g} "
                    "exceeds configured safe descent",
                    (destination[0] + 0.5, destination[2] + 0.5),
                    (previous, destination))
            horizontal_error = math.hypot(
                float(env.obs["x"]) - (destination[0] + 0.5),
                float(env.obs["z"]) - (destination[2] + 0.5),
            )
            horizontal_arrived = horizontal_error <= 0.4
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
                ) <= 0.08
                and destination[1] + 0.05 < live_y
                <= previous[1] + 1.35)
            flat_airborne = (
                destination[1] == previous[1]
                and 0.05 < live_y - previous[1] <= 1.35
                and PlanExecutor._on_horizontal_leg(
                    env, previous, destination, 0.7))
            if ascending_airborne or descending_airborne or flat_airborne:
                if self._route_attempts >= 8:
                    return PolicyVeto(
                        "progression_route_vertical_settle_exhausted",
                        f"player did not settle onto route leg "
                        f"{previous} -> {destination}",
                        (destination[0] + 0.5, destination[2] + 0.5),
                        (previous, destination))
                self._route_attempts += 1
                return [{"action": "wait", "ticks": 2}]
            vertical_landed_offcenter = (
                destination[1] != previous[1]
                and abs(live_y - destination[1]) <= 0.05
                and horizontal_error > 0.05
                and PlanExecutor._on_horizontal_leg(
                    env, previous, destination, 0.7))
            if vertical_landed_offcenter:
                if self._route_attempts >= 8:
                    return PolicyVeto(
                        "progression_route_vertical_recenter_exhausted",
                        f"player landed outside the center of vertical "
                        f"stand {destination}",
                        (destination[0] + 0.5, destination[2] + 0.5),
                        (destination,))
                result = self.face_and_move_to(
                    env, destination[0] + 0.5,
                    destination[2] + 0.5,
                    tolerance=0.05, sequential_hop=True,
                    step_up=False, allow_jump=False,
                    movement_cap_blocks=0.3)
                if not result:
                    return PolicyVeto(
                        "progression_route_vertical_recenter_unavailable",
                        f"controller produced no recenter move over "
                        f"vertical stand {destination}",
                        (destination[0] + 0.5, destination[2] + 0.5),
                        (destination,))
                self._route_attempts += 1
                return result
            if not self._on_leg(env, previous, destination):
                code = ("progression_route_origin_diverged"
                        if self._route_index == 1
                        else "progression_route_position_diverged")
                return PolicyVeto(
                    code,
                    f"player left planned leg {previous} -> {destination}",
                    (destination[0] + 0.5, destination[2] + 0.5),
                    (previous, destination))
            if self._lighting_active:
                self.observe_route_travel(env)
                self.remember_route_position(env)
                light = self.light_route(env)
                if light is not None:
                    return light
                if self.route_light_overdue():
                    debug = self.route_light_debug(env)
                    return PolicyVeto(
                        "progression_route_light_overdue",
                        "no verified wall light was placed before the "
                        f"configured route distance; lighting={debug!r}",
                        (destination[0] + 0.5, destination[2] + 0.5))
            if self._route_attempts >= 8:
                return PolicyVeto(
                    "progression_route_move_exhausted",
                    f"player did not reach planned stand {destination}",
                    (destination[0] + 0.5, destination[2] + 0.5),
                    (destination,))
            self._route_attempts += 1
            return self.face_and_move_to(
                env, destination[0] + 0.5, destination[2] + 0.5,
                # A descending diagonal can enter the destination x/z voxel
                # while still standing on the upper support.  Continue near
                # its center so gravity can settle the player; the ordinary
                # 0.4 route tolerance would return ``None`` forever before
                # the stricter vertical-settle condition is reached.
                tolerance=(0.05 if destination[1] < previous[1] else 0.4),
                sequential_hop=True,
                step_up=destination[1] > previous[1],
                allow_jump=destination[1] >= previous[1],
                movement_cap_blocks=(
                    0.3 if destination[1] != previous[1] else None))
        return None

    def _portal_actions(self, env):
        """Compatibility entry point for direct controller-level callers."""
        return self._portal_executor.actions(env)

    def actions(self, task, episode, case, turn):
        del task, case, turn
        env = episode.env
        guard = self._guard(env)
        if guard is not None:
            return guard
        if self.stage == "complete":
            return None

        while self._stage_index < len(self.spec.stages):
            stage = self.spec.stages[self._stage_index]
            if self._executor is None:
                self.plan = self._stage_plan_objects[self._stage_index]
                self._protected_cells = frozenset(
                    cell for target in _targets(self.plan)
                    for cell in target.removal.protected_support_cells)
                self._executor = self._stage_executor(env, stage)
                self.stage = f"mine:{stage.name}"
            if not self._executor.done:
                result = self._executor.actions(env)
                if result is not None:
                    return result

            # A multi-turn operation owns its stage until it completes.
            if self.stage not in PORTAL_STAGES:
                self.stage = f"milestone:{stage.name}"
            while self._operation_index < len(stage.after):
                operation = stage.after[self._operation_index]
                result = self._operation_actions(env, operation)
                if result is not None:
                    return result
                self._operation_index += 1

            self._stage_index += 1
            self._operation_index = 0
            self._executor = None

        self.stage = "complete"
        return None

__all__ = ("SurvivalProgressionPolicy",)
