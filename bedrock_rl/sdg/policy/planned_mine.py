"""Generic planned policies for small, declarative Minecraft tasks."""

from __future__ import annotations

from bedrock_rl.sdg.policy.execution import (
    MiningExecution,
    PlanExecutor,
)
from bedrock_rl.sdg.policy.planning import (
    GoalSpec,
    ProximityConstraint,
    ResourceRequirement,
    WorldPlanBuilder,
)
from bedrock_rl.sdg.policy import (
    MinecraftController,
    PolicyVeto,
)
from bedrock_rl.sdg.policy.world import SnapshotWorldMap
from bedrock_rl.sdg.generation import CaseRejected

def _values(value):
    return tuple(value) if isinstance(value, (list, tuple)) else (value,)


class PlannedMinePolicy:
    """Plan and execute a small natural-resource collection task.

    The task supplies the objective and spawn constraints.  The world
    snapshot supplies all coordinates. ``block_broken`` objectives require
    exact journal records. ``item_gained`` objectives additionally require
    the expected drop to enter inventory; model text is never a proxy for
    either kind of success.
    """

    def __init__(
        self,
        movement=None,
        behaviors=None,
        mining_ticks=80,
        target_blocks=None,
        drop_item=None,
        *,
        controller=None,
        world_factory=SnapshotWorldMap,
        plan_builder=WorldPlanBuilder,
    ):
        self.controller = (controller if controller is not None else
                           MinecraftController(
                               movement=movement, behaviors=behaviors))
        self.world_factory = world_factory
        self.plan_builder = plan_builder
        self.mining_ticks = int(mining_ticks)
        self.configured_target_blocks = (
            None if target_blocks is None else _values(target_blocks))
        self.configured_drop_item = drop_item
        if self.mining_ticks < 1:
            raise ValueError("mining_ticks must be positive")
        self.stage = "not_reset"
        self.plan = None
        self.executor = None

    def _objective(self, task):
        success = task.success
        if success.type == "block_broken":
            raw = success.spec.get("block")
            if raw is None:
                raise CaseRejected(
                    "block_broken success must constrain its block field",
                    code="planned_mine_missing_block")
            blocks = _values(raw)
            drop_item = None
        elif success.type == "item_gained":
            item = success.spec.get("item")
            if item is None:
                raise CaseRejected(
                    "item_gained success must constrain its item field",
                    code="planned_mine_missing_item")
            # Natural blocks whose inventory item has the same catalog name
            # need no policy-specific config (log is the common case).
            # Ores and custom harvests can declare target_blocks explicitly.
            blocks = self.configured_target_blocks or (item,)
            drop_item = self.configured_drop_item or item
        else:
            raise CaseRejected(
                "PlannedMinePolicy requires block_broken or item_gained "
                "success",
                code="planned_mine_unsupported_success")
        count = int(success.spec.get("count", 1))
        label = str(blocks[0]) if len(blocks) == 1 else "target_blocks"
        return label, blocks, count, drop_item

    @staticmethod
    def _goal(task, label, blocks, count):
        proximity = tuple(ProximityConstraint(
            constraint.block, (constraint.block,),
            constraint.min_distance, constraint.max_distance)
            for constraint in task.initial_state.constraints)
        return GoalSpec(
            resources=(ResourceRequirement(label, blocks, count),),
            proximity=proximity,
            hazard_clearance=0,
            node_cap=2_000,
            routes=("surface",),
        )

    @staticmethod
    def _targets(plan):
        targets = []
        for resource in plan.resources:
            for cluster in resource.clusters:
                for target in cluster.targets:
                    if target.route.kind != "surface":
                        raise CaseRejected(
                            "simple planned mining requires a surface "
                            "approach",
                            code="planned_mine_surface_route_missing")
                    targets.append(target)
        return tuple(targets)

    def reset(self, task, episode, case):
        self.controller.reset(task, episode, case)
        label, blocks, count, drop_item = self._objective(task)
        snapshot = getattr(episode.env, "snapshot_in", None)
        if not snapshot:
            raise CaseRejected(
                "planned mining requires an initial-state snapshot",
                code="planned_mine_snapshot_missing")
        with self.world_factory(snapshot) as world:
            self.plan = self.plan_builder(
                world, self._goal(task, label, blocks, count),
                decision_seed=case.decision_seed).build()
        self.target_blocks = blocks
        self.target_ids = frozenset(self.controller._block_ids(blocks))
        self.required = count
        self.success_type = task.success.type
        self.drop_item = drop_item
        self.item_baseline = (None if drop_item is None else
                              self.controller.item_count(
                                  episode.env, drop_item))
        executions = tuple(
            MiningExecution(
                target, block=blocks, drop_item=drop_item,
                drop_count=1 if drop_item is not None else 0)
            for target in self._targets(self.plan)
        )
        self.executor = PlanExecutor(
            self.controller,
            executions,
            mining_ticks=self.mining_ticks,
            tunnel_ticks=self.mining_ticks,
            health_loss_tolerance=0,
            veto_fluids=True,
        ).reset(episode.env)
        self.stage = "follow_surface_route"

    def provenance(self):
        """Planning evidence for trajectory provenance, never model chat."""

        return None if self.plan is None else self.plan.to_dict()

    def demonstration_grounding(self):
        targets = set(getattr(self, "target_blocks", ()))
        grounded = bool(targets) and targets.issubset({
            "log", "coal_ore",
        })
        return {
            "eligible": grounded,
            "target_selection": (
                "model-visible bearing and frame" if grounded
                else "privileged initial-state snapshot"),
            "execution": "public observation and journal events",
        }

    def diagnostics(self, env):
        return {
            "stage": self.stage,
            "target": (None if self.executor is None
                       else self.executor.target_index),
            "targets": (0 if self.executor is None
                        else len(self.executor.targets)),
            "verified_breaks": self._verified_breaks(env),
            "verified_items": self._verified_items(env),
            "required": getattr(self, "required", None),
            "required_breaks": getattr(self, "required", None),
            "snapshot_sha256": (None if self.plan is None else
                                self.plan.snapshot_sha256),
        }

    def _verified_breaks(self, env):
        return sum(
            1 for event in env.journal
            if event.name == "block_broken"
            and int(event.get("block", -1)) in self.target_ids)

    def _verified_items(self, env):
        if getattr(self, "drop_item", None) is None:
            return None
        return (self.controller.item_count(env, self.drop_item)
                - self.item_baseline)

    def _complete(self, env):
        if self.success_type == "item_gained":
            return self._verified_items(env) >= self.required
        return self._verified_breaks(env) >= self.required

    def actions(self, task, episode, case, turn):
        del task, case, turn
        env = episode.env
        if self._complete(env):
            self.stage = "complete"
            return None
        if self.executor is None:
            raise RuntimeError("PlannedMinePolicy must be reset before use")
        self.stage = "execute_verified_plan"
        actions = self.executor.actions(env)
        if isinstance(actions, PolicyVeto):
            return actions
        if actions:
            return actions
        if not self.executor.done or not self._complete(env):
            raise CaseRejected(
                "planned targets exhausted before journal verified success",
                code="planned_mine_progress_incomplete")
        self.stage = "complete"
        return None


__all__ = ["PlannedMinePolicy"]
