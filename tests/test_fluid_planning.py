import json
from pathlib import Path
import struct
from types import SimpleNamespace
import tempfile
import unittest

from bedrock_rl.sdg.policy.fluids import (
    FluidSourcePlan,
    _route_breaches_fluid,
    plan_fluid_sources,
)
from bedrock_rl.sdg.policy.structures import (
    PlaceStep,
    select_fluid_cast_site,
)
from bedrock_rl.sdg.policy.spatial import (
    DEFAULT_BLOCK_RULES,
    PlanRejected,
    RejectReason,
)
from bedrock_rl.sdg.policy.world import SnapshotWorldMap
from bedrock_rl.env.snapshots import (
    ANGLE_OFF,
    HEAD_SIZE,
    INV_OFF,
    INV_SLOTS,
    POS_OFF,
)


ITEM_COUNT_OFF = INV_OFF + INV_SLOTS * 3 * 4
REGION_HEADER_OFF = ITEM_COUNT_OFF + struct.calcsize("<I")


def write_fluid_snapshot(
        path, *, seed=403, protected_lava=False,
        file_first_far_lava=False, blocked_work=False,
        missing_work_support=False):
    rx, ry, rz = -8, -7, -7
    nx, ny, nz = 24, 15, 15
    buf = bytearray(HEAD_SIZE + nx * ny * nz * 2 + 4)
    struct.pack_into("<4sIqqii", buf, 0, b"BSNP", 1, seed, 3, 0, 0)
    struct.pack_into("<3d", buf, POS_OFF, 0.5, 2.0, 0.5)
    struct.pack_into("<2f", buf, ANGLE_OFF, 0.0, 0.0)
    struct.pack_into("<I", buf, ITEM_COUNT_OFF, 0)
    struct.pack_into("<6i", buf, REGION_HEADER_OFF,
                     rx, ry, rz, nx, ny, nz)

    def put(x, y, z, block_id, meta=0):
        index = ((x - rx) * ny + (y - ry)) * nz + (z - rz)
        struct.pack_into("<H", buf, HEAD_SIZE + index * 2,
                         block_id << 4 | meta)

    # Two-block-deep surface plus solid underground. Deep source routes must
    # use a staircase tunnel while surface sources retain ordinary paths.
    for x in range(rx, rx + nx):
        for y in range(ry, 2):
            for z in range(rz, rz + nz):
                put(x, y, z, 1)

    if blocked_work:
        put(0, 2, 0, 1)
        put(0, 3, 0, 1)
    if missing_work_support:
        put(0, 1, 0, 0)

    put(2, 2, 0, 9, 0)       # exact water source
    put(1, 2, 3, 9, 3)       # flowing water, never collectible
    put(4, 2, 0, 11, 0)      # surface lava source
    put(5, 2, 2, 10, 0)      # second source, alternate block id
    put(6, 2, -2, 11, 0)     # third dry-route surface source
    put(8, -3, 0, 11, 0)     # underground source forces tunnel routing
    put(3, 2, -3, 11, 2)     # flowing lava, never collectible
    if protected_lava:
        put(3, 2, 1, 11, 0)
    if file_first_far_lava:
        # First in SnapshotWorldMap x/y/z order and pinned against the lower
        # bounds, but much farther from work than the ordinary surface source.
        # A pre-ranking candidate limit would retain this cell incorrectly.
        put(-7, -6, -6, 11, 0)
    struct.pack_into("<I", buf, len(buf) - 4, 0)
    path.write_bytes(buf)


def cast_plan(world, *, casts_required=3, protected=()):
    return SimpleNamespace(
        world_seed=world.seed,
        snapshot_sha256=world.snapshot_sha256,
        casts_required=casts_required,
        work_stance=(0, 2, 0),
        protected_cells=tuple(protected),
    )


def cast_plan_with_ray(world, *, protected, ray, structural=()):
    """Small cast contract containing the fields source planning reads."""

    return SimpleNamespace(
        world_seed=world.seed,
        snapshot_sha256=world.snapshot_sha256,
        casts_required=1,
        work_stance=(0, 2, 0),
        protected_cells=tuple(protected),
        approach_stands=((0, 2, 0),),
        catalyst_recovery_stance=(0, 2, 0),
        result_mining_stance=(0, 2, 0),
        excavation_cells=tuple(structural),
        support_cells=(),
        backing_cells=(),
        temporary_cells=(),
        setup_placements=(),
        interactions=(SimpleNamespace(ray_cells=tuple(ray)),),
    )


class FluidSourcePlanningTests(unittest.TestCase):
    def test_new_tunnel_cells_cannot_open_sideways_into_fluid(self):
        class World:
            fluid = (1, 0, 0)

            @staticmethod
            def in_bounds(*_cell):
                return True

            def block(self, *cell):
                return 9 if tuple(cell) == self.fluid else 1

        world = World()
        self.assertTrue(_route_breaches_fluid(
            world, ((0, 0, 0),), rules=DEFAULT_BLOCK_RULES))
        world.fluid = (1, -1, 0)
        self.assertFalse(_route_breaches_fluid(
            world, ((0, 0, 0),), rules=DEFAULT_BLOCK_RULES))

    def open_world(self, directory, **kwargs):
        path = Path(directory) / "fluids.bsnp"
        write_fluid_snapshot(path, **kwargs)
        return SnapshotWorldMap(path)

    def test_exact_sources_return_to_mold_and_commit_overlay(self):
        with tempfile.TemporaryDirectory() as directory, \
                self.open_world(directory) as world:
            plan = plan_fluid_sources(
                world, cast_plan(world), decision_seed=71,
                node_cap=4_000, route_search_cap=256)

            self.assertIsInstance(plan, FluidSourcePlan)
            self.assertEqual(len(plan.steps), 4)
            self.assertEqual(plan.water_source.kind, "water")
            self.assertEqual(len(plan.lava_sources), 3)
            self.assertEqual(
                tuple(step.kind for step in plan.steps),
                ("water", "lava", "lava", "lava"))
            self.assertTrue(all(step.meta == 0 for step in plan.steps))
            sources = tuple(step.source_cell for step in plan.steps)
            self.assertEqual(len(sources), len(set(sources)))
            self.assertNotIn((1, 2, 3), sources)
            self.assertNotIn((3, 2, -3), sources)
            self.assertTrue(set(sources).issubset(plan.opened_cells))

            for step in plan.steps:
                self.assertEqual(step.outbound.stands[0], plan.work_stance)
                self.assertEqual(step.outbound.stands[-1],
                                 step.pickup_stance)
                self.assertEqual(step.return_route.stands[0],
                                 step.pickup_stance)
                self.assertEqual(step.return_route.stands[-1],
                                 plan.work_stance)
                self.assertEqual(step.pickup_ray_cells[-1],
                                 step.source_cell)
                self.assertLessEqual(
                    ((step.pickup_stance[0] + 0.5
                      - (step.source_cell[0] + 0.5)) ** 2
                     + (step.pickup_stance[1] + 1.62
                        - (step.source_cell[1] + 0.5)) ** 2
                     + (step.pickup_stance[2] + 0.5
                        - (step.source_cell[2] + 0.5)) ** 2) ** 0.5,
                    plan.max_reach + 1e-9)

            # Reaching the underground source would require excavating a
            # cell that lava can enter before pickup. The safe planner leaves
            # it unused and chooses the third naturally exposed source.
            self.assertNotIn((8, -3, 0), sources)
            self.assertIn((6, 2, -2), sources)

            payload = json.loads(plan.to_json(sort_keys=True))
            self.assertEqual(payload["schema"],
                             "bedrock.fluid_source_plan.v1")
            self.assertEqual(payload["world_seed"], world.seed)
            self.assertEqual(payload["decision_seed"], 71)
            self.assertEqual(
                [item["block"]["meta"] for item in payload["steps"]],
                [0, 0, 0, 0])
            self.assertTrue(all(
                item["block"]["source_only"]
                for item in payload["steps"]))

    def test_protected_source_is_skipped_and_never_opened(self):
        protected_source = (3, 2, 1)
        with tempfile.TemporaryDirectory() as directory, \
                self.open_world(
                    directory, protected_lava=True) as world:
            plan = plan_fluid_sources(
                world,
                cast_plan(
                    world, casts_required=3,
                    protected=(protected_source,)),
                decision_seed=9, node_cap=4_000,
                route_search_cap=256)

            self.assertNotIn(
                protected_source,
                {step.source_cell for step in plan.steps})
            self.assertNotIn(protected_source, plan.opened_cells)
            self.assertIn(protected_source, plan.protected_cells)
            self.assertTrue(all(
                protected_source not in step.outbound.mine_cells
                and protected_source not in step.return_route.mine_cells
                for step in plan.steps))

    def test_real_cast_plan_protects_complete_mold_during_source_routes(self):
        with tempfile.TemporaryDirectory() as directory, \
                self.open_world(directory) as world:
            mold = select_fluid_cast_site(
                world, decision_seed=12, start=(0, 2, 0),
                casts_required=3, search_radius=3,
                vertical_radius=0, hazard_clearance=0)
            plan = plan_fluid_sources(
                world, mold, decision_seed=13,
                node_cap=4_000, route_search_cap=256)

            self.assertTrue(
                set(mold.protected_cells).issubset(plan.protected_cells))
            self.assertFalse(
                set(mold.protected_cells).intersection(plan.opened_cells))
            self.assertTrue(all(
                not set(mold.protected_cells).intersection(
                    step.outbound.mine_cells
                    + step.return_route.mine_cells)
                for step in plan.steps))

    def test_plan_is_deterministic_for_decision_seed(self):
        with tempfile.TemporaryDirectory() as directory, \
                self.open_world(directory) as world:
            first = plan_fluid_sources(
                world, cast_plan(world), decision_seed=83,
                node_cap=4_000, route_search_cap=256)
            second = plan_fluid_sources(
                world, cast_plan(world), decision_seed=83,
                node_cap=4_000, route_search_cap=256)

            self.assertEqual(first.to_dict(), second.to_dict())

    def test_candidate_limit_keeps_nearest_not_first_snapshot_cell(self):
        with tempfile.TemporaryDirectory() as directory, \
                self.open_world(
                    directory, file_first_far_lava=True) as world:
            plan = plan_fluid_sources(
                world, cast_plan(world, casts_required=1),
                decision_seed=44, candidate_limit=1,
                node_cap=4_000, route_search_cap=32)

            self.assertEqual(len(plan.lava_sources), 1)
            self.assertEqual(plan.lava_sources[0].source_cell, (4, 2, 0))
            self.assertNotEqual(
                plan.lava_sources[0].source_cell, (-7, -6, -6))

    def test_prior_overlay_enables_route_and_separates_new_breaks(self):
        feet = (0, 2, 0)
        head = (0, 3, 0)
        with tempfile.TemporaryDirectory() as directory, \
                self.open_world(directory, blocked_work=True) as world:
            mold = cast_plan(
                world, casts_required=2, protected=(feet, head))

            with self.assertRaises(PlanRejected):
                plan_fluid_sources(
                    world, mold, decision_seed=31,
                    node_cap=4_000, route_search_cap=256)

            plan = plan_fluid_sources(
                world, mold, decision_seed=31,
                opened=(feet,), removed=(head,),
                placed={(4, 2, 0): 58},
                node_cap=4_000, route_search_cap=256)

            self.assertEqual(
                set(plan.initial_opened_cells), {feet, head})
            self.assertTrue(plan.newly_opened_cells)
            self.assertFalse(
                set(plan.initial_opened_cells).intersection(
                    plan.newly_opened_cells))
            self.assertEqual(
                set(plan.opened_cells),
                set(plan.initial_opened_cells)
                | set(plan.newly_opened_cells))
            self.assertEqual(plan.placed_blocks, (((4, 2, 0), 58),))
            self.assertNotIn(
                (4, 2, 0),
                {step.source_cell for step in plan.steps})

    def test_cast_setup_replaces_prior_air_before_source_planning(self):
        setup_cell = (1, 2, 1)
        with tempfile.TemporaryDirectory() as directory, \
                self.open_world(directory) as world:
            mold = cast_plan(
                world, casts_required=1, protected=(setup_cell,))
            mold.setup_placements = (
                PlaceStep.on(
                    setup_cell, (1, 1, 1), order=0,
                    material="cobblestone"),
            )

            plan = plan_fluid_sources(
                world, mold, decision_seed=31, opened=(setup_cell,),
                node_cap=4_000, route_search_cap=256)

            self.assertNotIn(setup_cell, plan.initial_opened_cells)
            self.assertIn(
                (setup_cell, "cobblestone"), plan.placed_blocks)
            self.assertTrue(all(
                setup_cell not in step.outbound.mine_cells
                and setup_cell not in step.return_route.mine_cells
                for step in plan.steps))

    def test_explicit_protected_passage_accepts_prior_open_route_body(self):
        feet = (0, 2, 0)
        head = (0, 3, 0)
        with tempfile.TemporaryDirectory() as directory, \
                self.open_world(directory, blocked_work=True) as world:
            mold = cast_plan(
                world, casts_required=2, protected=(feet, head))

            plan = plan_fluid_sources(
                world, mold, decision_seed=31,
                opened=(feet, head), protected=(feet, head),
                allowed_passage=(feet, head),
                placed={(4, 2, 0): 58}, node_cap=4_000,
                route_search_cap=256)

            self.assertEqual(
                set(plan.initial_opened_cells), {feet, head})
            self.assertTrue(all(
                {feet, head}.isdisjoint(step.outbound.mine_cells)
                and {feet, head}.isdisjoint(step.return_route.mine_cells)
                for step in plan.steps))

    def test_prior_open_intermediate_cast_ray_is_implicit_passage(self):
        ray_air = (0, 3, 1)
        ray_target = (0, 2, 2)
        with tempfile.TemporaryDirectory() as directory, \
                self.open_world(directory) as world:
            mold = cast_plan_with_ray(
                world, protected=(ray_air, ray_target),
                ray=((0, 3, 0), ray_air, ray_target))

            plan = plan_fluid_sources(
                world, mold, decision_seed=31, opened=(ray_air,),
                node_cap=4_000, route_search_cap=256)

            self.assertIn(ray_air, plan.initial_opened_cells)
            self.assertTrue(all(
                ray_air not in step.outbound.mine_cells
                and ray_air not in step.return_route.mine_cells
                for step in plan.steps))

    def test_cast_ray_target_or_unrelated_protected_air_stays_strict(self):
        ray_air = (0, 3, 1)
        ray_target = (0, 2, 2)
        unrelated = (1, 3, 1)
        structural = (1, 2, 1)
        with tempfile.TemporaryDirectory() as directory, \
                self.open_world(directory) as world:
            mold = cast_plan_with_ray(
                world,
                protected=(ray_air, ray_target, unrelated, structural),
                ray=((0, 3, 0), ray_air, ray_target),
                structural=(structural,))
            for opened in ((ray_target,), (unrelated,), (structural,)):
                with self.subTest(opened=opened), self.assertRaisesRegex(
                        ValueError, "opened cells overlap protected"):
                    plan_fluid_sources(
                        world, mold, decision_seed=31, opened=opened,
                        node_cap=4_000, route_search_cap=256)

    def test_fluid_passage_must_also_be_protected(self):
        with tempfile.TemporaryDirectory() as directory, \
                self.open_world(directory) as world:
            with self.assertRaisesRegex(ValueError, "subset of protected"):
                plan_fluid_sources(
                    world, cast_plan(world), decision_seed=1,
                    allowed_passage=((0, 2, 0),))

    def test_protected_air_cannot_become_fake_route_support(self):
        support = (0, 1, 0)
        with tempfile.TemporaryDirectory() as directory, \
                self.open_world(
                    directory, missing_work_support=True) as world:
            with self.assertRaises(PlanRejected):
                plan_fluid_sources(
                    world,
                    cast_plan(
                        world, casts_required=1, protected=(support,)),
                    decision_seed=37, node_cap=4_000,
                    route_search_cap=256)

    def test_search_radius_and_source_only_knobs_are_loud(self):
        with tempfile.TemporaryDirectory() as directory, \
                self.open_world(directory) as world:
            with self.assertRaises(PlanRejected):
                plan_fluid_sources(
                    world, cast_plan(world, casts_required=1),
                    decision_seed=41, search_radius=3,
                    node_cap=4_000, route_search_cap=256)
            plan = plan_fluid_sources(
                world, cast_plan(world, casts_required=1),
                decision_seed=41, search_radius=4,
                node_cap=4_000, route_search_cap=256)
            self.assertEqual(plan.search_radius, 4.0)
            with self.assertRaisesRegex(ValueError, "source_meta=0"):
                plan_fluid_sources(
                    world, cast_plan(world, casts_required=1),
                    decision_seed=41, source_meta=1)
            with self.assertRaisesRegex(ValueError, "unique"):
                plan_fluid_sources(
                    world, cast_plan(world, casts_required=1),
                    decision_seed=41, unique_casting_sources=False)

    def test_insufficient_meta_zero_sources_rejects_before_routes(self):
        with tempfile.TemporaryDirectory() as directory, \
                self.open_world(directory) as world:
            with self.assertRaises(PlanRejected) as raised:
                plan_fluid_sources(
                    world, cast_plan(world, casts_required=5),
                    decision_seed=5, node_cap=4_000,
                    route_search_cap=256)

            self.assertEqual(
                raised.exception.reason, RejectReason.UNREACHABLE_TARGET)
            self.assertIn("4/5", raised.exception.detail)
            self.assertIn("metadata-zero lava", raised.exception.detail)

    def test_route_search_budget_is_loud_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory, \
                self.open_world(directory) as world:
            with self.assertRaises(PlanRejected) as raised:
                plan_fluid_sources(
                    world, cast_plan(world), decision_seed=6,
                    node_cap=4_000, route_search_cap=1)

            self.assertEqual(raised.exception.reason, RejectReason.NODE_CAP)
            self.assertEqual(raised.exception.examined, 1)
            self.assertIn("route-search budget 1", raised.exception.detail)

    def test_snapshot_and_cast_provenance_must_match(self):
        with tempfile.TemporaryDirectory() as directory, \
                self.open_world(directory) as world:
            wrong = cast_plan(world)
            wrong.snapshot_sha256 = "0" * 64
            with self.assertRaisesRegex(ValueError, "provenance"):
                plan_fluid_sources(world, wrong, decision_seed=1)

            with self.assertRaisesRegex(TypeError, "SnapshotWorldMap"):
                plan_fluid_sources(
                    SimpleNamespace(closed=False), cast_plan(world),
                    decision_seed=1)


if __name__ == "__main__":
    unittest.main()
