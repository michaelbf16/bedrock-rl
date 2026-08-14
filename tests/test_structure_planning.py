import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from bedrock_rl.sdg.policy.spatial import (
    DEFAULT_BLOCK_RULES,
    PlanRejected,
    RejectReason,
    is_surface_stand,
)
from bedrock_rl.sdg.policy.structures import (
    FluidCastPlan,
    InteractionRay,
    PlaceStep,
    select_fluid_cast_site,
    select_nether_portal_completion_site,
    select_nether_portal_site,
    validate_support_order,
)
from bedrock_rl.sdg.policy.world import SnapshotWorldMap
from tests.test_world_planning import write_planning_snapshot


class FlatSnapshotWorld:
    """Small protocol-compatible map with snapshot-style provenance."""

    def __init__(self, *, shift=0, seed=73):
        self.low = (shift - 18, -3, -18)
        self.high = (shift + 18, 12, 18)
        self.blocks = {}
        self.seed = seed
        self.snapshot_sha256 = f"{shift + 100:064x}"
        self.player = SimpleNamespace(
            x=shift + 0.5, y=1.0, z=0.5)

    def in_bounds(self, *coordinates):
        cell = coordinates[0] if len(coordinates) == 1 else coordinates
        return all(self.low[index] <= cell[index] <= self.high[index]
                   for index in range(3))

    def block(self, *coordinates):
        cell = coordinates[0] if len(coordinates) == 1 else coordinates
        if not self.in_bounds(cell):
            return None
        return self.blocks.get(
            cell, "stone" if cell[1] <= 0 else "air")

    def put(self, block, *cells):
        for cell in cells:
            self.blocks[cell] = block


def shift_cell(cell, amount):
    return cell[0] + amount, cell[1], cell[2]


class PlaceStepTests(unittest.TestCase):
    def test_face_is_derived_and_matches_engine_journal_ids(self):
        steps = [
            PlaceStep.on((0, 1, 0), (0, 0, 0), order=0,
                         material="stone"),
            PlaceStep.on((-1, 1, 0), (0, 1, 0), order=1,
                         material="stone"),
            PlaceStep.on((-1, 1, 1), (-1, 1, 0), order=2,
                         material="stone"),
        ]
        self.assertEqual(
            [(step.face, step.face_id) for step in steps],
            [("up", 4), ("west", 1), ("south", 6)],
        )
        with self.assertRaisesRegex(ValueError, "not face-adjacent"):
            PlaceStep.on((2, 0, 0), (0, 0, 0), order=0,
                         material="stone")
        with self.assertRaisesRegex(ValueError, "does not match"):
            PlaceStep((0, 1, 0), (0, 0, 0), "north", 0, "stone")

    def test_support_validator_rejects_floating_or_out_of_order_steps(self):
        world = FlatSnapshotWorld()
        floating = PlaceStep.on(
            (0, 3, 0), (0, 2, 0), order=0, material="stone")
        with self.assertRaises(PlanRejected) as raised:
            validate_support_order(world, (floating,))
        self.assertEqual(raised.exception.reason, RejectReason.UNSAFE_CELL)

        valid = PlaceStep.on(
            (0, 1, 0), (0, 0, 0), order=1, material="stone")
        with self.assertRaisesRegex(ValueError, "contiguous"):
            validate_support_order(world, (valid,))


class NetherPortalStructureTests(unittest.TestCase):
    def test_completion_chamber_constructs_portal_in_solid_volume(self):
        world = FlatSnapshotWorld()
        for x in range(-8, 9):
            for y in range(1, 7):
                for z in range(-6, 7):
                    world.put("stone", (x, y, z))
        # A two-high mined corridor is enough; there is deliberately no
        # naturally empty 4x5 portal pad anywhere in the bounded search.
        for x in range(-6, 7):
            world.put("air", (x, 1, 0), (x, 2, 0))

        with self.assertRaises(PlanRejected):
            select_nether_portal_site(
                world, decision_seed=41, search_radius=3,
                vertical_radius=0, hazard_clearance=0)
        plan = select_nether_portal_completion_site(
            world, decision_seed=41, search_radius=3,
            vertical_radius=0, hazard_clearance=0)

        self.assertEqual(plan.kind, "nether_portal_completion_chamber")
        self.assertTrue(plan.clear_cells)
        self.assertEqual(len(plan.frame_cells), 10)
        self.assertEqual(len(plan.open_cells), 6)
        self.assertTrue(set(plan.frame_cells).intersection(plan.clear_cells))
        self.assertTrue(all(
            DEFAULT_BLOCK_RULES.is_excavatable(world.block(cell))
            for cell in plan.clear_cells))

    def test_exact_minimal_geometry_support_order_and_safe_stances(self):
        world = FlatSnapshotWorld()
        plan = select_nether_portal_site(world, decision_seed=19)
        axis = plan.orientation.split(":", 1)[0]
        horizontal = (1, 0, 0) if axis == "x" else (0, 0, 1)

        def at(across, up):
            return (
                plan.anchor[0] + horizontal[0] * across,
                plan.anchor[1] + up,
                plan.anchor[2] + horizontal[2] * across,
            )

        expected_frame = {
            at(1, 0), at(2, 0),
            at(0, 1), at(3, 1),
            at(0, 2), at(3, 2),
            at(0, 3), at(3, 3),
            at(1, 4), at(2, 4),
        }
        expected_open = {
            at(across, up)
            for across in (1, 2) for up in (1, 2, 3)
        }
        self.assertEqual(len(plan.frame_cells), 10)
        self.assertEqual(set(plan.frame_cells), expected_frame)
        self.assertEqual(len(plan.open_cells), 6)
        self.assertEqual(set(plan.open_cells), expected_open)
        self.assertEqual(len(plan.temporary_cells), 7)
        self.assertEqual(
            sum(step.material == "obsidian" for step in plan.placements), 10)
        self.assertEqual(
            sum(step.role == "scaffold" for step in plan.placements), 7)
        validate_support_order(world, plan.placements)

        self.assertEqual(plan.approach_stands[0], (0, 1, 0))
        self.assertEqual(plan.approach_stands[-1], plan.work_stance)
        for stance in (
                plan.work_stance, plan.ignition_stance, plan.exit_stance):
            self.assertTrue(is_surface_stand(
                world, stance, hazard_clearance=1))
        self.assertIn(plan.entry_stance, plan.open_cells)
        self.assertIn(plan.interaction_cell, plan.open_cells)
        self.assertIn(plan.interaction_support, plan.frame_cells)
        self.assertEqual(plan.interaction_face, "up")

        payload = json.loads(plan.to_json(sort_keys=True))
        self.assertEqual(payload["schema"], "bedrock.structure_plan.v1")
        self.assertEqual(payload["world_seed"], 73)
        self.assertEqual(payload["decision_seed"], 19)
        self.assertEqual(len(payload["occupied_cells"]), 10)

    def test_plan_translates_with_world_instead_of_embedding_coordinates(self):
        left_world = FlatSnapshotWorld(shift=0, seed=81)
        right_world = FlatSnapshotWorld(shift=47, seed=81)
        left = select_nether_portal_site(left_world, decision_seed=2026)
        right = select_nether_portal_site(right_world, decision_seed=2026)
        self.assertEqual(left.orientation, right.orientation)
        for first, second in (
            (left.anchor, right.anchor),
            (left.work_stance, right.work_stance),
            (left.ignition_stance, right.ignition_stance),
            (left.entry_stance, right.entry_stance),
            (left.exit_stance, right.exit_stance),
        ):
            self.assertEqual(shift_cell(first, 47), second)
        self.assertEqual(
            tuple(shift_cell(cell, 47) for cell in left.frame_cells),
            right.frame_cells,
        )
        self.assertEqual(
            tuple(shift_cell(step.cell, 47) for step in left.placements),
            tuple(step.cell for step in right.placements),
        )

    def test_hazard_near_one_site_is_avoided(self):
        world = FlatSnapshotWorld()
        initial = select_nether_portal_site(world, decision_seed=9)
        start = (0, 1, 0)
        lava = max(initial.frame_cells, key=lambda cell: sum(
            abs(cell[index] - start[index]) for index in range(3)))
        world.put("lava", lava)
        replanned = select_nether_portal_site(
            world, decision_seed=9, search_radius=8,
            hazard_clearance=1)
        relevant = (
            replanned.frame_cells + replanned.open_cells
            + replanned.temporary_cells
            + (replanned.work_stance, replanned.ignition_stance,
               replanned.entry_stance, replanned.exit_stance)
        )
        self.assertTrue(all(
            max(abs(cell[index] - lava[index]) for index in range(3)) > 1
            for cell in relevant
        ))

    def test_bounded_world_without_frame_headroom_rejects(self):
        world = FlatSnapshotWorld()
        world.high = (world.high[0], 4, world.high[2])
        with self.assertRaises(PlanRejected) as raised:
            select_nether_portal_site(
                world, decision_seed=1, search_radius=0,
                vertical_radius=0, hazard_clearance=0)
        self.assertEqual(raised.exception.reason, RejectReason.NO_PATH)
        self.assertGreaterEqual(raised.exception.examined, 1)

    def test_real_snapshot_supplies_seed_hash_pose_and_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "portal-world.bsnp"
            write_planning_snapshot(path, seed=991)
            with SnapshotWorldMap(path) as world:
                plan = select_nether_portal_site(
                    world, decision_seed=44,
                    search_radius=4, hazard_clearance=0)
                self.assertEqual(plan.world_seed, world.seed)
                self.assertEqual(plan.snapshot_sha256,
                                 world.snapshot_sha256)
                self.assertEqual(plan.approach_stands[0], (
                    int(world.player.x // 1), int(world.player.y // 1),
                    int(world.player.z // 1)))
                self.assertTrue(all(
                    DEFAULT_BLOCK_RULES.is_passable(
                        world.block(*cell))
                    for cell in plan.frame_cells + plan.open_cells
                ))


class FluidCastStructureTests(unittest.TestCase):
    def test_reusable_mold_has_contained_geometry_and_exact_rays(self):
        world = FlatSnapshotWorld()
        plan = select_fluid_cast_site(
            world, decision_seed=19, casts_required=10)

        self.assertIsInstance(plan, FluidCastPlan)
        self.assertEqual(plan.casts_required, 10)
        self.assertEqual(
            (plan.catalyst_fluid, plan.reactant_fluid, plan.result_block),
            ("water", "lava", "obsidian"))
        self.assertEqual(
            set(plan.excavation_cells),
            {plan.catalyst_cell, plan.result_cell})
        self.assertEqual(
            plan.excavation_cells,
            (plan.catalyst_cell, plan.result_cell),
        )
        self.assertEqual(plan.access_clear_cells, ())
        self.assertEqual(plan.fluid_envelope, plan.excavation_cells)
        self.assertEqual(plan.temporary_cells, plan.excavation_cells)
        self.assertEqual(plan.cleanup_cells, plan.excavation_cells)
        self.assertEqual(plan.final_open_cells, plan.excavation_cells)
        self.assertEqual(len(plan.support_cells), 2)
        self.assertTrue(all(
            cell[1] == plan.catalyst_cell[1] - 1
            for cell in plan.support_cells))

        boundary = {
            (cell[0] + dx, cell[1], cell[2] + dz)
            for cell in plan.excavation_cells
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))
        } - set(plan.excavation_cells)
        self.assertEqual(set(plan.backing_cells), boundary)
        self.assertTrue(all(
            DEFAULT_BLOCK_RULES.is_solid(world.block(cell))
            for cell in plan.support_cells + plan.backing_cells))

        self.assertEqual(plan.catalyst_placement.cell,
                         plan.catalyst_cell)
        self.assertEqual(plan.reactant_placement.cell, plan.result_cell)
        self.assertIn(plan.catalyst_placement.support,
                      plan.backing_cells)
        self.assertIn(plan.reactant_placement.support,
                      plan.backing_cells)
        self.assertEqual(
            tuple(item.action for item in plan.interactions),
            ("place_catalyst", "place_reactant",
             "recover_catalyst", "mine_result"))
        self.assertTrue(all(isinstance(item, InteractionRay)
                            for item in plan.interactions))
        self.assertTrue(all(item.ray_cells[-1] == item.target
                            for item in plan.interactions))
        self.assertEqual(plan.interactions[0].target,
                         plan.catalyst_placement.support)
        self.assertEqual(plan.interactions[1].target,
                         plan.reactant_placement.support)
        self.assertEqual(plan.interactions[2].target,
                         plan.catalyst_cell)
        self.assertEqual(plan.interactions[3].target, plan.result_cell)

        self.assertEqual(plan.approach_stands[0], (0, 1, 0))
        self.assertEqual(plan.approach_stands[-1], plan.work_stance)
        self.assertTrue({
            plan.catalyst_cell, plan.result_cell,
            *plan.support_cells, *plan.backing_cells,
        }.issubset(plan.protected_cells))
        payload = json.loads(plan.to_json(sort_keys=True))
        self.assertEqual(payload["schema"],
                         "bedrock.fluid_cast_plan.v3")
        self.assertNotIn("cleanup_cells", payload)
        self.assertEqual(payload["final_state"], {
            "default": "open_dry_mold",
            "final_open_cells": [list(cell)
                                 for cell in plan.final_open_cells],
            "restoration_placements": [],
        })
        self.assertEqual(payload["world_seed"], 73)
        self.assertEqual(payload["decision_seed"], 19)
        self.assertNotIn("source_cell", payload)
        self.assertNotIn("portal", payload["kind"])

    def test_cast_site_translates_without_seed_specific_coordinates(self):
        left_world = FlatSnapshotWorld(shift=0, seed=81)
        right_world = FlatSnapshotWorld(shift=47, seed=81)
        left = select_fluid_cast_site(left_world, decision_seed=2026)
        right = select_fluid_cast_site(right_world, decision_seed=2026)

        self.assertEqual(left.orientation, right.orientation)
        for first, second in (
            (left.anchor, right.anchor),
            (left.work_stance, right.work_stance),
            (left.catalyst_cell, right.catalyst_cell),
            (left.result_cell, right.result_cell),
        ):
            self.assertEqual(shift_cell(first, 47), second)
        self.assertEqual(
            tuple(shift_cell(cell, 47) for cell in left.backing_cells),
            right.backing_cells)
        self.assertEqual(
            tuple(shift_cell(cell, 47)
                  for cell in left.interactions[-1].ray_cells),
            right.interactions[-1].ray_cells)

    def test_protected_cells_are_excluded_from_all_cast_geometry(self):
        world = FlatSnapshotWorld()
        baseline = select_fluid_cast_site(
            world, decision_seed=19, casts_required=2)
        protected = frozenset((
            baseline.catalyst_cell,
            baseline.interactions[-1].target,
        ))

        plan = select_fluid_cast_site(
            world, decision_seed=19, casts_required=2,
            protected=protected)

        route_body = {
            body
            for x, y, z in plan.approach_stands
            for body in ((x, y - 1, z), (x, y, z), (x, y + 1, z))
        }
        interaction_rays = {
            cell for interaction in plan.interactions
            for cell in interaction.ray_cells
        }
        relevant = set(
            plan.excavation_cells + plan.support_cells
            + plan.backing_cells + plan.temporary_cells)
        relevant.update(route_body)
        relevant.update(interaction_rays)
        relevant.update(
            cell for step in plan.setup_placements
            for cell in (step.cell, step.support))
        self.assertTrue(protected.isdisjoint(relevant))

    def test_protected_solid_backing_is_reused_without_mutation(self):
        world = FlatSnapshotWorld()
        baseline = select_fluid_cast_site(
            world, decision_seed=1, search_radius=0,
            vertical_radius=0, hazard_clearance=0)
        immutable_backing = next(
            cell for cell in baseline.backing_cells
            if cell not in {
                (baseline.work_stance[0], baseline.work_stance[1] - 1,
                 baseline.work_stance[2])})

        plan = select_fluid_cast_site(
            world, decision_seed=1, search_radius=0,
            vertical_radius=0, hazard_clearance=0,
            protected=(immutable_backing,))

        self.assertEqual(plan.excavation_cells, baseline.excavation_cells)
        self.assertIn(immutable_backing, plan.backing_cells)
        self.assertNotIn(immutable_backing, plan.excavation_cells)
        self.assertFalse(plan.setup_placements)

    def test_missing_air_backing_is_built_from_exact_supported_steps(self):
        world = FlatSnapshotWorld()
        baseline = select_fluid_cast_site(
            world, decision_seed=1, search_radius=0,
            vertical_radius=0, hazard_clearance=0)
        work_support = (
            baseline.work_stance[0], baseline.work_stance[1] - 1,
            baseline.work_stance[2])
        missing = tuple(
            cell for cell in baseline.backing_cells
            if cell != work_support)
        world.put("air", *missing)

        plan = select_fluid_cast_site(
            world, decision_seed=1, search_radius=0,
            vertical_radius=0, hazard_clearance=0,
            setup_material="cobblestone")

        self.assertEqual(plan.kind, "constructed_contained_fluid_cast")
        self.assertEqual(
            {step.cell for step in plan.setup_placements}, set(missing))
        self.assertTrue(all(
            step.material == "cobblestone"
            and step.role == "cast_setup"
            and step.stance == plan.work_stance
            for step in plan.setup_placements))
        validate_support_order(
            type("Excavated", (), {
                "in_bounds": lambda _self, *coordinates:
                    world.in_bounds(*coordinates),
                "block": lambda _self, *coordinates: (
                    "air" if ((coordinates[0]
                               if len(coordinates) == 1 else coordinates)
                              in plan.excavation_cells)
                    else world.block(*coordinates)),
            })(),
            plan.setup_placements,
        )

    def test_missing_protected_backing_is_not_silently_filled(self):
        world = FlatSnapshotWorld()
        baseline = select_fluid_cast_site(
            world, decision_seed=1, search_radius=0,
            vertical_radius=0, hazard_clearance=0)
        missing = next(
            cell for cell in baseline.backing_cells
            if cell != (
                baseline.work_stance[0], baseline.work_stance[1] - 1,
                baseline.work_stance[2]))
        world.put("air", missing)

        plan = select_fluid_cast_site(
            world, decision_seed=1, search_radius=0,
            vertical_radius=0, hazard_clearance=0,
            protected=(missing,))

        self.assertNotIn(missing, plan.excavation_cells)
        self.assertNotIn(
            missing, {step.cell for step in plan.setup_placements})

    def test_protected_route_body_rejects_when_search_is_fixed(self):
        with self.assertRaises(PlanRejected) as raised:
            select_fluid_cast_site(
                FlatSnapshotWorld(), decision_seed=1,
                search_radius=0, vertical_radius=0,
                hazard_clearance=0, protected=((0, 1, 0),))

        self.assertEqual(raised.exception.reason, RejectReason.NO_PATH)
        self.assertEqual(raised.exception.examined, 8)

    def test_explicit_protected_passage_can_reach_a_cast_mold(self):
        world = FlatSnapshotWorld()
        passage = ((0, 1, 0), (0, 2, 0))

        plan = select_fluid_cast_site(
            world, decision_seed=1, search_radius=0, vertical_radius=0,
            hazard_clearance=0, protected=passage,
            allowed_passage=passage)

        self.assertEqual(plan.work_stance, (0, 1, 0))
        self.assertEqual(plan.approach_stands, ((0, 1, 0),))
        self.assertTrue(set(passage).isdisjoint(plan.excavation_cells))

    def test_cast_passage_must_also_be_protected(self):
        with self.assertRaisesRegex(ValueError, "subset of protected"):
            select_fluid_cast_site(
                FlatSnapshotWorld(), decision_seed=1,
                allowed_passage=((0, 1, 0),))

    def test_custom_cast_contract_is_data_not_portal_logic(self):
        plan = select_fluid_cast_site(
            FlatSnapshotWorld(), decision_seed=4, casts_required=3,
            catalyst_fluid="catalyst", reactant_fluid="reactant",
            result_block="product")

        self.assertEqual(plan.casts_required, 3)
        self.assertEqual(plan.catalyst_placement.material, "catalyst")
        self.assertEqual(plan.reactant_placement.material, "reactant")
        self.assertEqual(plan.result_block, "product")
        self.assertEqual(plan.setup_placements, ())

    def test_unexcavatable_mold_region_rejects_with_bounded_attempts(self):
        world = FlatSnapshotWorld()
        world.put("bedrock", *(
            (x, 0, z)
            for x in (-1, 0, 1) for z in (-1, 0, 1)
            if (x, z) != (0, 0)
        ))

        with self.assertRaises(PlanRejected) as raised:
            select_fluid_cast_site(
                world, decision_seed=1, search_radius=0,
                vertical_radius=0, hazard_clearance=0)

        self.assertEqual(raised.exception.reason, RejectReason.NO_PATH)
        self.assertEqual(raised.exception.examined, 8)

    def test_invalid_mold_geometry_never_spends_a_route_search(self):
        world = FlatSnapshotWorld()
        world.put("bedrock", *((x, 0, z)
                               for x in (-1, 0, 1)
                               for z in (-1, 0, 1)
                               if (x, z) != (0, 0)))

        with patch(
                "bedrock_rl.sdg.policy.structures."
                "plan_surface_path") as route:
            with self.assertRaises(PlanRejected):
                select_fluid_cast_site(
                    world, decision_seed=1, search_radius=0,
                    vertical_radius=0, hazard_clearance=0)

        route.assert_not_called()

    def test_cast_approach_routes_around_protected_geometry(self):
        world = FlatSnapshotWorld()
        world.put("bedrock", *((x, 0, z)
                               for x in range(-2, 3)
                               for z in range(-2, 3)))
        protected = {(0, 0, 1), (0, 1, 1), (0, 2, 1)}

        plan = select_fluid_cast_site(
            world, decision_seed=1, search_radius=6,
            hazard_clearance=0, protected=protected)

        route_cells = {
            body for x, y, z in plan.approach_stands
            for body in ((x, y - 1, z), (x, y, z), (x, y + 1, z))
        }
        self.assertTrue(protected.isdisjoint(route_cells))
        self.assertGreater(len(plan.approach_stands), 3)

    def test_exhaustive_route_failure_is_cached_per_work_stance(self):
        world = FlatSnapshotWorld()
        world.put("bedrock", *((x, 0, z)
                               for x in range(-2, 3)
                               for z in range(-2, 3)))

        with patch(
                "bedrock_rl.sdg.policy.structures."
                "plan_surface_path",
                side_effect=PlanRejected(RejectReason.NO_PATH)) as route:
            with self.assertRaises(PlanRejected):
                select_fluid_cast_site(
                    world, decision_seed=1, search_radius=6,
                    vertical_radius=0, hazard_clearance=0)

        goals = [call.args[2] for call in route.call_args_list]
        self.assertTrue(goals)
        self.assertEqual(len(goals), len(set(goals)))

    def test_already_open_cells_cannot_be_selected_as_fresh_mold(self):
        world = FlatSnapshotWorld()
        # Leave exactly one geometrically valid two-cell candidate. It is an
        # old cavity, not fresh ground; the remaining nearby cells are solid
        # but intentionally unbreakable so no alternative masks the result.
        world.put("bedrock", *(
            (x, 0, z)
            for x in (-1, 0, 1) for z in (-1, 0, 1)
            if (x, z) != (0, 0)
        ))
        world.put("air", (1, 0, 0), (1, 0, 1))

        with self.assertRaises(PlanRejected) as raised:
            select_fluid_cast_site(
                world, decision_seed=1, search_radius=0,
                vertical_radius=0, hazard_clearance=0)

        self.assertEqual(raised.exception.reason, RejectReason.NO_PATH)
        self.assertEqual(raised.exception.examined, 8)

    def test_interaction_ray_contract_rejects_truncated_evidence(self):
        with self.assertRaisesRegex(ValueError, "terminate"):
            InteractionRay(
                "mine_result", (0, 1, 0), (1, 0, 0),
                ((0, 2, 0),))


if __name__ == "__main__":
    unittest.main()
