import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bedrock_rl.sdg.policy.spatial import (
    PlanRejected,
    RejectReason,
    choose_mining_stance,
    clear_placement_ray,
    clear_selection_ray,
    clear_voxel_ray,
    is_excavation_stand,
    is_surface_stand,
    plan_safe_removal,
    plan_staircase_tunnel,
    plan_surface_path,
    safe_mining_stances,
    placement_face_point,
    voxel_ray,
)
from bedrock_rl.sdg.policy.world import SnapshotWorldMap
from tests.test_world_map import write_snapshot


class GridWorld:
    def __init__(self, low=(-8, -4, -8), high=(8, 8, 8), default="air"):
        self.low = low
        self.high = high
        self.default = default
        self.blocks = {}

    def block(self, cell):
        if not self.in_bounds(cell):
            return None
        return self.blocks.get(cell, self.default)

    def in_bounds(self, cell):
        return all(self.low[i] <= cell[i] <= self.high[i]
                   for i in range(3))

    def put(self, block, *cells):
        for cell in cells:
            self.blocks[cell] = block

    def floor(self, y=0):
        for x in range(self.low[0], self.high[0] + 1):
            for z in range(self.low[2], self.high[2] + 1):
                self.put("stone", (x, y, z))


class SurfacePlannerTests(unittest.TestCase):
    def test_real_snapshot_map_api_and_numeric_block_states(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "planner.bsnp"
            write_snapshot(snapshot)
            with SnapshotWorldMap(snapshot) as world:
                self.assertTrue(is_surface_stand(world, (10, 61, 0)))
                path = plan_surface_path(
                    world, (10, 61, 0), (11, 61, 0),
                    hazard_clearance=0)
                self.assertEqual(
                    path, ((10, 61, 0), (11, 61, 0)))
                # Water/lava BlockState ids remain non-passable even though
                # their concrete object type is unknown to this module.
                self.assertFalse(is_surface_stand(world, (10, 61, -1)))

    def test_surface_stand_needs_support_and_two_block_headroom(self):
        world = GridWorld()
        world.floor()
        self.assertTrue(is_surface_stand(world, (0, 1, 0)))
        world.put("stone", (0, 2, 0))
        self.assertFalse(is_surface_stand(world, (0, 1, 0)))
        world.put("air", (0, 2, 0), (0, 0, 0))
        self.assertFalse(is_surface_stand(world, (0, 1, 0)))

    def test_surface_astar_routes_around_wall_without_diagonals(self):
        world = GridWorld()
        world.floor()
        world.put("stone", (0, 1, 0), (0, 2, 0))
        path = plan_surface_path(world, (-2, 1, 0), (2, 1, 0))
        self.assertEqual(path[0], (-2, 1, 0))
        self.assertEqual(path[-1], (2, 1, 0))
        self.assertNotIn((0, 1, 0), path)
        self.assertTrue(all(
            abs(a[0] - b[0]) + abs(a[2] - b[2]) == 1
            and abs(a[1] - b[1]) <= 1
            for a, b in zip(path, path[1:])))

    def test_surface_descent_needs_old_level_head_clearance(self):
        world = GridWorld(low=(0, 0, 0), high=(1, 4, 0))
        world.floor()
        world.put("stone", (0, 1, 0), (1, 3, 0))
        start, goal = (0, 2, 0), (1, 1, 0)

        with self.assertRaises(PlanRejected):
            plan_surface_path(
                world, start, goal, hazard_clearance=0, node_cap=32)

        world.put("air", (1, 3, 0))
        self.assertEqual(
            plan_surface_path(
                world, start, goal, hazard_clearance=0, node_cap=32),
            (start, goal))
    def test_surface_path_steps_up_one_block(self):
        world = GridWorld(low=(-2, -2, -2), high=(3, 6, 2))
        world.floor()
        world.put("stone", (1, 1, 0), (2, 1, 0), (3, 1, 0))
        path = plan_surface_path(world, (0, 1, 0), (2, 2, 0))
        self.assertIn((1, 2, 0), path)
        self.assertEqual(path[-1], (2, 2, 0))

    def test_hazard_clearance_avoids_lava_adjacent_stands(self):
        world = GridWorld()
        world.floor()
        world.put("lava", (0, 1, 0))
        path = plan_surface_path(
            world, (-3, 1, 0), (3, 1, 0), hazard_clearance=1)
        self.assertTrue(all(abs(cell[2]) >= 2 for cell in path
                            if abs(cell[0]) <= 1))

    def test_node_cap_is_a_structured_rejection(self):
        world = GridWorld()
        world.floor()
        with self.assertRaises(PlanRejected) as raised:
            plan_surface_path(
                world, (-6, 1, -6), (6, 1, 6), node_cap=2)
        self.assertEqual(raised.exception.reason, RejectReason.NODE_CAP)
        self.assertEqual(raised.exception.examined, 0)
        self.assertIn("shortest possible route", raised.exception.detail)
        self.assertEqual(
            raised.exception.as_dict()["reason"], "node_cap")

    def test_seeded_tie_variance_is_reproducible(self):
        world = GridWorld()
        world.floor()
        world.put("stone", (0, 1, 0), (0, 2, 0))
        one = plan_surface_path(
            world, (-2, 1, 0), (2, 1, 0),
            decision_rng=random.Random(19))
        two = plan_surface_path(
            world, (-2, 1, 0), (2, 1, 0),
            decision_rng=random.Random(19))
        other = plan_surface_path(
            world, (-2, 1, 0), (2, 1, 0),
            decision_rng=random.Random(22))
        self.assertEqual(one, two)
        self.assertNotEqual(one, other)

    def test_surface_search_memoizes_stand_proofs(self):
        world = GridWorld()
        world.floor()
        world.put("stone", (0, 1, 0), (0, 2, 0))

        with patch(
                "bedrock_rl.sdg.policy.spatial.is_surface_stand",
                wraps=is_surface_stand) as checked:
            plan_surface_path(world, (-3, 1, 0), (3, 1, 0))

        feet = [call.args[1] for call in checked.call_args_list]
        self.assertEqual(len(feet), len(set(feet)))

    def test_surface_search_routes_around_forbidden_player_column(self):
        world = GridWorld()
        world.floor()
        forbidden = {(0, 0, 0), (0, 1, 0), (0, 2, 0)}

        path = plan_surface_path(
            world, (-2, 1, 0), (2, 1, 0),
            forbidden_cells=forbidden)

        route_cells = {
            body for x, y, z in path
            for body in ((x, y - 1, z), (x, y, z), (x, y + 1, z))
        }
        self.assertTrue(forbidden.isdisjoint(route_cells))
        self.assertNotIn((0, 1, 0), path)


class ExcavationPlannerTests(unittest.TestCase):
    def _solid_world(self):
        world = GridWorld(default="stone")
        # The endpoints are already-open tunnel cells; intermediate cells may
        # be solid and appear in the plan's explicit mining manifest.
        world.put("air", (0, 1, 0), (0, 2, 0), (4, -1, 0), (4, 0, 0))
        return world

    def test_staircase_reserves_body_and_lists_only_solid_mining_cells(self):
        world = self._solid_world()
        plan = plan_staircase_tunnel(
            world, (0, 1, 0), (4, -1, 0), hazard_clearance=0)
        self.assertEqual(plan.stands[0], (0, 1, 0))
        self.assertEqual(plan.stands[-1], (4, -1, 0))
        expected_reserved = {
            cell for feet in plan.stands
            for cell in (feet, (feet[0], feet[1] + 1, feet[2]))
        }
        expected_reserved.update(
            (right[0], right[1] + 2, right[2])
            for left, right in zip(plan.stands, plan.stands[1:])
            if right[1] < left[1])
        expected_reserved.update(
            (left[0], left[1] + 2, left[2])
            for left, right in zip(plan.stands, plan.stands[1:])
            if right[1] > left[1])
        self.assertEqual(set(plan.reserved_cells), expected_reserved)
        self.assertFalse(set(plan.reserved_cells) & set(plan.support_cells))
        self.assertFalse(set(plan.mine_cells) & set(plan.support_cells))
        self.assertTrue(all(world.block(cell) == "stone"
                            for cell in plan.mine_cells))
        self.assertTrue(all(
            abs(a[0] - b[0]) + abs(a[2] - b[2]) == 1
            and abs(a[1] - b[1]) <= 1
            for a, b in zip(plan.stands, plan.stands[1:])))

    def test_deep_tunnel_uses_safe_bounded_ramp_without_astar_explosion(self):
        world = GridWorld(
            low=(-20, -15, -20), high=(20, 20, 20), default="stone")
        start, goal = (0, 15, 0), (3, -10, 2)
        world.put("air", start, (0, 16, 0), goal, (3, -9, 2))

        plan = plan_staircase_tunnel(
            world, start, goal, hazard_clearance=0, node_cap=2,
            decision_rng=random.Random(7))

        self.assertEqual(plan.stands[0], start)
        self.assertEqual(plan.stands[-1], goal)
        self.assertFalse(set(plan.reserved_cells) & set(plan.support_cells))
        self.assertGreaterEqual(len(plan.stands) - 1, 25)
        self.assertTrue(all(
            (right[0], right[1] + 2, right[2]) in plan.reserved_cells
            for left, right in zip(plan.stands, plan.stands[1:])
            if right[1] < left[1]))
        self.assertTrue(all(
            (left[0], left[1] + 2, left[2]) in plan.reserved_cells
            for left, right in zip(plan.stands, plan.stands[1:])
            if right[1] > left[1]))

    def test_ascending_stair_reserves_origin_jump_headroom(self):
        world = GridWorld(default="stone")
        start, goal = (0, 1, 0), (1, 2, 0)
        world.put("air", start, (0, 2, 0), goal, (1, 3, 0))

        plan = plan_staircase_tunnel(
            world, start, goal, hazard_clearance=0)

        self.assertIn((0, 3, 0), plan.reserved_cells)
        self.assertIn((0, 3, 0), plan.mine_cells)

    def test_staircase_routes_around_unbreakable_transition_headroom(self):
        world = GridWorld(default="stone")
        start, goal = (0, 1, 0), (2, 2, 0)
        world.put("air", start, (0, 2, 0), goal, (2, 3, 0))
        # The shortest ascent would sweep this block. It is not part of the
        # ordinary feet/head stand predicate, so edge validation must reject
        # that transition and let A* choose another staircase.
        world.put("bedrock", (0, 3, 0))

        plan = plan_staircase_tunnel(
            world, start, goal, hazard_clearance=0, node_cap=1_000)

        self.assertNotIn((0, 3, 0), plan.reserved_cells)
        self.assertEqual(plan.stands[0], start)
        self.assertEqual(plan.stands[-1], goal)

    def test_surface_ascent_refuses_solid_origin_jump_headroom(self):
        world = GridWorld(default="stone")
        start, goal = (0, 1, 0), (1, 2, 0)
        world.put("air", start, (0, 2, 0), goal, (1, 3, 0))

        with self.assertRaises(PlanRejected):
            plan_surface_path(
                world, start, goal, hazard_clearance=0, node_cap=8)

    def test_staircase_retries_a_support_conflicting_astar_route(self):
        world = GridWorld(default="stone")
        start, goal = (0, 1, 0), (2, 2, 0)
        conflicting = (
            start, (1, 2, 0), (0, 2, 0), (1, 2, 0), goal)
        safe = (start, (1, 1, 0), goal)

        with patch(
                "bedrock_rl.sdg.policy.spatial."
                "_staircase_fast_paths", return_value=()), patch(
                "bedrock_rl.sdg.policy.spatial._astar",
                side_effect=(conflicting, safe)) as astar:
            plan = plan_staircase_tunnel(
                world, start, goal, hazard_clearance=0)

        self.assertEqual(plan.stands, safe)
        self.assertEqual(astar.call_count, 2)
        self.assertFalse(
            set(plan.reserved_cells).intersection(plan.support_cells))

    def test_staircase_refuses_bedrock_and_fluids(self):
        world = GridWorld(
            low=(0, -2, 0), high=(4, 4, 0), default="stone")
        world.put("air", (0, 1, 0), (0, 2, 0), (4, 1, 0), (4, 2, 0))
        world.put("bedrock", (1, 1, 0), (1, 2, 0))
        world.put("water", (2, 1, 0), (2, 2, 0))
        world.put("lava", (3, 1, 0), (3, 2, 0))
        with self.assertRaises(PlanRejected) as raised:
            plan_staircase_tunnel(
                world, (0, 1, 0), (4, 1, 0), hazard_clearance=0)
        self.assertEqual(raised.exception.reason, RejectReason.NO_PATH)

    def test_zero_clearance_allows_adjacency_not_body_or_support_fluid(self):
        world = GridWorld(default="stone")
        feet = (0, 1, 0)
        world.put("air", feet, (0, 2, 0))
        world.put("water", (1, 1, 0))
        self.assertTrue(is_excavation_stand(
            world, feet, hazard_clearance=0))
        self.assertFalse(is_excavation_stand(
            world, feet, hazard_clearance=1))
        for occupied in (feet, (0, 2, 0), (0, 0, 0)):
            with self.subTest(occupied=occupied):
                previous = world.blocks.get(occupied, world.default)
                world.put("water", occupied)
                self.assertFalse(is_excavation_stand(
                    world, feet, hazard_clearance=0))
                world.put(previous, occupied)

    def test_staircase_rejects_hazard_adjacency(self):
        world = GridWorld(default="stone")
        world.put("air", (0, 1, 0), (0, 2, 0), (4, 1, 0), (4, 2, 0))
        world.put("lava", (2, 1, 1))
        plan = plan_staircase_tunnel(
            world, (0, 1, 0), (4, 1, 0),
            hazard_clearance=1, node_cap=1_000)
        self.assertTrue(all(
            max(abs(cell[0] - 2), abs(cell[1] - 1), abs(cell[2] - 1)) > 1
            for cell in plan.reserved_cells))


class MiningStanceTests(unittest.TestCase):
    def test_placement_ray_proves_the_exact_support_face(self):
        world = GridWorld()
        world.floor()
        stand = (0, 1, 0)
        destination = (2, 1, 0)
        support = (2, 0, 0)

        self.assertEqual(
            placement_face_point(support, destination),
            (2.5, 0.99, 0.5),
        )
        self.assertEqual(
            placement_face_point(
                support, destination, observer=(0.5, 2.62, 0.8)),
            (2.12, 0.99, 0.8),
        )
        self.assertTrue(clear_placement_ray(
            world, stand, destination, support,
            max_reach=4.0, origin_margin=0.04))

        world.put("tallgrass", stand)
        self.assertFalse(clear_placement_ray(
            world, stand, destination, support,
            max_reach=4.0, origin_margin=0.04))
        world.put("air", stand)

        world.put("stone", (1, 1, 0))
        self.assertFalse(clear_placement_ray(
            world, stand, destination, support,
            max_reach=4.0, origin_margin=0.04))

    def test_placement_face_requires_face_adjacency(self):
        with self.assertRaisesRegex(ValueError, "face-adjacent"):
            placement_face_point((0, 0, 0), (1, 1, 0))

    def test_mining_stance_excludes_ray_occluding_passable_plant(self):
        world = GridWorld()
        world.floor()
        target = (1, 0, 0)
        world.put("stone", target)
        world.put("tallgrass", (0, 1, 0))

        stances = safe_mining_stances(
            world, target, hazard_clearance=0)

        self.assertNotIn((0, 1, 0), {stance.feet for stance in stances})

    def test_selection_ray_distinguishes_plants_from_fluids(self):
        world = GridWorld()
        world.put("stone", (2, 1, 0))
        start, end = (0.5, 2.5, 0.5), (2.5, 1.5, 0.5)
        world.put("tallgrass", (1, 1, 0))
        self.assertFalse(clear_selection_ray(
            world, start, end, target=(2, 1, 0)))
        world.put("water", (1, 1, 0))
        self.assertTrue(clear_selection_ray(
            world, start, end, target=(2, 1, 0)))

    def test_voxel_ray_is_finite_and_includes_both_ends(self):
        ray = voxel_ray((0.5, 1.5, 0.5), (3.5, 2.5, 0.5))
        self.assertEqual(ray[0], (0, 1, 0))
        self.assertEqual(ray[-1], (3, 2, 0))
        self.assertEqual(len(ray), len(set(ray)))

    def test_clear_ray_detects_occluder(self):
        world = GridWorld()
        world.put("diamond_ore", (3, 2, 0))
        start = (0.5, 2.62, 0.5)
        end = (3.5, 2.5, 0.5)
        self.assertTrue(clear_voxel_ray(
            world, start, end, target=(3, 2, 0)))
        world.put("stone", (2, 2, 0))
        self.assertFalse(clear_voxel_ray(
            world, start, end, target=(3, 2, 0)))

    def test_mining_ray_envelope_rejects_corner_grazing_occluder(self):
        world = GridWorld(
            low=(4, 84, 1), high=(10, 90, 7))
        target = (8, 87, 3)
        start = (6.5, 87.62, 5.5)
        end = (8.5, 87.5, 3.5)
        world.put("log", target)
        # The exact center DDA crosses the x/z boundary simultaneously and
        # skips this leaf. A player only 0.04 blocks off center hits it first.
        world.put("leaves", (6, 87, 4))

        self.assertTrue(clear_voxel_ray(
            world, start, end, target=target))
        self.assertFalse(clear_voxel_ray(
            world, start, end, target=target, origin_margin=0.04))

    def test_mining_candidates_are_within_reach_and_ray_verified(self):
        world = GridWorld()
        world.floor()
        target = (3, 2, 0)
        world.put("diamond_ore", target)
        candidates = safe_mining_stances(world, target)
        self.assertTrue(candidates)
        self.assertTrue(all(candidate.distance <= 5.0
                            for candidate in candidates))
        self.assertTrue(all(
            (candidate.feet[0], candidate.feet[1] - 1,
             candidate.feet[2]) != target
            for candidate in candidates))
        self.assertEqual(candidates[0], choose_mining_stance(world, target))
        self.assertTrue(clear_voxel_ray(
            world, candidates[0].eye,
            (target[0] + 0.5, target[1] + 0.5, target[2] + 0.5),
            target=target,
        ))

    def test_mining_stance_rejects_blocked_or_unsafe_target(self):
        world = GridWorld(low=(-4, -2, -4), high=(4, 6, 4))
        world.floor()
        target = (0, 2, 0)
        world.put("diamond_ore", target)
        # A complete shell blocks every eye ray to the target.
        for x in range(-1, 2):
            for y in range(1, 4):
                for z in range(-1, 2):
                    if (x, y, z) != target:
                        world.put("stone", (x, y, z))
        with self.assertRaises(PlanRejected) as raised:
            choose_mining_stance(world, target)
        self.assertEqual(
            raised.exception.reason, RejectReason.UNREACHABLE_TARGET)

        world.put("lava", target)
        with self.assertRaises(PlanRejected) as unsafe:
            safe_mining_stances(world, target)
        self.assertEqual(unsafe.exception.reason, RejectReason.UNSAFE_CELL)


class SafeRemovalTests(unittest.TestCase):
    def test_can_require_exact_water_seal_but_never_allow_lava(self):
        world = GridWorld()
        world.floor()
        target = (2, 1, 0)
        water = (2, 2, 0)
        world.put("obsidian", target)
        world.put("water", water)

        proof = plan_safe_removal(
            world, target, sealable_fluids=("water",))

        self.assertEqual(proof.fluid_seal_cells, (water,))
        self.assertEqual(proof.rule, "known_faces_sealable_fluid_v1")
        world.put("lava", water)
        with self.assertRaises(PlanRejected):
            plan_safe_removal(
                world, target, sealable_fluids=("water",))

    def test_proof_records_known_faces_and_route_support(self):
        world = GridWorld()
        world.floor()
        target = (2, 1, 0)
        world.put("log", target)
        proof = plan_safe_removal(
            world, target, route_stands=((0, 1, 0), (1, 1, 0)))
        self.assertEqual(proof.target, target)
        self.assertEqual(len(proof.inspected_cells), 7)
        self.assertIn((1, 0, 0), proof.protected_support_cells)
        self.assertEqual(proof.to_dict()["target"], [2, 1, 0])

    def test_rejects_side_or_underlying_lava(self):
        for lava in ((3, 1, 0), (2, 0, 0)):
            with self.subTest(lava=lava):
                world = GridWorld()
                world.floor()
                target = (2, 1, 0)
                world.put("log", target)
                world.put("lava", lava)
                with self.assertRaises(PlanRejected) as raised:
                    plan_safe_removal(world, target)
                self.assertEqual(
                    raised.exception.reason, RejectReason.UNSAFE_REMOVAL)
                self.assertIn("fluid or hazard", raised.exception.detail)

    def test_rejects_required_route_support(self):
        world = GridWorld()
        world.floor()
        target = (1, 0, 0)
        world.put("stone", target)
        with self.assertRaises(PlanRejected) as raised:
            plan_safe_removal(
                world, target, route_stands=((1, 1, 0),))
        self.assertEqual(
            raised.exception.reason, RejectReason.UNSAFE_REMOVAL)
        self.assertIn("supports", raised.exception.detail)

    def test_rejects_unknown_face_neighbor(self):
        world = GridWorld(low=(0, 0, 0), high=(2, 2, 2))
        target = (0, 1, 1)
        world.put("stone", target)
        with self.assertRaises(PlanRejected) as raised:
            plan_safe_removal(world, target)
        self.assertEqual(
            raised.exception.reason, RejectReason.UNSAFE_REMOVAL)
        self.assertIn("outside known", raised.exception.detail)


if __name__ == "__main__":
    unittest.main()
