import math
import unittest
from unittest.mock import patch

from bedrock_rl.sdg.policy import (
    MinecraftController, PolicyVeto, SurvivalBehaviors, SurvivalPolicy,
    WallTorchRouteMixin,
)
from bedrock_rl.sdg import (
    MinecraftController as CompatibleMinecraftController,
    WallTorchRouteMixin as CompatibleWallTorchRouteMixin,
)
from bedrock_rl.sdg.generation import Case
from bedrock_rl.env.journal import FLUID_VIEWPOINT


class FakeEvent(dict):
    def __init__(self, name, **values):
        super().__init__(values)
        self.name = name


class FakeEnv:
    def __init__(self, *, blocks=(), hotbar_ids=None, hotbar_counts=None,
                 selected=0, journal=(), x=0.0, y=64.0, z=0.0,
                 container=0):
        self.obs = {
            "x": x, "y": y, "z": z, "yaw": 0.0, "pitch": 0.0,
            "blocks": list(blocks),
            "hotbar_ids": list(hotbar_ids or [0] * 9),
            "hotbar_counts": list(hotbar_counts or [0] * 9),
            "hotbar_meta": [0] * 9,
            "hotbar_sel": selected,
            "container": container,
        }
        self.journal = list(journal)
        self.angles = {}

    def rel_angles_to(self, x, y, z):
        cell = (x, y, z)
        if cell in self.angles:
            return self.angles[cell]
        distance = math.dist(
            (self.obs["x"], self.obs["y"], self.obs["z"]),
            (float(x), float(y), float(z)))
        return 0.0, 0.0, distance


def controller(*, hazards=True):
    value = MinecraftController(
        movement={
            "min_blocks": 2.0, "max_blocks": 2.0,
            "min_pause_ticks": 0, "max_pause_ticks": 0,
        },
        behaviors={"hazards": hazards})
    value.reset(None, None, Case(0, 7, 11, {}))
    return value


class _LitRoutePolicy(WallTorchRouteMixin, SurvivalPolicy):
    def reset(self):
        super().reset(None, None, Case(0, 7, 11, {}))
        self.reset_route_lighting()


class WallTorchRouteTests(unittest.TestCase):
    def test_due_light_recovers_a_backpacked_torch(self):
        class LitController(WallTorchRouteMixin, MinecraftController):
            pass

        env = FakeEnv()
        env.obs.update(
            inventory_ids=[0] * 9 + [50] + [0] * 26,
            inventory_counts=[0] * 9 + [48] + [0] * 26,
            inventory_meta=[0] * 36,
        )
        policy = LitController(behaviors={"lighting": True})
        policy.reset(None, None, Case(0, 7, 11, {}))
        policy.reset_route_lighting()
        policy.start_route_lighting(env, immediate=True)

        actions = policy.light_route(env)

        self.assertEqual(actions[0], {
            "action": "key", "k": "e", "ticks": 1})
        self.assertEqual(actions[-1], {
            "action": "key", "k": "e", "ticks": 1})

    def test_failed_wall_immediately_tries_another_certified_support(self):
        policy = _LitRoutePolicy()
        policy.reset()
        policy._lighting_position = (0.5, 64.0, 0.5)
        policy.lighting_distance = 8.0
        policy.pending_light = ((0, 65, -1), (1, 65, -1),
                                (0.51, 65.0, -1.0), "wall")
        policy.light_aim_attempts = 4
        env = FakeEnv(
            x=0.5, y=64.0, z=0.5,
            hotbar_ids=[50] + [0] * 8,
            hotbar_counts=[2] + [0] * 8)
        next_site = ((0, 65, -1), (-1, 65, -1),
                     (-0.51, 65.0, -1.0), "wall")
        env.angles[next_site[2]] = (20.0, 0.0, 2.0)

        with patch.object(
                policy, "choose_route_light_site",
                return_value=next_site) as choose:
            self.assertEqual(
                policy.light_route(env), [{"action": "wait", "ticks": 1}])
            self.assertEqual(policy.rejected_light_supports,
                             {(1, 65, -1)})
            actions = policy.light_route(env)

        choose.assert_called_once_with(env)
        self.assertEqual(policy.pending_light, next_site)
        self.assertTrue(actions)

    def test_route_odometer_records_only_verified_placements(self):
        policy = _LitRoutePolicy()
        policy.reset()
        start = FakeEnv(x=0.0, y=64.0, z=0.0)
        policy.start_route_lighting(start, immediate=False)
        policy.observe_route_travel(FakeEnv(x=7.0, y=64.0, z=0.0))
        self.assertFalse(policy.route_light_due())
        policy.observe_route_travel(FakeEnv(x=8.5, y=64.0, z=0.0))
        self.assertTrue(policy.route_light_due())
        self.assertFalse(policy.route_light_overdue())
        policy.observe_route_travel(FakeEnv(x=11.5, y=64.0, z=0.0))

        self.assertTrue(policy.route_light_due())
        self.assertFalse(policy.route_light_overdue())
        policy.observe_route_travel(FakeEnv(x=13.5, y=64.0, z=0.0))
        self.assertTrue(policy.route_light_overdue())
        self.assertEqual(policy.verified_light_odometer, [])

        policy.route_light_placed()
        self.assertEqual(policy.verified_light_odometer, [13.0])
        self.assertEqual(policy.lighting_distance, 0.0)

    def test_route_odometer_does_not_count_backtracking_or_loops(self):
        policy = _LitRoutePolicy()
        policy.reset()
        policy.start_route_lighting(
            FakeEnv(x=0.5, y=64.0, z=0.5), immediate=False)
        for x, z in ((1.5, 0.5), (1.5, 1.5), (0.5, 1.5),
                     (0.5, 0.5), (0.5, 1.5), (1.5, 1.5),
                     (1.5, 0.5)):
            policy.observe_route_travel(FakeEnv(x=x, y=64.0, z=z))

        # Four unique unit edges, despite seven physical transitions.
        self.assertEqual(policy.lighting_distance, 4.0)
        self.assertEqual(policy.route_distance_total, 4.0)
        self.assertFalse(policy.route_light_due())

    def test_route_odometer_does_not_double_count_stair_settling(self):
        policy = _LitRoutePolicy()
        policy.reset()
        policy.start_route_lighting(
            FakeEnv(x=0.5, y=60.0, z=0.5), immediate=False)
        staircase = (
            (1.5, 60.0), (1.5, 61.0), (0.5, 61.0),
            (0.5, 62.0), (0.5, 61.0), (1.5, 61.0),
            (1.5, 60.0), (0.5, 60.0),
        )
        for x, y in staircase:
            policy.observe_route_travel(FakeEnv(x=x, y=y, z=0.5))

        # The four upward/forward edges are novel; settling back down them is
        # not another four blocks of mine progress.
        self.assertEqual(policy.lighting_distance, 4.0)
        self.assertFalse(policy.route_light_due())

    def test_existing_light_never_stretches_the_verified_interval(self):
        policy = _LitRoutePolicy()
        policy.reset()
        policy.start_route_lighting(
            FakeEnv(x=8.5, y=64.0, z=0.5), immediate=False)
        policy.observe_route_travel(FakeEnv(x=10.5, y=64.0, z=0.5))
        self.assertEqual(policy.lighting_distance, 2.0)

        policy.shown_lights.add((5, 65, 0))
        policy.observe_route_travel(FakeEnv(x=6.5, y=64.0, z=0.5))

        self.assertEqual(policy.lighting_distance, 6.0)
        self.assertFalse(policy.route_light_due())

    def test_consecutive_verified_light_odometer_gaps_never_exceed_ten(self):
        policy = _LitRoutePolicy()
        policy.reset()
        policy.behaviors = SurvivalBehaviors.parse({
            "lighting": {"every_blocks": 10, "grace_blocks": 0},
        })
        policy.start_route_lighting(
            FakeEnv(x=0.5, y=64.0, z=0.5), immediate=False)
        for base in (0, 10, 20):
            for offset in range(1, 11):
                policy.observe_route_travel(FakeEnv(
                    x=base + offset + 0.5, y=64.0, z=0.5))
            self.assertTrue(policy.route_light_due())
            self.assertFalse(policy.route_light_overdue())
            policy.route_light_placed()

        odometer = [0.0, *policy.verified_light_odometer]
        self.assertTrue(all(
            right - left <= 10.0
            for left, right in zip(odometer, odometer[1:])))

    def test_selects_only_a_prior_side_wall_not_the_path_ahead(self):
        policy = _LitRoutePolicy()
        policy.reset()
        env = FakeEnv(
            x=0.5, y=64.0, z=0.5,
            blocks=[(1, -1, 65, -1), (1, 1, 65, -1)])
        policy.recent_route_cells = [(0, 64, -1), (0, 64, 2)]

        self.assertEqual(policy.choose_route_light_site(env), (
            (0, 65, -1), (-1, 65, -1),
            (-0.51, 65, -1.0), "wall"))

    def test_due_light_needs_exact_face_and_verifies_wall_metadata(self):
        policy = _LitRoutePolicy()
        policy.reset()
        policy.start_route_lighting(FakeEnv(), immediate=True)
        policy.pending_light = (
            (0, 65, -1), (-1, 65, -1), (-0.55, 65, -1.0), "wall")
        env = FakeEnv(
            hotbar_ids=[50] + [0] * 8,
            hotbar_counts=[2] + [0] * 8,
            journal=[FakeEvent(
                "click_target", block=1, face=2, x=-1, y=65, z=-1)])

        self.assertEqual(policy.light_route(env), [
            {"action": "right_click"}])

        env.journal.append(FakeEvent(
            "block_placed", block=50, meta=2, x=0, y=65, z=-1))
        self.assertEqual(policy.light_route(env), [
            {"action": "wait", "ticks": 8}])
        self.assertEqual(policy.lighting_distance, 0.0)

        policy.pending_light = (
            (0, 65, -2), (-1, 65, -2), (-0.55, 65, -2.0), "wall")
        env.journal.append(FakeEvent(
            "block_placed", block=50, meta=5, x=0, y=65, z=-2))
        self.assertEqual(policy.light_route(env), [
            {"action": "wait", "ticks": 8}])

        policy.pending_light = (
            (0, 65, -3), (-1, 65, -3), (-0.55, 65, -3.0), "wall")
        env.journal.append(FakeEvent(
            "block_placed", block=50, meta=0, x=0, y=65, z=-3))
        veto = policy.light_route(env)
        self.assertIsInstance(veto, PolicyVeto)
        self.assertEqual(veto.code, "route_light_not_wall_mounted")

    def test_wrong_route_light_landing_is_structured_case_veto(self):
        policy = _LitRoutePolicy()
        policy.reset()
        policy.pending_light = (
            (0, 65, -2), (-1, 65, -2), (-0.55, 65, -2.0), "wall")
        env = FakeEnv(journal=[FakeEvent(
            "block_placed", block=50, meta=2, x=1, y=65, z=-2)])

        veto = policy.light_route(env)

        self.assertIsInstance(veto, PolicyVeto)
        self.assertEqual(veto.code, "route_light_landed_elsewhere")


class MinecraftControllerTests(unittest.TestCase):
    def test_full_inventory_observation_recovers_unjournaled_backpack_stack(self):
        env = FakeEnv()
        env.obs.update(
            inventory_ids=[0] * 9 + [280] + [0] * 26,
            inventory_counts=[0] * 9 + [2] + [0] * 26,
            inventory_meta=[0] * 36,
        )
        value = controller()

        self.assertEqual(value.observed_inventory_slot(env, (280,)), 9)
        self.assertEqual(
            value.observed_inventory_stack(env, (280,)), (9, 2, 0))
        self.assertEqual(
            value.recover_hotbar(env, "stick"),
            [
                {"action": "key", "k": "e", "ticks": 1},
                {"action": "left_click", "coordinate": [333, 540]},
                {"action": "left_click", "coordinate": [333, 782]},
                {"action": "key", "k": "e", "ticks": 1},
            ])

    def test_sdg_keeps_the_controller_compatibility_import(self):
        self.assertIs(CompatibleMinecraftController, MinecraftController)
        self.assertIs(CompatibleWallTorchRouteMixin, WallTorchRouteMixin)

    def test_public_event_and_inventory_queries(self):
        env = FakeEnv(
            hotbar_ids=[264, 4, 264, 0, 0, 0, 0, 0, 0],
            hotbar_counts=[2, 9, 1, 0, 0, 0, 0, 0, 0],
            journal=[
                FakeEvent("block_broken", block=56, x=1, y=6, z=2),
                FakeEvent("block_placed", block=50, x=1, y=7, z=2),
            ])
        control = controller()

        self.assertEqual(control.item_count(env, "diamond"), 3)
        self.assertEqual(control.event_count(
            env, "block_broken", block=56), 1)
        self.assertTrue(control.broken(env, (1, 6, 2), "diamond_ore"))
        self.assertTrue(control.placed(env, (1, 7, 2), "torch"))
        self.assertFalse(control.placed(
            env, (1, 7, 2), "torch", since=len(env.journal)))

    def test_face_and_move_never_emits_an_observed_unsafe_step(self):
        control = controller()
        env = FakeEnv(blocks=[(11, 1, 64, 0)])

        veto = control.face_and_move_to(env, 4.0, 0.0)

        self.assertIsInstance(veto, PolicyVeto)
        self.assertEqual(veto.code, "observed_hazard")

    def test_face_and_move_is_bounded_and_can_sneak_at_an_edge(self):
        control = controller(hazards=False)
        env = FakeEnv()

        actions = control.face_and_move_to(
            env, 1.0, 0.0, tolerance=0.1, sneak=True)

        moves = [action for action in actions
                 if action.get("action") == "key"
                 and isinstance(action.get("k"), list)]
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0]["k"], ["w", "sneak"])
        self.assertLessEqual(moves[0]["ticks"], 6)

    def test_known_step_up_jumps_proactively(self):
        control = controller(hazards=False)

        actions = control.face_and_move_to(
            FakeEnv(), 1.0, 0.0, tolerance=0.1, step_up=True)

        move = next(action for action in actions
                    if action.get("action") == "key")
        self.assertEqual(move["k"], ["w", "space"])

    def test_fine_movement_cap_bounds_hold_without_changing_profile(self):
        control = controller(hazards=False)

        actions = control.face_and_move_to(
            FakeEnv(), 3.0, 0.0, tolerance=0.1,
            movement_cap_blocks=0.3)

        move = next(action for action in actions
                    if action.get("action") == "key")
        self.assertLessEqual(move["ticks"], 2)

    def test_descending_retry_can_disable_jump_recovery(self):
        control = controller(hazards=False)
        env = FakeEnv()

        control.face_and_move_to(
            env, 1.0, 0.0, tolerance=0.1,
            sequential_hop=True, allow_jump=False)
        actions = control.face_and_move_to(
            env, 1.0, 0.0, tolerance=0.1,
            sequential_hop=True, allow_jump=False)

        self.assertFalse(any(
            action.get("k") == "space"
            or (isinstance(action.get("k"), list)
                and "space" in action["k"])
            for action in actions))

    def test_select_and_recover_use_only_public_hotbar_and_journal(self):
        control = controller()
        env = FakeEnv(
            hotbar_ids=[4, 1, 2, 3, 5, 6, 7, 8, 9],
            hotbar_counts=[8] * 9,
            selected=1,
            journal=[FakeEvent(
                "slot_changed", slot=12, item=264, count=1)])

        selected = control.select(env, "cobblestone")
        recovered = control.recover_hotbar(
            env, "diamond", replace=("cobblestone",))

        self.assertEqual(selected[0]["k"], "1")
        self.assertEqual(recovered[0], {
            "action": "key", "k": "e", "ticks": 1})
        self.assertEqual(recovered[-1], {
            "action": "key", "k": "e", "ticks": 1})

    def test_recipe_preparation_recovers_public_backpack_ingredient(self):
        from bedrock_rl.env import gui_layout

        control = controller()
        env = FakeEnv(
            hotbar_ids=[265, 270, 1, 2, 3, 4, 5, 6, 7],
            hotbar_counts=[1] * 9,
            journal=[FakeEvent(
                "slot_changed", slot=12, item=318, count=1, meta=0)])

        actions = control.prepare_recipe_ingredients(
            env, "flint_and_steel", 2, replace=("wooden_pickaxe",))

        self.assertEqual(actions, (
            gui_layout.open_screen()
            + gui_layout.click_slot(12, "left", "player")
            + gui_layout.click_slot(1, "left", "player")
            + gui_layout.close_screen()))

    def test_recipe_preparation_requires_a_complete_public_recipe(self):
        control = controller()
        env = FakeEnv(
            hotbar_ids=[265] + [0] * 8,
            hotbar_counts=[1] + [0] * 8)

        self.assertIsNone(control.prepare_recipe_ingredients(
            env, "flint_and_steel", 2))

    def test_hotbar_recovery_honors_multi_item_minimum(self):
        from bedrock_rl.env import gui_layout

        control = controller()
        env = FakeEnv(
            hotbar_ids=[274, 270, 1, 2, 3, 4, 5, 6, 7],
            hotbar_counts=[2] + [1] * 8,
            journal=[FakeEvent(
                "slot_changed", slot=12, item=274, count=1, meta=0)])

        actions = control.recover_hotbar(
            env, "stone_pickaxe", replace=("wooden_pickaxe",), minimum=3)

        self.assertEqual(actions, (
            gui_layout.open_screen()
            + gui_layout.click_slot(12, "left", "player")
            + gui_layout.click_slot(0, "left", "player")
            + gui_layout.close_screen()))

    def test_exact_mine_attacks_only_a_proven_target(self):
        control = controller()
        env = FakeEnv(
            hotbar_ids=[257] + [0] * 8,
            hotbar_counts=[1] + [0] * 8,
            journal=[FakeEvent(
                "click_target", block=56, face=1, x=1, y=64, z=0)])
        env.angles[(1, 64, 0)] = (0.0, 0.0, 2.0)

        actions = control.mine_exact(
            env, (1, 64, 0), tool="iron_pickaxe", block="diamond_ore")

        self.assertEqual(actions, [{
            "action": "left_click_hold", "ticks": 80}])

        env.journal = [FakeEvent(
            "click_target", block=1, face=1, x=0, y=64, z=0)]
        veto = control.mine_exact(
            env, (1, 64, 0), tool="iron_pickaxe", block="diamond_ore")
        self.assertIsInstance(veto, PolicyVeto)
        self.assertEqual(veto.code, "mine_target_unproven")

    def test_exact_mine_observes_tool_selection_before_attack(self):
        control = controller()
        env = FakeEnv(
            hotbar_ids=[5, 270] + [0] * 7,
            hotbar_counts=[1, 1] + [0] * 7,
            selected=0,
            journal=[FakeEvent(
                "click_target", block=1, face=1, x=1, y=64, z=0)])
        env.angles[(1, 64, 0)] = (0.0, 0.0, 2.0)

        select = control.mine_exact(
            env, (1, 64, 0), tool="wooden_pickaxe", block="stone")

        self.assertTrue(select)
        self.assertNotIn("left_click_hold", {
            action["action"] for action in select})
        env.obs["hotbar_sel"] = 1
        attack = control.mine_exact(
            env, (1, 64, 0), tool="wooden_pickaxe", block="stone")
        self.assertEqual(attack, [{
            "action": "left_click_hold", "ticks": 80}])

    def test_bucket_target_is_public_without_changing_ordinary_click_target(self):
        control = controller()
        env = FakeEnv(journal=[
            FakeEvent(
                "click_target", block=1, face=4, x=2, y=61, z=3),
            FakeEvent(
                "bucket_target", block=9, x=2, y=62, z=3,
                meta=0, slot=4),
        ])

        self.assertEqual(control.click_target(env), (1, (2, 61, 3)))
        self.assertEqual(control.events(env, "bucket_target"), [env.journal[1]])
        self.assertEqual(
            (env.journal[1]["block"], env.journal[1]["x"],
             env.journal[1]["y"], env.journal[1]["z"],
             env.journal[1]["meta"], env.journal[1]["slot"]),
            (9, 2, 62, 3, 0, 4))

    def test_exact_mine_recenters_inside_tight_stance_envelope(self):
        control = controller(hazards=False)
        env = FakeEnv(
            x=6.535, y=86.0, z=5.5,
            hotbar_ids=[257] + [0] * 8,
            hotbar_counts=[1] + [0] * 8,
            journal=[FakeEvent(
                "click_target", block=18, face=1, x=6, y=87, z=4)])
        env.angles[(8, 87, 3)] = (0.0, 0.0, 3.0)

        with patch.object(
                control, "recenter_interaction_stance",
                return_value=[{"action": "key", "k": "w", "ticks": 1}]
                ) as recenter:
            actions = control.mine_exact(
                env, (8, 87, 3), tool="iron_pickaxe", block="log",
                stand=(6.5, 5.5))

        self.assertIsInstance(actions, list)
        recenter.assert_called_once_with(
            env, 6.5, 5.5, tolerance=0.02)

    def test_exact_mine_aims_before_recentering_wrong_camera_trace(self):
        control = controller(hazards=False)
        env = FakeEnv(
            x=4.54, y=84.0, z=4.514,
            hotbar_ids=[270] + [0] * 8,
            hotbar_counts=[1] + [0] * 8,
            journal=[FakeEvent(
                "click_target", block=58, face=1, x=3, y=84, z=4)])
        env.angles[(5, 83, 4)] = (103.0, -40.0, 2.0)

        with patch.object(
                control, "recenter_interaction_stance") as recenter:
            actions = control.mine_exact(
                env, (5, 83, 4), tool="wooden_pickaxe",
                stand=(4.5, 4.5))

        self.assertEqual(actions[0]["action"], "mouse_move")
        recenter.assert_not_called()

    def test_exact_placement_recenters_at_its_planned_stance(self):
        control = controller(hazards=False)
        env = FakeEnv(
            x=0.54, y=64.0, z=0.5,
            hotbar_ids=[61] + [0] * 8,
            hotbar_counts=[1] + [0] * 8,
            journal=[FakeEvent(
                "click_target", block=1, face=5, x=1, y=64, z=0)])

        with patch.object(
                control, "recenter_interaction_stance",
                return_value=[{"action": "key", "k": "a", "ticks": 1}]
                ) as recenter:
            actions = control.place_workstation(
                env, "furnace", (1, 64, 0), (1, 63, 0),
                stand=(0.5, 0.5))

        self.assertEqual(actions, [
            {"action": "key", "k": "a", "ticks": 1}])
        recenter.assert_called_once_with(
            env, 0.5, 0.5, tolerance=0.02)

    def test_exact_placement_aims_at_visible_support_face_region(self):
        control = controller(hazards=False)
        env = FakeEnv(
            x=0.5, y=64.0, z=0.5,
            hotbar_ids=[61] + [0] * 8,
            hotbar_counts=[1] + [0] * 8,
            journal=[FakeEvent(
                "click_target", block=1, face=5, x=1, y=64, z=0)])
        aimed = [{"action": "mouse_move", "dx": 2, "dy": 1}]

        with patch.object(
                control, "aim_point", return_value=aimed) as aim:
            self.assertEqual(control.place_workstation(
                env, "furnace", (1, 64, 0), (1, 63, 0),
                stand=(0.5, 0.5)), aimed)

        aim.assert_called_once_with(
            env, (0.6200000000000001, 63.49, 0.0))

    def test_exact_top_face_placement_jumps_before_public_click_proof(self):
        control = controller(hazards=False)
        env = FakeEnv(
            x=0.5, y=64.0, z=0.5,
            hotbar_ids=[4] + [0] * 8,
            hotbar_counts=[3] + [0] * 8,
            journal=[FakeEvent(
                "click_target", block=1, face=1, x=0, y=66, z=1)])

        actions = control.place(
            env, "cobblestone", (0, 67, 1), (0, 66, 1),
            stand=(0.5, 0.5), jump_from_y=64)

        self.assertGreaterEqual(len(actions), 1)
        self.assertEqual(actions[-1], {
            "action": "key", "k": "space", "ticks": 2})
        self.assertNotIn(
            "right_click", {action["action"] for action in actions})

    def test_interaction_recenter_preserves_camera_and_damps_trace_oscillation(self):
        control = controller(hazards=False)
        right = FakeEnv(x=8.587, y=83.0, z=3.5)
        right.obs["yaw"] = -810.119
        left = FakeEnv(x=8.419, y=83.0, z=3.5)
        left.obs["yaw"] = -990.011

        right_action = control.recenter_interaction_stance(
            right, 8.5, 3.5)
        left_action = control.recenter_interaction_stance(
            left, 8.5, 3.5)

        for action in (right_action, left_action):
            self.assertEqual(len(action), 1)
            self.assertEqual(action[0]["action"], "key")
            self.assertEqual(action[0]["ticks"], 1)
            self.assertNotIn("mouse_move", repr(action))
        self.assertEqual(right_action[0]["k"], ["s", "sneak"])
        self.assertEqual(left_action[0]["k"], ["s", "sneak"])

        # A live one-tick crouched step on the pinned engine settles after
        # 0.059 blocks.  The two exact poses from the rejection trace thus
        # converge inside the 0.03 interaction envelope in one correction;
        # preserving each pose's yaw is what makes `s` point toward center on
        # both sides despite their cameras being 180 degrees apart.
        right_settled = FakeEnv(x=8.587 - 0.059, y=83.0, z=3.5)
        right_settled.obs["yaw"] = -810.119
        left_settled = FakeEnv(x=8.419 + 0.059, y=83.0, z=3.5)
        left_settled.obs["yaw"] = -990.011
        self.assertIsNone(control.recenter_interaction_stance(
            right_settled, 8.5, 3.5, tolerance=0.03))
        self.assertIsNone(control.recenter_interaction_stance(
            left_settled, 8.5, 3.5, tolerance=0.03))

        # Had the camera stayed at the first trace yaw after crossing center,
        # the same world-space correction reverses from back to forward.
        crossed = FakeEnv(x=8.419, y=83.0, z=3.5)
        crossed.obs["yaw"] = -810.119
        self.assertEqual(control.recenter_interaction_stance(
            crossed, 8.5, 3.5)[0]["k"], ["w", "sneak"])

    def test_exact_click_proof_wins_over_residual_aim_correction(self):
        control = controller(hazards=False)
        env = FakeEnv(
            x=6.535, y=86.0, z=5.5,
            hotbar_ids=[257] + [0] * 8,
            hotbar_counts=[1] + [0] * 8,
            journal=[FakeEvent(
                "click_target", block=17, face=1, x=8, y=87, z=3)])
        env.angles[(8, 87, 3)] = (3.0, 2.0, 3.0)

        with patch.object(control, "aim") as aim, \
                patch.object(
                    control, "recenter_interaction_stance") as recenter:
            actions = control.mine_exact(
                env, (8, 87, 3), tool="iron_pickaxe", block="log",
                stand=(6.5, 5.5))

        self.assertEqual(actions, [{
            "action": "left_click_hold", "ticks": 80}])
        aim.assert_not_called()
        recenter.assert_not_called()

    def test_safe_pickup_approach_uses_configured_bounded_stride(self):
        control = controller(hazards=False)
        env = FakeEnv(
            x=9.51, y=83.0, z=3.475,
            hotbar_ids=[0] * 9, hotbar_counts=[0] * 9)
        # Reproduce the live failure: the immediately preceding normal route
        # move had the same target and made only tiny progress.  That must not
        # suppress a new fine pickup correction unless this is explicitly a
        # sneak-to-edge collection site.
        control._walk_state = (8.5, 3.5, 9.51, 3.475)

        with patch.object(
                control, "face_and_move_to",
                return_value=[{"action": "key",
                               "k": "w", "ticks": 1}]
                ) as move:
            actions = control.pickup(
                env, "log", minimum=1, attempt=0,
                target=(8.5, 3.5), sneak=False)

        self.assertEqual(actions[0]["k"], "w")
        move.assert_called_once_with(
            env, 8.5, 3.5, tolerance=0.1, sneak=False,
            allow_jump=False,
            movement_cap_blocks=control.behaviors.pickup_step_blocks)

    def test_pickup_waits_when_explicit_ledge_sneak_is_stuck(self):
        control = controller(hazards=False)
        env = FakeEnv(x=9.51, y=83.0, z=3.475)
        control._walk_state = (8.5, 3.5, 9.51, 3.475)

        actions = control.pickup(
            env, "log", minimum=1, attempt=0,
            target=(8.5, 3.5), sneak=True)

        self.assertEqual(actions, [{"action": "wait", "ticks": 12}])

    def test_unstuck_explicit_ledge_pickup_keeps_sneak(self):
        control = controller(hazards=False)
        env = FakeEnv(x=9.5, y=83.0, z=3.5)
        with patch.object(
                control, "face_and_move_to",
                return_value=[{"action": "key",
                               "k": ["w", "sneak"], "ticks": 1}]
                ) as move:
            actions = control.pickup(
                env, "log", minimum=1, attempt=0,
                target=(8.5, 3.5), sneak=True)

        self.assertEqual(actions[0]["k"], ["w", "sneak"])
        self.assertTrue(move.call_args.kwargs["sneak"])
        self.assertEqual(
            move.call_args.kwargs["movement_cap_blocks"],
            1.0 / control.movement.ticks_per_block)

    def test_explicit_ledge_pickup_emits_one_closed_loop_crouch_tick(self):
        control = controller(hazards=False)
        env = FakeEnv(x=8.5, y=92.0, z=9.5)

        actions = control.pickup(
            env, "log", minimum=1, attempt=0,
            target=(11.5, 7.5), sneak=True)

        move = next(action for action in actions
                    if action.get("action") == "key")
        self.assertEqual(move["k"], ["w", "sneak"])
        self.assertEqual(move["ticks"], 1)

    def test_proved_lower_pickup_can_use_stuck_recovery_jump(self):
        control = controller(hazards=False)
        env = FakeEnv(x=3.3, y=79.0, z=2.5)
        with patch.object(
                control, "face_and_move_to",
                return_value=[{"action": "key",
                               "k": ["w", "space"], "ticks": 2}]
                ) as move:
            actions = control.pickup(
                env, "cobblestone", minimum=1, attempt=1,
                target=(2.5, 2.5), sneak=False, allow_jump=True)

        self.assertEqual(actions[0]["k"], ["w", "space"])
        self.assertTrue(move.call_args.kwargs["allow_jump"])

    def test_mining_hold_uses_exact_observed_block_and_tool(self):
        control = controller(hazards=False)
        env = FakeEnv(blocks=[
            (3, 4, 81, 3),
            (1, 4, 82, 3),
            (49, 4, 10, 7),
        ])

        self.assertEqual(control.mining_hold_ticks(
            env, (4, 81, 3), tool="wooden_pickaxe", fallback=188), 26)
        self.assertEqual(control.mining_hold_ticks(
            env, (4, 82, 3), tool="stone_pickaxe", fallback=188), 28)
        env.obs["blocks"].append((16, 5, 82, 3))
        self.assertEqual(control.mining_hold_ticks(
            env, (5, 82, 3), tool="wooden_pickaxe", fallback=28), 50)
        self.assertEqual(control.mining_hold_ticks(
            env, (4, 10, 7), tool="diamond_pickaxe", fallback=28), 188)
        self.assertEqual(control.mining_hold_ticks(
            env, (7, 7, 7), tool="stone_pickaxe", fallback=41), 41)

        env.journal.append(FakeEvent(
            "player_fluid_state", block=9, flags=FLUID_VIEWPOINT))
        self.assertEqual(control.mining_hold_ticks(
            env, (4, 82, 3), tool="diamond_pickaxe", fallback=188), 44)

    def test_repeated_exact_mine_can_ignore_an_earlier_break(self):
        control = controller(hazards=False)
        env = FakeEnv(journal=[
            FakeEvent("block_broken", block=49, x=1, y=64, z=0),
            FakeEvent("click_target", block=49, face=1,
                      x=1, y=64, z=0),
        ])
        env.angles[(1, 64, 0)] = (0.0, 0.0, 2.0)

        self.assertIsNone(control.mine_exact(
            env, (1, 64, 0), block="obsidian", ticks=188))
        self.assertEqual(control.mine_exact(
            env, (1, 64, 0), block="obsidian", ticks=188, since=1), [{
                "action": "left_click_hold", "ticks": 188}])

    def test_use_block_clicks_only_a_proven_target(self):
        control = controller()
        env = FakeEnv(journal=[FakeEvent(
            "click_target", block=58, face=1, x=1, y=64, z=0)])
        env.angles[(1, 64, 0)] = (0.0, 0.0, 2.0)

        self.assertEqual(
            control.use_block(env, (1, 64, 0)),
            [{"action": "right_click"}])
        env.journal = [FakeEvent(
            "click_target", block=1, face=1, x=0, y=64, z=0)]
        veto = control.use_block(env, (1, 64, 0))
        self.assertIsInstance(veto, PolicyVeto)
        self.assertEqual(veto.code, "use_target_unproven")

    def test_place_clicks_only_a_proven_destination(self):
        control = controller()
        env = FakeEnv(
            hotbar_ids=[4] + [0] * 8,
            hotbar_counts=[3] + [0] * 8,
            journal=[FakeEvent(
                "click_target", block=1, face=4, x=0, y=63, z=1)])

        actions = control.place(
            env, "cobblestone", (0, 64, 1), (0, 63, 1))

        self.assertEqual(actions, [{"action": "right_click"}])

    def test_exact_place_recenters_to_avoid_player_collision(self):
        control = controller()
        env = FakeEnv(
            x=0.86, y=64.0, z=0.5,
            hotbar_ids=[4] + [0] * 8,
            hotbar_counts=[3] + [0] * 8,
            journal=[FakeEvent(
                "click_target", block=1, face=4, x=1, y=63, z=0)])
        correction = [{"action": "key", "k": ["a", "sneak"],
                       "ticks": 1}]

        with patch.object(
                control, "recenter_interaction_stance",
                return_value=correction) as recenter:
            actions = control.place(
                env, "cobblestone", (1, 64, 0), (1, 63, 0),
                stand=(0.5, 0.5))

        self.assertEqual(actions, correction)
        recenter.assert_called_once_with(
            env, 0.5, 0.5, tolerance=0.02)

    def test_place_observes_hotbar_selection_before_clicking(self):
        control = controller()
        env = FakeEnv(
            hotbar_ids=[0, 4] + [0] * 7,
            hotbar_counts=[0, 3] + [0] * 7,
            journal=[FakeEvent(
                "click_target", block=1, face=4, x=0, y=63, z=1)])

        actions = control.place(
            env, "cobblestone", (0, 64, 1), (0, 63, 1))

        self.assertTrue(actions)
        self.assertTrue(all(action["action"] == "key" for action in actions))
        self.assertNotIn("right_click", {action["action"] for action in actions})
        self.assertEqual(control._workstation_attempts, {})

    def test_crafting_wrappers_and_missing_table_are_explicit(self):
        control = controller()
        env = FakeEnv()
        with patch(
                "bedrock_rl.sdg.policy.oracle.gui_craft_2x2",
                return_value=[{"action": "key", "k": "e", "ticks": 1}]):
            self.assertEqual(control.craft_2x2(env, "planks")[0]["k"], "e")

        veto = control.craft_3x3(env, "wooden_pickaxe")
        self.assertIsInstance(veto, PolicyVeto)
        self.assertEqual(veto.code, "crafting_table_unavailable")

    def test_3x3_craft_requires_the_exact_planned_open_table(self):
        control = controller()
        expected = (2, 64, 3)
        env = FakeEnv(container=1, journal=[FakeEvent(
            "container_opened", kind=1, x=9, y=64, z=9)])

        veto = control.craft_3x3(
            env, "wooden_pickaxe", table=expected)

        self.assertIsInstance(veto, PolicyVeto)
        self.assertEqual(veto.code, "crafting_table_origin_unproven")
        env.journal.append(FakeEvent(
            "container_opened", kind=1, x=2, y=64, z=3))
        with patch(
                "bedrock_rl.sdg.policy.oracle.gui_craft_3x3",
                return_value=[{"action": "left_click"}]):
            self.assertEqual(control.craft_3x3(
                env, "wooden_pickaxe", table=expected),
                [{"action": "left_click"}])

    def test_nearby_cells_and_pickup_are_closed_loop(self):
        control = controller(hazards=False)
        env = FakeEnv(blocks=[(17, 2, 64, 0), (17, 1, 64, 0)])
        env.angles[(2, 64, 0)] = (0.0, 0.0, 2.0)
        env.angles[(1, 64, 0)] = (0.0, 0.0, 1.0)

        self.assertEqual(
            control.nearby_cells(env, "log", 3.0),
            [(1, 64, 0), (2, 64, 0)])
        actions = control.pickup(env, "log", attempt=0)
        self.assertEqual(actions, [{"action": "wait", "ticks": 12}])

    def test_workstation_place_and_open_require_exact_live_proof(self):
        control = controller(hazards=False)
        furnace = (1, 64, 0)
        support = (1, 63, 0)
        env = FakeEnv(
            hotbar_ids=[61] + [0] * 8,
            hotbar_counts=[1] + [0] * 8,
            journal=[FakeEvent(
                "click_target", block=1, face=4, x=1, y=63, z=0)])

        self.assertEqual(control.place_workstation(
            env, "furnace", furnace, support), [
                {"action": "right_click"}])

        env.journal.extend([
            FakeEvent("block_placed", block=61, x=1, y=64, z=0),
            FakeEvent("click_target", block=61, face=1, x=1, y=64, z=0),
        ])
        self.assertEqual(control.open_workstation(
            env, furnace, block="furnace", container="furnace"), [
                {"action": "right_click"}])

        env.obs["container"] = 2
        env.journal.append(FakeEvent(
            "container_opened", kind=2, x=1, y=64, z=0))
        self.assertIsNone(control.open_workstation(
            env, furnace, block="furnace", container="furnace"))

        env.journal[-1] = FakeEvent(
            "container_opened", kind=2, x=9, y=64, z=9)
        veto = control.open_workstation(
            env, furnace, block="furnace", container="furnace")
        self.assertIsInstance(veto, PolicyVeto)
        self.assertEqual(veto.code, "workstation_origin_unproven")

    def test_workstation_retries_are_bounded_and_structured(self):
        control = controller(hazards=False)
        env = FakeEnv(
            hotbar_ids=[61] + [0] * 8,
            hotbar_counts=[1] + [0] * 8,
            journal=[FakeEvent(
                "click_target", block=1, face=4, x=1, y=63, z=0)])

        for _ in range(2):
            self.assertEqual(control.place_workstation(
                env, "furnace", (1, 64, 0), (1, 63, 0),
                max_attempts=2), [{"action": "right_click"}])
        veto = control.place_workstation(
            env, "furnace", (1, 64, 0), (1, 63, 0), max_attempts=2)

        self.assertIsInstance(veto, PolicyVeto)
        self.assertEqual(veto.code, "workstation_place_exhausted")
        self.assertEqual(veto.to_dict()["observed"], [[1, 64, 0]])

    def test_recover_cursor_uses_public_state_and_closes_cleanly(self):
        from bedrock_rl.env import gui_layout

        control = controller()
        env = FakeEnv(
            container=2,
            hotbar_ids=[1] * 8 + [0],
            hotbar_counts=[1] * 8 + [0],
            journal=[
                FakeEvent("container_opened", kind=2, x=1, y=64, z=0),
                FakeEvent("cursor_changed", item=265, count=4, meta=0),
            ])

        actions = control.recover_cursor(env)

        self.assertEqual(actions[0], gui_layout.click_slot(
            8, "left", "furnace")[0])
        self.assertEqual(actions[-1], gui_layout.close_screen()[0])

    def test_smelt_is_an_exact_verified_closed_loop_transaction(self):
        from bedrock_rl.env import gui_layout

        control = controller()
        furnace = (1, 64, 0)
        journal = [
            FakeEvent("block_placed", block=61, x=1, y=64, z=0),
            FakeEvent("container_opened", kind=2, x=1, y=64, z=0),
        ]
        env = FakeEnv(
            container=2,
            hotbar_ids=[15, 263] + [0] * 7,
            hotbar_counts=[4, 1] + [0] * 7,
            journal=journal)

        load = control.smelt(
            env, furnace, "iron_ore", "iron_ingot", count=4,
            output_replace=("stone_pickaxe",))
        self.assertEqual(load, (
            gui_layout.click_slot(0, "left", "furnace")
            + gui_layout.click_slot(46, "left", "furnace")
            + gui_layout.click_slot(1, "left", "furnace")
            + gui_layout.click_slot(47, "left", "furnace")))

        env.journal.extend([
            FakeEvent("slot_changed", slot=0, item=0, count=0, meta=0),
            FakeEvent("slot_changed", slot=1, item=0, count=0, meta=0),
            FakeEvent("cursor_changed", item=0, count=0, meta=0),
        ])
        self.assertEqual(control.smelt(
            env, furnace, "iron_ore", "iron_ingot", count=4,
            output_replace=("wooden_pickaxe",)), [
                {"action": "wait", "ticks": 200}])

        env.journal.extend([
            FakeEvent(
                "item_smelted", item=265, count=1, input=15, meta=0)
            for _ in range(4)
        ])
        take = control.smelt(
            env, furnace, "iron_ore", "iron_ingot", count=4)
        self.assertEqual(take, (
            gui_layout.click_slot(48, "left", "furnace")
            + gui_layout.click_slot(2, "left", "furnace")
            + gui_layout.close_screen()))

        env.obs["container"] = 0
        env.obs["hotbar_ids"][2] = 265
        env.obs["hotbar_counts"][2] = 4
        env.journal.extend([
            FakeEvent("slot_changed", slot=2, item=265, count=4, meta=0),
            FakeEvent("cursor_changed", item=0, count=0, meta=0),
            FakeEvent("container_closed", kind=2, x=1, y=64, z=0),
        ])
        self.assertIsNone(control.smelt(
            env, furnace, "iron_ore", "iron_ingot", count=4))
        self.assertTrue(control.furnace_session_complete(furnace))

    def test_smelt_rejects_unproven_load_without_replaying_clicks(self):
        control = controller()
        furnace = (1, 64, 0)
        env = FakeEnv(
            container=2,
            hotbar_ids=[15, 263] + [0] * 7,
            hotbar_counts=[1, 1] + [0] * 7,
            journal=[
                FakeEvent("block_placed", block=61, x=1, y=64, z=0),
                FakeEvent("container_opened", kind=2, x=1, y=64, z=0),
            ])
        control.smelt(env, furnace, "iron_ore", "iron_ingot")

        self.assertEqual(control.smelt(
            env, furnace, "iron_ore", "iron_ingot",
            max_attempts=2), [{"action": "wait", "ticks": 1}])
        veto = control.smelt(
            env, furnace, "iron_ore", "iron_ingot", max_attempts=2)

        self.assertIsInstance(veto, PolicyVeto)
        self.assertEqual(veto.code, "furnace_load_unverified")

    def test_smelt_requires_fresh_furnace_for_exact_output(self):
        control = controller()
        furnace = (1, 64, 0)
        env = FakeEnv(
            blocks=[(61, *furnace)],
            hotbar_ids=[15, 263] + [0] * 7,
            hotbar_counts=[1, 1] + [0] * 7)

        veto = control.smelt(
            env, furnace, "iron_ore", "iron_ingot")

        self.assertIsInstance(veto, PolicyVeto)
        self.assertEqual(veto.code, "furnace_freshness_unproven")

    def test_smelt_closes_workstation_before_backpack_recovery(self):
        from bedrock_rl.env import gui_layout

        control = controller()
        furnace = (1, 64, 0)
        env = FakeEnv(
            container=2,
            hotbar_ids=[263] + [0] * 8,
            hotbar_counts=[1] + [0] * 8,
            journal=[
                FakeEvent("block_placed", block=61, x=1, y=64, z=0),
                FakeEvent("slot_changed", slot=12, item=15, count=1),
                FakeEvent("container_opened", kind=2, x=1, y=64, z=0),
            ])

        self.assertEqual(control.smelt(
            env, furnace, "iron_ore", "iron_ingot"),
            gui_layout.close_screen())

    def test_smelt_recovers_split_backpack_input_until_exact_count_visible(self):
        from bedrock_rl.env import gui_layout

        control = controller()
        furnace = (1, 64, 0)
        env = FakeEnv(
            hotbar_ids=[15, 263] + [0] * 7,
            hotbar_counts=[5, 2] + [0] * 7,
            journal=[
                FakeEvent("block_placed", block=61, x=1, y=64, z=0),
                FakeEvent("slot_changed", slot=12, item=15, count=5),
            ])

        actions = control.smelt(
            env, furnace, "iron_ore", "iron_ingot", count=10,
            fuel_count=2)

        self.assertEqual(actions, (
            gui_layout.open_screen()
            + gui_layout.click_slot(12, "left", "player")
            + gui_layout.click_slot(0, "left", "player")
            + gui_layout.close_screen()))

    def test_smelt_wait_budget_rejects_a_stalled_furnace(self):
        control = controller()
        furnace = (1, 64, 0)
        env = FakeEnv(
            container=2,
            hotbar_ids=[15, 263] + [0] * 7,
            hotbar_counts=[1, 1] + [0] * 7,
            journal=[
                FakeEvent("block_placed", block=61, x=1, y=64, z=0),
                FakeEvent("container_opened", kind=2, x=1, y=64, z=0),
            ])
        control.smelt(env, furnace, "iron_ore", "iron_ingot")
        env.journal.extend([
            FakeEvent("slot_changed", slot=0, item=0, count=0),
            FakeEvent("slot_changed", slot=1, item=0, count=0),
        ])

        self.assertEqual(control.smelt(
            env, furnace, "iron_ore", "iron_ingot", max_waits=1), [
                {"action": "wait", "ticks": 200}])
        veto = control.smelt(
            env, furnace, "iron_ore", "iron_ingot", max_waits=1)

        self.assertIsInstance(veto, PolicyVeto)
        self.assertEqual(veto.code, "furnace_smelt_stalled")

    def test_large_exact_load_is_chunked_to_computer_action_budget(self):
        control = controller()
        furnace = (1, 64, 0)
        env = FakeEnv(
            container=2,
            hotbar_ids=[15, 263] + [0] * 7,
            hotbar_counts=[64, 1] + [0] * 7,
            journal=[
                FakeEvent("block_placed", block=61, x=1, y=64, z=0),
                FakeEvent("container_opened", kind=2, x=1, y=64, z=0),
            ])

        first = control.smelt(
            env, furnace, "iron_ore", "iron_ingot", count=25,
            max_actions=12)

        self.assertEqual(len(first), 12)
        self.assertEqual(sum(
            action["action"] == "right_click" for action in first), 10)
        env.obs["hotbar_counts"][0] = 54
        env.journal.append(FakeEvent(
            "slot_changed", slot=0, item=15, count=54))
        self.assertEqual(control.smelt(
            env, furnace, "iron_ore", "iron_ingot", count=25,
            max_actions=12), [{"action": "wait", "ticks": 1}])


if __name__ == "__main__":
    unittest.main()
