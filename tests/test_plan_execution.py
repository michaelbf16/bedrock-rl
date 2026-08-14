from dataclasses import replace
from types import SimpleNamespace
import json
import unittest

from bedrock_rl.sdg.policy.execution import (
    FluidCastExecution,
    FluidCastExecutor,
    FluidSourceExecution,
    MiningExecution,
    PlanExecutor,
)
from bedrock_rl.sdg.policy.planning import (
    MiningTargetPlan,
    RoutePlan,
)
from bedrock_rl.sdg.policy import PolicyVeto
from bedrock_rl.sdg.policy.spatial import RemovalProof
from bedrock_rl.sdg.policy.structures import (
    FluidCastPlan,
    InteractionRay,
    PlaceStep,
)
from bedrock_rl.env.journal import FLUID_BODY


class Event(dict):
    def __init__(self, name, **values):
        super().__init__(values)
        self.name = name


class FakeEnv:
    def __init__(self, position=(0.5, 1.0, 0.5)):
        self.obs = {
            "tick": 0,
            "x": position[0],
            "y": position[1],
            "z": position[2],
            "health": 20.0,
            "dead": False,
            "blocks": [],
            "hotbar_ids": [0] * 9,
            "hotbar_counts": [0] * 9,
        }
        self.journal = []


class FakeController:
    def __init__(self):
        self.mines = []
        self.moves = []
        self.pickups = []
        self.placements = []
        self.hazard_ids = frozenset({10, 11, 51})
        self.behaviors = SimpleNamespace(hazard_clearance_blocks=0.0)
        self.movement = SimpleNamespace(max_blocks=1.0)

    @staticmethod
    def _block_ids(value):
        names = {"log": 17, "stone": 1}
        values = value if isinstance(value, (list, tuple)) else (value,)
        return tuple(names.get(item, int(item) if isinstance(item, int)
                               else -1) for item in values)

    @staticmethod
    def _item_ids(value):
        names = {"log": 5, "diamond": 264}
        values = value if isinstance(value, (list, tuple)) else (value,)
        return tuple(names.get(item, int(item) if isinstance(item, int)
                               else -1) for item in values)

    @staticmethod
    def observed_block(env, cell):
        for block, x, y, z in env.obs["blocks"]:
            if (x, y, z) == cell:
                return block
        return None

    @classmethod
    def item_count(cls, env, item):
        wanted = frozenset(cls._item_ids(item))
        return sum(count for found, count in zip(
            env.obs["hotbar_ids"], env.obs["hotbar_counts"])
                   if found in wanted)

    @classmethod
    def inventory_item_count(cls, env, item):
        if "inventory_ids" not in env.obs:
            return cls.item_count(env, item)
        wanted = frozenset(cls._item_ids(item))
        return sum(count for found, count in zip(
            env.obs["inventory_ids"], env.obs["inventory_counts"])
                   if found in wanted)

    def mine_exact(self, env, cell, **kwargs):
        del env
        self.mines.append((cell, kwargs))
        return [{"action": "left_click_hold", "ticks": kwargs["ticks"]}]

    def face_and_move_to(self, env, x, z, **kwargs):
        del env
        self.moves.append((x, z, kwargs))
        return [{"action": "key", "k": "w", "ticks": 3}]

    def pickup(self, env, item, **kwargs):
        del env
        self.pickups.append((item, kwargs))
        return [{"action": "wait", "ticks": 2}]

    @staticmethod
    def drop_is_safe(current_y, landing_y):
        return current_y - landing_y <= 1

    @classmethod
    def placed(cls, env, cell, block):
        expected = frozenset(cls._block_ids(block))
        return any(event.name == "block_placed"
                   and event_cell(event) == cell
                   and int(event.get("block", 0)) in expected
                   for event in env.journal)

    def place(self, env, item, cell, support, **kwargs):
        del env
        self.placements.append((item, cell, support, kwargs))
        return [{"action": "right_click"}]


def event_cell(event):
    return tuple(event[key] for key in ("x", "y", "z"))


def tunnel_target():
    stands = ((0, 1, 0), (1, 1, 0), (2, 1, 0))
    mine_cells = ((1, 1, 0), (1, 2, 0))
    reserved = tuple(dict.fromkeys(
        cell for stand in stands
        for cell in (stand, (stand[0], stand[1] + 1, stand[2]))))
    route = RoutePlan(
        "tunnel", stands, reserved,
        tuple((stand[0], stand[1] - 1, stand[2]) for stand in stands),
        mine_cells,
    )
    target = (3, 1, 0)
    proof = RemovalProof(
        target,
        (target, (4, 1, 0), (2, 1, 0), (3, 2, 0),
         (3, 0, 0), (3, 1, 1), (3, 1, -1)),
        ((0, 0, 0), (1, 0, 0), (2, 0, 0)),
    )
    return MiningTargetPlan(target, stands[-1], route, proof)


def surface_target():
    stance = (0, 1, 0)
    route = RoutePlan.surface((stance,))
    target = (1, 1, 0)
    proof = RemovalProof(
        target,
        (target, (2, 1, 0), (0, 1, 0), (1, 2, 0),
         (1, 0, 0), (1, 1, 1), (1, 1, -1)),
        ((0, 0, 0),),
    )
    return MiningTargetPlan(target, stance, route, proof)


class PlanExecutorTests(unittest.TestCase):
    def test_exact_soft_ray_occluder_is_declared_and_cleared_first(self):
        class PlantOccludedController(FakeController):
            @staticmethod
            def click_target(env):
                del env
                return 31, (1, 2, 0)

            def mine_exact(self, env, cell, **kwargs):
                del env, kwargs
                return PolicyVeto(
                    "mine_target_unproven",
                    f"plant blocks {cell}", (1.5, 0.5), ((1, 2, 0),))

        declared = []
        executor = PlanExecutor(
            PlantOccludedController(), [surface_target()],
            break_declarer=lambda cells: declared.extend(cells)).reset(
                FakeEnv())

        action = executor.actions(FakeEnv())

        self.assertEqual(action, [{
            "action": "left_click_hold", "ticks": 2}])
        self.assertEqual(declared, [(1, 2, 0)])

    def test_soft_route_body_occluder_is_declared_before_solid_clear(self):
        class PlantOccludedController(FakeController):
            @staticmethod
            def click_target(env):
                del env
                return 31, (1, 2, 0)

            def mine_exact(self, env, cell, **kwargs):
                del env, kwargs
                return PolicyVeto(
                    "mine_target_unproven",
                    f"plant blocks {cell}", (1.5, 0.5), ((1, 2, 0),))

        route = RoutePlan(
            "tunnel", ((0, 1, 0), (1, 1, 0)),
            ((0, 1, 0), (0, 2, 0), (1, 1, 0), (1, 2, 0)),
            ((0, 0, 0), (1, 0, 0)), ((1, 1, 0),))
        target = MiningTargetPlan(
            (2, 1, 0), (1, 1, 0), route,
            RemovalProof((2, 1, 0), ((2, 1, 0),), ()))
        declared = []
        executor = PlanExecutor(
            PlantOccludedController(), [target],
            break_declarer=lambda cells: declared.extend(cells)).reset(
                FakeEnv())

        action = executor.actions(FakeEnv())

        self.assertEqual(action, [{
            "action": "left_click_hold", "ticks": 2}])
        self.assertEqual(declared, [(1, 2, 0)])

    def test_diagnostics_identify_active_plan_leg_and_bound_traces(self):
        env = FakeEnv()
        executor = PlanExecutor(FakeController(), [tunnel_target()]).reset(env)
        executor._route_index = 1
        executor._clear_control_trace[(1, 1, 0)] = [
            {"attempt": index} for index in range(6)]
        executor._mine_control_trace = [
            {"attempt": index} for index in range(6)]
        executor._soft_clear_attempts[(1, 2, 0)] = 1

        diagnostics = executor.diagnostics()
        json.dumps(diagnostics)

        self.assertEqual(diagnostics["target_index"], 0)
        self.assertEqual(diagnostics["active"]["target"], (3, 1, 0))
        self.assertEqual(
            diagnostics["active"]["mining_stance"], (2, 1, 0))
        self.assertEqual(
            diagnostics["active"]["route_previous"], (0, 1, 0))
        self.assertEqual(
            diagnostics["active"]["route_destination"], (1, 1, 0))
        self.assertEqual(len(diagnostics["recent_mine_control"]), 4)
        self.assertEqual(diagnostics["soft_clear_attempts"], {
            "(1, 2, 0)": 1})
        self.assertEqual(len(diagnostics["active"][
            "recent_clear_control"]["(1, 1, 0)"]), 4)

    def test_seals_planner_approved_water_before_mining(self):
        stance = (0, 1, 0)
        target_cell = (1, 1, 0)
        seal_cell = (1, 2, 0)
        target = MiningTargetPlan(
            target_cell, stance, RoutePlan.surface((stance,)),
            RemovalProof(
                target_cell, (target_cell, seal_cell), (),
                "known_faces_sealable_fluid_v1", (seal_cell,)))
        controller = FakeController()
        env = FakeEnv()
        execution = MiningExecution(
            target, block="stone", fluid_seal_item="stone")
        executor = PlanExecutor(controller, [execution]).reset(env)

        seal = executor.actions(env)

        self.assertEqual(seal, [{"action": "right_click"}])
        self.assertEqual(
            controller.placements[-1][1:3], (seal_cell, target_cell))
        self.assertFalse(controller.mines)
        env.journal.append(Event(
            "block_placed", block=1, x=1, y=2, z=0))
        mine = executor.actions(env)
        self.assertEqual(mine[0]["action"], "left_click_hold")

    def test_long_movement_profile_stays_on_one_certified_route_cell(self):
        stands = ((0, 1, 0), (1, 1, 0), (2, 1, 0), (3, 1, 0))
        route = RoutePlan.surface(stands)
        target_cell = (4, 1, 0)
        target = MiningTargetPlan(
            target_cell, stands[-1], route,
            RemovalProof(target_cell, (target_cell,), ()))
        controller = FakeController()
        controller.movement.max_blocks = 2.5
        executor = PlanExecutor(controller, [target]).reset(FakeEnv())

        result = executor.actions(FakeEnv())

        self.assertEqual(result[0]["action"], "key")
        self.assertEqual(controller.moves[-1][:2], (1.5, 0.5))

    def test_recovers_hidden_tool_before_irreversible_mining(self):
        env = FakeEnv()
        controller = FakeController()
        recovery = []

        def recover(_env, item):
            recovery.append(item)
            return [{"action": "key", "k": "e"}]

        execution = MiningExecution(
            surface_target(), block="stone", tool="stone_pickaxe")
        executor = PlanExecutor(
            controller, [execution], item_recovery=recover).reset(env)

        result = executor.actions(env)

        self.assertEqual(result, [{"action": "key", "k": "e"}])
        self.assertEqual(recovery, ["stone_pickaxe"])
        self.assertFalse(controller.mines)

    def test_surface_momentum_resynchronizes_only_to_later_proved_stand(self):
        stands = ((0, 2, 0), (1, 1, 0), (2, 1, 0), (3, 1, 0))
        target_cell = (4, 1, 0)
        target = MiningTargetPlan(
            target_cell, stands[-1], RoutePlan.surface(stands),
            RemovalProof(target_cell, (target_cell,), ()))
        env = FakeEnv((0.5, 2.0, 0.5))
        controller = FakeController()
        executor = PlanExecutor(controller, [target]).reset(env)
        executor.actions(env)
        # Momentum passed the descending destination but landed centered on
        # the immediately following snapshot-proved surface stand.
        env.obs.update(x=2.5, y=1.0, z=0.5)

        action = executor.actions(env)

        self.assertEqual(executor.route_index, 3)
        self.assertEqual(action[0]["action"], "key")
        self.assertEqual(controller.moves[-1][:2], (3.5, 0.5))

    def test_descending_leg_waits_for_exact_level_before_advancing(self):
        stands = ((0, 2, 0), (1, 1, 0))
        target_cell = (2, 1, 0)
        target = MiningTargetPlan(
            target_cell, stands[-1], RoutePlan.surface(stands),
            RemovalProof(target_cell, (target_cell,), ()))
        env = FakeEnv((0.5, 2.0, 0.5))
        controller = FakeController()
        executor = PlanExecutor(controller, [target]).reset(env)

        move = executor.actions(env)
        self.assertEqual(move[0]["action"], "key")
        self.assertNotIn("sneak", controller.moves[-1][2])
        self.assertEqual(controller.moves[-1][2]["tolerance"], 0.05)
        self.assertFalse(controller.moves[-1][2]["allow_jump"])
        self.assertEqual(
            controller.moves[-1][2]["movement_cap_blocks"], 0.3)
        env.obs.update(x=1.5, y=2.0, z=0.5)
        settle = executor.actions(env)
        self.assertEqual(settle, [{"action": "wait", "ticks": 2}])
        env.obs.update(x=1.09, y=1.0)
        correction = executor.actions(env)
        self.assertEqual(correction[0]["action"], "key")
        env.obs["y"] = 1.0
        env.obs["x"] = 1.5
        mine = executor.actions(env)
        self.assertEqual(mine[0]["action"], "left_click_hold")

    def test_ascending_leg_uses_a_bounded_closed_loop_jump(self):
        stands = ((8, 92, 9), (9, 93, 9))
        target_cell = (11, 93, 7)
        target = MiningTargetPlan(
            target_cell, stands[-1], RoutePlan.surface(stands),
            RemovalProof(target_cell, (target_cell,), ()))
        env = FakeEnv((8.5, 92.0, 9.5))
        controller = FakeController()
        controller.movement.max_blocks = 2.5
        executor = PlanExecutor(controller, [target]).reset(env)

        move = executor.actions(env)

        self.assertEqual(move[0]["action"], "key")
        self.assertTrue(controller.moves[-1][2]["step_up"])
        self.assertEqual(
            controller.moves[-1][2]["movement_cap_blocks"], 0.3)

    def test_ascending_jump_arc_waits_then_accepts_proved_landing(self):
        stands = ((8, 92, 9), (9, 93, 9))
        target_cell = (11, 93, 7)
        target = MiningTargetPlan(
            target_cell, stands[-1], RoutePlan.surface(stands),
            RemovalProof(target_cell, (target_cell,), ()))
        env = FakeEnv((8.5, 92.0, 9.5))
        executor = PlanExecutor(FakeController(), [target]).reset(env)
        executor.actions(env)
        # Exact live regression: the old full-stride jump reached destination
        # x/z while still 1.249 blocks above its certified landing.
        env.obs.update(x=9.378, y=94.249, z=9.541)

        wait = executor.actions(env)

        self.assertEqual(wait, [{"action": "wait", "ticks": 2}])
        env.obs.update(x=9.5, y=93.0, z=9.5)
        mine = executor.actions(env)
        self.assertEqual(mine[0]["action"], "left_click_hold")

    def test_ascending_lip_perch_recenters_without_another_jump(self):
        stands = ((2, 72, 13), (3, 73, 13))
        target_cell = (-13, 65, 0)
        target = MiningTargetPlan(
            target_cell, stands[-1], RoutePlan.surface(stands),
            RemovalProof(target_cell, (target_cell,), ()))
        env = FakeEnv((2.5, 72.0, 13.5))
        controller = FakeController()
        executor = PlanExecutor(controller, [target]).reset(env)
        executor.actions(env)
        # Exact live canary regression: the player body caught the raised
        # neighboring block while its feet remained inside the destination
        # cell.  Waiting at this stable y can never settle the jump.
        env.obs.update(x=3.513, y=74.0, z=13.883)

        correction = executor.actions(env)

        self.assertEqual(correction[0]["action"], "key")
        self.assertEqual(controller.moves[-1][:2], (3.5, 13.5))
        self.assertEqual(controller.moves[-1][2]["tolerance"], 0.05)
        self.assertFalse(controller.moves[-1][2]["step_up"])
        self.assertFalse(controller.moves[-1][2]["allow_jump"])
        self.assertEqual(
            controller.moves[-1][2]["movement_cap_blocks"], 0.3)

        env.obs.update(x=3.5, y=73.0, z=13.5)
        mine = executor.actions(env)
        self.assertEqual(mine[0]["action"], "left_click_hold")

    def test_centered_ascending_height_mismatch_waits_instead_of_moving(self):
        stands = ((2, 72, 13), (3, 73, 13))
        target_cell = (-13, 65, 0)
        target = MiningTargetPlan(
            target_cell, stands[-1], RoutePlan.surface(stands),
            RemovalProof(target_cell, (target_cell,), ()))
        env = FakeEnv((2.5, 72.0, 13.5))
        controller = FakeController()
        executor = PlanExecutor(controller, [target]).reset(env)
        executor.actions(env)
        env.obs.update(x=3.5, y=74.0, z=13.5)

        wait = executor.actions(env)

        self.assertEqual(wait, [{"action": "wait", "ticks": 2}])
        self.assertEqual(len(controller.moves), 1)

    def test_unproved_upward_overshoot_still_vetoes(self):
        stands = ((8, 92, 9), (9, 93, 9))
        target_cell = (11, 93, 7)
        target = MiningTargetPlan(
            target_cell, stands[-1], RoutePlan.surface(stands),
            RemovalProof(target_cell, (target_cell,), ()))
        env = FakeEnv((8.5, 92.0, 9.5))
        executor = PlanExecutor(FakeController(), [target]).reset(env)
        executor.actions(env)
        env.obs.update(x=9.5, y=94.5, z=9.5)

        veto = executor.actions(env)

        self.assertEqual(veto.code, "route_position_diverged")

    def test_ascending_settle_wait_is_bounded(self):
        stands = ((8, 92, 9), (9, 93, 9))
        target_cell = (11, 93, 7)
        target = MiningTargetPlan(
            target_cell, stands[-1], RoutePlan.surface(stands),
            RemovalProof(target_cell, (target_cell,), ()))
        env = FakeEnv((8.5, 92.0, 9.5))
        executor = PlanExecutor(
            FakeController(), [target], max_move_attempts=2).reset(env)
        executor.actions(env)
        env.obs.update(x=9.211, y=94.249, z=9.480)

        self.assertEqual(
            executor.actions(env), [{"action": "wait", "ticks": 2}])
        veto = executor.actions(env)
        self.assertEqual(veto.code, "route_vertical_settle_exhausted")

    def test_executor_enforces_controller_descent_policy(self):
        stands = ((0, 2, 0), (1, 1, 0))
        target_cell = (2, 1, 0)
        target = MiningTargetPlan(
            target_cell, stands[-1], RoutePlan.surface(stands),
            RemovalProof(target_cell, (target_cell,), ()))
        controller = FakeController()
        controller.drop_is_safe = lambda current, landing: False
        env = FakeEnv((0.5, 2.0, 0.5))

        veto = PlanExecutor(controller, [target]).reset(env).actions(env)

        self.assertEqual(veto.code, "route_drop_rejected")
        self.assertFalse(controller.moves)

    def test_descending_tunnel_clears_body_before_gravity_wait(self):
        stands = ((0, 2, 0), (1, 1, 0))
        reserved = list(
            cell for stand in stands
            for cell in (stand, (stand[0], stand[1] + 1, stand[2])))
        reserved.append((1, 3, 0))
        mine_cells = ((1, 1, 0), (1, 2, 0), (1, 3, 0))
        route = RoutePlan(
            "tunnel", stands, tuple(reserved),
            ((0, 1, 0), (1, 0, 0)), mine_cells)
        target_cell = (2, 1, 0)
        target = MiningTargetPlan(
            target_cell, stands[-1], route,
            RemovalProof(target_cell, (target_cell,), ()))
        env = FakeEnv((0.5, 2.0, 0.5))
        controller = FakeController()
        executor = PlanExecutor(controller, [target]).reset(env)

        action = executor.actions(env)

        self.assertEqual(action[0]["action"], "left_click_hold")
        self.assertEqual(controller.mines[-1][0], (1, 3, 0))
        self.assertFalse(controller.moves)

    def test_ascending_tunnel_clears_origin_jump_headroom_before_move(self):
        stands = ((0, 1, 0), (1, 2, 0))
        headroom = (0, 3, 0)
        route = RoutePlan(
            "tunnel", stands,
            ((0, 1, 0), (0, 2, 0), headroom,
             (1, 2, 0), (1, 3, 0)),
            ((0, 0, 0), (1, 1, 0)), (headroom,))
        target_cell = (2, 2, 0)
        target = MiningTargetPlan(
            target_cell, stands[-1], route,
            RemovalProof(target_cell, (target_cell,), ()))
        env = FakeEnv((0.5, 1.0, 0.5))
        controller = FakeController()
        executor = PlanExecutor(controller, [target]).reset(env)

        action = executor.actions(env)

        self.assertEqual(action[0]["action"], "left_click_hold")
        self.assertEqual(controller.mines[-1][0], headroom)
        self.assertFalse(controller.moves)

        incomplete = RoutePlan(
            "tunnel", stands,
            ((0, 1, 0), (0, 2, 0), (1, 2, 0), (1, 3, 0)),
            ((0, 0, 0), (1, 1, 0)), ())
        with self.assertRaisesRegex(ValueError, "origin jump headroom"):
            PlanExecutor(controller, [MiningTargetPlan(
                target_cell, stands[-1], incomplete,
                RemovalProof(target_cell, (target_cell,), ()))])

    def test_ascending_tunnel_clears_nearest_equal_height_cell_first(self):
        stands = ((0, 1, 0), (1, 2, 0))
        near_headroom = (0, 3, 0)
        far_headroom = (1, 3, 0)
        route = RoutePlan(
            "tunnel", stands,
            ((0, 1, 0), (0, 2, 0), near_headroom,
             (1, 2, 0), far_headroom),
            ((0, 0, 0), (1, 1, 0)),
            # Reverse physical order to prove execution, not planner
            # insertion order, chooses the unobstructed nearest cell.
            (far_headroom, near_headroom))
        target_cell = (2, 2, 0)
        target = MiningTargetPlan(
            target_cell, stands[-1], route,
            RemovalProof(target_cell, (target_cell,), ()))
        env = FakeEnv((0.5, 1.0, 0.5))
        controller = FakeController()

        action = PlanExecutor(controller, [target]).reset(env).actions(env)

        self.assertEqual(action[0]["action"], "left_click_hold")
        self.assertEqual(controller.mines[-1][0], near_headroom)

    def test_overdue_lighting_vetoes_instead_of_silently_walking(self):
        class Lighting:
            def observe_route_travel(self, env):
                del env

            def remember_route_position(self, env):
                del env

            def light_route(self, env):
                del env
                return None

            def route_light_overdue(self):
                return True

        env = FakeEnv()
        executor = PlanExecutor(
            FakeController(), [tunnel_target()],
            lighting=Lighting()).reset(env)

        veto = executor.actions(env)

        self.assertIsInstance(veto, PolicyVeto)
        self.assertEqual(veto.code, "route_light_overdue")

    def test_clears_tunnel_before_walking_and_mines_at_proved_stance(self):
        env = FakeEnv()
        controller = FakeController()
        executor = PlanExecutor(
            controller, [tunnel_target()], target_block="log",
            tunnel_tool="stone_pickaxe").reset(env)

        first = executor.actions(env)
        self.assertEqual(first[0]["action"], "left_click_hold")
        self.assertEqual(controller.mines[-1][0], (1, 2, 0))
        self.assertFalse(controller.moves)
        env.journal.append(Event(
            "block_broken", block=1, drop=4, x=1, y=2, z=0))

        second = executor.actions(env)
        self.assertEqual(second[0]["action"], "left_click_hold")
        self.assertEqual(controller.mines[-1][0], (1, 1, 0))
        self.assertFalse(controller.moves)
        env.journal.append(Event(
            "block_broken", block=1, drop=4, x=1, y=1, z=0))

        move_one = executor.actions(env)
        self.assertEqual(move_one[0]["action"], "key")
        env.obs.update(x=1.5, y=1.0, z=0.5, tick=1)
        move_two = executor.actions(env)
        self.assertEqual(move_two[0]["action"], "key")
        env.obs.update(x=2.5, y=1.0, z=0.5, tick=2)

        target_action = executor.actions(env)
        self.assertEqual(target_action[0]["action"], "left_click_hold")
        self.assertEqual(controller.mines[-1][0], (3, 1, 0))
        self.assertEqual(controller.mines[-1][1]["block"], "log")
        env.journal.append(Event(
            "block_broken", block=17, drop=5, x=3, y=1, z=0))

        self.assertIsNone(executor.actions(env))
        self.assertTrue(executor.done)
        self.assertEqual(
            [event_cell(event) for event in env.journal],
            [(1, 2, 0), (1, 1, 0), (3, 1, 0)])

    def test_final_mining_stance_recenters_before_irreversible_click(self):
        class RecenteringController(FakeController):
            def __init__(self):
                super().__init__()
                self.recenters = []

            def recenter_interaction_stance(
                    self, env, x, z, *, tolerance):
                del env
                self.recenters.append((x, z, tolerance))
                return [{"action": "key", "k": ["a", "sneak"],
                         "ticks": 1}]

        env = FakeEnv((0.556, 1.0, 0.797))
        controller = RecenteringController()
        executor = PlanExecutor(
            controller, [surface_target()]).reset(env)
        executor._route_index = 1

        correction = executor.actions(env)

        self.assertEqual(correction[0]["k"], ["a", "sneak"])
        self.assertEqual(controller.recenters, [(0.5, 0.5, 0.05)])
        self.assertFalse(controller.mines)
        env.obs.update(x=0.52, z=0.51)
        attack = executor.actions(env)
        self.assertEqual(attack[0]["action"], "left_click_hold")

    def test_final_mining_stance_recovers_adjacent_boundary_rounding(self):
        env = FakeEnv((0.5, 1.0, -0.037))
        controller = FakeController()
        executor = PlanExecutor(
            controller, [surface_target()]).reset(env)
        executor._route_index = 1

        correction = executor.actions(env)

        self.assertEqual(correction[0]["action"], "key")
        self.assertEqual(controller.moves[-1][0:2], (0.5, 0.5))
        self.assertFalse(controller.mines)

    def test_uncleared_descent_recenters_residual_momentum_before_aiming(self):
        class RecenteringController(FakeController):
            def __init__(self):
                super().__init__()
                self.recenters = []

            def recenter_interaction_stance(
                    self, env, x, z, *, tolerance):
                del env
                self.recenters.append((x, z, tolerance))
                return [{"action": "key", "k": ["s", "sneak"],
                         "ticks": 1}]

        stands = ((0, 2, 0), (1, 1, 0))
        route = RoutePlan(
            "tunnel", stands,
            ((0, 2, 0), (0, 3, 0),
             (1, 1, 0), (1, 2, 0), (1, 3, 0)),
            ((0, 1, 0), (1, 0, 0)),
            ((1, 1, 0), (1, 2, 0), (1, 3, 0)),
        )
        target_cell = (2, 1, 0)
        target = MiningTargetPlan(
            target_cell, stands[-1], route,
            RemovalProof(target_cell, (target_cell,), ()))
        env = FakeEnv((0.5, 2.0, 0.5))
        controller = RecenteringController()
        executor = PlanExecutor(controller, [target]).reset(env)
        # Reproduce the live transition after the origin was certified: a
        # bounded move reached it, then residual momentum carried the player
        # 0.751 blocks into the still-solid descending column during an aim
        # tick. The pose remains on the proved leg and old level.
        executor._route_index = 1
        env.obs.update(x=1.251, y=2.0, z=0.5)

        correction = executor.actions(env)

        self.assertEqual(correction[0]["k"], ["s", "sneak"])
        self.assertEqual(controller.recenters, [(0.5, 0.5, 0.05)])
        self.assertFalse(controller.mines)
        env.obs.update(x=0.55, y=2.0)
        attack = executor.actions(env)
        self.assertEqual(attack[0]["action"], "left_click_hold")
        self.assertEqual(controller.mines[-1][0], (1, 3, 0))

    def test_never_walks_without_journal_verified_clearance(self):
        env = FakeEnv()
        controller = FakeController()
        executor = PlanExecutor(
            controller, [tunnel_target()], max_clear_attempts=2).reset(env)

        self.assertIsInstance(executor.actions(env), list)
        self.assertIsInstance(executor.actions(env), list)
        veto = executor.actions(env)

        self.assertIsInstance(veto, PolicyVeto)
        self.assertEqual(veto.code, "tunnel_clear_unverified")
        self.assertFalse(controller.moves)

    def test_aim_actions_do_not_consume_exact_break_attempts(self):
        class AimingController(FakeController):
            def mine_exact(self, env, cell, **kwargs):
                del env
                self.mines.append((cell, kwargs))
                if len(self.mines) <= 8:
                    return [{"action": "mouse_move",
                             "coordinate": [500, 500]}]
                return [{"action": "left_click_hold",
                         "ticks": kwargs["ticks"]}]

        env = FakeEnv()
        controller = AimingController()
        executor = PlanExecutor(
            controller, [tunnel_target()],
            max_clear_attempts=8).reset(env)

        for _ in range(8):
            self.assertEqual(
                executor.actions(env)[0]["action"], "mouse_move")
        attack = executor.actions(env)

        self.assertEqual(attack[0]["action"], "left_click_hold")
        self.assertEqual(executor._clear_attempts[(1, 2, 0)], 1)

    def test_tunnel_hold_uses_controller_block_tool_calibration(self):
        class CalibratedController(FakeController):
            @staticmethod
            def mining_hold_ticks(env, cell, *, tool, fallback):
                del env, cell
                self_values = {"tool": tool, "fallback": fallback}
                CalibratedController.values = self_values
                return 26

        env = FakeEnv()
        controller = CalibratedController()
        executor = PlanExecutor(
            controller, [tunnel_target()], tunnel_tool="wooden_pickaxe",
            tunnel_ticks=188).reset(env)

        action = executor.actions(env)

        self.assertEqual(action, [
            {"action": "left_click_hold", "ticks": 26}])
        self.assertEqual(CalibratedController.values, {
            "tool": "wooden_pickaxe", "fallback": 188})

    def test_rejects_route_position_and_protected_support_divergence(self):
        target = tunnel_target()
        controller = FakeController()
        shifted = FakeEnv((2.5, 1.0, 4.5))
        origin = PlanExecutor(controller, [target]).reset(shifted)
        veto = origin.actions(shifted)
        self.assertEqual(veto.code, "route_origin_diverged")

        env = FakeEnv()
        executor = PlanExecutor(controller, [target]).reset(env)
        env.journal.append(Event(
            "block_broken", block=1, drop=4, x=1, y=0, z=0))
        veto = executor.actions(env)
        self.assertEqual(veto.code, "protected_support_removed")

    def test_vetoes_health_loss_fluid_contact_and_visible_hazard(self):
        for code in ("health_lost", "fluid_contact", "hazard_contact"):
            with self.subTest(code=code):
                env = FakeEnv()
                executor = PlanExecutor(
                    FakeController(), [surface_target()]).reset(env)
                if code == "health_lost":
                    env.obs["health"] = 19.0
                    env.journal.append(Event(
                        "vitals_changed", health=19, was_health=20))
                elif code == "fluid_contact":
                    env.journal.append(Event(
                        "player_fluid_state", block=10, flags=FLUID_BODY,
                        x=0, y=1, z=0))
                else:
                    env.obs["blocks"] = [(51, 0, 1, 0)]
                veto = executor.actions(env)
                self.assertIsInstance(veto, PolicyVeto)
                self.assertEqual(veto.code, code)

    def test_revalidates_snapshot_removal_faces_against_live_scan(self):
        env = FakeEnv()
        env.obs["blocks"] = [(10, 2, 1, 0)]
        executor = PlanExecutor(
            FakeController(), [surface_target()]).reset(env)

        veto = executor.actions(env)

        self.assertEqual(veto.code, "removal_hazard_diverged")
        self.assertEqual(veto.observed, ((2, 1, 0),))

    def test_collateral_break_outside_exact_plan_hard_vetoes(self):
        env = FakeEnv()
        executor = PlanExecutor(
            FakeController(), [surface_target()]).reset(env)
        env.journal.append(Event(
            "block_broken", block=3, drop=3, x=1, y=0, z=0))

        veto = executor.actions(env)

        self.assertEqual(veto.code, "unplanned_block_removed")
        self.assertEqual(veto.observed, ((1, 0, 0),))

    def test_expected_drop_is_both_produced_and_collected(self):
        env = FakeEnv()
        controller = FakeController()
        execution = MiningExecution(
            surface_target(), block="log", tool="wooden_axe",
            drop_item="log", drop_count=1)
        executor = PlanExecutor(
            controller, [execution], max_pickup_attempts=2).reset(env)

        self.assertEqual(executor.actions(env)[0]["action"],
                         "left_click_hold")
        env.journal.append(Event(
            "block_broken", block=17, drop=5, x=1, y=1, z=0))
        self.assertEqual(executor.actions(env)[0]["action"], "wait")
        self.assertEqual(controller.pickups[-1][0], "log")
        self.assertFalse(controller.pickups[-1][1]["allow_jump"])
        env.journal.append(Event(
            "item_picked_up", item=5, count=1, x=1, y=1, z=0))

        self.assertIsNone(executor.actions(env))
        self.assertTrue(executor.done)

    def test_exact_inventory_delta_overrides_early_pickup_event(self):
        env = FakeEnv()
        env.obs.update(
            inventory_ids=[0] * 36,
            inventory_counts=[0] * 36,
            inventory_meta=[0] * 36)
        controller = FakeController()
        execution = MiningExecution(
            surface_target(), block="log", drop_item="log", drop_count=1)
        executor = PlanExecutor(
            controller, [execution], max_pickup_attempts=2).reset(env)

        executor.actions(env)
        env.journal.extend((
            Event("block_broken", block=17, drop=5, x=1, y=1, z=0),
            Event("item_picked_up", item=5, count=1, x=1, y=1, z=0),
        ))

        self.assertEqual(executor.actions(env)[0]["action"], "wait")
        self.assertFalse(executor.done)
        env.obs["inventory_ids"][9] = 5
        env.obs["inventory_counts"][9] = 1
        self.assertIsNone(executor.actions(env))
        self.assertTrue(executor.done)

    def test_safe_drop_navigation_does_not_consume_collection_waits(self):
        class RoutedPickupController(FakeController):
            def pickup(self, env, item, **kwargs):
                del env
                self.pickups.append((item, kwargs))
                if len(self.pickups) <= 5:
                    return [
                        {"action": "key", "k": "w", "ticks": 1},
                        {"action": "wait", "ticks": 1},
                    ]
                return [{"action": "wait", "ticks": 2}]

        env = FakeEnv()
        controller = RoutedPickupController()
        execution = MiningExecution(
            surface_target(), block="log", drop_item="log", drop_count=1)
        executor = PlanExecutor(
            controller, [execution], max_pickup_attempts=1).reset(env)
        executor.actions(env)
        env.journal.append(Event(
            "block_broken", block=17, drop=5, x=1, y=1, z=0))

        navigation = [executor.actions(env) for _ in range(5)]

        self.assertTrue(all(actions[0]["action"] == "key"
                            for actions in navigation))
        self.assertEqual(executor._pickup_attempts, 0)
        self.assertEqual(executor._pickup_navigation_attempts, 5)
        wait = executor.actions(env)
        self.assertEqual(wait[0]["action"], "wait")
        self.assertEqual(executor._pickup_attempts, 1)

    def test_pickup_controller_cannot_spin_without_engine_action(self):
        class EmptyPickupController(FakeController):
            def pickup(self, env, item, **kwargs):
                del env, item, kwargs
                return None

        env = FakeEnv()
        execution = MiningExecution(
            surface_target(), block="log", drop_item="log", drop_count=1)
        executor = PlanExecutor(
            EmptyPickupController(), [execution]).reset(env)
        executor.actions(env)
        env.journal.append(Event(
            "block_broken", block=17, drop=5, x=1, y=1, z=0))

        result = executor.actions(env)

        self.assertEqual(result.code, "target_drop_pickup_no_progress")

    def test_pickup_may_leave_mining_stance_after_exact_break(self):
        env = FakeEnv()
        controller = FakeController()
        execution = MiningExecution(
            surface_target(), block="log", drop_item="log", drop_count=1)
        executor = PlanExecutor(controller, [execution]).reset(env)
        executor.actions(env)
        env.journal.append(Event(
            "block_broken", block=17, drop=5, x=1, y=1, z=0))
        env.obs.update(x=1.5, z=0.5)

        pickup = executor.actions(env)

        self.assertEqual(pickup[0]["action"], "wait")
        self.assertEqual(controller.pickups[-1][1]["target"], (1.5, 0.5))

    def test_wrong_tool_drop_is_rejected_before_pickup(self):
        env = FakeEnv()
        execution = MiningExecution(
            surface_target(), block="log", tool="wooden_axe",
            drop_item="log", drop_count=1)
        executor = PlanExecutor(FakeController(), [execution]).reset(env)
        executor.actions(env)
        env.journal.append(Event(
            "block_broken", block=17, drop=0, x=1, y=1, z=0))

        veto = executor.actions(env)

        self.assertEqual(veto.code, "target_drop_mismatch")

    def test_destroyed_block_metadata_is_part_of_live_contract(self):
        env = FakeEnv()
        execution = MiningExecution(
            surface_target(), block="stone", block_metadata=(0,))
        executor = PlanExecutor(FakeController(), [execution]).reset(env)
        executor.actions(env)
        env.journal.append(Event(
            "block_broken", block=1, meta=1, drop=1,
            x=1, y=1, z=0))

        veto = executor.actions(env)

        self.assertEqual(veto.code, "target_block_state_mismatch")
        self.assertIn("expected=(0,)", veto.reason)
        self.assertIn("observed=(1,)", veto.reason)

    def test_drop_mismatch_reports_item_events_and_inventory(self):
        env = FakeEnv()
        env.obs.update(
            hotbar_ids=[4, 270] + [0] * 7,
            hotbar_counts=[3, 1] + [0] * 7,
            hotbar_meta=[0, 7] + [0] * 7,
            hotbar_sel=1,
            inv_counts={"cobblestone": 3, "wooden_pickaxe": 1},
        )
        execution = MiningExecution(
            surface_target(), block="stone", tool=270,
            drop_item=4, drop_count=1)
        executor = PlanExecutor(FakeController(), [execution]).reset(env)
        executor.actions(env)
        env.journal.extend([
            Event("block_broken", tick=22, block=1, meta=1, drop=1,
                  tool=270, x=1, y=1, z=0),
            Event("item_dropped", tick=22, item=1, count=1, meta=1,
                  cause="harvest"),
        ])

        veto = executor.actions(env)

        self.assertEqual(veto.code, "target_drop_mismatch")
        self.assertIn("'meta': 1", veto.reason)
        self.assertIn("item_events=", veto.reason)
        self.assertIn("'event': 'item_dropped'", veto.reason)
        self.assertIn("selected=(1, 270)", veto.reason)
        self.assertIn("inventory={'cobblestone': 3", veto.reason)

    def test_optional_lighting_runs_only_at_proved_stands(self):
        env = FakeEnv()
        controller = FakeController()
        responses = [[{"action": "wait", "ticks": 1}], None]
        calls = []

        def lighting(live_env):
            calls.append((live_env.obs["x"], live_env.obs["z"]))
            return responses.pop(0)

        executor = PlanExecutor(
            controller, [tunnel_target()], lighting=lighting).reset(env)
        light = executor.actions(env)
        mine = executor.actions(env)

        self.assertEqual(light[0]["action"], "wait")
        self.assertEqual(mine[0]["action"], "left_click_hold")
        self.assertEqual(calls, [(0.5, 0.5), (0.5, 0.5)])
        self.assertFalse(controller.moves)

    def test_rejects_disconnected_target_legs_at_construction(self):
        first = surface_target()
        other_stance = (8, 1, 0)
        other_cell = (9, 1, 0)
        second = MiningTargetPlan(
            other_cell, other_stance, RoutePlan.surface((other_stance,)),
            RemovalProof(other_cell, (other_cell,), ()))

        with self.assertRaisesRegex(ValueError, "preceding mining stance"):
            PlanExecutor(FakeController(), [first, second])


def fluid_cast_execution(casts=1):
    work = (0, 1, 0)
    catalyst = (1, 0, 0)
    result = (1, 0, 1)
    catalyst_step = PlaceStep.on(
        catalyst, (0, 0, 0), order=0, material="water",
        role="cast_catalyst", stance=work)
    reactant_step = PlaceStep.on(
        result, (0, 0, 1), order=1, material="lava",
        role="cast_reactant", stance=work)
    plan = FluidCastPlan(
        kind="contained_fluid_cast", snapshot_sha256="a" * 64,
        world_seed=7, decision_seed=11, orientation="test",
        anchor=catalyst, casts_required=casts,
        catalyst_fluid="water", reactant_fluid="lava",
        result_block="obsidian", catalyst_cell=catalyst,
        result_cell=result, access_clear_cells=(),
        excavation_cells=(catalyst, result),
        support_cells=((1, -1, 0), (1, -1, 1)),
        backing_cells=((0, 0, 0), (0, 0, 1)),
        fluid_envelope=(catalyst, result),
        temporary_cells=(catalyst, result),
        cleanup_cells=(catalyst, result), setup_placements=(),
        catalyst_placement=catalyst_step,
        reactant_placement=reactant_step,
        approach_stands=(work,), work_stance=work,
        catalyst_recovery_stance=work, result_mining_stance=work,
        interactions=(
            InteractionRay("place_catalyst", work, (0, 0, 0),
                           ((0, 0, 0),)),
            InteractionRay("place_reactant", work, (0, 0, 1),
                           ((0, 0, 1),)),
            InteractionRay("recover_catalyst", work, catalyst,
                           (catalyst,)),
            InteractionRay("mine_result", work, result, (result,)),
        ), hazard_clearance=1)
    coolant = FluidSourceExecution(
        "coolant", (-1, 1, 1),
        RoutePlan.surface((work, (-1, 1, 0))),
        RoutePlan.surface(((-1, 1, 0), work)),
        "water", "water_bucket")
    casting = []
    for index in range(casts):
        intermediate = (() if index == 0 else ((index, 1, 0),))
        stands = (work, *intermediate, (index, 1, -1))
        casting.append(FluidSourceExecution(
            "casting", (index, 1, -2),
            RoutePlan.surface(stands),
            RoutePlan.surface(tuple(reversed(stands))),
            "lava", "lava_bucket"))
    return FluidCastExecution(plan, (coolant, *casting))


class FluidController(FakeController):
    BLOCKS = {
        "stone": 1, "flowing_water": 8, "water": 9,
        "flowing_lava": 10, "lava": 11, "obsidian": 49,
    }
    ITEMS = {
        "bucket": 325, "water_bucket": 326, "lava_bucket": 327,
        "obsidian": 49, "diamond_pickaxe": 278,
    }

    def __init__(self):
        super().__init__()
        self.target = (0, None)
        self.destination = None

    @classmethod
    def _block_ids(cls, value):
        values = value if isinstance(value, (list, tuple)) else (value,)
        return tuple(cls.BLOCKS.get(item, item) for item in values)

    @classmethod
    def _item_ids(cls, value):
        values = value if isinstance(value, (list, tuple)) else (value,)
        return tuple(cls.ITEMS.get(item, item) for item in values)

    @classmethod
    def select(cls, env, item):
        wanted = frozenset(cls._item_ids(item))
        slot = next((index for index, (found, count) in enumerate(zip(
            env.obs["hotbar_ids"], env.obs["hotbar_counts"]))
                     if found in wanted and count > 0), None)
        if slot is None:
            return None
        if env.obs.get("hotbar_sel", 0) == slot:
            return []
        return [{"action": "hotbar", "slot": slot}]

    def click_target(self, env):
        del env
        return self.target

    def click_destination(self, env):
        del env
        return self.destination

    @staticmethod
    def aim(env, cell):
        del env, cell
        return [{"action": "mouse_move", "dx": 1, "dy": 0}]

    @staticmethod
    def aim_point(env, point):
        del env, point
        return [{"action": "mouse_move", "dx": 1, "dy": 0}]


class FluidCastExecutorTests(unittest.TestCase):
    def env(self):
        env = FakeEnv((0.5, 1.0, 0.5))
        env.obs.update(
            hotbar_sel=0,
            hotbar_ids=[325] + [0] * 8,
            hotbar_counts=[1] + [0] * 8,
            yaw=0.0, pitch=0.0)
        return env

    @staticmethod
    def event(env, name, **values):
        env.journal.append(Event(name, **values))

    @staticmethod
    def set_slot(env, item, count=1, slot=0):
        env.obs["hotbar_ids"][slot] = item
        env.obs["hotbar_counts"][slot] = count

    def test_input_requires_unique_ordered_source_contract(self):
        execution = fluid_cast_execution(2)
        duplicate = FluidSourceExecution(
            "casting", execution.sources[1].source_cell,
            execution.sources[2].outbound_route,
            execution.sources[2].return_route,
            "lava", "lava_bucket")
        with self.assertRaisesRegex(ValueError, "unique"):
            FluidCastExecution(
                execution.plan,
                (execution.sources[0], execution.sources[1], duplicate))
        with self.assertRaisesRegex(ValueError, "coolant first"):
            FluidCastExecution(
                execution.plan,
                (execution.sources[1], execution.sources[0],
                 execution.sources[2]))

    def test_execution_validates_wait_and_open_dry_coolant_contract(self):
        execution = fluid_cast_execution()
        with self.assertRaisesRegex(ValueError, "casting_wait_ticks"):
            FluidCastExecution(
                execution.plan, execution.sources, casting_wait_ticks=0)
        with self.assertRaisesRegex(ValueError, "recover_coolant=True"):
            FluidCastExecution(
                execution.plan, execution.sources, recover_coolant=False)

    def test_distinct_return_route_is_validated_and_executed_directly(self):
        execution = fluid_cast_execution()
        work = execution.plan.work_stance
        outbound = RoutePlan.surface((work, (-1, 1, 0)))
        returned_surface = RoutePlan.surface((
            (-1, 1, 0), (-1, 1, 1), (0, 1, 1), work))
        return_only_mine = (-1, 2, 1)
        returned = RoutePlan(
            "tunnel", returned_surface.stands,
            (*returned_surface.reserved_cells, return_only_mine),
            returned_surface.support_cells, (return_only_mine,))
        coolant = FluidSourceExecution(
            "coolant", (-1, 1, 2), outbound, returned,
            "water", "water_bucket")
        execution = FluidCastExecution(
            execution.plan, (coolant, *execution.sources[1:]))
        env = self.env()
        env.obs.update(x=-0.5, z=0.5)
        controller = FluidController()
        executor = FluidCastExecutor(controller, execution).reset(env)
        executor._set_phase(env, "coolant_route_back")
        self.event(env, "block_broken", block=1, drop=1,
                   x=-1, y=2, z=1)

        action = executor.actions(env)

        self.assertEqual(action[0]["action"], "key")
        self.assertEqual(controller.moves[-1][:2], (-0.5, 1.5))

        disconnected = RoutePlan.surface(((2, 1, 0), work))
        with self.assertRaisesRegex(ValueError, "same dry pickup stance"):
            FluidSourceExecution(
                "coolant", (-1, 1, 2), outbound, disconnected,
                "water", "water_bucket")

    def test_source_route_caps_ascending_transition_to_one_observation(self):
        env = self.env()
        controller = FluidController()
        executor = FluidCastExecutor(
            controller, fluid_cast_execution()).reset(env)
        stands = ((0, 1, 0), (1, 2, 0))
        route = SimpleNamespace(kind="tunnel", mine_cells=())

        action = executor._follow_stands(env, stands, route=route)

        self.assertEqual(action[0]["action"], "key")
        self.assertTrue(controller.moves[-1][2]["step_up"])
        self.assertEqual(
            controller.moves[-1][2]["movement_cap_blocks"], 0.3)

    def test_source_route_swims_through_safe_coolant_current(self):
        env = self.env()
        controller = FluidController()
        executor = FluidCastExecutor(
            controller, fluid_cast_execution()).reset(env)
        self.event(env, "player_fluid_state", block=9, flags=FLUID_BODY,
                   x=0, y=1, z=0)
        stands = ((0, 1, 0), (1, 2, 0))
        route = SimpleNamespace(kind="tunnel", mine_cells=())

        action = executor._follow_stands(env, stands, route=route)

        self.assertEqual(action[0]["action"], "key")
        self.assertTrue(controller.moves[-1][2]["swim"])
        self.assertEqual(
            controller.moves[-1][2]["movement_cap_blocks"], 1.5)

    def test_source_route_recovers_from_safe_coolant_current_drift(self):
        env = self.env()
        env.obs.update(x=0.5, z=1.7)
        controller = FluidController()
        executor = FluidCastExecutor(
            controller, fluid_cast_execution()).reset(env)
        executor._route_index = 1
        self.event(env, "player_fluid_state", block=8, flags=FLUID_BODY,
                   x=0, y=1, z=1)
        stands = ((0, 1, 0), (1, 1, 0))
        route = SimpleNamespace(kind="tunnel", mine_cells=())

        action = executor._follow_stands(env, stands, route=route)

        self.assertEqual(action[0]["action"], "key")
        self.assertEqual(controller.moves[-1][:2], (0.5, 0.5))
        self.assertTrue(controller.moves[-1][2]["swim"])

    def test_source_route_recenters_exactly_before_bucket_interaction(self):
        env = self.env()
        env.obs.update(z=0.84)
        controller = FluidController()
        executor = FluidCastExecutor(
            controller, fluid_cast_execution()).reset(env)

        action = executor._follow_stands(env, ((0, 1, 0),))

        self.assertEqual(action[0]["action"], "key")
        self.assertEqual(controller.moves[-1][:2], (0.5, 0.5))
        self.assertFalse(controller.moves[-1][2]["allow_jump"])
        self.assertEqual(
            controller.moves[-1][2]["movement_cap_blocks"], 0.3)

    def test_source_route_uses_crouched_micro_recenter_at_final_stance(self):
        class MicroController(FluidController):
            def __init__(self):
                super().__init__()
                self.recenters = []

            def recenter_interaction_stance(
                    self, env, x, z, *, tolerance):
                del env
                self.recenters.append((x, z, tolerance))
                return [{"action": "key", "k": ["a", "sneak"],
                         "ticks": 1}]

        env = self.env()
        env.obs.update(z=0.59)
        controller = MicroController()
        executor = FluidCastExecutor(
            controller, fluid_cast_execution()).reset(env)

        action = executor._follow_stands(env, ((0, 1, 0),))

        self.assertEqual(action, [{
            "action": "key", "k": ["a", "sneak"], "ticks": 1}])
        self.assertEqual(controller.recenters, [(0.5, 0.5, 0.02)])
        self.assertFalse(controller.moves)

    def test_source_route_reclears_exact_reoccupied_headroom(self):
        class OccludedController(FluidController):
            def mine_exact(self, env, cell, **kwargs):
                if cell == (1, 3, 0):
                    return PolicyVeto(
                        "mine_target_unproven",
                        "the declared prior headroom occludes this target",
                        (1.5, 0.5), ((0, 3, 0),))
                return super().mine_exact(env, cell, **kwargs)

        env = self.env()
        self.set_slot(env, 278, slot=1)
        env.obs["hotbar_sel"] = 1
        controller = OccludedController()
        controller.target = (1, (0, 3, 0))
        executor = FluidCastExecutor(
            controller, fluid_cast_execution()).reset(env)
        route = RoutePlan(
            "tunnel", ((0, 1, 0), (1, 2, 0)),
            ((0, 1, 0), (0, 2, 0), (0, 3, 0),
             (1, 2, 0), (1, 3, 0)),
            ((0, 0, 0), (1, 1, 0)),
            ((0, 3, 0), (1, 3, 0)))
        self.event(env, "block_broken", block=13, drop=0,
                   x=0, y=3, z=0)

        action = executor._follow_stands(
            env, route.stands, route=route)

        self.assertEqual(action, [{
            "action": "left_click_hold", "ticks": 80}])
        self.assertEqual(executor._route_clear_attempts[(0, 3, 0)], 1)

    def test_source_route_diagnostics_keep_json_safe_control_trace(self):
        env = self.env()
        env.obs.update(hotbar_sel=1)
        self.set_slot(env, 278, slot=1)
        controller = FluidController()
        controller.target = (1, (1, 2, 0))
        executor = FluidCastExecutor(
            controller, fluid_cast_execution()).reset(env)
        route = RoutePlan(
            "tunnel", ((0, 1, 0), (1, 1, 0)),
            ((0, 1, 0), (0, 2, 0), (1, 1, 0), (1, 2, 0)),
            ((0, 0, 0), (1, 0, 0)), ((1, 2, 0),))

        action = executor._follow_stands(env, route.stands, route=route)
        diagnostics = executor.diagnostics()

        self.assertEqual(action[0]["action"], "left_click_hold")
        trace = diagnostics["recent_route_clear_control"]["(1, 2, 0)"]
        self.assertEqual(trace[-1]["click"], (1, (1, 2, 0)))
        self.assertEqual(trace[-1]["selected"], (1, 278))
        json.dumps(diagnostics)

    def test_setup_recovers_backpacked_scaffold_before_placement(self):
        execution = fluid_cast_execution()
        step = PlaceStep.on(
            (2, 0, 0), (1, 0, 0), order=0, material="stone",
            role="mold_scaffold", stance=execution.plan.work_stance)
        plan = replace(execution.plan, setup_placements=(step,))
        execution = replace(execution, plan=plan)
        recovered = []

        def recover(_env, item, minimum):
            recovered.append((item, minimum))
            return [{"action": "key", "k": "e", "ticks": 1}]

        env = self.env()
        executor = FluidCastExecutor(
            FluidController(), execution,
            item_recovery=recover).reset(env)
        executor._setup_index = len(
            plan.access_clear_cells + plan.excavation_cells)

        action = executor._setup_actions(env)

        self.assertEqual(action, [{
            "action": "key", "k": "e", "ticks": 1}])
        self.assertEqual(recovered, [("stone", 1)])

    def test_scoop_is_bucket_target_and_same_slot_gated(self):
        env = self.env()
        controller = FluidController()
        executor = FluidCastExecutor(
            controller, fluid_cast_execution()).reset(env)
        # Prove setup excavation and move to the coolant source.
        executor.actions(env)
        self.event(env, "block_broken", block=1, drop=1,
                   x=1, y=0, z=0)
        executor.actions(env)
        self.event(env, "block_broken", block=1, drop=1,
                   x=1, y=0, z=1)
        executor.actions(env)
        env.obs.update(x=-0.5, z=0.5)
        self.assertEqual(
            executor.actions(env), [{"action": "wait", "ticks": 2}])
        aimed = executor.actions(env)
        self.assertEqual(aimed[0]["action"], "mouse_move")
        self.event(env, "bucket_target", block=9, x=-1, y=1, z=1,
                   meta=0, slot=0)
        click = executor.actions(env)
        self.assertEqual(click, [{"action": "right_click"}])
        # A filled bucket without the exact same-slot transition cannot pass.
        self.set_slot(env, 326)
        self.event(env, "block_broken", block=9, drop=0,
                   x=-1, y=1, z=1)
        missing = executor.actions(env)
        self.assertIsInstance(missing, PolicyVeto)
        self.assertEqual(missing.code, "fluid_cast_empty_bucket_missing")

    def test_scoop_waits_for_bucket_probe_when_camera_is_already_aimed(self):
        class SettledAimController(FluidController):
            @staticmethod
            def aim(env, cell):
                del env, cell
                return []

        env = self.env()
        execution = fluid_cast_execution()
        executor = FluidCastExecutor(
            SettledAimController(), execution).reset(env)

        action = executor._scoop(env, execution.sources[0])

        self.assertEqual(action, [{"action": "wait", "ticks": 1}])
        self.assertEqual(executor._control_attempts, 1)

    def test_reaction_wait_uses_execution_setting(self):
        base = fluid_cast_execution()
        execution = FluidCastExecution(
            base.plan, base.sources, casting_wait_ticks=7)
        env = self.env()
        executor = FluidCastExecutor(
            FluidController(), execution).reset(env)
        executor._set_phase(env, "verify_reaction")

        self.assertEqual(
            executor.actions(env), [{"action": "wait", "ticks": 7}])

    def test_excavation_hold_resolves_live_block_hardness(self):
        class DynamicController(FluidController):
            def __init__(self):
                super().__init__()
                self.hold_calls = []

            def mining_hold_ticks(self, env, cell, *, tool, fallback):
                del env
                self.hold_calls.append((cell, tool, fallback))
                return 13

        env = self.env()
        controller = DynamicController()
        execution = fluid_cast_execution()
        action = FluidCastExecutor(controller, execution).reset(env).actions(env)

        self.assertEqual(action, [{"action": "left_click_hold", "ticks": 13}])
        self.assertEqual(controller.hold_calls, [(
            execution.plan.excavation_cells[0],
            execution.excavation_tool,
            execution.excavation_ticks,
        )])

    def test_result_pickup_can_delegate_backpack_recovery(self):
        env = self.env()
        calls = []

        def recover(live_env, item, minimum):
            calls.append((live_env, item, minimum))
            return [{"action": "inventory_swap", "slot": 12}]

        executor = FluidCastExecutor(
            FluidController(), fluid_cast_execution(),
            result_recovery=recover).reset(env)
        executor._mine_event_start = 0
        executor._result_item_baseline = 0
        executor._set_phase(env, "pickup_result")
        self.event(env, "item_picked_up", item=49, count=1, meta=0,
                   x=1, y=0, z=1)

        action = executor.actions(env)

        self.assertEqual(action, [{"action": "inventory_swap", "slot": 12}])
        self.assertEqual(calls, [(env, "obsidian", 1)])
        self.assertEqual(executor._phase_attempts, 0)

    def test_repeated_cast_reuses_live_water_without_replacing_it(self):
        env = self.env()
        controller = FluidController()
        executor = FluidCastExecutor(
            controller, fluid_cast_execution(casts=2)).reset(env)
        executor._mine_event_start = 0
        executor._result_item_baseline = 0
        executor._set_phase(env, "pickup_result")
        self.event(env, "item_picked_up", item=49, count=1, meta=0,
                   x=1, y=0, z=1)
        self.set_slot(env, 49, slot=1)

        action = executor.actions(env)

        self.assertEqual(executor._cast_index, 1)
        self.assertEqual(executor.phase, "casting_route_out")
        self.assertEqual(action[0]["action"], "key")
        self.assertNotIn(
            "place_catalyst",
            tuple(item["to"] for item in executor._phase_trace))

    def test_repeated_cast_returns_to_work_stance_after_drop_pickup(self):
        env = self.env()
        controller = FluidController()
        executor = FluidCastExecutor(
            controller, fluid_cast_execution(casts=2)).reset(env)
        executor._mine_event_start = 0
        executor._result_item_baseline = 0
        executor._set_phase(env, "pickup_result")
        self.event(env, "item_picked_up", item=49, count=1, meta=0,
                   x=1, y=0, z=1)
        self.set_slot(env, 49, slot=1)
        env.obs.update(x=1.5, y=0.0, z=1.5)

        action = executor.actions(env)

        self.assertEqual(executor.phase, "post_pickup_recenter")
        self.assertEqual(action[0]["action"], "key")
        self.assertEqual(controller.moves[-1][:2], (0.5, 0.5))
        self.assertTrue(controller.moves[-1][2]["step_up"])

        for _ in range(executor.max_move_attempts - 1):
            self.assertIsInstance(executor.actions(env), list)
        self.assertEqual(
            executor._phase_attempts, executor.max_move_attempts)

    def test_one_cast_keeps_water_down_while_mining_then_recovers_it(self):
        env = self.env()
        controller = FluidController()
        execution = fluid_cast_execution()
        executor = FluidCastExecutor(controller, execution).reset(env)

        # Excavate the exact two-cell mold.
        self.assertEqual(executor.actions(env)[0]["action"],
                         "left_click_hold")
        self.event(env, "block_broken", block=1, drop=1,
                   x=1, y=0, z=0)
        self.assertEqual(executor.actions(env)[0]["action"],
                         "left_click_hold")
        self.event(env, "block_broken", block=1, drop=1,
                   x=1, y=0, z=1)

        # Route out, scoop one source-only water, and route back.
        self.assertEqual(executor.actions(env)[0]["action"], "key")
        env.obs.update(x=-0.5, z=0.5)
        self.event(env, "bucket_target", block=9, x=-1, y=1, z=1,
                   meta=0, slot=0)
        self.assertEqual(
            executor.actions(env), [{"action": "wait", "ticks": 2}])
        self.assertEqual(executor.actions(env), [{"action": "right_click"}])
        self.event(env, "slot_changed", slot=0, was_item=325,
                   was_count=1, item=326, count=1, meta=0)
        self.event(env, "block_broken", block=9, drop=0,
                   x=-1, y=1, z=1)
        self.set_slot(env, 326)
        self.assertEqual(executor.actions(env)[0]["action"], "key")
        env.obs.update(x=0.5, z=0.5)
        self.assertEqual(
            executor.actions(env), [{"action": "wait", "ticks": 2}])

        # Establish catalyst before any lava is collected.
        controller.target = (1, (0, 0, 0))
        controller.destination = (1, 0, 0)
        self.assertEqual(executor.actions(env), [{"action": "right_click"}])
        self.event(env, "block_placed", block=9, item=326, slot=0,
                   x=1, y=0, z=0)
        self.event(env, "slot_changed", slot=0, was_item=326,
                   was_count=1, item=325, count=1, meta=0)
        self.set_slot(env, 325)

        # Route to one distinct lava source, scoop, and return to the mold.
        self.assertEqual(executor.actions(env)[0]["action"], "key")
        env.obs.update(x=0.5, z=-0.5)
        self.event(env, "bucket_target", block=11, x=0, y=1, z=-2,
                   meta=0, slot=0)
        self.assertEqual(
            executor.actions(env), [{"action": "wait", "ticks": 2}])
        self.assertEqual(executor.actions(env), [{"action": "right_click"}])
        self.event(env, "slot_changed", slot=0, was_item=325,
                   was_count=1, item=327, count=1, meta=0)
        self.event(env, "block_broken", block=11, drop=0,
                   x=0, y=1, z=-2)
        self.set_slot(env, 327)
        self.assertEqual(executor.actions(env)[0]["action"], "key")
        env.obs.update(x=0.5, z=0.5)
        self.assertEqual(
            executor.actions(env), [{"action": "wait", "ticks": 2}])

        # Place reactant; the live view proves the same-tick reaction result.
        controller.target = (1, (0, 0, 1))
        controller.destination = (1, 0, 1)
        self.assertEqual(executor.actions(env), [{"action": "right_click"}])
        self.event(env, "block_placed", block=11, item=327, slot=0,
                   x=1, y=0, z=1)
        self.event(env, "slot_changed", slot=0, was_item=327,
                   was_count=1, item=325, count=1, meta=0)
        self.set_slot(env, 325)
        env.obs["blocks"] = [(9, 1, 0, 0), (49, 1, 0, 1)]

        # Mine with the water still down, shielding the exposed mold.
        controller.target = (49, (1, 0, 1))
        controller.destination = None
        self.assertEqual(executor.actions(env)[0]["action"],
                         "left_click_hold")

        # Exact result break and pickup are proved before catalyst recovery.
        self.event(env, "block_broken", block=49, tool=278, drop=49,
                   x=1, y=0, z=1)
        env.obs["blocks"] = [(9, 1, 0, 0)]
        self.assertEqual(executor.actions(env)[0]["action"], "wait")
        self.event(env, "item_picked_up", item=49, count=1, meta=0,
                   x=1, y=0, z=1)
        self.set_slot(env, 49, slot=1)

        # Recover the singleton water bucket only after the final result.
        self.event(env, "bucket_target", block=9, x=1, y=0, z=0,
                   meta=0, slot=0)
        self.assertEqual(executor.actions(env), [{"action": "right_click"}])
        self.event(env, "slot_changed", slot=0, was_item=325,
                   was_count=1, item=326, count=1, meta=0)
        self.event(env, "block_broken", block=9, drop=0,
                   x=1, y=0, z=0)
        self.set_slot(env, 326)
        env.obs["blocks"] = []
        self.assertIsNone(executor.actions(env))
        self.assertTrue(executor.done)
        self.assertEqual(env.obs["hotbar_ids"][0], 326)

    def test_completion_recenters_at_exact_certified_work_stance(self):
        env = self.env()
        controller = FluidController()
        executor = FluidCastExecutor(
            controller, fluid_cast_execution()).reset(env)
        self.set_slot(env, 326)
        executor._set_phase(env, "cleanup")
        env.obs.update(x=1.5, y=0.0, z=1.5)

        action = executor.actions(env)

        self.assertEqual(action[0]["action"], "key")
        self.assertFalse(executor.done)
        self.assertEqual(controller.moves[-1][:2], (0.5, 0.5))
        self.assertTrue(controller.moves[-1][2]["step_up"])

        env.obs.update(x=0.5, y=1.0, z=0.5)
        self.assertIsNone(executor.actions(env))
        self.assertTrue(executor.done)

    def test_player_fluid_contact_is_a_structured_veto(self):
        env = self.env()
        executor = FluidCastExecutor(
            FluidController(), fluid_cast_execution()).reset(env)
        self.event(env, "player_fluid_state", block=11, flags=FLUID_BODY,
                   x=0, y=1, z=0)

        veto = executor.actions(env)

        self.assertEqual(veto.code, "fluid_cast_fluid_contact")

    def test_coolant_contact_is_safe_but_casting_fluid_is_not(self):
        env = self.env()
        executor = FluidCastExecutor(
            FluidController(), fluid_cast_execution()).reset(env)
        self.event(env, "player_fluid_state", block=9, flags=FLUID_BODY,
                   x=0, y=1, z=0)

        action = executor.actions(env)

        self.assertIsInstance(action, list)
        self.assertNotIsInstance(action, PolicyVeto)

        flowing = self.env()
        flowing_executor = FluidCastExecutor(
            FluidController(), fluid_cast_execution()).reset(flowing)
        self.event(flowing, "player_fluid_state", block=8,
                   flags=FLUID_BODY, x=0, y=1, z=0)
        self.assertNotIsInstance(
            flowing_executor.actions(flowing), PolicyVeto)

        dangerous = self.env()
        dangerous_executor = FluidCastExecutor(
            FluidController(), fluid_cast_execution()).reset(dangerous)
        self.event(dangerous, "player_fluid_state", block=10,
                   flags=FLUID_BODY, x=0, y=1, z=0)
        self.assertEqual(
            dangerous_executor.actions(dangerous).code,
            "fluid_cast_fluid_contact")

    def test_source_routes_offer_existing_lighting_before_movement(self):
        class Lighting:
            def __init__(self):
                self.observed = 0
                self.remembered = 0
                self.lit = 0

            def observe_route_travel(self, env):
                del env
                self.observed += 1

            def remember_route_position(self, env):
                del env
                self.remembered += 1

            def light_route(self, env):
                del env
                self.lit += 1
                return [{"action": "wait", "ticks": 1}]

        env = self.env()
        controller = FluidController()
        lighting = Lighting()
        executor = FluidCastExecutor(
            controller, fluid_cast_execution(),
            lighting=lighting).reset(env)
        executor._set_phase(env, "coolant_route_out")

        action = executor.actions(env)

        self.assertEqual(action, [{"action": "wait", "ticks": 1}])
        self.assertEqual(lighting.observed, 1)
        # Reset remembers the existing cadence; the proved source-route
        # observation updates it once more without restarting the odometer.
        self.assertEqual(lighting.remembered, 2)
        self.assertEqual(lighting.lit, 1)
        self.assertFalse(controller.moves)

    def test_source_route_overdue_light_veto_is_strict_and_optional(self):
        class Lighting:
            @staticmethod
            def light_route(env):
                del env
                return None

            @staticmethod
            def route_light_overdue():
                return True

            @staticmethod
            def route_light_debug(env):
                del env
                return {"distance": 11}

        env = self.env()
        executor = FluidCastExecutor(
            FluidController(), fluid_cast_execution(),
            lighting=Lighting()).reset(env)
        executor._set_phase(env, "coolant_route_out")

        veto = executor.actions(env)

        self.assertEqual(veto.code, "fluid_cast_route_light_overdue")
        self.assertIn("distance", veto.reason)

        controller = FluidController()
        filtered = FluidCastExecutor(
            controller, fluid_cast_execution(), lighting=Lighting(),
            lighting_route_kinds=("tunnel",)).reset(env)
        filtered._set_phase(env, "coolant_route_out")
        self.assertEqual(filtered.actions(env)[0]["action"], "key")


if __name__ == "__main__":
    unittest.main()
