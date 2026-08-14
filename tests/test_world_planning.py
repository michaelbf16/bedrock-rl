import json
import struct
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from bedrock_rl.sdg.policy.planning import (
    ClusterPlan,
    GoalSpec,
    MiningTargetPlan,
    ResourcePlan,
    ResourceRequirement,
    RoutePlan,
    WorldPlan,
    WorldPlanBuilder,
    WorldPlanningRejected,
    _drop_landing_within_reach,
    _drop_fall_within,
    _excavation_clear_rays,
    _excavation_falling_blocks_stable,
    _target_falling_blocks_stable,
    fluid_seal_ledger,
    natural_harvest_drop,
    require_fluid_seal_budget,
    require_tool_durability_budget,
    tool_durability_ledgers,
)
from bedrock_rl.sdg.policy.spatial import (
    ExcavationPlan,
    RemovalProof,
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


def write_planning_snapshot(path, *, shift=0, seed=77, include_diamond=True):
    rx, ry, rz = shift - 8, -5, -6
    nx, ny, nz = 24, 14, 13
    buf = bytearray(HEAD_SIZE + nx * ny * nz * 2 + 4)
    struct.pack_into("<4sIqqii", buf, 0, b"BSNP", 1, seed, 3, 0, 0)
    struct.pack_into("<3d", buf, POS_OFF, shift + 0.5, 1.0, 0.5)
    struct.pack_into("<2f", buf, ANGLE_OFF, 0.0, 0.0)
    struct.pack_into("<I", buf, ITEM_COUNT_OFF, 0)
    struct.pack_into("<6i", buf, REGION_HEADER_OFF,
                     rx, ry, rz, nx, ny, nz)

    def put(x, y, z, block_id, meta=0):
        index = ((x - rx) * ny + (y - ry)) * nz + (z - rz)
        struct.pack_into("<H", buf, HEAD_SIZE + index * 2,
                         block_id << 4 | meta)

    # Solid underground and a clear surface make both walking and mined
    # staircase routes testable without special coordinate knowledge.
    for x in range(rx, rx + nx):
        for y in range(ry, 1):
            for z in range(rz, rz + nz):
                put(x, y, z, 1)

    deposits = {
        17: [(3, 1, 0), (3, 2, 0), (3, 3, 0)],
        16: [(5, -1, 0), (5, -1, 1)],
        15: [(7, -2, 0), (7, -2, 1), (7, -1, 0)],
        13: [(2, 0, 3), (2, 1, 3)],
        49: [(6, 0, 3), (6, 0, 4), (7, 0, 3)],
    }
    if include_diamond:
        deposits[56] = [(9, -3, 0), (9, -3, 1), (10, -3, 0)]
    for block_id, cells in deposits.items():
        for x, y, z in cells:
            put(shift + x, y, z, block_id)
    struct.pack_into("<I", buf, len(buf) - 4, 0)
    path.write_bytes(buf)


def write_quarry_snapshot(path, *, count=24, variant_count=0):
    """Write a long, surface-accessible stone face with stable supports."""

    rx, ry, rz = -3, 0, -3
    nx, ny, nz = count + 8, 6, 7
    buf = bytearray(HEAD_SIZE + nx * ny * nz * 2 + 4)
    struct.pack_into("<4sIqqii", buf, 0, b"BSNP", 1, 81, 3, 0, 0)
    struct.pack_into("<3d", buf, POS_OFF, 0.5, 2.0, 0.5)
    struct.pack_into("<2f", buf, ANGLE_OFF, 0.0, 0.0)
    struct.pack_into("<I", buf, ITEM_COUNT_OFF, 0)
    struct.pack_into("<6i", buf, REGION_HEADER_OFF,
                     rx, ry, rz, nx, ny, nz)

    def put(x, y, z, block_id, meta=0):
        index = ((x - rx) * ny + (y - ry)) * nz + (z - rz)
        struct.pack_into("<H", buf, HEAD_SIZE + index * 2,
                         block_id << 4 | meta)

    # Dirt avoids merging the target face into an enormous stone floor.
    for x in range(rx, rx + nx):
        for z in range(rz, rz + nz):
            put(x, 0, z, 3)
            put(x, 1, z, 3)
    for x in range(2, count + 2):
        put(x, 2, 1, 1, 1 if x < variant_count + 2 else 0)
    struct.pack_into("<I", buf, len(buf) - 4, 0)
    path.write_bytes(buf)


def write_protected_clusters_snapshot(
        path, *, upper_block=17, horizontal=False):
    """Two reachable log clusters; one cell in the nearer cluster is kept."""

    rx, ry, rz = -3, 0, -3
    nx, ny, nz = 12, 7, 7
    buf = bytearray(HEAD_SIZE + nx * ny * nz * 2 + 4)
    struct.pack_into("<4sIqqii", buf, 0, b"BSNP", 1, 91, 3, 0, 0)
    struct.pack_into("<3d", buf, POS_OFF, 0.5, 2.0, 0.5)
    struct.pack_into("<2f", buf, ANGLE_OFF, 0.0, 0.0)
    struct.pack_into("<I", buf, ITEM_COUNT_OFF, 0)
    struct.pack_into("<6i", buf, REGION_HEADER_OFF,
                     rx, ry, rz, nx, ny, nz)

    def put(x, y, z, block_id):
        index = ((x - rx) * ny + (y - ry)) * nz + (z - rz)
        struct.pack_into("<H", buf, HEAD_SIZE + index * 2,
                         block_id << 4)

    for x in range(rx, rx + nx):
        for z in range(rz, rz + nz):
            put(x, 0, z, 3)
            put(x, 1, z, 3)
    # The near cluster has two cells but only its lower cell is usable. The
    # second one-cell cluster must supply the remaining requested log.
    put(2, 2, 0, 17)
    if horizontal:
        put(3, 2, 0, upper_block)
    else:
        put(2, 3, 0, upper_block)
    put(5, 2, 0, 17)
    struct.pack_into("<I", buf, len(buf) - 4, 0)
    path.write_bytes(buf)


GOAL = {
    "requirements": {
        "log": 2,
        "coal_ore": 2,
        "iron_ore": 2,
        "diamond_ore": 2,
        "gravel": 2,
        "obsidian": 2,
    },
    "proximity": {"log": [2.0, 4.0]},
    "natural_obsidian": True,
    "hazard_clearance": 0,
    "node_cap": 5_000,
}


def translated(cell, amount):
    return cell[0] + amount, cell[1], cell[2]


class WorldPlanningTests(unittest.TestCase):
    @staticmethod
    def seal_plan(*seal_groups):
        targets = []
        for index, seals in enumerate(seal_groups):
            target = index, 2, 0
            stance = index, 1, 1
            targets.append(MiningTargetPlan(
                target,
                stance,
                RoutePlan.surface((stance,)),
                RemovalProof(
                    target,
                    (target, *tuple(seals)),
                    (),
                    "known_faces_sealable_fluid_v1",
                    tuple(seals),
                ),
            ))
        cluster = ClusterPlan(
            targets[0].cell, len(targets), len(targets), (49,),
            tuple(targets))
        resource = ResourcePlan(
            "obsidian", ("obsidian",), len(targets), len(targets),
            len(targets), (cluster,), natural=True)
        goal = GoalSpec(resources=(ResourceRequirement(
            "obsidian", ("obsidian",), len(targets),
            sealable_fluids=("water",)),), natural_obsidian=True)
        return WorldPlan(
            "0" * 64, 1, 2, (0, 1, 0), goal, (), (resource,), ())

    @staticmethod
    def durability_plan():
        def resource(name, block, groups):
            targets = []
            for index, (mine_cells, target_cell) in enumerate(groups):
                stance = index, 1, len(targets)
                targets.append(MiningTargetPlan(
                    target_cell,
                    stance,
                    RoutePlan(
                        "tunnel", (stance,), (), (), tuple(mine_cells)),
                    RemovalProof(target_cell, (target_cell,), ()),
                ))
            cluster = ClusterPlan(
                targets[0].cell, len(targets), len(targets),
                (block,), tuple(targets))
            return ResourcePlan(
                name, (block,), len(targets), len(targets),
                len(targets), (cluster,))

        resources = (
            resource("stone", 1, (
                (((10, 0, 0), (11, 0, 0)), (12, 0, 0)),
                (((11, 0, 0), (13, 0, 0)), (14, 0, 0)),
            )),
            resource("iron", 15, (
                (((14, 0, 0), (15, 0, 0)), (16, 0, 0)),
            )),
            resource("gravel", 13, (
                (((17, 0, 0),), (18, 0, 0)),
            )),
        )
        goal = GoalSpec(resources=tuple(ResourceRequirement(
            item.name, item.blocks, item.minimum) for item in resources))
        return WorldPlan(
            "1" * 64, 3, 4, (0, 1, 0), goal, (), resources, ())

    def test_natural_gravel_drop_matches_engine_coordinate_hash(self):
        self.assertEqual(natural_harvest_drop(13, (4, 74, 4)), 13)
        self.assertEqual(natural_harvest_drop(13, (2, 75, 3)), 318)

    def test_natural_stone_drop_respects_variant_metadata(self):
        self.assertEqual(
            natural_harvest_drop(1, (0, 64, 0), block_meta=0), 4)
        for meta in range(1, 7):
            with self.subTest(meta=meta):
                self.assertEqual(
                    natural_harvest_drop(
                        1, (0, 64, 0), block_meta=meta),
                    1)

    def test_both_legacy_log_blocks_have_their_matching_item_drop(self):
        self.assertEqual(natural_harvest_drop(17, (0, 64, 0)), 17)
        self.assertEqual(natural_harvest_drop(162, (0, 64, 0)), 162)

    def test_resource_metadata_parses_strictly_and_round_trips(self):
        requirement = ResourceRequirement.parse({
            "name": "plain_stone",
            "blocks": ["stone"],
            "minimum": 3,
            "metadata": 0,
            "required_drop": "cobblestone",
        })

        self.assertEqual(requirement.metadata, (0,))
        self.assertEqual(requirement.to_dict()["metadata"], [0])
        with self.assertRaisesRegex(ValueError, "metadata"):
            ResourceRequirement(
                "bad", ("stone",), 1, metadata=(0, 16))

    def test_tunnel_clear_ray_rejects_protected_overhead_occluder(self):
        class World:
            def __init__(self, blocker=0):
                self.blocker = blocker

            @staticmethod
            def in_bounds(_cell):
                return True

            def block(self, cell):
                return self.blocker if cell == (0, 1, 0) else 0

        plan = ExcavationPlan(
            stands=((1, 0, 0), (0, 0, 0)),
            reserved_cells=((1, 0, 0), (1, 1, 0),
                            (0, 0, 0), (0, 1, 0)),
            support_cells=((1, -1, 0), (0, -1, 0)),
            mine_cells=((0, 0, 0),),
        )
        self.assertTrue(_excavation_clear_rays(World(), plan))
        self.assertFalse(_excavation_clear_rays(
            World(13), plan, protected=((0, 1, 0),)))

    def test_tunnel_rejects_gravel_that_would_collapse_into_body(self):
        class World:
            def __init__(self, overhead):
                self.overhead = overhead

            @staticmethod
            def in_bounds(_cell):
                return True

            def block(self, cell):
                if cell == (0, 2, 0):
                    return self.overhead
                if cell[1] < 0:
                    return 1
                return 0

        plan = ExcavationPlan(
            stands=((1, 0, 0), (0, 0, 0)),
            reserved_cells=((1, 0, 0), (1, 1, 0),
                            (0, 0, 0), (0, 1, 0)),
            support_cells=((1, -1, 0), (0, -1, 0)),
            mine_cells=((0, 0, 0), (0, 1, 0)),
        )
        self.assertTrue(_excavation_falling_blocks_stable(World(1), plan))
        self.assertFalse(_excavation_falling_blocks_stable(World(13), plan))

        class FallingClear(World):
            def block(self, cell):
                return 13 if cell == (0, 1, 0) else super().block(cell)

        self.assertFalse(_excavation_falling_blocks_stable(
            FallingClear(1), plan))

    def test_target_removal_rejects_overhead_falling_block(self):
        class World:
            def __init__(self, blocks):
                self.blocks = blocks

            @staticmethod
            def in_bounds(cell):
                return cell[1] <= 8

            def block(self, cell):
                return self.blocks.get(cell, 0)

        target = (0, 2, 0)
        self.assertTrue(_target_falling_blocks_stable(
            World({(0, 4, 0): 1}), target))
        self.assertFalse(_target_falling_blocks_stable(
            World({(0, 4, 0): 13}), target))
        self.assertFalse(_target_falling_blocks_stable(
            World({(0, 3, 0): 12}), target))

    def test_resource_planner_skips_target_below_falling_block(self):
        goal = {
            "requirements": [{
                "name": "logs", "blocks": ["log"], "count": 1,
                "routes": ["surface"], "max_drop_fall": 1,
            }],
            "hazard_clearance": 0,
            "node_cap": 256,
            "route_search_cap": 16,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "falling-target.bsnp"
            write_protected_clusters_snapshot(path, upper_block=13)
            with SnapshotWorldMap(path) as world:
                plan = WorldPlanBuilder(
                    world, goal, decision_seed=991).build()

        self.assertEqual(
            plan.resources[0].clusters[0].targets[0].cell,
            (5, 2, 0),
        )

    def test_drop_landing_must_stay_within_collection_stance(self):
        class World:
            def __init__(self, blocks):
                self.blocks = blocks

            @staticmethod
            def in_bounds(cell):
                return cell[1] >= 0

            def block(self, cell):
                return self.blocks.get(cell, 0)

        target = (0, 6, 0)
        stance = (1, 5, 0)

        self.assertTrue(_drop_landing_within_reach(
            World({(0, 5, 0): 1}), target, stance))
        self.assertFalse(_drop_landing_within_reach(
            World({(0, 5, 0): 1}), target, (0, 5, 2)))
        self.assertFalse(_drop_landing_within_reach(
            World({(0, 2, 0): 1}), target, stance))
        self.assertFalse(_drop_landing_within_reach(
            World({(0, 3, 0): 1}), target, (1, 4, 0), max_fall=1))
        self.assertFalse(_drop_landing_within_reach(
            World({(0, 5, 0): 10}), target, stance))
        self.assertTrue(_drop_landing_within_reach(
            World({(0, 4, 0): 1}), target, (1, 6, 0), max_fall=1))
        self.assertFalse(_drop_landing_within_reach(
            World({(0, 4, 0): 1, (0, 7, 0): 3}),
            target, (1, 6, 0), max_fall=1))
        self.assertTrue(_drop_fall_within(
            World({(0, 5, 0): 1}), target, 1))
        self.assertFalse(_drop_fall_within(
            World({(0, 3, 0): 1}), target, 1))

    def test_goal_spec_is_data_and_validates_natural_obsidian(self):
        goal = GoalSpec.parse(GOAL)
        self.assertEqual(goal.resources[0].name, "log")
        self.assertEqual(goal.resources[0].minimum, 2)
        self.assertEqual(goal.proximity[0].minimum, 2.0)
        self.assertTrue(goal.natural_obsidian)
        self.assertEqual(goal.to_dict()["requirements"][0]["blocks"],
                         ["log"])
        with self.assertRaisesRegex(ValueError, "obsidian"):
            GoalSpec.parse({
                "requirements": {"log": 1},
                "natural_obsidian": True,
            })

    def test_goal_mappings_reject_unknown_keys(self):
        cases = (
            (
                {"requirements": {"log": 1},
                 "hazzard_clearance": 1},
                "unknown world goal key: hazzard_clearance",
            ),
            (
                {"requirements": {"log": {
                    "count": 1, "prefer_expsed": True,
                }}},
                "unknown resource requirement key: prefer_expsed",
            ),
            (
                {"requirements": {"log": 1}, "proximity": {
                    "log": {"distance": [1, 4], "maximim": 4},
                }},
                "unknown proximity constraint key: maximim",
            ),
        )
        for value, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    GoalSpec.parse(value)

    def test_goal_boolean_fields_require_real_booleans(self):
        with self.assertRaisesRegex(
                TypeError, "resource prefer_exposed must be true or false"):
            GoalSpec.parse({"requirements": {"log": {
                "count": 1, "prefer_exposed": "false",
            }}})
        with self.assertRaisesRegex(
                TypeError, "goal natural_obsidian must be true or false"):
            GoalSpec.parse({
                "requirements": {"obsidian": 1},
                "natural_obsidian": "false",
            })

        goal = GoalSpec.parse({
            "requirements": {"log": {
                "count": 1, "prefer_exposed": False,
            }},
            "natural_obsidian": False,
        })
        self.assertFalse(goal.resources[0].prefer_exposed)
        self.assertFalse(goal.natural_obsidian)

    def test_plan_is_immutable_json_ready_and_provenanced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "world.bsnp"
            write_planning_snapshot(path)
            with SnapshotWorldMap(path) as world:
                plan = WorldPlanBuilder(
                    world, GOAL, decision_seed=991).build()
                self.assertEqual(plan.snapshot_sha256,
                                 world.snapshot_sha256)
                self.assertEqual(plan.world_seed, 77)
                self.assertEqual(plan.decision_seed, 991)
                self.assertEqual(plan.proximity[0].name, "log")
                self.assertGreaterEqual(plan.proximity[0].distance, 2.0)
                self.assertLessEqual(plan.proximity[0].distance, 4.0)
                payload = json.loads(plan.to_json(sort_keys=True))
            self.assertEqual(payload["schema"], "bedrock.world_plan.v1")
            self.assertEqual(payload["world_seed"], 77)
            self.assertTrue(payload["safe_cells"])
            self.assertTrue(all(
                item["selected_count"] >= item["minimum"]
                for item in payload["resources"]))
            route_kinds = {
                cluster["route"]["kind"]
                for item in payload["resources"]
                for cluster in item["clusters"]
            }
            self.assertEqual(route_kinds, {"surface", "tunnel"})
            self.assertTrue(all(
                not (set(map(tuple, cluster["route"]["mine_cells"]))
                     & set(map(tuple, cluster["route"]["support_cells"])))
                for item in payload["resources"]
                for cluster in item["clusters"]
            ))
            obsidian = next(item for item in payload["resources"]
                            if item["name"] == "obsidian")
            self.assertTrue(obsidian["natural"])
            self.assertTrue(all(
                cluster["source"] == "initial_snapshot"
                for cluster in obsidian["clusters"]))
            cursor = plan.start
            for resource in plan.resources:
                for cluster in resource.clusters:
                    self.assertEqual(cluster.selected_count,
                                     len(cluster.targets))
                    for target in cluster.targets:
                        self.assertEqual(target.route.stands[0], cursor)
                        self.assertEqual(target.route.stands[-1],
                                         target.mining_stance)
                        self.assertEqual(target.removal.target, target.cell)
                        if target.route.kind == "tunnel":
                            self.assertLessEqual(
                                abs(target.cell[0]
                                    - target.mining_stance[0])
                                + abs(target.cell[2]
                                      - target.mining_stance[2]),
                                1,
                            )
                        self.assertNotIn(target.cell, (
                            target.mining_stance,
                            (target.mining_stance[0],
                             target.mining_stance[1] + 1,
                             target.mining_stance[2]),
                        ))
                        self.assertTrue(all(target.cell not in (
                            stand, (stand[0], stand[1] + 1, stand[2]))
                            for stand in target.route.stands))
                        self.assertNotIn(
                            target.cell,
                            target.removal.protected_support_cells)
                        cursor = target.mining_stance
            with self.assertRaises(AttributeError):
                plan.world_seed = 9

    def test_fluid_seal_ledger_counts_unique_cells_and_preserves_reserve(self):
        shared = (8, 2, 0)
        plan = self.seal_plan(
            ((7, 2, 0), shared),
            (shared, (9, 2, 0)),
        )

        self.assertEqual(plan.fluid_seal_cells, (
            (7, 2, 0), shared, (9, 2, 0)))
        self.assertEqual(plan.resources[0].fluid_seal_cells,
                         plan.fluid_seal_cells)
        self.assertEqual(
            plan.resources[0].clusters[0].fluid_seal_cells,
            plan.fluid_seal_cells)
        payload = plan.to_dict()
        self.assertEqual(payload["fluid_seal_count"], 3)
        self.assertEqual(
            payload["resources"][0]["fluid_seal_count"], 3)
        self.assertEqual(
            payload["resources"][0]["clusters"][0][
                "fluid_seal_count"], 3)

        ledger = require_fluid_seal_budget(
            plan, available=6, reserve=3, resource="obsidian")
        self.assertTrue(ledger.within_budget)
        self.assertEqual(ledger.required, 3)
        self.assertEqual(ledger.usable, 3)
        self.assertEqual(ledger.shortfall, 0)

    def test_fluid_seal_budget_rejects_aggregate_shortfall(self):
        plan = self.seal_plan(
            ((7, 2, 0), (8, 2, 0)),
            ((9, 2, 0),),
        )

        ledger = fluid_seal_ledger(plan, available=5, reserve=3)
        self.assertFalse(ledger.within_budget)
        self.assertEqual(ledger.required, 3)
        self.assertEqual(ledger.usable, 2)
        self.assertEqual(ledger.shortfall, 1)
        with self.assertRaises(WorldPlanningRejected) as raised:
            require_fluid_seal_budget(
                plan, available=5, reserve=3, resource="obsidian")
        self.assertEqual(
            raised.exception.code, "world_plan_fluid_seal_budget")
        self.assertEqual(raised.exception.stage, "resources")
        self.assertEqual(raised.exception.resource, "obsidian")
        self.assertEqual(raised.exception.details, ledger.to_dict())

    def test_fluid_seal_budget_rejects_invalid_inventory_accounting(self):
        plan = self.seal_plan(())
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            fluid_seal_ledger(plan, available=True)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            fluid_seal_ledger(plan, available=2, reserve=3)

    def test_tool_durability_ledgers_preserve_order_and_deduplicate_cells(self):
        plan = self.durability_plan()

        ledgers = tool_durability_ledgers(
            plan,
            capacities={"wooden_pickaxe": 3, "stone_pickaxe": 2},
            copies={"wooden_pickaxe": 2, "stone_pickaxe": 2},
            route_tools={
                "stone": "wooden_pickaxe",
                "iron": "stone_pickaxe",
                "gravel": "stone_pickaxe",
            },
            target_tools={
                "stone": "wooden_pickaxe",
                "iron": "stone_pickaxe",
                "gravel": None,
            },
            extra_breaks={
                "wooden_pickaxe": 1,
                "stone_pickaxe": 1,
            },
        )

        wood, stone = ledgers
        self.assertEqual(wood.planned_cells, tuple(
            (x, 0, 0) for x in range(10, 15)))
        self.assertEqual(wood.planned_breaks, 5)
        self.assertEqual(wood.required, 6)
        self.assertEqual(wood.available, 6)
        self.assertTrue(wood.within_budget)
        # Cell 14 was already charged to the earlier wooden-pick target.
        # The hand-mined gravel target 18 consumes no tracked durability.
        self.assertEqual(stone.planned_cells, (
            (15, 0, 0), (16, 0, 0), (17, 0, 0)))
        self.assertEqual(stone.required, 4)
        self.assertEqual(stone.available, 4)
        self.assertTrue(stone.within_budget)

    def test_tool_durability_budget_rejects_only_failing_tools(self):
        plan = self.durability_plan()

        with self.assertRaises(WorldPlanningRejected) as raised:
            require_tool_durability_budget(
                plan,
                capacities={"wooden_pickaxe": 4, "stone_pickaxe": 4},
                route_tools={
                    "stone": "wooden_pickaxe",
                    "iron": "stone_pickaxe",
                },
                target_tools={
                    "stone": "wooden_pickaxe",
                    "iron": "stone_pickaxe",
                },
            )

        error = raised.exception
        self.assertEqual(
            error.code, "world_plan_tool_durability_budget")
        self.assertEqual(error.stage, "resources")
        self.assertEqual(len(error.details["tools"]), 1)
        failure = error.details["tools"][0]
        self.assertEqual(failure["tool"], "wooden_pickaxe")
        self.assertEqual(failure["required"], 5)
        self.assertEqual(failure["available"], 4)
        self.assertEqual(failure["shortfall"], 1)

    def test_tool_durability_ledger_requires_caller_supplied_capacities(self):
        plan = self.durability_plan()
        with self.assertRaisesRegex(ValueError, "capacity missing"):
            tool_durability_ledgers(
                plan,
                capacities={"wooden_pickaxe": 59},
                route_tools={"iron": "stone_pickaxe"},
            )
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            tool_durability_ledgers(
                plan,
                capacities={"wooden_pickaxe": 59},
                route_tools={"stone": "wooden_pickaxe"},
                copies={"wooden_pickaxe": True},
            )

    def test_two_translated_worlds_produce_translated_safe_plans(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / "one.bsnp",
                     Path(directory) / "two.bsnp"]
            write_planning_snapshot(paths[0], shift=0, seed=17)
            write_planning_snapshot(paths[1], shift=40, seed=29)
            plans = []
            for path in paths:
                with SnapshotWorldMap(path) as world:
                    plans.append(WorldPlanBuilder(
                        world, GOAL, decision_seed=1234).build())

        left, right = plans
        self.assertEqual(translated(left.start, 40), right.start)
        self.assertNotEqual(left.world_seed, right.world_seed)
        self.assertNotEqual(left.snapshot_sha256, right.snapshot_sha256)
        self.assertEqual(
            translated(left.proximity[0].nearest_cell, 40),
            right.proximity[0].nearest_cell)
        self.assertEqual(len(left.resources), len(right.resources))
        for one, two in zip(left.resources, right.resources):
            self.assertEqual(one.name, two.name)
            self.assertEqual(one.selected_count, two.selected_count)
            self.assertEqual(len(one.clusters), len(two.clusters))
            for first, second in zip(one.clusters, two.clusters):
                self.assertEqual(translated(first.anchor, 40), second.anchor)
                self.assertEqual(
                    translated(first.mining_stance, 40),
                    second.mining_stance)
                self.assertEqual(
                    tuple(translated(cell, 40)
                          for cell in first.route.stands),
                    second.route.stands)
                self.assertEqual(len(first.targets), len(second.targets))
                for target_one, target_two in zip(
                        first.targets, second.targets):
                    self.assertEqual(
                        translated(target_one.cell, 40), target_two.cell)
                    self.assertEqual(
                        translated(target_one.mining_stance, 40),
                        target_two.mining_stance)
                    self.assertEqual(
                        tuple(translated(cell, 40)
                              for cell in target_one.route.stands),
                        target_two.route.stands)

    def test_same_world_supports_repeatable_decision_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "world.bsnp"
            write_planning_snapshot(path)
            with SnapshotWorldMap(path) as world:
                first = WorldPlanBuilder(
                    world, GOAL, decision_seed=101).build()
                again = WorldPlanBuilder(
                    world, GOAL, decision_seed=101).build()
                another = WorldPlanBuilder(
                    world, GOAL, decision_seed=202).build()
        self.assertEqual(first.to_dict(), again.to_dict())
        self.assertEqual(first.snapshot_sha256, another.snapshot_sha256)
        self.assertNotEqual(first.decision_seed, another.decision_seed)

    def test_rejections_have_codes_and_preflight_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.bsnp"
            write_planning_snapshot(path, include_diamond=False)
            with SnapshotWorldMap(path) as world:
                with self.assertRaises(WorldPlanningRejected) as raised:
                    WorldPlanBuilder(
                        world, GOAL, decision_seed=3).build()
        error = raised.exception
        self.assertEqual(error.code, "world_plan_resource_missing")
        self.assertEqual(error.stage, "resources")
        self.assertEqual(error.resource, "diamond_ore")
        self.assertEqual(error.to_dict()["details"]["available"], 0)

    def test_proximity_rejection_is_categorized(self):
        goal = dict(GOAL)
        goal["proximity"] = {"log": [8, 12]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "too-close.bsnp"
            write_planning_snapshot(path)
            with SnapshotWorldMap(path) as world:
                with self.assertRaises(WorldPlanningRejected) as raised:
                    WorldPlanBuilder(
                        world, goal, decision_seed=7).build()
        self.assertEqual(
            raised.exception.code, "world_plan_proximity_out_of_range")
        self.assertEqual(raised.exception.stage, "proximity")

    def test_aggregate_route_search_budget_rejects_hostile_world(self):
        goal = dict(GOAL)
        goal["route_search_cap"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bounded.bsnp"
            write_planning_snapshot(path)
            with SnapshotWorldMap(path) as world:
                with self.assertRaises(WorldPlanningRejected) as raised:
                    WorldPlanBuilder(
                        world, goal, decision_seed=3).build()
        self.assertEqual(raised.exception.code,
                         "world_plan_search_budget")
        self.assertEqual(raised.exception.details,
                         {"attempted": 1, "limit": 1})
        self.assertEqual(raised.exception.stage, "routes")

    def test_long_stone_quarry_preserves_budget_for_all_drop_safe_legs(self):
        goal = {
            "requirements": [{
                "name": "stone",
                "blocks": ["stone"],
                "count": 24,
                "routes": ["surface"],
                "max_drop_fall": 1,
            }],
            "hazard_clearance": 0,
            "node_cap": 256,
            "route_search_cap": 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quarry.bsnp"
            write_quarry_snapshot(path)
            with SnapshotWorldMap(path) as world:
                plan = WorldPlanBuilder(
                    world, goal, decision_seed=991).build()

        stone = plan.resources[0]
        self.assertEqual(stone.selected_count, 24)
        self.assertEqual(
            [target.cell for target in stone.clusters[0].targets],
            [(x, 2, 1) for x in range(2, 26)],
        )

    def test_abundant_resource_planning_skips_component_flood_fill(self):
        goal = {
            "requirements": [{
                "name": "stone",
                "blocks": ["stone"],
                "count": 24,
                "routes": ["surface"],
                "max_drop_fall": 1,
            }],
            "hazard_clearance": 0,
            "node_cap": 256,
            "route_search_cap": 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "abundant-quarry.bsnp"
            write_quarry_snapshot(path, count=128)
            with SnapshotWorldMap(path) as world:
                with mock.patch.object(
                        SnapshotWorldMap, "resource_clusters",
                        side_effect=AssertionError(
                            "abundant fast path flooded components")):
                    first = WorldPlanBuilder(
                        world, goal, decision_seed=991,
                        cluster_sample_limit=64).build()
                    again = WorldPlanBuilder(
                        world, goal, decision_seed=991,
                        cluster_sample_limit=64).build()
                targets = tuple(
                    target.cell
                    for cluster in first.resources[0].clusters
                    for target in cluster.targets)
                self.assertTrue(all(
                    world.block_id(*target) == 1 for target in targets))

        self.assertEqual(first.to_dict(), again.to_dict())
        self.assertEqual(first.resources[0].available_count, 128)
        self.assertEqual(first.resources[0].selected_count, 24)
        self.assertEqual(
            {cluster.source for cluster in first.resources[0].clusters},
            {"initial_snapshot_target_pool"},
        )

    def test_metadata_drop_filter_keeps_abundant_stone_on_fast_path(self):
        goal = {
            "requirements": [{
                "name": "stone",
                "blocks": ["stone"],
                "count": 24,
                "routes": ["surface"],
                "max_drop_fall": 1,
                "metadata": [0],
                "required_drop": "cobblestone",
            }],
            "hazard_clearance": 0,
            "node_cap": 256,
            "route_search_cap": 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed-stone-quarry.bsnp"
            write_quarry_snapshot(path, count=152, variant_count=24)
            with SnapshotWorldMap(path) as world:
                with mock.patch.object(
                        SnapshotWorldMap, "resource_clusters",
                        side_effect=AssertionError(
                            "metadata fast path flooded components")):
                    plan = WorldPlanBuilder(
                        world, goal, decision_seed=991,
                        cluster_sample_limit=64).build()
                targets = tuple(
                    target.cell
                    for cluster in plan.resources[0].clusters
                    for target in cluster.targets)
                self.assertTrue(all(
                    world.block(*target).meta == 0 for target in targets))

        self.assertEqual(plan.resources[0].available_count, 128)
        self.assertEqual(plan.resources[0].selected_count, 24)
        self.assertEqual(
            {cluster.source for cluster in plan.resources[0].clusters},
            {"initial_snapshot_target_pool"},
        )

    def test_fast_and_exact_paths_certify_equivalent_target_validity(self):
        goal = {
            "requirements": [{
                "name": "stone",
                "blocks": ["stone"],
                "count": 24,
                "routes": ["surface"],
                "max_drop_fall": 1,
            }],
            "hazard_clearance": 0,
            "node_cap": 256,
            "route_search_cap": 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "equivalent-quarry.bsnp"
            write_quarry_snapshot(path, count=128)
            with SnapshotWorldMap(path) as world:
                fast = WorldPlanBuilder(
                    world, goal, decision_seed=991,
                    cluster_sample_limit=64).build()
                exact = WorldPlanBuilder(
                    world, goal, decision_seed=991,
                    cluster_sample_limit=128).build()
                fast_targets = tuple(
                    target.cell for cluster in fast.resources[0].clusters
                    for target in cluster.targets)
                exact_targets = tuple(
                    target.cell for cluster in exact.resources[0].clusters
                    for target in cluster.targets)
                self.assertTrue(all(
                    world.block_id(*cell) == 1
                    for cell in fast_targets + exact_targets))

        self.assertEqual(len(fast_targets), len(exact_targets))
        self.assertEqual(fast.resources[0].available_count,
                         exact.resources[0].available_count)

    def test_protected_cell_does_not_discard_usable_cluster_prefix(self):
        goal = {
            "requirements": [{
                "name": "logs",
                "blocks": ["log"],
                "count": 2,
                "routes": ["surface"],
                "max_drop_fall": 1,
            }],
            "hazard_clearance": 0,
            "node_cap": 256,
            "route_search_cap": 16,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protected-clusters.bsnp"
            write_protected_clusters_snapshot(path, horizontal=True)
            with SnapshotWorldMap(path) as world:
                plan = WorldPlanBuilder(
                    world, goal, decision_seed=991,
                    protected=((3, 2, 0),),
                ).build()

        logs = plan.resources[0]
        self.assertEqual(logs.selected_count, 2)
        self.assertEqual(
            [target.cell for cluster in logs.clusters
             for target in cluster.targets],
            [(2, 2, 0), (5, 2, 0)],
        )


if __name__ == "__main__":
    unittest.main()
