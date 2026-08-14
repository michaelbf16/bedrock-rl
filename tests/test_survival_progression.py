import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bedrock_rl.sdg.policy import (
    MovementProfile, PolicyVeto, SurvivalBehaviors,
)
from bedrock_rl.sdg.policy.planning import RoutePlan
from bedrock_rl.sdg.policy.progression import (
    BeginLightingOperation,
    BucketCastMineOperation,
    CraftOperation,
    HotbarPolicy,
    NetherPortalOperation,
    PortalSettings,
    RepeatDropOperation,
    ResourceStage,
    SmeltOperation,
    SurvivalProgressionPolicy,
    SurvivalProgressionSpec,
    WorkstationOperation,
    WorkstationSite,
    _ReturnBreadcrumb,
    _portal_protected_cells,
    _workstation_sites,
)
from bedrock_rl.sdg.generation import Case, CaseRejected
from bedrock_rl.env.journal import FLUID_BODY


class Event(dict):
    def __init__(self, name, **values):
        super().__init__(values)
        self.name = name


class Env:
    def __init__(self, *, x=0.5, y=64.0, z=0.5, health=20.0,
                 journal=(), hotbar_ids=None, hotbar_counts=None):
        self.obs = {
            "x": x, "y": y, "z": z, "yaw": 0.0, "pitch": 0.0,
            "health": health, "dead": False, "dimension": 0,
            "container": 0, "blocks": [], "hotbar_sel": 0,
            "hotbar_ids": list(hotbar_ids or [0] * 9),
            "hotbar_counts": list(hotbar_counts or [0] * 9),
            "hotbar_meta": [0] * 9,
        }
        self.journal = list(journal)

    def rel_angles_to(self, x, y, z):
        return 0.0, 0.0, ((x - self.obs["x"]) ** 2
                          + (y - self.obs["y"]) ** 2
                          + (z - self.obs["z"]) ** 2) ** 0.5


class ReturnBreadcrumbTests(unittest.TestCase):
    def test_revisit_splices_abandoned_loop(self):
        breadcrumb = _ReturnBreadcrumb((0, 64, 0))
        breadcrumb.extend((
            (0, 64, 0), (1, 64, 0), (2, 64, 0),
        ))
        breadcrumb.extend((
            (2, 64, 0), (1, 64, 0), (1, 64, 1),
        ))

        self.assertEqual(breadcrumb.stands, (
            (0, 64, 0), (1, 64, 0), (1, 64, 1),
        ))
        self.assertNotIn((2, 64, 0), breadcrumb.passage_cells)
        self.assertNotIn((2, 63, 0), breadcrumb.support_cells)

    def test_vertical_headroom_is_recomputed_after_loop_splice(self):
        breadcrumb = _ReturnBreadcrumb((0, 64, 0))
        breadcrumb.extend((
            (0, 64, 0), (1, 63, 0), (2, 63, 0),
        ))
        self.assertIn((1, 65, 0), breadcrumb.passage_cells)
        breadcrumb.extend((
            (2, 63, 0), (1, 63, 0), (1, 64, 1),
        ))

        self.assertEqual(breadcrumb.stands, (
            (0, 64, 0), (1, 63, 0), (1, 64, 1),
        ))
        self.assertNotIn((2, 65, 0), breadcrumb.passage_cells)
        self.assertTrue(
            breadcrumb.passage_cells.isdisjoint(
                breadcrumb.support_cells))

    def test_disconnected_or_noncardinal_leg_breaks_certificate(self):
        for leg in (
            ((9, 64, 9), (10, 64, 9)),
            ((0, 64, 0), (1, 64, 1)),
        ):
            with self.subTest(leg=leg):
                breadcrumb = _ReturnBreadcrumb((0, 64, 0))
                breadcrumb.extend(leg)
                self.assertTrue(breadcrumb.broken)
                self.assertEqual(breadcrumb.stands, ())


class SurvivalProgressionSpecTests(unittest.TestCase):
    def test_named_portal_preset_is_explicit_and_omission_stays_compatible(self):
        named = SurvivalProgressionSpec.parse("nether_portal")
        implicit = SurvivalProgressionSpec.parse()

        self.assertEqual(named, implicit)
        self.assertEqual(named.stages[-1].name, "diamonds")
        with self.assertRaisesRegex(ValueError, "unknown.*preset"):
            SurvivalProgressionSpec.parse("not_a_preset")

    def test_portal_turn_grounding_excludes_clairvoyant_stages(self):
        policy = SurvivalProgressionPolicy(spec="nether_portal")
        policy.stage = "mine:logs"
        self.assertTrue(policy.turn_grounding()["eligible"])
        policy.stage = "mine:coal"
        self.assertTrue(policy.turn_grounding()["eligible"])
        for stage in ("mine:iron", "mine:diamonds", "fluid_cast",
                      "portal_frame"):
            with self.subTest(stage=stage):
                policy.stage = stage
                self.assertFalse(policy.turn_grounding()["eligible"])

    def test_portal_preset_is_a_data_only_exact_recipe_graph(self):
        spec = SurvivalProgressionSpec.nether_portal()

        self.assertEqual(
            [(stage.name, stage.count, stage.tool) for stage in spec.stages],
            [
                ("logs", 12, None),
                ("stone", 27, "wooden_pickaxe"),
                ("coal", 18, "stone_pickaxe"),
                ("gravel", 1, None),
                ("iron", 10, "stone_pickaxe"),
                ("diamonds", 3, "iron_pickaxe"),
            ])
        self.assertFalse(spec.goal.natural_obsidian)
        self.assertFalse(any(stage.natural_only for stage in spec.stages))
        self.assertEqual(spec.hotbar.swap_priority, (
            "dirt", "planks", "stick", "crafting_table",
            "wooden_pickaxe", "stone_pickaxe", "cobblestone", "gravel",
            "torch"))
        self.assertEqual(spec.goal.hazard_clearance, 1)
        self.assertEqual(spec.route_search_cap, 512)
        stone = next(stage for stage in spec.stages
                     if stage.name == "stone")
        self.assertEqual(stone.required_drop, "cobblestone")
        self.assertEqual(stone.metadata, (0,))
        gravel = next(stage for stage in spec.stages
                      if stage.name == "gravel")
        self.assertIsNone(gravel.tool)
        self.assertEqual(gravel.tunnel_tool, "stone_pickaxe")
        self.assertEqual(gravel.tunnel_ticks, 70)
        self.assertEqual(gravel.drop_item, "flint")
        self.assertEqual(gravel.required_drop, "flint")
        self.assertFalse(gravel.after)
        cast = next(
            operation for operation in spec.stages[-1].after
            if isinstance(operation, BucketCastMineOperation))
        self.assertIsInstance(spec.stages[-1].after[-1],
                              NetherPortalOperation)
        self.assertEqual(
            (cast.item, cast.minimum, cast.tool, cast.mining_ticks),
            ("obsidian", 10, "diamond_pickaxe", 188))
        self.assertEqual(spec.stages[0].mining_ticks, 65)
        crafts = {
            operation.item: operation.minimum
            for stage in spec.stages for operation in stage.after
            if isinstance(operation, CraftOperation)
        }
        self.assertEqual(crafts["planks"], 48)
        self.assertEqual(crafts["stick"], 32)
        self.assertEqual(crafts["wooden_pickaxe"], 2)
        self.assertEqual(crafts["stone_pickaxe"], 3)
        self.assertEqual(crafts["iron_pickaxe"], 2)
        self.assertEqual(crafts["torch"], 64)
        self.assertEqual(crafts["bucket"], 1)
        iron = next(stage for stage in spec.stages
                    if stage.name == "iron")
        smelt = next(operation for operation in iron.after
                     if isinstance(operation, SmeltOperation))
        self.assertEqual(
            (smelt.count, smelt.fuel_item, smelt.fuel_count),
            (10, "coal", 2))
        self.assertEqual(
            [stage.mining_ticks for stage in spec.stages[1:]],
            [28, 28, 26, 28, 28])
        payload = spec.to_dict()
        self.assertEqual(payload["portal"]["scaffold_item"], "cobblestone")
        self.assertEqual(payload["portal"]["method"], "bucket_cast_mined")
        self.assertEqual(
            payload["portal"]["strategy"], "completion_chamber")
        self.assertEqual(payload["portal"]["fluid_search_radius"], 160)
        self.assertEqual(payload["portal"]["source_meta"], 0)
        self.assertTrue(payload["portal"]["unique_casting_sources"])
        self.assertTrue(payload["portal"]["recover_coolant"])
        self.assertFalse(payload["veto_fluids"])
        self.assertEqual(
            SurvivalProgressionSpec.parse(payload).to_dict(), payload)
        self.assertEqual(
            [requirement.to_dict()
             for requirement in spec.fluid_source_requirements],
            [{
                "role": "coolant", "fluid": "water", "count": 1,
                "metadata": 0, "source_only": True, "unique": True,
                "filled_bucket_item": "water_bucket", "recover": True,
            }, {
                "role": "casting", "fluid": "lava", "count": 10,
                "metadata": 0, "source_only": True, "unique": True,
                "filled_bucket_item": "lava_bucket", "recover": False,
            }])
        # Geometry belongs to the snapshot plans, never this reusable spec.
        self.assertNotIn("seed", repr(payload).lower())
        self.assertNotIn("coordinate", repr(payload).lower())

    def test_bucket_cast_must_be_final_live_moving_operation(self):
        cast = BucketCastMineOperation(
            "obsidian", 1, "diamond_pickaxe")
        portal = PortalSettings(
            enabled=True, method="bucket_cast_mined")
        cases = (
            (
                ResourceStage(
                    "first", ("stone",), 1, "cobblestone",
                    after=(cast,)),
                ResourceStage(
                    "later", ("log",), 1, "log"),
            ),
            (
                ResourceStage(
                    "only", ("stone",), 1, "cobblestone",
                    after=(cast, WorkstationOperation(
                        "late", "crafting_table", "crafting_table",
                        "crafting_table"))),
            ),
        )
        for stages in cases:
            with self.subTest(stages=tuple(stage.name for stage in stages)):
                with self.assertRaisesRegex(
                        ValueError, "bucket_cast_mine"):
                    SurvivalProgressionSpec(stages=stages, portal=portal)

    def test_legacy_enabled_portal_is_upgraded_to_terminal_operation(self):
        spec = SurvivalProgressionSpec(
            stages=(ResourceStage(
                "stone", ("stone",), 1, "cobblestone"),),
            portal=PortalSettings(enabled=True))

        self.assertIsInstance(spec.stages[-1].after[-1],
                              NetherPortalOperation)
        self.assertEqual(
            spec.to_dict()["stages"][-1]["after"][-1],
            {"type": "nether_portal"})

    def test_terminal_portal_operation_owns_the_endgame_dispatch(self):
        policy = SurvivalProgressionPolicy(spec="nether_portal")
        policy.stage = "milestone:diamonds"
        policy._plan_portal_return = Mock()
        policy._portal_executor = SimpleNamespace(
            actions=Mock(return_value=[{"action": "wait", "ticks": 2}]))
        env = object()

        result = policy._operation_actions(env, NetherPortalOperation())

        policy._plan_portal_return.assert_called_once_with(env)
        policy._portal_executor.actions.assert_called_once_with(env)
        self.assertEqual(result, [{"action": "wait", "ticks": 2}])

    def test_custom_progression_round_trips_without_subclassing(self):
        raw = {
            "stages": [{
                "name": "wood",
                "blocks": ["log"],
                "count": 2,
                "drop": "log",
                "route_kinds": ["surface"],
                "after": [
                    {"type": "craft", "item": "planks",
                     "minimum": 8, "grid": 2},
                    {"type": "workstation", "key": "bench",
                     "item": "crafting_table"},
                    {"type": "craft", "item": "wooden_pickaxe",
                     "minimum": 1, "grid": 3, "table": "bench"},
                ],
            }],
            "portal": False,
            "hotbar": {"swap_priority": ["sand", "dirt"]},
            "veto_fluids": False,
        }

        spec = SurvivalProgressionSpec.parse(raw)

        self.assertFalse(spec.portal.enabled)
        self.assertFalse(spec.veto_fluids)
        self.assertEqual(spec.hotbar, HotbarPolicy(("sand", "dirt")))
        self.assertEqual(spec.stages[0].after[1].key, "bench")
        payload = spec.to_dict()
        self.assertEqual(payload["stages"][0]["count"], 2)
        self.assertEqual(
            SurvivalProgressionSpec.parse(payload).to_dict(), payload)

    def test_inline_specs_default_portal_off_and_normalize_direct_booleans(self):
        raw = {"stages": [{
            "name": "wood", "blocks": ["log"], "count": 1,
            "drop": "log",
        }]}

        self.assertFalse(SurvivalProgressionSpec.parse(raw).portal.enabled)
        self.assertFalse(SurvivalProgressionSpec(
            stages=(ResourceStage.parse(raw["stages"][0]),),
            portal=False).portal.enabled)
        self.assertTrue(SurvivalProgressionSpec(
            stages=(ResourceStage.parse(raw["stages"][0]),),
            portal=True).portal.enabled)

    def test_progression_parsers_reject_typos_and_string_booleans(self):
        stage = {
            "name": "wood", "blocks": ["log"], "count": 1,
            "drop": "log",
        }
        with self.assertRaisesRegex(ValueError, "unknown resource stage key"):
            ResourceStage.parse({**stage, "prefer_expsed": True})
        with self.assertRaisesRegex(TypeError, "prefer_exposed.*true or false"):
            ResourceStage.parse({**stage, "prefer_exposed": "false"})
        with self.assertRaisesRegex(ValueError, "unknown craft operation key"):
            ResourceStage.parse({**stage, "after": [{
                "type": "craft", "item": "planks", "grid": 2,
                "minimun": 4,
            }]})
        with self.assertRaisesRegex(TypeError, "veto_fluids.*true or false"):
            SurvivalProgressionSpec.parse({
                "stages": [stage], "veto_fluids": "false"})
        with self.assertRaisesRegex(TypeError, "portal enabled.*true or false"):
            PortalSettings.parse({"enabled": "false"})
        with self.assertRaisesRegex(TypeError, "natural_only.*true or false"):
            ResourceStage.parse({**stage, "natural_only": "false"})
        with self.assertRaisesRegex(ValueError, "unknown hotbar policy key"):
            SurvivalProgressionSpec.parse({
                "stages": [stage], "hotbar": {"swap_priorities": []}})

    def test_workstations_must_be_declared_once_before_use(self):
        with self.assertRaisesRegex(ValueError, "before it is declared"):
            SurvivalProgressionSpec(stages=(ResourceStage.parse({
                "name": "wood", "count": 1, "drop": "log",
                "after": [{"type": "craft", "item": "wooden_pickaxe",
                           "grid": 3, "table": "later"},
                          {"type": "workstation", "key": "later",
                           "item": "crafting_table"}],
            }),))
        with self.assertRaisesRegex(ValueError, "duplicated"):
            SurvivalProgressionSpec(stages=(ResourceStage.parse({
                "name": "wood", "count": 1, "drop": "log",
                "after": [
                    {"type": "workstation", "key": "bench",
                     "item": "crafting_table"},
                    {"type": "workstation", "key": "bench",
                     "item": "crafting_table"},
                ],
            }),))

    def test_portal_runtime_fields_parse_and_round_trip(self):
        settings = PortalSettings.parse({
            "enabled": True,
            "entry_wait_ticks": 83,
            "return_tool": "iron_pickaxe",
            "return_ticks": 37,
            "active_block": "netherrack",
            "igniter_item": "torch",
        })

        self.assertEqual(settings.entry_wait_ticks, 83)
        self.assertEqual(settings.return_tool, "iron_pickaxe")
        self.assertEqual(settings.return_ticks, 37)
        self.assertEqual(settings.active_block, "netherrack")
        self.assertEqual(settings.igniter_item, "torch")
        self.assertEqual(PortalSettings.parse(settings.to_dict()), settings)

    def test_bucket_cast_portal_fields_parse_strictly_and_round_trip(self):
        settings = PortalSettings.parse({
            "enabled": True,
            "method": "bucket_cast_mined",
            "bucket_item": "iron_bucket",
            "casting_fluid": "hot_fluid",
            "coolant_fluid": "cold_fluid",
            "cast_block": "reaction_block",
            "casting_backing_item": "mold_block",
            "source_meta": 2,
            "fluid_search_radius": 96,
            "casting_wait_ticks": 3,
            "unique_casting_sources": False,
            "recover_coolant": False,
        })

        self.assertEqual(settings.method, "bucket_cast_mined")
        self.assertEqual(settings.bucket_item, "iron_bucket")
        self.assertEqual(settings.cast_block, "reaction_block")
        self.assertEqual(settings.fluid_search_radius, 96)
        self.assertEqual(PortalSettings.parse(settings.to_dict()), settings)
        with self.assertRaisesRegex(ValueError, "portal method"):
            PortalSettings(method="guess")
        with self.assertRaisesRegex(TypeError, "unique_casting_sources"):
            PortalSettings.parse({"unique_casting_sources": "false"})

    def test_bucket_cast_operation_requires_declared_method_and_materials(self):
        payload = SurvivalProgressionSpec.nether_portal().to_dict()
        payload["portal"]["method"] = "placed_blocks"
        with self.assertRaisesRegex(ValueError, "needs an enabled"):
            SurvivalProgressionSpec.parse(payload)

        payload = SurvivalProgressionSpec.nether_portal().to_dict()
        payload["stages"][-1]["after"] = [
            operation for operation in payload["stages"][-1]["after"]
            if operation["type"] != "bucket_cast_mine"]
        with self.assertRaisesRegex(ValueError, "exactly one"):
            SurvivalProgressionSpec.parse(payload)

    def test_portal_entry_wait_defaults_to_client_visible_boundary(self):
        # The pinned runtime changes dimension after its 82nd portal-contact
        # tick.  Waiting 80 relied on movement/frame ticks outside the policy
        # action and could require a second bounded entry attempt.
        self.assertEqual(PortalSettings(enabled=True).entry_wait_ticks, 82)
        with self.assertRaisesRegex(ValueError, "entry_wait_ticks"):
            PortalSettings(enabled=True, entry_wait_ticks=0)

    def test_portal_strategy_is_explicit_and_validated(self):
        settings = PortalSettings.parse({
            "enabled": True,
            "strategy": "completion_chamber",
        })

        self.assertEqual(settings.strategy, "completion_chamber")
        self.assertEqual(PortalSettings.parse(settings.to_dict()), settings)
        with self.assertRaisesRegex(ValueError, "portal strategy"):
            PortalSettings(enabled=True, strategy="lucky_natural_pad")


class SurvivalProgressionRuntimeTests(unittest.TestCase):
    def test_actions_does_not_clobber_an_active_portal_stage(self):
        policy = SurvivalProgressionPolicy.__new__(SurvivalProgressionPolicy)
        stage = SimpleNamespace(name="final", after=[object()])
        policy.spec = SimpleNamespace(stages=[stage])
        policy.stage = "return_to_surface"
        policy._stage_index = 0
        policy._operation_index = 0
        policy._executor = SimpleNamespace(done=True)
        policy._guard = lambda _env: None
        seen = []
        policy._operation_actions = lambda _env, _operation: (
            seen.append(policy.stage) or [{"action": "wait", "ticks": 1}])

        result = policy.actions(
            None, SimpleNamespace(env=SimpleNamespace()), None, 0)

        self.assertEqual(result, [{"action": "wait", "ticks": 1}])
        self.assertEqual(seen, ["return_to_surface"])

    def policy(self):
        spec = SurvivalProgressionSpec(stages=(ResourceStage(
            "wood", ("log",), 1, "log", route_kinds=("surface",)),),
            portal=False)
        policy = SurvivalProgressionPolicy(spec=spec)
        # Runtime-unit tests start after snapshot planning. The shared
        # controller reset is still real, including behavior variance.
        super(SurvivalProgressionPolicy, policy).reset(
            None, None, Case(0, 7, 11, {}))
        policy.reset_route_lighting()
        policy._event_start = 0
        policy._baseline_health = 20.0
        policy.plan = None
        policy.portal_plan = None
        policy.workstation_sites = {}
        policy._lighting_initial_site = None
        policy._lighting_active = False
        return policy

    @staticmethod
    def planned_resource(name, *, stands, mine_cells=(), target=(9, 64, 0),
                         kind="tunnel"):
        route = SimpleNamespace(
            kind=kind, stands=tuple(stands), mine_cells=tuple(mine_cells))
        planned = SimpleNamespace(
            cell=target, route=route, removal=SimpleNamespace(
                protected_support_cells=()), mining_stance=stands[-1])
        return SimpleNamespace(resources=(SimpleNamespace(
            name=name, clusters=(SimpleNamespace(targets=(planned,)),)),))

    def test_portal_protection_preserves_certified_stand_volumes(self):
        plan = SimpleNamespace(
            occupied_cells=((10, 64, 10),),
            open_cells=((10, 65, 10),),
            temporary_cells=((11, 64, 10),),
            approach_stands=((0, 64, 0), (1, 64, 0)),
            work_stance=(2, 64, 0),
            interaction_stance=(3, 64, 0),
            entry_stance=(4, 65, 0),
            exit_stance=(5, 64, 0),
        )

        protected = _portal_protected_cells(plan)

        for x, y, z in (
                *plan.approach_stands,
                plan.work_stance,
                plan.interaction_stance,
                plan.exit_stance):
            self.assertTrue({
                (x, y - 1, z), (x, y, z), (x, y + 1, z),
            }.issubset(protected))
        self.assertIn(plan.entry_stance, protected)
        self.assertIn((4, 66, 0), protected)
        # Entry support is supplied by the future frame. It is protected in a
        # real plan through occupied_cells, not invented as existing terrain.
        self.assertNotIn((4, 64, 0), protected)
        self.assertTrue(set(
            plan.occupied_cells + plan.open_cells + plan.temporary_cells
        ).issubset(protected))

    def test_completion_portal_starts_locally_without_a_return_leg(self):
        policy = self.policy()
        policy.spec = SurvivalProgressionSpec(
            stages=policy.spec.stages,
            portal=PortalSettings(
                enabled=True, strategy="completion_chamber"),
        )
        policy.portal_plan = SimpleNamespace(
            approach_stands=((6, 64, 5), (7, 64, 5)),
        )
        policy._portal_return_certificate = None
        policy._portal_return_plan = Mock(
            side_effect=AssertionError("completion portal must stay local"))

        policy._plan_portal_return(Env(x=6.5, y=64, z=5.5))

        policy._portal_return_plan.assert_not_called()
        self.assertEqual(policy.stage, "portal_approach")
        self.assertEqual(policy._route, policy.portal_plan.approach_stands)
        self.assertEqual(policy._route_mine_cells, frozenset())

    def test_only_initial_portals_preserve_traversed_return_routes(self):
        stage = ResourceStage(
            "wood", ("log",), 1, "log", route_kinds=("surface",))
        plan = self.planned_resource(
            "wood", stands=((0, 64, 0),), target=(1, 64, 0),
            kind="surface")
        plan.fluid_seal_cells = ()
        plan.to_dict = lambda: {"resource": "wood"}
        target = plan.resources[0].clusters[0].targets[0]
        target.route.support_cells = ()

        for strategy, expected in (
                ("completion_chamber", False), ("initial", True)):
            with self.subTest(strategy=strategy):
                policy = self.policy()
                policy.spec = SurvivalProgressionSpec(
                    stages=(stage,), portal=PortalSettings(
                        enabled=True, strategy=strategy))
                policy._world = Mock()
                policy._portal_cells = frozenset()
                policy._opened = set()
                policy._removed = set()
                policy._placed = {}
                policy._return_passage_cells = set()
                policy._return_support_cells = set()
                policy._decision_seed = 23
                policy._stage_plans = []
                policy._provenance_workstations = []
                policy.workstation_sites = {}
                planner = Mock(return_value=SimpleNamespace(
                    build=lambda: plan))
                policy.plan_builder = planner

                with patch(
                        "bedrock_rl.sdg.policy.progression.policy."
                        "_workstation_sites", return_value=({}, None)) as sites:
                    policy._plan_stage(Env(x=0.5, y=64, z=0.5), stage)

                self.assertIs(
                    planner.call_args.kwargs["preserve_traversed_route"],
                    expected)
                self.assertIs(
                    sites.call_args.kwargs["preserve_return_route"],
                    expected)

    def test_fluid_seal_commit_is_visible_and_protected_in_later_stage(self):
        seal_stage = ResourceStage(
            "sealed", ("stone",), 1, "cobblestone",
            route_kinds=("tunnel",), sealable_fluids=("water",),
            fluid_seal_item="cobblestone")
        later_stage = ResourceStage(
            "later", ("stone",), 1, "cobblestone",
            route_kinds=("tunnel",))
        policy = self.policy()
        policy.spec = SurvivalProgressionSpec(
            stages=(seal_stage, later_stage), portal=False)
        policy._opened = set()
        policy._removed = set()
        policy._placed = {}
        policy._portal_cells = frozenset({(20, 64, 20)})
        policy._decision_seed = 23
        policy._world = Mock()
        policy._stage_plans = []
        policy._provenance_workstations = []
        policy.workstation_sites = {}

        first_target = SimpleNamespace(
            cell=(3, 63, 0), mining_stance=(2, 63, 0),
            route=SimpleNamespace(
                kind="tunnel", stands=((2, 63, 0),),
                mine_cells=((2, 64, 0),), support_cells=()),
            removal=SimpleNamespace(protected_support_cells=()),
        )
        policy.plan = SimpleNamespace(
            resources=(SimpleNamespace(
                name="sealed", clusters=(SimpleNamespace(
                    targets=(first_target,)),)),),
            fluid_seal_cells=((4, 63, 0),),
        )
        policy._commit_stage_plan(seal_stage)

        later_target = SimpleNamespace(
            cell=(8, 62, 0), mining_stance=(7, 62, 0),
            route=SimpleNamespace(
                kind="tunnel", stands=((7, 62, 0),),
                mine_cells=(), support_cells=()),
            removal=SimpleNamespace(protected_support_cells=()),
        )
        later_plan = SimpleNamespace(
            resources=(SimpleNamespace(
                name="later", clusters=(SimpleNamespace(
                    targets=(later_target,)),)),),
            fluid_seal_cells=(),
            to_dict=lambda: {"resource": "later"},
        )
        planner_calls = []

        def plan_builder(_world, _goal, **kwargs):
            planner_calls.append(kwargs)
            return SimpleNamespace(build=lambda: later_plan)

        policy.plan_builder = plan_builder
        with patch(
                "bedrock_rl.sdg.policy.progression.policy."
                "_workstation_sites", return_value=({}, None)) as sites:
            policy._plan_stage(Env(x=2.5, y=63.0, z=0.5), later_stage)

        cobblestone = policy._block_ids("cobblestone")[0]
        self.assertEqual(policy._placed, {(4, 63, 0): cobblestone})
        self.assertEqual(
            planner_calls[0]["placed"], {(4, 63, 0): cobblestone})
        self.assertIn((4, 63, 0), planner_calls[0]["protected"])
        self.assertIn((20, 64, 20), planner_calls[0]["protected"])
        self.assertEqual(
            sites.call_args.kwargs["protected"],
            {(20, 64, 20), (4, 63, 0)})

    def test_cost_certificate_accounts_routes_workstations_and_tools(self):
        prep = ResourceStage(
            "wood", ("log",), 1, "log", route_kinds=("surface",),
            after=(
                WorkstationOperation(
                    "bench", "crafting_table", "crafting_table",
                    "crafting_table"),
                CraftOperation("wooden_pickaxe", 1, 3, "bench"),
                CraftOperation("torch", 4, 2),
                BeginLightingOperation("bench"),
            ))
        stone = ResourceStage(
            "stone", ("stone",), 1, "cobblestone",
            tool="wooden_pickaxe", route_kinds=("tunnel",))
        policy = self.policy()
        policy.spec = SurvivalProgressionSpec(
            stages=(prep, stone), portal=False)
        policy.workstation_sites = {"bench": WorkstationSite(
            "bench", (1, 64, 0), (1, 63, 0), (0, 64, 0),
            approach_stands=((0, 64, 0),))}
        policy._stage_plan_objects = [
            self.planned_resource(
                "wood", stands=((0, 64, 0),), target=(2, 64, 0),
                kind="surface"),
            self.planned_resource(
                "stone", stands=tuple((x, 64, 0) for x in range(11)),
                mine_cells=((5, 65, 0),), target=(11, 64, 0)),
        ]
        policy._portal_return_certificate = None
        policy._costs = policy._build_cost_certificate()

        costs = policy.cost_certificate()
        self.assertEqual(costs["schema"], "bedrock.progression_costs.v1")
        self.assertEqual(costs["stages"][1]["route_stands"], 11)
        self.assertEqual(costs["totals"]["workstations"], 1)
        self.assertEqual(costs["lighting"]["required"], 2)
        self.assertTrue(costs["lighting"]["within_budget"])
        tool = costs["tools"][0]
        self.assertEqual(tool["tool"], "wooden_pickaxe")
        self.assertEqual(tool["required"], 2)
        self.assertEqual(tool["available"], 59)
        self.assertNotIn("planned_cells", tool)
        self.assertEqual(
            tool["planned_cells_sha256"],
            "bf94b5db7b116031a4d806784aa71a7f"
            "ab2dd37985922ea10c17ee4995120cc5")

    def test_cost_certificate_counts_reused_bucket_and_repeated_cast_breaks(self):
        stage = ResourceStage(
            "materials", ("log",), 1, "log", route_kinds=("surface",),
            after=(
                WorkstationOperation(
                    "bench", "crafting_table", "crafting_table",
                    "crafting_table"),
                CraftOperation("bucket", 1, 3, "bench"),
                CraftOperation("diamond_pickaxe", 1, 3, "bench"),
                BucketCastMineOperation(
                    "obsidian", 10, "diamond_pickaxe", 188),
            ))
        policy = self.policy()
        policy.spec = SurvivalProgressionSpec(
            stages=(stage,), portal=PortalSettings(
                enabled=True, method="bucket_cast_mined"))
        policy.workstation_sites = {"bench": WorkstationSite(
            "bench", (1, 64, 0), (1, 63, 0), (0, 64, 0),
            approach_stands=((0, 64, 0),))}
        policy._stage_plan_objects = [self.planned_resource(
            "materials", stands=((0, 64, 0),), target=(2, 64, 0),
            kind="surface")]
        policy._portal_return_certificate = None
        policy.portal_plan = SimpleNamespace(
            placements=(), approach_stands=())
        policy._costs = policy._build_cost_certificate()

        casting = policy._costs["portal"]["casting"]
        self.assertEqual(casting["reusable_buckets"], 1)
        self.assertEqual(casting["casting_sources"], 10)
        self.assertEqual(casting["coolant_sources"], 1)
        self.assertEqual(casting["result_breaks"], 10)
        self.assertTrue(casting["casting_sources_unique"])
        self.assertEqual(
            policy._material_budget("obsidian")["produced"], 10)
        self.assertEqual(policy._material_budget("bucket")["available"], 1)
        tool = next(item for item in policy._costs["tools"]
                    if item["tool"] == "diamond_pickaxe")
        self.assertEqual(tool["planned_breaks"], 0)
        self.assertEqual(tool["extra_breaks"], 10)
        self.assertEqual(tool["required"], 10)
        self.assertTrue(tool["within_budget"])

    def test_two_crafted_iron_pickaxes_certify_five_hundred_breaks(self):
        prep = ResourceStage(
            "wood", ("log",), 1, "log", route_kinds=("surface",),
            after=(
                WorkstationOperation(
                    "bench", "crafting_table", "crafting_table",
                    "crafting_table"),
                CraftOperation("iron_pickaxe", 2, 3, "bench"),
            ))
        ore = ResourceStage(
            "ore", ("stone",), 1, "cobblestone",
            tool="iron_pickaxe", route_kinds=("tunnel",))
        policy = self.policy()
        policy.spec = SurvivalProgressionSpec(
            stages=(prep, ore), portal=False)
        policy.workstation_sites = {"bench": WorkstationSite(
            "bench", (1, 64, 0), (1, 63, 0), (0, 64, 0),
            approach_stands=((0, 64, 0),))}
        policy._stage_plan_objects = [
            self.planned_resource(
                "wood", stands=((0, 64, 0),), target=(2, 64, 0),
                kind="surface"),
            self.planned_resource(
                "ore", stands=((0, 64, 0),),
                mine_cells=tuple((x, 65, 0) for x in range(481)),
                target=(481, 64, 0)),
        ]
        policy._portal_return_certificate = None

        tool = next(item for item in policy._tool_ledgers()
                    if item.tool == "iron_pickaxe")

        self.assertEqual(tool.copies, 2)
        self.assertEqual(tool.capacity_per_copy, 250)
        self.assertEqual(tool.available, 500)
        self.assertEqual(tool.required, 482)
        self.assertTrue(tool.within_budget)

    def test_preflight_rejects_light_and_tool_budget_shortfalls(self):
        prep = ResourceStage(
            "wood", ("log",), 1, "log", route_kinds=("surface",),
            after=(
                WorkstationOperation(
                    "bench", "crafting_table", "crafting_table",
                    "crafting_table"),
                CraftOperation("wooden_pickaxe", 1, 3, "bench"),
                CraftOperation("torch", 1, 2),
                BeginLightingOperation("bench"),
            ))
        stone = ResourceStage(
            "stone", ("stone",), 1, "cobblestone",
            tool="wooden_pickaxe", route_kinds=("tunnel",))
        policy = self.policy()
        policy.spec = SurvivalProgressionSpec(
            stages=(prep, stone), portal=False)
        policy.workstation_sites = {"bench": WorkstationSite(
            "bench", (1, 64, 0), (1, 63, 0), (0, 64, 0),
            approach_stands=((0, 64, 0),))}
        policy._stage_plan_objects = [
            self.planned_resource(
                "wood", stands=((0, 64, 0),), target=(2, 64, 0),
                kind="surface"),
            self.planned_resource(
                "stone", stands=tuple((x, 64, 0) for x in range(61)),
                mine_cells=tuple((x, 65, 0) for x in range(60)),
                target=(61, 64, 0)),
        ]
        policy._portal_return_certificate = None
        policy._costs = policy._build_cost_certificate()

        self.assertFalse(policy._costs["lighting"]["within_budget"])
        with self.assertRaises(CaseRejected) as light:
            policy._require_light_budget()
        self.assertEqual(light.exception.code, "progression_light_budget")
        with self.assertRaises(CaseRejected) as tool:
            policy._require_tool_durability_budget()
        self.assertEqual(
            tool.exception.code, "progression_tool_durability_budget")
        self.assertIn("tools", tool.exception.details)

    def test_fluid_seal_budget_reserves_structure_scaffolds(self):
        stage = ResourceStage(
            "stone", ("stone",), 6, "cobblestone",
            route_kinds=("tunnel",), sealable_fluids=("water",),
            fluid_seal_item="cobblestone")
        policy = self.policy()
        policy.spec = SurvivalProgressionSpec(
            stages=(stage,), portal=PortalSettings(enabled=True))
        policy._stage_plan_objects = [SimpleNamespace(
            fluid_seal_cells=((1, 60, 0), (2, 60, 0), (3, 60, 0)))]
        policy.portal_plan = SimpleNamespace(placements=(
            SimpleNamespace(role="scaffold"),
            SimpleNamespace(role="frame"),
            SimpleNamespace(role="scaffold"),
            SimpleNamespace(role="scaffold"),
        ))

        budget = policy._fluid_seal_budgets()[0]

        self.assertEqual(budget["produced"], 6)
        self.assertEqual(budget["required"], 3)
        self.assertEqual(budget["reserve"], 3)
        self.assertEqual(budget["usable"], 3)
        self.assertTrue(budget["within_budget"])

        policy.spec = SurvivalProgressionSpec(stages=(ResourceStage(
            "stone", ("stone",), 5, "cobblestone",
            route_kinds=("tunnel",), sealable_fluids=("water",),
            fluid_seal_item="cobblestone"),),
            portal=PortalSettings(enabled=True))
        with self.assertRaises(CaseRejected) as raised:
            policy._fluid_seal_budgets()
        self.assertEqual(
            raised.exception.code, "progression_fluid_seal_budget")
        self.assertEqual(raised.exception.details["shortfall"], 1)

    def test_hotbar_priority_is_configurable_and_excludes_wanted_items(self):
        policy = self.policy()
        policy.spec = SurvivalProgressionSpec(
            stages=policy.spec.stages, portal=False,
            hotbar=HotbarPolicy(("sand", "dirt")))
        env = Env(
            hotbar_ids=[3] + [0] * 7 + [12],
            hotbar_counts=[1] + [0] * 7 + [1])

        self.assertEqual(policy._hotbar_swaps(env), ("sand",))
        self.assertEqual(policy._hotbar_swaps(env, "sand"), ("dirt",))

        policy.workstation_sites["bench"] = WorkstationSite(
            "bench", (1, 64, 0), (1, 63, 0), (0, 64, 0))
        policy.place_workstation = Mock(return_value=[{
            "action": "right_click"}])
        operation = WorkstationOperation(
            "bench", "crafting_table", "crafting_table", "chest")

        self.assertEqual(policy._operation_actions(env, operation), [{
            "action": "right_click"}])
        self.assertEqual(
            policy.place_workstation.call_args.kwargs["replace"],
            ("sand",))

    def test_visible_tool_does_not_trigger_phantom_backpack_recovery(self):
        policy = self.policy()
        policy.workstation_sites["bench"] = WorkstationSite(
            "bench", (1, 64, 0), (1, 63, 0), (0, 64, 0))
        policy._workstation_ready = Mock(return_value=True)
        policy.prepare_recipe_ingredients = Mock(return_value=[])
        policy.craft_3x3 = Mock(return_value=[{
            "action": "left_click"}])
        env = Env(
            hotbar_ids=[270, 5] + [0] * 7,
            hotbar_counts=[1, 10] + [0] * 7,
            journal=[
                Event("slot_changed", slot=0, item=270, count=1),
                Event("container_opened", kind=1, x=1, y=64, z=0),
            ])
        env.obs["container"] = 1

        result = policy._operation_actions(
            env, CraftOperation("wooden_pickaxe", 2, 3, "bench"))

        self.assertEqual(result, [{"action": "left_click"}])
        policy.craft_3x3.assert_called_once_with(
            env, "wooden_pickaxe", table=(1, 64, 0),
            output_replace=())

    def test_craft_recovery_never_swaps_out_active_ingredients(self):
        policy = self.policy()
        policy.spec = SurvivalProgressionSpec(
            stages=policy.spec.stages, portal=False,
            hotbar=HotbarPolicy(("stick", "cobblestone")))
        policy.workstation_sites["bench"] = WorkstationSite(
            "bench", (1, 64, 0), (1, 63, 0), (0, 64, 0))
        policy._workstation_ready = Mock(return_value=True)
        policy.prepare_recipe_ingredients = Mock(return_value=[])
        policy.craft_3x3 = Mock(return_value=[{
            "action": "left_click"}])
        env = Env(
            hotbar_ids=[264, 280, 4, 1, 2, 3, 5, 6, 7],
            hotbar_counts=[3, 2, 12, 1, 1, 1, 1, 1, 1])

        result = policy._operation_actions(
            env, CraftOperation("diamond_pickaxe", 1, 3, "bench"))

        self.assertEqual(result, [{"action": "left_click"}])
        policy.prepare_recipe_ingredients.assert_called_once_with(
            env, "diamond_pickaxe", 3, replace=("cobblestone",))
        policy.craft_3x3.assert_called_once_with(
            env, "diamond_pickaxe", table=(1, 64, 0),
            output_replace=("cobblestone",))

    def test_craft_recovery_falls_back_to_any_unprotected_hotbar_stack(self):
        policy = self.policy()
        policy.spec = SurvivalProgressionSpec(
            stages=policy.spec.stages, portal=False,
            hotbar=HotbarPolicy(("sand", "dirt")))
        policy.workstation_sites["bench"] = WorkstationSite(
            "bench", (1, 64, 0), (1, 63, 0), (0, 64, 0))
        policy._workstation_ready = Mock(return_value=True)
        policy.prepare_recipe_ingredients = Mock(return_value=[])
        policy.craft_3x3 = Mock(return_value=[{
            "action": "left_click"}])
        env = Env(
            hotbar_ids=[264, 280, 1, 2, 5, 6, 7, 8, 9],
            hotbar_counts=[3, 2, 12, 1, 1, 1, 1, 1, 1])

        result = policy._operation_actions(
            env, CraftOperation("diamond_pickaxe", 1, 3, "bench"))

        self.assertEqual(result, [{"action": "left_click"}])
        policy.prepare_recipe_ingredients.assert_called_once_with(
            env, "diamond_pickaxe", 3, replace=(1,))
        policy.craft_3x3.assert_called_once_with(
            env, "diamond_pickaxe", table=(1, 64, 0),
            output_replace=(1,))

    def test_workstation_sneaks_when_placed_on_interactive_support(self):
        policy = self.policy()
        policy.workstation_sites = {
            "furnace": WorkstationSite(
                "furnace", (1, 63, 0), (1, 62, 0), (0, 64, 0)),
            "bench": WorkstationSite(
                "bench", (1, 64, 0), (1, 63, 0), (0, 64, 0)),
        }
        policy.place_workstation = Mock(return_value=[{
            "action": "right_click", "sneak": True}])
        operation = WorkstationOperation(
            "bench", "crafting_table", "crafting_table", "chest")

        policy._operation_actions(Env(), operation)

        self.assertTrue(
            policy.place_workstation.call_args.kwargs["sneak"])

    def test_workstation_alcove_is_mined_before_placement(self):
        policy = self.policy()
        policy.spec = SurvivalProgressionSpec(stages=(ResourceStage(
            "stone", ("stone",), 1, "cobblestone",
            tool="wooden_pickaxe", route_kinds=("tunnel",)),),
            portal=False)
        policy._stage_index = 0
        cell = (1, 64, 0)
        policy.workstation_sites["bench"] = WorkstationSite(
            "bench", cell, (1, 63, 0), (0, 64, 0),
            clear_cells=(cell,))
        policy.mine_exact = Mock(return_value=[{
            "action": "left_click_hold", "ticks": 28}])
        env = Env(
            hotbar_ids=[270] + [0] * 8,
            hotbar_counts=[1] + [0] * 8)
        operation = WorkstationOperation(
            "bench", "crafting_table", "crafting_table", "chest")

        self.assertEqual(policy._operation_actions(env, operation), [{
            "action": "left_click_hold", "ticks": 28}])
        policy.mine_exact.assert_called_once_with(
            env, cell, tool="wooden_pickaxe", ticks=80,
            stand=(0.5, 0.5))

        policy._stage_plan_objects = []
        policy._portal_return_certificate = None
        self.assertIn(cell, policy._planned_break_cells())

    def test_workstation_overlay_resolves_declared_block_generically(self):
        operation = WorkstationOperation(
            "storage", "chest", "chest", "chest")
        stage = ResourceStage(
            "wood", ("log",), 1, "log", route_kinds=("surface",),
            hazard_clearance=0, after=(operation,))
        spec = SurvivalProgressionSpec(
            stages=(stage,), portal=False, hazard_clearance=3)
        target = SimpleNamespace(
            cell=(4, 64, 0), mining_stance=(0, 64, 0),
            route=SimpleNamespace(mine_cells=(), stands=((0, 64, 0),)))
        plan = SimpleNamespace(resources=(SimpleNamespace(
            name="wood", clusters=(SimpleNamespace(
                targets=(target,)),)),))
        world = Mock(seed=7, snapshot_sha256="sha", player={})
        world.in_bounds.return_value = True
        world.block.side_effect = lambda _x, y, _z: 1 if y == 63 else 0

        with patch(
                "bedrock_rl.sdg.policy.progression.spec."
                "is_surface_stand", return_value=True) as safe_stand, \
                patch(
                    "bedrock_rl.sdg.policy.progression.spec."
                    "resolve_blocks", return_value=(54,)) as resolver:
            sites, changed = _workstation_sites(
                world, plan, spec, decision_seed=11)

        resolver.assert_called_once_with(["chest"])
        self.assertEqual(changed.placed[sites["storage"].cell], 54)
        self.assertTrue(all(
            call.kwargs["hazard_clearance"] == 0
            for call in safe_stand.call_args_list))

    def test_workstation_site_requires_a_proven_placement_face(self):
        operation = WorkstationOperation(
            "storage", "chest", "chest", "chest")
        stage = ResourceStage(
            "wood", ("log",), 1, "log", route_kinds=("surface",),
            hazard_clearance=0, after=(operation,))
        spec = SurvivalProgressionSpec(
            stages=(stage,), portal=False, hazard_clearance=0)
        target = SimpleNamespace(
            cell=(4, 64, 0), mining_stance=(0, 64, 0),
            route=SimpleNamespace(
                mine_cells=(), stands=((0, 64, 0), (1, 64, 0))))
        plan = SimpleNamespace(resources=(SimpleNamespace(
            name="wood", clusters=(SimpleNamespace(
                targets=(target,)),)),))
        world = Mock(seed=7, snapshot_sha256="sha", player={})
        world.in_bounds.return_value = True
        world.block.side_effect = lambda _x, y, _z: 1 if y == 63 else 0

        with patch(
                "bedrock_rl.sdg.policy.progression.spec."
                "is_surface_stand", return_value=True), patch(
                "bedrock_rl.sdg.policy.progression.spec."
                "clear_placement_ray", return_value=False):
            with self.assertRaises(CaseRejected) as raised:
                _workstation_sites(
                    world, plan, spec, decision_seed=11)

        self.assertEqual(
            raised.exception.code, "progression_workstation_site_missing")

    def test_workstation_replaces_a_head_height_cavity_with_side_alcove(self):
        operation = WorkstationOperation(
            "storage", "chest", "chest", "chest")
        stage = ResourceStage(
            "wood", ("log",), 1, "log", route_kinds=("tunnel",),
            hazard_clearance=0, after=(operation,))
        spec = SurvivalProgressionSpec(stages=(stage,), portal=False)
        target = SimpleNamespace(
            cell=(1, 65, 0), mining_stance=(0, 64, 0),
            route=SimpleNamespace(
                mine_cells=(), stands=((0, 64, 0),)))
        plan = SimpleNamespace(resources=(SimpleNamespace(
            name="wood", clusters=(SimpleNamespace(
                targets=(target,)),)),))
        world = Mock(seed=7, snapshot_sha256="sha", player={})
        world.in_bounds.return_value = True
        world.block.side_effect = lambda _x, y, _z: 1 if y <= 64 else 0

        with patch(
                "bedrock_rl.sdg.policy.progression.spec."
                "is_surface_stand", return_value=True), patch(
                "bedrock_rl.sdg.policy.progression.spec."
                "clear_placement_ray", return_value=True):
            sites, _changed = _workstation_sites(
                world, plan, spec, decision_seed=11)

        site = sites["storage"]
        self.assertEqual(site.cell[1], site.stand[1])
        self.assertEqual(site.clear_cells, (site.cell,))

    def test_portal_return_uses_configured_tool_and_duration(self):
        policy = self.policy()
        policy.spec = SurvivalProgressionSpec(
            stages=policy.spec.stages,
            portal=PortalSettings(
                enabled=True, return_tool="iron_pickaxe",
                return_ticks=37))
        policy._route = ((0, 64, 0), (1, 64, 0))
        policy._route_index = 1
        policy._route_mine_cells = frozenset({(1, 64, 0)})
        policy._route_clear_attempts = {}
        policy.mine_exact = Mock(return_value=[{
            "action": "left_click_hold", "ticks": 37}])

        result = policy._clear_return_leg(Env())

        self.assertEqual(result[0]["ticks"], 37)
        self.assertEqual(
            policy.mine_exact.call_args.kwargs["tool"], "iron_pickaxe")
        self.assertEqual(policy.mine_exact.call_args.kwargs["ticks"], 37)

    def test_portal_activation_and_igniter_use_configured_catalog_names(self):
        policy = self.policy()
        policy.spec = SurvivalProgressionSpec(
            stages=policy.spec.stages,
            portal=PortalSettings(
                enabled=True, active_block="netherrack",
                igniter_item="torch"))
        policy.stage = "portal_ignite"
        policy._portal_ignite_attempts = 0
        policy.portal_plan = SimpleNamespace(
            open_cells=((1, 64, 0),),
            interaction_stance=(0, 64, 0),
            interaction_support=(1, 63, 0),
            interaction_cell=(1, 64, 0),
            entry_stance=(2, 64, 0))
        env = Env(hotbar_ids=[50] + [0] * 8,
                  hotbar_counts=[1] + [0] * 8)
        policy._block_ids = Mock(return_value=(87,))
        policy.observed_block = Mock(return_value=0)
        policy.select = Mock(return_value=[])
        policy.click_target = Mock(return_value=(1, (1, 63, 0)))
        policy.click_destination = Mock(return_value=(1, 64, 0))

        result = policy._portal_actions(env)

        self.assertEqual(result[0], {"action": "right_click"})
        self.assertEqual(result[-1], {"action": "wait", "ticks": 4})
        policy._block_ids.assert_called_once_with("netherrack")
        policy.select.assert_called_once_with(env, "torch")
        self.assertEqual(policy.observed_block.call_args.args[1], (1, 64, 0))
        self.assertEqual(policy.stage, "portal_ignite")

    def test_portal_ignition_observes_selected_tool_before_click(self):
        policy = self.policy()
        policy.spec = SurvivalProgressionSpec(
            stages=policy.spec.stages,
            portal=PortalSettings(enabled=True))
        policy.stage = "portal_ignite"
        policy._portal_ignite_attempts = 0
        policy.portal_plan = SimpleNamespace(
            open_cells=((1, 64, 0),),
            interaction_stance=(0, 64, 0),
            interaction_support=(1, 63, 0),
            interaction_cell=(1, 64, 0),
            entry_stance=(2, 64, 0))
        env = Env(x=0.5, y=64.0, z=0.5)
        policy.observed_block = Mock(return_value=0)
        selection = [{"action": "key", "k": "9", "ticks": 1}]
        policy.select = Mock(return_value=selection)
        policy.click_target = Mock(return_value=(1, (1, 63, 0)))
        policy.click_destination = Mock(return_value=(1, 64, 0))

        result = policy._portal_actions(env)

        self.assertEqual(result, selection)
        self.assertEqual(policy._portal_ignite_attempts, 0)

    def test_portal_activation_accepts_exact_ray_when_sparse_scan_omits_it(self):
        policy = self.policy()
        policy.stage = "portal_ignite"
        policy._portal_entry_attempts = 0
        interior = (1, 64, 0)
        policy.portal_plan = SimpleNamespace(
            open_cells=(interior,), interaction_stance=(0, 64, 0),
            interaction_support=(1, 63, 0), interaction_cell=interior,
            entry_stance=(2, 64, 0))
        policy.observed_block = Mock(return_value=None)
        policy.click_target = Mock(return_value=(90, interior))
        entered = [{"action": "key", "k": "w", "ticks": 2}]
        policy.face_and_move_to = Mock(return_value=entered)

        result = policy._portal_actions(Env())

        self.assertEqual(result, entered)
        self.assertEqual(policy.stage, "portal_enter")
        self.assertEqual(policy._portal_entry_attempts, 1)

    def test_portal_ignition_reports_observed_frame_when_fire_remains(self):
        policy = self.policy()
        policy.stage = "portal_ignite"
        policy._portal_ignite_attempts = 1
        interior = (1, 64, 0)
        frame = ((1, 63, 0), (2, 63, 0))
        policy.portal_plan = SimpleNamespace(
            open_cells=(interior,), occupied_cells=frame,
            interaction_stance=(0, 64, 0),
            interaction_support=frame[0], interaction_cell=interior,
            entry_stance=(2, 64, 0))
        observed = {interior: 51, frame[0]: 49, frame[1]: 4}
        policy.observed_block = Mock(
            side_effect=lambda _env, cell: observed.get(cell))

        result = policy._portal_actions(Env())

        self.assertIsInstance(result, PolicyVeto)
        self.assertEqual(result.code, "progression_portal_frame_invalid")
        self.assertIn("((2, 63, 0), 4)", result.reason)
        self.assertEqual(result.observed, (interior,))

    def test_portal_entry_wait_uses_configured_82_tick_boundary(self):
        policy = self.policy()
        policy.spec = SurvivalProgressionSpec(
            stages=policy.spec.stages,
            portal=PortalSettings(enabled=True, entry_wait_ticks=82))
        policy.stage = "portal_enter"
        policy._portal_entry_attempts = 0
        policy.portal_plan = SimpleNamespace(entry_stance=(2, 64, 0))
        policy.face_and_move_to = Mock(return_value=None)

        result = policy._portal_actions(Env(x=2.5, y=64.0, z=0.5))

        self.assertEqual(result, [{"action": "wait", "ticks": 82}])
        self.assertEqual(policy._portal_entry_attempts, 1)

    def test_portal_clear_does_not_count_aim_as_failed_attack(self):
        policy = self.policy()
        policy.stage = "portal_clear"
        policy._portal_clear_attempts = {}
        policy._portal_clear_control_attempts = {}
        policy.portal_plan = SimpleNamespace(
            clear_cells=((2, 64, 0),), work_stance=(1, 64, 0))
        policy.events = Mock(return_value=())
        policy.mining_hold_ticks = Mock(return_value=80)
        policy.mine_exact = Mock(return_value=[{
            "action": "mouse_move", "dx": 1, "dy": 1}])

        for _ in range(5):
            result = policy._portal_actions(Env())
            self.assertEqual(result[0]["action"], "mouse_move")

        self.assertEqual(policy._portal_clear_attempts, {})
        self.assertEqual(policy._portal_clear_control_attempts, {
            (2, 64, 0): 5})

    def test_portal_frame_recovers_material_from_backpack(self):
        policy = self.policy()
        policy.stage = "portal_frame"
        policy._placement_index = 0
        policy.portal_plan = SimpleNamespace(placements=(SimpleNamespace(
            role="frame", material="obsidian", cell=(2, 64, 0),
            support=(2, 63, 0), stance=(1, 64, 0)),),
            work_stance=(1, 64, 0))
        policy.placed = Mock(return_value=False)
        policy.place = Mock(return_value=None)
        policy._hotbar_swaps = Mock(return_value=(4,))
        recovered = [{"action": "inventory_swap", "slot": 9}]
        policy.recover_hotbar = Mock(return_value=recovered)
        env = Env(x=1.5, y=64.0, z=0.5)

        result = policy._portal_actions(env)

        self.assertEqual(result, recovered)
        policy.place.assert_called_once_with(
            env, "obsidian", (2, 64, 0), (2, 63, 0),
            block="obsidian", stand=(1.5, 0.5), jump_from_y=None,
            since=0)
        policy.recover_hotbar.assert_called_once_with(
            env, "obsidian", replace=(4,), minimum=1)

    def test_portal_frame_does_not_trust_stale_placement_event(self):
        policy = self.policy()
        policy.stage = "portal_frame"
        policy._placement_index = 0
        step = SimpleNamespace(
            role="frame", material="obsidian", cell=(2, 64, 0),
            support=(2, 63, 0), stance=(1, 64, 0), jump=False)
        policy.portal_plan = SimpleNamespace(
            placements=(step,), work_stance=(1, 64, 0))
        policy._portal_placement_event_start = 1
        placement = [{"action": "right_click"}]
        policy.place = Mock(return_value=placement)
        env = Env(x=1.5, y=64.0, z=0.5)
        env.journal.append(Event(
            "block_placed", block=49, x=2, y=64, z=0))

        result = policy._portal_actions(env)

        self.assertEqual(result, placement)
        policy.place.assert_called_once_with(
            env, "obsidian", step.cell, step.support,
            block="obsidian", stand=(1.5, 0.5), jump_from_y=None,
            since=1)

    def test_portal_frame_uses_certified_airborne_jump_pose(self):
        policy = self.policy()
        policy.stage = "portal_frame"
        policy._placement_index = 0
        step = SimpleNamespace(
            role="scaffold", material="scaffold", cell=(2, 68, 0),
            support=(2, 67, 0), stance=(1, 65, 0), jump=True)
        policy.portal_plan = SimpleNamespace(
            placements=(step,), work_stance=(1, 64, 0))
        policy.placed = Mock(return_value=False)
        policy.place = Mock(return_value=[{"action": "right_click"}])
        env = Env(x=1.5, y=66.2, z=0.5)

        result = policy._portal_actions(env)

        self.assertEqual(result, [{"action": "right_click"}])
        policy.place.assert_called_once_with(
            env, "cobblestone", step.cell, step.support,
            block="cobblestone", stand=(1.5, 0.5), jump_from_y=65,
            since=0)

    def test_health_and_body_fluid_contact_veto_every_stage(self):
        policy = self.policy()
        health = policy._guard(Env(health=19.5))
        self.assertIsInstance(health, PolicyVeto)
        self.assertEqual(health.code, "health_lost")

        fluid = policy._guard(Env(journal=[Event(
            "player_fluid_state", block=11, flags=FLUID_BODY,
            x=2, y=63, z=4)]))
        self.assertIsInstance(fluid, PolicyVeto)
        self.assertEqual(fluid.code, "fluid_contact")
        self.assertEqual(fluid.observed, ((2, 63, 4),))

        # A milestone boundary may start after the contact record.  The
        # latest fluid state still describes the player's current body.
        policy._guard_event_index = 1
        current = policy._guard(Env(journal=[Event(
            "player_fluid_state", block=11, flags=FLUID_BODY,
            x=2, y=63, z=4)]))
        self.assertIsInstance(current, PolicyVeto)
        self.assertEqual(current.code, "fluid_contact")

    def test_global_break_guard_rejects_collateral_in_every_phase(self):
        contexts = (
            ("resource executor", "mine:wood"),
            ("repeat-drop milestone", "milestone:gravel"),
            ("between resource stages", "milestone:wood"),
            ("portal return", "return_to_surface"),
        )
        for label, stage in contexts:
            with self.subTest(label=label):
                policy = self.policy()
                policy.stage = stage
                policy._allowed_break_cells = frozenset({(1, 64, 0)})
                veto = policy._guard(Env(journal=[Event(
                    "block_broken", block=1, drop=4,
                    x=2, y=64, z=0)]))

                self.assertIsInstance(veto, PolicyVeto)
                self.assertEqual(veto.code, "unplanned_block_removed")
                self.assertEqual(veto.observed, ((2, 64, 0),))

    def test_global_break_guard_accepts_every_declared_break_kind(self):
        policy = self.policy()
        target = (1, 64, 0)
        tunnel = (2, 64, 0)
        repeated = (3, 64, 0)
        portal_return = (4, 64, 0)
        policy._allowed_break_cells = frozenset({
            target, tunnel, repeated, portal_return,
        })
        env = Env(journal=[
            Event("block_broken", block=17, drop=5,
                  x=target[0], y=target[1], z=target[2]),
            Event("block_broken", block=1, drop=4,
                  x=tunnel[0], y=tunnel[1], z=tunnel[2]),
            Event("block_broken", block=13, drop=13,
                  x=repeated[0], y=repeated[1], z=repeated[2]),
            # Repeated-drop may intentionally break the same declared cell
            # several times without expanding the coordinate ledger.
            Event("block_broken", block=13, drop=318,
                  x=repeated[0], y=repeated[1], z=repeated[2]),
            Event("block_broken", block=1, drop=4,
                  x=portal_return[0], y=portal_return[1],
                  z=portal_return[2]),
        ])

        self.assertIsNone(policy._guard(env))
        self.assertEqual(policy._guard_event_index, len(env.journal))

    def test_lighting_begins_immediately_at_declared_workstation(self):
        policy = self.policy()
        policy.workstation_sites["bench"] = WorkstationSite(
            "bench", (1, 64, 0), (1, 63, 0), (0, 64, 0),
            ((2, 64, 0),))
        policy.light_route = Mock(return_value=[{
            "action": "right_click"}])
        policy._initial_light_vantage_route = Mock(
            return_value=((0, 64, 0),))
        env = Env(hotbar_ids=[50] + [0] * 8,
                  hotbar_counts=[16] + [0] * 8)

        result = policy._operation_actions(
            env, BeginLightingOperation("bench"))

        self.assertEqual(result, [{"action": "right_click"}])
        self.assertTrue(policy._lighting_active)
        self.assertGreaterEqual(
            policy.lighting_distance,
            policy.behaviors.light_every_blocks)
        self.assertEqual(policy._lighting_initial_site[1], (1, 64, 0))
        self.assertEqual(
            policy._lighting_initial_site[2], (1.49, 64, 0.0))

    def test_route_light_uses_authoritative_world_not_sparse_summary(self):
        policy = self.policy()
        policy.recent_route_cells = [(-1, 64, 0)]
        world = Mock()
        world.in_bounds.return_value = True
        world.block.side_effect = lambda cell: (
            1 if tuple(cell) == (-1, 65, 1) else 0)
        policy._live_world = Mock(return_value=world)
        env = Env(x=0.5, y=64.0, z=0.5)
        # The bounded model summary deliberately contains no wall blocks.
        self.assertEqual(env.obs["blocks"], [])

        site = policy.choose_route_light_site(env)

        self.assertIsNotNone(site)
        self.assertEqual(site[0], (-1, 65, 0))
        self.assertEqual(site[1], (-1, 65, 1))

    def test_route_light_skips_unknown_wall_beyond_snapshot_boundary(self):
        policy = self.policy()
        policy.recent_route_cells = [(167, 24, -34)]
        world = Mock()
        world.in_bounds.side_effect = lambda cell: tuple(cell)[0] <= 167

        def block(cell):
            cell = tuple(cell)
            if cell[0] > 167:
                raise IndexError(f"block {cell} is outside Bounds(test)")
            return 1 if cell == (167, 23, -34) else 0

        world.block.side_effect = block
        policy._live_world = Mock(return_value=world)
        env = Env(x=167.5, y=24.0, z=-33.5)

        site = policy.choose_route_light_site(env)

        self.assertIsNone(site)
        self.assertNotIn(
            (168, 24, -34),
            [tuple(call.args[0]) for call in world.block.call_args_list])

    def test_route_light_releases_excavated_stair_body_for_floor_torch(self):
        policy = self.policy()
        stand = (-1, 63, 0)
        route = RoutePlan(
            "tunnel", ((0, 64, 0), stand),
            ((0, 64, 0), (0, 65, 0), stand, (-1, 64, 0)),
            ((0, 63, 0), (-1, 62, 0)), (stand, (-1, 64, 0)))
        policy.fluid_source_plan = SimpleNamespace(steps=(SimpleNamespace(
            source_cell=(-3, 63, 0), pickup_ray_cells=((-3, 63, 0),),
            outbound=route, return_route=route),))
        policy.recent_route_cells = [stand]
        world = Mock()
        world.in_bounds.return_value = True
        world.block.side_effect = lambda cell: (
            1 if tuple(cell) == (-1, 62, 0) else 0)
        policy._live_world = Mock(return_value=world)
        env = Env(x=0.5, y=64.0, z=0.5, journal=(
            Event("block_broken", block=1, drop=4,
                  x=stand[0], y=stand[1], z=stand[2]),
        ))

        site = policy.choose_route_light_site(env)

        self.assertIsNotNone(site)
        self.assertEqual(site[:2], (stand, (-1, 62, 0)))
        self.assertEqual(site[3], "floor")

    def test_route_follower_accepts_partial_progress_not_just_endpoints(self):
        policy = self.policy()
        policy._start_route(((0, 64, 0), (1, 64, 0), (2, 64, 0)),
                            "test_route")
        policy.face_and_move_to = Mock(return_value=[{
            "action": "key", "k": "w", "ticks": 2}])

        result = policy._follow_route(Env(x=1.0, y=64.0, z=0.5))

        self.assertIsInstance(result, list)
        self.assertEqual(result[0]["action"], "key")
        policy.face_and_move_to.assert_called_once()

    def test_route_follower_does_not_coast_across_multiple_plan_cells(self):
        policy = self.policy()
        policy.movement = MovementProfile(max_blocks=2.5)
        policy._start_route(
            ((0, 64, 0), (1, 64, 0), (2, 64, 0)), "test_route")
        policy.face_and_move_to = Mock(return_value=[{
            "action": "key", "k": "w", "ticks": 2}])

        policy._follow_route(Env(x=0.5, y=64.0, z=0.5))

        policy.face_and_move_to.assert_called_once_with(
            unittest.mock.ANY, 1.5, 0.5, tolerance=0.4,
            sequential_hop=True, step_up=False, allow_jump=True,
            movement_cap_blocks=None)

    def test_route_origin_recovery_waits_for_transient_gravity_settle(self):
        policy = self.policy()
        env = Env(x=-14.05, y=71.0, z=65.5)

        settle = policy.recover_route_origin(env, (-15, 70, 65))

        self.assertEqual(settle, [{"action": "wait", "ticks": 2}])
        self.assertEqual(policy._origin_settle_attempts, 1)
        env.obs["x"] = -14.5
        env.obs["y"] = 70.0
        self.assertIsNone(
            policy.recover_route_origin(env, (-15, 70, 65)))
        self.assertEqual(policy._origin_settle_attempts, 0)

    def test_route_origin_recovery_centers_walk_tolerance_pose(self):
        policy = self.policy()
        policy.face_and_move_to = Mock(return_value=[{
            "action": "key", "k": "w", "ticks": 1}])
        env = Env(x=5.502, y=79.0, z=-0.186)

        result = policy.recover_route_origin(env, (5, 79, -1))

        self.assertEqual(result[0]["action"], "key")
        policy.face_and_move_to.assert_called_once_with(
            env, 5.5, -0.5, tolerance=0.05, allow_jump=False,
            movement_cap_blocks=0.3)

    def test_route_origin_recovery_uses_crouch_step_near_center(self):
        policy = self.policy()
        policy.recenter_interaction_stance = Mock(return_value=[{
            "action": "key", "k": ["a", "sneak"], "ticks": 1}])
        env = Env(x=10.438, y=75.0, z=-3.556)

        result = policy.recover_route_origin(env, (10, 75, -4))

        self.assertEqual(result[0]["k"], ["a", "sneak"])
        policy.recenter_interaction_stance.assert_called_once_with(
            env, 10.5, -3.5, tolerance=0.05)

    def test_route_origin_recovery_handles_adjacent_boundary_rounding(self):
        policy = self.policy()
        policy.face_and_move_to = Mock(return_value=[{
            "action": "key", "k": ["s", "sneak"], "ticks": 1}])
        env = Env(x=-65.508, y=14.024, z=-56.998)

        result = policy.recover_route_origin(env, (-66, 14, -58))

        self.assertEqual(result[0]["action"], "key")
        policy.face_and_move_to.assert_called_once_with(
            env, -65.5, -57.5, tolerance=0.05, allow_jump=False,
            movement_cap_blocks=0.3)

    def test_live_planning_start_recovers_to_adjacent_safe_stand(self):
        policy = self.policy()
        policy._live_world = Mock(return_value=Mock())
        policy.face_and_move_to = Mock(return_value=[{
            "action": "key", "k": "w", "ticks": 1}])
        env = Env(x=0.5, y=64.0, z=0.95)

        with patch(
                "bedrock_rl.sdg.policy.progression.policy."
                "is_surface_stand",
                side_effect=lambda _world, cell, **_:
                tuple(cell) == (0, 64, 1)):
            result = policy._recover_live_planning_start(env)

        self.assertEqual(result[0]["action"], "key")
        policy.face_and_move_to.assert_called_once_with(
            env, 0.5, 1.5, tolerance=0.05, sequential_hop=True,
            step_up=False, allow_jump=True, movement_cap_blocks=0.3)

    def test_descending_leg_continues_to_center_before_gravity_settles(self):
        policy = self.policy()
        policy._start_route(((3, 73, 3), (2, 72, 3)), "descent")
        policy.face_and_move_to = Mock(return_value=[{
            "action": "key", "k": "w", "ticks": 1}])

        result = policy._follow_route(Env(
            x=2.856, y=73.0, z=3.479))

        self.assertEqual(result[0]["action"], "key")
        policy.face_and_move_to.assert_called_once_with(
            unittest.mock.ANY, 2.5, 3.5, tolerance=0.05,
            sequential_hop=True, step_up=False, allow_jump=False,
            movement_cap_blocks=0.3)

    def test_singleton_route_centers_within_same_cell_before_completion(self):
        policy = self.policy()
        policy._start_route(((4, 78, 2),), "singleton")
        policy.face_and_move_to = Mock(return_value=[{
            "action": "key", "k": "w", "ticks": 1}])

        result = policy._follow_route(Env(x=4.574, y=78.0, z=2.006))

        self.assertEqual(result[0]["action"], "key")
        policy.face_and_move_to.assert_called_once_with(
            unittest.mock.ANY, 4.5, 2.5, tolerance=0.4,
            allow_jump=False, movement_cap_blocks=0.3)

    def test_stage_executor_uses_configured_pickup_attempt_bound(self):
        policy = self.policy()
        policy.behaviors = SurvivalBehaviors.parse({
            "pickups": {"attempts": 9},
        })
        target = SimpleNamespace()
        policy.plan = SimpleNamespace(resources=(SimpleNamespace(
            name="wood", clusters=(SimpleNamespace(
                targets=(target,)),)),))

        with patch(
                "bedrock_rl.sdg.policy.progression.policy."
                "MiningExecution", return_value=object()), \
                patch(
                    "bedrock_rl.sdg.policy.progression.policy."
                    "PlanExecutor") as executor_cls:
            policy._stage_executor(Env(), policy.spec.stages[0])

        self.assertEqual(
            executor_cls.call_args.kwargs["max_pickup_attempts"], 9)

    def test_elevated_drop_uses_safe_same_level_approach_not_solid_support(self):
        policy = self.policy()
        policy._world = Mock()
        policy._decision_seed = 11
        env = Env(x=-6.5, y=65.0, z=8.5)
        target = SimpleNamespace(cell=(-8, 66, 9))
        current = (-7, 65, 8)
        safe_side = (-7, 65, 9)

        with patch.object(policy, "_live_world", return_value=Mock()), \
                patch(
                    "bedrock_rl.sdg.policy.progression.policy."
                    "is_surface_stand",
                    side_effect=lambda _world, cell, **_: cell == safe_side), \
                patch(
                    "bedrock_rl.sdg.policy.progression.policy."
                    "plan_surface_path",
                    return_value=(current, safe_side)) as planner:
            point, sneak = policy.drop_pickup_site(env, target)

        self.assertEqual(point, (-6.5, 9.5))
        self.assertFalse(sneak)
        self.assertTrue(policy._drop_pickup_allow_jump)
        self.assertEqual(planner.call_args.args[1:], (current, safe_side))

    def test_cardinal_drop_neighbor_sneaks_to_safe_ledge_edge(self):
        policy = self.policy()
        policy._world = Mock()
        env = Env(x=9.5, y=83.0, z=3.5)
        target = SimpleNamespace(cell=(8, 84, 3))
        current = (9, 83, 3)

        with patch.object(policy, "_live_world", return_value=Mock()), \
                patch(
                    "bedrock_rl.sdg.policy.progression.policy."
                    "is_surface_stand",
                    side_effect=lambda _world, cell, **_: cell == current), \
                patch(
                    "bedrock_rl.sdg.policy.progression.policy."
                    "plan_surface_path", return_value=(current,)):
            point, sneak = policy.drop_pickup_site(env, target)

        self.assertEqual(point, (8.5, 3.5))
        self.assertTrue(sneak)

    def test_diagonal_drop_neighbor_sneaks_to_safe_ledge_corner(self):
        policy = self.policy()
        policy._world = Mock()
        env = Env(x=-8.5, y=65.0, z=5.5)
        target = SimpleNamespace(cell=(-10, 66, 4))
        current = (-9, 65, 5)

        with patch.object(policy, "_live_world", return_value=Mock()), \
                patch(
                    "bedrock_rl.sdg.policy.progression.policy."
                    "is_surface_stand",
                    side_effect=lambda _world, cell, **_: cell == current), \
                patch(
                    "bedrock_rl.sdg.policy.progression.policy."
                    "plan_surface_path", return_value=(current,)):
            point, sneak = policy.drop_pickup_site(env, target)

        self.assertEqual(point, (-9.5, 4.5))
        self.assertTrue(sneak)

    def test_removed_lower_target_is_used_as_safe_collection_stand(self):
        policy = self.policy()
        policy._world = Mock()
        env = Env(x=4.5, y=84.0, z=4.5)
        target = SimpleNamespace(cell=(4, 83, 3))

        with patch.object(policy, "_live_world", return_value=Mock()), \
                patch(
                    "bedrock_rl.sdg.policy.progression.policy."
                    "is_surface_stand",
                    side_effect=lambda _world, cell, **_: cell == target.cell):
            point, sneak = policy.drop_pickup_site(env, target)

        self.assertEqual(point, (4.5, 3.5))
        self.assertFalse(sneak)

    def test_live_safe_drop_cell_may_relax_planning_buffer_only(self):
        policy = self.policy()
        policy._world = Mock()
        env = Env(x=3.518, y=79.0, z=3.196)
        target = SimpleNamespace(cell=(2, 79, 3))

        def surface(_world, cell, *, hazard_clearance, **_):
            self.assertEqual(cell, target.cell)
            return hazard_clearance == 0

        with patch.object(policy, "_live_world", return_value=Mock()), \
                patch(
                    "bedrock_rl.sdg.policy.progression.policy."
                    "is_surface_stand", side_effect=surface):
            point, sneak = policy.drop_pickup_site(env, target)

        self.assertEqual(point, (2.5, 3.5))
        self.assertFalse(sneak)

    def test_one_block_drop_column_descends_normally_to_proved_landing(self):
        policy = self.policy()
        policy._world = Mock()
        env = Env(x=3.518, y=79.0, z=3.196)
        target = SimpleNamespace(cell=(2, 79, 3))
        world = Mock()
        world.block.side_effect = lambda cell: (
            1 if tuple(cell) == (2, 77, 3) else 0)

        def surface(_world, cell, *, hazard_clearance, **_):
            return tuple(cell) == (2, 78, 3) and hazard_clearance == 0

        with patch.object(policy, "_live_world", return_value=world), \
                patch(
                    "bedrock_rl.sdg.policy.progression.policy."
                    "is_surface_stand", side_effect=surface):
            point, sneak = policy.drop_pickup_site(env, target)

        self.assertEqual(point, (2.5, 3.5))
        self.assertFalse(sneak)

    def test_blocked_log_column_routes_to_safe_ground_neighbor(self):
        policy = self.policy()
        policy._world = Mock()
        env = Env(x=9.5, y=93.0, z=9.5)
        target = SimpleNamespace(cell=(11, 91, 7))
        world = Mock()
        # Exact rejection geometry: dirt directly below the removed log, but
        # another trunk block keeps the drop column itself unstandable.
        world.block.side_effect = lambda cell: (
            3 if tuple(cell) == (11, 90, 7) else 0)
        current = (9, 93, 9)
        safe_ground = (10, 91, 7)
        first_descent = (9, 92, 8)

        def surface(_world, cell, *, hazard_clearance, **_):
            del hazard_clearance
            return tuple(cell) == safe_ground

        with patch.object(policy, "_live_world", return_value=world), \
                patch(
                    "bedrock_rl.sdg.policy.progression.policy."
                    "is_surface_stand", side_effect=surface), \
                patch(
                    "bedrock_rl.sdg.policy.progression.policy."
                    "plan_surface_path",
                    return_value=(current, first_descent, safe_ground)
                ) as planner:
            point, sneak = policy.drop_pickup_site(env, target)

        self.assertEqual(point, (9.5, 8.5))
        self.assertFalse(sneak)
        self.assertFalse(policy._drop_pickup_allow_jump)
        planner.assert_called_once()
        self.assertEqual(planner.call_args.args[1:],
                         (current, safe_ground))
        self.assertEqual(
            policy._drop_pickup_trace["candidate_levels"], (93, 91))

    def test_upper_log_drop_scans_to_planner_proved_ground_landing(self):
        policy = self.policy()
        policy._world = Mock()
        env = Env(x=-13.508, y=71.0, z=64.948)
        target = SimpleNamespace(
            cell=(-14, 75, 65), mining_stance=(-14, 71, 64))
        world = Mock()
        # Exact canary regression: lower trunk blocks were already removed,
        # so the item falls through y=74..71 onto natural ground at y=70.
        world.block.side_effect = lambda cell: (
            2 if tuple(cell) == (-14, 70, 65) else 0)
        current = (-14, 71, 64)
        landing = (-14, 71, 65)

        def surface(_world, cell, *, hazard_clearance, **_):
            del hazard_clearance
            return tuple(cell) in (current, landing)

        with patch.object(policy, "_live_world", return_value=world), \
                patch(
                    "bedrock_rl.sdg.policy.progression.policy."
                    "is_surface_stand", side_effect=surface), \
                patch(
                    "bedrock_rl.sdg.policy.progression.policy."
                    "plan_surface_path",
                    return_value=(current, landing)) as planner:
            point, sneak = policy.drop_pickup_site(env, target)

        self.assertEqual(point, (-13.5, 65.5))
        self.assertFalse(sneak)
        self.assertEqual(
            policy._drop_pickup_trace["landing"], landing)
        self.assertEqual(
            policy._drop_pickup_trace["supports"], [
                ((-14, 74, 65), 0),
                ((-14, 73, 65), 0),
                ((-14, 72, 65), 0),
                ((-14, 71, 65), 0),
                ((-14, 70, 65), 2),
            ])
        planner.assert_called_once_with(
            world, current, landing,
            hazard_clearance=policy.spec.hazard_clearance,
            node_cap=policy.spec.node_cap)

    def test_distant_ledge_is_skipped_for_pickup_reachable_neighbor(self):
        policy = self.policy()
        policy._world = Mock()
        env = Env(x=4.5, y=79.0, z=2.5)
        target = SimpleNamespace(cell=(2, 79, 2))
        current = (4, 79, 2)
        near = (3, 79, 2)

        def surface(_world, cell, **_):
            return cell in (current, near)

        def route(_world, start, goal, **_):
            if goal == current:
                return (current,)
            if goal == near:
                return (current, near)
            raise AssertionError(goal)

        with patch.object(policy, "_live_world", return_value=Mock()), \
                patch(
                    "bedrock_rl.sdg.policy.progression.policy."
                    "is_surface_stand", side_effect=surface), \
                patch(
                    "bedrock_rl.sdg.policy.progression.policy."
                    "plan_surface_path", side_effect=route):
            point, sneak = policy.drop_pickup_site(env, target)

        self.assertEqual(point, (3.5, 2.5))
        self.assertFalse(sneak)

    def test_repeat_drop_reuses_one_planned_cell_with_bounded_live_steps(self):
        policy = self.policy()
        target = SimpleNamespace(
            cell=(1, 64, 0), mining_stance=(0, 64, 0))
        policy.plan = SimpleNamespace(resources=(SimpleNamespace(
            name="gravel", clusters=(SimpleNamespace(
                targets=(target,)),)),))
        operation = RepeatDropOperation(
            "gravel", "gravel", "flint", "gravel", max_attempts=3)
        env = Env(hotbar_ids=[13] + [0] * 8,
                  hotbar_counts=[1] + [0] * 8)
        policy._place_repeated = Mock(return_value=[{
            "action": "right_click"}])

        place = policy._repeat_drop_actions(env, operation)

        self.assertEqual(place[0]["action"], "right_click")
        policy._place_repeated.assert_called_once_with(
            env, operation, (1, 64, 0), (1, 63, 0), (0, 64, 0))

        start = policy._repeat_states[operation]["event_start"]
        self.assertEqual(start, 0)
        env.journal.append(Event(
            "block_placed", block=13, x=1, y=64, z=0))
        policy._mine_repeated = Mock(return_value=[{
            "action": "left_click_hold", "ticks": 26}])
        mine = policy._repeat_drop_actions(env, operation)
        self.assertEqual(mine[0]["action"], "left_click_hold")

        env.obs["hotbar_ids"][0] = 318
        self.assertIsNone(policy._repeat_drop_actions(env, operation))

    def test_repeat_drop_observes_selection_before_irreversible_use(self):
        policy = self.policy()
        operation = RepeatDropOperation(
            "gravel", "gravel", "flint", "gravel")
        env = Env(
            hotbar_ids=[0, 13] + [0] * 7,
            hotbar_counts=[0, 1] + [0] * 7)

        selecting = policy._place_repeated(
            env, operation, (1, 64, 0), (1, 63, 0), (0, 64, 0))

        self.assertTrue(selecting)
        self.assertFalse(any(action.get("action") == "right_click"
                             for action in selecting))
        env.obs["hotbar_sel"] = 1
        env.journal.append(Event(
            "click_target", block=1, x=1, y=63, z=0, face=4))
        self.assertEqual(policy._place_repeated(
            env, operation, (1, 64, 0), (1, 63, 0), (0, 64, 0)), [
                {"action": "right_click"},
            ])

    def test_repeat_drop_uses_live_pickup_geometry_after_break(self):
        policy = self.policy()
        target = SimpleNamespace(
            cell=(1, 64, 0), mining_stance=(0, 64, 0))
        policy.plan = SimpleNamespace(resources=(SimpleNamespace(
            name="gravel", clusters=(SimpleNamespace(
                targets=(target,)),)),))
        operation = RepeatDropOperation(
            "gravel", "gravel", "flint", "gravel")
        env = Env(journal=[Event(
            "block_broken", block=13, drop=13, x=1, y=64, z=0)])
        policy._repeat_states[operation] = {
            "event_start": 0, "events_seen": 0,
            "pickup_attempts": 0, "place_attempts": 0,
            "mine_attempts": 0,
        }
        policy.drop_pickup_site = Mock(return_value=((1.5, 0.5), False))
        policy.pickup = Mock(return_value=[{"action": "wait", "ticks": 1}])

        result = policy._repeat_drop_actions(env, operation)

        self.assertEqual(result, [{"action": "wait", "ticks": 1}])
        policy.drop_pickup_site.assert_called_once_with(env, target)
        self.assertEqual(policy.pickup.call_args.kwargs["target"], (1.5, 0.5))

    def test_reset_plans_each_stage_from_prior_endpoint_and_overlay(self):
        stages = (
            ResourceStage(
                "wood", ("log",), 1, "log", route_kinds=("surface",)),
            ResourceStage(
                "stone", ("stone",), 1, "cobblestone",
                route_kinds=("tunnel",)),
        )
        world = Mock()
        world.count_blocks_many.return_value = {"wood": 1, "stone": 1}
        world_factory = Mock(return_value=world)
        calls = []

        def target(cell, stance, mine_cell):
            return SimpleNamespace(
                cell=cell,
                mining_stance=stance,
                route=SimpleNamespace(
                    kind=("surface" if cell[1] == 64 else "tunnel"),
                    mine_cells=(mine_cell,), stands=(stance,),
                    support_cells=()),
                removal=SimpleNamespace(protected_support_cells=()),
            )

        targets = {
            "wood": target((3, 64, 1), (3, 64, 0), (2, 65, 0)),
            "stone": target((7, 62, 1), (7, 62, 0), (6, 63, 0)),
        }

        def plan_builder(_world, goal, **kwargs):
            name = goal.resources[0].name
            calls.append((name, {
                **kwargs,
                "opened": set(kwargs["opened"]),
                "removed": set(kwargs["removed"]),
                "placed": dict(kwargs["placed"]),
            }))
            resource = SimpleNamespace(
                name=name,
                clusters=(SimpleNamespace(targets=(targets[name],)),))
            plan = SimpleNamespace(
                resources=(resource,),
                to_dict=lambda name=name: {"resource": name},
            )
            return SimpleNamespace(build=lambda: plan)

        policy = SurvivalProgressionPolicy(
            spec=SurvivalProgressionSpec(
                stages=stages, portal=PortalSettings(enabled=False)),
            world_factory=world_factory,
            plan_builder=plan_builder,
        )
        env = Env(x=0.5, y=64.0, z=0.5)
        env.snapshot_in = "/tmp/unit-test.bsnp"
        episode = SimpleNamespace(env=env)

        with patch(
                "bedrock_rl.sdg.policy.progression.policy."
                "_workstation_sites", return_value=({}, None)):
            policy.reset(None, episode, Case(0, 17, 23, {}))

        self.assertEqual([name for name, _ in calls], ["wood", "stone"])
        self.assertEqual(calls[0][1]["start"], (0, 64, 0))
        self.assertEqual(calls[1][1]["start"], (3, 64, 0))
        self.assertEqual(calls[1][1]["opened"], {(2, 65, 0)})
        self.assertEqual(calls[1][1]["removed"], {(3, 64, 1)})
        self.assertEqual(
            [item["resource"] for item in policy.provenance()["stage_plans"]],
            ["wood", "stone"],
        )
        self.assertEqual(policy._allowed_break_cells, frozenset({
            (3, 64, 1), (2, 65, 0),
            (7, 62, 1), (6, 63, 0),
        }))
        self.assertIsNone(policy.plan)
        world.close.assert_not_called()
        policy.close()
        world.close.assert_called_once_with()

    def test_later_stages_preserve_prior_workstation_geometry(self):
        stages = (
            ResourceStage(
                "wood", ("log",), 1, "log", route_kinds=("surface",),
                after=(WorkstationOperation(
                    "bench", "crafting_table", "crafting_table",
                    "crafting_table"),)),
            ResourceStage(
                "stone", ("stone",), 1, "cobblestone",
                route_kinds=("tunnel",)),
        )
        world = Mock()
        world.count_blocks_many.return_value = {"wood": 1, "stone": 1}
        calls = []

        def target(name):
            cell = (3, 64, 1) if name == "wood" else (7, 62, 1)
            stance = (3, 64, 0) if name == "wood" else (7, 62, 0)
            return SimpleNamespace(
                cell=cell, mining_stance=stance,
                route=SimpleNamespace(
                    kind="surface" if name == "wood" else "tunnel",
                    mine_cells=(), stands=(stance,), support_cells=()),
                removal=SimpleNamespace(protected_support_cells=()))

        def plan_builder(_world, goal, **kwargs):
            name = goal.resources[0].name
            calls.append((name, set(kwargs["protected"])))
            planned = target(name)
            plan = SimpleNamespace(
                resources=(SimpleNamespace(
                    name=name, clusters=(SimpleNamespace(
                        targets=(planned,)),)),),
                to_dict=lambda name=name: {"resource": name})
            return SimpleNamespace(build=lambda: plan)

        site = WorkstationSite(
            "bench", (4, 64, 0), (4, 63, 0), (3, 64, 0),
            light_cells=((4, 64, 1),))
        placements = iter((({"bench": site}, None), ({}, None)))
        policy = SurvivalProgressionPolicy(
            spec=SurvivalProgressionSpec(stages=stages, portal=False),
            world_factory=Mock(return_value=world),
            plan_builder=plan_builder)
        env = Env(x=0.5, y=64.0, z=0.5)
        env.snapshot_in = "/tmp/unit-test.bsnp"

        with patch(
                "bedrock_rl.sdg.policy.progression.policy."
                "_workstation_sites", side_effect=lambda *_args, **_kwargs:
                next(placements)):
            policy.reset(None, SimpleNamespace(env=env), Case(0, 17, 23, {}))

        self.assertEqual(calls[0], ("wood", set()))
        self.assertTrue({site.cell, site.support}.issubset(calls[1][1]))
        self.assertTrue(set(site.light_cells).isdisjoint(calls[1][1]))

    def test_cast_preflight_uses_final_overlay_and_portal_return_starts_at_cast(self):
        stage = ResourceStage(
            "ore", ("stone",), 1, "cobblestone",
            tool="diamond_pickaxe", route_kinds=("tunnel",),
            after=(BucketCastMineOperation(
                "obsidian", 1, "diamond_pickaxe"),))
        world = Mock(seed=7, snapshot_sha256="sha", player={})
        world.count_blocks_many.return_value = {"ore": 1}
        portal = SimpleNamespace(
            occupied_cells=(), open_cells=(), temporary_cells=(),
            approach_stands=((9, 64, 9),), work_stance=(9, 64, 9),
            interaction_stance=(9, 64, 9), exit_stance=(9, 64, 9),
            entry_stance=(9, 64, 9), placements=(),
            to_dict=lambda: {"schema": "portal"})
        target = SimpleNamespace(
            cell=(2, 64, 0), mining_stance=(1, 64, 0),
            route=SimpleNamespace(
                kind="tunnel", mine_cells=((1, 65, 0),),
                stands=((0, 64, 0), (1, 64, 0)), support_cells=()),
            removal=SimpleNamespace(protected_support_cells=()))
        plan = SimpleNamespace(
            resources=(SimpleNamespace(
                name="ore", clusters=(SimpleNamespace(
                    targets=(target,)),)),),
            fluid_seal_cells=(),
            to_dict=lambda: {"resource": "ore"})
        cast = SimpleNamespace(
            casts_required=1, approach_stands=(
                (1, 64, 0), (2, 64, 0), (3, 64, 0), (4, 64, 0)),
            work_stance=(4, 64, 0), access_clear_cells=(),
            excavation_cells=((5, 64, 0),),
            final_open_cells=((5, 64, 0),), result_cell=(5, 64, 0),
            support_cells=(), backing_cells=(), temporary_cells=(),
            setup_placements=(), interactions=(SimpleNamespace(
                ray_cells=((4, 65, 0), target.cell, (5, 64, 0))),),
            to_dict=lambda: {"schema": "bedrock.fluid_cast_plan.v3"})
        outbound = RoutePlan(
            "tunnel", ((4, 64, 0), (5, 64, 0), (6, 64, 0)), (), (),
            ((6, 65, 0),))
        returned = RoutePlan(
            "tunnel", ((6, 64, 0), (5, 64, 0), (4, 64, 0)), (), (),
            ((7, 65, 0),))
        source_step = SimpleNamespace(
            kind="water", source_cell=(6, 64, 1),
            outbound=outbound, return_route=returned)
        sources = SimpleNamespace(
            steps=(source_step,),
            newly_opened_cells=((6, 64, 1), (6, 65, 0), (7, 65, 0)),
            to_dict=lambda: {"schema": "bedrock.fluid_source_plan.v1"})
        cast_selector = Mock(return_value=cast)
        source_planner = Mock(return_value=sources)

        policy = SurvivalProgressionPolicy(
            spec=SurvivalProgressionSpec(
                stages=(stage,), portal=PortalSettings(
                    enabled=True, method="bucket_cast_mined")),
            world_factory=Mock(return_value=world),
            plan_builder=lambda *_args, **_kwargs: SimpleNamespace(
                build=lambda: plan),
            portal_selector=Mock(return_value=portal),
            cast_selector=cast_selector,
            fluid_source_planner=source_planner,
        )
        env = Env()
        env.snapshot_in = "/tmp/unit-test.bsnp"
        episode = SimpleNamespace(env=env)
        portal_return = ((4, 64, 0), (9, 64, 9))

        with patch(
                "bedrock_rl.sdg.policy.progression.policy."
                "_workstation_sites", return_value=({}, None)), patch.object(
                    policy, "_portal_return_plan",
                    return_value=(portal_return,
                                  frozenset({(8, 65, 8)}))) as planner, \
                patch.object(policy, "_build_cost_certificate",
                             return_value={"tools": [], "lighting": {
                                 "required": 0, "within_budget": True}}), \
                patch.object(policy, "_require_tool_durability_budget"), \
                patch.object(policy, "_require_light_budget"):
            policy.reset(None, episode, Case(0, 7, 11, {}))

        self.assertEqual(cast_selector.call_args.kwargs["start"],
                         target.mining_stance)
        changed = cast_selector.call_args.args[0]
        self.assertEqual(changed.block(target.cell), 0)
        self.assertEqual(changed.block((1, 65, 0)), 0)
        self.assertIs(source_planner.call_args.args[0], world)
        self.assertEqual(source_planner.call_args.kwargs["removed"],
                         {target.cell})
        planner.assert_called_once_with(cast.work_stance)
        self.assertEqual(
            policy._portal_return_certificate["start"],
            list(cast.work_stance))
        self.assertTrue({
            source_step.source_cell, *outbound.mine_cells,
            *returned.mine_cells, (8, 65, 8), cast.result_cell,
        }.issubset(policy._allowed_break_cells))
        provenance = policy.provenance()
        self.assertEqual(provenance["fluid_cast_plan"]["schema"],
                         "bedrock.fluid_cast_plan.v3")
        self.assertEqual(provenance["fluid_source_plan"]["schema"],
                         "bedrock.fluid_source_plan.v1")
        self.assertEqual(set(provenance["preflight_timing_seconds"]), {
            "resource_counts", "portal_site", "resource_stage:ore",
            "fluid_cast_site", "fluid_sources", "portal_return",
            "cost_certificate",
        })
        self.assertTrue(all(
            value >= 0 for value in
            provenance["preflight_timing_seconds"].values()))

    def test_bucket_cast_operation_delegates_once_without_fallthrough(self):
        policy = self.policy()
        policy.cast_plan = SimpleNamespace(result_cell=(2, 64, 0))
        policy.fluid_source_plan = SimpleNamespace()
        policy.item_count = Mock(side_effect=lambda _env, item: (
            1 if item == "bucket" else 0))
        executor = Mock(done=False)
        executor.actions.return_value = [{"action": "wait", "ticks": 1}]
        policy._fluid_cast_executor = Mock(return_value=executor)
        operation = BucketCastMineOperation(
            "obsidian", 1, "diamond_pickaxe")

        first = policy._operation_actions(Env(), operation)
        second = policy._operation_actions(Env(), operation)

        self.assertEqual(first, [{"action": "wait", "ticks": 1}])
        self.assertEqual(second, first)
        policy._fluid_cast_executor.assert_called_once()
        self.assertIs(policy._cast_executor, executor)

    def test_active_bucket_cast_finishes_cleanup_after_inventory_goal(self):
        policy = self.policy()
        policy.cast_plan = SimpleNamespace(result_cell=(2, 64, 0))
        policy.item_count = Mock(return_value=10)
        executor = Mock(done=False)
        executor.actions.return_value = [{"action": "wait", "ticks": 1}]
        policy._cast_executor = executor
        operation = BucketCastMineOperation(
            "obsidian", 10, "diamond_pickaxe")

        result = policy._bucket_cast_actions(Env(), operation)

        self.assertEqual(result, [{"action": "wait", "ticks": 1}])
        executor.actions.assert_called_once()

    def test_fluid_geometry_is_protected_from_route_lighting(self):
        policy = self.policy()
        policy.plan = None
        policy.cast_plan = SimpleNamespace(
            protected_cells=((2, 64, 0), (2, 65, 0)))
        route = RoutePlan(
            "tunnel", ((0, 64, 0), (1, 64, 0)),
            ((1, 64, 0), (1, 65, 0)), ((1, 63, 0),),
            ((1, 65, 0),))
        policy.fluid_source_plan = SimpleNamespace(steps=(SimpleNamespace(
            source_cell=(3, 63, 0),
            pickup_ray_cells=((3, 64, 0), (3, 63, 0)),
            outbound=route, return_route=route,
        ),))

        protected = policy.route_light_protected_cells()

        self.assertTrue({
            (2, 64, 0), (2, 65, 0),
            (3, 63, 0), (3, 64, 0),
            *route.reserved_cells, *route.support_cells, *route.mine_cells,
        }.issubset(protected))
        forbidden_supports = policy.route_light_forbidden_support_cells()
        self.assertTrue({
            (2, 64, 0), (2, 65, 0), (3, 63, 0),
            *route.mine_cells,
        }.issubset(forbidden_supports))
        self.assertFalse(set(route.support_cells) & forbidden_supports)

    def test_portal_return_falls_back_when_live_origin_differs(self):
        policy = self.policy()
        policy._world = Mock(seed=1, snapshot_sha256="sha", player={})
        policy._opened = {(1, 64, 0)}
        policy._removed = {(2, 64, 0)}
        policy._placed = {(3, 64, 0): 58}
        policy._portal_cells = frozenset()
        policy._decision_seed = 23
        policy.portal_plan = SimpleNamespace(
            approach_stands=((9, 64, 9),))
        fresh_route = ((6, 64, 5), (7, 64, 6), (9, 64, 9))
        env = Env(x=6.5, y=64.0, z=5.5)

        with patch(
                "bedrock_rl.sdg.policy.progression.policy."
                "plan_surface_path", return_value=fresh_route) as planner:
            policy._plan_portal_return(env)

        args = planner.call_args.args
        self.assertEqual(args[1:], ((6, 64, 5), (9, 64, 9)))
        self.assertEqual(policy._route, fresh_route)
        self.assertEqual(policy._route_mine_cells, frozenset())

    def test_portal_return_reuses_matching_preflight_certificate(self):
        policy = self.policy()
        policy._allowed_break_cells = frozenset({(1, 64, 0)})
        policy._portal_return_certificate = {
            "start": [6, 64, 5], "goal": [9, 64, 9],
            "stands": [[6, 64, 5], [7, 64, 5], [8, 64, 5]],
            "mine_cells": [[7, 65, 5]],
        }

        with patch.object(policy, "_portal_return_plan") as planner:
            policy._plan_portal_return(Env(x=6.5, y=64.0, z=5.5))

        planner.assert_not_called()
        self.assertEqual(policy._route, (
            (6, 64, 5), (7, 64, 5), (8, 64, 5)))
        self.assertEqual(policy._route_mine_cells, {(7, 65, 5)})
        self.assertIn((7, 65, 5), policy._allowed_break_cells)

    def test_breadcrumb_reverses_complete_progression_spine(self):
        class World:
            @staticmethod
            def in_bounds(*_cell):
                return True

            @staticmethod
            def block(*coordinates):
                cell = coordinates[0] if len(coordinates) == 1 \
                    else coordinates
                return "stone" if cell[1] == 63 else "air"

        stage = ResourceStage(
            "ore", ("stone",), 1, "cobblestone",
            route_kinds=("surface",))
        target = SimpleNamespace(
            route=RoutePlan.surface(((0, 64, 0), (1, 64, 0))))
        plan = SimpleNamespace(resources=(SimpleNamespace(
            name="ore", clusters=(SimpleNamespace(
                targets=(target,)),)),))
        policy = self.policy()
        policy.spec = SurvivalProgressionSpec(stages=(stage,))
        policy._stage_plan_objects = [plan]
        policy.cast_plan = SimpleNamespace(
            approach_stands=((1, 64, 0), (2, 64, 0)))
        policy._return_breadcrumb = _ReturnBreadcrumb((0, 64, 0))
        policy._return_breadcrumb.extend(target.route.stands)
        policy._return_breadcrumb.extend(policy.cast_plan.approach_stands)

        route = policy._portal_breadcrumb_return(
            World(), (2, 64, 0), (0, 64, 0))

        self.assertEqual(route, (
            (2, 64, 0), (1, 64, 0), (0, 64, 0)))

    def test_breadcrumb_rejects_final_overlay_with_missing_support(self):
        class World:
            @staticmethod
            def in_bounds(*_cell):
                return True

            @staticmethod
            def block(*coordinates):
                cell = coordinates[0] if len(coordinates) == 1 \
                    else coordinates
                if cell == (1, 63, 0):
                    return "air"
                return "stone" if cell[1] == 63 else "air"

        stage = ResourceStage(
            "ore", ("stone",), 1, "cobblestone",
            route_kinds=("surface",))
        target = SimpleNamespace(
            route=RoutePlan.surface(((0, 64, 0), (1, 64, 0))))
        plan = SimpleNamespace(resources=(SimpleNamespace(
            name="ore", clusters=(SimpleNamespace(
                targets=(target,)),)),))
        policy = self.policy()
        policy.spec = SurvivalProgressionSpec(stages=(stage,))
        policy._stage_plan_objects = [plan]
        policy.cast_plan = SimpleNamespace(
            approach_stands=((1, 64, 0), (2, 64, 0)))
        policy._return_breadcrumb = _ReturnBreadcrumb((0, 64, 0))
        policy._return_breadcrumb.extend(target.route.stands)
        policy._return_breadcrumb.extend(policy.cast_plan.approach_stands)

        self.assertIsNone(policy._portal_breadcrumb_return(
            World(), (2, 64, 0), (0, 64, 0)))

    def test_return_clear_includes_ascending_transition_headroom(self):
        policy = self.policy()
        policy._route = ((0, 64, 0), (1, 65, 0))
        policy._route_index = 1
        policy._route_mine_cells = frozenset({(0, 66, 0)})
        policy._route_clear_attempts = {}
        policy.mine_exact = Mock(return_value=[{"action": "attack"}])

        result = policy._clear_return_leg(Env())

        self.assertEqual(result, [{"action": "attack"}])
        self.assertEqual(policy.mine_exact.call_args.args[1], (0, 66, 0))

    def test_fresh_portal_return_plan_declares_its_break_cells(self):
        policy = self.policy()
        policy.portal_plan = SimpleNamespace(
            approach_stands=((9, 64, 9),))
        policy._allowed_break_cells = frozenset({(1, 64, 0)})
        route = ((6, 64, 5), (7, 64, 6))
        return_cells = frozenset({(7, 64, 6), (7, 65, 6)})

        with patch.object(
                policy, "_portal_return_plan",
                return_value=(route, return_cells)):
            policy._plan_portal_return(Env(x=6.5, y=64.0, z=5.5))

        self.assertEqual(policy._route_mine_cells, return_cells)
        self.assertEqual(policy._allowed_break_cells, frozenset({
            (1, 64, 0), *return_cells,
        }))


if __name__ == "__main__":
    unittest.main()
