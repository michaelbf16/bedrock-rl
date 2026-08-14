import unittest

from bedrock_rl.env.initial_state import InitialStateSpec


class InitialStateTests(unittest.TestCase):
    def test_absent_initial_state_does_not_create_a_snapshot(self):
        spec = InitialStateSpec.parse(None)
        self.assertFalse(spec.enabled)
        self.assertEqual(spec.world, {})

    def test_inventory_vitals_spawn_constraints_and_world_parse(self):
        spec = InitialStateSpec.parse({
            "inventory": [
                {"item": "iron_pickaxe", "count": 1,
                 "slot": "random_hotbar"},
                {"item": "torch", "count": 16, "slot": 8},
            ],
            "player": {"health": 7, "food": 12, "saturation": 2.5,
                       "selected_item": "iron_pickaxe"},
            "spawn": {
                "points": [{"x": 1, "y": 70, "z": -2}],
                "hazard_clearance": 1,
                "constraints": [{"block": "iron_ore",
                                 "min_distance": 2,
                                 "max_distance": 12}],
            },
            "world": {"mode": "survival", "difficulty": "normal",
                      "weather": "clear", "time": 1000, "mobs": False},
        })
        self.assertTrue(spec.enabled)
        self.assertEqual(spec.inventory_slots(5), spec.inventory_slots(5))
        self.assertEqual(spec.inventory_slots(5)[8][1], 16)
        self.assertEqual(spec.player["health"], 7.0)
        self.assertEqual(spec.player["selected_item"], "iron_pickaxe")
        self.assertEqual(spec.spawn_hazard_clearance, 1)
        self.assertEqual(spec.world["time"], 1000)

    def test_random_inventory_reserves_later_fixed_slots(self):
        spec = InitialStateSpec.parse({"inventory": [
            {"item": "dirt", "slot": "random_hotbar"},
            {"item": "torch", "slot": 4},
        ]})

        for seed in range(500):
            slots = spec.inventory_slots(seed)
            self.assertEqual(slots[4][1], 1)
            self.assertEqual(len(slots), 2)

    def test_spawn_hazard_clearance_is_a_nonnegative_integer(self):
        with self.assertRaisesRegex(TypeError, "must be an integer"):
            InitialStateSpec.parse({"spawn": {"hazard_clearance": True}})
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            InitialStateSpec.parse({"spawn": {"hazard_clearance": -1}})

    def test_unsupported_engine_world_settings_are_refused(self):
        for key, value in (("mode", "creative"), ("difficulty", "hard"),
                           ("weather", "rain")):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    InitialStateSpec.parse({"world": {key: value}})

    def test_world_booleans_are_not_coerced_from_strings(self):
        with self.assertRaisesRegex(TypeError, "mobs must be true or false"):
            InitialStateSpec.parse({"world": {"mobs": "false"}})
        with self.assertRaisesRegex(TypeError, "time must be an integer"):
                InitialStateSpec.parse({"world": {"time": True}})

    def test_ordered_block_volumes_support_single_cells_and_ranges(self):
        spec = InitialStateSpec.parse({"blocks": [
            {"block": "air", "x": [-2, 2], "y": [65, 68],
             "z": [-2, 2]},
            {"block": "diamond_ore", "x": 1, "y": 65, "z": 3},
        ]})
        self.assertTrue(spec.enabled)
        self.assertEqual(len(list(spec.blocks[0].cells())), 100)
        self.assertEqual(spec.blocks[1].block_id, 56)
        self.assertEqual(spec.to_dict()["blocks"][0]["x"], [-2, 2])

    def test_block_volume_validation_is_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown initial_state block"):
            InitialStateSpec.parse({"blocks": [
                {"block": "stone", "x": 0, "y": 1, "z": 2,
                 "ignored": True},
            ]})


if __name__ == "__main__":
    unittest.main()
