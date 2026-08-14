import unittest
from types import SimpleNamespace

from bedrock_rl.sdg.policy.oracle import expert_cu
from bedrock_rl.env.catalog import resolve_items
from bedrock_rl.env.task import Check


class ObsAtLeast:
    def __init__(self, field, value):
        self.field = field
        self.value = value

    def holds(self, env, start_obs):
        del start_obs
        return env.obs[self.field] >= self.value


class SelectedItemTests(unittest.TestCase):
    def setUp(self):
        self.iron = resolve_items(["iron_pickaxe"])[0]
        self.stone = resolve_items(["stone_pickaxe"])[0]
        self.check = Check({
            "type": "selected_item",
            "items": ["iron_pickaxe"],
        })
        self.task = SimpleNamespace(
            success=SimpleNamespace(
                type="selected_item", spec={"items": ["iron_pickaxe"]}))

    def env(self, selected):
        return SimpleNamespace(obs={
            "hotbar_sel": selected,
            "hotbar_ids": (self.stone, self.iron) + (0,) * 7,
        })

    def test_check_reads_the_live_selected_slot(self):
        self.assertFalse(self.check.holds(self.env(0), {}))
        self.assertTrue(self.check.holds(self.env(1), {}))

    def test_oracle_selects_the_matching_number_key(self):
        self.assertEqual(expert_cu(self.task, self.env(0)), [
            {"action": "key", "k": "2", "ticks": 1},
        ])
        self.assertEqual(expert_cu(self.task, self.env(1)), [
            {"action": "wait", "ticks": 1},
        ])

    def test_custom_check_is_a_configured_import_path(self):
        check = Check({
            "type": "tests.test_selected_item:ObsAtLeast",
            "config": {"field": "hotbar_sel", "value": 1},
            "reward": 0.75,
        })
        self.assertFalse(check.holds(self.env(0), {}))
        self.assertTrue(check.holds(self.env(1), {}))
        self.assertEqual(check.reward, 0.75)


if __name__ == "__main__":
    unittest.main()
