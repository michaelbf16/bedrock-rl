import types
import unittest

from bedrock_rl.env.task import Check


class AtLeast:
    def __init__(self, field, value):
        self.field = field
        self.value = value

    def holds(self, env, start_observation):
        del start_observation
        return env.obs[self.field] >= self.value


def changed(env, start_observation):
    return env.obs["value"] != start_observation["value"]


class CustomCheckTests(unittest.TestCase):
    def test_configured_object_reads_live_state(self):
        check = Check({
            "type": f"{__name__}:AtLeast",
            "config": {"field": "health", "value": 10},
            "reward": 2,
        })
        env = types.SimpleNamespace(obs={"health": 12})
        self.assertTrue(check.holds(env, {}))
        self.assertEqual(check.reward, 2.0)

    def test_unconfigured_function_is_the_predicate(self):
        check = Check({"type": f"{__name__}:changed"})
        env = types.SimpleNamespace(obs={"value": 2})
        self.assertTrue(check.holds(env, {"value": 1}))

    def test_custom_check_rejects_ambiguous_top_level_settings(self):
        with self.assertRaisesRegex(ValueError, "unknown custom check key"):
            Check({"type": f"{__name__}:AtLeast", "field": "health"})


if __name__ == "__main__":
    unittest.main()
