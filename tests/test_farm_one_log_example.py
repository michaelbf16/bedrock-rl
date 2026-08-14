import unittest
from pathlib import Path

import yaml

from bedrock_rl.sdg.generation import GenerationSpec
from bedrock_rl.env.task import Task


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "farm_one_log_sdg"


class FarmOneLogExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = yaml.safe_load((EXAMPLE / "data.yaml").read_text())
        cls.spec = GenerationSpec.load(EXAMPLE / "data.yaml")
        cls.task = Task.load(EXAMPLE / "task.yaml")

    def test_task_requires_a_newly_collected_log(self):
        self.assertEqual(self.task.success.type, "item_gained")
        self.assertEqual(
            self.task.success.spec,
            {"type": "item_gained", "item": "log", "count": 1,
             "reward": 1.0},
        )
        constraint = self.task.initial_state.constraints[0]
        self.assertEqual(
            (constraint.block, constraint.min_distance,
             constraint.max_distance),
            ("log", 4.0, 12.0),
        )

    def test_manifest_is_generic_and_parallel(self):
        self.assertEqual(self.spec.worlds, 100)
        self.assertEqual(self.spec.max_world_attempts, 3000)
        self.assertEqual(
            self.spec.sampler.type,
            "bedrock_rl.sdg:RandomCases",
        )
        self.assertEqual(
            self.raw["generator"]["policy"]["type"],
            "bedrock_rl.sdg.policy.planned_mine:"
            "PlannedMinePolicy",
        )
        self.assertEqual(
            self.spec.executor.type,
            "bedrock_rl.sdg:ProcessExecutor",
        )
        self.assertEqual(self.spec.executor.config["workers"], "auto")
        self.assertTrue(self.spec.rejections_output.endswith(
            "datasets/farm_one_log.rejections.jsonl"))

    def test_readme_shows_the_modular_example_and_verified_gif(self):
        readme = (ROOT / "README.md").read_text()
        self.assertIn("Modular synthetic data generation", readme)
        self.assertIn("examples/farm_one_log_sdg/data.yaml", readme)
        self.assertIn("docs/assets/farm_one_log.views.gif", readme)
        self.assertNotIn("## Synthetic survival dataset target", readme)
        gif = ROOT / "docs" / "assets" / "farm_one_log.views.gif"
        self.assertGreater(gif.stat().st_size, 1_000_000)


if __name__ == "__main__":
    unittest.main()
