import tempfile
import unittest
from pathlib import Path

import yaml

from bedrock_rl.config import training as config
from bedrock_rl.cli.main import task_yaml
from bedrock_rl.resources import task_files
from bedrock_rl.sdg.generation import GenerationSpec


ROOT = Path(__file__).resolve().parents[1]
FARM = ROOT / "examples" / "farm_one_log_sdg"


class ConfigFilenameTests(unittest.TestCase):
    def test_bare_names_resolve_only_in_complete_examples(self):
        self.assertEqual(
            Path(task_yaml("equip_pickaxe_grpo")),
            ROOT / "examples" / "equip_pickaxe_grpo" / "task.yaml",
        )
        self.assertEqual(
            {Path(path).relative_to(ROOT).as_posix() for path in task_files()},
            {
                "examples/equip_pickaxe_grpo/task.yaml",
                "examples/farm_one_log_sdg/task.yaml",
            },
        )

    def test_yaml_roles_do_not_require_conventional_filenames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_path = root / "harvest_a_natural_log.yml"
            data_path = root / "make_my_wood_trajectories.yaml"
            train_path = root / "fine_tune_the_wood_policy.yaml"

            task_path.write_text((FARM / "task.yaml").read_text())

            data = yaml.safe_load((FARM / "data.yaml").read_text())
            data["generator"]["task"] = task_path.name
            data_path.write_text(yaml.safe_dump(data, sort_keys=False))

            spec = GenerationSpec.load(data_path)
            generator = spec.generator.create(
                base_dir=spec.base_dir, metadata=spec.metadata)
            self.assertEqual(generator.task_path, task_path.resolve())

            train_path.write_text(yaml.safe_dump({
                "trainer": "rl",
                "task": task_path.name,
                "model": "qwen3-vl-2b",
                "steps": 1,
                "algorithm": "rloo",
            }, sort_keys=False))
            plan = config.train_plan(
                config.load(train_path),
                resolve=lambda value: str((root / value).resolve()),
                origin=str(train_path),
            )
            self.assertEqual(plan.task, task_path.name)
            self.assertEqual(
                Path(task_yaml(plan.task, base_dir=root)).resolve(),
                task_path.resolve(),
            )

    def test_readme_calls_names_conventions_and_leads_with_sdg(self):
        readme = (ROOT / "README.md").read_text()
        self.assertIn("ordinary filenames, not schemas or registry entries",
                      readme)
        self.assertLess(
            readme.index("## Modular synthetic data generation"),
            readme.index("## Example training run"),
        )


if __name__ == "__main__":
    unittest.main()
