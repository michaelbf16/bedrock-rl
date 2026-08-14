"""CLI contracts that must hold without a bedrock-rl source checkout."""

from __future__ import annotations

import io
import tempfile
import textwrap
import tomllib
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from bedrock_rl import resources
from bedrock_rl.cli import main as cli


TASK = """
name: wheel_task
goal: Equip the iron pickaxe.
seeds: [7]
max_turns: 4
success: {type: selected_item, items: [iron_pickaxe]}
"""

RUN = """
task:
  name: wheel_run
  instruction: Complete the Minecraft task.
  environment:
    type: bedrock_rl.adapters.netherite.environment:NetheriteEnvironment
    task_yaml: task.yaml
  verifier: bedrock_rl.adapters.netherite.reward:EpisodeReward
model:
  type: bedrock_rl.adapters.chat_completions:ChatCompletionsModel
  model: local-model
tools:
  - bedrock_rl.adapters.netherite.tools:NetheriteComputerTool
max_turns: 4
"""

DATA = """
count: 1
sampler: bedrock_rl.sdg:RandomCases
generator:
  type: bedrock_rl.sdg:TaskVariantGenerator
  template: task.yaml
executor: bedrock_rl.sdg:SerialExecutor
writer: bedrock_rl.sdg:JSONLWriter
output: output.jsonl
"""

TRAIN = """
trainer: rl
task: task.yaml
model: qwen3-vl-2b
steps: 2
algorithm: rloo
rollout_n: 2
"""


class InstalledWheelCLITests(unittest.TestCase):
    def test_core_install_declares_default_image_runtime(self):
        """A plain install can decode the default Minecraft view."""

        project = tomllib.loads(
            (Path(__file__).resolve().parents[1] / "pyproject.toml")
            .read_text())
        dependencies = project["project"]["dependencies"]
        self.assertTrue(any(
            dependency.lower().startswith("pillow")
            for dependency in dependencies
        ))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name, value in (("task.yaml", TASK), ("run.yaml", RUN),
                            ("data.yaml", DATA), ("train.yaml", TRAIN)):
            (self.root / name).write_text(textwrap.dedent(value).lstrip())

    def tearDown(self):
        self.tmp.cleanup()

    def invoke(self, argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with (patch.object(resources, "repo_root", return_value=None),
              redirect_stdout(stdout), redirect_stderr(stderr)):
            code = cli.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_registry_commands_do_not_need_a_checkout(self):
        self.assertEqual(self.invoke(["models", "list"])[0], 0)
        self.assertEqual(
            self.invoke(["models", "resolve", "qwen3-vl-2b"])[0], 0)

    def test_explicit_task_show_and_validate_do_not_need_catalog(self):
        task = str(self.root / "task.yaml")
        self.assertEqual(self.invoke(["task", "show", task])[0], 0)
        self.assertEqual(self.invoke(["task", "validate", task])[0], 0)

    def test_trial_check_resolves_package_components(self):
        code, _, error = self.invoke(
            ["trial", str(self.root / "run.yaml"), "--check"])
        self.assertEqual(code, 0, error)

    def test_generate_check_runs_without_checkout_environment(self):
        code, _, error = self.invoke(
            ["generate", str(self.root / "data.yaml"), "--check"])
        self.assertEqual(code, 0, error)

    def test_eval_summarize_is_a_public_cli_workflow(self):
        records = self.root / "base.records.json"
        summary = self.root / "curve.json"
        svg = self.root / "curve.svg"
        records.write_text("[]\n")
        with patch("bedrock_rl.eval.summary.main", return_value=0) as build:
            code, _, error = self.invoke([
                "eval", "summarize",
                "--point", "base", "0", str(records),
                "--summary", str(summary),
                "--svg", str(svg),
            ])
        self.assertEqual(code, 0, error)
        argv = build.call_args.args[0]
        self.assertIn(str(records), argv)
        self.assertIn(str(summary), argv)
        self.assertIn(str(svg), argv)

    def test_train_dry_run_resolves_config_but_execution_refuses(self):
        path = str(self.root / "train.yaml")
        with patch.object(cli, "run", return_value=0) as launched:
            code, _, error = self.invoke(["train", path, "--dry-run"])
        self.assertEqual(code, 0, error)
        argv = launched.call_args.args[0]
        self.assertEqual(argv[:2], ["bash", "bin/rl.sh"])
        self.assertIn(str(self.root / "task.yaml"), argv)

        code, output, error = self.invoke(["train", path])
        self.assertEqual(code, 1)
        self.assertIn("needs the bedrock-rl checkout", output + error)

    def test_doctor_reports_an_installed_package_instead_of_crashing(self):
        code, _, _ = self.invoke(["doctor", "--runtime-only"])
        # A fresh machine is expected to fail its engine checks.  The command
        # itself still ran and reported the wheel boundary.
        self.assertIn(code, (0, 1))
        from bedrock_rl.cli import doctor

        with patch.object(resources, "repo_root", return_value=None):
            check = doctor.repo_check()
        self.assertEqual(check.state, "warn")
        self.assertIn("installed from a wheel", check.detail)


if __name__ == "__main__":
    unittest.main()
