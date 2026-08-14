import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from bedrock_rl.cli import main as cli


class TaskCommandDispatchTests(unittest.TestCase):
    def test_validate_checks_the_computer_prompt_against_task_budgets(self):
        task = SimpleNamespace(name="tiny", shaping=(), fail=None, seeds=(7,))
        args = SimpleNamespace(tasks=["tiny.yaml"])
        with (
            patch.object(cli, "task_yaml", return_value="/tiny.yaml"),
            patch.object(cli, "_load", return_value=task),
            patch(
                "bedrock_rl.adapters.netherite.computer.validate_cu_example"
            ) as validate,
        ):
            self.assertEqual(cli.cmd_task_validate(args), 0)

        validate.assert_called_once_with(task)

    def test_render_dispatches_to_the_packaged_module(self):
        args = SimpleNamespace(
            task="task.yaml",
            out="preview",
            play_oracle=True,
            dry_run=False,
        )
        with (
            patch.object(cli, "package_env", return_value={"X": "1"}),
            patch.object(cli, "task_yaml", return_value="/task.yaml"),
            patch.object(cli, "here", return_value="/preview"),
            patch.object(cli, "run", return_value=0) as run,
        ):
            self.assertEqual(cli.cmd_task_render(args), 0)

        argv = run.call_args.args[0]
        self.assertEqual(
            argv,
            [
                sys.executable,
                "-m",
                "bedrock_rl.cli.task_render",
                "/task.yaml",
                "/preview",
                "--play-oracle",
            ],
        )
        self.assertEqual(run.call_args.kwargs["env"], {"X": "1"})
        self.assertNotIn("cwd", run.call_args.kwargs)

    def test_snapshots_dispatches_to_the_packaged_module(self):
        args = SimpleNamespace(task="task.yaml", dry_run=False)
        with (
            patch.object(cli, "package_env", return_value={"X": "1"}),
            patch.object(cli, "task_yaml", return_value="/task.yaml"),
            patch.object(cli, "run", return_value=0) as run,
        ):
            self.assertEqual(cli.cmd_task_snapshots(args), 0)

        self.assertEqual(
            run.call_args.args[0],
            [
                sys.executable,
                "-m",
                "bedrock_rl.cli.task_snapshots",
                "--task",
                "/task.yaml",
            ],
        )
        self.assertEqual(run.call_args.kwargs["env"], {"X": "1"})
        self.assertNotIn("cwd", run.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
