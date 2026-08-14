import io
import sys
import unittest
from unittest.mock import patch

from bedrock_rl.adapters.verl import console
from bedrock_rl.adapters.verl.console import parse_step


class VerlConsoleTests(unittest.TestCase):
    def test_parse_step_preserves_numpy_scalar_metrics(self):
        step, metrics = parse_step(
            "step:3 - actor/grad_norm:np.float64(0.003321) - "
            "num_turns:min:np.int32(3) - reward:-0.42"
        )
        self.assertEqual(step, 3)
        self.assertEqual(metrics, {
            "actor/grad_norm": 0.003321,
            "num_turns:min": 3.0,
            "reward": -0.42,
        })

    def test_console_owns_the_single_wandb_run(self):
        with (patch.object(console.ui_mod, "Reporter") as reporter,
              patch.object(sys, "stdin", io.StringIO(""))):
            console.main(["--run", "test-run", "--steps", "1"])

        self.assertNotIn("wandb", reporter.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
