import re
from pathlib import Path
import tomllib
import unittest
from unittest.mock import patch

from bedrock_rl.cli import main as cli


ROOT = Path(__file__).resolve().parents[1]
SETUP = (ROOT / "bin" / "setup_deps.sh").read_text()
README = (ROOT / "README.md").read_text()


class OnboardingContractTests(unittest.TestCase):
    def test_release_version_metadata_agrees(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(project["project"]["version"], "0.1.0")
        with patch("importlib.metadata.version", return_value="0.1.0"):
            self.assertEqual(cli._version(), "v0.1.0")

    def test_runtime_only_setup_stops_before_the_training_stack(self):
        guard = 'if [ "${BRL_RUNTIME_ONLY:-0}" = 1 ]; then'
        self.assertIn(guard, SETUP)
        self.assertLess(SETUP.index("if ! brl_write_manifest"),
                        SETUP.index(guard))
        self.assertLess(SETUP.index(guard),
                        SETUP.index("# ---- verl (trainer)"))
        block = SETUP[SETUP.index(guard):
                      SETUP.index("# ---- verl (trainer)")]
        self.assertIn("exit 0", block)

    def test_runtime_only_setup_is_forwarded_by_the_cli(self):
        parser, _groups = cli.build_parser()
        args = parser.parse_args(["setup", "--runtime-only"])
        with (patch.object(cli, "checkout_env", return_value={}),
              patch.object(cli, "script", return_value="setup_deps.sh"),
              patch.object(cli, "run", return_value=0) as run):
            self.assertEqual(cli.cmd_setup(args), 0)
        _argv, kwargs = run.call_args
        self.assertEqual(kwargs["env"]["BRL_RUNTIME_ONLY"], "1")

    def test_plain_setup_does_not_inherit_runtime_only(self):
        parser, _groups = cli.build_parser()
        args = parser.parse_args(["setup"])
        inherited = {"BRL_RUNTIME_ONLY": "1"}
        with (patch.object(cli, "checkout_env", return_value=inherited),
              patch.object(cli, "script", return_value="setup_deps.sh"),
              patch.object(cli, "run", return_value=0) as run):
            self.assertEqual(cli.cmd_setup(args), 0)
        _argv, kwargs = run.call_args
        self.assertNotIn("BRL_RUNTIME_ONLY", kwargs["env"])

    def test_runtime_only_rejects_training_only_options(self):
        for option in ("--python", "--no-flash-attn",
                       "--install-deepspeed"):
            argv = ["setup", "--runtime-only", option]
            if option == "--python":
                argv.append("3.12")
            parser, _groups = cli.build_parser()
            args = parser.parse_args(argv)
            with (patch.object(cli, "checkout_env", return_value={}),
                  self.assertRaisesRegex(SystemExit, option)):
                cli.cmd_setup(args)

    def test_public_clone_commands_use_the_uv_environment(self):
        paths = [ROOT / "README.md", *(ROOT / "docs").glob("*.md")]
        paths.extend((ROOT / "examples").glob("*/README.md"))
        bare = []
        pattern = re.compile(r"^\s*(?:netherite\b|python -m unittest\b)")
        for path in paths:
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if pattern.match(line):
                    bare.append(f"{path.relative_to(ROOT)}:{number}: {line}")
        self.assertEqual(bare, [])

    def test_quick_start_is_runtime_only_and_docker_roles_are_honest(self):
        self.assertIn("uv run bedrock setup --runtime-only", README)
        self.assertIn("uv run bedrock doctor --runtime-only", README)
        self.assertNotIn("CPU Docker target covers generation and evaluation",
                         README)
        self.assertIn("remote-API\n`bedrock trial`", README)

    def test_equip_slot_balance_names_worlds_and_rows(self):
        text = (ROOT / "examples" / "equip_pickaxe_grpo" /
                "README.md").read_text()
        self.assertIn("Among the 36 unique worlds", text)
        self.assertIn("every slot appears in 32 rows", text)


if __name__ == "__main__":
    unittest.main()
