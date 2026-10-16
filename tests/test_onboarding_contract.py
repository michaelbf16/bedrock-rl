import importlib.util
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from bedrock_rl.cli import main as cli


ROOT = Path(__file__).resolve().parents[1]
SETUP = (ROOT / "bin" / "setup_deps.sh").read_text()
README = (ROOT / "README.md").read_text()
CI = (ROOT / ".github" / "workflows" / "ci.yml").read_text()


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

    def test_quick_start_installs_native_prerequisites_and_checks_pinned_uv(
            self):
        quick_start = README.split("## Quick start", 1)[1].split(
            "\n## ", 1)[0]
        ci_uv = re.search(r'UV_VERSION: "([^"]+)"', CI)
        self.assertIsNotNone(ci_uv)
        installer = (
            f"https://astral.sh/uv/{ci_uv.group(1)}/install.sh"
        )
        self.assertIn(installer, quick_start)
        self.assertNotIn("https://astral.sh/uv/install.sh", quick_start)
        self.assertIn(
            "if ! uv --version >/dev/null 2>&1; then", quick_start)
        self.assertNotIn("command -v uv", quick_start)
        recovery = quick_start.split(
            "if ! uv --version >/dev/null 2>&1; then", 1)[1].split(
                "\nfi", 1)[0]
        self.assertIn(installer, recovery)
        self.assertIn('export PATH="$HOME/.local/bin:$PATH"', recovery)
        after_recovery = quick_start.split("\nfi", 1)[1]
        self.assertIn("\nuv --version", after_recovery)
        self.assertLess(
            quick_start.index("\nuv --version", quick_start.index("\nfi")),
            quick_start.index("uv sync --extra netherite --extra data"),
        )
        for package in (
                "build-essential", "libsdl2-dev", "pkg-config", "xxd",
                "unzip"):
            with self.subTest(package=package):
                self.assertIn(package, quick_start)
                self.assertIn(package, CI)

    def test_every_documented_eval_summary_installs_plot_extra_beside_it(self):
        paths = [ROOT / "README.md", *(ROOT / "docs").glob("*.md")]
        paths.extend((ROOT / "examples").glob("*/README.md"))
        workflows = []
        for path in paths:
            for block in re.findall(
                    r"```bash\n(.*?)\n```", path.read_text(), re.DOTALL):
                if "bedrock eval summarize" not in block:
                    continue
                workflows.append(path.relative_to(ROOT))
                self.assertIn("uv sync --extra plot", block)
                self.assertLess(
                    block.index("uv sync --extra plot"),
                    block.index("bedrock eval summarize"),
                )
        self.assertEqual(
            workflows,
            [Path("examples/equip_pickaxe_grpo/README.md")],
        )

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib"),
        "the executable SVG contract runs in CI's plot-extra step",
    )
    def test_documented_eval_summary_runs_through_public_entrypoint(self):
        path = ROOT / "examples" / "equip_pickaxe_grpo" / "README.md"
        block = next(
            block for block in re.findall(
                r"```bash\n(.*?)\n```", path.read_text(), re.DOTALL)
            if "bedrock eval summarize" in block
        )
        documented = block[block.index("uv run bedrock"):]
        argv = shlex.split(documented.replace("\\\n", " "))
        self.assertEqual(argv[:3], ["uv", "run", "bedrock"])
        entrypoint = Path(sys.executable).with_name("bedrock")
        self.assertTrue(entrypoint.is_file(), entrypoint)
        outputs = (
            Path("/tmp/equip-pickaxe-dev-summary.json"),
            Path("/tmp/equip-pickaxe-dev-curve.svg"),
        )
        for output in outputs:
            output.unlink(missing_ok=True)
        try:
            result = subprocess.run(
                [str(entrypoint), *argv[3:]],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for output in outputs:
                self.assertTrue(output.is_file(), output)
        finally:
            for output in outputs:
                output.unlink(missing_ok=True)

    def test_generated_caches_leave_clean_repo_but_evidence_is_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory)
            subprocess.run(
                ["git", "init", "-q", str(checkout)],
                check=True,
            )
            shutil.copyfile(ROOT / ".gitignore", checkout / ".gitignore")
            (checkout / "tracked.txt").write_text("baseline\n")
            subprocess.run(
                ["git", "-C", str(checkout), "add", "."],
                check=True,
            )
            subprocess.run(
                [
                    "git", "-C", str(checkout),
                    "-c", "user.name=Bedrock Contract",
                    "-c", "user.email=contract@example.invalid",
                    "commit", "-qm", "clean baseline",
                ],
                check=True,
            )

            status = ["git", "-C", str(checkout), "status", "--porcelain",
                      "--untracked-files=all"]
            baseline = subprocess.run(
                status, check=True, capture_output=True, text=True,
            ).stdout
            self.assertEqual(baseline, "")
            generated = (
                Path("bedrock_rl/env/__pycache__/task.cpython-312.pyc"),
                Path("bedrock_rl/legacy.pyc"),
                Path("examples/demo/.bedrock/snapshots/demo/cache.bsnp"),
            )
            for relative in generated:
                path = checkout / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"generated")
                ignored = subprocess.run(
                    ["git", "-C", str(checkout), "check-ignore", "--quiet",
                     str(relative)],
                    check=False,
                )
                self.assertEqual(ignored.returncode, 0, relative)
            after = subprocess.run(
                status, check=True, capture_output=True, text=True,
            ).stdout
            self.assertEqual(after, "")

            evidence = Path("examples/demo/.bedrock/evidence.json")
            evidence_path = checkout / evidence
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_path.write_text("{}\n")
            ignored = subprocess.run(
                ["git", "-C", str(checkout), "check-ignore", "--quiet",
                 str(evidence)],
                check=False,
            )
            self.assertEqual(ignored.returncode, 1)
            visible = subprocess.run(
                status, check=True, capture_output=True, text=True,
            ).stdout
            self.assertEqual(visible, f"?? {evidence}\n")

    def test_equip_slot_balance_names_worlds_and_rows(self):
        text = (ROOT / "examples" / "equip_pickaxe_grpo" /
                "README.md").read_text()
        self.assertIn("Among the 36 unique worlds", text)
        self.assertIn("every slot appears in 32 rows", text)


if __name__ == "__main__":
    unittest.main()
