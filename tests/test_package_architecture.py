import ast
import os
from pathlib import Path
import subprocess
import sys
import unittest

import yaml

from bedrock_rl.core.components import load_object


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "bedrock_rl"


def package_imports(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return tuple(name for name in imports if name.startswith("bedrock_rl"))


class PackageArchitectureTests(unittest.TestCase):
    def assert_layer_avoids(self, layer, forbidden):
        violations = []
        for path in sorted((PACKAGE / layer).rglob("*.py")):
            for imported in package_imports(path):
                if imported.startswith(forbidden):
                    violations.append(
                        f"{path.relative_to(ROOT)} imports {imported}"
                    )
        self.assertEqual(violations, [])

    def test_core_has_no_environment_or_training_dependency(self):
        for forbidden in (
            "bedrock_rl.adapters",
            "bedrock_rl.env",
            "bedrock_rl.sdg",
            "bedrock_rl.train",
        ):
            self.assert_layer_avoids("core", forbidden)

    def test_environment_has_no_adapter_or_training_dependency(self):
        for forbidden in (
            "bedrock_rl.adapters",
            "bedrock_rl.sdg",
            "bedrock_rl.train",
        ):
            self.assert_layer_avoids("env", forbidden)

    def test_adapters_do_not_import_product_workflows(self):
        for forbidden in (
            "bedrock_rl.cli",
            "bedrock_rl.eval",
            "bedrock_rl.launch",
            "bedrock_rl.sdg",
            "bedrock_rl.train",
        ):
            self.assert_layer_avoids("adapters", forbidden)

    def test_sdg_does_not_import_front_ends_or_trainers(self):
        for forbidden in (
            "bedrock_rl.cli",
            "bedrock_rl.eval",
            "bedrock_rl.launch",
            "bedrock_rl.train",
        ):
            self.assert_layer_avoids("sdg", forbidden)

    def test_launchers_do_not_import_cli_implementation(self):
        self.assert_layer_avoids("launch", "bedrock_rl.cli")

    def test_every_shipped_yaml_component_path_resolves(self):
        references = set()

        def collect(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if (key == "type" and isinstance(child, str)
                            and ":" in child):
                        references.add(child)
                    collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)

        for directory in (ROOT / "examples",):
            for path in directory.rglob("*.yaml"):
                collect(yaml.safe_load(path.read_text()))
        failures = []
        for reference in sorted(references):
            try:
                load_object(reference)
            except Exception as exc:
                failures.append(f"{reference}: {exc}")
        self.assertEqual(failures, [])

    def test_root_import_is_lightweight(self):
        command = (
            "import sys; import bedrock_rl; "
            "assert 'torch' not in sys.modules; "
            "assert 'PIL' not in sys.modules; "
            "assert 'bedrock_rl.env.engine' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", command], cwd=ROOT, check=True)

    def test_package_root_contains_only_public_facades(self):
        modules = {path.name for path in PACKAGE.glob("*.py")}
        self.assertEqual(modules, {
            "__init__.py", "data.py", "lora.py", "models.py",
            "reporting.py", "resources.py",
        })

    def test_bin_contains_only_shell_launchers(self):
        files = {path.name for path in (ROOT / "bin").iterdir()
                 if path.is_file()}
        self.assertTrue(files)
        self.assertTrue(all(name.endswith(".sh") for name in files))
        self.assertTrue((ROOT / "tools" / "stub_atlases.py").is_file())

    def test_only_required_engine_patches_ship(self):
        patches = sorted(
            path.name for path in (ROOT / "patches" / "netherite").glob("*.diff")
        )
        self.assertEqual(patches, [
            "0001-rl-env.diff",
            "0002-event-journal.diff",
            "0003-vanilla-obsidian-speed.diff",
            "0004-complete-crafting.diff",
            "0005-safe-portal-arrival.diff",
            "0006-long-episode-drops.diff",
            "0007-sneak-use-placement.diff",
            "0008-route-contract-test.diff",
            "0009-ten-chunk-snapshots.diff",
            "0010-emerald-harvest.diff",
            "0011-redstone-harvest.diff",
            "0012-full-inventory-observation.diff",
        ])
        event_patch = (ROOT / "patches" / "netherite" /
                       "0002-event-journal.diff").read_text().lower()
        self.assertNotIn("warm reset", event_patch)
        self.assertNotIn("warm_reset", event_patch)
        self.assertIn("gm_raycast_bucket", event_patch)
        self.assertIn("gm_player_bucket_target_state", event_patch)
        self.assertIn("rlj_emit_flags(rlj_block_broken", event_patch)
        self.assertIn("gm_world_meta(r->world, edits[i].wx", event_patch)
        self.assertIn("flags=meta", event_patch)
        mining_patch = (ROOT / "patches" / "netherite" /
                        "0003-vanilla-obsidian-speed.diff").read_text().lower()
        self.assertIn("blk_obsidian", mining_patch)
        self.assertIn("ita_block_is_tool_effective", mining_patch)
        recipe_patch = (ROOT / "patches" / "netherite" /
                        "0004-complete-crafting.diff").read_text().lower()
        self.assertTrue((ROOT / "tools" /
                         "build_recipe_registry.py").is_file())
        self.assertFalse((ROOT / "bin" /
                          "build_recipe_registry.py").exists())
        self.assertIn("generated by bedrock-rl/tools/", recipe_patch)
        self.assertIn("crf_nrecipes 360", recipe_patch)
        self.assertIn("milk buckets", recipe_patch)
        portal_patch = (ROOT / "patches" / "netherite" /
                        "0005-safe-portal-arrival.diff").read_text().lower()
        self.assertIn("arrival volume", portal_patch)
        drops_patch = (ROOT / "patches" / "netherite" /
                       "0006-long-episode-drops.diff").read_text().lower()
        self.assertIn("#define gm_live_max 256", drops_patch)
        self.assertIn("#define gm_live_overflow_max 256", drops_patch)
        sneak_patch = (ROOT / "patches" / "netherite" /
                       "0007-sneak-use-placement.diff").read_text().lower()
        self.assertIn("!action.sneak", sneak_patch)
        self.assertIn("isr_is_empty(&held_now)", sneak_patch)
        contract_patch = (ROOT / "patches" / "netherite" /
                          "0008-route-contract-test.diff").read_text().lower()
        self.assertIn("selectable non-solid block occludes itembucket ray",
                      contract_patch)
        self.assertIn("game/rl_journal.o", contract_patch)
        self.assertIn("world/gen_prefetch.o", contract_patch)
        snapshot_patch = (ROOT / "patches" / "netherite" /
                          "0009-ten-chunk-snapshots.diff").read_text().lower()
        self.assertIn("if (radius > 160) radius = 160", snapshot_patch)

    def test_bundled_model_registry_keeps_proven_defaults(self):
        from bedrock_rl.models import MODELS

        by_key = {model.key: model for model in MODELS}
        self.assertEqual(len(by_key), len(MODELS))
        self.assertTrue({"qwen2-vl", "qwen3-vl"} <= set(by_key))
        self.assertTrue(all(by_key[key].status == "tested"
                            for key in ("qwen2-vl", "qwen3-vl")))
        self.assertTrue(all(model.image_span for model in MODELS))
        self.assertTrue(all(model.model_types for model in MODELS))

    def test_runtime_doctor_does_not_require_training_stack(self):
        from unittest.mock import patch

        from bedrock_rl.cli import doctor

        runtime = doctor.Check("engine", "ok")
        with (patch.object(doctor, "check_uv", return_value=runtime),
              patch.object(doctor, "repo_check", return_value=runtime),
              patch.object(doctor, "commit_check", return_value=runtime),
              patch.object(doctor, "engine_checks",
                           return_value=[runtime]),
              patch.object(doctor, "verl_home_check") as trainer):
            checks = doctor.run(runtime_only=True)
        self.assertEqual(len(checks), 4)
        trainer.assert_not_called()

    def test_container_doctor_does_not_warn_about_intentionally_absent_git(self):
        from unittest.mock import patch

        from bedrock_rl.cli import doctor

        with (patch.dict(os.environ, {"BRL_CONTAINER_IMAGE": "1"}),
              patch.object(doctor.resources, "repo_root", return_value=ROOT),
              patch.object(doctor, "_run", return_value=None)):
            result = doctor.commit_check()
        self.assertEqual(result.state, "skip")
        self.assertIn("image digest", result.detail)


if __name__ == "__main__":
    unittest.main()
