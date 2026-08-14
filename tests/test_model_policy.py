import subprocess
import unittest
from unittest.mock import patch

from bedrock_rl import models
from bedrock_rl.cli import model_check
from bedrock_rl.core.messages import validate_transcript
from bedrock_rl.core.sampling import (MAX_TOOL_CALLS_PER_TURN,
                                      multimodal_image_limit)


class ModelPolicyTests(unittest.TestCase):
    def test_image_acceptance_limit_covers_multi_call_turns(self):
        limit = multimodal_image_limit(4)

        self.assertEqual(limit, 17)
        self.assertGreaterEqual(limit, 1 + 4 * 2)
        self.assertEqual(
            multimodal_image_limit(80, keep_latest_images=1), 1)
        environment = models.shell_env("qwen3-vl", max_turns=4)
        self.assertIn("BRL_LIMIT_IMAGES=17", environment)
        self.assertIn(
            f"BRL_MAX_TOOL_CALLS_PER_TURN={MAX_TOOL_CALLS_PER_TURN}",
            environment)

    def test_verl_override_array_preserves_spaces(self):
        overrides = ["+model.path=/tmp/model with spaces", "+flag=true"]
        with patch.object(models, "verl_overrides", return_value=overrides):
            environment = models.shell_env("qwen3-vl", max_turns=4)
        shell_lines = [
            environment,
            "args=()",
            "while IFS= read -r override; do",
            "  [ -z \"$override\" ] || args+=(\"$override\")",
            "done <<< \"$BRL_VERL_OVERRIDES\"",
            "printf '<%s>\\n' \"${args[@]}\"",
        ]
        result = subprocess.run(
            ["bash", "-c", "\n".join(shell_lines)], text=True,
            capture_output=True, check=True)

        self.assertEqual(result.stdout.splitlines(), [
            "<+model.path=/tmp/model with spaces>", "<+flag=true>"])

    def test_model_check_uses_the_canonical_visual_tool_transcript(self):
        chain = validate_transcript(model_check.CHAIN)
        self.assertEqual([message["role"] for message in chain],
                         ["user", "assistant", "tool", "user"])
        call = chain[1]["tool_calls"][0]
        self.assertEqual(call["id"], chain[2]["tool_call_id"])
        self.assertIsInstance(call["function"]["arguments"], str)
        self.assertEqual(chain[2]["content"][0]["text"], "OBS_MARKER")
        self.assertEqual(chain[3]["content"][0]["type"], "image_url")

    def test_fit_report_refuses_memory_fraction_with_no_actor_room(self):
        spec = models.resolve("qwen3-vl-2b")
        for value in (0, -0.1, 0.95, 1.0):
            with self.subTest(value=value):
                report, ok = models.fit_report(
                    spec, "2B", 1, gpu_mem_util=value)
                self.assertFalse(ok)
                self.assertIn("below 0.95", report)

    def test_unknown_sizing_is_explicitly_advisory(self):
        spec = models.resolve("qwen3-vl-2b")
        with patch.object(models, "card_memory_gib", return_value=None):
            report, ok = models.fit_report(spec, "2B", 1)
        self.assertTrue(ok)
        self.assertIn("advisory unavailable", report)
        self.assertIn("no fit claim", report)

    def test_full_fine_tune_and_lora_have_distinct_fit_estimates(self):
        spec = models.resolve("qwen3-vl-2b")
        full = models.sizing(spec, "2B", 1, card_gib=24, lora=False)
        adapter = models.sizing(spec, "2B", 1, card_gib=24, lora=True)

        self.assertGreater(full["actor_per_card_gib"],
                           adapter["actor_per_card_gib"])
        self.assertFalse(full["fits"])
        self.assertTrue(adapter["fits"])
        self.assertIn("Adam moments", full["actor_basis"])
        self.assertIn("measured LoRA", adapter["actor_basis"])

    def test_offload_fit_is_explicitly_advisory(self):
        spec = models.resolve("qwen3-vl-2b")
        with patch.object(models, "card_memory_gib", return_value=24):
            report, ok = models.fit_report(
                spec, "2B", 1, offload=True)
        self.assertTrue(ok)
        self.assertIn("offload has no calibrated peak", report)
        self.assertIn("no fit claim", report)

    def test_shared_location_rule_prefers_existing_relative_checkpoint(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoints" / "run-10"
            checkpoint.mkdir(parents=True)
            resolve = lambda value: str(root / value)
            self.assertEqual(
                models.resolve_model_location("checkpoints/run-10", resolve),
                str(checkpoint))
            self.assertEqual(
                models.resolve_model_location(
                    "Qwen/Qwen3-VL-2B-Instruct", resolve),
                "Qwen/Qwen3-VL-2B-Instruct")

    def test_doctor_does_not_certify_ambiguous_missing_path_as_hub(self):
        from bedrock_rl.cli.doctor import model_check

        missing = model_check("checkpoints/definitely-missing")
        hub = model_check("Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(missing.state, "fail")
        self.assertNotIn("a hub id", missing.detail)
        self.assertEqual(hub.state, "ok")
        self.assertIn("a hub id", hub.detail)


if __name__ == "__main__":
    unittest.main()
