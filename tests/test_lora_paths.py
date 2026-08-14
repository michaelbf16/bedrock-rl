import json
import os
import tempfile
import unittest
from pathlib import Path

from bedrock_rl import lora, models


def write_adapter(path, base):
    os.makedirs(path)
    with open(os.path.join(path, lora.ADAPTER_CONFIG), "w") as f:
        json.dump({"base_model_name_or_path": base, "r": 32,
                   "lora_alpha": 32}, f)


class ProcessorPathTests(unittest.TestCase):
    def test_sft_uses_continued_adapters_saved_processor(self):
        source = (Path(__file__).parents[1] / "bedrock_rl" / "train" /
                  "sft.py").read_text()
        self.assertIn("lora.processor_path(path, a.lora_adapter)", source)

    def test_full_sft_loads_fp32_master_weights(self):
        source = (Path(__file__).parents[1] / "bedrock_rl" / "train" /
                  "sft.py").read_text()
        self.assertIn(
            "weight_dtype = torch.bfloat16 if use_lora else torch.float32",
            source)
        self.assertIn("a.lr = 1e-5 if use_lora else 1e-6", source)

    def test_full_checkpoint_uses_processor_saved_in_run_parent(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = os.path.join(root, "checkpoint-40")
            os.mkdir(checkpoint)
            with open(os.path.join(root, "preprocessor_config.json"), "w"):
                pass
            self.assertEqual(lora.processor_path(checkpoint), root)

    def test_adapter_checkpoint_parent_precedes_base_processor(self):
        with tempfile.TemporaryDirectory() as root:
            base = os.path.join(root, "base")
            run = os.path.join(root, "run")
            adapter = os.path.join(run, "checkpoint-20")
            os.makedirs(base)
            os.makedirs(adapter)
            for directory in (base, run):
                with open(os.path.join(directory,
                                       "preprocessor_config.json"), "w"):
                    pass
            self.assertEqual(lora.processor_path(base, adapter), run)

    def test_non_checkpoint_without_processor_falls_back_to_base(self):
        self.assertEqual(lora.processor_path("Qwen/base", "/tmp/adapter"),
                         "Qwen/base")


class PolicyPathTests(unittest.TestCase):
    def test_explicit_adapter_overrides_inferred_adapter_not_its_base(self):
        with tempfile.TemporaryDirectory() as root:
            inferred = os.path.join(root, "inferred")
            explicit = os.path.join(root, "explicit")
            write_adapter(inferred, "Qwen/Qwen2-VL-2B-Instruct")
            write_adapter(explicit, "Qwen/Qwen2-VL-2B-Instruct")
            self.assertEqual(
                models.policy_paths(inferred, explicit),
                ("Qwen/Qwen2-VL-2B-Instruct", explicit),
            )

    def test_adapter_provenance_names_its_base_as_the_hub_model(self):
        with tempfile.TemporaryDirectory() as root:
            adapter = os.path.join(root, "adapter")
            write_adapter(adapter, "Qwen/Qwen2-VL-2B-Instruct")
            provenance = models.provenance(adapter)
            self.assertEqual(provenance["requested"], adapter)
            self.assertEqual(provenance["resolved"],
                             "Qwen/Qwen2-VL-2B-Instruct")
            self.assertEqual(provenance["hub_id"],
                             "Qwen/Qwen2-VL-2B-Instruct")
            self.assertEqual(provenance["adapter"], adapter)
            self.assertEqual(provenance["adapter_rank"], 32)
            self.assertEqual(provenance["adapter_alpha"], 32)

    def test_explicit_adapter_must_exist_and_declare_a_valid_rank(self):
        with self.assertRaisesRegex(models.PolicyError, "does not exist"):
            models.policy_paths("qwen3-vl-2b", "/definitely/missing/adapter")
        with tempfile.TemporaryDirectory() as root:
            adapter = os.path.join(root, "adapter")
            write_adapter(adapter, "Qwen/Qwen3-VL-2B-Instruct")
            config = os.path.join(adapter, lora.ADAPTER_CONFIG)
            with open(config, "w") as stream:
                json.dump({"base_model_name_or_path":
                           "Qwen/Qwen3-VL-2B-Instruct", "r": 0}, stream)
            with self.assertRaisesRegex(models.PolicyError, "invalid rank"):
                models.policy_paths("qwen3-vl-2b", adapter)

    def test_explicit_adapter_must_match_the_selected_base(self):
        with tempfile.TemporaryDirectory() as root:
            adapter = os.path.join(root, "adapter")
            write_adapter(adapter, "Qwen/Qwen2-VL-2B-Instruct")
            with self.assertRaisesRegex(models.PolicyError, "was trained on"):
                models.policy_paths("qwen3-vl-2b", adapter)

    def test_emitted_run_provenance_includes_explicit_adapter(self):
        with tempfile.TemporaryDirectory() as root:
            adapter = os.path.join(root, "adapter")
            write_adapter(adapter, "Qwen/Qwen3-VL-2B-Instruct")
            output = os.path.join(root, "model.yaml")
            models.emit_config("qwen3-vl-2b", output, adapter)
            with open(os.path.join(root, "model.provenance.json")) as stream:
                provenance = json.load(stream)
            self.assertEqual(provenance["adapter"], adapter)
            self.assertEqual(provenance["adapter_base"],
                             "Qwen/Qwen3-VL-2B-Instruct")
            self.assertEqual(provenance["adapter_rank"], 32)


if __name__ == "__main__":
    unittest.main()
