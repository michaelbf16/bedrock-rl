"""CPU regressions for causal training data, harvest freshness and portability."""
import json
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bedrock_rl import lora, models
from bedrock_rl.adapters.netherite.chat import prepare_sft_multiturn
from bedrock_rl.train.sft_data import CausalSFTDataset


ROOT = Path(__file__).resolve().parents[1]
BASE = "Qwen/Qwen3-VL-2B-Instruct"
COMMIT = "a" * 40
HAS_TORCH = importlib.util.find_spec("torch") is not None


def conversation():
    return [
        {"role": "user", "content": "<image>first observation"},
        {"role": "assistant", "content": "first action"},
        {"role": "user", "content": "<image>second observation"},
        {"role": "assistant", "content": "second action"},
        {"role": "user", "content": "<image>terminal observation"},
    ]


class CausalTrainingTests(unittest.TestCase):
    def test_full_rows_expand_to_actions_before_applying_image_window(self):
        original = {"messages": json.dumps(conversation()),
                    "images": ["first", "second", "terminal"]}
        rows = CausalSFTDataset([original])
        self.assertEqual(len(rows), 2)
        for index, expected in enumerate(("first", "second")):
            row = rows[index]
            messages, images = prepare_sft_multiturn(
                json.loads(row["messages"]), row["images"],
                last_assistant_only=row["loss_last_assistant_only"],
                keep_latest_images=1)
            self.assertEqual(images, [expected])
            self.assertEqual(messages[-1]["content"], f"{expected} action")
            self.assertNotIn("terminal", json.dumps(messages))
            self.assertEqual(sum(m["role"] == "assistant" for m in messages),
                             index + 1)
        self.assertEqual(original["images"], ["first", "second", "terminal"])

    def test_existing_last_action_masks_are_not_expanded(self):
        row = {"messages": json.dumps(conversation()), "images": [1, 2, 3],
               "loss_last_assistant_only": True}
        rows = CausalSFTDataset([row])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0], row)

    def test_unexpanded_multi_action_windows_fail_before_training(self):
        with self.assertRaisesRegex(ValueError, "one causal prefix"):
            prepare_sft_multiturn(conversation(), [1, 2, 3],
                                 keep_latest_images=1)

    def test_single_action_window_excludes_terminal_observation(self):
        view, images = prepare_sft_multiturn(
            conversation()[:3], ["input", "terminal"], keep_latest_images=1)
        self.assertEqual(images, ["input"])
        self.assertEqual(view[-1]["role"], "assistant")

    @unittest.skipUnless(HAS_TORCH, "torch is an optional training dependency")
    def test_harvest_keeps_each_actions_pixels_and_exact_sample(self):
        from PIL import Image
        from bedrock_rl.data import png_bytes
        from bedrock_rl.train.self_distill import harvest_rows
        from bedrock_rl.train.rollout import Rollout

        rollout = Rollout({"seed": 17}, 0)
        rollout.success, rollout.reward, rollout.turns = True, 1.0, 2
        rollout.messages = conversation()
        rollout.assistant_raw = [" exact first action\n", " exact second action\n"]
        rollout.images = [Image.new("RGB", (2, 2), color)
                          for color in ("red", "green", "blue")]
        with patch.dict(os.environ, {"BRL_KEEP_LATEST_IMAGES": "1"}):
            rows, failed = harvest_rows([rollout], None)
        self.assertEqual(failed, [])
        self.assertEqual(len(rows), 2)
        for index, row in enumerate(rows):
            self.assertTrue(row["loss_last_assistant_only"])
            self.assertEqual(row["images"][0]["bytes"],
                             png_bytes(rollout.images[index]))
            messages = json.loads(row["messages"])
            self.assertEqual(messages[-1]["content"], rollout.assistant_raw[index])
            self.assertNotIn("terminal", json.dumps(messages))


class HarvestFreshnessTests(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH, "torch is an optional training dependency")
    def test_empty_harvest_receipt_does_not_claim_previous_output(self):
        from bedrock_rl.train import self_distill

        class Reporter:
            def __init__(self, **_):
                pass

            def __getattr__(self, _):
                return lambda *args, **kwargs: None

        with tempfile.TemporaryDirectory() as directory:
            out, receipt = Path(directory) / "data.parquet", Path(directory) / "receipt"
            out.write_bytes(b"previous accepted harvest")
            argv = ["harvest", "--task", "unused", "--data", "unused",
                    "--out", str(out), "--receipt", str(receipt),
                    "--episodes", "1", "--batch", "1"]
            model = SimpleNamespace(eval=lambda: None)
            spec = SimpleNamespace(read_chat_template=lambda: None)
            failed = SimpleNamespace(success=False, stopped=True, turn_hinted=False)
            with (patch.object(sys, "argv", argv),
                  patch.object(self_distill.Task, "load", return_value=SimpleNamespace(name="test")),
                  patch.object(self_distill, "check_engine", return_value="hf"),
                  patch.object(self_distill.hints, "hint_text", return_value=None),
                  patch.object(self_distill, "model_banner", return_value="fake"),
                  patch.object(self_distill.console, "Reporter", Reporter),
                  patch.object(self_distill, "load_episode_specs", return_value=[{"seed": 1}]),
                  patch.object(self_distill, "load_model", return_value=(model, object(), spec)),
                  patch.object(self_distill, "Coder", return_value=object()),
                  patch.object(self_distill, "build_sampler", return_value=model),
                  patch.object(self_distill, "play", return_value=[failed])):
                self_distill.main()
            self.assertEqual(out.read_bytes(), b"previous accepted harvest")
            result = json.loads(receipt.read_text())
            self.assertEqual(result["examples"], 0)
            self.assertEqual(result["out"], str(out))

    def test_launcher_requires_successful_current_receipt(self):
        # Exercise the actual shell flow with only the GPU invocations stubbed.
        common = r'''
set -euo pipefail
brl_setup() { BRL_ROOT="$FIXTURE_ROOT"; TASK=test; MODEL=unused; BRL_TASK_NAME=test; }
brl_resolve_model() { :; }
brl_paths() { BRL_SLUG=test; BRL_METRICS="$BRL_ROOT/metrics.jsonl"; BRL_LOG="$BRL_ROOT/log"; OUT="$BRL_ROOT/out"; }
brl_flag() { :; }
brl_lora_flags() { :; }
brl_task_yaml() { echo unused; }
brl_run() { "$FIXTURE_PYTHON" "$FIXTURE_ROOT/invoke.py" "$@"; }
'''
        invoke = '''
import json, os, pathlib, subprocess, sys
args = sys.argv[2:]
if args[:2] == ["-m", "bedrock_rl.train.self_distill"]:
    out = pathlib.Path(args[args.index("--out") + 1])
    count = int(os.environ["FIXTURE_EXAMPLES"])
    if count:
        out.write_text("fresh accepted harvest")
    receipt = pathlib.Path(args[args.index("--receipt") + 1])
    receipt.write_text(json.dumps({"examples": count, "out": os.path.abspath(out)}))
elif args[:2] == ["-m", "bedrock_rl.train.sft"]:
    pathlib.Path(os.environ["FIXTURE_ROOT"], "trained").touch()
else:
    sys.exit(subprocess.call([sys.executable, *args]))
'''
        for count, harvest_only in ((0, False), (0, True), (2, False)):
            with self.subTest(count=count, harvest_only=harvest_only), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                shutil.copy(ROOT / "bin/self_distill.sh", root / "self_distill.sh")
                (root / "_common.sh").write_text(common)
                (root / "invoke.py").write_text(invoke)
                data = root / "data/test_cu/self_distill_procedure_self_distill.parquet"
                data.parent.mkdir(parents=True)
                data.write_text("old accepted harvest")
                env = {**os.environ, "FIXTURE_ROOT": directory,
                       "FIXTURE_PYTHON": sys.executable,
                       "FIXTURE_EXAMPLES": str(count)}
                for key in ("HINT_LEVEL", "TURN_HINT", "RUN_TAG", "HARVEST_ONLY"):
                    env.pop(key, None)
                if harvest_only:
                    env["HARVEST_ONLY"] = "1"
                result = subprocess.run(["bash", str(root / "self_distill.sh")],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode,
                                 0 if count or harvest_only else 1, result.stderr)
                self.assertEqual((root / "trained").exists(), count > 0)
                if not count:
                    self.assertEqual(data.read_text(), "old accepted harvest")


class AdapterPortabilityTests(unittest.TestCase):
    def make_adapter(self, directory, base=BASE, revision=None):
        path = Path(directory) / "adapter"
        path.mkdir()
        (path / lora.ADAPTER_CONFIG).write_text(json.dumps({
            "base_model_name_or_path": base, "revision": revision,
            "r": 4, "lora_alpha": 4}))
        return path

    def test_saved_cache_snapshot_becomes_hub_id_and_exact_revision(self):
        from bedrock_rl.train.peft_util import set_adapter_base
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots" / COMMIT
            snapshot.mkdir(parents=True)
            adapter = self.make_adapter(directory, str(snapshot))
            set_adapter_base(adapter, str(snapshot))
            config = lora.read_adapter_config(adapter)
            self.assertEqual(config["base_model_name_or_path"], BASE)
            self.assertEqual(config["revision"], COMMIT)
            shutil.rmtree(snapshot.parent.parent)
            with patch.object(models, "download", return_value="/new-host/snapshot") as fetch:
                self.assertEqual(models.policy_paths(adapter, snapshot=True),
                                 ("/new-host/snapshot", str(adapter)))
            fetch.assert_called_once_with(BASE, revision=COMMIT)
            self.assertEqual(models.provenance(adapter)["revision"], COMMIT)

    def test_verl_export_writes_supplied_revision(self):
        from bedrock_rl.train import export_adapter
        with tempfile.TemporaryDirectory() as directory:
            actor = Path(directory) / "global_step_1/actor"
            actor.mkdir(parents=True)

            def merge(_actor, target):
                adapter = Path(target) / lora.MERGED_ADAPTER
                adapter.mkdir(parents=True)
                (adapter / lora.ADAPTER_CONFIG).write_text(json.dumps({
                    "r": 4, "lora_alpha": 4}))
                return 0

            with patch.object(export_adapter, "merge", side_effect=merge):
                adapter = export_adapter.export(directory, BASE, revision=COMMIT)
            config = lora.read_adapter_config(adapter)
            self.assertEqual(config["base_model_name_or_path"], BASE)
            self.assertEqual(config["revision"], COMMIT)

    @unittest.skipUnless(HAS_TORCH and importlib.util.find_spec("peft"),
                         "PEFT is an optional training dependency")
    def test_periodic_peft_save_already_has_portable_metadata(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        from bedrock_rl.train.peft_util import create
        with tempfile.TemporaryDirectory() as directory:
            config = LlamaConfig(vocab_size=32, hidden_size=16,
                                 intermediate_size=32, num_hidden_layers=1,
                                 num_attention_heads=2, num_key_value_heads=2)
            config._name_or_path = str(Path(directory)
                / "models--test-org--tiny-llama/snapshots" / COMMIT)
            model = create(LlamaForCausalLM(config), None, 4, 4,
                           ["q_proj"], None)
            out = Path(directory) / "checkpoint-1"
            model.save_pretrained(out, save_embedding_layers=False)
            saved = lora.read_adapter_config(out)
            self.assertEqual(saved["base_model_name_or_path"], "test-org/tiny-llama")
            self.assertEqual(saved["revision"], COMMIT)

    def test_explicit_adapter_matches_cached_identity_but_checks_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.make_adapter(directory, revision=COMMIT)
            snapshot = Path(directory) / "models--Qwen--Qwen3-VL-2B-Instruct/snapshots" / COMMIT
            snapshot.mkdir(parents=True)
            self.assertEqual(models.resolve_policy(str(snapshot), adapter).adapter,
                             str(adapter))
            wrong = snapshot.with_name("b" * 40)
            wrong.mkdir()
            with self.assertRaisesRegex(models.PolicyError, "was trained at"):
                models.resolve_policy(str(wrong), adapter)

    def test_custom_local_base_is_not_relabelled_as_hub_weights(self):
        from bedrock_rl.train.peft_util import set_adapter_base
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "custom-checkpoint"
            base.mkdir()
            adapter = self.make_adapter(directory)
            set_adapter_base(adapter, base)
            config = lora.read_adapter_config(adapter)
            self.assertEqual(config["base_model_name_or_path"], str(base))
            self.assertIsNone(config["revision"])

    def test_continued_adapter_keeps_non_registry_revision(self):
        from bedrock_rl.train.peft_util import set_adapter_base
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.make_adapter(directory, revision=COMMIT)
            set_adapter_base(adapter, BASE)
            self.assertEqual(lora.read_adapter_config(adapter)["revision"], COMMIT)
            with patch.object(models, "weights_path", return_value="/snapshot") as fetch:
                output = models.shell_env(adapter, snapshot=True)
            fetch.assert_called_once_with(BASE, revision=COMMIT)
            self.assertIn(f"BRL_MODEL_REVISION='{COMMIT}'", output)

    def test_unpinned_hub_variant_exports_the_downloaded_commit(self):
        base = "Qwen/Qwen3-VL-2B-Custom"
        self.assertIsNone(models.model_revision(base))
        snapshot = f"/another-host/models--Qwen--Qwen3-VL-2B-Custom/snapshots/{COMMIT}"
        with patch.object(models, "weights_path", return_value=snapshot):
            output = models.shell_env(base, snapshot=True)
        self.assertIn(f"BRL_MODEL_ID='{base}'", output)
        self.assertIn(f"BRL_MODEL_REVISION='{COMMIT}'", output)

    def test_snapshot_commit_cannot_conflict_with_the_exported_pin(self):
        snapshot = f"/another-host/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/{COMMIT}"
        with patch.object(models, "weights_path", return_value=snapshot):
            with self.assertRaisesRegex(models.PolicyError, "not revision"):
                models.shell_env(BASE, snapshot=True)

    def test_continued_merged_layout_retains_original_hub_identity(self):
        from bedrock_rl.train.peft_util import set_adapter_base
        with tempfile.TemporaryDirectory() as directory:
            merged = Path(directory) / "merged"
            adapter = merged / lora.MERGED_ADAPTER
            adapter.mkdir(parents=True)
            (merged / "config.json").write_text('{"model_type":"qwen3_vl"}')
            (adapter / lora.ADAPTER_CONFIG).write_text(json.dumps({
                "base_model_name_or_path": BASE, "revision": COMMIT,
                "r": 4, "lora_alpha": 4}))
            output = models.shell_env(merged, snapshot=True)
            self.assertIn(f"BRL_MODEL_PATH='{merged}'", output)
            self.assertIn(f"BRL_MODEL_ID='{BASE}'", output)
            self.assertIn(f"BRL_MODEL_REVISION='{COMMIT}'", output)
            exported = self.make_adapter(directory, base=str(merged))
            set_adapter_base(exported, merged)
            config = lora.read_adapter_config(exported)
            self.assertEqual(config["base_model_name_or_path"], BASE)
            self.assertEqual(config["revision"], COMMIT)


@unittest.skipUnless(HAS_TORCH, "torch is an optional training dependency")
class VLLMStartupTests(unittest.TestCase):
    def test_spawn_is_selected_before_constructing_engine(self):
        from bedrock_rl.train.rollout import VLLMSampler
        seen = []

        class LLM:
            def __init__(self, **kwargs):
                seen.append(os.environ.get("VLLM_WORKER_MULTIPROC_METHOD"))

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            os.environ.pop("VLLM_WORKER_MULTIPROC_METHOD", None)
            with patch.dict(sys.modules, {"vllm": SimpleNamespace(LLM=LLM)}):
                sampler = VLLMSampler(directory, SimpleNamespace(spec=None),
                                      workdir=directory, say=lambda *_: None)
                sampler.close()
            self.assertEqual(seen, ["spawn"])


if __name__ == "__main__":
    unittest.main()
