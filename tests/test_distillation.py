from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bedrock_rl.adapters.verl.distillation import (
    check_tokenizers, default_gpus_per_node, hydra_overrides, load_teachers,
    stage_teachers,
)


class DistillationConfigTests(unittest.TestCase):
    def test_teacher_staging_binds_runtime_config_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / ("a" * 40)
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}")
            provenance = Path(directory) / "teachers.json"
            payload = json.dumps({
                "expert": {"model": "Qwen/Qwen3-VL-32B-Instruct"}
            })
            with patch("bedrock_rl.models.weights_path",
                       return_value=str(snapshot)), patch(
                           "bedrock_rl.models.model_revision",
                           return_value="pinned-commit"):
                staged = json.loads(stage_teachers(
                    payload, "opd", str(provenance)))
            self.assertEqual(staged["expert"]["model"], str(snapshot))
            record = json.loads(provenance.read_text())
            self.assertEqual(record["teachers"]["expert"]["requested"],
                             "Qwen/Qwen3-VL-32B-Instruct")
            self.assertEqual(record["teachers"]["expert"]["revision"],
                             "pinned-commit")
            self.assertIn("config.json",
                          record["teachers"]["expert"]["files"])

    def test_teacher_adapter_is_rejected_instead_of_serving_its_base(self):
        payload = json.dumps({"expert": {"model": "/tmp/adapter"}})
        with patch("bedrock_rl.models.policy_paths",
                   return_value=("Qwen/base", "/tmp/adapter")):
            with self.assertRaisesRegex(ValueError, "merge the adapter"):
                stage_teachers(payload, "opd")

    def test_tokenizer_preflight_uses_exact_paths_without_remote_code(self):
        calls = []

        class Tokenizer:
            eos_token_id = 1

            @classmethod
            def from_pretrained(cls, path, **kwargs):
                calls.append((path, kwargs))
                return cls()

            def get_vocab(self):
                return {"x": 0}

        teachers = load_teachers(json.dumps({
            "expert": {"model": "/snapshots/teacher"}
        }), "opd")
        module = SimpleNamespace(AutoTokenizer=Tokenizer)
        with patch.dict(sys.modules, {"transformers": module}):
            check_tokenizers("/snapshots/student", teachers)
        self.assertEqual([call[0] for call in calls],
                         ["/snapshots/student", "/snapshots/teacher"])
        self.assertTrue(all(call[1]["trust_remote_code"] is False
                            for call in calls))

    def test_teacher_pipeline_parallel_is_refused_before_gpu_allocation(self):
        with self.assertRaisesRegex(ValueError, "not implemented"):
            load_teachers(json.dumps({
                "expert": {"model": "Qwen/teacher", "pipeline_parallel": 2}
            }), "opd")

    def test_opd_uses_verl_default_teacher_entry(self):
        teachers = load_teachers(json.dumps({
            "expert": {"model": "Qwen/teacher", "tensor_parallel": 2}
        }), "opd")
        self.assertEqual(default_gpus_per_node(teachers, "opd", 2), 2)
        args = hydra_overrides(
            teachers, "opd", teacher_gpus_per_node=2, teacher_nnodes=2,
            teacher_key="data_source", max_model_len=8192,
            loss_mode="k1", topk=64, use_task_rewards="False",
            use_policy_gradient="True", coefficient=1.0,
            loss_max_clamp="10.0", log_prob_min_clamp="-10.0")
        self.assertIn("distillation.enabled=True", args)
        self.assertIn(
            "distillation.teacher_models.teacher_model.model_path=Qwen/teacher",
            args)

    def test_mopd_routes_named_teachers_and_checks_pool_size(self):
        teachers = load_teachers(json.dumps({
            "mine": {"model": "mine-teacher", "route": "mining",
                     "tensor_parallel": 2},
            "craft": {"model": "craft-teacher", "route": "crafting",
                      "tensor_parallel": 2},
        }), "mopd")
        args = hydra_overrides(
            teachers, "mopd", teacher_gpus_per_node=4, teacher_nnodes=1,
            teacher_key="data_source", max_model_len=8192,
            loss_mode="k1", topk=64, use_task_rewards="False",
            use_policy_gradient="True", coefficient=1.0,
            loss_max_clamp="10.0", log_prob_min_clamp="-10.0")
        self.assertIn("+distillation.teacher_models.mine.key=mining", args)
        with self.assertRaisesRegex(ValueError, "teacher pool"):
            hydra_overrides(
                teachers, "mopd", teacher_gpus_per_node=2,
                teacher_nnodes=1, teacher_key="data_source",
                max_model_len=8192, loss_mode="k1", topk=64,
                use_task_rewards="False", use_policy_gradient="True",
                coefficient=1.0, loss_max_clamp="10.0",
                log_prob_min_clamp="-10.0")

    def test_launcher_stages_before_preflight_and_hydra_rendering(self):
        source = (Path(__file__).parents[1] / "bin" / "rl.sh").read_text()
        self.assertLess(source.index("stage-teachers"),
                        source.index("check-tokenizers"))
        self.assertLess(source.index("stage-teachers"),
                        source.index("distillation hydra"))


if __name__ == "__main__":
    unittest.main()
