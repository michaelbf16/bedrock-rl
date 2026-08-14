"""Contract tests that run only in CI's exact pinned-verl environment."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

from bedrock_rl.adapters.verl.distillation import (hydra_overrides,
                                                      load_teachers)


HAS_VERL = importlib.util.find_spec("verl") is not None
HAS_HYDRA = importlib.util.find_spec("hydra") is not None


@unittest.skipUnless(HAS_VERL and HAS_HYDRA,
                     "the exact verl runtime is tested in its CI job")
class VerlDistillationContractTests(unittest.TestCase):
    def test_rl_estimators_and_gspo_loss_exist_at_the_pin(self):
        from verl.trainer.ppo.core_algos import (get_adv_estimator_fn,
                                                 get_policy_loss_fn)

        for estimator in ("grpo", "rloo", "reinforce_plus_plus", "remax"):
            with self.subTest(estimator=estimator):
                self.assertTrue(callable(get_adv_estimator_fn(estimator)))
        self.assertTrue(callable(get_policy_loss_fn("gspo")))

    @staticmethod
    def _instantiate(mode, payload, pool):
        import verl
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate

        overrides = hydra_overrides(
            load_teachers(json.dumps(payload), mode), mode,
            teacher_gpus_per_node=pool, teacher_nnodes=1,
            teacher_key="data_source", max_model_len=8192,
            loss_mode="k1", topk=64, use_task_rewards="False",
            use_policy_gradient="True", coefficient=1.0,
            loss_max_clamp="10.0", log_prob_min_clamp="-10.0")
        config_dir = Path(verl.__file__).parent / "trainer" / "config"
        with initialize_config_dir(config_dir=str(config_dir),
                                   version_base=None):
            config = compose(config_name="ppo_trainer", overrides=overrides)
        return instantiate(config.distillation, _convert_="partial")

    def test_single_teacher_opd_matches_the_pinned_verl_schema(self):
        config = self._instantiate("opd", {
            "expert": {
                "model": "Qwen/Qwen3-VL-8B-Instruct",
                "tensor_parallel": 2,
            }
        }, 2)

        self.assertEqual(set(config.teacher_models), {"default"})
        self.assertEqual(config.n_gpus_per_node * config.nnodes, 2)
        self.assertEqual(config.distillation_loss.loss_mode, "k1")
        self.assertTrue(config.distillation_loss.use_policy_gradient)
        self.assertFalse(config.distillation_loss.use_task_rewards)

    def test_routed_mopd_matches_the_pinned_verl_schema(self):
        config = self._instantiate("mopd", {
            "mine": {
                "model": "mining-teacher", "route": "mining",
                "tensor_parallel": 2,
            },
            "craft": {
                "model": "crafting-teacher", "route": "crafting",
                "tensor_parallel": 2,
            },
        }, 4)

        self.assertEqual(set(config.teacher_models), {"mining", "crafting"})
        self.assertEqual(config.n_gpus_per_node * config.nnodes, 4)


if __name__ == "__main__":
    unittest.main()
