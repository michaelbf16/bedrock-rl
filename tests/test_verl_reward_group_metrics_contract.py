"""Executable contract for GRPO group metrics at the pinned verl revision."""

import importlib.util
import unittest
from types import SimpleNamespace


HAS_VERL = importlib.util.find_spec("verl") is not None


@unittest.skipUnless(HAS_VERL, "the exact patched verl runtime is tested in CI")
class VerlRewardGroupMetricsContractTests(unittest.TestCase):
    def test_actual_reward_contrast_is_grouped_by_sibling_uid(self):
        from verl.trainer.ppo.reward_group_metrics import (
            compute_grpo_group_reward_metrics,
        )

        metrics = compute_grpo_group_reward_metrics(
            sample_uids=["a", "b", "a", "b"],
            sequence_rewards=[0.0, 0.5, 1.0, 0.5],
        )

        self.assertEqual(metrics["grpo/reward_group/count"], 2)
        self.assertEqual(metrics["grpo/reward_group/zero_contrast/count"], 1)
        self.assertEqual(
            metrics["grpo/reward_group/nonzero_contrast/count"], 1)
        self.assertEqual(metrics["grpo/reward_group/contrast/max"], 1.0)
        self.assertNotIn(
            "grpo/success_group/all_unsuccessful/count", metrics)
        self.assertNotIn("grpo/success_group/all_successful/count", metrics)

    def test_success_composition_requires_explicit_boolean_outcomes(self):
        from verl.trainer.ppo.reward_group_metrics import (
            compute_grpo_group_reward_metrics,
        )

        metrics = compute_grpo_group_reward_metrics(
            sample_uids=["same", "same"],
            sequence_rewards=[0.0, 0.0],
            successes=[False, True],
        )

        self.assertEqual(metrics["grpo/reward_group/zero_contrast/count"], 1)
        self.assertEqual(metrics["grpo/success_group/mixed/count"], 1)

        with self.assertRaisesRegex(TypeError, "authoritative booleans"):
            compute_grpo_group_reward_metrics(
                sample_uids=["same"],
                sequence_rewards=[0.0],
                successes=[0],
            )

    def test_pinned_postprocess_and_exact_trainer_hook_share_aligned_fields(self):
        import numpy as np
        import torch
        from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
        from verl.trainer.ppo.reward_group_metrics import (
            ROLLOUT_EPISODE_SUCCESS_FIELD,
            compute_grpo_group_reward_metrics_from_batch,
            update_non_tensor_batch_with_reward_extra_info,
        )

        from bedrock_rl.adapters.verl.agent_loop import (
            VERL_EPISODE_SUCCESS_FIELD,
        )

        self.assertEqual(
            ROLLOUT_EPISODE_SUCCESS_FIELD, VERL_EPISODE_SUCCESS_FIELD)

        def output(success):
            return SimpleNamespace(
                prompt_ids=torch.tensor([[1, 2]]),
                response_ids=torch.tensor([[3, 4]]),
                response_mask=torch.tensor([[1, 1]]),
                attention_mask=torch.tensor([[1, 1, 1, 1]]),
                input_ids=torch.tensor([[1, 2, 3, 4]]),
                position_ids=torch.tensor([[0, 1, 2, 3]]),
                response_logprobs=None,
                routed_experts=None,
                teacher_logprobs=None,
                teacher_ids=None,
                reward_score=0.0,
                num_turns=1,
                multi_modal_inputs=None,
                extra_fields={VERL_EPISODE_SUCCESS_FIELD: success},
                metrics=SimpleNamespace(model_dump=lambda: {}),
            )

        # This order models repeat followed by global token balancing: sibling
        # uids are aligned but deliberately noncontiguous.
        uids = np.array(["uid-a", "uid-b", "uid-a", "uid-b"], dtype=object)
        worker = object.__new__(AgentLoopWorker)
        worker.reward_loop_worker_handles = None
        batch = worker._postprocess(
            [output(False), output(True), output(True), output(True)],
            input_non_tensor_batch={
                "uid": uids,
                # An unrelated generic field must not become task outcomes.
                "success": np.array([True, True, True, True]),
                # _postprocess must overwrite even a colliding dataset field
                # with the rollout-owned values from output.extra_fields.
                VERL_EPISODE_SUCCESS_FIELD: np.array(
                    [True, True, True, True]),
            },
        )

        self.assertEqual(
            batch.non_tensor_batch[VERL_EPISODE_SUCCESS_FIELD].tolist(),
            [False, True, True, True],
        )
        self.assertEqual(batch.non_tensor_batch["uid"].tolist(), uids.tolist())

        # Scores represent the pre-KL outcome. Only token_level_rewards,
        # representing the post-KL values at the advantage boundary, have
        # contrast for uid-a.
        batch.batch["token_level_scores"] = torch.tensor([
            [1.0, 0.0], [2.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        ])
        batch.batch["token_level_rewards"] = torch.tensor([
            [0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [0.5, 0.0],
        ])

        metrics = compute_grpo_group_reward_metrics_from_batch(batch)

        self.assertEqual(metrics["grpo/reward_group/zero_contrast/count"], 1)
        self.assertEqual(
            metrics["grpo/reward_group/nonzero_contrast/count"], 1)
        self.assertEqual(metrics["grpo/reward_group/contrast/max"], 1.0)
        self.assertEqual(metrics["grpo/success_group/mixed/count"], 1)
        self.assertEqual(
            metrics["grpo/success_group/all_successful/count"], 1)

        with self.assertRaisesRegex(ValueError, "rollout-owned field"):
            update_non_tensor_batch_with_reward_extra_info(
                batch,
                {VERL_EPISODE_SUCCESS_FIELD: np.array(
                    [True, True, True, True])},
            )

        batch.non_tensor_batch[VERL_EPISODE_SUCCESS_FIELD] = np.array(
            [False, True])
        with self.assertRaisesRegex(ValueError, "same length"):
            compute_grpo_group_reward_metrics_from_batch(batch)


if __name__ == "__main__":
    unittest.main()
