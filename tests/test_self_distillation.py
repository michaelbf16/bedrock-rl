import unittest
import random
from types import SimpleNamespace
from unittest.mock import patch

from bedrock_rl.sdg.guidance import (TurnHinter,
                                                   assert_hint_removable,
                                                   strip, with_hint)


def feedback_provider(task, env, turn_index, level):
    return f"state={env.value} turn={turn_index} level={level}"


class DynamicFeedbackTests(unittest.TestCase):
    def test_teacher_only_feedback_never_enters_student_messages(self):
        task = SimpleNamespace(
            name="test",
            raw={"turn_hint_provider":
                 f"{__name__}:feedback_provider"})
        hinter = TurnHinter(task, "direction", visible=False)
        messages = [{"role": "user", "content": "observation"}]
        hinter.attach(messages, SimpleNamespace(value=7), 0)
        self.assertEqual(messages[0]["content"], "observation")
        self.assertEqual(hinter.marks, [])
        teacher = hinter.teacher_view(messages)
        self.assertEqual(messages[0]["content"], "observation")
        self.assertIn("state=7", teacher[0]["content"])

    def test_static_and_dynamic_privilege_are_removed_exactly(self):
        task = SimpleNamespace(
            name="test",
            raw={"turn_hint_provider": f"{__name__}:feedback_provider"})
        plain = [{"role": "user", "content": "first observation"}]
        messages = with_hint(plain, "private procedure")
        hinter = TurnHinter(task, "direction", visible=True)
        hinter.attach(messages, SimpleNamespace(value=1), 0)
        messages.extend([
            {"role": "assistant", "content": "act"},
            {"role": "user", "content": "second observation"},
        ])
        hinter.attach(messages, SimpleNamespace(value=2), 1)
        unprivileged = [
            {"role": "user", "content": "first observation"},
            {"role": "assistant", "content": "act"},
            {"role": "user", "content": "second observation"},
        ]

        self.assertEqual(strip(messages, "private procedure", hinter.marks),
                         unprivileged)
        damaged = [dict(message) for message in messages]
        damaged[-1]["content"] += " changed"
        with self.assertRaisesRegex(ValueError, "refusing to guess"):
            strip(damaged, "private procedure", hinter.marks)

    def test_static_hint_removal_is_exact_in_rendered_model_prompt(self):
        body = [{"role": "user", "content": "inspect the block"}]

        def render(messages):
            messages = list(messages)
            if messages and messages[0]["role"] == "system":
                system = messages.pop(0)["content"]
            else:
                system = "default system"
            turns = "".join(
                f"<{message['role']}>{message['content']}</{message['role']}>"
                for message in messages)
            return f"<system>{system}</system>{turns}"

        block = assert_hint_removable(
            render, body, "line one\nline two", family="test")

        self.assertIn("line one\nline two", block)

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"),
                         "torch is an optional training dependency")
    def test_none_static_level_does_not_disable_dynamic_feedback(self):
        from bedrock_rl.train.rollout import play
        task = SimpleNamespace(
            name="test",
            raw={"turn_hint_provider": f"{__name__}:feedback_provider"})
        args = SimpleNamespace(seed=9, turn_hint="direction",
                               turn_hint_p=1.0)

        def fake_turns(model, coder, rollouts, task, hint, args, pool):
            del model, coder, task, hint, args, pool
            for rollout in rollouts:
                rollout.ep = SimpleNamespace(
                    final_reward=lambda: 0.0, success=False,
                    close=lambda: None)

        with patch("bedrock_rl.train.rollout._play_turns", fake_turns):
            rollouts = play(object(), object(), [{"seed": 1}], task,
                            None, args, None)

        self.assertTrue(rollouts[0].turn_hinted)
        self.assertEqual(rollouts[0].hinter.level, "direction")


@unittest.skipUnless(__import__("importlib").util.find_spec("torch"),
                     "torch is an optional training dependency")
class SDFTCoreTests(unittest.TestCase):
    def test_ema_teacher_shares_frozen_base_and_copies_trainable_state(self):
        import torch
        from bedrock_rl.train.sdft import ema_teacher, update_teacher

        student = torch.nn.Module()
        student.register_parameter(
            "base", torch.nn.Parameter(torch.tensor([3.0]),
                                        requires_grad=False))
        student.register_parameter(
            "adapter", torch.nn.Parameter(torch.tensor([1.0])))
        teacher = ema_teacher(student)

        self.assertIs(teacher.base, student.base)
        self.assertIsNot(teacher.adapter, student.adapter)
        student.adapter.data.fill_(5.0)
        update_teacher(teacher, student, 0.25)
        self.assertEqual(float(student.base), 3.0)
        self.assertEqual(float(teacher.base), 3.0)
        self.assertEqual(float(teacher.adapter), 2.0)

    def test_demonstrations_are_paired_to_the_same_world_seed(self):
        from bedrock_rl.train.sdft import (Demonstration,
                                             demonstrations_for)

        demonstrations = [
            Demonstration("route for seven", seed=7),
            Demonstration("alternate route for seven", seed=7),
            Demonstration("route for nine", seed=9),
        ]

        selected = demonstrations_for(
            [{"seed": 9}, {"seed": 7}], demonstrations, random.Random(3))

        self.assertEqual(selected[0], "route for nine")
        self.assertIn(selected[1], {
            "route for seven", "alternate route for seven"})

    def test_missing_paired_demonstration_fails_loudly(self):
        from bedrock_rl.train.sdft import (Demonstration,
                                             demonstrations_for)

        with self.assertRaisesRegex(ValueError, "world seed 9"):
            demonstrations_for(
                [{"seed": 9}], [Demonstration("route", seed=7)],
                random.Random(0))

    def test_random_demonstration_pairing_is_an_explicit_ablation(self):
        from bedrock_rl.train.sdft import demonstrations_for

        selected = demonstrations_for(
            [{"seed": 9}], ["cross-world route"], random.Random(0),
            match="random")

        self.assertEqual(selected, ["cross-world route"])

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"),
                         "torch is an optional training dependency")
    def test_sdft_is_exact_forward_kl_and_skips_prefix(self):
        import torch
        from bedrock_rl.train.sdft import forward_kl
        teacher = torch.log(torch.tensor([[[0.8, 0.2], [0.4, 0.6]]]))
        student = torch.log(torch.tensor([[[0.5, 0.5], [0.4, 0.6]]]))
        mask = torch.ones(1, 2)
        full, _ = forward_kl(student, teacher, mask, skip_tokens=0)
        skipped, metrics = forward_kl(student, teacher, mask, skip_tokens=1)
        self.assertGreater(float(full), 0.0)
        self.assertAlmostEqual(float(skipped), 0.0, places=6)
        self.assertEqual(metrics["sdft/distilled_tokens"], 1.0)


class SDPOLossTests(unittest.TestCase):
    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"),
                         "torch is an optional training dependency")
    def test_sdpo_teacher_rate_is_validated(self):
        from bedrock_rl.train.sdpo import SDPOConfig

        with self.assertRaisesRegex(ValueError, "teacher_update_rate"):
            SDPOConfig(teacher_update_rate=-0.1)
        self.assertEqual(SDPOConfig(teacher_update_rate=0.0)
                         .teacher_update_rate, 0.0)

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"),
                         "torch is an optional training dependency")
    def test_sdpo_step_scores_privileged_view_with_ema_teacher(self):
        import torch
        from bedrock_rl.train.sdpo import SDPOConfig
        from bedrock_rl.train.sdpo_trainer import sdpo_step

        student = object()
        teacher = object()
        logits = torch.nn.Parameter(torch.tensor([[0.0, 0.0]]))
        teacher_logps = torch.log(torch.tensor([[0.8, 0.2]]))
        seen = []

        def fake_logprobs(module, coder, ids, mask, images, device,
                          topk=None, topk_indices=None):
            del coder, ids, mask, images, device, topk
            seen.append(module)
            indices = (torch.tensor([[0, 1]]) if topk_indices is None
                       else topk_indices)
            if module is teacher:
                values = teacher_logps.gather(-1, indices)
                return values[:, 0], values, indices
            values = torch.log_softmax(logits, dim=-1).gather(-1, indices)
            return values[:, 0], values, indices

        views = [{
            "student": ([1], [1]), "teacher": ([2], [1]),
            "images": [], "trajectory": 0, "trajectory_weight": 1.0,
            "sampling_log_probs": None,
        }]
        with patch("bedrock_rl.train.sdpo_trainer.resp_logprobs",
                   fake_logprobs):
            loss = sdpo_step(
                student, teacher, object(), views,
                SDPOConfig(alpha=0.0, distillation_topk=2, is_clip=None),
                "cpu", {})

        self.assertGreater(loss, 0.0)
        self.assertEqual(seen, [student, teacher])
        self.assertIsNotNone(logits.grad)

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"),
                         "torch is an optional training dependency")
    def test_forward_and_reverse_kl_match_closed_form(self):
        import torch
        from bedrock_rl.train.sdpo import SDPOConfig, self_distillation_loss

        student = torch.log(torch.tensor([[[0.5, 0.5]]]))
        teacher = torch.log(torch.tensor([[[0.8, 0.2]]]))
        sampled_student = student[:, :, 0]
        sampled_teacher = teacher[:, :, 0]
        mask = torch.ones(1, 1)

        forward, _ = self_distillation_loss(
            sampled_student, sampled_teacher, mask,
            SDPOConfig(alpha=0.0, distillation_topk=None,
                       distillation_add_tail=False, is_clip=None),
            student_topk_log_probs=student,
            teacher_topk_log_probs=teacher)
        reverse, _ = self_distillation_loss(
            sampled_student, sampled_teacher, mask,
            SDPOConfig(alpha=1.0, distillation_topk=None,
                       distillation_add_tail=False, is_clip=None),
            student_topk_log_probs=student,
            teacher_topk_log_probs=teacher)
        expected_forward = (torch.tensor([0.8, 0.2]) * (
            torch.log(torch.tensor([0.8, 0.2]))
            - torch.log(torch.tensor([0.5, 0.5])))).sum()
        expected_reverse = (torch.tensor([0.5, 0.5]) * (
            torch.log(torch.tensor([0.5, 0.5]))
            - torch.log(torch.tensor([0.8, 0.2])))).sum()

        self.assertAlmostEqual(float(forward), float(expected_forward),
                               places=6)
        self.assertAlmostEqual(float(reverse), float(expected_reverse),
                               places=6)

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"),
                         "torch is an optional training dependency")
    def test_distillation_mask_excludes_whole_rollouts(self):
        import torch
        from bedrock_rl.train.sdpo import SDPOConfig, self_distillation_loss

        student = torch.log(torch.tensor([
            [[0.5, 0.5]], [[0.99, 0.01]],
        ]))
        teacher = torch.log(torch.tensor([
            [[0.8, 0.2]], [[0.01, 0.99]],
        ]))
        cfg = SDPOConfig(alpha=0.0, distillation_topk=None,
                         distillation_add_tail=False, is_clip=None)
        loss, metrics = self_distillation_loss(
            student[:, :, 0], teacher[:, :, 0], torch.ones(2, 1), cfg,
            student_topk_log_probs=student,
            teacher_topk_log_probs=teacher,
            self_distillation_mask=torch.tensor([1.0, 0.0]))
        expected = (torch.tensor([0.8, 0.2]) * (
            torch.log(torch.tensor([0.8, 0.2]))
            - torch.log(torch.tensor([0.5, 0.5])))).sum()

        self.assertAlmostEqual(float(loss), float(expected), places=6)
        self.assertEqual(metrics["sdpo/distilled_tokens"], 1.0)

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"),
                         "torch is an optional training dependency")
    def test_gradient_step_moves_student_toward_teacher(self):
        import torch
        from bedrock_rl.train.sdpo import SDPOConfig, self_distillation_loss

        logits = torch.tensor([[[0.0, 0.0]]], requires_grad=True)
        teacher = torch.log(torch.tensor([[[0.9, 0.1]]]))
        cfg = SDPOConfig(alpha=0.0, distillation_topk=None,
                         distillation_add_tail=False, is_clip=None)

        def objective(values):
            student = torch.log_softmax(values, dim=-1)
            return self_distillation_loss(
                student[:, :, 0], teacher[:, :, 0], torch.ones(1, 1), cfg,
                student_topk_log_probs=student,
                teacher_topk_log_probs=teacher)[0]

        before = objective(logits)
        before.backward()
        updated = (logits - 0.25 * logits.grad).detach()
        after = objective(updated)

        self.assertGreater(float(before.detach()), 0.0)
        self.assertLess(float(after), float(before.detach()))


if __name__ == "__main__":
    unittest.main()
