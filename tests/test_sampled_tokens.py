import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bedrock_rl.adapters.netherite.chat import (
    Coder, IMAGE_TAG, SFT_IMAGE_TAG, normalize_legacy_sft_markers,
    template_messages,
)


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return "|".join(
            f"{message['role']}:{message.get('content') or ''}"
            for message in messages)

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return SimpleNamespace(input_ids=[ord(char) for char in text])


class SampledTokenTests(unittest.TestCase):
    def coder(self):
        processor = SimpleNamespace(
            tokenizer=_Tokenizer(), image_processor=object())
        return Coder(processor, tools=[], spec=None)

    def test_sampled_response_ids_are_never_retokenized(self):
        coder = self.coder()
        # These deliberately bear no relation to the rendered tool call.
        # A parse/re-render path cannot reproduce them.
        response = [70001, 42, 70002]
        view = coder.sampled_view(
            [10, 11], response,
            [{"role": "system", "content": "privileged"},
             {"role": "user", "content": "screen"}], ())
        self.assertEqual(view["student"][0][-3:], response)
        self.assertEqual(view["teacher"][0][-3:], response)
        self.assertEqual(view["student"][1], [0, 0, 1, 1, 1])
        self.assertEqual(view["teacher"][1][-3:], [1, 1, 1])

    def test_model_template_view_normalizes_null_tool_call_content(self):
        messages = [{"role": "assistant", "content": None,
                     "tool_calls": [{"id": "call_0"}]}]
        prepared = template_messages(messages)
        self.assertEqual(prepared[0]["content"], "")
        self.assertIsNone(messages[0]["content"])

    def test_coder_expands_only_reserved_image_placeholders(self):
        processor = SimpleNamespace(
            tokenizer=_Tokenizer(), image_processor=object())
        spec = SimpleNamespace(key="test", image_span=("[", "P", "]"))
        coder = Coder(processor, tools=[], spec=spec)

        legacy = coder.render([{
            "role": "user",
            "content": "<image>\nThe literal spelling is <image>.",
        }], [2])
        self.assertIn("[PP]\nThe literal spelling is <image>.", legacy)

        typed = coder.render([{
            "role": "user",
            "content": [
                {"type": "text", "text": SFT_IMAGE_TAG},
                {"type": "text", "text": "literal <image>"},
                {"type": "text", "text": "<image>"},
            ],
        }], [1])
        self.assertIn("[P]", typed)
        self.assertIn("literal <image>", typed)
        self.assertEqual(typed.count("<image>"), 2)

        system_literal = coder.render([
            {"role": "system", "content": "<image> means a screenshot"},
            {"role": "user", "content": "<image>\nlook"},
        ], [1])
        self.assertIn("system:<image> means a screenshot", system_literal)
        self.assertIn("user:[P]\nlook", system_literal)

    def test_legacy_typed_sft_image_markers_are_upgraded_unambiguously(self):
        legacy = [{"role": "user", "content": [
            {"type": "text", "text": IMAGE_TAG},
            {"type": "text", "text": "task"},
        ]}]
        migrated = normalize_legacy_sft_markers(legacy, 1)
        self.assertEqual(migrated[0]["content"][0]["text"], SFT_IMAGE_TAG)
        self.assertEqual(legacy[0]["content"][0]["text"], IMAGE_TAG)

        canonical = [{"role": "user", "content": [
            {"type": "text", "text": IMAGE_TAG},
            {"type": "text", "text": SFT_IMAGE_TAG},
        ]}]
        preserved = normalize_legacy_sft_markers(canonical, 1)
        self.assertEqual(preserved, canonical)

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"),
                         "torch is an optional training dependency")
    def test_vllm_sampling_log_probs_follow_chosen_tokens(self):
        from bedrock_rl.train.rollout import VLLMSampler

        positions = [
            {4: SimpleNamespace(logprob=-0.25)},
            {7: SimpleNamespace(logprob=-1.5)},
        ]
        self.assertEqual(VLLMSampler.chosen_log_probs((4, 7), positions),
                         (-0.25, -1.5))

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"),
                         "torch is an optional training dependency")
    def test_vllm_engine_explicitly_requests_raw_policy_logprobs(self):
        from bedrock_rl.train.rollout import VLLMSampler

        seen = {}

        class LLM:
            def __init__(self, **kwargs):
                seen.update(kwargs)

        with tempfile.TemporaryDirectory() as directory, patch.dict(
                sys.modules, {"vllm": SimpleNamespace(LLM=LLM)}), patch(
                    "bedrock_rl.models.weights_path",
                    return_value="/immutable/model"):
            sampler = VLLMSampler(
                "/immutable/model", SimpleNamespace(spec=None),
                lora_rank=8, workdir=directory, max_turns=2,
                say=lambda *_: None)
            self.assertEqual(seen["logprobs_mode"], "raw_logprobs")
            self.assertTrue(seen["enable_lora"])
            sampler.close()
            self.assertTrue(Path(directory).is_dir())

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"),
                         "torch is an optional training dependency")
    def test_vllm_sampling_refuses_processor_prompt_length_drift(self):
        from bedrock_rl.train.rollout import StalePolicyError, VLLMSampler

        VLLMSampler.verify_prompt_length((10, 11), (20, 21))
        with self.assertRaisesRegex(StalePolicyError,
                                    "sampled from a prompt of length 1"):
            VLLMSampler.verify_prompt_length((10, 11), (20,))
        with self.assertRaisesRegex(StalePolicyError, "length missing"):
            VLLMSampler.verify_prompt_length((10, 11), None)

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"),
                         "torch is an optional training dependency")
    def test_sdft_builds_one_exact_view_per_sampled_turn(self):
        from bedrock_rl.train.rollout import SampledTurn
        from bedrock_rl.train.sdft import SDFTConfig, build_views

        turn = SampledTurn(
            "prose plus oddly spaced call", (10, 11), (91, 92, 93),
            ({"role": "user", "content": "screen"},),
            ({"role": "user", "content": "screen"},), (), ())
        rollout = SimpleNamespace(sampled_turns=[turn])
        views = build_views(self.coder(), [rollout], ["expert"],
                            SDFTConfig())
        self.assertEqual(len(views), 1)
        self.assertEqual(views[0]["student"][0][-3:], [91, 92, 93])
        self.assertEqual(views[0]["teacher"][0][-3:], [91, 92, 93])
        self.assertEqual(views[0]["skip_tokens"], 3)

        second = SampledTurn(
            "short", (10, 11), (94,),
            ({"role": "user", "content": "screen"},),
            ({"role": "user", "content": "screen"},), (), ())
        rollout.sampled_turns.append(second)
        views = build_views(self.coder(), [rollout], ["expert"],
                            SDFTConfig())
        self.assertEqual([view["skip_tokens"] for view in views], [3, 0])
        self.assertEqual([view["trajectory_weight"] for view in views],
                         [0.0, 1.0])

        short_first = SampledTurn(
            "short", (10,), (95, 96),
            ({"role": "user", "content": "screen"},),
            ({"role": "user", "content": "screen"},), (), ())
        rollout.sampled_turns = [short_first, turn]
        views = build_views(self.coder(), [rollout], ["expert"],
                            SDFTConfig())
        self.assertEqual([view["skip_tokens"] for view in views], [2, 1])
        self.assertEqual([view["trajectory_weight"] for view in views],
                         [0.0, 1.0])

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"),
                         "torch is an optional training dependency")
    def test_sdpo_builds_exact_views_for_prose_and_tool_tokens(self):
        from bedrock_rl.train.rollout import SampledTurn
        from bedrock_rl.train.sdpo import SDPOConfig
        from bedrock_rl.train.sdpo_trainer import build_views

        turn = SampledTurn(
            'analysis text\n<tool_call>{  "name": "computer" }</tool_call>',
            (10, 11), (801, 17, 802),
            ({"role": "user", "content": "screen"},),
            ({"role": "user", "content": "screen\n\nturn hint"},), (), ())
        second = SampledTurn(
            "short", (10, 11), (803,),
            ({"role": "user", "content": "screen"},),
            ({"role": "user", "content": "screen"},), (), ())
        rollout = SimpleNamespace(
            sampled_turns=[turn, second], group=0, success=True, hinter=None,
            response_text=lambda: turn.text)
        views = build_views(
            self.coder(), [rollout], "static hint", SDPOConfig())
        self.assertEqual(len(views), 2)
        self.assertEqual(views[0]["student"][0][-3:], [801, 17, 802])
        self.assertEqual(views[0]["teacher"][0][-3:], [801, 17, 802])
        self.assertEqual([view["trajectory_weight"] for view in views],
                         [0.75, 0.25])

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"),
                         "torch is an optional training dependency")
    def test_sdpo_static_none_keeps_dynamic_teacher_feedback(self):
        from bedrock_rl.train.rollout import SampledTurn
        from bedrock_rl.train.sdpo import SDPOConfig
        from bedrock_rl.train.sdpo_trainer import build_views

        turn = SampledTurn(
            "move", (10,), (801,),
            ({"role": "user", "content": "screen"},),
            ({"role": "user", "content": "screen\n\nturn direction"},),
            (), ())
        rollout = SimpleNamespace(
            sampled_turns=[turn], group=0, success=False, hinter=None,
            response_text=lambda: turn.text)

        views = build_views(
            self.coder(), [rollout], None,
            SDPOConfig(teacher_hint=False, teacher_turn_hint=True,
                       teacher_demo=False))

        self.assertTrue(views[0]["distil"])
        self.assertNotEqual(views[0]["student"][0],
                            views[0]["teacher"][0])
        self.assertEqual(views[0]["trajectory_weight"], 1.0)

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"),
                         "torch is an optional training dependency")
    def test_sdpo_static_guidance_distils_an_all_failure_group(self):
        from bedrock_rl.train.rollout import SampledTurn
        from bedrock_rl.train.sdpo import SDPOConfig
        from bedrock_rl.train.sdpo_trainer import build_views

        turn = SampledTurn(
            "move", (10,), (801,),
            ({"role": "user", "content": "screen"},),
            ({"role": "user", "content": "screen"},), (), ())
        rollout = SimpleNamespace(
            sampled_turns=[turn], group=0, success=False, hinter=None,
            response_text=lambda: turn.text)

        views = build_views(
            self.coder(), [rollout], "private procedure",
            SDPOConfig(teacher_hint=True, teacher_turn_hint=False,
                       teacher_demo=False))

        self.assertTrue(views[0]["distil"])
        self.assertNotEqual(views[0]["student"][0],
                            views[0]["teacher"][0])
        self.assertEqual(views[0]["trajectory_weight"], 1.0)

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"),
                         "torch is an optional training dependency")
    def test_sdpo_success_without_private_context_is_not_a_target(self):
        from bedrock_rl.train.rollout import SampledTurn
        from bedrock_rl.train.sdpo import SDPOConfig
        from bedrock_rl.train.sdpo_trainer import build_views

        turn = SampledTurn(
            "move", (10,), (801,),
            ({"role": "user", "content": "screen"},),
            ({"role": "user", "content": "screen"},), (), ())
        rollout = SimpleNamespace(
            sampled_turns=[turn], group=0, success=True, hinter=None,
            response_text=lambda: turn.text)

        views = build_views(
            self.coder(), [rollout], None,
            SDPOConfig(teacher_hint=False, teacher_turn_hint=False,
                       teacher_demo=True))

        self.assertFalse(views[0]["distil"])
        self.assertEqual(views[0]["trajectory_weight"], 0.0)

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"),
                         "torch is an optional training dependency")
    def test_ema_teacher_preserves_fp32_master_values(self):
        import torch
        from bedrock_rl.train.sdft import ema_teacher, update_teacher

        student = torch.nn.Linear(1, 1, bias=False).to(torch.float32)
        teacher = ema_teacher(student)
        self.assertEqual(teacher.weight.dtype, torch.float32)
        self.assertFalse(teacher.weight.requires_grad)
        before = teacher.weight.detach().clone()
        student.weight.data.add_(1e-4)
        update_teacher(teacher, student, 0.01)
        self.assertFalse(torch.equal(teacher.weight, before))

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"),
                         "torch is an optional training dependency")
    def test_continued_adapter_master_is_lifted_to_fp32(self):
        import torch
        from bedrock_rl.train.peft_util import lift_trainable

        model = torch.nn.Linear(1, 1).to(torch.bfloat16)
        model.bias.requires_grad_(False)
        lift_trainable(model, torch.float32)
        self.assertEqual(model.weight.dtype, torch.float32)
        self.assertEqual(model.bias.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
