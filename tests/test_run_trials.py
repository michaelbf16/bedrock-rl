import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


from bedrock_rl.eval import trials as RUN_TRIALS


ROOT = Path(__file__).resolve().parents[1]


class RunTrialsOutputTests(unittest.TestCase):
    def test_generation_failure_isolated_to_its_trial(self):
        failed = []

        def roll(name):
            return SimpleNamespace(
                name=name,
                fail=lambda doing, error: failed.append(
                    (name, doing, type(error).__name__)))

        rolls = [roll("a"), roll("bad"), roll("c")]

        def generate(requests):
            if "bad" in requests:
                raise ValueError(
                    "At most 4 image(s) may be provided in one prompt")
            return [f"output:{request}" for request in requests]

        sampled = RUN_TRIALS.generate_isolated(
            rolls, ["a", "bad", "c"], generate)

        self.assertEqual([(item.name, output) for item, output in sampled],
                         [("a", "output:a"), ("c", "output:c")])
        self.assertEqual(failed, [
            ("bad", "the model could not sample this trial", "ValueError")])

    def test_global_generation_failure_is_not_retried_at_a_new_batch_size(self):
        failures = []
        rolls = [SimpleNamespace(fail=lambda *args: failures.append(args))
                 for _ in range(2)]
        calls = []

        def generate(requests):
            calls.append(list(requests))
            raise RuntimeError("CUDA out of memory")

        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            RUN_TRIALS.generate_isolated(rolls, ["a", "b"], generate)

        self.assertEqual(calls, [["a", "b"]])
        self.assertEqual(failures, [])

    def test_t0_typed_image_conversion_preserves_separator_byte(self):
        self.assertEqual(
            RUN_TRIALS.without_leading_image_marker("<image>\nTASK: mine"),
            "\nTASK: mine")

    def test_explicit_task_must_match_frozen_rows(self):
        task = ROOT / "examples" / "equip_pickaxe_grpo" / "task.yaml"
        other = ROOT / "examples" / "farm_one_log_sdg" / "task.yaml"
        rows = [{"extra_info": {"task_yaml": str(task)}}]
        self.assertEqual(
            Path(RUN_TRIALS.evaluation_task(rows, str(task)).path), task)
        with self.assertRaisesRegex(SystemExit, "refusing to score"):
            RUN_TRIALS.evaluation_task(rows, str(other))

    def test_eval_rejects_an_empty_frozen_set(self):
        with self.assertRaisesRegex(SystemExit, "no prompt rows"):
            RUN_TRIALS.evaluation_task([])

    def test_evaluation_executes_every_tool_call_in_one_assistant_turn(self):
        class Episode:
            done = False

            def __init__(self):
                self.calls = []

            def step(self, payload, *, count_turn=True):
                self.calls.append((json.loads(payload), count_turn))
                return "obs", None, 0.0, False

        episode = Episode()
        text = (
            '<tool_call>{"name":"computer","arguments":{"actions":['
            '{"action":"wait","ticks":1}]}}</tool_call>'
            '<tool_call>{"name":"computer","arguments":{"actions":['
            '{"action":"wait","ticks":2}]}}</tool_call>')
        calls, _, _, _ = RUN_TRIALS.execute_payloads(episode, text)
        self.assertEqual(len(calls), 2)
        self.assertEqual([count for _payload, count in episode.calls],
                         [True, False])

    def test_evaluation_caps_calls_and_resolves_every_emitted_call(self):
        class Episode:
            done = False

            def __init__(self):
                self.calls = 0

            def step(self, payload, *, count_turn=True):
                del payload, count_turn
                self.calls += 1
                return "obs", None, 0.0, False

        episode = Episode()
        envelope = (
            '<tool_call>{"name":"computer","arguments":'
            '{"actions":[]}}</tool_call>')
        calls, messages, _frames, _done = RUN_TRIALS.execute_turn(
            episode, envelope * 5)
        tool_messages = [message for message in messages
                         if message["role"] == "tool"]

        self.assertEqual(len(calls), 5)
        self.assertEqual(len(tool_messages), 5)
        self.assertEqual(episode.calls, 4)
        self.assertEqual(
            tool_messages[-1]["content"],
            RUN_TRIALS.TOOL_CALL_LIMIT_MESSAGE)

    def test_multi_call_eval_context_matches_training_turn_shape(self):
        class Episode:
            done = False

            def __init__(self):
                self.calls = 0

            def step(self, payload, *, count_turn=True):
                del payload, count_turn
                self.calls += 1
                return (f"obs {self.calls}", f"frame {self.calls}",
                        0.0, False)

        text = (
            '<tool_call>{"name":"computer","arguments":{"actions":['
            '{"action":"wait","ticks":1}]}}</tool_call>'
            '<tool_call>{"name":"computer","arguments":{"actions":['
            '{"action":"wait","ticks":2}]}}</tool_call>')
        calls, messages, frames, done = RUN_TRIALS.execute_turn(
            Episode(), text, 3)

        self.assertFalse(done)
        self.assertEqual([call["id"] for call in calls],
                         ["eval_3_0", "eval_3_1"])
        self.assertEqual(messages[0]["content"], text)
        self.assertNotIn("tool_calls", messages[0])
        self.assertEqual(
            [message["tool_call_id"] for message in messages
             if message["role"] == "tool"],
            ["eval_3_0", "eval_3_1"])
        self.assertIn("obs 1", messages[1]["content"])
        self.assertIn("obs 2", messages[2]["content"])
        self.assertEqual(
            [message["role"] for message in messages],
            ["assistant", "tool", "tool", "user", "user"])
        self.assertEqual(frames, ["frame 1", "frame 2"])

    def test_malformed_and_unknown_tool_envelopes_spend_a_turn(self):
        class Episode:
            done = False

            def __init__(self):
                self.payloads = []

            def step(self, payload, *, count_turn=True):
                self.payloads.append((payload, count_turn))
                return "parse error", None, -0.2, False

        for text in (
                "<tool_call>{bad json}</tool_call>",
                '<tool_call>{"name":"invented","arguments":{}}'
                "</tool_call>"):
            with self.subTest(text=text):
                episode = Episode()
                calls, _messages, _frames, _done = RUN_TRIALS.execute_turn(
                    episode, text)
                self.assertEqual(len(calls), 1)
                self.assertEqual(episode.payloads, [("{", True)])

    def test_engine_failure_is_not_reclassified_as_model_malformation(self):
        episode = SimpleNamespace(
            done=False,
            step=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("engine died")))
        text = ('<tool_call>{"name":"computer","arguments":'
                '{"actions":[]}}</tool_call>')

        with self.assertRaisesRegex(RuntimeError, "engine died"):
            RUN_TRIALS.execute_turn(episode, text)

    def test_eval_no_call_uses_same_refusal_floor_as_training(self):
        episode = SimpleNamespace(
            success=False, failed=False, turns=0,
            final_reward=lambda: 0.0,
            env=SimpleNamespace(journal=[]))
        rollout = SimpleNamespace(
            ep=episode, broken=None, seed=1, row_index=0, k=0,
            init_yaw=0.0, init_pitch=0.0, turns=["plain prose"],
            protocol={})

        result = RUN_TRIALS.record(rollout)

        self.assertEqual(result["reward"], RUN_TRIALS.NO_TOOL_CALL_REWARD)

    def test_image_window_is_explicit_not_ambient(self):
        args = RUN_TRIALS.parser().parse_args(
            ["--data", "d", "--model", "m"])
        self.assertEqual(args.keep_latest_images, 0)

    def test_vllm_uses_registered_vision_backend(self):
        kwargs = RUN_TRIALS.vllm_engine_kwargs(
            "qwen3-vl-2b", {"enable_lora": True})
        self.assertEqual(kwargs["mm_encoder_attn_backend"], "TORCH_SDPA")
        self.assertTrue(kwargs["enable_lora"])

    def test_outputs_create_nested_parent_directories(self):
        record = {
            "seed": 1,
            "success": False,
            "actions": [],
            "broken": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dump = root / "actions" / "dev" / "actions.json"
            records = root / "records" / "dev" / "records.json"

            RUN_TRIALS.write_dump([record], str(dump))
            args = SimpleNamespace(
                dump_actions=None,
                records_out=str(records),
            )
            RUN_TRIALS.report(args, [record], whole=False)

            self.assertEqual(json.loads(dump.read_text())[0]["seed"], 1)
            self.assertEqual(json.loads(records.read_text())[0]["seed"], 1)


if __name__ == "__main__":
    unittest.main()
