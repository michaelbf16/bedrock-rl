import base64
import struct
import copy
import json
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

from bedrock_rl.sdg import (
    TrajectorySFTWriter,
    trajectory_sft_row,
    trajectory_sft_rows,
)
from bedrock_rl.sdg.pipeline import _message_content
from bedrock_rl.adapters.netherite.chat import (SFT_IMAGE_TAG,
                                                prepare_sft_multiturn)
from bedrock_rl.core.messages import (decode_data_image_url, image_url_part,
                                        text_part)
from bedrock_rl.core.reward import RewardResult
from bedrock_rl.core.trajectory import Trajectory


def png(rgb):
    def chunk(kind, data):
        body = kind + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xffffffff))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2,
                                           0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00" + bytes(rgb)))
            + chunk(b"IEND", b""))


FIRST = png((255, 0, 0))
SECOND = png((0, 255, 0))
THIRD = png((0, 0, 255))


def embedded(data):
    payload = base64.b64encode(data).decode("ascii")
    return image_url_part(f"data:image/png;base64,{payload}")


def successful_trajectory():
    messages = [
        {"role": "user", "content": [
            text_part("before"), embedded(FIRST), text_part("middle"),
            embedded(SECOND), text_part("after"),
        ]},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "computer",
                         "arguments": json.dumps({"actions": []})},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "name": "computer",
         "content": [text_part("observation")]},
        {"role": "user", "content": [embedded(THIRD)]},
        {"role": "assistant", "content": "done"},
    ]
    trajectory = Trajectory(
        messages=messages,
        reward=RewardResult(1.25, {"netherite": 1.25},
                            {"success": True, "turns": 2}),
        provenance={
            "generator": "test", "nested": {"version": 1},
            "demonstration_grounding": {
                "eligible": True,
                "target_selection": "public_observation",
                "execution": "public_interface",
            },
        },
        metadata={"generation_case": {"seed": 91,
                                      "decision_seed": 17}},
        id="trajectory-1", finished_at=2.0)
    trajectory.state.record(2, "terminal", {
        "episode": {"success": True, "turns": 2,
                    "spec": {"seed": 91}},
    })
    return trajectory


class TrajectorySFTWriterTests(unittest.TestCase):
    def test_non_writer_string_markers_round_trip_into_sft_rows(self):
        messages = [
            {"role": "user", "content": "<image>\nTASK: farm one log"},
            {"role": "assistant", "content": "act"},
            {"role": "tool", "content": "<image>\nnext observation",
             "tool_call_id": "call_1", "name": "computer"},
        ]

        view, images = prepare_sft_multiturn(messages, [FIRST, SECOND])

        self.assertEqual(images, [FIRST, SECOND])
        self.assertEqual(view[0]["content"], [
            text_part(SFT_IMAGE_TAG), text_part("\nTASK: farm one log")])
        self.assertEqual(view[2]["content"], [
            text_part(SFT_IMAGE_TAG), text_part("\nnext observation")])

    def test_sft_marker_conversion_preserves_online_prompt_bytes(self):
        online = "<image>\nTASK: farm one log"
        view, _ = prepare_sft_multiturn(
            [{"role": "user", "content": online}], [FIRST])
        rendered = "".join(part["text"] for part in view[0]["content"])

        self.assertEqual(rendered.replace(SFT_IMAGE_TAG, "<image>"), online)

    def test_scripted_message_image_switch_is_explicit(self):
        visual = _message_content(FIRST, "state", include_image=True)
        text_only = _message_content(FIRST, "state", include_image=False)

        self.assertEqual(decode_data_image_url(
            visual[0]["image_url"]["url"]), FIRST)
        self.assertEqual(visual[1], text_part("state"))
        self.assertEqual(text_only, [text_part("state")])

    def test_canonical_artifact_validates_openai_image_url_shape(self):
        malformed = Trajectory(messages=[{
            "role": "user", "content": [
                {"type": "image_url", "image_url": "not-a-mapping"}]}])
        with self.assertRaisesRegex(TypeError, "must be a mapping"):
            malformed.to_dict()

        malformed.messages[0]["content"][0] = image_url_part(
            "data:image/png;base64,not base64!")
        with self.assertRaisesRegex(ValueError, "invalid base64"):
            malformed.to_dict()

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"),
                         "torch is an optional training dependency")
    def test_local_trainer_rollout_exports_embedded_openai_images(self):
        from bedrock_rl.train.rollout import Rollout
        rollout = Rollout({"task_yaml": "task.yaml", "seed": 4,
                           "view": {"type": "semantic"}}, group=0)
        rollout.messages = [
            {"role": "user", "content": "<image>\nstart"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "local_0_0", "type": "function",
                "function": {"name": "computer",
                             "arguments": {"actions": []}},
            }]},
            {"role": "tool", "tool_call_id": "local_0_0",
             "name": "computer", "content": "<image>\nafter"},
        ]
        rollout.images = [FIRST, SECOND]
        messages = rollout.canonical_messages()
        self.assertEqual(messages[0]["content"][0]["type"], "image_url")
        self.assertTrue(messages[0]["content"][0]["image_url"]["url"]
                        .startswith("data:image/png;base64,"))
        self.assertEqual(messages[2]["role"], "tool")
        self.assertEqual(messages[3]["role"], "user")
        self.assertEqual(messages[3]["content"][0]["type"], "image_url")
        self.assertEqual(decode_data_image_url(
            messages[0]["content"][0]["image_url"]["url"]), FIRST)
        self.assertEqual(decode_data_image_url(
            messages[3]["content"][0]["image_url"]["url"]), SECOND)
        Trajectory(messages=messages).to_dict()

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"),
                         "torch is an optional training dependency")
    def test_local_multicall_rollout_defers_images_until_all_results(self):
        from bedrock_rl.train.rollout import Rollout
        rollout = Rollout({"task_yaml": "task.yaml", "seed": 4}, group=0)
        rollout.messages = [
            {"role": "user", "content": "<image>\nstart"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "a", "type": "function", "function": {
                    "name": "computer", "arguments": {"actions": []}}},
                {"id": "b", "type": "function", "function": {
                    "name": "computer", "arguments": {"actions": []}}},
            ]},
            {"role": "tool", "tool_call_id": "a", "name": "computer",
             "content": "<image>\nfirst"},
            {"role": "tool", "tool_call_id": "b", "name": "computer",
             "content": "<image>\nsecond"},
        ]
        rollout.images = [FIRST, SECOND, THIRD]

        messages = rollout.canonical_messages()
        self.assertEqual([message["role"] for message in messages],
                         ["user", "assistant", "tool", "tool", "user"])
        self.assertEqual(len(messages[-1]["content"]), 2)
        self.assertEqual(
            [decode_data_image_url(part["image_url"]["url"])
             for part in messages[-1]["content"]], [SECOND, THIRD])
        Trajectory(messages=messages).to_dict()

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"),
                         "torch is an optional training dependency")
    def test_local_rollout_does_not_treat_literal_image_text_as_a_frame(self):
        from bedrock_rl.train.rollout import Rollout
        rollout = Rollout({"task_yaml": "task.yaml", "seed": 4}, group=0)
        rollout.messages = [{
            "role": "user",
            "content": "<image>\nDescribe the literal token <image>.",
        }]
        rollout.images = [FIRST]

        messages = rollout.canonical_messages()

        self.assertEqual(messages[0]["content"][0]["type"], "image_url")
        self.assertEqual(messages[0]["content"][1],
                         text_part("\nDescribe the literal token <image>."))

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"),
                         "torch is an optional training dependency")
    def test_image_window_ignores_literal_image_text(self):
        from bedrock_rl.train.rollout import Rollout
        rollout = Rollout({"task_yaml": "task.yaml", "seed": 4}, group=0)
        rollout.messages = [
            {"role": "user", "content": "<image>\nLiteral <image>."},
            {"role": "user", "content": "<image>"},
        ]
        rollout.images = [FIRST, SECOND]

        with patch.dict("os.environ", {"BRL_KEEP_LATEST_IMAGES": "1"}):
            messages, images = rollout.model_view()

        self.assertEqual(messages[0]["content"], "\nLiteral <image>.")
        self.assertEqual(messages[1]["content"], "<image>")
        self.assertEqual(images, [SECOND])

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"),
                         "torch is an optional training dependency")
    def test_local_model_history_preserves_sampled_assistant_bytes(self):
        from bedrock_rl.train.rollout import Rollout, VLLMSampler
        sampled = ('I will inspect first.\n<tool_call>\n'
                   '{"name": "computer", "arguments": {"actions": []}}\n'
                   '</tool_call>')
        rollout = Rollout({"task_yaml": "task.yaml", "seed": 4}, group=0)
        rollout.messages = [
            {"role": "user", "content": "<image>\nstart"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "local_0_0", "type": "function",
                "function": {"name": "computer",
                             "arguments": "{\"actions\":[]}"},
            }]},
            {"role": "tool", "tool_call_id": "local_0_0",
             "name": "computer", "content": "after"},
        ]
        rollout.assistant_raw = [sampled]
        rollout.images = [FIRST]

        messages, _images = rollout.model_view()

        self.assertEqual(messages[1],
                         {"role": "assistant", "content": sampled})
        self.assertEqual(rollout.canonical_messages()[1]["content"], None)

        class RecordingCoder:
            def render(self, rendered_messages, image_pads, add_generation):
                self.messages = rendered_messages
                self.image_pads = image_pads
                self.add_generation = add_generation
                return rendered_messages[1]["content"]

        coder = RecordingCoder()
        self.assertEqual(VLLMSampler.prompt(coder, rollout), sampled)
        self.assertEqual(coder.messages[1]["content"], sampled)
        self.assertEqual(coder.image_pads, [1])
        self.assertTrue(coder.add_generation)

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"),
                         "torch is an optional training dependency")
    def test_local_rollout_caps_calls_and_resolves_every_emitted_call(self):
        from bedrock_rl.core.sampling import TOOL_CALL_LIMIT_MESSAGE
        from bedrock_rl.train.rollout import Rollout, step_rollout

        class Episode:
            done = False

            def __init__(self):
                self.calls = 0

            def step(self, payload, *, count_turn=True):
                del payload, count_turn
                self.calls += 1
                return "obs", None, 0.0, False

        rollout = Rollout({"task_yaml": "task.yaml", "seed": 4}, group=0)
        rollout.ep = Episode()
        rollout.capture_state = lambda *args: None
        envelope = (
            '<tool_call>{"name":"computer","arguments":'
            '{"actions":[]}}</tool_call>')

        step_rollout(rollout, envelope * 5, max_turns=2)
        tool_messages = [message for message in rollout.messages
                         if message["role"] == "tool"]

        self.assertEqual(rollout.ep.calls, 4)
        self.assertEqual(len(tool_messages), 5)
        self.assertEqual(tool_messages[-1]["content"],
                         TOOL_CALL_LIMIT_MESSAGE)

    def test_extracts_images_in_message_order_without_flattening_tools(self):
        trajectory = successful_trajectory()
        original = copy.deepcopy(trajectory.messages)

        row = trajectory_sft_row(trajectory)
        messages = json.loads(row["messages"])

        self.assertEqual([image["bytes"] for image in row["images"]],
                         [FIRST, SECOND, THIRD])
        self.assertEqual(messages[0]["content"], [
            {"type": "text", "text": "before"},
            {"type": "text", "text": SFT_IMAGE_TAG},
            {"type": "text", "text": "middle"},
            {"type": "text", "text": SFT_IMAGE_TAG},
            {"type": "text", "text": "after"},
        ])
        self.assertEqual(messages[1]["tool_calls"],
                         original[1]["tool_calls"])
        self.assertEqual(messages[1]["content"], "")
        self.assertEqual(messages[2]["role"], "tool")
        self.assertEqual(messages[2]["tool_call_id"], "call_1")
        self.assertEqual(messages[2]["name"], "computer")
        self.assertEqual(messages[2]["content"], [
            {"type": "text", "text": "observation"},
        ])
        self.assertEqual(messages[3]["content"], [
            {"type": "text", "text": SFT_IMAGE_TAG},
        ])
        self.assertEqual(trajectory.messages, original)

    def test_preserves_training_and_audit_metadata(self):
        row = trajectory_sft_row(successful_trajectory().to_dict())

        self.assertEqual(row["reward"], 1.25)
        self.assertEqual(json.loads(row["reward_json"])["components"],
                         {"netherite": 1.25})
        self.assertEqual(row["seed"], 91)
        self.assertEqual(row["turns"], 2)
        self.assertEqual(row["trajectory_id"], "trajectory-1")
        self.assertEqual(row["source_trajectory_id"], "trajectory-1")
        self.assertEqual(row["curriculum_stage"], 2)
        self.assertIs(row["loss_last_assistant_only"], True)
        self.assertEqual(json.loads(row["provenance"])["nested"],
                         {"version": 1})
        self.assertEqual(
            json.loads(row["trajectory_metadata"])["generation_case"]
            ["decision_seed"], 17)

    def test_sft_window_is_causal_and_excludes_post_action_frames(self):
        trajectory = successful_trajectory()
        trajectory.messages.pop()
        row = trajectory_sft_row(trajectory)
        messages, images = prepare_sft_multiturn(
            json.loads(row["messages"]), row["images"],
            last_assistant_only=True, keep_latest_images=1)

        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertEqual(messages[-1]["tool_calls"][0]["id"], "call_1")
        self.assertEqual([image["bytes"] for image in images], [SECOND])
        self.assertEqual(sum(
            part.get("text") == SFT_IMAGE_TAG
            for message in messages
            for part in (message.get("content")
                         if isinstance(message.get("content"), list) else [])
            if isinstance(part, dict)), 1)

    def test_export_writes_one_bounded_causal_row_per_assistant(self):
        rows = list(trajectory_sft_rows(
            successful_trajectory(), keep_latest_images=1))

        self.assertEqual(len(rows), 2)
        self.assertEqual([row["curriculum_stage"] for row in rows], [1, 2])
        self.assertEqual([len(row["images"]) for row in rows], [1, 1])
        for row in rows:
            messages = json.loads(row["messages"])
            self.assertEqual(messages[-1]["role"], "assistant")
            self.assertTrue(row["loss_last_assistant_only"])

    def test_sink_streams_rows_through_shared_parquet_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "sft.parquet")
            with patch("bedrock_rl.sdg.writers.write_rows",
                       return_value=1) as writer:
                result = TrajectorySFTWriter(output, batch_size=3).write(
                    [successful_trajectory()])

        self.assertEqual(result, output)
        writer.assert_called_once()
        rows, path = writer.call_args.args
        self.assertEqual(path, output)
        self.assertNotIsInstance(rows, list)
        self.assertEqual(writer.call_args.kwargs, {"batch_size": 3})
        self.assertEqual(next(rows)["seed"], 91)

    def test_sink_rejects_ungrounded_trajectories_by_default(self):
        trajectory = successful_trajectory()
        trajectory.provenance["demonstration_grounding"]["eligible"] = False
        with tempfile.TemporaryDirectory() as directory:
            writer = TrajectorySFTWriter(str(Path(directory) / "sft.parquet"))
            with patch(
                    "bedrock_rl.sdg.writers.write_rows",
                    side_effect=lambda rows, *_args, **_kwargs: list(rows)):
                with self.assertRaisesRegex(
                        ValueError, "no assistant turn declared"):
                    writer.write([trajectory])

    def test_grounded_export_keeps_only_eligible_assistant_turns(self):
        trajectory = successful_trajectory()
        trajectory.provenance["demonstration_grounding"]["eligible"] = False
        trajectory.messages[1]["metadata"] = {
            "demonstration_grounding": {"eligible": True},
            "policy_stage": "mine:logs",
        }
        trajectory.messages[4]["metadata"] = {
            "demonstration_grounding": {"eligible": False},
            "policy_stage": "mine:diamonds",
        }

        rows = list(trajectory_sft_rows(
            trajectory, keep_latest_images=1, require_grounded=True))

        self.assertEqual(len(rows), 1)
        messages = json.loads(rows[0]["messages"])
        self.assertEqual(messages[-1]["metadata"]["policy_stage"],
                         "mine:logs")

    def test_rejects_unfinished_failed_and_unscored_trajectories(self):
        cases = []
        unfinished = successful_trajectory()
        unfinished.finished_at = None
        cases.append((unfinished, "unfinished"))
        failed = successful_trajectory()
        failed.reward.metrics["success"] = False
        cases.append((failed, "non-success"))
        unscored = successful_trajectory()
        unscored.reward = None
        cases.append((unscored, "without reward"))

        for trajectory, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    trajectory_sft_row(trajectory)

    def test_rejects_trajectory_with_fabricated_malformed_call_payload(self):
        trajectory = successful_trajectory()
        trajectory.messages[1]["metadata"] = {
            "malformed_tool_calls": {
                "call_1": {"id": "call_1", "function": {
                    "name": "computer", "arguments": "{"}},
            },
        }
        with self.assertRaisesRegex(ValueError, "malformed tool calls"):
            trajectory_sft_row(trajectory)

    def test_rejects_malformed_image_and_tool_transcripts(self):
        external = successful_trajectory()
        external.messages[0]["content"][1] = image_url_part(
            "https://example.test/frame.png")
        with self.assertRaisesRegex(ValueError, "embedded data-URL"):
            trajectory_sft_row(external)

        bad_base64 = successful_trajectory()
        bad_base64.messages[0]["content"][1] = image_url_part(
            "data:image/png;base64,not base64!")
        with self.assertRaisesRegex(ValueError, "invalid base64"):
            trajectory_sft_row(bad_base64)

        wrong_payload = successful_trajectory()
        wrong_payload.messages[0]["content"][1] = image_url_part(
            "data:image/png;base64," +
            base64.b64encode(b"not a png").decode("ascii"))
        with self.assertRaisesRegex(ValueError, "does not match"):
            trajectory_sft_row(wrong_payload)

        runtime_part = successful_trajectory()
        runtime_part.messages[0]["content"][1] = {"type": "image"}
        with self.assertRaisesRegex(ValueError, "canonical OpenAI chat"):
            trajectory_sft_row(runtime_part)

        wrong_role = successful_trajectory()
        wrong_role.messages[2]["content"].append(embedded(THIRD))
        with self.assertRaisesRegex(ValueError, "only on user messages"):
            trajectory_sft_row(wrong_role)

        orphan = successful_trajectory()
        orphan.messages[2]["tool_call_id"] = "missing"
        with self.assertRaisesRegex(ValueError, "unknown or already-resolved"):
            trajectory_sft_row(orphan)

        unresolved = successful_trajectory()
        del unresolved.messages[2]
        with self.assertRaisesRegex(ValueError, "before tool results"):
            trajectory_sft_row(unresolved)

    def test_preserves_literal_image_markers_without_confusing_sft_frames(self):
        trajectory = successful_trajectory()
        trajectory.messages[0]["content"][0]["text"] = "<image>"

        row = trajectory_sft_row(trajectory)
        messages = json.loads(row["messages"])

        self.assertEqual(messages[0]["content"][0],
                         text_part("<image>"))
        self.assertEqual(messages[0]["content"][1],
                         text_part(SFT_IMAGE_TAG))


if __name__ == "__main__":
    unittest.main()
