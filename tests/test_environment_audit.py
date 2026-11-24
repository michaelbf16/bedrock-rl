import os
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bedrock_rl.adapters.netherite.environment import NetheriteEnvironment, NetheriteSession
from bedrock_rl.adapters.netherite.computer import cu_payload
from bedrock_rl.core.reward import EventCountReward
from bedrock_rl.core.sampling import EPISODE_DONE_TOOL_MESSAGE
from bedrock_rl.core.state import StateTrace
from bedrock_rl.core.tools import ToolCall
from bedrock_rl.core.trajectory import Trajectory
from bedrock_rl.env.engine import engine_argv
from bedrock_rl.env.episode import Episode
from bedrock_rl.env.journal import Event, EVENTS
from bedrock_rl.env.task import Task
from bedrock_rl.sdg.media import _assistant_programs


class EnvironmentAuditTests(unittest.TestCase):
    def test_default_trial_requests_rendered_pixels(self):
        path = "examples/equip_pickaxe_grpo/task.yaml"
        for options, enabled in (({}, True), ({"capture_frames": False}, False)):
            with self.subTest(options=options), patch.object(Episode, "draw") as draw:
                NetheriteEnvironment(path, **options)._open_blocking(path)
                self.assertIs(draw.call_args.kwargs["capture_frames"], enabled)
                self.assertEqual("frames_root" in draw.call_args.kwargs, enabled)

    def test_verifiers_see_only_events_after_reset(self):
        aimed_at = Event("aimed_at", EVENTS["aimed_at"], 1, (3, 4, 0, 0, 0, 0))
        journal = [aimed_at]
        session = NetheriteSession(
            SimpleNamespace(journal_start=1, env=SimpleNamespace(journal=journal)), None)
        state = StateTrace()
        reward = EventCountReward("aimed_at")
        state.extend_events(session.drain_events())
        self.assertEqual(reward.evaluate(None, state).total, 0.0)
        journal.append(aimed_at)
        state.extend_events(session.drain_events())
        state.extend_events(session.drain_events())
        self.assertEqual(reward.evaluate(None, state).metrics["event:aimed_at.count"], 1)

    def test_spawn_angles_follow_the_seeded_point_and_per_axis_override(self):
        task = Task({
            "name": "look", "goal": "Look up", "seeds": [4],
            "success": {"type": "pitch_below", "value": -85},
            "initial_state": {"spawn": {"points": [
                {"x": 1, "y": 70, "z": 1, "yaw": 25, "pitch": 12},
                {"x": 2, "y": 70, "z": 2, "yaw": 90, "pitch": -18}]}},
        })
        for seed in range(12):
            point = task.initial_state.point_for(seed)
            self.assertEqual(task.sample_init(random.Random(13), seed),
                             (point.yaw, point.pitch))
        task.init = {"yaw": [33, 33]}
        self.assertEqual(task.sample_init(random.Random(13)),
                         (33, task.initial_state.point_for(4).pitch))

    def test_replay_validates_render_settings_and_legacy_default(self):
        from tests.test_episode_contracts import TickEngine, open_episode

        with patch.object(TickEngine, "render_settings",
                          ("hide_gui=1", "no_hand=1"), create=True):
            episode = open_episode()
            self.addCleanup(episode.close)
            self.assertEqual(episode.spec["view"]["engine_render_set"],
                             "hide_gui=1,no_hand=1")

        class RecordedEpisode(Episode):
            def __init__(self, *args, **kwargs):
                self.spec = {"view": {"type": "procedural", "engine_render_set":
                             os.environ.get("NETHERITE_RENDER_SET", "")}}

            def close(self):
                pass

        spec = {"seed": 0, "init_yaw": 0, "init_pitch": 0, "task_yaml": "world.yaml",
                "view": {"type": "procedural"}}
        with patch.dict(os.environ, NETHERITE_RENDER_SET=""):
            RecordedEpisode.from_spec(spec, parser=None, task=object())
        with patch.dict(os.environ, NETHERITE_RENDER_SET="hide_gui=1,no_hand=1"):
            self.assertEqual(engine_argv("game", 0, 12, 12, 1)[-4:],
                             ["--set", "hide_gui=1", "--set", "no_hand=1"])
            with self.assertRaisesRegex(RuntimeError, "recorded episode view"):
                RecordedEpisode.from_spec(spec, parser=None, task=object())
            spec["view"]["engine_render_set"] = "hide_gui=1,no_hand=1"
            RecordedEpisode.from_spec(spec, parser=None, task=object())

    def test_standalone_trainers_keep_the_recorded_view(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is optional")
        from bedrock_rl.sdg.pipeline import load_episode_specs
        view = {"engine_frame_size": [428, 240], "engine_render_set": "hide_gui=1"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.parquet"
            pq.write_table(pa.Table.from_pylist([{"extra_info": {
                "task_yaml": "world.yaml", "seed": 0, "init_yaw": 0.,
                "init_pitch": 0., "view": view}}]), path)
            self.assertEqual(load_episode_specs(path)[0]["view"], view)

    def test_replay_skips_unexecuted_calls_and_counts_each_turn_once(self):
        arguments = {"actions": [{"action": "wait", "ticks": 1}]}
        def call(key):
            return ToolCall(key, "computer", arguments).to_openai()

        trajectory = Trajectory(messages=[
            {"role": "assistant", "tool_calls": [call("a"), call("b"), call("c")]},
            {"role": "assistant", "tool_calls": [call("b"), call("c")]},
            {"role": "tool", "tool_call_id": "c", "content": EPISODE_DONE_TOOL_MESSAGE},
        ], metadata={"tool_results": [
            {"turn": 0, "call_id": "b", "metadata": {"skipped": True}},
        ]})
        self.assertEqual(_assistant_programs(trajectory), [
            (cu_payload(arguments), True), (cu_payload(arguments), False),
            (cu_payload(arguments), True),
        ])
