import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bedrock_rl.sdg.media import (
    align_frame_sequences, compose_labeled_panels, replay_trajectory_view,
    save_frames_gif, save_replayed_views_gif, save_semantic_gif,
    save_trajectory_gif, semantic_image,
)
from bedrock_rl.core.messages import data_image_part
from bedrock_rl.sdg import TrajectoryArtifactWriter
from bedrock_rl.core.trajectory import Trajectory
from bedrock_rl.env.engine import CAM_H, CAM_W


class SemanticMediaTests(unittest.TestCase):
    def sample(self, block, depth=0, edge=0):
        return [block] * (CAM_W * CAM_H), [depth] * (CAM_W * CAM_H), \
            [edge] * (CAM_W * CAM_H)

    def test_palette_depth_edges_and_nearest_neighbor_scale(self):
        camera, depth, edge = self.sample(2)
        image = semantic_image(camera, depth, edge, width=128)
        self.assertEqual(image.size, (128, 72))
        self.assertEqual(image.getpixel((0, 0)), (95, 159, 53))

        image = semantic_image(*self.sample(2, depth=256, edge=1), width=64)
        self.assertEqual(image.getpixel((0, 0)), (26, 44, 15))

    def test_gif_comes_from_saved_state_samples(self):
        trajectory = Trajectory()
        for turn, block in enumerate((2, 49, 87)):
            camera, depth, edge = self.sample(block)
            trajectory.state.record(turn, "after_tools", {"observation": {
                "cam": camera, "depth": depth, "edge": edge,
            }})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preview.gif"
            save_semantic_gif(trajectory, path, width=64)
            self.assertGreater(path.stat().st_size, 0)

    def test_compact_single_trajectory_is_still_plain_json(self):
        trajectory = Trajectory()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.json"
            TrajectoryArtifactWriter(path, compact=True).write([trajectory])
            text = path.read_text()
            self.assertEqual(text.count("\n"), 1)
            self.assertEqual(
                json.loads(text)["schema"],
                "bedrock.trajectory.v1")

    def test_repository_preview_can_redact_images_without_mutating_source(self):
        from PIL import Image
        import io

        output = io.BytesIO()
        Image.new("RGB", (2, 2), "violet").save(output, format="PNG")
        trajectory = Trajectory(messages=[{
            "role": "user", "content": [data_image_part(output.getvalue())],
        }])
        original = trajectory.messages[0]["content"][0]["image_url"]["url"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.json"
            TrajectoryArtifactWriter(
                path, compact=True, redact_images=True).write([trajectory])
            saved = Trajectory.load(path)
        redacted = saved.messages[0]["content"][0]["image_url"]["url"]
        self.assertEqual(
            redacted, "https://bedrock-rl.invalid/base64_redacted")
        self.assertEqual(
            trajectory.messages[0]["content"][0]["image_url"]["url"],
            original)
        self.assertEqual(saved.metadata["artifact_images"]["count"], 1)

    def test_artifact_gif_uses_embedded_model_view_not_semantic_state(self):
        from PIL import Image
        import io

        def png(color):
            output = io.BytesIO()
            Image.new("RGB", (8, 6), color).save(output, format="PNG")
            return output.getvalue()

        trajectory = Trajectory(messages=[
            {"role": "user", "content": [data_image_part(png("red"))]},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "computer", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "call_1",
             "content": "observation"},
            {"role": "user", "content": [data_image_part(png("blue"))]},
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "view.gif"
            save_trajectory_gif(trajectory, path, width=16)
            with Image.open(path) as image:
                self.assertEqual(image.size, (16, 12))
                self.assertEqual(image.n_frames, 2)

    def test_dense_recording_drives_gif_but_is_not_leaked_to_json(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording = root / "dense.recording"
            frames = recording / "seed7_7_1"
            frames.mkdir(parents=True)
            Image.new("RGB", (8, 6), "red").save(
                frames / "frame_000002.ppm")
            Image.new("RGB", (8, 6), "blue").save(
                frames / "frame_000010.ppm")
            trajectory = Trajectory(metadata={
                "_recording_frames_dir": str(recording),
                "recording_every": 2,
            })
            output = root / "trajectory.json"
            gif = root / "dense.gif"
            TrajectoryArtifactWriter(
                output, gif=gif, gif_width=16, compact=True).write(
                    [trajectory])
            self.assertTrue(gif.is_file())
            self.assertNotIn("_recording_frames_dir", output.read_text())
            self.assertFalse(recording.exists())

    def test_artifact_sink_streams_many_records_and_cleans_recordings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []

            def records():
                for index in range(500):
                    recording = root / f"recording-{index}"
                    recording.mkdir()
                    paths.append(recording)
                    yield Trajectory(
                        id=f"trajectory-{index}",
                        metadata={"_recording_frames_dir": str(recording)},
                    )
                    # The previous non-preview recording is released before
                    # the producer is asked for the next record.
                    if index > 0:
                        self.assertFalse(recording.exists())

            output = root / "trajectories.jsonl"
            TrajectoryArtifactWriter(output, compact=True).write(records())
            self.assertEqual(len(output.read_text().splitlines()), 500)
            self.assertTrue(all(not path.exists() for path in paths))

    def test_replay_panel_alignment_holds_shorter_view_and_labels(self):
        from PIL import Image

        semantic = [Image.new("RGB", (16, 16), "red")]
        procedural = [Image.new("RGB", (16, 16), "blue"),
                      Image.new("RGB", (16, 16), "green")]
        aligned = align_frame_sequences([semantic, procedural])
        self.assertEqual([len(value) for value in aligned], [2, 2])
        self.assertIs(aligned[0][0], aligned[0][1])
        panels = compose_labeled_panels(
            [semantic, procedural], ["Semantic", "Netherite"],
            panel_width=32)
        self.assertEqual(len(panels), 2)
        self.assertEqual(panels[0].size, (64, 32))
        # Padding holds the red semantic view while the other view advances.
        self.assertEqual(panels[1].getpixel((16, 30)), (255, 0, 0))
        self.assertEqual(panels[1].getpixel((48, 30)), (0, 128, 0))

    def test_portal_panel_dimensions_are_github_readable_1284_by_240(self):
        from PIL import Image

        source = [[Image.new("RGB", (16, 9), color)]
                  for color in ("red", "green", "blue")]
        frame = compose_labeled_panels(
            source, ["Semantic", "Netherite", "Minecraft"],
            panel_width=428)[0]
        self.assertEqual(frame.size, (1284, 240))

    def test_labeled_panels_hold_geometry_across_gui_aspect_changes(self):
        from PIL import Image

        sequences = [
            [Image.new("RGB", (16, 9), "red"),
             Image.new("RGB", (4, 3), "blue")],
            [Image.new("RGB", (16, 9), "green"),
             Image.new("RGB", (16, 9), "green")],
        ]

        frames = compose_labeled_panels(
            sequences, ["Semantic", "Minecraft"], panel_width=32)

        self.assertEqual([frame.size for frame in frames], [(64, 18)] * 2)

    def test_replay_pixels_wait_for_exact_tick_not_newest_capture(self):
        from PIL import Image
        from bedrock_rl.sdg.media import (
            _ReplayFrameCollector,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            class Env:
                journal = []
                ticks = 1
                frame_w = 2
                frame_h = 1
                timeout = 1.0
                failed = False
                _frames_path = root

                def describe(self):
                    return "fake periodic replay"

            env = Env()
            collector = _ReplayFrameCollector("minecraft-official", 5)
            collector(env)  # t0 is intentionally not part of the replay.
            # A later capture is already discoverable while the capture for
            # the current tick has not appeared yet. Asking for "newest"
            # here used to serve frame 10 for tick 5 and then duplicate it.
            Image.new("RGB", (2, 1), "blue").save(
                root / "frame_000010.ppm")

            for tick in range(2, 5):
                env.ticks = tick
                collector(env)

            expected = root / "frame_000005.ppm"

            def finish_delayed_capture(_seconds):
                Image.new("RGB", (2, 1), "green").save(expected)

            env.ticks = 5
            with mock.patch(
                    "bedrock_rl.sdg.media.time.sleep",
                    side_effect=finish_delayed_capture) as sleep:
                collector(env)
            sleep.assert_called_once()

            for tick in range(6, 11):
                env.ticks = tick
                collector(env)
            self.assertEqual(collector.frame_indices, [5, 10])
            self.assertEqual(
                [frame.getpixel((0, 0)) for frame in collector.frames],
                [(0, 128, 0), (0, 0, 255)])

    def test_replay_semantic_frame_is_sampled_on_its_capture_tick(self):
        from bedrock_rl.sdg.media import (
            _ReplayFrameCollector,
        )

        class Env:
            journal = []
            ticks = 1

            def __init__(self, observation):
                self.obs = observation

        def observation(block):
            camera, depth, edge = self.sample(block)
            return {"cam": camera, "depth": depth, "edge": edge}

        env = Env(observation(1))
        collector = _ReplayFrameCollector("semantic", 2)
        collector(env)
        env.ticks = 2
        env.obs = observation(2)
        collector(env)
        # Simulate discovery occurring only after the observation has moved
        # on. Semantic replay must already own tick 2's exact model state.
        env.ticks = 3
        env.obs = observation(49)
        collector(env)
        env.ticks = 4
        env.obs = observation(87)
        collector(env)

        self.assertEqual(collector.frame_indices, [2, 4])
        self.assertEqual(
            [frame.getpixel((0, 0)) for frame in collector.frames],
            [(95, 159, 53), (112, 52, 49)])

    def test_streaming_gif_accepts_one_shot_frames_and_coalesces_duplicates(
            self):
        from PIL import Image

        yielded = []

        def frames():
            for color in ("red", "green", "green", "blue"):
                yielded.append(color)
                yield Image.new("RGB", (8, 6), color)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "streamed.gif"
            save_frames_gif(
                frames(), path, duration_ms=30, final_hold_ms=90,
                max_bytes=10_000)
            self.assertEqual(yielded, ["red", "green", "green", "blue"])
            with Image.open(path) as image:
                self.assertEqual(image.n_frames, 3)
                self.assertEqual(image.size, (8, 6))
                durations = []
                pixels = []
                for index in range(image.n_frames):
                    image.seek(index)
                    durations.append(image.info["duration"])
                    pixels.append(image.convert("RGB").getpixel((0, 0)))
            self.assertEqual(durations, [30, 60, 90])
            self.assertEqual(
                pixels, [(255, 0, 0), (0, 128, 0), (0, 0, 255)])

    def test_streaming_gif_size_failure_is_atomic_and_cleans_temporary(self):
        from PIL import Image
        from bedrock_rl.sdg.media import _GifTooLarge

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "bounded.gif"
            path.write_bytes(b"existing artifact")
            frames = [Image.new("RGB", (32, 24), "red")]
            with self.assertRaises(_GifTooLarge):
                save_frames_gif(frames, path, max_bytes=1)
            self.assertEqual(path.read_bytes(), b"existing artifact")
            self.assertEqual(list(root.iterdir()), [path])

    def test_streaming_gif_delta_frames_restore_changed_pixels(self):
        from PIL import Image

        frames = [Image.new("RGB", (8, 6), "black") for _ in range(3)]
        frames[1].putpixel((7, 5), (255, 0, 0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "delta.gif"
            save_frames_gif(frames, path, duration_ms=30)
            with Image.open(path) as image:
                pixels = []
                for index in range(image.n_frames):
                    image.seek(index)
                    pixels.append(image.convert("RGB").getpixel((7, 5)))
            self.assertEqual(
                pixels, [(0, 0, 0), (255, 0, 0), (0, 0, 0)])

    def test_spooled_collector_resizes_unlinks_and_caps_memory(self):
        from PIL import Image
        from bedrock_rl.sdg.media import (
            _ReplayFrameCollector,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "capture"
            spool = root / "spool"
            capture.mkdir()

            class Env:
                journal = []
                ticks = 1
                frame_w = 8
                frame_h = 6
                timeout = 1.0
                failed = False
                _frames_path = capture

                def describe(self):
                    return "spool test"

            env = Env()
            collector = _ReplayFrameCollector(
                "minecraft-official", 2, spool_dir=spool, frame_width=4)
            collector(env)
            env.ticks = 2
            source = capture / "frame_000002.ppm"
            Image.new("RGB", (8, 6), "orange").save(source)
            collector(env)
            self.assertFalse(source.exists())
            self.assertEqual(collector.frame_indices, [2])
            self.assertIsInstance(collector.frames[0], Path)
            with Image.open(collector.frames[0]) as saved:
                self.assertEqual(saved.size, (4, 3))
            # Replays intentionally mix the semantic camera and full engine
            # GUI frames. Their integer aspect rounding may differ, but a
            # panel's encoded dimensions must remain constant.
            collector._store(Image.new("RGB", (9, 6), "blue"), 4)
            with Image.open(collector.frames[1]) as saved:
                self.assertEqual(saved.size, (4, 3))
            self.assertGreater(collector.spool_bytes, 0)

    def test_three_view_gif_requires_exact_same_tick_ids(self):
        from PIL import Image
        from bedrock_rl.sdg.media import ReplayViewResult

        def result(view, ticks):
            frames = [Image.new("RGB", (8, 6), "violet") for _ in ticks]
            return ReplayViewResult(
                view, frames, {"success": True}, tuple(ticks))

        values = iter([
            result("semantic", (5, 10)),
            result("netherite-procedural", (5, 15)),
            result("minecraft-official", (5, 10)),
        ])
        with tempfile.TemporaryDirectory() as directory, mock.patch(
                "bedrock_rl.sdg.media."
                "replay_trajectory_view", side_effect=lambda *_a, **_k: next(
                    values)):
            with self.assertRaisesRegex(RuntimeError, "identical engine ticks"):
                save_replayed_views_gif(
                    Trajectory(), Path(directory) / "views.gif",
                    replay_workers=1)

    def test_three_view_replays_run_concurrently_but_keep_panel_order(self):
        import threading

        from PIL import Image
        from bedrock_rl.sdg.media import ReplayViewResult

        barrier = threading.Barrier(3)
        official_done = threading.Event()
        procedural_done = threading.Event()
        finished = []
        colors = {
            "semantic": "red",
            "netherite-procedural": "green",
            "minecraft-official": "blue",
        }
        def replay(_trajectory, view, **_kwargs):
            # A serial implementation cannot pass this barrier. Events force
            # completion order to differ from requested panel order without
            # relying on scheduler timing.
            barrier.wait(timeout=1.0)
            if view == "semantic":
                if not procedural_done.wait(timeout=1.0):
                    raise RuntimeError("procedural replay did not finish")
            elif view == "netherite-procedural":
                if not official_done.wait(timeout=1.0):
                    raise RuntimeError("official replay did not finish")
            finished.append(view)
            if view == "minecraft-official":
                official_done.set()
            elif view == "netherite-procedural":
                procedural_done.set()
            frame = Image.new("RGB", (80, 60), colors[view])
            return ReplayViewResult(
                view, [frame], {"success": True}, (5,))

        views = (
            "semantic", "netherite-procedural", "minecraft-official",
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch(
                "bedrock_rl.sdg.media."
                "replay_trajectory_view", side_effect=replay):
            path = Path(directory) / "views.gif"
            summary = save_replayed_views_gif(
                Trajectory(), path, views=views, labels=views,
                panel_width=80)
            self.assertEqual(finished, list(reversed(views)))
            self.assertEqual(summary["views"], list(views))
            with Image.open(path) as image:
                image.seek(0)
                rgb = image.convert("RGB")
                self.assertEqual(rgb.getpixel((40, 50)), (255, 0, 0))
                self.assertEqual(rgb.getpixel((120, 50)), (0, 128, 0))
                self.assertEqual(rgb.getpixel((200, 50)), (0, 0, 255))

    def test_replay_workers_can_force_serial_and_reject_invalid_values(self):
        from PIL import Image
        from bedrock_rl.sdg.media import ReplayViewResult

        active = 0
        peak = 0

        def replay(_trajectory, view, **_kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            active -= 1
            return ReplayViewResult(
                view, [Image.new("RGB", (8, 6), "violet")],
                {"success": True}, (5,))

        with tempfile.TemporaryDirectory() as directory, mock.patch(
                "bedrock_rl.sdg.media."
                "replay_trajectory_view", side_effect=replay):
            save_replayed_views_gif(
                Trajectory(), Path(directory) / "serial.gif",
                replay_workers=1)
            self.assertEqual(peak, 1)
            with self.assertRaisesRegex(ValueError, "must be positive"):
                save_replayed_views_gif(
                    Trajectory(), Path(directory) / "invalid.gif",
                    replay_workers=0)
            with self.assertRaisesRegex(TypeError, "'auto' or an integer"):
                save_replayed_views_gif(
                    Trajectory(), Path(directory) / "invalid-bool.gif",
                    replay_workers=True)

    def test_three_view_gif_must_be_denser_than_trajectory_images(self):
        from PIL import Image
        from bedrock_rl.sdg.media import ReplayViewResult

        ticks = (5, 10)

        def replay(_trajectory, view, **_kwargs):
            return ReplayViewResult(
                view, [Image.new("RGB", (8, 6), "red") for _ in ticks],
                {"success": True}, ticks)

        trajectory = Trajectory(messages=[{
            "role": "user",
            "content": [{"type": "image_url", "image_url": {
                "url": "data:image/png;base64,unused",
            }}],
        }] * 2)
        with tempfile.TemporaryDirectory() as directory, mock.patch(
                "bedrock_rl.sdg.media."
                "replay_trajectory_view", side_effect=replay):
            with self.assertRaisesRegex(RuntimeError, "higher cadence"):
                save_replayed_views_gif(
                    trajectory, Path(directory) / "views.gif")

    def test_three_view_sampling_uses_shared_ticks_and_preserves_duration(self):
        from PIL import Image
        from bedrock_rl.sdg.media import ReplayViewResult

        ticks = tuple(range(5, 1506, 5))

        def replay(_trajectory, view, **_kwargs):
            color = {
                "semantic": "red",
                "netherite-procedural": "green",
                "minecraft-official": "blue",
            }[view]
            frames = [Image.new("RGB", (8, 6), color) for _ in ticks]
            return ReplayViewResult(
                view, frames, {"success": True}, ticks)

        with tempfile.TemporaryDirectory() as directory, mock.patch(
                "bedrock_rl.sdg.media."
                "replay_trajectory_view", side_effect=replay), mock.patch(
                "bedrock_rl.sdg.media._INITIAL_GIF_FRAMES",
                23):
            path = Path(directory) / "views.gif"
            summary = save_replayed_views_gif(
                Trajectory(), path, panel_width=8, duration_ms=30,
                max_bytes=1_000_000)
            self.assertEqual(summary["frames"], 23)
            self.assertEqual(summary["source_frames"], len(ticks))
            self.assertEqual(summary["sampled_ticks"][0], ticks[0])
            self.assertEqual(summary["sampled_ticks"][-1], ticks[-1])
            with Image.open(path) as image:
                # Every source frame is identical, so the GIF writer folds
                # the 23 rows while retaining their full elapsed duration.
                self.assertEqual(image.n_frames, 1)
                self.assertEqual(
                    image.info["duration"], len(ticks) * 30)

    def test_mock_replay_preserves_exact_saved_argument_strings(self):
        from PIL import Image

        programs = [
            '{"actions": [{"action": "wait", "ticks": 1}]}',
            '{"actions":[{"action":"wait","ticks":2}]}',
        ]
        trajectory = Trajectory(messages=[
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"call_{index}", "type": "function",
                "function": {"name": "computer", "arguments": value},
            }]}
            for index, value in enumerate(programs)
        ], task={"path": "examples/farm_one_log_sdg/task.yaml"})
        trajectory.state.record(2, "after_tools", {"episode": {
            "done": True, "success": True, "failed": False,
            "turns": 2, "actions": 2, "reward": 1.0,
            "spec": {
                "task_yaml": "examples/farm_one_log_sdg/task.yaml",
                "seed": 0, "init_yaw": 3.0, "init_pitch": -1.0,
            },
        }})

        class Collector:
            ticks = []

            def __init__(self, view, recording_every):
                self.view = view
                self.recording_every = recording_every
                self.frames = [Image.new("RGB", (8, 6), "violet")]

            def __call__(self, env):
                self.ticks.append(env.frame_ticks)

        class FakeEnv:
            journal = []

            def __init__(self):
                self.frame_ticks = 0

        class FakeEpisode:
            seen = []

            def __init__(self, task, seed, yaw, pitch, **kwargs):
                self.task = task
                self.env = FakeEnv()
                self.done = self.success = self.failed = False
                self.turns = self.nlines = 0
                self._observer = kwargs["tick_observer"]
                self.identity = (seed, yaw, pitch)

            def step(self, program):
                self.seen.append(program)
                self.turns += 1
                self.nlines += 1
                if self.turns == 2:
                    self.done = self.success = True
                # Episode.step takes exactly one observation frame after the
                # action batch, even when that turn reaches terminal state.
                self.env.frame_ticks += 1
                self._observer(self.env)

            def final_reward(self):
                return 1.0

            def close(self):
                pass

        with mock.patch(
                "bedrock_rl.sdg.media._ReplayFrameCollector",
                Collector):
            result = replay_trajectory_view(
                trajectory, "semantic", episode_class=FakeEpisode)
        self.assertEqual(FakeEpisode.seen, [
            '[{"action": "wait", "ticks": 1}]',
            '[{"action": "wait", "ticks": 2}]',
        ])
        self.assertEqual(Collector.ticks, [1, 2])
        self.assertEqual(result.terminal["success"], True)
        self.assertEqual(trajectory.messages[0]["tool_calls"][0]
                         ["function"]["arguments"], programs[0])

    def test_replay_prefers_single_provenance_episode_spec(self):
        from bedrock_rl.sdg.media import _recorded_episode

        spec = {
            "task_yaml": "examples/farm_one_log_sdg/task.yaml",
            "seed": 17, "init_yaw": 3.0, "init_pitch": -1.0,
        }
        trajectory = Trajectory(provenance={"episode_spec": spec})
        trajectory.state.record(2, "after_tools", {"episode": {
            "done": True, "success": True, "failed": False,
            "turns": 2, "actions": 2, "reward": 1.0,
        }})

        recorded, replay_spec = _recorded_episode(trajectory)

        self.assertTrue(recorded["success"])
        self.assertIs(replay_spec, spec)
        recorded["spec"] = {**spec, "seed": 99}
        _recorded, replay_spec = _recorded_episode(trajectory)
        self.assertIs(replay_spec, spec)

    def test_sink_replay_config_is_forwarded_without_mutating_artifact(self):
        trajectory = Trajectory(metadata={"marker": "canonical"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "trajectory.json"
            replay_gif = root / "views.gif"
            replay_result = {"gif": str(replay_gif), "frames": 4}
            with mock.patch(
                    "bedrock_rl.sdg.media."
                    "save_replayed_views_gif",
                    return_value=replay_result) as save:
                result = TrajectoryArtifactWriter(output, compact=True, replay={
                    "gif": replay_gif,
                    "recording_every": 7,
                    "panel_width": 160,
                    "replay_workers": 2,
                }).write([trajectory])
            save.assert_called_once()
            _, path = save.call_args.args
            self.assertEqual(path, replay_gif)
            self.assertEqual(save.call_args.kwargs["recording_every"], 7)
            self.assertEqual(save.call_args.kwargs["panel_width"], 160)
            self.assertEqual(save.call_args.kwargs["replay_workers"], 2)
            self.assertEqual(result["replay"], replay_result)
            self.assertEqual(trajectory.metadata, {"marker": "canonical"})

    def test_replay_rejects_any_event_payload_or_tick_drift(self):
        from bedrock_rl.sdg.media import _verify_replay
        from bedrock_rl.core.state import StateEvent

        trajectory = Trajectory()
        trajectory.state.extend_events([
            StateEvent("block_broken", {"block": 17}, tick=12),
        ])
        recorded = {
            "done": True, "success": True, "failed": False,
            "turns": 1, "actions": 1, "reward": 1.0,
        }

        class Event:
            name = "block_broken"

            def as_dict(self):
                return {"event": self.name, "block": 17, "tick": 13}

        episode = mock.Mock(
            done=True, success=True, failed=False, turns=1, nlines=1,
            env=mock.Mock(journal=[Event()]),
        )
        episode.final_reward.return_value = 1.0
        with self.assertRaisesRegex(RuntimeError, "event stream"):
            _verify_replay(trajectory, recorded, episode,
                           "minecraft-official")


if __name__ == "__main__":
    unittest.main()
