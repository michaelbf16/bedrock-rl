import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from bedrock_rl.env.engine import CAM_H, CAM_W
from bedrock_rl.env.task import Task
from bedrock_rl.env.views import (EnginePixelsView, SemanticView, ViewSpec,
                                     capture_png, open_view,
                                     read_asset_manifest)


class CustomView:
    def __init__(self, label="custom"):
        self.label = label

    def capture(self, episode):
        del episode
        return None


class FakeEnv:
    def __init__(self):
        count = CAM_W * CAM_H
        self.obs = {"cam": [2] * count, "depth": [0] * count,
                    "edge": [0] * count}
        self.acts = 0
        self.frames = 0

    def act(self):
        self.acts += 1
        return self.obs

    def frame(self):
        self.frames += 1
        return b"exact-png"


class FakeEpisode:
    def __init__(self, gui_open=False):
        self.env = FakeEnv()
        self.gui_open = gui_open
        self.frame_width = 128
        self.frame_height = 72


class ViewTests(unittest.TestCase):
    def test_builtin_names_aliases_and_task_default(self):
        self.assertEqual(ViewSpec.parse("procedural").type,
                         "netherite-procedural")
        self.assertEqual(ViewSpec.parse("jar").type,
                         "minecraft-official")
        task = Task({"name": "v", "goal": "see",
                     "success": {"type": "pitch_below", "value": 0}})
        self.assertEqual(task.view.type, "netherite-procedural")

    def test_custom_import_path_is_the_extension_point(self):
        reference = f"{CustomView.__module__}:CustomView"
        spec = ViewSpec.parse({"type": reference,
                               "label": "mine"})
        view, home, provenance = open_view(spec, ".")
        self.assertIsInstance(view, CustomView)
        self.assertEqual(view.label, "mine")
        self.assertEqual(home, Path.cwd().resolve())
        self.assertEqual(provenance["type"], reference)

    def test_semantic_view_spends_one_tick_and_returns_exact_size(self):
        episode = FakeEpisode()
        data = capture_png(SemanticView(), episode)
        self.assertTrue(data.startswith(b"\x89PNG"))
        self.assertEqual(episode.env.acts, 1)
        self.assertEqual(episode.env.frames, 0)
        with Image.open(io.BytesIO(data)) as image:
            self.assertEqual(image.size, (128, 72))
            self.assertEqual(image.getpixel((0, 0)), (95, 159, 53))

    def test_semantic_gui_falls_back_to_clickable_engine_frame(self):
        episode = FakeEpisode(gui_open=True)
        self.assertEqual(capture_png(SemanticView(), episode), b"exact-png")
        self.assertEqual(episode.env.frames, 1)
        self.assertEqual(episode.env.acts, 0)

    def test_asset_identity_is_read_and_mismatch_is_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".bedrock-rl-assets").write_text(
                "kind=procedural\nsource=test\nsha256=abc\n")
            self.assertEqual(read_asset_manifest(root)["kind"], "procedural")
            EnginePixelsView(name="netherite-procedural",
                             asset_kind="procedural").validate(root)
            with self.assertRaisesRegex(RuntimeError, "requires a jar"):
                EnginePixelsView(name="minecraft-official",
                                 asset_kind="jar").validate(root)

    def test_unknown_names_and_ambiguous_config_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown task view"):
            ViewSpec.parse("pretty")
        with self.assertRaisesRegex(ValueError, "both beside"):
            ViewSpec.parse({"type": "semantic", "width": 64,
                            "config": {"height": 36}})

    def test_official_view_binds_texture_pack_and_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".bedrock-rl-assets").write_text(
                "kind=jar\nsource=minecraft-client.jar+resource-pack\n"
                "texture_pack=faithful.zip\ntexture_pack_sha256=abc123\n")
            view = ViewSpec.parse({
                "type": "minecraft-official",
                "texture_pack": "packs/faithful.zip",
                "texture_pack_sha256": "ABC123",
            }).create()
            view.validate(root)
            with self.assertRaisesRegex(RuntimeError, "requests vanilla"):
                EnginePixelsView(name="minecraft-official",
                                 asset_kind="jar").validate(root)
            with self.assertRaisesRegex(RuntimeError, "requires texture pack"):
                EnginePixelsView(name="minecraft-official", asset_kind="jar",
                                 texture_pack="other.zip").validate(root)


if __name__ == "__main__":
    unittest.main()
