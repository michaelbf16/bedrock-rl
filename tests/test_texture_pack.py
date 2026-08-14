from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.compose_texture_pack import compose, pack_digest


class TexturePackTests(unittest.TestCase):
    def test_pack_overrides_base_and_digest_ignores_zip_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "client.jar"
            pack = root / "pack.zip"
            output = root / "composite.jar"
            with zipfile.ZipFile(base, "w") as archive:
                archive.writestr("assets/minecraft/textures/blocks/stone.png",
                                 b"vanilla")
                archive.writestr("version.json", b"base")
            with zipfile.ZipFile(pack, "w") as archive:
                archive.writestr("pack.mcmeta", b"{}")
                archive.writestr("assets/minecraft/textures/blocks/stone.png",
                                 b"custom")
            compose(base, pack, output)
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(archive.read(
                    "assets/minecraft/textures/blocks/stone.png"), b"custom")
                self.assertEqual(archive.read("version.json"), b"base")
            first = pack_digest(pack)
            with zipfile.ZipFile(pack, "w") as archive:
                archive.writestr("assets/minecraft/textures/blocks/stone.png",
                                 b"custom")
                archive.writestr("pack.mcmeta", b"{}")
            self.assertEqual(pack_digest(pack), first)

    def test_pack_refuses_traversal_and_non_minecraft_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            pack = Path(directory) / "bad.zip"
            with zipfile.ZipFile(pack, "w") as archive:
                archive.writestr("../escape", b"bad")
            with self.assertRaisesRegex(ValueError, "unsafe"):
                pack_digest(pack)


if __name__ == "__main__":
    unittest.main()
