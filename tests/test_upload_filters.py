"""Exercise the same Docker-style matcher and walker Modal uploads use."""

import importlib.util
import tempfile
import unittest
from pathlib import Path, PurePosixPath

from bedrock_rl.launch.uploads import DATA_UPLOAD_IGNORE


ROOT = Path(__file__).resolve().parents[1]
HAS_MODAL = importlib.util.find_spec("modal") is not None


@unittest.skipUnless(HAS_MODAL, "Modal is an optional launch dependency")
class UploadFilterTests(unittest.TestCase):
    def matchers(self):
        from modal import FilePatternMatcher
        return {
            "source": FilePatternMatcher.from_file(ROOT / ".dockerignore"),
            "explicit_data": FilePatternMatcher(*DATA_UPLOAD_IGNORE),
        }

    def test_credentials_and_caches_are_excluded_at_every_depth(self):
        private = (
            ".env", ".env.local", ".env.production", ".modal.toml",
            ".modal/token", ".huggingface/token", ".cache/huggingface/token",
            ".aws/credentials", ".azure/accessTokens.json", ".ssh/id_ed25519",
            ".config/gcloud/application_default_credentials.json",
            ".netrc", ".pypirc", ".npmrc", ".git-credentials",
            "credentials.json", "token.json", "service.pem", "service.key",
            ".git/config", ".venv/lib/module.py", "venv/bin/python",
            "__pycache__/module.pyc", ".pytest_cache/cache.json",
        )
        for name, matcher in self.matchers().items():
            for prefix in ("", "custom/", "nested/custom/"):
                for filename in private:
                    path = Path(prefix + filename)
                    with self.subTest(upload=name, path=path):
                        self.assertTrue(matcher(path))

    def test_source_components_and_example_environment_files_are_retained(self):
        public = (
            ".env.example", "custom/.env.example",
            "bedrock_rl/launch/training_app.py", "custom/generator.py",
            "custom/train.yaml", "custom/manifest.json",
            "examples/example/data.yaml", "requirements/verl-cu130.lock",
        )
        for name, matcher in self.matchers().items():
            for filename in public:
                with self.subTest(upload=name, path=filename):
                    self.assertFalse(matcher(Path(filename)))

    def test_data_mount_walk_retains_payloads_and_filters_private_files(self):
        from modal.file_pattern_matcher import _ignore_fn
        from modal.mount import _MountDir

        public = (
            "train.parquet", "nested/validation.jsonl", "frames/0001.png",
            "arrays/features.npy", "checkpoints/step-1/config.json",
            "checkpoints/step-1/model.safetensors", "checkpoints/optimizer.pt",
            "outputs/model.pth", "data/custom/manifest.json", ".env.example",
            "custom/.env.example",
        )
        private = (
            ".modal.toml", "nested/.env", "nested/.cache/huggingface/token",
            "checkpoints/.git/config", "nested/service.key", "token.json",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in (*public, *private):
                path = root / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("synthetic fixture")
            # Image.add_local_dir uses this exact sequence conversion and
            # mount walker before either mounting or copying an image layer.
            mount = _MountDir(root, PurePosixPath("/data"),
                              _ignore_fn(DATA_UPLOAD_IGNORE), recursive=True)
            uploaded = {
                str(remote.relative_to("/data"))
                for _local, remote in mount.get_files_to_upload()
            }
        self.assertEqual(uploaded, set(public))

    def test_data_patterns_remain_part_of_source_exclusions(self):
        patterns = {
            line.strip() for line in (ROOT / ".dockerignore").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertTrue(set(DATA_UPLOAD_IGNORE).issubset(patterns))


if __name__ == "__main__":
    unittest.main()
