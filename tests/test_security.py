"""Regression checks for untrusted named chat-template metadata."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from bedrock_rl.security import (guard_transformers_saves,
                                 validate_chat_templates)


class TemplatePathTests(unittest.TestCase):
    def test_processor_also_validates_nested_tokenizer(self):
        owner = SimpleNamespace(
            chat_template="safe", tokenizer=SimpleNamespace(
                chat_template={"../../escape": "untrusted"}))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Invalid chat template"):
                validate_chat_templates(owner, directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_legacy_names_and_cross_platform_separators(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("../../escape", "..\\escape", "/tmp/escape"):
                with self.subTest(name=name):
                    owner = SimpleNamespace(chat_template=[
                        {"name": name, "template": "untrusted"}])
                    with self.assertRaises(ValueError):
                        validate_chat_templates(owner, directory)

    def test_named_template_cannot_follow_an_existing_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "checkpoint"
            (destination / "additional_chat_templates").mkdir(parents=True)
            outside = root / "outside.jinja"
            (destination / "additional_chat_templates/tool.jinja").symlink_to(outside)
            with self.assertRaises(ValueError):
                validate_chat_templates(
                    SimpleNamespace(chat_template={"tool": "untrusted"}),
                    destination)
            self.assertFalse(outside.exists())

    def test_real_tokenizer_rejects_escape_and_saves_valid_templates(self):
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import WordLevel
            from transformers import PreTrainedTokenizerFast
        except ImportError:
            self.skipTest("training dependencies are not installed")
        guard_transformers_saves()
        guard_transformers_saves()  # repeated model loads are idempotent
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, "[UNK]")),
            unk_token="[UNK]")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer.chat_template = {"../../escape": "untrusted"}
            with self.assertRaisesRegex(ValueError, "Invalid chat template"):
                tokenizer.save_pretrained(root / "bad")
            self.assertFalse((root / "escape.jinja").exists())
            self.assertFalse((root / "bad").exists())
            tokenizer.chat_template = {"default": "default", "tool": "tool"}
            tokenizer.save_pretrained(root / "good")
            self.assertEqual(
                (root / "good/additional_chat_templates/tool.jinja").read_text(), "tool")

    def test_real_processor_rejects_escape_before_writing(self):
        try:
            from transformers import ProcessorMixin
        except ImportError:
            self.skipTest("training dependencies are not installed")
        guard_transformers_saves()

        class MinimalProcessor(ProcessorMixin):
            attributes = []

            def __init__(self):
                self.chat_template = {"../../escape": "untrusted"}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "checkpoint"
            with self.assertRaisesRegex(ValueError, "Invalid chat template"):
                MinimalProcessor().save_pretrained(output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
