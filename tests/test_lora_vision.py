import unittest
from types import SimpleNamespace

from bedrock_rl import lora


class _Parameter:
    def __init__(self, trainable):
        self.requires_grad = trainable


class _Model:
    def __init__(self, parameters):
        self.parameters = parameters

    def named_parameters(self):
        return iter(self.parameters)


class LoraVisionTests(unittest.TestCase):
    def setUp(self):
        self.spec = SimpleNamespace(vision_patterns=(".visual.", "visual."))

    def test_text_only_adapter_is_accepted(self):
        model = _Model([
            ("base_model.model.language_model.q_proj.lora_A", _Parameter(True)),
            ("base_model.model.visual.qkv.weight", _Parameter(False)),
        ])
        lora.assert_no_trainable_vision(model, self.spec)

    def test_trainable_vision_adapter_is_refused(self):
        model = _Model([
            ("base_model.model.visual.qkv.lora_A", _Parameter(True)),
        ])
        with self.assertRaisesRegex(ValueError, "vLLM cannot serve"):
            lora.assert_no_trainable_vision(model, self.spec)


if __name__ == "__main__":
    unittest.main()
