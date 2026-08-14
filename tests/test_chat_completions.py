import os
import unittest
from unittest.mock import patch

from bedrock_rl.adapters.chat_completions import ChatCompletionsModel


class ChatCompletionsAuthenticationTests(unittest.TestCase):
    def test_global_openai_key_is_not_sent_to_an_arbitrary_endpoint(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "secret"}):
            model = ChatCompletionsModel("model", "https://example.com/v1")
        self.assertIsNone(model.api_key)

    def test_official_openai_endpoint_uses_the_standard_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "secret"}):
            model = ChatCompletionsModel(
                "model", "https://api.openai.com/v1")
        self.assertEqual(model.api_key, "secret")

    def test_custom_endpoint_key_requires_explicit_configuration(self):
        with patch.dict(os.environ, {"MY_SERVER_KEY": "custom"}):
            model = ChatCompletionsModel(
                "model", "https://example.com/v1",
                api_key_env="MY_SERVER_KEY")
        self.assertEqual(model.api_key, "custom")


if __name__ == "__main__":
    unittest.main()
