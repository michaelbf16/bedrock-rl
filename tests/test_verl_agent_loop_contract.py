import asyncio
import importlib.util
import inspect
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from PIL import Image

from bedrock_rl.core.messages import validate_transcript
from bedrock_rl.core.tools import FunctionTool, ToolRegistry, ToolResult


class _Record:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _fake_verl_modules():
    modules = {}
    for name in (
            "verl", "verl.experimental", "verl.experimental.agent_loop",
            "verl.experimental.agent_loop.agent_loop",
            "verl.experimental.agent_loop.tool_parser", "verl.tools",
            "verl.tools.schemas"):
        module = ModuleType(name)
        module.__path__ = []
        modules[name] = module

    class AgentLoopBase:
        pass

    class ToolParser:
        @classmethod
        def get_tool_parser(cls, *args, **kwargs):
            del args, kwargs
            return None

    modules["verl.experimental.agent_loop.agent_loop"].AgentLoopBase = (
        AgentLoopBase)
    modules["verl.experimental.agent_loop.agent_loop"].AgentLoopMetrics = (
        _Record)
    modules["verl.experimental.agent_loop.agent_loop"].AgentLoopOutput = (
        _Record)
    modules["verl.experimental.agent_loop.tool_parser"].ToolParser = ToolParser
    modules["verl.tools.schemas"].OpenAIFunctionToolSchema = _Record
    return modules


def _load_agent_loop_module():
    path = (Path(__file__).parents[1] / "bedrock_rl" / "adapters" /
            "verl" / "agent_loop.py")
    spec = importlib.util.spec_from_file_location(
        "_netherite_test_verl_agent_loop", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, _fake_verl_modules()):
        spec.loader.exec_module(module)
    return module


class VerlAgentLoopContractTests(unittest.TestCase):
    def test_visual_rollouts_get_a_private_frame_root_by_default(self):
        module = _load_agent_loop_module()
        with patch.dict("os.environ", {}, clear=True):
            root = module._frames_root(True)

        self.assertTrue(Path(root).is_dir())
        self.assertIsNone(module._frames_root(False))

    def test_each_turn_is_capped_to_remaining_response_tokens(self):
        bound = _load_agent_loop_module().bounded_sampling_params

        self.assertEqual(
            bound({"max_tokens": 128, "temperature": 1.0}, 90, 100),
            {"max_tokens": 10, "temperature": 1.0})
        self.assertEqual(bound({"temperature": 0.0}, 7, 10)["max_tokens"],
                         3)
        self.assertIsNone(bound({"max_tokens": 8}, 10, 10))

    def test_multimodal_control_tokens_are_banned_without_losing_user_bans(self):
        module = _load_agent_loop_module()
        tokenizer = SimpleNamespace(
            additional_special_tokens=[
                "<|vision_start|>", "<|vision_end|>",
                "<|image_pad|>", "<|video_pad|>",
            ],
            convert_ids_to_tokens=lambda token_id: {
                101: "<|image_pad|>", 102: "<|video_pad|>",
            }[token_id],
        )
        processor = SimpleNamespace(image_token_id=101, video_token_id=102)

        forbidden = module._multimodal_control_tokens(processor, tokenizer)
        params = module.forbid_multimodal_output(
            {"temperature": 1.0, "bad_words": ["user-ban"]}, forbidden)

        self.assertEqual(params["bad_words"], [
            "user-ban", "<|image_pad|>", "<|video_pad|>",
            "<|vision_start|>", "<|vision_end|>",
        ])

    def test_text_only_rollouts_do_not_add_a_bad_words_parameter(self):
        module = _load_agent_loop_module()
        params = {"temperature": 1.0}

        self.assertEqual(module._multimodal_control_tokens(
            None, SimpleNamespace()), ())
        self.assertIs(module.forbid_multimodal_output(params, ()), params)

    @unittest.skipUnless(importlib.util.find_spec("verl"),
                         "the real verl surface is tested in its pinned CI job")
    def test_real_pinned_verl_output_schema_accepts_the_adapter_contract(self):
        from verl.experimental.agent_loop.agent_loop import (
            AgentLoopBase, AgentLoopMetrics, AgentLoopOutput,
        )
        from bedrock_rl.adapters.verl.agent_loop import NetheriteAgentLoop

        self.assertTrue(issubclass(NetheriteAgentLoop, AgentLoopBase))
        metrics = AgentLoopMetrics(
            generate_sequences=0.0, tool_calls=0.0,
            compute_score=0.0, num_preempted=0)
        output = AgentLoopOutput(
            prompt_ids=[], response_ids=[], response_mask=[],
            response_logprobs=None, multi_modal_data={},
            mm_processor_kwargs={}, reward_score=0.0, num_turns=1,
            metrics=metrics, extra_fields={})
        self.assertEqual(output.response_ids, [])

    def test_custom_tools_extend_the_default_computer_tool(self):
        module = _load_agent_loop_module()
        module.NetheriteAgentLoop.rollout_config = SimpleNamespace(
            multi_turn=SimpleNamespace(format="hermes"), response_length=32)
        module.NetheriteAgentLoop.tokenizer = SimpleNamespace()

        extra = FunctionTool(
            "recipe_book", "look up a recipe", {"type": "object"},
            lambda arguments, context: ToolResult("recipe"))
        instance = module.NetheriteAgentLoop(core_tools=[extra])

        self.assertIn("computer", instance.tools)
        self.assertIn("recipe_book", instance.tools)

    def test_custom_computer_must_keep_unknown_call_accounting(self):
        module = _load_agent_loop_module()
        module.NetheriteAgentLoop.rollout_config = SimpleNamespace(
            multi_turn=SimpleNamespace(format="hermes"), response_length=32)
        module.NetheriteAgentLoop.tokenizer = SimpleNamespace()
        replacement = FunctionTool(
            "computer", "act", {"type": "object"},
            lambda arguments, context: ToolResult("ok"))

        with self.assertRaisesRegex(ValueError, "malformed_fallback=True"):
            module.NetheriteAgentLoop(core_tools=[replacement])

    def test_run_signature_and_cpu_behavior(self):
        module = _load_agent_loop_module()
        signature = inspect.signature(module.NetheriteAgentLoop.run)
        self.assertEqual(list(signature.parameters),
                         ["self", "sampling_params", "kwargs"])
        self.assertEqual(signature.parameters["kwargs"].kind,
                         inspect.Parameter.VAR_KEYWORD)
        self.assertTrue(inspect.iscoroutinefunction(
            module.NetheriteAgentLoop.run))

        async def exercise():
            instance = object.__new__(module.NetheriteAgentLoop)
            session = SimpleNamespace()
            session.task = SimpleNamespace(
                name="contract", path="task.yaml", goal="finish",
                max_turns=2)
            session.done = False
            session.closed = False
            session.episode = SimpleNamespace(
                view_provenance={"type": "test"}, success=True, turns=1,
                final_reward=lambda: 1.0)

            def malformed_step(payload, *, count_turn=True):
                self.assertEqual(payload, "{")
                session.episode.turns += int(count_turn)
                return "malformed observation", image1, -0.2, False

            session.episode.step = malformed_step
            session.drain_events = lambda: []

            async def close():
                session.closed = True

            session.close = close

            class Environment:
                async def open(self, task):
                    del task
                    return session

            module.NetheriteEnvironment = lambda *args, **kwargs: Environment()

            executed = []

            def computer(arguments, context):
                del arguments
                executed.append(context.turn)
                context.session.done = True
                return ToolResult(
                    "done", done=True, attachments={"image": image1})

            tool = FunctionTool("computer", "act", {"type": "object"},
                                computer)

            async def malformed(error, context):
                return await module.NetheriteComputerTool().malformed(
                    error, context)

            tool.malformed = malformed
            tool.malformed_fallback = True
            instance.tools = ToolRegistry([tool])
            instance.tool_schemas = []
            instance.keep_last_images = 1
            instance.save_trajectories = None
            instance.response_length = 32
            instance.verifier = None
            instance.loop = asyncio.get_running_loop()
            instance.tokenizer = SimpleNamespace(
                decode=lambda ids: "sampled-call")

            class Probes:
                async def capture(self, active_session):
                    del active_session
                    return {}

            instance.probes = Probes()

            class Parser:
                stop_token_ids = []

                async def extract_tool_calls(self, token_ids, schemas):
                    del token_ids, schemas
                    calls = [
                        SimpleNamespace(name="invented", arguments={}),
                        SimpleNamespace(name="computer", arguments=[]),
                        SimpleNamespace(name="computer", arguments={}),
                        SimpleNamespace(name="computer", arguments={}),
                    ]
                    return None, calls

            instance.tool_parser = Parser()

            image0 = Image.new("RGB", (1, 1), (255, 0, 0))
            image1 = Image.new("RGB", (1, 1), (0, 255, 0))

            async def process_multi_modal_info(messages):
                del messages
                return {"images": [image0]}

            instance.process_multi_modal_info = process_multi_modal_info

            async def apply_chat_template(messages, *, images=None, **kwargs):
                del messages, kwargs
                return [100 + len(images or ())]

            instance.apply_chat_template = apply_chat_template
            instance._get_mm_processor_kwargs = lambda: {"test": True}

            class Server:
                async def generate(self, **kwargs):
                    del kwargs
                    return SimpleNamespace(
                        token_ids=[7], num_preempted=0,
                        routed_experts=None, extra_fields={},
                        log_probs=[-0.25])

            instance.server_manager = Server()
            saved = []

            async def save(trajectory):
                saved.append(trajectory)
                return None

            instance._save = save
            raw_prompt = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "finish"},
                ],
            }]
            output = await instance.run(
                {"temperature": 0.0}, raw_prompt=raw_prompt,
                tools_kwargs={"computer": {"create_kwargs": {
                    "task_yaml": "task.yaml"}}})

            self.assertTrue(session.closed)
            self.assertEqual(executed, [0])
            self.assertEqual(output.prompt_ids, [101])
            self.assertEqual(output.response_ids, [7, 100])
            self.assertEqual(output.response_mask, [1, 0])
            self.assertEqual(output.multi_modal_data["images"], [image0])
            self.assertEqual(output.extra_fields[
                "context_masked_assistant_turns"], 0)
            self.assertEqual(len(saved), 1)
            validate_transcript(saved[0].messages)
            results = saved[0].metadata["tool_results"]
            self.assertEqual([result["call_id"] for result in results],
                             ["call_1_0", "call_1_1", "call_1_2",
                              "call_1_3"])
            self.assertIn("unknown tool", results[0]["metadata"]["malformed"])
            self.assertIn("JSON object", results[1]["metadata"]["malformed"])
            # The invented tool name is charged as one malformed computer
            # attempt instead of escaping the episode reward path.
            self.assertEqual(session.episode.turns, 2)
            self.assertTrue(results[3]["metadata"]["skipped"])

        with patch.dict("os.environ", {"BRL_TRAJECTORY_DIR": ""}):
            asyncio.run(exercise())

    def test_no_call_floor_survives_an_opt_in_verifier(self):
        module = _load_agent_loop_module()

        async def exercise():
            session = SimpleNamespace(episode=SimpleNamespace(
                success=False, turns=0, final_reward=lambda: 0.0))
            recorder = SimpleNamespace(
                trajectory=SimpleNamespace(), state=SimpleNamespace())

            reward = await module._rollout_reward(
                lambda trajectory, state: 0.75,
                recorder, session, tool_calls=0, computer_calls=0)
            self.assertEqual(reward.total, -1.0)
            self.assertEqual(reward.metrics["tool_calls"], 0)

        asyncio.run(exercise())

    def test_custom_tool_call_outranks_the_no_call_floor(self):
        module = _load_agent_loop_module()

        async def exercise():
            session = SimpleNamespace(episode=SimpleNamespace(
                success=False, turns=0, final_reward=lambda: 0.0))
            recorder = SimpleNamespace(
                trajectory=SimpleNamespace(), state=SimpleNamespace())

            reward = await module._rollout_reward(
                None, recorder, session, tool_calls=1, computer_calls=0)
            self.assertEqual(reward.total, 0.0)
            self.assertGreater(reward.total, -1.0)

        asyncio.run(exercise())

    def test_unhandled_custom_malformed_call_pays_format_penalty(self):
        module = _load_agent_loop_module()

        async def exercise():
            session = SimpleNamespace(episode=SimpleNamespace(
                success=False, turns=0, final_reward=lambda: 0.0))
            recorder = SimpleNamespace(
                trajectory=SimpleNamespace(), state=SimpleNamespace())

            reward = await module._rollout_reward(
                None, recorder, session, tool_calls=1, computer_calls=0,
                format_penalty=0.2)
            self.assertEqual(reward.total, -0.2)
            self.assertEqual(reward.components["bedrock_format"], -0.2)

        asyncio.run(exercise())

    def test_custom_tool_before_computer_does_not_consume_computer_turn(self):
        module = _load_agent_loop_module()
        computer = module.NetheriteComputerTool()
        custom = FunctionTool(
            "lookup", "look up", {"type": "object"},
            lambda arguments, context: ToolResult("ok"))
        registry = ToolRegistry([computer, custom])

        self.assertFalse(module._counts_computer_turn(
            registry, SimpleNamespace(name="lookup")))
        self.assertTrue(module._counts_computer_turn(
            registry, SimpleNamespace(name="computer")))
        self.assertTrue(module._counts_computer_turn(
            registry, SimpleNamespace(name="invented")))


if __name__ == "__main__":
    unittest.main()
