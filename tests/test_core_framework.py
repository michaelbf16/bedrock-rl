import json
import random
import tempfile
import unittest
from pathlib import Path

from bedrock_rl.core.components import ComponentSpec, load_object
from bedrock_rl.core.context import (ContextRequest, KeepLastImages,
                                       KeepLastTurns, SummarizeOldTurns,
                                       apply_context)
from bedrock_rl.core.guidance import (GuidancePolicy, StaticGuidance,
                                        group_guidance)
from bedrock_rl.core.harness import ToolHarness
from bedrock_rl.core.messages import image_url_part, text_part
from bedrock_rl.core.messages import validate_transcript
from bedrock_rl.core.reward import (EventSequenceReward, StateValueReward,
                                      evaluate)
from bedrock_rl.core.run import RunSpec
from bedrock_rl.core.state import FunctionProbe
from bedrock_rl.core.task import TaskSpec
from bedrock_rl.core.tools import (FunctionTool, ToolCall, ToolContext,
                                   ToolRegistry, ToolResult)
from bedrock_rl.core.trajectory import Trajectory
from bedrock_rl.adapters.verl.segments import (TokenSegment,
                                                 finalize_segments,
                                                 render_segments)
from bedrock_rl.adapters.netherite.chat import tool_schema


def import_path_summary(messages, request):
    return f"summarized {len(messages)} messages at turn {request.turn}"


class ContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_last_images_is_a_view_not_a_history_mutation(self):
        messages = [{
            "role": "user",
            "content": [text_part("state"),
                        image_url_part("https://example.test/one"),
                        image_url_part("https://example.test/two"),
                        image_url_part("https://example.test/three")],
        }]
        original = json.loads(json.dumps(messages))
        view = await apply_context(KeepLastImages(2), messages,
                                   ContextRequest(0))
        self.assertEqual(messages, original)
        self.assertEqual([part["image_url"]["url"]
                          for part in view.messages[0]["content"]
                          if part["type"] == "image_url"],
                         ["https://example.test/two",
                          "https://example.test/three"])
        self.assertEqual(view.transforms[0].details["removed"], [[0, 1]])

    async def test_summarizer_is_injected_and_audited(self):
        calls = []

        async def summarize(messages, request):
            calls.append((messages, request.turn))
            return "earlier actions"

        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "tool_call_id": "a", "content": "obs"},
            {"role": "assistant", "content": "b"},
            {"role": "tool", "tool_call_id": "b", "content": "latest"},
        ]
        view = await apply_context(SummarizeOldTurns(summarize, keep_last=1),
                                   messages, ContextRequest(2))
        self.assertEqual(calls[0][1], 2)
        self.assertEqual(view.messages[0]["content"], "rules")
        self.assertEqual(view.messages[1]["content"], "start")
        self.assertIn("earlier actions", view.messages[2]["content"])
        self.assertEqual(view.messages[-1]["content"], "latest")

    async def test_summarizer_resolves_from_run_config_import_path(self):
        messages = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "new"},
        ]
        policy = SummarizeOldTurns(
            f"{__name__}:import_path_summary", keep_last=0)
        view = await apply_context(policy, messages, ContextRequest(3))
        self.assertEqual(view.messages[0]["content"], "old")
        self.assertIn("summarized", view.messages[1]["content"])

    async def test_last_turns_never_orphans_a_tool_result(self):
        messages = [
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "one", "type": "function",
                "function": {"name": "act", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "one", "content": "one"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "two", "type": "function",
                "function": {"name": "act", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "two", "content": "two"},
        ]
        view = await apply_context(KeepLastTurns(1), messages,
                                   ContextRequest(2))
        self.assertEqual([message["role"] for message in view.messages],
                         ["user", "assistant", "tool"])
        self.assertEqual(view.messages[0]["content"], "start")
        self.assertEqual(view.messages[1]["tool_calls"][0]["id"], "two")

    async def test_context_policies_never_drop_opening_instruction(self):
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "the task"},
            {"role": "assistant", "content": "old"},
            {"role": "user", "content": "observation"},
            {"role": "assistant", "content": "latest"},
        ]
        kept = await apply_context(KeepLastTurns(1), messages,
                                   ContextRequest(2))
        self.assertEqual(kept.messages[:2], messages[:2])

        async def summarize(old, request):
            del old, request
            return "history"

        summarized = await apply_context(
            SummarizeOldTurns(summarize, keep_last=1), messages,
            ContextRequest(2))
        self.assertEqual(summarized.messages[:2], messages[:2])


class FakeSession:
    def __init__(self):
        self.value = 0
        self.done = False
        self.closed = False
        self.events = []

    def initial_messages(self, task):
        return [{"role": "user", "content": [
            text_part(task.instruction),
            image_url_part("https://example.test/initial")]}]

    def drain_events(self):
        events, self.events = self.events, []
        return events

    def close(self):
        self.closed = True


class FakeEnvironment:
    def __init__(self):
        self.session = FakeSession()

    def open(self, task):
        return self.session


class FakeModel:
    async def complete(self, messages, tools, sampling):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "increment",
                             "arguments": json.dumps({"amount": 1})},
            }],
        }


class HarnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_trajectory_views_state_tools_and_reward(self):
        environment = FakeEnvironment()

        def increment(arguments, context):
            context.session.value += arguments["amount"]
            context.session.done = True
            context.session.events.append(
                {"name": "incremented", "payload": {"by": 1}})
            return ToolResult([
                text_part("now one"),
                image_url_part("https://example.test/new")],
                              done=True)

        tool = FunctionTool(
            "increment", "increase a counter",
            {"type": "object", "properties": {
                "amount": {"type": "integer"}}, "required": ["amount"]},
            increment)
        task = TaskSpec(
            "counter", "make the counter one",
            ComponentSpec("tests.test_core_framework:FakeEnvironment"))
        guidance = GuidancePolicy(
            StaticGuidance({"teacher": "privileged static"},
                           dynamic=lambda request, level:
                           f"turn {request.turn} state {request.state['counter']}"),
            static_level="teacher", dynamic_level="direction")
        harness = ToolHarness(
            FakeModel(), environment=environment, tools=[tool],
            context=KeepLastImages(1), guidance=guidance,
            probes=[FunctionProbe("counter", lambda session: session.value)],
            verifier=StateValueReward("solved", "counter", equals=1))
        trajectory = await harness.run(task)

        self.assertTrue(environment.session.closed)
        self.assertEqual([m["role"] for m in trajectory.messages],
                         ["user", "assistant", "tool", "user"])
        self.assertEqual(trajectory.messages[2]["content"],
                         [text_part("now one")])
        self.assertEqual(trajectory.messages[3]["content"],
                         [image_url_part("https://example.test/new")])
        self.assertNotIn("privileged", json.dumps(trajectory.messages))
        self.assertIn("privileged", json.dumps(trajectory.views[0].messages))
        self.assertEqual(trajectory.reward.total, 1.0)
        self.assertEqual(trajectory.state.latest["counter"], 1)
        self.assertEqual([event.name for event in trajectory.state.events],
                         ["incremented"])
        self.assertEqual(
            trajectory.metadata["tool_results"][0]["call_id"], "call_1")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollout.json"
            trajectory.save(path)
            loaded = Trajectory.load(path)
        self.assertEqual(loaded.to_dict(), trajectory.to_dict())

    async def test_terminal_tool_call_skips_remaining_calls_but_resolves_them(self):
        environment = FakeEnvironment()
        executed = []

        class MultiCallModel:
            async def complete(self, messages, tools, sampling):
                del messages, tools, sampling
                return {
                    "role": "assistant", "content": None,
                    "tool_calls": [
                        {"id": "first", "type": "function", "function": {
                            "name": "finish", "arguments": "{}"}},
                        {"id": "second", "type": "function", "function": {
                            "name": "finish", "arguments": "{}"}},
                    ],
                }

        def finish(arguments, context):
            del arguments
            executed.append(context.turn)
            context.session.done = True
            return ToolResult("finished", done=True)

        tool = FunctionTool("finish", "finish the session",
                            {"type": "object"}, finish)
        task = TaskSpec(
            "terminal", "finish once",
            ComponentSpec("tests.test_core_framework:FakeEnvironment"))
        trajectory = await ToolHarness(
            MultiCallModel(), environment=environment, tools=[tool]).run(task)

        self.assertEqual(executed, [0])
        validate_transcript(trajectory.messages)
        results = trajectory.metadata["tool_results"]
        self.assertEqual([result["call_id"] for result in results],
                         ["first", "second"])
        self.assertTrue(results[1]["metadata"]["skipped"])
        self.assertEqual(results[1]["metadata"]["reason"], "episode_done")

    async def test_malformed_and_unknown_tool_calls_become_results(self):
        environment = FakeEnvironment()

        class BrokenCallModel:
            async def complete(self, messages, tools, sampling):
                del messages, tools, sampling
                return {
                    "role": "assistant", "content": None,
                    "tool_calls": [
                        {"id": "bad-json", "type": "function", "function": {
                            "name": "increment", "arguments": "{"}},
                        {"id": "invented", "type": "function", "function": {
                            "name": "teleport", "arguments": "{}"}},
                    ],
                }

        def increment(arguments, context):
            raise AssertionError("malformed arguments must not execute")

        tool = FunctionTool("increment", "increase a counter",
                            {"type": "object"}, increment)
        task = TaskSpec(
            "malformed", "try a call",
            ComponentSpec("tests.test_core_framework:FakeEnvironment"))
        trajectory = await ToolHarness(
            BrokenCallModel(), environment=environment, tools=[tool],
            max_turns=1).run(task)

        validate_transcript(trajectory.messages)
        results = trajectory.metadata["tool_results"]
        self.assertEqual([row["call_id"] for row in results],
                         ["bad-json", "invented"])
        self.assertIn("invalid JSON", results[0]["metadata"]["malformed"])
        self.assertIn("unknown tool", results[1]["metadata"]["malformed"])

    async def test_harness_caps_calls_and_resolves_every_emitted_call(self):
        environment = FakeEnvironment()

        class ManyCallModel:
            async def complete(self, messages, tools, sampling):
                del messages, tools, sampling
                return {
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": f"call_{index}", "type": "function",
                        "function": {"name": "increment",
                                     "arguments": {"amount": 1}},
                    } for index in range(5)],
                }

        def increment(arguments, context):
            context.session.value += arguments["amount"]
            return ToolResult("incremented")

        tool = FunctionTool(
            "increment", "increase a counter",
            {"type": "object", "properties": {
                "amount": {"type": "integer"}}}, increment)
        task = TaskSpec(
            "capped", "try many calls",
            ComponentSpec("tests.test_core_framework:FakeEnvironment"))
        trajectory = await ToolHarness(
            ManyCallModel(), environment=environment, tools=[tool],
            max_turns=1).run(task)

        self.assertEqual(environment.session.value, 4)
        results = trajectory.metadata["tool_results"]
        self.assertEqual(len(results), 5)
        self.assertEqual(results[-1]["metadata"]["reason"],
                         "turn_call_limit")

    async def test_malformed_call_uses_registered_tool_accounting_hook(self):
        environment = FakeEnvironment()
        observed = []

        class BrokenCallModel:
            async def complete(self, messages, tools, sampling):
                del messages, tools, sampling
                return {
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": "bad-json", "type": "function",
                        "function": {
                            "name": "increment", "arguments": "{"},
                    }],
                }

        class StatefulTool:
            name = "increment"
            schema = {
                "type": "function", "function": {
                    "name": name, "description": "increment",
                    "parameters": {"type": "object"},
                },
            }

            def execute(self, call, context):
                del call, context
                raise AssertionError("malformed arguments must not execute")

            async def malformed(self, error, context):
                observed.append((str(error), context.metadata["count_turn"]))
                context.session.value -= 1
                context.session.done = True
                return ToolResult(
                    "charged malformed turn",
                    {"malformed": str(error), "step_reward": -0.2},
                    done=True, reward=-0.2)

        task = TaskSpec(
            "malformed", "try a call",
            ComponentSpec("tests.test_core_framework:FakeEnvironment"))
        trajectory = await ToolHarness(
            BrokenCallModel(), environment=environment,
            tools=[StatefulTool()],
            probes=[FunctionProbe("counter", lambda session: session.value)],
            max_turns=1).run(task)

        self.assertEqual(len(observed), 1)
        self.assertIn("invalid JSON", observed[0][0])
        self.assertTrue(observed[0][1])
        self.assertEqual(environment.session.value, -1)
        self.assertEqual(trajectory.messages[-1]["content"],
                         "charged malformed turn")
        result = trajectory.metadata["tool_results"][0]
        self.assertEqual(result["reward"], -0.2)
        self.assertEqual(result["metadata"]["step_reward"], -0.2)

    async def test_unknown_call_uses_the_registry_malformed_fallback(self):
        environment = FakeEnvironment()

        class UnknownToolModel:
            async def complete(self, messages, tools, sampling):
                del messages, tools, sampling
                return {
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": "unknown", "type": "function",
                        "function": {"name": "invented", "arguments": {}},
                    }],
                }

        class StatefulFallback:
            name = "stateful"
            malformed_fallback = True
            schema = {"type": "function", "function": {
                "name": name, "description": "stateful",
                "parameters": {"type": "object"}}}

            def execute(self, call, context):
                del call, context
                raise AssertionError("the unknown tool must not execute")

            def malformed(self, error, context):
                context.session.value -= 1
                context.session.done = True
                return ToolResult(
                    "charged unknown call", {"malformed": str(error)},
                    done=True, reward=-0.2)

        task = TaskSpec(
            "unknown", "try a call",
            ComponentSpec("tests.test_core_framework:FakeEnvironment"))
        trajectory = await ToolHarness(
            UnknownToolModel(), environment=environment,
            tools=[StatefulFallback()], max_turns=1).run(task)

        self.assertEqual(environment.session.value, -1)
        self.assertEqual(trajectory.messages[-1]["content"],
                         "charged unknown call")
        self.assertIn("unknown tool", trajectory.metadata[
            "tool_results"][0]["metadata"]["malformed"])

    async def test_known_tool_without_handler_does_not_use_fallback(self):
        fallback_calls = []

        class Extension:
            name = "lookup"
            schema = {"type": "function", "function": {
                "name": name, "description": "lookup",
                "parameters": {"type": "object"}}}

            def execute(self, call, context):
                del call, context
                return ToolResult("ok")

        class Fallback:
            name = "computer"
            malformed_fallback = True
            schema = {"type": "function", "function": {
                "name": name, "description": "stateful",
                "parameters": {"type": "object"}}}

            def execute(self, call, context):
                del call, context
                return ToolResult("ok")

            def malformed(self, error, context):
                fallback_calls.append((error, context))
                return ToolResult("charged")

        registry = ToolRegistry([Fallback(), Extension()])
        result, accounted_by = await registry.account_malformed(
            ToolCall("call", "lookup", {}), ValueError("bad arguments"),
            ToolContext(None, None, None, 0))

        self.assertIsNone(result)
        self.assertIsNone(accounted_by)
        self.assertEqual(fallback_calls, [])

    async def test_event_sequence_reward(self):
        environment = FakeEnvironment()
        environment.session.events = [
            {"name": "opened"}, {"name": "crafted"}, {"name": "closed"}]
        # Use a tiny completed trajectory from the harness test's primitives.
        from bedrock_rl.core.state import StateTrace
        from bedrock_rl.core.trajectory import Trajectory
        state = StateTrace()
        state.extend_events(environment.session.drain_events())
        result = await evaluate(EventSequenceReward(["opened", "closed"]),
                                Trajectory(), state)
        self.assertEqual(result.total, 1.0)


class ComponentTests(unittest.TestCase):
    def test_builtin_computer_schema_has_no_external_config_file(self):
        schema = tool_schema()[0]
        self.assertEqual(schema["function"]["name"], "computer")
        parameters = schema["function"]["parameters"]
        self.assertEqual(parameters["required"], ["actions"])
        self.assertEqual(parameters["properties"]["actions"]["type"],
                         "array")

    def test_import_paths_replace_central_registries(self):
        self.assertIs(load_object("json:loads"), json.loads)
        spec = ComponentSpec.parse({"type": "builtins:dict", "answer": 42})
        self.assertEqual(spec.create(), {"answer": 42})

    def test_group_guidance_is_homogeneous(self):
        decisions = group_guidance(["a", "a", "b", "b"], 0.5,
                                   rng=random.Random(3))
        self.assertEqual(decisions[0], decisions[1])
        self.assertEqual(decisions[2], decisions[3])

    def test_run_check_resolves_every_component_without_constructing(self):
        run = RunSpec.from_dict({
            "task": {
                "name": "counter", "instruction": "count",
                "environment": "tests.test_core_framework:FakeEnvironment",
            },
            "model": "tests.test_core_framework:FakeModel",
            "tools": ["builtins:dict"],
        })
        names = [name for name, _ in run.validate_components()]
        self.assertEqual(names,
                         ["task.environment", "model", "harness", "tool[0]"])

    def test_run_rejects_unsupported_schema_version(self):
        with self.assertRaisesRegex(
                ValueError, "unsupported run schema_version 2"):
            RunSpec.from_dict({
                "schema_version": 2,
                "task": {
                    "name": "counter", "instruction": "count",
                    "environment":
                    "tests.test_core_framework:FakeEnvironment",
                },
                "model": "tests.test_core_framework:FakeModel",
            })

    def test_dynamic_image_context_never_reencodes_sampled_tokens(self):
        segments = [
            TokenSegment("initial", [10, 11], ("i0",), ("image0",), [10]),
            TokenSegment("assistant", [20], sampled_image_ids=("i0",),
                         logprobs=[-0.2]),
            TokenSegment("tool", [30, 31], ("i1",), ("image1",), [30]),
            TokenSegment("assistant", [40], sampled_image_ids=("i1",),
                         logprobs=[-0.4]),
        ]
        prompt, images, selected = render_segments(segments[:3], 1)
        self.assertEqual(prompt, [10, 20, 30, 31])
        self.assertEqual(images, ["image1"])
        self.assertEqual(selected, ("i1",))
        out = finalize_segments(segments, 1)
        self.assertEqual(out[0], [10])
        self.assertEqual(out[1], [20, 30, 31, 40])
        # Turn 0 sampled with i0, which the final view dropped. Turn 1's
        # exact sampled token remains loss-carrying.
        self.assertEqual(out[2], [0, 0, 0, 1])
        self.assertEqual(out[3], [0.0, 0.0, 0.0, -0.4])
        self.assertEqual(out[4], ["image1"])
        self.assertEqual(out[5], 1)

    def test_image_window_can_cut_through_a_multi_image_segment(self):
        segments = [
            TokenSegment(
                "initial", [10, 11, 12], ("i0", "i1"),
                ("image0", "image1"), [10],
                image_variants={("i1",): [10, 12]}),
            TokenSegment("assistant", [20], sampled_image_ids=("i1",),
                         logprobs=[-0.2]),
        ]

        prompt, images, selected = render_segments(segments[:1], 1)
        self.assertEqual(prompt, [10, 12])
        self.assertEqual(images, ["image1"])
        self.assertEqual(selected, ("i1",))
        out = finalize_segments(segments, 1)
        self.assertEqual(out[0], [10, 12])
        self.assertEqual(out[1], [20])
        self.assertEqual(out[2], [1])
        self.assertEqual(out[4], ["image1"])

    def test_terminal_tool_frame_does_not_displace_training_context(self):
        segments = [
            TokenSegment("initial", [10, 11], ("before",),
                         ("before-image",), [10]),
            TokenSegment("assistant", [20, 21],
                         sampled_image_ids=("before",),
                         logprobs=[-0.2, -0.3]),
            TokenSegment("tool", [30, 31], ("terminal",),
                         ("terminal-image",), [30]),
        ]

        out = finalize_segments(segments, 1)

        self.assertEqual(out[0], [10, 11])
        self.assertEqual(out[1], [20, 21, 30])
        self.assertEqual(out[2], [1, 1, 0])
        self.assertEqual(out[3], [-0.2, -0.3, 0.0])
        self.assertEqual(out[4], ["before-image"])
        self.assertEqual(out[5], 0)


if __name__ == "__main__":
    unittest.main()
