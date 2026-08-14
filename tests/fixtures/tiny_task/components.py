"""Small components used to exercise the generic task runner."""

import json

from bedrock_rl.core import RewardResult, ToolResult


class CounterSession:
    def __init__(self, target=3):
        self.count = 0
        self.target = int(target)

    @property
    def done(self):
        return self.count >= self.target

    def initial_messages(self, task):
        return [{"role": "user", "content": task.instruction}]

    def drain_events(self):
        return []

    def close(self):
        return None


class CounterEnvironment:
    def __init__(self, target=3):
        self.target = int(target)

    def open(self, task):
        return CounterSession(self.target)


class IncrementTool:
    name = "increment"
    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": "Increase the counter.",
            "parameters": {
                "type": "object",
                "properties": {"amount": {"type": "integer", "minimum": 1}},
                "required": ["amount"],
            },
        },
    }

    def execute(self, call, context):
        context.session.count += int(call.arguments["amount"])
        return ToolResult(json.dumps({"count": context.session.count}),
                          done=context.session.done)


class ScriptedModel:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools, sampling):
        self.calls += 1
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": f"call_{self.calls}",
                "type": "function",
                "function": {"name": "increment",
                             "arguments": json.dumps({"amount": 1})},
            }],
        }


class CounterProbe:
    name = "counter"

    def capture(self, session):
        return {"count": session.count, "target": session.target}


class CounterReward:
    def evaluate(self, trajectory, state):
        final = state.latest["counter"]
        success = final["count"] >= final["target"]
        score = 1.0 if success else 0.0
        return RewardResult(score, {"target_reached": score},
                            {"success": success})


class DemoGuidance:
    def static_hint(self, request, level):
        return "Use the increment tool until the target is reached."

    def dynamic_hint(self, request, level):
        counter = (request.state or {}).get("counter", {})
        return (f"Current progress: {counter.get('count', 0)}/"
                f"{counter.get('target', '?')}.")
