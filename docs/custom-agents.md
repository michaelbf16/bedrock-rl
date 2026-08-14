# Customize the agent

Tasks define Minecraft. Agent components define how a model sees history,
chooses tools, and runs its loop. They are connected by import paths, so a
custom component does not require editing bedrock-rl.

Component import paths execute Python. Only run agent, task, data, or training
configs you trust.

## Compose an agent

A standalone run manifest can use the built-in Minecraft session while
replacing any agent component. `run.yaml` is a convention, not a required
filename; `bedrock trial` accepts an explicit path to any YAML file:

```yaml
task:
  name: my_agent
  instruction: Complete the Minecraft task.
  environment:
    type: bedrock_rl.adapters.netherite.environment:NetheriteEnvironment
    task_yaml: objective.yaml
  verifier: bedrock_rl.adapters.netherite.reward:EpisodeReward

model:
  type: bedrock_rl.adapters.chat_completions:ChatCompletionsModel
  model: Qwen/Qwen3-VL-2B-Instruct
  base_url: http://127.0.0.1:8000/v1

tools:
  - bedrock_rl.adapters.netherite.tools:NetheriteComputerTool
  - my_project.tools:RecipeBookTool

context:
  - {type: bedrock_rl.core.context:KeepLastImages, count: 2}
  - {type: bedrock_rl.core.context:KeepLastTurns, count: 4}

probes:
  - bedrock_rl.adapters.netherite.probes:EpisodeProbe
  - bedrock_rl.adapters.netherite.probes:ObservationProbe

max_turns: 20
output: outputs/trajectory.json
```

```bash
uv run bedrock trial agent-run.yaml --check
uv run bedrock trial agent-run.yaml
```

The saved trajectory retains canonical history and captured state. Context
policies create the model-facing view without deleting that history.
Put custom components in any importable Python package; `my_project` below is
only an example name and does not need to live inside bedrock-rl.

## Add a tool

A tool needs a name, an OpenAI function schema, and `execute`. It may be sync
or async and returns `ToolResult`, text, content parts, or the equivalent
mapping.

```python
import json

from bedrock_rl.core.tools import ToolResult


RECIPES = {
    "iron_pickaxe": ["3 iron_ingot across the top", "2 stick in the center"],
}


class RecipeBookTool:
    name = "recipe_book"
    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": "Look up a Minecraft recipe.",
            "parameters": {
                "type": "object",
                "properties": {"item": {"type": "string"}},
                "required": ["item"],
            },
        },
    }

    def execute(self, call, context):
        item = call.arguments["item"]
        content = json.dumps({"item": item, "steps": RECIPES.get(item)})
        return ToolResult(content, metadata={"source": "minecraft-1.11"})
```

`context.session` is the live environment session. Keep tool outputs in
OpenAI message form; put runtime-only objects such as PIL images in
`ToolResult.attachments`.

## Control context

Built-ins cover full history, the last N turns, the last N images, and old
turn summarization:

```yaml
context:
  - type: bedrock_rl.core.context:SummarizeOldTurns
    summarizer: my_project.context:summarize
    keep_last: 2
  - {type: bedrock_rl.core.context:KeepLastImages, count: 2}
```

A custom policy exposes `apply(view, request) -> ContextView`. This adaptive
example delegates transcript-safe turn selection to the built-in policy and
records its own decision:

```python
from bedrock_rl.core.context import ContextTransform, KeepLastTurns


class AdaptiveWindow:
    async def apply(self, view, request):
        inventory = (request.state or {}).get("observation", {}).get(
            "inv_counts", {})
        count = 6 if inventory.get("iron_ingot", 0) else 2
        result = await KeepLastTurns(count).apply(view, request)
        result.transforms.append(ContextTransform(
            "adaptive_window", {"turns": count, "turn": request.turn}))
        return result
```

Select it as `my_project.context:AdaptiveWindow`. Context policies receive the
turn, task, captured state, and metadata; they alter only the exact model view,
never canonical history.

## Add private guidance or state

A guidance provider can implement:

```python
def static_hint(self, request, level): ...
def dynamic_hint(self, request, level): ...
```

Dynamic guidance is recomputed after every turn. It changes a teacher view,
not canonical student messages, which is the safe path for a custom harness,
SDPO, or the self-distillation harvesting workflow. SDFT instead conditions
its EMA teacher on a paired successful demonstration.

A state probe exposes a unique `name` and `capture(session)`. Its return value
must be JSON-serializable. Use probes for reward inputs and debugging data
that should be saved but never shown to the model.

## Replace the harness

Most agents can use `ToolHarness`. Replace it when orchestration itself
changes—for example planning, parallel tool calls, reflection, or another
turn protocol. Its constructor receives `model` plus `environment`, `tools`,
`context`, `guidance`, `probes`, `verifier`, `max_turns`, `view_name`, and
`provenance`. It implements:

```python
async def run(self, task, **kwargs) -> Trajectory: ...
```

Select it with `harness: my_project.harness:Harness` in the run manifest. For
verl training, select a custom verl agent loop instead:

```yaml
trainer: rl
task: objective.yaml
model: qwen3-vl-2b
steps: 40
agent_loop_config: my_agent_loop.yaml
agent_loop: my_agent
```

The built-in verl loop can add tools, probes, or a verifier without replacing
its rollout logic. `core_tools` extends the built-in `computer` tool; a
component whose name is `computer` intentionally replaces it:

```yaml
- name: bedrock
  _target_: bedrock_rl.adapters.verl.agent_loop.NetheriteAgentLoop
  keep_last_images: 2
  core_tools: [{type: my_project.tools:RecipeBookTool}]
  probes: [{type: my_project.state:ProgressProbe}]
  verifier: {type: my_project.reward:Verifier}
```

Point `agent_loop_config` at that file. Context rewriting beyond the built-in
image window changes token causality, so it belongs in a custom verl agent
loop rather than being applied after sampling. A stateful tool that should
account for invented tool names may expose `malformed_fallback = True` and a
`malformed(error, context)` handler; the registry permits exactly one such
fallback. Regardless of a custom verifier, a rollout that emits no tool call
keeps the global no-call reward floor.

The useful extension points are intentionally small:

| Change | Implement |
| --- | --- |
| New action or API | `Tool` |
| Model-visible history | context policy |
| Static or per-turn private help | guidance provider |
| Captured state | state probe |
| Agent orchestration | harness or verl agent loop |
| Reward | task check or verifier |
