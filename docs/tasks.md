# Make a task

A task owns the Minecraft reset, the instruction, interaction limits, and the
reward. It does not choose a model, trainer, or train/test split.

Start by copying an existing task definition. `task.yaml` is only the example
convention: save the mapping under any `.yaml` or `.yml` filename and pass its
path to the CLI. Bare names such as `equip_pickaxe_grpo` optionally resolve to
`task.yaml` in the matching included example; user-authored tasks do not need
to be registered anywhere. This is a complete small task:

```yaml
name: select_pickaxe
view: netherite-procedural
goal: Select the iron pickaxe in the hotbar.
goal_cu: Press its number key, then stop.

max_turns: 1
max_lines: 1
max_ticks: 40
max_ticks_per_turn: 5

initial_state:
  inventory:
    - {item: iron_pickaxe, count: 1, slot: random_hotbar}
    - {item: dirt, count: 32, slot: random_hotbar}
  player: {health: 20, food: 20}
  world: {mode: survival, difficulty: normal, weather: clear, mobs: false}

no_commit: -0.25
success: {type: selected_item, items: [iron_pickaxe], reward: 1.0}
```

Check the environment before writing any training config:

```bash
uv run bedrock task validate path/to/select-pickaxe.yaml
uv run bedrock task show path/to/select-pickaxe.yaml
uv run bedrock task render path/to/select-pickaxe.yaml /tmp/task-preview
uv run bedrock smoke path/to/select-pickaxe.yaml
```

## What a task can set

- `initial_state.inventory`: item, count, metadata, and fixed or randomized
  hotbar slots.
- `initial_state.player`: selected item, health, food, saturation, and
  exhaustion.
- `initial_state.spawn`: fixed points with pose, or distance constraints from
  blocks. Constraint-selected spawns may set integer `hazard_clearance` to
  require that many dry, non-hazard voxels around the player's body.
- `initial_state.world`: time and mobs. The optional `mode`, `difficulty`,
  and `weather` declarations can document the engine defaults, but Netherite
  currently rejects anything except `survival`, `normal`, and `clear` rather
  than pretending unsupported world rules took effect.
- `initial_state.blocks`: ordered single-block or cuboid world edits.
- `view`: `semantic`, `netherite-procedural`, `minecraft-official`, or a
  custom view component.
- `max_turns`, `max_lines`, and tick limits: the action budget.
- `success`, `fail`, `shaping`, and `no_commit`: the reward contract.

Checks can use live state or journaled events such as movement, selected
items, visible blocks, mining, placement, crafting, smelting, GUI use, and
compositions (`all`/`any`), and ordered `sequence` checks. Reward-only state
is never added to model messages. A journal-backed check always enables the
event stream even if the process-level performance default disables it;
`journal: false` is rejected for such a task. Numeric event constraints use
task units (for example, blocks), including equality as well as `min_` and
`max_` bounds.

## Choose the model's Minecraft view

The task—not the trainer—selects its pixels:

```yaml
view: semantic              # block/depth/edge camera; GUI fallback
view: netherite-procedural  # deterministic redistributable textures
view: minecraft-official    # textures built from a client jar you own
```

The strip below is an illustrative comparison of one 428×240 observation per
view; it is not a reproducibility record for a particular seed or camera pose:

![Example Minecraft observations in semantic, Netherite procedural, and Minecraft official views](assets/views.png)

Training, evaluation, synthetic generation, and saved model views all use the
task's selection. Renderer identity is recorded in the trajectory.

For an authored area, block edits are applied in order:

```yaml
initial_state:
  blocks:
    - {block: air, x: [-8, 8], y: [65, 72], z: [-8, 8]}
    - {block: stone, x: [-8, 8], y: 64, z: [-8, 8]}
    - {block: iron_ore, x: 4, y: 65, z: 6}
```

Use natural seeds when the task is about survival-world generalization. Use
authored blocks when exact geometry is part of the environment.

## Custom behavior

Framework components are normal `package.module:object` imports. A custom
reward check can inspect the live environment without changing the agent:

```yaml
success:
  type: my_project.rewards:ReachedStructure
  config: {block: beacon, radius: 6}
  reward: 1.0
```

`ReachedStructure(block, radius)` may be a callable
`(env, start_observation) -> bool` or expose
`holds(env, start_observation) -> bool`. The task stores only JSON config;
the component reads the live engine state when reward is evaluated.

Use the same component mechanism for custom views, tools, context policies,
state probes, and harnesses. The generic `bedrock trial` runner is useful
when the whole agent harness is custom; verl training can select a custom
agent loop from a training manifest (called `train.yaml` in the examples).

See [`examples/equip_pickaxe_grpo`](../examples/equip_pickaxe_grpo) for a
minimal complete RL task and [`examples/farm_one_log_sdg`](../examples/farm_one_log_sdg)
for a user-authored synthetic-data task.
