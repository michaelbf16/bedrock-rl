# Documentation

A common project layout uses three files:

```text
my_task/
├── task.yaml   # Minecraft world, instruction, and reward
├── data.yaml   # generated cases or demonstrations
└── train.yaml  # model, trainer, and compute
```

These names are conventions used by the included examples, not required
filenames or registry keys. `bedrock task`, `bedrock generate`, and
`bedrock train` accept explicit paths to user-authored `.yaml` or `.yml`
files with any names. A `task:` path inside a generation or training manifest
is resolved relative to that manifest. Only a bare task name uses the optional
lookup under `examples/<name>/task.yaml`.

Start with the page for the thing you are changing:

| Goal | Guide |
| --- | --- |
| Understand package boundaries and extension points | [Architecture](architecture.md) |
| Create a Minecraft environment and reward | [Make a task](tasks.md) |
| Generate prompts or successful demonstrations | [Synthetic data](synthetic-data.md) |
| Train locally, on several nodes, or on Modal | [Launch training](training.md) |
| Use or register another VLM family | [Models](models.md) |
| Add tools, context management, hints, probes, or a harness | [Customize the agent](custom-agents.md) |
| Understand the implemented learning methods | [Training methods](trainers.md) |

The concerns are separate so changing a trainer cannot change the task, and
changing a data split cannot change its verifier. You only need a task
definition to build an environment. Add generation and training manifests
when the project needs them.

The included GRPO example is the smallest complete reference:

```bash
uv run bedrock task validate examples/equip_pickaxe_grpo/task.yaml
uv run bedrock generate examples/equip_pickaxe_grpo/data.yaml --check
uv run bedrock train examples/equip_pickaxe_grpo/train.yaml --dry-run
```

Framework-owned relative paths inside these manifests resolve beside the file
that contains them, so an experiment directory can be copied without
rewriting paths.
