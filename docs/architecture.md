# Architecture

bedrock-rl separates reusable agent contracts, the Minecraft runtime adapter,
synthetic-data generation, and training. Imports point from general contracts
toward concrete workflows:

```text
core                         dependency-light agent and trajectory contracts
  └── env                    task loading, episodes, observations, reward
      └── adapters/netherite Minecraft computer tool and engine integration
          └── sdg            generation runtime and scripted experts

adapters/verl                verl integration over core + env + Netherite
train                        standalone trainers over those runtime layers
eval                         paired evaluation and reporting
cli + launch                 user and infrastructure front ends
```

`core` does not import the environment, adapters, SDG, or trainers. `env` does
not import adapters, SDG, or trainers. Adapters do not import product workflows.
SDG may use `core`, `env`, and adapters, but not the CLI, launchers, evaluation,
or trainers. Architecture tests enforce those directions.

## Package ownership

| Path | Sole responsibility |
| --- | --- |
| `bedrock_rl/core/` | Generic messages, tools, context, guidance, state, reward, task, run, harness, and trajectory contracts. |
| `bedrock_rl/env/` | The patched Minecraft engine lifecycle, task schema, snapshots, observations, perception, and world data. |
| `bedrock_rl/adapters/netherite/` | Translation between the generic contracts and Netherite's computer tool, chat format, probes, and live environment. It contains no data-generation policy. |
| `bedrock_rl/adapters/verl/` | verl agent-loop, token-segment, console, and distillation integration. |
| `bedrock_rl/sdg/` | Samplers, executors, rejection accounting, writers, privileged guidance, replay media, and the Netherite demonstration pipeline. |
| `bedrock_rl/sdg/policy/` | Scripted Minecraft expert behavior. `execution/` advances certified plans; `progression/` composes longer objectives. |
| `bedrock_rl/train/` | SFT, SDFT, SDPO, self-distillation, rollout, PEFT, and export implementations. `sft_data.py` is trainer-owned input validation and lazy encoding. |
| `bedrock_rl/eval/` | Trial execution, paired summaries, clustered confidence intervals, and plots. |
| `bedrock_rl/config/` | Training configuration, overrides, and path resolution shared by front ends. |
| `bedrock_rl/cli/` | Implementations of user-facing `bedrock` commands. |
| `bedrock_rl/launch/` | Local/cluster and Modal launch adapters; it may use config but not CLI internals. |
| `bedrock_rl/templates/` | Shipped chat, DeepSpeed, and verl configuration data. |

The modules directly under `bedrock_rl/` are intentionally small shared
facades rather than an unowned utility directory: `data.py` handles common
serialization, `lora.py` owns adapter metadata, `models.py` owns the model
registry, `reporting.py` owns metrics presentation, and `resources.py` resolves
checkout and wheel resources.

## Repository ownership

| Path | Contents |
| --- | --- |
| `.github/workflows/` | CI only. |
| `THIRD_PARTY_LICENSES/` | Notices for vendored or patched dependencies. |
| `bin/` | Shell launchers for setup and GPU training environments. Python product commands belong in `cli/`; Python build utilities belong in `tools/`. |
| `docs/` | Maintained user and contributor guides; `docs/assets/` contains only media referenced by those guides. |
| `examples/` | Complete runnable workflows, each owning its task, generation, training, and evidence files. |
| `patches/` | Ordered, pinned upstream Netherite and verl changes consumed identically by setup, Docker, and CI. |
| `requirements/` | The separately pinned accelerator environment. |
| `tests/` | Behavioral and architecture-contract tests. They remain flat because CI and cross-test fixtures use their stable module names; group by product prefix, not cosmetic directories. |
| `tools/` | Repository/build maintenance utilities that are not user-facing commands. |

Generated datasets, snapshots, checkpoints, logs, and run outputs stay outside
the package and Git history. Writers replace final outputs atomically. Accepted
rows retain world and decision seeds plus component provenance so executor
parallelism does not change reproducibility.

## Extension points

Generation YAML selects a sampler, generator, executor, and writer by
`package.module:object` import path. A custom project can replace any one of
those components. Task YAML remains declarative and owns the episode's world,
budgets, failure conditions, and reward checks; generation code must not weaken
those checks to fill a dataset.

The checked-in examples use conventional filenames—`task.yaml`, `data.yaml`,
and `train.yaml`—but commands accept explicit YAML paths with any name.
