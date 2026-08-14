# Launch training

Training consumes a task and generated data. It does not redefine the world
or its reward.

Generate the requested split first, then create a training manifest. The
examples call it `train.yaml`, but `bedrock train` accepts any YAML path and
does not register or infer configs by filename:

```bash
uv run bedrock generate path/to/dataset-plan.yaml train
```

```yaml
trainer: rl
task: objective.yaml
model: qwen3-vl-2b
steps: 40
algorithm: rloo

train_bs: 8
rollout_n: 4
micro_bs: 4
model_len: 32768
save_freq: 10
```

Inspect the resolved command before allocating a GPU:

```bash
uv run bedrock train path/to/learner.yaml --dry-run
```

For local NVIDIA training, build and check the pinned Linux/CUDA stack once,
then launch the same config:

```bash
uv run bedrock setup
uv run bedrock doctor
uv run bedrock train path/to/learner.yaml
```

Use `--set steps=2` for a short smoke run. To see every accepted setting for
a trainer, run `uv run bedrock train --knobs rl` and replace `rl` with `sft`,
`sdft`, `sdpo`, `opd`, or `mopd`.

## Choose a method

| `trainer` | Use it for |
| --- | --- |
| `rl` | verl-backed GRPO, RLOO, REINFORCE++, ReMax, or GSPO loss |
| `sft` | supervised training on successful interactions |
| `sdft` | on-policy self-distillation from demonstrations |
| `sdpo` | self-distillation with static or per-turn private guidance |
| `opd` | on-policy distillation from one separate teacher |
| `mopd` | routed on-policy distillation from several teachers |

The exact losses and method-specific constraints are in [Training
methods](trainers.md).

## GPUs and clusters

When `cuda_visible_devices` is omitted, the launcher uses the GPUs assigned by
the scheduler or all local GPUs reported by `nvidia-smi`.

The experimental Modal multi-node adapter launches the same config; it does
not require another YAML file. The manifest may have any filename, but it and
its referenced project files must live inside the checkout copied into the
Modal image:

```bash
uv sync --extra modal
uv run bedrock train path/to/learner.yaml --modal --gpu H100
uv run bedrock train path/to/learner.yaml --modal --nodes 2 --gpu H100
```

Give a long run a stable output identity with `--modal-run RUN_ID`. Repeating
the same command with that ID reuses its durable output directory and asks SFT
or verl to resume from any complete checkpoint already there. The launcher
commits checkpoints and metrics on trainer failure before propagating the
error; a resume can only continue from checkpoints the trainer had actually
written.

Visual verl rollouts create private frame scratch space automatically and
require one rendered observation after every computer turn. Set
`brl_frames_root` only when that scratch location must live on a specific
filesystem; leaving it unset does not disable per-turn pixels.

The image is locked to CUDA 12.8. The launcher rejects `B300` and the
ambiguous `B200+` selector because those may require a CUDA 13.1 runtime;
request `B200` explicitly when that family is intended.

Require `--modal-smoke` to pass in your workspace before treating the Modal
multi-node path as supported; it checks CUDA, cross-node NCCL, and Ray without
training. For a non-Modal multi-node verl job, start one Ray cluster, make the
engine and weights available at the same paths on every node, and set
`RAY_ADDRESS`, `NNODES`, and `GPUS_PER_NODE`.

## Metrics and evaluation

Every trainer writes JSONL metrics. The shared reporter sends the same
step-by-step metrics to W&B, while verl trainers can also write native
TensorBoard events:

```yaml
wandb: true
wandb_project: minecraft-agents
tensorboard: true
tensorboard_dir: outputs/tensorboard
```

Evaluate a checkpoint on a frozen dev set:

```bash
uv run bedrock eval trials path/to/objective.yaml \
  --data path/to/dev_prompts.parquet \
  --model PATH_TO_CHECKPOINT -n 256
```

Select checkpoints on dev. Run the sealed test set only after selection, and
compare it with the base model on the same cases. A successful launch or a
falling loss is not evidence that the Minecraft policy improved.
