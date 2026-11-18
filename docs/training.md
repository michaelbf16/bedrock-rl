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

With a bounded image history, multi-turn SFT expands each full conversation
into one causal prefix per assistant action. Each prefix uses the images
available before that action and applies loss only to its final response.
Rows already marked `loss_last_assistant_only: true` keep their chosen target.
Self-distillation harvests use the same prefixes: `kept` and `yield` count
accepted episodes, while `examples` counts supervised actions. The launcher
requires a fresh receipt from that harvest before starting SFT; an empty run
preserves an older parquet but never trains on it.

Hub-backed LoRA checkpoints and exported verl adapters record the base model's
repository and exact revision, including when training loaded a local Hub
cache snapshot. Moving the adapter to another host reloads that same commit.
An adapter trained on custom local weights still names those local weights;
move the base checkpoint too and update its reference when relocating it.
The standalone vLLM sampler defaults to spawned workers so processor or model
setup can initialize CUDA before the rollout engine starts.

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

The image is locked to CUDA 13.0, PyTorch 2.10.0 and vLLM 0.17.1.
It requires an NVIDIA driver supporting CUDA 13 (580 or newer).
The compatible FlashAttention wheel is built for that same Torch/CUDA ABI.
Transformers remains on 4.57.6 for vLLM compatibility; setup and Docker apply
the upstream CVE-2026-9856 template-path checks, and standalone processor
loads enable the same save validation. Re-run setup after reinstalling the
training environment.

The launcher rejects `B300` and the
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

New prompt rows carry the resolved task definition and its digest. Evaluation
checks that binding before opening an episode; `--task` must match its content,
even when the original file was edited in place. Records fingerprint the task,
hint text, selected prompt/image rows, rendered initial prompt, processor,
runtime, engine binary, patch/asset manifests, and restored snapshot. Matching
paths alone cannot certify a paired comparison, and moving identical inputs
to another path does not change their content identities.

Regenerate older prompt files when possible. To deliberately evaluate rows
without a recorded task digest, pass `-- --allow-legacy-inputs`; the new records
retain an unverified frozen-task status. To summarize historical records or
these unverified-input runs, `bedrock eval summarize` requires
`--allow-legacy-protocol`. The JSON, CLI output, and plots identify the weaker
provenance. Historical and content-verified protocols cannot be mixed in one
paired comparison; opting in does not add fingerprints to old evidence.

Select checkpoints on dev. Run the sealed test set only after selection, and
compare it with the base model on the same cases. A successful launch or a
falling loss is not evidence that the Minecraft policy improved.
