# Training methods

All trainers operate outside the task boundary. Changing a trainer does not
change the Minecraft reset, tools, pixels, verifier, or saved interaction.
`uv run bedrock train --knobs NAME` prints the exact settings accepted by one
method. Filenames in the snippets are illustrative: the training manifest can
have any YAML filename, and each relative `task:` path resolves beside that
manifest.

## verl RL

```yaml
trainer: rl
task: objective.yaml
model: qwen3-vl-2b
steps: 40
algorithm: grpo       # grpo | rloo | reinforce_plus_plus | remax
rollout_n: 4
loss_mode: vanilla    # or gspo
```

The pinned [verl 0.8 commit](https://github.com/volcengine/verl/tree/7aed6b230776f963fa09509c10d9c3a767d1102c)
owns the optimizers and distributed runtime. `algorithm` selects the advantage
estimator; `loss_mode` is an independent policy-loss choice, so GSPO is, for
example, GRPO advantages with sequence-level ratios. GRPO and RLOO require at
least two rollouts per prompt and the launcher refuses `rollout_n: 1`. ReMax
adds the greedy baseline rollout itself. REINFORCE++ computes token returns and
whitens them over valid response tokens.

The default is the common RLVR recipe with reference KL disabled. It is not
silently fixed to that choice:

```yaml
use_kl_in_reward: true
kl_coef: 0.001
# alternatively:
use_kl_loss: true
kl_loss_coef: 0.001
kl_loss_type: low_var_kl
```

Primary references: [GRPO / DeepSeekMath](https://arxiv.org/abs/2402.03300),
[RLOO](https://arxiv.org/abs/2402.14740),
[REINFORCE++](https://arxiv.org/abs/2501.03262),
[ReMax](https://proceedings.mlr.press/v235/li24cd.html), and
[GSPO](https://arxiv.org/abs/2507.18071).

## SFT

`trainer: sft` uses Transformers causal-language-model training. The shared
model registry supplies the processor and exact rollout chat template. Input
frames remain multimodal inputs, labels cover assistant spans only, and user,
system, observation, and tool-result tokens are `-100`. Multi-turn generated
trajectories can supervise every assistant action or causal prefixes can
supervise only their final action. `TrajectorySFTWriter` emits one bounded
causal prefix per action, and the trainer tensorizes rows lazily instead of
materializing every frame in host RAM. Full fine-tuning, LoRA/QLoRA, DDP, and the
shipped DeepSpeed ZeRO presets are supported.

The SFT trainer, RL rollout, and evaluation all retain full causal image
history by default (`keep_latest_images: 0`). Set the same explicit nonzero
window in every stage when memory requires a bounded view. SFT ingestion
accepts both writer-native typed image parts and the leading `<image>` string
used by question/answer generation and self-distillation, while preserving
the exact separator byte after each image.

## SDFT

`trainer: sdft` implements on-policy Self-Distillation Fine-Tuning. The student
samples without the demonstration. An EMA copy of that same policy evaluates
the exact sampled tokens with an expert demonstration in context, and the
student minimizes exact forward KL at its own visited states. Defaults match
the released practical recipe: EMA update `0.01` and the first three response
tokens omitted. The Hugging Face rollout path is used because this trainer
needs aligned sampling probabilities; it does not claim the current vLLM path
is equivalent. Demonstrations are paired to prompt rows by world seed by
default; `demonstration_match: random` is available only as an explicit
cross-world ablation.

Ordinary LoRA is supported. QLoRA is rejected because this standalone EMA
teacher path does not implement a quantized base; use `trainer: sft` when the
frozen base must be 4-bit.

References: [SDFT paper](https://arxiv.org/abs/2601.19897), the
[authors' release](https://github.com/idanshen/Self-Distillation), and
[TRL SDFT documentation](https://huggingface.co/docs/trl/sdft_trainer).

## SDPO

`trainer: sdpo` is Self-Distilled Policy Optimization, not preference-pair
DPO. Groups are sampled on-policy. An EMA copy of the same policy becomes a
private teacher by receiving static guidance, state-dependent per-turn
feedback, and/or a successful sibling demonstration, then scores the student's
exact response. The released teacher update rate is `0.05` and is configurable
with `teacher_update_rate`; set it to `0` for a fixed reference teacher.
Ordinary LoRA is supported; QLoRA is rejected for the same EMA-teacher reason.
The implementation carries the paper's full-logit top-k distribution, tail
bucket, generalized Jensen-Shannon interpolation, and importance-ratio clamp.
The multimodal adaptation preserves full computer-use chat rather than
flattening it to a text question; success comes from the live verifier.

References: [SDPO paper](https://arxiv.org/abs/2601.20802), the
[authors' implementation](https://github.com/lasgroup/SDPO), and
[TRL SDPO documentation](https://huggingface.co/docs/trl/sdpo_trainer).

## Self-distillation workflow

`workflow: self_distill` is a two-phase data workflow, not another optimizer.
It samples with static or per-turn guidance, keeps trajectories accepted by
the live Minecraft verifier, removes guidance from the student view, and runs
SFT on those successful interactions. Use `trainer: sdft` for the on-policy
forward-KL method described in the SDFT paper.

## OPD and MOPD

OPD distills a separate teacher on the student's own rollouts. MOPD uses the
same method with multiple specialist teachers and routes each dataset row by
`data_source`. Both are direct configuration layers over the pinned verl
teacher manager and distillation losses; Netherite does not maintain a second
implementation.

One teacher:

```yaml
trainer: opd
task: objective.yaml
model: Qwen/Qwen3-VL-8B-Instruct
steps: 100
teachers:
  expert:
    model: Qwen/Qwen3-VL-32B-Instruct
    tensor_parallel: 2
teacher_gpus_per_node: 4
actor_gpus_per_node: 4
```

Several specialists:

```yaml
trainer: mopd
task: objective.yaml
model: Qwen/Qwen3-VL-8B-Instruct
steps: 100
train_files: [data/mining.parquet, data/crafting.parquet]
teachers:
  mining:
    model: /models/mining-teacher
    route: mining
    tensor_parallel: 2
  crafting:
    model: /models/crafting-teacher
    route: crafting
    tensor_parallel: 2
teacher_key: data_source
teacher_gpus_per_node: 4
actor_gpus_per_node: 4
```

Set the route while generating each file:

```yaml
type: bedrock_rl.sdg:EpisodeDatasetGenerator
config: {task: objective.yaml, mode: rl, data_source: mining}
```

verl scores student token IDs under each teacher, so student and teachers must
share the exact tokenizer vocabulary. Before taking GPU resources, the
launcher resolves every teacher to an immutable local snapshot and runs the
tokenizer check against that exact path. verl then consumes the same path, and
`teachers.provenance.json` records the requested id, resolved snapshot,
revision, and artifact inventory. Modal prefetch commits both student and
teacher snapshots to its shared volume. The teacher pool is separate from the
actor pool and their per-node counts must fit the GPUs visible to the job.
MOPD additionally checks
that `replicas × tensor_parallel × data_parallel × pipeline_parallel` exactly
equals the declared teacher pool. The pinned verl rollout engines currently
require `pipeline_parallel: 1`, which Netherite checks before allocation. The
same accounting is used on local Ray, multinode Ray, and Modal.

Defaults use verl's policy-gradient OPD recipe (`k1`, top-k 64, task reward
off). Set `use_task_rewards: true` and `distillation_loss_coef` for a hybrid
Minecraft-reward objective, or `use_policy_gradient: false` with an appropriate
direct distillation loss. See [verl OPD](https://verl.readthedocs.io/en/latest/algo/opd.html),
the [on-policy distillation / GKD paper](https://arxiv.org/abs/2306.13649), and
the [MOPD paper](https://arxiv.org/abs/2606.30406).

## Logging and artifacts

Every method writes compact JSONL metrics. The shared reporter owns W&B so a
run has one client and the JSONL and hosted step series match; verl trainers
(`rl`, `opd`, and `mopd`) can additionally use verl's TensorBoard backend.
Minecraft-playing trainers save canonical OpenAI-chat trajectories with model
views, typed image parts, game state/events, reward components, seeds, and run
provenance.
