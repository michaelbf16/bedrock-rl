# Equip the iron pickaxe with GRPO

The hotbar contains nine randomized items. Qwen3-VL 2B must read the frame and
press the number key for the iron pickaxe. The verifier reads the live selected
slot; generated text and action heuristics do not affect reward.

## Ten-step smoke run

Qwen3-VL 2B was trained directly with GRPO, without an SFT warm-start. Every
checkpoint was evaluated at temperature 0 on the same 288 frozen development
prompts. The lower panel is the mean reward of every on-policy training batch;
the upper panel is checkpoint success on that frozen development set.

![Development evaluation through GRPO step 10](curves/learning_curve.svg)

| Policy | Success | Mean reward |
| --- | ---: | ---: |
| Base | 27/288 (9.4%) | -0.158 |
| GRPO step 5 | 33/288 (11.5%) | -0.111 |
| GRPO step 10 | 39/288 (13.5%) | -0.085 |

The step-10 paired change was 38 failures fixed and 26 base successes lost
(two-sided exact `p=0.169`). Treat this as an end-to-end smoke result, not a
claim of statistically established learning. The raw trial records,
seed-clustered intervals, summary, curve, and all ten training metric rows are
in [`curves`](curves). The sealed test job was not opened because no
development checkpoint met that stronger bar.

At temperature 1 with three samples per prompt, pass@3 was 31/288 (10.8%) for
the base and 64/288 (22.2%) at step 10. The paired comparison had 37 failures
fixed, 4 successes lost, and a two-sided exact `p=1.03e-7`. This is a
stochastic candidate-coverage result; it does not replace the deterministic
single-attempt curve. Both 864-trial record files and
`pass3_summary.json` are included for audit.

Among the 36 unique worlds in each of train and dev, every possible target
slot appears exactly four times. Each 288-row generated split cycles those
worlds eight times, so every slot appears in 32 rows. The world seeds are
disjoint: train uses seeds in the 1-60 range and dev uses seeds in the
10001-10117 range. Sibling GRPO samples share the same world and hotbar, so
group-relative advantages compare decisions at the same state. The `test` job
samples fresh worlds only after a checkpoint has been selected on dev.

## Run it

This example has exactly three YAML files:

- `task.yaml` defines the Minecraft state, instruction, budget, and reward.
- `data.yaml` defines independent `train`, `dev`, and held-out `test` jobs.
- `train.yaml` selects Qwen3-VL 2B, GRPO, and compute settings.

Those names describe this example's layout; they are not required names.
Copied files may be renamed, provided the task paths inside the generation and
training manifests still point to the chosen task file.

Paths are relative to the file that contains them, so this directory can be
copied as a unit. See the [documentation index](../../docs/README.md) to make
a task, generate data, launch training, or add a model.

```bash
uv run bedrock generate examples/equip_pickaxe_grpo/data.yaml --check
uv run bedrock generate examples/equip_pickaxe_grpo/data.yaml train
uv run bedrock generate examples/equip_pickaxe_grpo/data.yaml dev
uv run bedrock train examples/equip_pickaxe_grpo/train.yaml
```

Evaluate checkpoints on the frozen development rows:

```bash
uv run bedrock eval trials examples/equip_pickaxe_grpo/task.yaml \
  --data data/equip_pickaxe_grpo_cu/heldout_prompts.parquet \
  --model PATH_TO_CHECKPOINT -n 288 \
  -- --temp 0 --dp 8 --records-out curves/POLICY.records.json
```

After selecting a checkpoint, generate `data.yaml test` once and compare only
the base and selected policy on those fresh paired trials. Keep every raw
record and use `bedrock eval summarize` for paired deltas and seed-clustered
confidence intervals; do not select a checkpoint on the test split.
