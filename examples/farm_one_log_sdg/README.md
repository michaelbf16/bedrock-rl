# Farm one log

This is the smallest complete synthetic-data and warm-start recipe: find a natural tree,
break one trunk block, collect its drop, and verify that a new log entered the
inventory. Coordinates come from each world's snapshot; the YAML contains no
seed-specific route.

This directory uses the conventional names `task.yaml` and `data.yaml` only
for readability. Copy them under any filenames, then keep the generation
manifest's relative `generator.task` path pointed at the task file you chose.
No registration step is required.

```bash
uv run bedrock generate examples/farm_one_log_sdg/data.yaml --check
uv run bedrock generate examples/farm_one_log_sdg/data.yaml
uv run bedrock train examples/farm_one_log_sdg/train.yaml warmstart
```

The manifest asks for 100 accepted worlds. Unreachable or unsuitable seeds
are replaced automatically, while accepted rows retain their world seed,
independent policy seed, snapshot hash, selected target, safe route, and exact
computer actions. `TrajectorySFTWriter` writes one causal row per assistant
action, keeps only its newest model-visible frame, and refuses trajectories
whose policy does not declare the target selection model-grounded.

To continue the resulting LoRA with RL, generate ordinary hint-free prompt
rows and select the second training preset:

```bash
uv run bedrock generate examples/farm_one_log_sdg/rl-prompts.yaml
uv run bedrock train examples/farm_one_log_sdg/train.yaml rl
```

The two training presets share the same base model and image window. The RL
preset loads the adapter written by `warmstart`; neither stage receives the
planner snapshot or its provenance.
