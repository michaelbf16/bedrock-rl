# Synthetic data

A generation manifest owns which worlds and decision variants are generated.
The examples call this file `data.yaml`, but `bedrock generate` accepts any
user-chosen YAML path; there is no filename-based registry. It can write RL
prompt rows, successful demonstrations, reviewable trajectories, or custom
records. Framework-owned relative paths—including the task path and outputs—
resolve beside the generation manifest.

## RL prompts

This manifest makes disjoint named jobs without duplicating the generator:

```yaml
defaults:
  generator:
    type: bedrock_rl.sdg:EpisodeDatasetGenerator
    task: objective.yaml
    mode: rl
  executor:
    type: bedrock_rl.sdg:ProcessExecutor
    workers: auto
  writer: bedrock_rl.sdg:ParquetWriter

jobs:
  train:
    count: 500
    seed: 1001
    sampler: bedrock_rl.sdg:RandomCases
    output: ../../data/my_task/rl_prompts.parquet
    metadata: {split: train}

  dev:
    count: 128
    seed: 2001
    sampler: bedrock_rl.sdg:RandomCases
    output: ../../data/my_task/dev_prompts.parquet
    metadata: {split: dev}
```

```bash
uv run bedrock generate path/to/dataset-plan.yaml --check
uv run bedrock generate path/to/dataset-plan.yaml train
uv run bedrock generate path/to/dataset-plan.yaml dev
```

`RandomCases` creates unique world seeds and separate decision seeds.
`SeedCases` uses an explicit seed list. Keep train, dev, and test worlds
disjoint.

`CaseFile` is the deterministic handoff from a cheap seed scout to a final
materializer. Its `path` is JSONL containing either exact serialized cases or
rows with the case under a `case` key. It preserves the selected world seed,
decision seed, variations, and source indexes byte-for-byte; it never
re-samples or shuffles them. This lets a large CPU fleet filter headlessly and
a smaller machine replay only accepted cases with the task's authoritative
renderer:

```yaml
sampler:
  type: bedrock_rl.sdg:CaseFile
  path: accepted-cases.jsonl
generator:
  decision_attempts: 1
```

Use `world_variables` for world setup and `variables` for independent policy
augmentation:

```yaml
sampler:
  type: bedrock_rl.sdg:RandomCases
  world_variables:
    tree_max_distance: {integer: [8, 16]}
  variables:
    movement_max_blocks: {uniform: [2.25, 2.75]}
generator:
  type: bedrock_rl.sdg:ScriptedEpisodeGenerator
  task: objective.yaml
  task_bindings:
    initial_state.spawn.constraints.0.max_distance: world_values.tree_max_distance
  policy: {type: my_project.policy:Policy}
```

`world_variables` are sampled once per world and copied into every variant;
`variables` are redrawn from each variant's independent decision seed. The two
maps may not reuse a key. `task_bindings` can replace any dotted mapping or
list path in the task before its snapshot and episode are created. If a
binding accidentally depends on a per-sample value and gives variants of one
world different task definitions, generation fails before running that world.

For verified demonstrations, request worlds directly:

```yaml
worlds: 50
samples_per_world: 3
max_world_attempts: 1500
```

This finds 50 distinct accepted world seeds and produces three independent
decision variants in each world: exactly 150 rows. When a task declares an
`initial_state`, its per-world snapshot is cached and reused by those variants.
A world is the scheduling and acceptance unit. If any of its three
variants fails verification, all three are discarded and another candidate
world is tried. Set `samples_per_world: 1` for exactly one run in each of 50
different worlds. Use `count` when only the total number of independent cases
matters. Increasing `samples_per_world` preserves
the selected world sequence and every existing sample index; it only adds new
decision variants.

Generation either replaces the destination with exactly
`worlds * samples_per_world` verified rows or exits with an error and leaves an
existing JSONL/parquet destination untouched. `CaseRejected` is for expected
world or policy vetoes and is summarized by reason. Any other generator
exception stops the batch immediately instead of being mistaken for a bad
seed.

## Successful demonstrations

A scripted policy is a closed-loop agent. It receives a fresh live episode
each turn and returns ordinary mouse/keyboard actions accepted by the public
`computer` tool:

```python
import random

from bedrock_rl.env import gui_layout
from bedrock_rl.env.catalog import resolve_items

IRON_PICKAXE = resolve_items(["iron_pickaxe"])[0]


class Policy:
    def reset(self, task, episode, case):
        self.rng = random.Random(case.decision_seed)

    def actions(self, task, episode, case, turn):
        slot = gui_layout.find_stack(episode.env, (IRON_PICKAXE,), 1)
        if slot is None or episode.success:
            return []
        return [{"action": "key", "k": str(slot + 1), "ticks": 1}]
```

Reference it from a demonstration job:

```yaml
worlds: 500
samples_per_world: 1
seed: 3001
max_world_attempts: 5000
sampler:
  type: bedrock_rl.sdg:RandomCases
  variables:
    movement_min_blocks: {uniform: [1.5, 2.0]}
    movement_max_blocks: {uniform: [2.5, 3.0]}
generator:
  type: bedrock_rl.sdg:ScriptedEpisodeGenerator
  task: objective.yaml
  policy: {type: my_project.policy:Policy}
  # Optional: callable(observation_text, actions, turn) -> grounded text.
  assistant_content: my_project.policy:assistant_content
  decision_attempts: 4
  include_images: true
executor:
  type: bedrock_rl.sdg:ProcessExecutor
  workers: auto
writer:
  type: bedrock_rl.sdg:TrajectorySFTWriter
  keep_latest_images: 1
  require_grounded: true
output: ../../data/my_task/sft.parquet
```

The generator accepts a world group only when every run's live verifier
succeeds. Candidate worlds consume `max_world_attempts`; failed or partial groups
never enter the dataset.

`decision_attempts` deterministically tries a bounded number of independent
policy seeds against the same world and sampled variables. It does not retry
missing terrain, globally unreachable resources, exhausted world-plan search,
engine timeouts, or failed snapshot qualification. An accepted trajectory
records its original seed, selected seed, attempt number, and prior rejection
codes; an exhausted candidate writes the same selection history to the
rejection ledger. This improves yield without hiding failures or changing the
natural world.

A policy may optionally expose `provenance()`, `cost_certificate()`, and
`demonstration_grounding()` methods
that return JSON mappings. The trajectory records both without teaching the
generic runner what a route, recipe, or portal means. Expected `CaseRejected`
errors may likewise carry structured `details`; rejection summaries preserve
one example per stable code alongside draw and preflight timings.

Each trajectory keeps its portable episode reconstruction spec once under
`provenance.episode_spec`. Per-turn state samples contain only changing episode
status and captured Minecraft observations, so long demonstrations do not
repeat task paths and view configuration hundreds of times.

For movement policies, use `MovementProfile` from
`bedrock_rl.sdg.policy`. It limits each decision to a realistic
distance, adds optional pauses, and reads the sampled `movement_*` variables
through `profile.vary(case.values)`. World seed and decision variance remain
independent.

Policies that perform ordinary survival play can subclass `SurvivalPolicy`
from the same module. Its `movement_actions(env, x, z)` controller emits a
bounded action only when the observed path is safe, returns a structured
`PolicyVeto` for visible hazards, and supports `sneak=True` for edge movement.
It also adds configurable public-observation guardrails while
the subclass still owns the objective and route:

```yaml
policy:
  type: my_project.policy:Policy
  behaviors:
    hazards:
      enabled: true
      blocks: [flowing_lava, lava, fire, cactus, magma]
      clearance_blocks: 0.55
    descent: {enabled: true, max_drop_blocks: 1}
    pickups: {enabled: true, attempts: 4, wait_ticks: 12}
    lighting: {enabled: true, every_blocks: 10, item: torch}
```

Each section can be disabled independently. `SurvivalPolicy` supplies hazard
inspection and enforced straight-path refusal, bounded-drop checks, pickup retry
timing, public-journal recovery when a pickup lands in the backpack, and
distance-based lighting markers. It does not invent resources, choose a seed,
or decide success. Keep task-specific crafting and navigation in the subclass.
For scripted planners, subclass `MinecraftController`. It adds task-agnostic
public controls for journal and inventory queries, nearby-block discovery,
hotbar selection and recovery, bounded `face_and_move_to`, aiming, exact
mining and placement, 2x2/3x3 GUI crafting, and pickup retries. Exact mining
and placement are deliberately two-turn operations when needed: the first
turn aims, and a later turn clicks only after the live `click_target` event
proves the requested voxel or destination. The controller contains no world
coordinates, resource order, or goal-specific state machine.

For small natural-resource objectives, the shipped
`bedrock_rl.sdg.policy.planned_mine:PlannedMinePolicy` avoids a
custom subclass. It derives targets from a task whose `success` is
`block_broken` or `item_gained`, plans surface approaches from the natural
snapshot, and checks the block journal or inventory delta respectively. Use
`target_blocks` and `drop_item` when a block and its collected item do not
share a catalog name. This is declarative objective matching, not an English
goal parser: policies are selected explicitly in the generation manifest.
See [`farm_one_log_sdg`](../examples/farm_one_log_sdg) for the complete task
and generation pair.

The implementation follows those boundaries in separate modules under
`bedrock_rl/sdg/policy/`. `controller.py` owns the generic live controller;
`lighting.py`, `mining.py`, and `recipes.py` own reusable behaviors;
`planning.py`, `spatial.py`, `world.py`, `structures.py`, and `fluids.py` own
immutable planning and geometry; `execution/` advances plans from observed
engine transitions; and `progression/` composes longer objectives. Portal
build/ignite/enter execution is an isolated terminal-operation executor rather
than an implicit policy tail.
Tasks inside the existing mine/craft/smelt family can replace one layer
without copying the survival state machine. New behavior families such as
combat, redstone, or arbitrary structures still need their own controller and
planner components.

### Grounding boundary

Scripted execution is model-grounded: movement, aiming, GUI clicks, mining,
placement, and recovery react to the same frame, observation fields, and
journal events available at rollout time. Target selection is a separate
question. Snapshot planners can inspect the full initial voxel map. For the
small log and coal objectives, the selected target is also exposed by the
model-visible bearing text and frame. The portal progression's underground
iron, diamond, water, and lava routes are not; those route choices are useful
planner and controller tests, but they are not valid behavior-cloning targets
for a frame-only policy.

Policies record this distinction at the trajectory level through
`demonstration_grounding()` and may narrow it per assistant action through
`turn_grounding()`. `TrajectorySFTWriter` requires grounded data by default. It
keeps eligible turns and skips privileged stages; a trajectory with no
eligible assistant action fails loudly. The portal policy currently exposes
only its log and coal stages because those targets have model-visible bearings;
its clairvoyant underground resource, fluid, and structure decisions never
enter default behavior-cloning loss. The exporter produces one causal row per
eligible action, drops post-action terminal frames, and retains only the newest
image by default. Set `keep_latest_images: 0` only when full image history is
both trainable and used by deployment. Assistant content stays empty unless
the manifest supplies an `assistant_content` callable; that callable receives
only the current public observation text, emitted actions, and turn number,
so it can add grounded narration without copying hidden plan provenance into
the loss.

For a declarative survival progression, use
`SurvivalProgressionPolicy`. Its `spec` is the objective graph: natural
resources, tools, counts, crafting or smelting milestones, and an optional
final portal. Built-in presets are selected explicitly; custom goals can
provide the same data inline instead:

```yaml
policy:
  type: bedrock_rl.sdg.policy.progression:SurvivalProgressionPolicy
  spec:
    stages:
      - name: logs
        blocks: [log]
        count: 2
        drop: log
        route_kinds: [surface]
        after:
          - {type: craft, item: planks, minimum: 8, grid: 2}
    portal: false
```

This changes the objective without changing the snapshot planner, live action
executor, hazard behavior, parallel world sampler, or trajectory writer.

## Choosing components

| Need | Component |
| --- | --- |
| Parallel C-engine episodes | `ProcessExecutor` |
| Thread-safe remote model/API calls | `ThreadExecutor` |
| Deterministic debugging | `SerialExecutor` |
| Training prompts | `EpisodeDatasetGenerator` + `ParquetWriter` |
| VLM demonstrations | `ScriptedEpisodeGenerator` + `TrajectorySFTWriter` |
| JSON trajectory and GIF | `TrajectoryArtifactWriter` |

`ThreadExecutor` shares one generator instance. Custom generators that keep
mutable per-world caches should declare `thread_safe = False`; the executor
then refuses them with a pointer to `ProcessExecutor` or `SerialExecutor`
instead of racing that state.

CPU processes are normally the fastest way to scout many Netherite seeds;
one CUDA rasterizer per worker wastes VRAM. A custom sampler, generator,
executor, policy, or writer is just another `package.module:object` component.
`workers: auto` uses the smallest of the available CPUs, a one-GiB-per-worker
memory budget with two GiB left for the parent and operating system, and the
number of candidate worlds available. It has no hidden 32-process ceiling on
large hosts. An explicit integer overrides the estimate. Work submission stays
bounded to two queued or active worlds per worker by default; `chunksize`
changes that prefetch multiplier. Long episodes benefit from 2–4 because quick
seed rejections cannot leave workers idle behind one slow, earlier candidate.
Once the requested number of worlds succeeds, pending processes are
terminated.

For a much larger CPU fleet, install the optional Modal client and submit the
same grouped manifest as a deterministic frameless scout:

```bash
uv sync --extra modal
uv run bedrock generate examples/farm_one_log_sdg/data.yaml \
  --modal-scout --scout-worlds 120 --scout-attempts 3000 \
  --scout-containers 64 --scout-run farm-one-log-100
```

The manifest may have any filename, but Modal scouting requires it and its
referenced task/components to live inside the checkout copied into the remote
image.

The detached coordinator commits candidates in sampler order, stops the tail
once the exact target is reached, and stores durable progress plus an accepted
case manifest in the named Modal Dict (default `bedrock-sdg-scout-v1`). The
workers run the complete live policy and verifier with rendering disabled;
they do not approximate actions or success. Rejected rows collapse into reason
counts in the manifest, while every candidate result remains durable and
accepted rows retain their selected decision seed and action digest. Relaunch
the same command with the same `--scout-run` to reuse completed candidates; a
changed manifest, referenced task, package/policy source, engine patch/pin,
named job, or override set is refused under that run ID so results from
different execution contracts cannot mix. Expected policy/seed
rejections are recorded without cancelling unrelated candidates. A worker
failure or a candidate whose worker never publishes a result is not mislabeled
as a rejected seed: the coordinator writes an incomplete manifest, leaves that
candidate uncommitted, and a relaunch retries it while reusing every completed
row. Workers write attempt-specific rows; only the coordinator publishes a
stable candidate decision, so a late timed-out worker cannot overwrite a later
relaunch. Use
`modal dict get DICT RUN:progress` while it runs and
`modal dict get DICT RUN:manifest` when complete. Write the manifest's
`accepted_cases` as JSONL, point `CaseFile` at it, set `decision_attempts: 1`,
and run ordinary generation with the authoritative renderer. Keeping a small
acceptance cushion (for example, scout 550 to materialize 500) leaves room for
machine faults without weakening final verification.

Pathological custom planners or engine calls can be bounded without letting
one early candidate hold every later result behind it:

```yaml
executor:
  type: bedrock_rl.sdg:ProcessExecutor
  workers: auto
  candidate_timeout_seconds: 120
```

The deadline covers the complete candidate world group, including cheap
qualification and every `samples_per_world` variant. A timeout discards any
staged siblings, reports `candidate_timeout`, and advances to the next
deterministic candidate. Timeout-enabled workers retire after each candidate,
so partially mutated policy or native-extension state cannot leak into the
replacement world; the pool maintains concurrency with clean worker slots.
This option uses POSIX process timers and is disabled by default. Wall-clock
cutoffs can select different worlds on substantially slower hardware, so
deterministic node, turn, and engine-tick budgets should remain the primary
limits; set this deadline comfortably above ordinary successful candidates as
an operational backstop.

The CLI reports progress to stderr as candidates commit, including
completed/committed candidates, accepted worlds, top rejection codes, elapsed
time, throughput, and a rough ETA. Ordinary updates are throttled until either
10 candidates or 10 seconds have passed. `ProcessExecutor` also reports while
an earlier ordered candidate is still running; set its `heartbeat_seconds`
(10 seconds by default) to change that operator heartbeat. `SerialExecutor`
and `ThreadExecutor` cannot report from inside one long generator call. The
Python `GenerationSpec.execute()` API remains quiet unless passed a progress
callback. Ctrl-C uses the same cleanup path.

When CUDA environment rendering is explicitly configured with
`NETHERITE_ENV_GPUS=0,1,...`, auto mode starts at most one process per listed
device and pins each worker to one device. Use CPU rendering for broad seed
scouts and short episodes. Reserve CUDA for long image-heavy episodes where
the raster saving exceeds process startup and VRAM cost.

Writers stream accepted records, and `TrajectoryArtifactWriter` retains only the
first one when it needs a GIF. One sample at a time lives in each active
worker; completed siblings are staged to executor-owned temporary storage
until the whole world is accepted. Samples from one world run sequentially in
the same worker so they reuse one snapshot. Lower `workers` when individual
image trajectories are large.

See [`farm_one_log_sdg`](../examples/farm_one_log_sdg) for a compact natural
world task with seed filtering, safe movement, parallel fresh-world
acceptance, trajectory SFT export, and an end-to-end training manifest.
