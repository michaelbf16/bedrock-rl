# benchmark

25 Minecraft eval tasks in four difficulty tiers, run as independent
`bedrock trial` episodes against any OpenAI-compatible endpoint
(OpenRouter by default). Success is judged by the engine's event journal
and live state, never by the model's text.

| Tier | Tasks | Skills |
| --- | --- | --- |
| t1 | select_pickaxe, select_flint, look_at_sky, face_tree, walk_forward, open_inventory | reading the screen, camera, movement, one keypress |
| t2 | break_one_log, dig_dirt, place_block, place_torch, reach_water | one real interaction with the world |
| t3 | craft_planks, craft_sticks, craft_table, use_crafting_table, mine_stone, craft_torches, open_furnace, gather_three_logs | short chains, 2x2 crafting GUI |
| t4 | craft_wooden_pickaxe, stone_pickaxe_scratch, smelt_iron, mine_iron_arena, enter_nether_lit, enter_nether_build | long-horizon chains, 3x3 crafting, smelting, the Nether |

## Run it

```bash
# 1. one-time, no API cost: verify 3 workable world seeds per task
uv run python benchmark/prequalify.py

# 2. sanity-check the episode list
export OPENROUTER_API_KEY=...
uv run python benchmark/run.py \
    --model anthropic/claude-sonnet-4.5 --dry-run

# 3. the sweep (resumable: re-run the same command to continue)
uv run python benchmark/run.py \
    --model anthropic/claude-sonnet-4.5

# 4. summarize
uv run python benchmark/summarize.py \
    benchmark/results/anthropic_claude-sonnet-4.5
```

The model needs vision and tool calling. Useful knobs: `--tasks t1_...`
to subset, `--seeds-per-task`, `--temperature`, `--timeout` (wall clock
per episode), `--keep-images` (screenshots kept in context; raise for
better memory at higher cost).

Every episode writes `results/<model>/<task>/seed<N>.json` (full
trajectory with the reward verdict), a `.log` with the trial's console
output, and a line in `runlog.jsonl`. Existing `seed<N>.json` files are
skipped on re-run, so a crashed sweep continues where it stopped.

## Notes

- `seeds.json` (from prequalify.py) pins, per task, world seeds whose
  terrain satisfies the task's spawn constraints. Without it the tasks'
  unverified `seeds:` lists are used.
- The t4 arena tasks (`mine_iron_arena`, `enter_nether_*`) author their
  geometry with `initial_state.blocks` on a stone floor at y=64, so any
  seed works; the seed only changes the scenery.
- Rendering uses the deterministic stub textures unless the engine was
  built from a client jar (`bedrock setup --mc-jar ...`).
