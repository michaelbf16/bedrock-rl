#!/usr/bin/env python
"""Verify, per task, which world seeds can draw a legal start. No API calls.

A task with spawn constraints (a tree within 4-12 blocks, water within 14,
...) cannot start on a seed whose terrain never satisfies them. This draws a
real engine episode for each candidate seed and keeps the first N that work,
writing seeds.json for run.py.

    uv run python benchmark/prequalify.py
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS_DIR = HERE / "tasks"

# Candidates shared by every task so trivially-constrained tasks land on the
# same worlds as constrained ones, which keeps cross-task comparisons fair.
CANDIDATE_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds-per-task", type=int, default=3)
    parser.add_argument("--tasks", nargs="*")
    args = parser.parse_args()

    from bedrock_rl.adapters.netherite.computer import cu_parser
    from bedrock_rl.env.episode import Episode
    from bedrock_rl.env.task import Task

    task_files = sorted(TASKS_DIR.glob("t*.yaml"))
    if args.tasks:
        task_files = [tf for tf in task_files if tf.stem in set(args.tasks)]

    out_path = HERE / "seeds.json"
    qualified = json.loads(out_path.read_text()) if out_path.exists() else {}
    problems = []
    for task_file in task_files:
        task = Task.load(task_file)
        if len(qualified.get(task.name, [])) >= args.seeds_per_task:
            print(f"{task.name}: already qualified {qualified[task.name]}")
            continue
        parser_ = cu_parser(task)
        good = []
        for seed in CANDIDATE_SEEDS:
            if len(good) >= args.seeds_per_task:
                break
            started = time.time()
            try:
                # Same rng derivation as the driver's run manifests, so the
                # pose qualified here is the pose the eval will replay.
                episode = Episode.draw(task, seed, random.Random(seed),
                                       parser_)
                episode.close()
                good.append(seed)
                print(f"{task.name}: seed {seed} ok "
                      f"({time.time() - started:.1f}s)")
            except Exception as exc:  # noqa: BLE001 - reject and move on
                print(f"{task.name}: seed {seed} rejected: "
                      f"{type(exc).__name__}: {exc}")
        qualified[task.name] = good
        if len(good) < args.seeds_per_task:
            problems.append(f"{task.name}: only {len(good)} of "
                            f"{args.seeds_per_task} seeds qualified")
        out_path.write_text(json.dumps(qualified, indent=2) + "\n")

    print(f"\nwrote {out_path}")
    if problems:
        print("\nNOT fully qualified:", file=sys.stderr)
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
