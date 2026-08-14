"""Pre-build the cached start-state snapshots for one task."""

from __future__ import annotations

import argparse
import os

from bedrock_rl.env.snapshots import ensure_seed
from bedrock_rl.env.task import Task


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="bedrock task snapshots",
        description="pre-build every configured seed snapshot for a task",
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--netherite-home", default=None)
    args = parser.parse_args(argv)

    task = Task.load(args.task)
    if not task.snapshots_dir:
        parser.error(f"{task.name} has no snapshots_dir")

    ready = 0
    for seed in task.seeds:
        try:
            path = ensure_seed(task, seed, args.netherite_home)
            print(f"seed {seed}: {path} ({os.path.getsize(path)} B)")
            ready += 1
        except RuntimeError as exc:
            print(f"seed {seed}: {exc}")
    print(f"{ready}/{len(task.seeds)} snapshots ready")
    return 0 if ready == len(task.seeds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
