#!/usr/bin/env python
"""Summarize an overnight sweep: pass rate per task and per tier.

    uv run python evals/overnight_v1/summarize.py \
        evals/overnight_v1/results/<model slug>
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

TIERS = {"t1": "tier 1 (perception/basics)",
         "t2": "tier 2 (world interaction)",
         "t3": "tier 3 (short chains)",
         "t4": "tier 4 (long horizon)"}


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = Path(sys.argv[1])
    if not root.is_dir():
        sys.exit(f"{root} is not a directory")

    rows = []
    for traj_path in sorted(root.glob("*/seed*.json")):
        task = traj_path.parent.name
        seed = traj_path.stem.removeprefix("seed")
        try:
            doc = json.loads(traj_path.read_text())
        except json.JSONDecodeError:
            rows.append((task, seed, None, None, None))
            continue
        reward = doc.get("reward") or {}
        metrics = reward.get("metrics") or {}
        rows.append((task, seed, bool(metrics.get("success")),
                     reward.get("total"), metrics.get("turns")))

    if not rows:
        sys.exit(f"no seed*.json trajectories under {root}")

    by_task = defaultdict(list)
    for task, seed, success, total, turns in rows:
        by_task[task].append((seed, success, total, turns))

    print(f"{'task':34} {'pass':>9} {'mean reward':>12} {'mean turns':>11}")
    tier_pass = defaultdict(lambda: [0, 0])
    for task in sorted(by_task):
        episodes = by_task[task]
        scored = [e for e in episodes if e[1] is not None]
        passes = sum(1 for e in scored if e[1])
        tier = task.split("_", 1)[0]
        tier_pass[tier][0] += passes
        tier_pass[tier][1] += len(scored)
        rewards = [e[2] for e in scored if e[2] is not None]
        turns = [e[3] for e in scored if e[3] is not None]
        mean_r = f"{sum(rewards) / len(rewards):+.3f}" if rewards else "-"
        mean_t = f"{sum(turns) / len(turns):.1f}" if turns else "-"
        broken = len(episodes) - len(scored)
        note = f"  ({broken} unreadable)" if broken else ""
        print(f"{task:34} {passes}/{len(scored):>2}{'':6} "
              f"{mean_r:>12} {mean_t:>11}{note}")

    print()
    for tier in sorted(tier_pass):
        passes, total = tier_pass[tier]
        label = TIERS.get(tier, tier)
        pct = f"{100 * passes / total:.0f}%" if total else "-"
        print(f"{label:34} {passes}/{total} ({pct})")
    overall = [sum(v[0] for v in tier_pass.values()),
               sum(v[1] for v in tier_pass.values())]
    print(f"{'overall':34} {overall[0]}/{overall[1]} "
          f"({100 * overall[0] / overall[1]:.0f}%)")


if __name__ == "__main__":
    main()
