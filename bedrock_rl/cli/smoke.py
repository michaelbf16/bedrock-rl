"""Env + task + reward sanity, no GPU and no verl needed.

  NETHERITE_HOME=/path/to/netherite uv run bedrock smoke [task_yaml]

Exercises the shipped computer-use path end to end: prompt -> oracle ->
Episode -> direct state reward. The oracle is asked once per turn,
because one that answers with the whole solve and one that answers with
the next thing to do are both oracles here, and a task whose route runs
through a container screen is the second kind.

Exits non-zero on anything short of a full oracle sweep. It printed
`expert success 0/8` and exited 0 once, against an engine that was not at
the pinned SHA, and a green smoke test on the wrong binary is worse than
no smoke test. MagmaEnv warns about the engine separately.
"""
import json
import random
import sys


from bedrock_rl.env.task import Task
from bedrock_rl.adapters.netherite.computer import cu_parser, cu_user_prompt
from bedrock_rl.sdg.policy.oracle import expert_cu
from bedrock_rl.env.episode import Episode, NO_TOOL_CALL_REWARD


def play_oracle(task, episode, rng):
    """Drive the one-turn oracle until the episode ends."""
    turns = []
    while not episode.done and len(turns) < task.max_turns:
        actions = expert_cu(task, episode.env, rng)
        if not actions:
            return []
        episode.step(json.dumps(actions))
        turns.append(actions)
    return turns


def main():
    task_path = sys.argv[1] if len(sys.argv) > 1 \
        else "examples/equip_pickaxe_grpo/task.yaml"
    task = Task.load(task_path)
    rng = random.Random(0)
    failures = []

    print("=== prompt (computer-use) ===")
    # Episode.draw everywhere below, the same screened pose the data
    # paths get. An oracle run from a pose that already had the crosshair
    # on the target reports a success nobody had to aim for, and a smoke
    # test that can pass vacuously is not a smoke test.
    ep = Episode.draw(task, task.seeds[0], rng, parser=cu_parser(task))
    print(cu_user_prompt(task, ep.env))
    ep.close()

    print("\n=== expert episodes (computer-use) ===")
    succ = 0
    n = 8
    first = None
    for i in range(n):
        seed = task.seeds[i % len(task.seeds)]
        ep = Episode.draw(task, seed, rng, parser=cu_parser(task))
        yaw, pitch = ep.init_pose          # the pose ACCEPTED, not drawn
        try:
            turns = play_oracle(task, ep, rng)
            if not turns:
                raise SystemExit(f"no CU oracle for task {task.name}")
            succ += bool(ep.success)
            if first is None:
                # Keep the exact accepted episode for the deterministic
                # repeat check below.
                first = (dict(ep.spec), turns, ep.final_reward())
            print(f"seed {seed} init=({yaw:.0f},{pitch:.0f}) "
                  f"turns={len(turns)} "
                  f"acts={json.dumps([a for t in turns for a in t])} -> "
                  f"score={ep.final_reward():.3f} success={ep.success} "
                  f"ticks={ep.env.ticks}")
        finally:
            ep.close()
    print(f"expert success {succ}/{n}")
    if succ != n:
        failures.append(f"expert oracle solved {succ}/{n} episodes")

    print("\n=== direct episode reward ===")
    spec, turns, live = first
    repeated = Episode.from_spec(spec, cu_parser(task), task=task)
    try:
        for actions in turns:
            repeated.step(json.dumps(actions))
        again = float(repeated.final_reward())
    finally:
        repeated.close()
    print(f"expert   -> {live:.3f}")
    print(f"repeat   -> {again:.3f}")
    print(f"no tool  -> {NO_TOOL_CALL_REWARD:.3f}")
    if again != live:
        failures.append("the same deterministic episode and actions scored "
                        f"{live:.6f} then {again:.6f}")

    if failures:
        print("\nSMOKE FAILED")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nsmoke ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
