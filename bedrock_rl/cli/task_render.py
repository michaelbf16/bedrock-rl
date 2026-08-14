"""Render the initial model view and optional scripted-policy replay."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from bedrock_rl.adapters.netherite.computer import cu_parser, cu_user_prompt
from bedrock_rl.sdg.policy.oracle import expert_cu
from bedrock_rl.env.episode import Episode
from bedrock_rl.env.task import Task


def play_oracle(task, out):
    rng = random.Random(0)
    seed = task.seeds[0]
    yaw, pitch = task.sample_init(rng)
    episode = Episode(
        task,
        seed,
        yaw,
        pitch,
        frames_root=str(out / "frames_tmp"),
        parser=cu_parser(task),
    )
    try:
        turn = 0
        while not episode.done:
            actions = expert_cu(task, episode.env, rng)
            if not actions:
                message = (
                    "no computer-use oracle for this task"
                    if turn == 0
                    else "the oracle could not continue from here"
                )
                print(message)
                break
            observation, frame, reward, done = episode.step(
                json.dumps(actions)
            )
            if frame is None:
                raise RuntimeError(
                    f"episode returned no model-visible frame at turn {turn}"
                )
            frame.save(out / f"turn{turn}.png")
            (out / f"turn{turn}.txt").write_text(
                json.dumps(actions) + "\n\n" + observation
            )
            print(
                f"  turn {turn}: {json.dumps(actions)} -> "
                f"reward {reward:+.3f}{' DONE' if done else ''}"
            )
            turn += 1
        print(
            f"oracle success={episode.success} "
            f"final={episode.final_reward():.3f}"
        )
    finally:
        episode.close()


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="bedrock task render",
        description="render the exact initial frame and prompt shown to a model",
    )
    parser.add_argument("task")
    parser.add_argument("out")
    parser.add_argument("--play-oracle", action="store_true")
    args = parser.parse_args(argv)

    task = Task.load(args.task)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    episode = Episode(
        task,
        task.seeds[0],
        *task.sample_init(rng),
        parser=cu_parser(task),
        frames_root=os.fspath(out / "frames_tmp"),
    )
    try:
        image = episode.frame()
        if image is None:
            raise RuntimeError("episode returned no initial model-visible frame")
        image.save(out / "sample.png", format="PNG")
        (out / "sample_prompt.txt").write_text(
            cu_user_prompt(task, episode.env)
        )
    finally:
        episode.close()
    print(f"wrote {out / 'sample.png'} and {out / 'sample_prompt.txt'}")
    if args.play_oracle:
        play_oracle(task, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
