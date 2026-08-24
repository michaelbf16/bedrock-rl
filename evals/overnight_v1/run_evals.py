#!/usr/bin/env python
"""Overnight eval driver: every task x N seeds through `bedrock trial`.

Each episode is one `bedrock trial` subprocess against an OpenAI-compatible
endpoint (OpenRouter by default), so an engine or API crash loses one
episode, never the sweep. Episodes whose trajectory JSON already exists are
skipped, which makes the sweep resumable by re-running the same command.

    export OPENROUTER_API_KEY=...
    uv run python evals/overnight_v1/run_evals.py \
        --model anthropic/claude-sonnet-4.5

Run prequalify.py first; it verifies, per task, which world seeds can
actually draw a legal start (a tree/water/stone within the constraint) and
writes seeds.json. Without that file the task's own `seeds:` list is used
unverified.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
TASKS_DIR = HERE / "tasks"

# Provider-specific sampling extras, merged into every request for that
# model. Kimi K2.6's default reasoning is unbounded on this prompt: it hits
# any max_tokens ceiling mid-thought and returns no tool call at all, so its
# episodes are run with reasoning off (noted in reports).
MODEL_SAMPLING = {
    "moonshotai/kimi-k2.6": {"reasoning": {"enabled": False}},
}


def load_dotenv(path):
    """Minimal KEY=VALUE .env loader; existing environment wins."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        os.environ.setdefault(key, value.strip().strip("'\""))


def build_run_document(task_file, doc, seed, args, output):
    return {
        "task": {
            "name": f"{doc['name']}__seed{seed}",
            "instruction": doc["goal"],
            "environment": {
                "type": ("bedrock_rl.adapters.netherite.environment:"
                         "NetheriteEnvironment"),
                "task_yaml": str(task_file),
                "seed": int(seed),
                "random_seed": int(seed),
                # Without a frames root the engine runs frameless, t0_png is
                # None, and the model gets a TEXT-ONLY prompt: the eval would
                # silently test blind play. See NetheriteEnvironment/Episode.
                "frames_root": str(output.parent / f"frames_seed{seed}"),
            },
            "verifier": "bedrock_rl.adapters.netherite.reward:EpisodeReward",
        },
        "model": {
            "type": "eval_components:TextEnvelopeModel",
            "model": args.model,
            "base_url": args.base_url,
            "api_key_env": args.api_key_env,
            "timeout": 300,
        },
        "tools": ["bedrock_rl.adapters.netherite.tools:NetheriteComputerTool"],
        "context": [
            {"type": "bedrock_rl.core.context:KeepLastImages",
             "count": args.keep_images},
        ],
        "probes": [
            "bedrock_rl.adapters.netherite.probes:EpisodeProbe",
            "bedrock_rl.adapters.netherite.probes:ObservationProbe",
        ],
        # The episode also enforces the task's own budget; mirroring it here
        # keeps the harness from cutting a long task off at the default 8.
        "max_turns": int(doc.get("max_turns", 8)),
        "sampling": {"temperature": args.temperature,
                     "max_tokens": args.max_tokens,
                     **MODEL_SAMPLING.get(args.model, {})},
        "output": str(output),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="model id on the endpoint, e.g. "
                             "anthropic/claude-sonnet-4.5")
    parser.add_argument("--base-url",
                        default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--seeds-per-task", type=int, default=3)
    parser.add_argument("--tasks", nargs="*",
                        help="task names to include (default: every task)")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="wall-clock seconds allowed per episode")
    parser.add_argument("--temperature", type=float, default=0.0)
    # Reasoning models (Kimi K2.6, GPT-5 family) spend completion budget on
    # hidden reasoning BEFORE the tool call; at 1024 they can return an empty
    # message with no tool call at all.
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--keep-images", type=int, default=2,
                        help="screenshots kept in the model's context window")
    parser.add_argument("--results", default=None,
                        help="results root (default results/<model slug>)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the episodes without running any")
    args = parser.parse_args()

    if not os.environ.get(args.api_key_env):
        load_dotenv(HERE.parents[1] / ".env")
    if not args.dry_run and not os.environ.get(args.api_key_env):
        sys.exit(f"{args.api_key_env} is not set; export it or put it in "
                 "the repo-root .env before an overnight run.")

    seeds_path = HERE / "seeds.json"
    qualified = (json.loads(seeds_path.read_text())
                 if seeds_path.exists() else {})
    if not qualified:
        print("warning: no seeds.json; using each task's unverified "
              "`seeds:` list. Run prequalify.py first.", file=sys.stderr)

    task_files = sorted(TASKS_DIR.glob("t*.yaml"))
    if args.tasks:
        wanted = set(args.tasks)
        task_files = [tf for tf in task_files if tf.stem in wanted]
        missing = wanted - {tf.stem for tf in task_files}
        if missing:
            sys.exit(f"unknown tasks: {', '.join(sorted(missing))}")
    if not task_files:
        sys.exit(f"no task YAMLs found under {TASKS_DIR}")

    episodes = []
    for task_file in task_files:
        doc = yaml.safe_load(task_file.read_text())
        seeds = qualified.get(doc["name"], doc.get("seeds") or [0])
        for seed in seeds[: args.seeds_per_task]:
            episodes.append((task_file, doc, seed))

    slug = args.model.replace("/", "_").replace(":", "_")
    results = Path(args.results) if args.results else HERE / "results" / slug
    runlog = results / "runlog.jsonl"
    print(f"{len(episodes)} episodes across {len(task_files)} tasks "
          f"-> {results}")
    if args.dry_run:
        for task_file, doc, seed in episodes:
            print(f"  {doc['name']} seed {seed}")
        return

    done = failed = skipped = 0
    for index, (task_file, doc, seed) in enumerate(episodes, 1):
        out = results / doc["name"] / f"seed{seed}.json"
        if out.exists():
            skipped += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        run_doc = build_run_document(task_file, doc, seed, args, out)
        run_yaml = out.parent / f"run_seed{seed}.yaml"
        run_yaml.write_text(yaml.safe_dump(run_doc, sort_keys=False))
        log_path = out.parent / f"seed{seed}.log"
        started = time.time()
        print(f"[{index}/{len(episodes)}] {doc['name']} seed {seed} ...",
              flush=True)
        record = {"task": doc["name"], "seed": seed,
                  "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
        env = dict(os.environ)
        env["PYTHONPATH"] = (str(HERE) + os.pathsep + env["PYTHONPATH"]
                             if env.get("PYTHONPATH") else str(HERE))
        try:
            with log_path.open("w") as log:
                proc = subprocess.run(
                    ["uv", "run", "bedrock", "trial", str(run_yaml)],
                    stdout=log, stderr=subprocess.STDOUT,
                    timeout=args.timeout, cwd=HERE.parents[1], env=env)
            record["returncode"] = proc.returncode
            record["ok"] = proc.returncode == 0 and out.exists()
        except subprocess.TimeoutExpired:
            record["returncode"] = None
            record["ok"] = False
            record["error"] = f"timeout after {args.timeout}s"
        record["seconds"] = round(time.time() - started, 1)
        if record["ok"]:
            done += 1
        else:
            failed += 1
            print(f"    FAILED ({record.get('error', 'see ' + str(log_path))})",
                  flush=True)
        runlog.parent.mkdir(parents=True, exist_ok=True)
        with runlog.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    print(f"finished: {done} ran, {failed} failed, {skipped} already done")
    print(f"summarize with: uv run python {HERE / 'summarize.py'} {results}")


if __name__ == "__main__":
    main()
