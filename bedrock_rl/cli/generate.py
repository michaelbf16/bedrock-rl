"""Execute one import-path data-generation job.

A file may be a single job or a manifest of named jobs under shared defaults.
Jobs are selected explicitly and never chained.
"""

import argparse
import json
import sys
from pathlib import Path

from bedrock_rl.config.overrides import apply_override
from bedrock_rl.sdg.generation import GenerationSpec, load_generation_jobs


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="bedrock generate",
        description="generate records through import-path components")
    parser.add_argument("config")
    parser.add_argument("job", nargs="?",
                        help="named job in a multi-job data manifest")
    parser.add_argument("--set", action="append", default=[],
                        metavar="KEY=VALUE")
    parser.add_argument("--check", action="store_true",
                        help="resolve components without constructing them")
    args = parser.parse_args(argv)
    path = Path(args.config).resolve()
    jobs = load_generation_jobs(path)
    if args.job is not None:
        if args.job not in jobs:
            parser.error(f"{path} has no job {args.job!r}; choose "
                         + ", ".join(str(name) for name in jobs))
        selected = {args.job: jobs[args.job]}
    elif len(jobs) == 1:
        selected = jobs
    elif args.check:
        selected = jobs
    else:
        parser.error(f"{path} has multiple jobs; choose one: "
                     + ", ".join(str(name) for name in jobs))
    for document in selected.values():
        for expression in args.set:
            apply_override(document, expression)
    specs = {
        name: GenerationSpec.from_dict(document, path=str(path))
        for name, document in selected.items()
    }
    if args.check:
        many = len(specs) > 1
        for job, spec in specs.items():
            if many:
                print(f"[{job}]")
            spec.validate_components()
            for name in ("sampler", "generator", "executor", "writer"):
                print(f"{name:10} {getattr(spec, name).type}")
        return 0
    spec = next(iter(specs.values()))
    def progress(value):
        eta = value["eta_seconds"]
        eta_text = "unknown" if eta is None else f"{eta:.0f}s"
        unit = "worlds" if value["grouped"] else "records"
        rejections = ",".join(
            f"{code}:{count}" for code, count in
            value["rejection_counts"].items()) or "none"
        print(
            "progress "
            f"{unit}={value['accepted']}/{value['wanted']} "
            f"candidates={value['completed']}/{value['attempt_limit']} "
            f"committed={value['committed']} "
            f"rejected={value['rejected']} "
            f"rejections={rejections} "
            f"elapsed={value['elapsed_seconds']:.0f}s "
            f"rate={value['rate']:.2f}/s eta={eta_text}",
            file=sys.stderr, flush=True)

    result = spec.execute(progress=progress)
    job = next(iter(specs))
    if job is not None:
        result["job"] = job
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
