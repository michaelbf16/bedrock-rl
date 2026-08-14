"""Paired held-out evaluation summaries and Matplotlib curves.

``bedrock eval trials --records-out`` writes one JSON list per evaluated
checkpoint.  This module turns several of those lists into one auditable
comparison.  It deliberately refuses partial or unpaired inputs: a curve is
only meaningful when every checkpoint was scored on the same trials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


SCHEMA = "bedrock.eval-curve.v1"
Z_95 = 1.959963984540054
PAIR_FIELDS = ("seed", "row_index", "sample_index", "init_yaw", "init_pitch")


class EvaluationError(ValueError):
    """An evaluation input cannot support the comparison it claims."""


@dataclass(frozen=True)
class PointSpec:
    """One labeled checkpoint evaluation on the curve."""

    label: str
    step: float
    records: str | Path


def wilson_interval(successes: float, trials: float,
                    z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a Bernoulli success rate."""
    successes, trials = float(successes), float(trials)
    if trials < 1:
        raise EvaluationError("a success interval needs at least one trial")
    if not 0 <= successes <= trials:
        raise EvaluationError(
            f"successes must be between 0 and {trials}, got {successes}")
    p = successes / trials
    z2 = z * z
    scale = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / scale
    radius = z * math.sqrt(
        p * (1.0 - p) / trials + z2 / (4.0 * trials * trials)) / scale
    low = 0.0 if successes == 0 else max(0.0, center - radius)
    high = 1.0 if successes == trials else min(1.0, center + radius)
    return low, high


def seed_clustered_interval(records: list[dict[str, Any]],
                            z: float = Z_95) -> tuple[float, float]:
    """Conservative interval over independent, possibly uneven seed groups."""
    groups = {}
    for record in records:
        groups.setdefault(record["seed"], []).append(float(record["success"]))
    means = [fmean(values) for values in groups.values()]
    rate = fmean(float(record["success"]) for record in records)
    if len(means) == 1:
        return 0.0, 1.0
    total = sum(len(values) for values in groups.values())
    weights = [len(values) / total for values in groups.values()]
    # Cluster-robust variance for the row-weighted rate. Using an unweighted
    # stdev of per-seed means around this weighted center mixes two different
    # estimands when seeds have unequal replicate counts.
    variance = (len(means) / (len(means) - 1)
                * sum(weight * weight * (mean - rate) ** 2
                      for weight, mean in zip(weights, means)))
    radius = z * math.sqrt(variance)
    normal = (max(0.0, rate - radius), min(1.0, rate + radius))
    effective_clusters = 1.0 / sum(weight * weight for weight in weights)
    wilson = wilson_interval(
        rate * effective_clusters, effective_clusters, z)
    return min(normal[0], wilson[0]), max(normal[1], wilson[1])


def _emitted_tool_call(record: dict[str, Any], identity: str) -> bool:
    """Read the any-turn tool-attempt flag, with legacy compatibility."""
    if "emitted_tool_call" not in record:
        return record.get("actions") is not None
    value = record["emitted_tool_call"]
    if type(value) is not bool:
        raise EvaluationError(
            f"trial {identity} has non-boolean emitted_tool_call: {value!r}")
    return value


def _number(value: Any, *, field: str, trial: str,
            nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(
            f"trial {trial} has non-numeric {field}: {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise EvaluationError(f"trial {trial} has non-finite {field}")
    if nonnegative and number < 0:
        raise EvaluationError(f"trial {trial} has negative {field}: {number}")
    return number


def _load(path: str | Path) -> tuple[list[dict[str, Any]], str]:
    path = Path(path)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise EvaluationError(f"cannot read evaluation records {path}: {exc}") \
            from exc
    try:
        records = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"evaluation records {path} are not JSON: {exc}") \
            from exc
    if not isinstance(records, list) or not records:
        raise EvaluationError(
            f"evaluation records {path} must be a non-empty JSON list")
    if not all(isinstance(record, dict) for record in records):
        raise EvaluationError(
            f"evaluation records {path} must contain JSON objects")
    return records, hashlib.sha256(payload).hexdigest()


def _identity_mode(records: list[dict[str, Any]]) -> str:
    explicit = ["trial_id" in record for record in records]
    if any(explicit) and not all(explicit):
        raise EvaluationError(
            "trial_id is present on only part of an evaluation; every trial "
            "must use explicit identities or none may")
    return "trial_id" if all(explicit) else "position_and_seed"


def _identity(record: dict[str, Any], index: int, mode: str) -> str:
    if mode == "trial_id":
        value = record["trial_id"]
        if isinstance(value, (dict, list)) or value is None:
            raise EvaluationError(
                f"trial {index} has an invalid trial_id: {value!r}")
        return f"trial_id:{value}"
    if "seed" not in record:
        raise EvaluationError(f"trial {index} has no seed")
    # run_trials preserves row/sample order. Until its record schema grows a
    # trial_id, the position plus world seed is the strongest identity the
    # artifact contains. Including both catches reordered or changed rows.
    return f"position:{index}:seed:{record['seed']}"


def _validated(path: str | Path) -> tuple[dict[str, dict[str, Any]], dict]:
    records, digest = _load(path)
    mode = _identity_mode(records)
    indexed = {}
    for index, record in enumerate(records):
        identity = _identity(record, index, mode)
        if identity in indexed:
            raise EvaluationError(
                f"evaluation {path} repeats trial identity {identity!r}")
        broken = record.get("broken")
        if broken:
            raise EvaluationError(
                f"evaluation {path} is partial: trial {identity} did not "
                f"run ({broken})")
        if type(record.get("success")) is not bool:
            raise EvaluationError(
                f"trial {identity} has non-boolean success: "
                f"{record.get('success')!r}")
        if type(record.get("opened")) is not bool:
            raise EvaluationError(
                f"trial {identity} has non-boolean opened: "
                f"{record.get('opened')!r}")
        if "seed" not in record:
            raise EvaluationError(f"trial {identity} has no seed")
        if "actions" not in record:
            raise EvaluationError(
                f"trial {identity} has no actions field, so tool-call rate "
                "cannot be measured")
        _emitted_tool_call(record, identity)
        protocol = record.get("protocol")
        if not isinstance(protocol, dict) or not protocol:
            raise EvaluationError(
                f"trial {identity} has no evaluation protocol; sampling, "
                "hint, and image-window provenance cannot be paired")
        _number(record.get("reward"), field="reward", trial=identity)
        _number(record.get("turns"), field="turns", trial=identity,
                nonnegative=True)
        indexed[identity] = record
    solved = sum(record["success"] for record in records)
    low, high = seed_clustered_interval(records)
    independent = len({record["seed"] for record in records})
    metrics = {
        "trials": len(records),
        "successes": solved,
        "success_rate": solved / len(records),
        "success_ci95": [low, high],
        "success_ci95_method": "seed-clustered",
        "independent_seed_clusters": independent,
        "reward_mean": fmean(float(record["reward"]) for record in records),
        "turns_mean": fmean(float(record["turns"]) for record in records),
        "tool_call_rate": (
            sum(_emitted_tool_call(
                record, _identity(record, index, mode))
                for index, record in enumerate(records))
            / len(records)),
        "workstation_open_rate": (
            sum(record["opened"] for record in records) / len(records)),
    }
    source = {"path": str(Path(path)), "sha256": digest,
              "identity_mode": mode}
    return indexed, {"metrics": metrics, "source": source}


def _paired_delta(base: dict[str, dict[str, Any]],
                  point: dict[str, dict[str, Any]]) -> dict[str, Any]:
    identities = tuple(base)
    transition = {"failure_to_success": 0, "success_to_failure": 0,
                  "both_success": 0, "both_failure": 0}
    for identity in identities:
        before = base[identity]["success"]
        after = point[identity]["success"]
        if before and after:
            transition["both_success"] += 1
        elif before:
            transition["success_to_failure"] += 1
        elif after:
            transition["failure_to_success"] += 1
        else:
            transition["both_failure"] += 1

    def delta(field, transform=float):
        return fmean(transform(point[key][field])
                     - transform(base[key][field]) for key in identities)

    return {
        "success_rate": delta("success", float),
        "reward_mean": delta("reward"),
        "turns_mean": delta("turns"),
        "tool_call_rate": fmean(
            float(_emitted_tool_call(point[key], key))
            - float(_emitted_tool_call(base[key], key))
            for key in identities),
        "workstation_open_rate": delta("opened", float),
        "transitions": transition,
    }


def analyze(points: Iterable[PointSpec], base: str | None = None) -> dict:
    """Validate and summarize a paired checkpoint evaluation curve."""
    points = list(points)
    if len(points) < 2:
        raise EvaluationError("an evaluation curve needs at least two points")
    labels = [point.label for point in points]
    if any(not label.strip() for label in labels):
        raise EvaluationError("evaluation point labels cannot be empty")
    if len(set(labels)) != len(labels):
        raise EvaluationError("evaluation point labels must be unique")
    if any(not math.isfinite(float(point.step)) for point in points):
        raise EvaluationError("evaluation steps must be finite numbers")
    if len({float(point.step) for point in points}) != len(points):
        raise EvaluationError("evaluation steps must be unique")
    base = base or labels[0]
    if base not in labels:
        raise EvaluationError(
            f"base label {base!r} is not one of {', '.join(labels)}")

    loaded = {}
    summaries = {}
    for point in points:
        records, summary = _validated(point.records)
        loaded[point.label] = records
        summaries[point.label] = summary

    base_records = loaded[base]
    base_ids = set(base_records)
    base_mode = summaries[base]["source"]["identity_mode"]
    for point in points:
        label = point.label
        mode = summaries[label]["source"]["identity_mode"]
        if mode != base_mode:
            raise EvaluationError(
                f"{label!r} uses {mode} identities but base {base!r} uses "
                f"{base_mode}")
        ids = set(loaded[label])
        if ids != base_ids:
            missing = sorted(base_ids - ids)
            extra = sorted(ids - base_ids)
            detail = []
            if missing:
                detail.append(f"missing {missing[0]!r}")
            if extra:
                detail.append(f"extra {extra[0]!r}")
            raise EvaluationError(
                f"{label!r} is not paired with base {base!r}: "
                + ", ".join(detail))
        for identity in base_ids:
            before, after = base_records[identity], loaded[label][identity]
            if before["protocol"] != after["protocol"]:
                raise EvaluationError(
                    f"{label!r} is not paired with base {base!r}: trial "
                    f"{identity!r} changed evaluation protocol")
            for field in PAIR_FIELDS:
                # New run_trials artifacts carry all five fields. Older
                # artifacts may only have seed; a field cannot silently
                # appear, disappear, or change across checkpoints.
                if before.get(field) != after.get(field):
                    raise EvaluationError(
                        f"{label!r} is not paired with base {base!r}: trial "
                        f"{identity!r} changed {field} from "
                        f"{before.get(field)!r} to {after.get(field)!r}")

    ordered = []
    for point in sorted(points, key=lambda item: float(item.step)):
        summary = summaries[point.label]
        step = float(point.step)
        ordered.append({
            "label": point.label,
            "step": int(step) if step.is_integer() else step,
            **summary,
            "delta_vs_base": _paired_delta(
                base_records, loaded[point.label]),
        })
    return {
        "schema": SCHEMA,
        "base": base,
        "trials_per_point": len(base_records),
        "identity_mode": base_mode,
        "points": ordered,
    }


def _atomic_text(path: str | Path, content: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_summary(summary: dict, path: str | Path):
    _atomic_text(path, json.dumps(summary, indent=2, sort_keys=True,
                                  allow_nan=False) + "\n")
    return str(path)


def svg(summary: dict, title: str = "Held-out task success",
        success_max: float = 1.0) -> str:
    """Render the simple dark Matplotlib held-out curve."""
    del title  # The public chart deliberately has no header.
    try:
        from bedrock_rl.eval.plotting import heldout_svg
        return heldout_svg(summary, success_max)
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            raise EvaluationError(
                "plotting needs the optional dependency: "
                "uv sync --extra plot") from exc
        raise


def write_svg(summary: dict, path: str | Path,
              title: str = "Held-out task success",
              success_max: float = 1.0):
    _atomic_text(path, svg(summary, title, success_max))
    return str(path)


def load_training_scores(path: str | Path,
                         field: str = "critic/score/mean") -> dict:
    """Read the append-only trainer metrics used behind a combined curve."""
    path = Path(path)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise EvaluationError(f"cannot read training metrics {path}: {exc}") \
            from exc
    points = []
    for number, raw in enumerate(payload.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EvaluationError(
                f"training metrics {path}:{number} are not JSON: {exc}") \
                from exc
        if not isinstance(row, dict):
            raise EvaluationError(
                f"training metrics {path}:{number} must be an object")
        step = _number(row.get("step", row.get("training/global_step")),
                       field="step", trial=f"line {number}",
                       nonnegative=True)
        score = _number(row.get(field), field=field, trial=f"line {number}")
        points.append({"step": step, "score": score})
    if not points:
        raise EvaluationError(f"training metrics {path} contain no points")
    steps = [point["step"] for point in points]
    if steps != sorted(steps) or len(set(steps)) != len(steps):
        raise EvaluationError("training metric steps must be unique and ordered")
    return {
        "field": field,
        "points": points,
        "source": {"path": str(path),
                   "sha256": hashlib.sha256(payload).hexdigest()},
    }


def combined_svg(training: dict, heldout: Iterable[tuple[str, dict]],
                 title: str = "Training reward and held-out success",
                 success_max: float = 1.0) -> str:
    """Render the optional Matplotlib learning curve as SVG."""
    try:
        from bedrock_rl.eval.plotting import combined_svg as render
        return render(training, heldout, title, success_max)
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            raise EvaluationError(
                "plotting needs the optional dependency: "
                "uv sync --extra plot") from exc
        raise


def write_combined_svg(training: dict, heldout: Iterable[tuple[str, dict]],
                       path: str | Path, title: str,
                       success_max: float = 1.0):
    _atomic_text(path, combined_svg(training, heldout, title, success_max))
    return str(path)


def _step(value: str) -> float:
    try:
        step = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("step must be numeric") from exc
    if not math.isfinite(step):
        raise argparse.ArgumentTypeError("step must be finite")
    return step


def _success_max(value: str) -> float:
    try:
        percent = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "success maximum must be a percentage") from exc
    if not math.isfinite(percent) or percent <= 0 or percent > 100:
        raise argparse.ArgumentTypeError(
            "success maximum must be in (0, 100]")
    return percent / 100


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="build an honest paired held-out evaluation curve")
    parser.add_argument(
        "--point", action="append", nargs=3, required=True,
        metavar=("LABEL", "STEP", "RECORDS"),
        help="labeled bedrock eval trials --records-out file; repeat for "
             "base and checkpoints")
    parser.add_argument("--base", default=None,
                        help="baseline label; defaults to the first point")
    parser.add_argument("--summary", required=True,
                        help="output summary JSON")
    parser.add_argument("--svg", required=True, dest="svg_path",
                        help="output held-out success SVG")
    parser.add_argument("--title", default="Held-out task success")
    parser.add_argument(
        "--success-max", type=_success_max, default=1.0,
        metavar="PERCENT",
        help="top of the held-out success axis in percent (default: 100)")
    parser.add_argument(
        "--training-metrics", default=None,
        help="optional trainer JSONL; when set, plot every training score "
             "with held-out success in one SVG")
    parser.add_argument(
        "--heldout-summary", action="append", nargs=2, default=[],
        metavar=("LABEL", "SUMMARY"),
        help="additional evaluation summary to mark on a combined curve")
    args = parser.parse_args(argv)
    try:
        specs = [PointSpec(label, _step(step), records)
                 for label, step, records in args.point]
        result = analyze(specs, args.base)
        write_summary(result, args.summary)
        if args.training_metrics:
            training = load_training_scores(args.training_metrics)
            heldout = [("held-out eval", result)]
            for label, path in args.heldout_summary:
                try:
                    extra = json.loads(Path(path).read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    raise EvaluationError(
                        f"cannot read held-out summary {path}: {exc}") from exc
                if extra.get("schema") != SCHEMA or not extra.get("points"):
                    raise EvaluationError(
                        f"held-out summary {path} is not a {SCHEMA} summary")
                heldout.append((label, extra))
            observed = max(
                float(point["metrics"]["success_rate"])
                for _label, summary in heldout
                for point in summary["points"])
            if observed > args.success_max:
                raise EvaluationError(
                    f"success axis maximum {args.success_max * 100:g}% "
                    f"would clip observed {observed * 100:.1f}%")
            write_combined_svg(training, heldout, args.svg_path, args.title,
                               args.success_max)
        else:
            observed = max(float(point["metrics"]["success_rate"])
                           for point in result["points"])
            if observed > args.success_max:
                raise EvaluationError(
                    f"success axis maximum {args.success_max * 100:g}% "
                    f"would clip observed {observed * 100:.1f}%")
            write_svg(result, args.svg_path, args.title, args.success_max)
    except (argparse.ArgumentTypeError, EvaluationError) as exc:
        parser.error(str(exc))
    print(f"paired held-out trials: {result['trials_per_point']}")
    print("label                 step   success (95% CI)       reward  "
          "turns  tool calls  delta")
    for point in result["points"]:
        metrics = point["metrics"]
        low, high = metrics["success_ci95"]
        delta = point["delta_vs_base"]["success_rate"]
        print(f"{point['label']:<20} {str(point['step']):>6}  "
              f"{metrics['success_rate'] * 100:6.1f}% "
              f"[{low * 100:5.1f}, {high * 100:5.1f}]  "
              f"{metrics['reward_mean']:+7.3f}  "
              f"{metrics['turns_mean']:5.2f}  "
              f"{metrics['tool_call_rate'] * 100:6.1f}%  "
              f"{delta * 100:+6.1f} pp")
    print(f"summary {args.summary}")
    print(f"curve   {args.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
