"""Modal CPU fleet for deterministic, frameless SDG seed scouting.

The fleet proves full scripted trajectories but retains only accepted cases
and compact rejection counters. Final image materialization stays an ordinary
``bedrock generate`` run, typically with :class:`CaseFile` as its sampler.
"""

from __future__ import annotations

import collections
import copy
import hashlib
import itertools
import json
import os
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import modal

from bedrock_rl.config.overrides import apply_override
from bedrock_rl.launch.modal import modal_cpu_dockerfile


ROOT = Path(__file__).resolve().parents[2]
REMOTE_ROOT = Path("/opt/bedrock-rl")
CONFIG = REMOTE_ROOT / os.environ.get(
    "BRL_MODAL_SDG_CONFIG", "missing-generation-config.yaml")
JOB = os.environ.get("BRL_MODAL_SDG_JOB") or None
SETS = tuple(json.loads(os.environ.get("BRL_MODAL_SDG_SETS", "[]")))
RESULTS_NAME = os.environ.get(
    "BRL_MODAL_SDG_RESULTS", "bedrock-sdg-scout-v1")
MAX_CONTAINERS = int(os.environ.get("BRL_MODAL_SDG_CONTAINERS", "1000"))
REMOTE_ENV = {
    key: value for key, value in os.environ.items()
    if key.startswith("BRL_MODAL_SDG_")
}

app = modal.App("bedrock-sdg-scout")

if modal.is_local():
    pin = (ROOT / "patches/netherite/PINNED_SHA").read_text().strip()
    source = modal_cpu_dockerfile((ROOT / "Dockerfile").read_text())
    digest = hashlib.sha256(source.encode()).hexdigest()[:12]
    dockerfile = Path(tempfile.gettempdir()) / (
        f"bedrock-rl-modal-cpu-{digest}.Dockerfile")
    dockerfile.write_text(source)
    scout_image = modal.Image.from_dockerfile(
        dockerfile, context_dir=ROOT,
        build_args={"NETHERITE_SHA": pin})
else:
    # Function metadata already points at the locally assembled image.
    scout_image = modal.Image.debian_slim()

results = modal.Dict.from_name(RESULTS_NAME, create_if_missing=True)


def _result_key(run_id, invocation_id, ordinal):
    return f"{run_id}:{invocation_id}:case:{int(ordinal):08d}"


def _attempt_key(run_id, invocation_id, attempt_id, ordinal):
    return (f"{run_id}:{invocation_id}:attempt:{attempt_id}:"
            f"{int(ordinal):08d}")


def _committed_result(row):
    """Return only worker-evaluated durable rows.

    Older releases wrote coordinator timeouts into the worker's result key.
    Those rows did not evaluate a candidate and must be retried rather than
    becoming permanent deterministic rejections.
    """
    if isinstance(row, dict) and (
            row.get("code") == "scout_result_timeout" or row.get("fatal")):
        return None
    return row


def _invocation_id(run_id, identity=None):
    """Stable namespace for one run and one execution contract."""
    value = str(run_id) if identity is None else f"{run_id}\0{identity}"
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def _configuration_identity(root=None, config=None):
    """Hash the inputs that can change one scout candidate decision."""
    root = REMOTE_ROOT if root is None else Path(root)
    config = CONFIG if config is None else Path(config)
    candidates = {config, root / "Dockerfile",
                  root / "pyproject.toml", root / "uv.lock"}
    for pattern in (
            "**/*.py", "**/*.yaml", "**/*.yml",
            "bedrock_rl/**/*.json", "bedrock_rl/**/*.jinja",
            "patches/**/*", "requirements/**/*"):
        candidates.update(root.glob(pattern))
    ignored = {".git", ".venv", "__pycache__", "outputs", "snapshots",
               "artifacts", "data", "docs", "logs", "tests", "work"}
    source = hashlib.sha256()
    files = (
        path for path in candidates
        if path.is_file() and not ignored.intersection(
            path.relative_to(root).parts)
    )
    for path in sorted(
            files, key=lambda item: str(item.relative_to(root))):
        relative = str(path.relative_to(root))
        body = path.read_bytes()
        source.update(relative.encode())
        source.update(b"\0")
        source.update(str(len(body)).encode())
        source.update(b"\0")
        source.update(body)
    payload = {
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "source_sha256": source.hexdigest(),
        "job": JOB,
        "sets": list(SETS),
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _store_progress(run_id, invocation_id, progress):
    """Write the internal key and the stable user-facing lookup alias."""
    results[f"{run_id}:{invocation_id}:progress"] = progress
    results[f"{run_id}:progress"] = progress


def _spec():
    from bedrock_rl.sdg.generation import GenerationSpec, load_generation_jobs

    jobs = load_generation_jobs(CONFIG)
    if JOB is None:
        if len(jobs) != 1:
            raise ValueError(
                f"{CONFIG} has named jobs; choose one in the launch")
        document = next(iter(jobs.values()))
    else:
        try:
            document = jobs[JOB]
        except KeyError as exc:
            raise ValueError(f"{CONFIG} has no job {JOB!r}") from exc
    for expression in SETS:
        apply_override(document, expression)
    generator = document.get("generator")
    if not isinstance(generator, dict):
        generator = {"type": generator}
        document["generator"] = generator
    generator["include_images"] = False
    generator["headless_qualification"] = False
    return GenerationSpec.from_dict(document, path=str(CONFIG))


def _components(*, with_generator):
    from bedrock_rl.env.task import Task

    spec = _spec()
    sampler = spec.sampler.create(
        base_dir=spec.base_dir, metadata=dict(spec.metadata))
    if not with_generator:
        return spec, sampler, None
    generator = spec.generator.create(
        base_dir=spec.base_dir, metadata=dict(spec.metadata))
    if not (hasattr(generator, "task_template")
            and hasattr(generator, "_action_digest")):
        raise TypeError(
            "Modal seed scouting requires ScriptedEpisodeGenerator or a "
            "compatible deterministic scripted generator")
    # Scouting is frameless. The policy reads exact live state, not pixels;
    # selecting the semantic view avoids requiring proprietary renderer assets
    # while retaining every simulator transition and public action.
    generator.task_template["view"] = "semantic"
    generator.task = Task(copy.deepcopy(generator.task_template),
                          path=str(generator.task_path))
    generator._resolved_task = None
    return spec, sampler, generator


def _case(value):
    from bedrock_rl.sdg.generation import Case

    return Case(
        index=int(value["index"]), seed=int(value["seed"]),
        decision_seed=int(value["decision_seed"]),
        values=copy.deepcopy(value.get("values") or {}),
        world_index=value.get("world_index"),
        sample_index=value.get("sample_index"),
        world_values=copy.deepcopy(value.get("world_values") or {}))


@app.function(
    image=scout_image,
    cpu=1.0,
    memory=2048,
    timeout=2400,
    max_containers=MAX_CONTAINERS,
    env=REMOTE_ENV,
)
def scout(run_id: str, invocation_id: str, attempt_id: str, ordinal: int,
          case_values: list[dict]):
    from bedrock_rl.sdg.generation import CaseRejected

    cases = tuple(_case(value) for value in case_values)
    # Workers only own attempt-specific rows. The coordinator is the sole
    # writer of stable candidate keys, so a timed-out worker cannot overwrite
    # the result selected by a later relaunch.
    key = _attempt_key(run_id, invocation_id, attempt_id, ordinal)
    started = time.perf_counter()
    try:
        _spec_value, _sampler, generator = _components(with_generator=True)
        qualify = getattr(generator, "qualify_group", None)
        if qualify is not None:
            qualify(cases)
        accepted = []
        for original in cases:
            trajectory = generator(original)
            digest, turns = generator._action_digest(trajectory)
            accepted.append({
                "case": copy.deepcopy(
                    trajectory.metadata["generation_case"]),
                "decision_selection": copy.deepcopy(
                    trajectory.metadata.get("decision_selection") or {}),
                "action_sha256": digest,
                "turns": int(turns),
            })
        row = {
            "ok": True,
            "ordinal": int(ordinal),
            "source_cases": [case.to_dict() for case in cases],
            "accepted": accepted,
            "elapsed_seconds": time.perf_counter() - started,
        }
    except CaseRejected as exc:
        row = {
            "ok": False,
            "ordinal": int(ordinal),
            "source_cases": [case.to_dict() for case in cases],
            "code": exc.code,
            "reason": str(exc)[:512],
            "elapsed_seconds": time.perf_counter() - started,
        }
    except Exception as exc:
        row = {
            "ok": False,
            "fatal": True,
            "ordinal": int(ordinal),
            "source_cases": [case.to_dict() for case in cases],
            "code": "generator_error",
            "reason": f"{type(exc).__name__}: {exc}"[:1024],
            "elapsed_seconds": time.perf_counter() - started,
        }
    results[key] = row
    return {"ordinal": ordinal, "ok": row["ok"]}


@app.function(
    image=scout_image, cpu=1.0, memory=1024, timeout=86400,
    env=REMOTE_ENV,
)
def coordinate(run_id: str, invocation_id: str, start: int,
               max_candidates: int, target: int):
    attempt_id = uuid4().hex
    identity = _configuration_identity()
    # Dict has no cross-key transaction. Namespace durable candidates by the
    # full execution identity so two concurrent first launches with the same
    # human run ID but different configs can race the alias, never the data.
    expected_invocation_id = _invocation_id(run_id, identity)
    if invocation_id != expected_invocation_id:
        raise ValueError(
            "local and remote scout inputs differ; rebuild the launch image "
            "from the current checkout")
    identity_key = f"{run_id}:identity"
    previous_identity = results.get(identity_key)
    if previous_identity is not None and previous_identity != identity:
        raise ValueError(
            f"scout run {run_id!r} already belongs to a different config; "
            "choose a new --scout-run instead of mixing candidates")
    results[identity_key] = identity
    spec, sampler, _generator = _components(with_generator=False)
    if spec.worlds is None:
        raise ValueError("Modal scouting requires grouped `worlds` generation")
    groups = itertools.islice(
        sampler.sample_worlds(
            start + max_candidates, spec.samples_per_world, spec.seed),
        start, start + max_candidates)
    ordinals = range(start, start + max_candidates)

    def missing():
        for ordinal, group in zip(ordinals, groups):
            if _committed_result(results.get(_result_key(
                    run_id, invocation_id, ordinal))) is None:
                yield ordinal, [case.to_dict() for case in group]

    pending = iter(missing())
    try:
        first = next(pending)
    except StopIteration:
        calls = None
    else:
        streams = itertools.tee(itertools.chain((first,), pending), 5)
        calls = scout.experimental_spawn_map(
            (run_id for _ in streams[0]),
            (invocation_id for _ in streams[1]),
            (attempt_id for _ in streams[2]),
            (item[0] for item in streams[3]),
            (item[1] for item in streams[4]))
    accepted = []
    rejected = 0
    rejection_counts = collections.Counter()
    fatal = None
    fatal_count = 0
    infrastructure_failures = []
    infrastructure_failure_count = 0
    committed = 0
    started = time.time()
    result_deadline = started + 2700
    for ordinal in ordinals:
        key = _result_key(run_id, invocation_id, ordinal)
        attempt_key = _attempt_key(
            run_id, invocation_id, attempt_id, ordinal)
        row = None
        while row is None:
            row = _committed_result(results.get(key))
            if row is None:
                attempt_row = results.get(attempt_key)
                if isinstance(attempt_row, dict) and attempt_row.get("fatal"):
                    fatal = fatal or attempt_row
                    fatal_count += 1
                    infrastructure_failure_count += 1
                    if len(infrastructure_failures) < 100:
                        infrastructure_failures.append(attempt_row)
                    try:
                        del results[attempt_key]
                    except KeyError:
                        pass
                    break
                if attempt_row is not None:
                    # Only evaluated, non-fatal outcomes become durable
                    # candidate decisions. Expected CaseRejected outcomes are
                    # deterministic; infrastructure failures remain retryable.
                    results[key] = attempt_row
                    row = attempt_row
                    try:
                        del results[attempt_key]
                    except KeyError:
                        pass
            if row is None:
                _store_progress(run_id, invocation_id, {
                    "schema": "bedrock.modal.seed-scout-progress.v1",
                    "run_id": run_id,
                    "invocation_id": invocation_id,
                    "attempt_id": attempt_id,
                    "committed": committed,
                    "accepted": len(accepted),
                    "rejected": rejected,
                    "waiting_ordinal": ordinal,
                    "elapsed_seconds": time.time() - started,
                })
                if time.time() >= result_deadline:
                    # No worker evaluated this candidate, so it is not a seed
                    # rejection. Leave the stable key empty, keep draining
                    # completed siblings, and make the manifest incomplete so
                    # a relaunch retries only missing/fatal candidates.
                    timeout = {
                        "code": "scout_result_timeout",
                        "ordinal": ordinal,
                        "reason": (
                            "mapped scout produced no durable result within "
                            "2700 seconds"),
                        "elapsed_seconds": time.time() - started,
                    }
                    infrastructure_failure_count += 1
                    if len(infrastructure_failures) < 100:
                        infrastructure_failures.append(timeout)
                    break
                time.sleep(5)
        if row is None:
            continue
        committed += 1
        if row["ok"]:
            if len(accepted) < target:
                accepted.append(row)
        else:
            rejected += 1
            rejection_counts[row["code"]] += 1
        if (len(accepted) == target
                and infrastructure_failure_count == 0):
            break
    # Map calls are cancellable as a group, but Modal does not permit the
    # ``terminate_containers`` option for a mapped FunctionCall handle.
    if calls is not None:
        calls.cancel()
    manifest = {
        "schema": "bedrock.modal.seed-scout.v1",
        "run_id": run_id,
        "invocation_id": invocation_id,
        "attempt_id": attempt_id,
        "config": str(CONFIG.relative_to(REMOTE_ROOT)),
        "job": JOB,
        "sets": list(SETS),
        "start": start,
        "target": target,
        "accepted_cases": [
            item["case"] for row in accepted for item in row["accepted"]
        ],
        "accepted_keys": [
            _result_key(run_id, invocation_id, row["ordinal"])
            for row in accepted
        ],
        "rejected": rejected,
        "rejection_counts": dict(rejection_counts),
        "fatal": fatal,
        "fatal_count": fatal_count,
        "infrastructure_failure": (
            infrastructure_failures[0]
            if infrastructure_failures else None),
        "infrastructure_failures": infrastructure_failures,
        "infrastructure_failure_count": infrastructure_failure_count,
        "complete": (len(accepted) == target
                     and infrastructure_failure_count == 0),
        "elapsed_seconds": time.time() - started,
    }
    results[f"{run_id}:{invocation_id}:manifest"] = manifest
    results[f"{run_id}:manifest"] = manifest
    _store_progress(run_id, invocation_id, {
        "schema": "bedrock.modal.seed-scout-progress.v1",
        "run_id": run_id,
        "invocation_id": invocation_id,
        "attempt_id": attempt_id,
        "committed": committed,
        "accepted": len(accepted),
        "rejected": rejected,
        "complete": manifest["complete"],
        "elapsed_seconds": manifest["elapsed_seconds"],
    })
    return manifest


@app.local_entrypoint()
def main(start: int = 0, max_candidates: int = 1,
         target: int = 1, run_id: str = ""):
    if start < 0 or max_candidates < 1 or target < 1:
        raise ValueError("start, max_candidates and target must be positive")
    if target > max_candidates:
        raise ValueError("target cannot exceed max_candidates")
    if not run_id:
        raise ValueError("run_id is required for durable result lookup")
    local_config = ROOT / CONFIG.relative_to(REMOTE_ROOT)
    identity = _configuration_identity(ROOT, local_config)
    invocation_id = _invocation_id(run_id, identity)
    call = coordinate.spawn(
        run_id, invocation_id, start, max_candidates, target)
    print(json.dumps({
        "schema": "bedrock.modal.seed-scout-launch.v1",
        "run_id": run_id,
        "invocation_id": invocation_id,
        "results": RESULTS_NAME,
        "progress_key": f"{run_id}:progress",
        "manifest_key": f"{run_id}:manifest",
        "coordinator_call_id": call.object_id,
    }, sort_keys=True))
