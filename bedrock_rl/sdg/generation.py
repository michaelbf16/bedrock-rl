"""Small import-path contracts for reproducible data generation."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import pickle
import random
import signal
import shutil
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from itertools import chain, islice
from multiprocessing import get_context
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Mapping
from uuid import uuid4

from bedrock_rl.core.components import ComponentSpec
from bedrock_rl.core.state import json_value


class CaseRejected(Exception):
    """An expected unusable generation attempt with a useful reason."""

    def __init__(self, reason, *, code="case_rejected", details=None):
        super().__init__(str(reason))
        code = str(code).strip()
        if not code or not code.replace("_", "").isalnum():
            raise ValueError("rejection code must contain letters, numbers, "
                             "or underscores")
        self.code = code
        if details is not None and not isinstance(details, Mapping):
            raise TypeError("rejection details must be a mapping")
        self.details = json_value(
            dict(details or {}), what="rejection details")


@dataclass(frozen=True)
class Case:
    """One deterministic generation attempt.

    ``seed`` is the environment/world seed. ``decision_seed`` is separate,
    so changing agent variance never silently changes which world is used.
    """

    index: int
    seed: int
    decision_seed: int
    values: dict[str, Any] = field(default_factory=dict)
    world_index: int | None = None
    sample_index: int | None = None
    world_values: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        value = {"index": self.index, "seed": self.seed,
                 "decision_seed": self.decision_seed,
                 "values": json_value(self.values, what="case values")}
        if self.world_values:
            value["world_values"] = json_value(
                self.world_values, what="case world values")
        if self.world_index is not None:
            value["world_index"] = int(self.world_index)
            value["sample_index"] = int(self.sample_index or 0)
        return value


def _derived_decision_seed(seed, world_seed, sample_index, used):
    """A stable policy seed independent of world-stream consumption."""
    nonce = 0
    while True:
        rng = random.Random(
            f"netherite-decision-v1:{int(seed)}:{int(world_seed)}:"
            f"{int(sample_index)}:{nonce}")
        value = rng.randrange(0, 2**63 - 1)
        if value not in used:
            return value
        nonce += 1


class RandomCases:
    """Unique world seeds plus independently controlled random variables.

    ``world_variables`` are drawn once from a seed-derived world stream and
    shared by all samples in that world. ``variables`` are drawn separately
    from each case's decision seed.

    Variable declarations are deliberately data, not a registry::

        movement_blocks: {uniform: [1.5, 3.0]}
        branch_length: {integer: [6, 14]}
        route: {choice: [left, right], weights: [1, 2]}
        explore: {boolean: 0.15}
        fixed: {constant: value}
    """

    def __init__(self, variables=None, world_variables=None, world_seed_min=0,
                 world_seed_max=2**31 - 1, **_):
        self.variables = dict(variables or {})
        self.world_variables = dict(world_variables or {})
        overlap = sorted(set(self.variables) & set(self.world_variables))
        if overlap:
            raise ValueError(
                "variables and world_variables overlap: "
                + ", ".join(overlap))
        self.world_seed_min = int(world_seed_min)
        self.world_seed_max = int(world_seed_max)
        if self.world_seed_max <= self.world_seed_min:
            raise ValueError("world_seed_max must be greater than "
                             "world_seed_min")

    @staticmethod
    def _value(rng, name, declaration):
        if not isinstance(declaration, Mapping) or len(declaration) != 1:
            raise ValueError(
                f"variation {name!r} must have exactly one distribution")
        kind, value = next(iter(declaration.items()))
        if kind == "constant":
            return copy.deepcopy(value)
        if kind == "uniform":
            lo, hi = value
            return rng.uniform(float(lo), float(hi))
        if kind == "integer":
            lo, hi = value
            return rng.randint(int(lo), int(hi))
        if kind == "boolean":
            probability = float(value)
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"variation {name!r} boolean probability "
                                 "must be in [0, 1]")
            return rng.random() < probability
        if kind == "choice":
            choices = value
            weights = None
            if isinstance(value, Mapping):
                choices = value.get("values")
                weights = value.get("weights")
            if not choices:
                raise ValueError(f"variation {name!r} has no choices")
            return copy.deepcopy(
                rng.choices(list(choices), weights=weights, k=1)[0])
        raise ValueError(f"variation {name!r} uses unknown distribution "
                         f"{kind!r}")

    def sample(self, count, seed):
        count = int(count)
        capacity = self.world_seed_max - self.world_seed_min
        if count > capacity:
            raise ValueError(f"cannot draw {count} unique world seeds from "
                             f"a range containing {capacity}")
        rng = random.Random(int(seed))
        worlds = rng.sample(range(self.world_seed_min,
                                  self.world_seed_max), count)
        for index, world_seed in enumerate(worlds):
            world_rng = random.Random(
                f"netherite-world-variables-v1:{int(seed)}:{world_seed}")
            world_values = {
                name: self._value(world_rng, name, declaration)
                for name, declaration in self.world_variables.items()
            }
            decision_seed = rng.randrange(0, 2**63 - 1)
            variation_rng = random.Random(decision_seed)
            values = {**copy.deepcopy(world_values), **{
                name: self._value(variation_rng, name, declaration)
                for name, declaration in self.variables.items()
            }}
            yield Case(index, world_seed, decision_seed, values,
                       world_values=copy.deepcopy(world_values))

    def sample_worlds(self, count, samples_per_world, seed):
        """Distinct candidate worlds, each with independent policy samples.

        A group is the scheduling unit: one process receives every decision
        variant for a world, so an expensive start-state snapshot is built at
        most once there. The outer generator remains lazy even when
        ``max_attempts`` is very large.
        """
        count = int(count)
        samples_per_world = int(samples_per_world)
        capacity = self.world_seed_max - self.world_seed_min
        if count > capacity:
            raise ValueError(f"cannot draw {count} unique world seeds from "
                             f"a range containing {capacity}")
        if samples_per_world < 1:
            raise ValueError("samples_per_world must be at least one")
        # World selection and policy variance are separate random streams.
        # Changing samples_per_world must not silently change the worlds a
        # manifest selects, which matters when users increase augmentation
        # after inspecting a fixed set of natural worlds.
        world_rng = random.Random(int(seed))
        seen = set()
        for world_index in range(count):
            while True:
                world_seed = world_rng.randrange(
                    self.world_seed_min, self.world_seed_max)
                if world_seed not in seen:
                    seen.add(world_seed)
                    break
            group = []
            group_decisions = set()
            world_variation_rng = random.Random(
                f"netherite-world-variables-v1:{int(seed)}:{world_seed}")
            world_values = {
                name: self._value(
                    world_variation_rng, name, declaration)
                for name, declaration in self.world_variables.items()
            }
            for sample_index in range(samples_per_world):
                decision_seed = _derived_decision_seed(
                    seed, world_seed, sample_index, group_decisions)
                group_decisions.add(decision_seed)
                variation_rng = random.Random(decision_seed)
                values = {**copy.deepcopy(world_values), **{
                    name: self._value(variation_rng, name, declaration)
                    for name, declaration in self.variables.items()
                }}
                group.append(Case(
                    world_index * samples_per_world + sample_index,
                    world_seed, decision_seed, values,
                    world_values=copy.deepcopy(world_values),
                    world_index=world_index, sample_index=sample_index))
            yield tuple(group)


class SeedCases:
    """Cycle an explicit, reproducibly shuffled set of world seeds.

    A task defines Minecraft behavior; a data manifest defines which worlds
    belong to train, development, and test.  Keeping split membership here
    avoids cloning an otherwise identical task YAML just to change ``seeds``.
    """

    def __init__(self, seeds, variables=None, world_variables=None,
                 decision_seeds=None, **_):
        self.seeds = [int(seed) for seed in seeds]
        if not self.seeds:
            raise ValueError("SeedCases needs at least one world seed")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("SeedCases world seeds must be unique")
        self.variables = dict(variables or {})
        self.world_variables = dict(world_variables or {})
        overlap = sorted(set(self.variables) & set(self.world_variables))
        if overlap:
            raise ValueError(
                "variables and world_variables overlap: "
                + ", ".join(overlap))
        self.decision_seeds = ([int(value) for value in decision_seeds]
                               if decision_seeds is not None else None)
        if self.decision_seeds is not None and not self.decision_seeds:
            raise ValueError("SeedCases decision_seeds cannot be empty")

    def sample(self, count, seed):
        rng = random.Random(int(seed))
        worlds = list(self.seeds)
        rng.shuffle(worlds)
        for index in range(int(count)):
            world_seed = worlds[index % len(worlds)]
            world_rng = random.Random(
                f"netherite-world-variables-v1:{int(seed)}:{world_seed}")
            world_values = {
                name: RandomCases._value(world_rng, name, declaration)
                for name, declaration in self.world_variables.items()
            }
            decision_seed = (self.decision_seeds[
                index % len(self.decision_seeds)]
                if self.decision_seeds is not None else
                rng.randrange(0, 2**63 - 1))
            decision_rng = random.Random(decision_seed)
            values = {**copy.deepcopy(world_values), **{
                name: RandomCases._value(decision_rng, name, declaration)
                for name, declaration in self.variables.items()
            }}
            yield Case(index, world_seed, decision_seed, values,
                       world_values=copy.deepcopy(world_values))

    def sample_worlds(self, count, samples_per_world, seed):
        """Use each explicit world once and vary only policy decisions."""
        count = int(count)
        samples_per_world = int(samples_per_world)
        if count > len(self.seeds):
            raise ValueError(
                f"cannot draw {count} distinct worlds from the "
                f"{len(self.seeds)} explicit SeedCases seeds")
        if samples_per_world < 1:
            raise ValueError("samples_per_world must be at least one")
        selected_decisions = (None if self.decision_seeds is None else [
            self.decision_seeds[index % len(self.decision_seeds)]
            for index in range(samples_per_world)])
        if (selected_decisions is not None
                and len(set(selected_decisions)) < samples_per_world):
            raise ValueError(
                "SeedCases needs at least samples_per_world distinct "
                "decision_seeds for independent variants")
        rng = random.Random(int(seed))
        worlds = list(self.seeds)
        rng.shuffle(worlds)
        for world_index, world_seed in enumerate(worlds[:count]):
            group = []
            group_decisions = set()
            world_variation_rng = random.Random(
                f"netherite-world-variables-v1:{int(seed)}:{world_seed}")
            world_values = {
                name: RandomCases._value(
                    world_variation_rng, name, declaration)
                for name, declaration in self.world_variables.items()
            }
            for sample_index in range(samples_per_world):
                offset = world_index * samples_per_world + sample_index
                if self.decision_seeds is not None:
                    decision_seed = self.decision_seeds[
                        sample_index % len(self.decision_seeds)]
                else:
                    decision_seed = _derived_decision_seed(
                        seed, world_seed, sample_index, group_decisions)
                group_decisions.add(decision_seed)
                decision_rng = random.Random(decision_seed)
                values = {**copy.deepcopy(world_values), **{
                    name: RandomCases._value(
                        decision_rng, name, declaration)
                    for name, declaration in self.variables.items()
                }}
                group.append(Case(
                    offset, world_seed, decision_seed, values,
                    world_values=copy.deepcopy(world_values),
                    world_index=world_index, sample_index=sample_index))
            yield tuple(group)


class CaseFile:
    """Replay exact cases from a newline-delimited JSON manifest.

    This is the handoff between a cheap distributed seed scout and an
    authoritative image materializer. Each line is either a serialized
    :class:`Case` or an object with that mapping under ``case``. Nothing is
    re-sampled or shuffled: world seed, decision seed, variations and source
    indexes remain auditable across machines.
    """

    def __init__(self, path, *, base_dir=".", **_):
        value = Path(str(path)).expanduser()
        self.path = value if value.is_absolute() else Path(base_dir) / value
        self.path = self.path.resolve()

    @staticmethod
    def _case(value, line):
        if isinstance(value, Mapping) and "case" in value:
            value = value["case"]
        if not isinstance(value, Mapping):
            raise TypeError(f"case file line {line} must be a JSON object")
        required = {"index", "seed", "decision_seed"}
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(
                f"case file line {line} is missing " + ", ".join(missing))
        return Case(
            index=int(value["index"]),
            seed=int(value["seed"]),
            decision_seed=int(value["decision_seed"]),
            values=copy.deepcopy(dict(value.get("values") or {})),
            world_index=(None if value.get("world_index") is None else
                         int(value["world_index"])),
            sample_index=(None if value.get("sample_index") is None else
                          int(value["sample_index"])),
            world_values=copy.deepcopy(
                dict(value.get("world_values") or {})),
        )

    def _read(self):
        with self.path.open(encoding="utf-8") as stream:
            for line_number, text in enumerate(stream, 1):
                if not text.strip():
                    continue
                try:
                    value = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON on case file line {line_number}: "
                        f"{exc.msg}") from exc
                yield self._case(value, line_number)

    @staticmethod
    def _take(values, count, path):
        values = iter(values)
        for index in range(int(count)):
            try:
                yield next(values)
            except StopIteration as exc:
                raise ValueError(
                    f"case file {path} contains only {index} cases; "
                    f"requested {count}") from exc

    def sample(self, count, seed):
        del seed
        yield from self._take(self._read(), int(count), self.path)

    def sample_worlds(self, count, samples_per_world, seed):
        del seed
        samples_per_world = int(samples_per_world)
        if samples_per_world < 1:
            raise ValueError("samples_per_world must be at least one")
        values = iter(self._read())
        for world in range(int(count)):
            group = []
            for sample in range(samples_per_world):
                try:
                    group.append(next(values))
                except StopIteration as exc:
                    consumed = world * samples_per_world + sample
                    raise ValueError(
                        f"case file {self.path} contains only {consumed} "
                        f"cases; requested {count} worlds x "
                        f"{samples_per_world} samples") from exc
            yield tuple(group)


class JSONLWriter:
    """Dependency-free, streaming writer with atomic replacement."""

    def __init__(self, output, **_):
        self.output = Path(output)

    def write(self, records):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_name(
            f".{self.output.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w") as stream:
                for record in records:
                    if hasattr(record, "to_dict"):
                        record = record.to_dict()
                    stream.write(json.dumps(json_value(record), sort_keys=True)
                                 + "\n")
            os.replace(temporary, self.output)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return str(self.output)


def assign_path(document, dotted, value):
    """Set one dotted mapping/list path, creating intermediate mappings."""
    target = document
    parts = [part for part in str(dotted).split(".") if part]
    if not parts:
        raise ValueError("binding target cannot be empty")
    for part in parts[:-1]:
        if isinstance(target, list):
            if not part.isdigit() or int(part) >= len(target):
                raise ValueError(
                    f"cannot bind {dotted!r}: {part!r} is not a valid "
                    "list index")
            child = target[int(part)]
        elif isinstance(target, dict):
            child = target.setdefault(part, {})
        else:
            raise ValueError(
                f"cannot bind {dotted!r}: {part!r} is not a container")
        if not isinstance(child, (dict, list)):
            raise ValueError(
                f"cannot bind {dotted!r}: {part!r} is not a container")
        target = child
    last = parts[-1]
    if isinstance(target, list):
        if not last.isdigit() or int(last) >= len(target):
            raise ValueError(
                f"cannot bind {dotted!r}: {last!r} is not a valid list index")
        target[int(last)] = copy.deepcopy(value)
    elif isinstance(target, dict):
        target[last] = copy.deepcopy(value)
    else:
        raise ValueError(
            f"cannot bind {dotted!r}: target is not a container")


def case_value(case, source):
    """Resolve a built-in case field or sampled value by name."""
    source = str(source)
    if source in ("index", "seed", "decision_seed",
                  "world_index", "sample_index", "values", "world_values"):
        return getattr(case, source)
    if source.startswith("world_values."):
        key = source.removeprefix("world_values.")
        if key in case.world_values:
            return case.world_values[key]
        raise KeyError(f"case has no world value {source!r}")
    key = source.removeprefix("values.")
    if key in case.values:
        return case.values[key]
    raise KeyError(f"case has no value {source!r}")


class TrialGenerator:
    """Execute a component-based run template once per generated case.

    ``bindings`` maps dotted fields in the run document to case fields such
    as ``seed``, ``decision_seed``, or ``values.route``. This makes the
    same generator useful for local models, remote APIs, arbitrary harnesses,
    and any environment adapter. Static ``overrides`` are applied last.
    """

    def __init__(self, run, bindings=None, overrides=None, *, base_dir=".",
                 metadata=None):
        import yaml

        self.path = Path(run)
        if not self.path.is_absolute():
            self.path = Path(base_dir) / self.path
        self.path = self.path.resolve()
        with self.path.open() as stream:
            self.template = yaml.safe_load(stream)
        if not isinstance(self.template, Mapping):
            raise TypeError("trial run template must be a mapping")
        self.bindings = dict(bindings or {})
        self.overrides = dict(overrides or {})
        self.metadata = dict(metadata or {})

    def __call__(self, case):
        from bedrock_rl.core.run import RunSpec

        document = copy.deepcopy(self.template)
        for target, source in self.bindings.items():
            assign_path(document, target, case_value(case, source))
        for target, value in self.overrides.items():
            assign_path(document, target, value)
        document["output"] = None
        run = RunSpec.from_dict(document, path=str(self.path))
        trajectory = asyncio.run(run.execute())
        trajectory.metadata.setdefault("generation_case", case.to_dict())
        trajectory.metadata.setdefault("generation", self.metadata)
        return trajectory


# A live Netherite engine is about 500 MiB before Python, one in-progress
# trajectory, image encoding, and the process itself. Budgeting one GiB per
# worker plus two GiB for the parent/OS keeps `auto` from turning a 16-GiB
# workstation into swap. There is deliberately no arbitrary process cap:
# exact-N jobs on large CPU/RAM hosts should use the hardware the operator
# made available. An explicit integer remains the escape hatch for measured
# workloads with a different footprint.
_AUTO_WORKER_BYTES = 1024**3
_AUTO_MEMORY_RESERVE = 2 * 1024**3


def _available_cpu_count():
    """CPUs this process may use, not merely CPUs installed on the host."""

    limits = [max(1, os.cpu_count() or 1)]
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            limits.append(max(1, len(affinity(0))))
        except OSError:
            pass
    quotas = (
        (Path("/sys/fs/cgroup/cpu.max"), None),
        (Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
         Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")),
    )
    for quota_path, period_path in quotas:
        try:
            if period_path is None:
                raw_quota, raw_period = quota_path.read_text().split()[:2]
            else:
                raw_quota = quota_path.read_text().strip()
                raw_period = period_path.read_text().strip()
            if raw_quota == "max":
                continue
            quota, period = int(raw_quota), int(raw_period)
            if quota > 0 and period > 0:
                limits.append(max(1, quota // period))
                break
        except (OSError, TypeError, ValueError):
            continue
    return min(limits)


def _memory_limit_bytes():
    """Smallest discoverable host/cgroup memory ceiling, or None."""

    limits = []
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if pages > 0 and page_size > 0:
            limits.append(pages * page_size)
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    # Containers commonly report the host's physical RAM through sysconf.
    # Respect both cgroup v2 and v1 limits when either is tighter.
    for path in (Path("/sys/fs/cgroup/memory.max"),
                 Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        try:
            raw = path.read_text().strip()
            value = int(raw)
        except (OSError, TypeError, ValueError):
            continue
        # v1 uses a huge sentinel for "unlimited"; values larger than the
        # physical host limit cannot constrain this process.
        if value > 0 and (not limits or value < min(limits)):
            limits.append(value)
    return min(limits) if limits else None


def _memory_worker_ceiling():
    limit = _memory_limit_bytes()
    if limit is None:
        # Unknown memory must not introduce a hidden fixed ceiling. The CPU
        # limit remains a finite conservative bound in _worker_ceiling().
        return _available_cpu_count()
    usable = max(_AUTO_WORKER_BYTES, limit - _AUTO_MEMORY_RESERVE)
    return max(1, usable // _AUTO_WORKER_BYTES)


def _worker_ceiling(value):
    if value in (None, "auto"):
        return min(_available_cpu_count(), _memory_worker_ceiling())
    n = int(value)
    if n < 1:
        raise ValueError("executor workers must be at least one")
    return n


def _sized_prefix(values, maximum):
    """Peek at most one worker's worth without requiring a sized input."""
    iterator = iter(values)
    prefix = list(islice(iterator, int(maximum)))
    return prefix, chain(prefix, iterator)


_PROCESS_GENERATOR = None
_PROCESS_INIT_ERROR = None
_PROCESS_CANDIDATE_TIMEOUT = None


class _CandidateTimedOut(BaseException):
    """Private control-flow exception raised inside one process worker.

    It deliberately inherits from BaseException so a generator's broad
    ``except Exception`` cannot turn an operator deadline into a fatal
    generator error. Normal ``finally`` cleanup still runs before the worker
    reports an ordinary rejected candidate to the parent.
    """

    def __init__(self, seconds):
        self.seconds = float(seconds)
        super().__init__(f"candidate exceeded {self.seconds:g} seconds")


def _deadline_call(function, *args):
    """Run one worker call under its optional POSIX wall-clock deadline."""

    seconds = _PROCESS_CANDIDATE_TIMEOUT
    if seconds is None:
        return function(*args)

    def expired(_signum, _frame):
        raise _CandidateTimedOut(seconds)

    prior_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, expired)
    prior_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return function(*args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prior_handler)
        # A ProcessExecutor worker owns this timer, but preserve a timer a
        # custom worker initializer installed instead of silently deleting it.
        if prior_timer[0] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, *prior_timer)


def _construct(spec, base_dir, metadata):
    return spec.create(base_dir=base_dir, metadata=dict(metadata or {}))


def _environment_gpus():
    """Explicit environment-renderer devices, in requested order."""

    raw = os.environ.get("NETHERITE_ENV_GPUS")
    if raw is None:
        return ()
    devices = tuple(value.strip() for value in raw.split(",")
                    if value.strip())
    if len(set(devices)) != len(devices):
        raise ValueError("NETHERITE_ENV_GPUS must not repeat a device")
    return devices


def _pin_process_gpu(devices, counter):
    """Give one process one renderer GPU without changing the parent."""

    if not devices or counter is None:
        return
    with counter.get_lock():
        index = int(counter.value)
        counter.value += 1
    # Explicit worker counts may intentionally exceed the GPU count. Cycle in
    # that case instead of silently dropping workers; auto is capped below.
    os.environ["NETHERITE_ENV_GPUS"] = devices[index % len(devices)]


def _process_init(spec, base_dir, metadata, gpu_devices=(), gpu_counter=None,
                  candidate_timeout_seconds=None, artifact_owner_pid=None):
    global _PROCESS_GENERATOR, _PROCESS_INIT_ERROR
    global _PROCESS_CANDIDATE_TIMEOUT
    # ProcessExecutor owns data parallelism.  Letting NumPy/OpenBLAS create a
    # second thread pool inside every world worker can turn independent
    # engines into thousands of mostly idle threads and severe scheduler
    # contention.  Set these before constructing the generator (and therefore
    # before the snapshot fast path imports NumPy).  A workload that actually
    # needs intra-case BLAS parallelism belongs behind ThreadExecutor or a
    # custom executor where it can own that choice explicitly.
    for name in (
        "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
    _PROCESS_CANDIDATE_TIMEOUT = candidate_timeout_seconds
    _pin_process_gpu(tuple(gpu_devices), gpu_counter)
    try:
        _PROCESS_GENERATOR = _construct(spec, base_dir, metadata)
        if (artifact_owner_pid is not None
                and hasattr(_PROCESS_GENERATOR, "_artifact_owner_pid")):
            _PROCESS_GENERATOR._artifact_owner_pid = int(artifact_owner_pid)
        _PROCESS_INIT_ERROR = None
    except Exception as exc:
        # Raising from a Pool initializer makes multiprocessing replace the
        # failed worker forever. Keep the worker alive and report the error
        # through the ordinary rejected-case path instead.
        _PROCESS_GENERATOR = None
        _PROCESS_INIT_ERROR = f"{type(exc).__name__}: {exc}"


def _call(generator, case):
    try:
        value = generator(case)
        if value is None:
            return {"ok": False, "case": case.to_dict(),
                    "code": "generator_rejected",
                    "reason": "generator rejected the case"}
        return {"ok": True, "case": case.to_dict(), "value": value}
    except CaseRejected as exc:
        outcome = {"ok": False, "case": case.to_dict(),
                   "code": exc.code, "reason": str(exc)}
        if exc.details:
            outcome["details"] = exc.details
        return outcome
    except Exception as exc:
        return {"ok": False, "case": case.to_dict(),
                "fatal": True,
                "code": "generator_error",
                "reason": f"{type(exc).__name__}: {exc}"}


@dataclass(frozen=True)
class _StagedGroup:
    """Trusted executor-internal reference to one accepted world group.

    This pickle spool is an implementation detail for values created by the
    current run. It is deliberately not an input format and must never be
    constructed from or used to deserialize an untrusted external path.
    """

    directory: str
    count: int

    def values(self):
        """Load in sample order, unlinking each value before yielding it."""

        directory = Path(self.directory)

        def staged_values():
            try:
                for index in range(self.count):
                    path = directory / f"{index:08d}.pickle"
                    with path.open("rb") as stream:
                        value = pickle.load(stream)
                    path.unlink()
                    yield value
            finally:
                shutil.rmtree(directory, ignore_errors=True)

        return staged_values()

    def discard(self):
        shutil.rmtree(self.directory, ignore_errors=True)


def _group_outcome_values(outcome):
    """Return a lazy ordered iterator for a built-in or custom group outcome."""

    staged = outcome.get("spool")
    if staged is not None:
        if not isinstance(staged, _StagedGroup):
            raise TypeError("executor returned an invalid internal group spool")
        return staged.count, staged.values()
    # Custom executors may return values in memory; built-in executors spool.
    values = outcome["values"]
    return len(values), iter(values)


def _discard_group_outcome(outcome):
    staged = outcome.get("spool")
    if isinstance(staged, _StagedGroup):
        staged.discard()


def _stage_group_value(directory, index, value):
    """Serialize one generated value without retaining its siblings in RAM."""

    path = Path(directory) / f"{int(index):08d}.pickle"
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _process_call(case):
    if _PROCESS_INIT_ERROR is not None:
        return {"ok": False, "case": case.to_dict(),
                "fatal": True,
                "code": "generator_init_error",
                "reason": "generator initialization failed: "
                          + _PROCESS_INIT_ERROR}
    try:
        return _deadline_call(_call, _PROCESS_GENERATOR, case)
    except _CandidateTimedOut as exc:
        return {
            "ok": False, "case": case.to_dict(), "fatal": False,
            "code": "candidate_timeout",
            "reason": str(exc),
            "details": {"timeout_seconds": exc.seconds,
                        "scope": "case"},
        }


def _call_group(generator, cases, spool_root=None):
    """Run one world's variants atomically through one generator instance."""
    cases = tuple(cases)
    if not cases:
        return {"ok": False, "cases": [], "attempted": 0,
                "code": "empty_world_group",
                "reason": "sampler produced an empty world group"}
    identify = getattr(generator, "group_identity", None)
    if identify is not None:
        try:
            identities = [json_value(
                identify(case), what="generator world-group identity")
                for case in cases]
        except Exception as exc:
            return {
                "ok": False, "fatal": True,
                "case": cases[0].to_dict(),
                "cases": [item.to_dict() for item in cases],
                "attempted": 0, "code": "group_identity_error",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        if any(identity != identities[0] for identity in identities[1:]):
            return {
                "ok": False, "fatal": True,
                "case": cases[0].to_dict(),
                "cases": [item.to_dict() for item in cases],
                "attempted": 0, "code": "group_identity_mismatch",
                "reason": ("samples_per_world resolved different world/task "
                           "identities; task setup may depend only on values "
                           "shared by every variant in a world"),
            }
    # A grouped generator may expose a cheap, read-only world qualifier.  It
    # runs before the first (potentially expensive) sample and receives the
    # complete group so decision-dependent checks can prove that every
    # variant is viable.  This is deliberately a generator protocol instead
    # of a generation-core registry: Minecraft snapshots, simulators, remote
    # sandboxes, and custom environments can each implement their own filter.
    qualify = getattr(generator, "qualify_group", None)
    if qualify is not None:
        try:
            qualify(cases)
        except CaseRejected as exc:
            return {
                "ok": False,
                "case": cases[0].to_dict(),
                "cases": [item.to_dict() for item in cases],
                "attempted": 0,
                "discarded": 0,
                "fatal": False,
                "code": exc.code,
                **({"details": exc.details} if exc.details else {}),
                "reason": f"world qualification rejected: {exc}",
            }
        except Exception as exc:
            return {
                "ok": False,
                "case": cases[0].to_dict(),
                "cases": [item.to_dict() for item in cases],
                "attempted": 0,
                "discarded": 0,
                "fatal": True,
                "code": "group_qualification_error",
                "reason": ("world qualification failed: "
                           f"{type(exc).__name__}: {exc}"),
            }
    directory = Path(tempfile.mkdtemp(
        prefix="bedrock-world-", dir=spool_root))
    attempted = 0
    staged = 0
    try:
        for case in cases:
            outcome = _call(generator, case)
            attempted += 1
            if not outcome["ok"]:
                shutil.rmtree(directory, ignore_errors=True)
                return {
                    "ok": False,
                    "case": cases[0].to_dict(),
                    "cases": [item.to_dict() for item in cases],
                    "attempted": attempted,
                    "discarded": staged,
                    "fatal": bool(outcome.get("fatal", False)),
                    "code": outcome.get("code", "case_rejected"),
                    **({"details": outcome["details"]}
                       if outcome.get("details") else {}),
                    "reason": (f"sample {case.sample_index} rejected: "
                               f"{outcome['reason']}"),
                }
            try:
                _stage_group_value(directory, staged, outcome["value"])
            except Exception as exc:
                shutil.rmtree(directory, ignore_errors=True)
                return {
                    "ok": False,
                    "case": cases[0].to_dict(),
                    "cases": [item.to_dict() for item in cases],
                    "attempted": attempted,
                    "discarded": staged,
                    "fatal": True,
                    "code": "group_spool_error",
                    "reason": (f"sample {case.sample_index} could not be "
                               "staged: "
                               f"{type(exc).__name__}: {exc}"),
                }
            staged += 1
            del outcome
        return {
            "ok": True,
            "case": cases[0].to_dict(),
            "cases": [item.to_dict() for item in cases],
            "attempted": attempted,
            "spool": _StagedGroup(str(directory), staged),
        }
    except _CandidateTimedOut as exc:
        shutil.rmtree(directory, ignore_errors=True)
        return {
            "ok": False,
            "case": cases[0].to_dict(),
            "cases": [item.to_dict() for item in cases],
            "attempted": attempted,
            "discarded": staged,
            "fatal": False,
            "code": "candidate_timeout",
            "reason": str(exc),
            "details": {"timeout_seconds": exc.seconds,
                        "scope": "world_group"},
        }
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise


def _validated_world_groups(groups, samples_per_world):
    """Validate a sampler's grouped contract without buffering its cases."""
    seen_worlds = set()
    for ordinal, value in enumerate(groups):
        group = tuple(value)
        if len(group) != samples_per_world:
            raise ValueError(
                f"world sampler group {ordinal} contains {len(group)} "
                f"cases; expected {samples_per_world}")
        if not group:
            raise ValueError(f"world sampler group {ordinal} is empty")
        if any(not isinstance(case, Case) for case in group):
            raise TypeError(
                f"world sampler group {ordinal} must contain Case objects")
        seeds = {int(case.seed) for case in group}
        if len(seeds) != 1:
            raise ValueError(
                f"world sampler group {ordinal} mixes seeds "
                + ", ".join(str(seed) for seed in sorted(seeds)))
        world = next(iter(seeds))
        if world in seen_worlds:
            raise ValueError(
                f"world sampler repeated candidate seed {world}; worlds "
                "mode requires distinct natural worlds")
        seen_worlds.add(world)
        decisions = {int(case.decision_seed) for case in group}
        if len(decisions) != len(group):
            raise ValueError(
                f"world sampler repeated a decision seed inside world "
                f"{world}; samples_per_world must be distinct variants")
        world_values = [json_value(case.world_values) for case in group]
        if any(value != world_values[0] for value in world_values[1:]):
            raise ValueError(
                f"world sampler changed world_values inside world {world}")
        world_indexes = {case.world_index for case in group}
        if len(world_indexes) != 1 or None in world_indexes:
            shown_indexes = sorted(
                world_indexes, key=lambda item: -1 if item is None else item)
            raise ValueError(
                f"world sampler group {ordinal} has world_index values "
                f"{shown_indexes}; expected one non-null group identity")
        sample_indexes = [case.sample_index for case in group]
        if sample_indexes != list(range(samples_per_world)):
            raise ValueError(
                f"world sampler group {ordinal} has sample_index values "
                f"{sample_indexes}; expected "
                f"{list(range(samples_per_world))}")
        indexes = [case.index for case in group]
        if len(set(indexes)) != len(indexes):
            raise ValueError(
                f"world sampler group {ordinal} repeats case indexes "
                f"{indexes}")
        yield group


def _process_group_call(cases, spool_root):
    if _PROCESS_INIT_ERROR is not None:
        cases = tuple(cases)
        return {"ok": False,
                "case": cases[0].to_dict() if cases else {},
                "cases": [case.to_dict() for case in cases],
                "attempted": 0,
                "fatal": True,
                "code": "generator_init_error",
                "reason": "generator initialization failed: "
                          + _PROCESS_INIT_ERROR}
    try:
        return _deadline_call(
            _call_group, _PROCESS_GENERATOR, cases, spool_root)
    except _CandidateTimedOut as exc:
        # The deadline may fire during group identity or qualification,
        # before _call_group has created a spool it can clean itself.
        cases = tuple(cases)
        return {
            "ok": False,
            "case": cases[0].to_dict() if cases else {},
            "cases": [case.to_dict() for case in cases],
            "attempted": 0,
            "discarded": 0,
            "fatal": False,
            "code": "candidate_timeout",
            "reason": str(exc),
            "details": {"timeout_seconds": exc.seconds,
                        "scope": "world_group"},
        }


def _thread_map_bounded(values, fn, workers, prefetch=2):
    """Bounded busy-worker thread map with deterministic output order.

    The window counts every submitted result that has not been emitted, not
    merely futures that are still running.  Otherwise one slow early value
    lets completed later values accumulate without bound in ``ready``.
    """
    prefix, values = _sized_prefix(values, workers)
    workers = min(workers, len(prefix))
    if not workers:
        return iter(())

    def results():
        pool = ThreadPoolExecutor(max_workers=workers)
        source = iter(values)
        pending = {}
        ready = {}
        submitted = 0
        emitted = 0
        window = workers * max(1, int(prefetch))

        def submit(value):
            nonlocal submitted
            pending[pool.submit(fn, value)] = submitted
            submitted += 1

        def refill():
            while submitted - emitted < window:
                try:
                    value = next(source)
                except StopIteration:
                    return
                submit(value)

        try:
            refill()
            while pending or ready:
                while emitted in ready:
                    value = ready.pop(emitted)
                    emitted += 1
                    # Refill before yielding so a consumer processing this
                    # result does not leave an otherwise idle worker behind.
                    # Advancing ``emitted`` first keeps active + ready results
                    # within the fixed window.
                    refill()
                    yield value
                if pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        ready[pending.pop(future)] = future.result()
        finally:
            for future in pending:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
    return results()


class SerialExecutor:
    def __init__(self, **_):
        pass

    def run(self, cases, generator, *, base_dir, metadata):
        return self._run(
            cases, generator, _call,
            base_dir=base_dir, metadata=metadata, grouped=False)

    def run_groups(self, groups, generator, *, base_dir, metadata):
        return self._run(
            groups, generator, _call_group,
            base_dir=base_dir, metadata=metadata, grouped=True)

    @staticmethod
    def _run(values, generator, fn, *, base_dir, metadata, grouped):
        def results():
            instance = _construct(generator, base_dir, metadata)
            spool = (tempfile.TemporaryDirectory(
                prefix="bedrock-generation-groups-") if grouped else None)
            try:
                for value in values:
                    yield (fn(instance, value, spool.name) if grouped
                           else fn(instance, value))
            finally:
                close = getattr(instance, "close", None)
                if close is not None:
                    close()
                if spool is not None:
                    spool.cleanup()

        return results()


class ThreadExecutor:
    """Shared thread-safe generator for remote APIs and model servers."""

    def __init__(self, workers="auto", **_):
        self.workers = workers

    def run(self, cases, generator, *, base_dir, metadata):
        return self._run(cases, generator, _call,
                         base_dir=base_dir, metadata=metadata, grouped=False)

    def run_groups(self, groups, generator, *, base_dir, metadata):
        return self._run(groups, generator, _call_group,
                         base_dir=base_dir, metadata=metadata, grouped=True)

    def _run(self, values, generator, fn, *, base_dir, metadata, grouped):
        def results():
            instance = _construct(generator, base_dir, metadata)
            if getattr(instance, "thread_safe", True) is False:
                close_instance = getattr(instance, "close", None)
                if close_instance is not None:
                    close_instance()
                raise TypeError(
                    f"{type(instance).__name__} is stateful and cannot be "
                    "shared by ThreadExecutor; use ProcessExecutor or "
                    "SerialExecutor")
            spool = (tempfile.TemporaryDirectory(
                prefix="bedrock-generation-groups-") if grouped else None)
            outcomes = _thread_map_bounded(
                values, lambda value: (
                    fn(instance, value, spool.name) if grouped
                    else fn(instance, value)),
                _worker_ceiling(self.workers))
            try:
                yield from outcomes
            finally:
                close_outcomes = getattr(outcomes, "close", None)
                if close_outcomes is not None:
                    close_outcomes()
                close_instance = getattr(instance, "close", None)
                if close_instance is not None:
                    close_instance()
                if spool is not None:
                    spool.cleanup()

        return results()


def _process_pool_failure(pool, known_processes, expected_pids):
    """Describe a worker death/replacement that cannot deliver its callback."""
    processes = getattr(pool, "_pool", None)
    if processes is None:
        return None
    known_ids = {id(process) for process in known_processes}
    for process in processes:
        if id(process) not in known_ids:
            known_processes.append(process)
            known_ids.add(id(process))
    dead = [(getattr(process, "pid", None),
             getattr(process, "exitcode", None))
            for process in known_processes
            if getattr(process, "exitcode", None) not in (None, 0)]
    if dead:
        return f"process worker exited before returning a result: {dead}"
    current = {getattr(process, "pid", None) for process in processes}
    current.discard(None)
    if expected_pids and current and current != expected_pids:
        return ("process worker set changed while work was pending "
                f"({sorted(expected_pids)} -> {sorted(current)}); a worker "
                "was replaced before its callback could be delivered")
    return None


class ProcessExecutor:
    """One generator and, for Netherite, one C engine per worker.

    CPU rendering is the default intentionally: a CUDA raster process uses
    gigabytes of VRAM, so it is the wrong throughput path for hundreds of
    independent seeds. Models can instead use ``ThreadExecutor`` against one
    batched inference server. ``candidate_timeout_seconds`` is an optional
    POSIX wall-clock backstop over a complete case or atomic world group.
    Timeout mode retires each worker after one candidate so interrupted custom
    state can never reach the next world.
    """

    def __init__(self, workers="auto", start_method=None, chunksize=2,
                 heartbeat_seconds=10, candidate_timeout_seconds=None, **_):
        self.workers = workers
        self.start_method = start_method or (
            "forkserver" if os.name != "nt" else "spawn")
        self.chunksize = int(chunksize)
        if self.chunksize < 1:
            raise ValueError("executor chunksize must be at least one")
        self.heartbeat_seconds = float(heartbeat_seconds)
        if self.heartbeat_seconds <= 0:
            raise ValueError("executor heartbeat_seconds must be positive")
        self.candidate_timeout_seconds = (
            None if candidate_timeout_seconds is None
            else float(candidate_timeout_seconds))
        if (self.candidate_timeout_seconds is not None
                and (not math.isfinite(self.candidate_timeout_seconds)
                     or self.candidate_timeout_seconds <= 0)):
            raise ValueError(
                "executor candidate_timeout_seconds must be positive")
        if (self.candidate_timeout_seconds is not None
                and not (hasattr(signal, "setitimer")
                         and hasattr(signal, "SIGALRM"))):
            raise ValueError(
                "executor candidate_timeout_seconds requires a POSIX "
                "process worker")

    def run(self, cases, generator, *, base_dir, metadata, heartbeat=None):
        return self._run(cases, generator, _process_call,
                         base_dir=base_dir, metadata=metadata, grouped=False,
                         heartbeat=heartbeat)

    def run_groups(self, groups, generator, *, base_dir, metadata,
                   heartbeat=None):
        # Keeping a world's samples in one work item is what makes an
        # initial-state snapshot a once-per-world cost instead of N racing
        # builds in N processes.
        return self._run(groups, generator, _process_group_call,
                         base_dir=base_dir, metadata=metadata, grouped=True,
                         heartbeat=heartbeat)

    def _run(self, values, generator, fn, *, base_dir, metadata, grouped,
             heartbeat=None):
        # Fail before starting a pool when an import path or constructor is
        # invalid. Worker initialization still guards its own environment so
        # a worker-only failure is returned instead of causing a respawn loop.
        preflight = _construct(generator, base_dir, metadata)
        close = getattr(preflight, "close", None)
        if close is not None:
            close()
        ceiling = _worker_ceiling(self.workers)
        context = get_context(self.start_method)
        gpu_devices = _environment_gpus()
        # NETHERITE_ENV_GPUS is specifically the CUDA environment-renderer
        # pool. With auto sizing, one worker per listed device avoids all
        # engines selecting the first visible GPU and exhausting its VRAM.
        # An explicit worker count remains an intentional oversubscription.
        if self.workers in (None, "auto") and gpu_devices:
            ceiling = min(ceiling, len(gpu_devices))
        gpu_counter = (context.Value("Q", 0)
                       if gpu_devices and ceiling > 1 else None)

        def results():
            spool = (tempfile.TemporaryDirectory(
                prefix="bedrock-generation-groups-") if grouped else None)
            try:
                prefix, stream = _sized_prefix(values, ceiling)
                workers = min(ceiling, len(prefix))
                if not workers:
                    return
                if workers == 1 and self.candidate_timeout_seconds is None:
                    instance = _construct(generator, base_dir, metadata)
                    if hasattr(instance, "_artifact_owner_pid"):
                        instance._artifact_owner_pid = os.getpid()
                    try:
                        for value in stream:
                            yield (_call_group(instance, value, spool.name)
                                   if grouped else _call(instance, value))
                    finally:
                        close = getattr(instance, "close", None)
                        if close is not None:
                            close()
                    return
                initargs = (
                    generator, base_dir, metadata, gpu_devices,
                    gpu_counter, self.candidate_timeout_seconds,
                    os.getpid())
                if self.candidate_timeout_seconds is not None:
                    # An asynchronous deadline may interrupt arbitrary custom
                    # generator code after it mutated process-local state.
                    # Retiring the worker after this candidate means neither
                    # that state nor a half-torn-down native extension can
                    # contaminate the replacement world. Pool preserves the
                    # configured concurrency by spawning a clean slot.
                    pool = context.Pool(
                        workers, initializer=_process_init,
                        initargs=initargs, maxtasksperchild=1)
                else:
                    pool = context.Pool(
                        workers, initializer=_process_init,
                        initargs=initargs)
                expected_pids = {
                    getattr(process, "pid", None)
                    for process in getattr(pool, "_pool", ())}
                expected_pids.discard(None)
                if self.candidate_timeout_seconds is not None:
                    expected_pids.clear()
                known_processes = list(getattr(pool, "_pool", ()))
                source = iter(stream)
                completed = Queue()
                pending = 0
                ready = {}
                submitted = 0
                emitted = 0
                complete = False
                window = workers * max(1, self.chunksize)

                def submit(value):
                    nonlocal pending, submitted
                    ordinal = submitted
                    submitted += 1
                    pending += 1
                    args = ((value, spool.name) if grouped else (value,))
                    pool.apply_async(
                        fn, args,
                        callback=lambda result, index=ordinal:
                            completed.put((index, True, result)),
                        error_callback=lambda error, index=ordinal:
                            completed.put((index, False, error)))

                def refill():
                    while submitted - emitted < window:
                        try:
                            value = next(source)
                        except StopIteration:
                            return
                        submit(value)

                try:
                    refill()
                    while pending or ready:
                        while emitted in ready:
                            value = ready.pop(emitted)
                            emitted += 1
                            # Completed out-of-order values occupy the same
                            # bounded window as running work. Refill only after
                            # an ordered value has left the parent process.
                            refill()
                            yield value
                        if pending:
                            try:
                                index, ok, value = completed.get(
                                    timeout=self.heartbeat_seconds)
                            except Empty:
                                failure = _process_pool_failure(
                                    pool, known_processes, expected_pids)
                                if failure is not None:
                                    raise RuntimeError(failure)
                                if heartbeat is not None:
                                    heartbeat({
                                        "submitted": submitted,
                                        "completed": emitted + len(ready),
                                        "committed": emitted,
                                        "running": pending,
                                    })
                                continue
                            pending -= 1
                            if not ok:
                                raise value
                            ready[index] = value
                    complete = True
                finally:
                    if complete:
                        pool.close()
                    else:
                        # Exact-count generation closes this iterator as soon
                        # as enough worlds succeed. Kill only the bounded
                        # in-flight tail instead of waiting for every candidate.
                        pool.terminate()
                    pool.join()
            finally:
                if spool is not None:
                    spool.cleanup()

        return results()


GENERATION_KEYS = frozenset((
    "schema_version", "count", "worlds", "samples_per_world", "seed",
    "max_attempts", "max_world_attempts", "sampler",
    "generator", "executor", "writer", "output", "rejections_output",
    "metadata",
    "unique_worlds",
))
GENERATION_MANIFEST_KEYS = frozenset(("defaults", "jobs"))


def _merged(base, overlay):
    """Recursively merge mappings without mutating either input."""
    if not isinstance(base, Mapping) or not isinstance(overlay, Mapping):
        return copy.deepcopy(overlay)
    # Import-path components are atomic when their implementation changes.
    # Retaining constructor keys from the old type creates a hybrid component
    # that neither class accepts. Matching types may still refine shared
    # constructor settings job by job.
    if ("type" in base and "type" in overlay
            and base["type"] != overlay["type"]):
        return copy.deepcopy(dict(overlay))
    out = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        out[key] = (_merged(out[key], value)
                    if key in out else copy.deepcopy(value))
    return out


def generation_jobs(value):
    """Expand one generation document into named, independent jobs.

    A traditional single-job document remains valid.  A manifest uses
    ``defaults`` plus ``jobs`` so train/dev/test/demo definitions can live in
    one readable file without introducing an execution pipeline.
    """
    if not isinstance(value, Mapping):
        raise TypeError("generation config must be a mapping")
    if "jobs" not in value:
        return {None: copy.deepcopy(dict(value))}
    unknown = sorted(set(value) - GENERATION_MANIFEST_KEYS)
    if unknown:
        raise ValueError("unknown generation manifest keys: "
                         + ", ".join(unknown))
    defaults = value.get("defaults") or {}
    jobs = value.get("jobs")
    if not isinstance(defaults, Mapping):
        raise TypeError("generation manifest defaults must be a mapping")
    if not isinstance(jobs, Mapping) or not jobs:
        raise ValueError("generation manifest jobs must be a non-empty mapping")
    expanded = {}
    for name, job in jobs.items():
        name = str(name).strip()
        if not name:
            raise ValueError("generation job names cannot be empty")
        if not isinstance(job, Mapping):
            raise TypeError(f"generation job {name!r} must be a mapping")
        expanded[name] = _merged(defaults, job)
    return expanded


def load_generation_jobs(path):
    """Load every job in a single config or named-job manifest."""
    import yaml
    path = Path(path).resolve()
    with path.open() as stream:
        return generation_jobs(yaml.safe_load(stream) or {})


@dataclass(frozen=True)
class GenerationSpec:
    count: int
    generator: ComponentSpec
    output: str
    rejections_output: str | None = None
    worlds: int | None = None
    samples_per_world: int = 1
    sampler: ComponentSpec = field(default_factory=lambda: ComponentSpec(
        "bedrock_rl.sdg:RandomCases"))
    executor: ComponentSpec = field(default_factory=lambda: ComponentSpec(
        "bedrock_rl.sdg:ProcessExecutor"))
    writer: ComponentSpec = field(default_factory=lambda: ComponentSpec(
        "bedrock_rl.sdg:JSONLWriter"))
    seed: int = 0
    max_attempts: int | None = None
    max_world_attempts: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1
    unique_worlds: bool = False
    path: str | None = None

    @classmethod
    def from_dict(cls, value, *, path=None):
        if not isinstance(value, Mapping):
            raise TypeError("generation config must be a mapping")
        raw = dict(value)
        unknown = sorted(set(raw) - GENERATION_KEYS)
        if unknown:
            raise ValueError("unknown generation keys: " + ", ".join(unknown))
        schema_version = int(raw.get("schema_version", 1))
        if schema_version != 1:
            raise ValueError(
                f"unsupported generation schema_version {schema_version}; "
                "this release supports only 1")
        has_count, has_worlds = "count" in raw, "worlds" in raw
        if has_count == has_worlds:
            raise ValueError(
                "generation config needs exactly one of count (independent "
                "cases) "
                "or worlds (distinct accepted world groups)")
        worlds = int(raw["worlds"]) if has_worlds else None
        samples_per_world = int(raw.get("samples_per_world", 1))
        if worlds is not None and worlds < 1:
            raise ValueError("generation worlds must be at least one")
        if samples_per_world < 1:
            raise ValueError("samples_per_world must be at least one")
        if worlds is None and "samples_per_world" in raw:
            raise ValueError("samples_per_world requires worlds")
        count = (int(raw["count"]) if worlds is None else
                 worlds * samples_per_world)
        if count < 1:
            raise ValueError("generation count must be at least one")
        if "generator" not in raw:
            raise ValueError("generation config needs a generator component")
        if not raw.get("output"):
            raise ValueError("generation config needs an output path")
        base = Path(path).resolve().parent if path else Path.cwd()
        output = Path(str(raw["output"]))
        if not output.is_absolute():
            output = base / output
        rejections_output = raw.get("rejections_output")
        if rejections_output is not None:
            if not str(rejections_output).strip():
                raise ValueError(
                    "generation rejections_output cannot be empty")
            rejections_output = Path(str(rejections_output))
            if not rejections_output.is_absolute():
                rejections_output = base / rejections_output
            if rejections_output == output:
                raise ValueError(
                    "rejections_output must differ from output")
        if worlds is None and "max_world_attempts" in raw:
            raise ValueError("max_world_attempts requires worlds")
        if worlds is not None and "max_attempts" in raw:
            raise ValueError(
                "use max_world_attempts with worlds; max_attempts counts "
                "independent rows")
        attempt_key = ("max_world_attempts" if worlds is not None else
                       "max_attempts")
        attempts = raw.get(attempt_key)
        attempts = None if attempts is None else int(attempts)
        minimum_attempts = worlds if worlds is not None else count
        if attempts is not None and attempts < minimum_attempts:
            raise ValueError(
                f"{attempt_key} cannot be smaller than "
                + ("worlds" if worlds is not None else "count"))
        unique_worlds = raw.get("unique_worlds", False)
        if not isinstance(unique_worlds, bool):
            raise TypeError("generation unique_worlds must be true or false")
        if worlds is not None:
            unique_worlds = True
        return cls(
            count=count, worlds=worlds,
            samples_per_world=samples_per_world,
            generator=ComponentSpec.parse(raw["generator"],
                                          what="generator"),
            output=str(output),
            rejections_output=(None if rejections_output is None else
                               str(rejections_output)),
            sampler=ComponentSpec.parse(raw.get(
                "sampler", "bedrock_rl.sdg:RandomCases"),
                what="sampler"),
            executor=ComponentSpec.parse(raw.get(
                "executor", "bedrock_rl.sdg:ProcessExecutor"),
                what="executor"),
            writer=ComponentSpec.parse(raw.get(
                "writer", "bedrock_rl.sdg:JSONLWriter"),
                what="writer"),
            seed=int(raw.get("seed", 0)),
            max_attempts=(attempts if worlds is None else None),
            max_world_attempts=(attempts if worlds is not None else None),
            metadata=json_value(dict(raw.get("metadata") or {})),
            schema_version=schema_version,
            unique_worlds=unique_worlds, path=path)

    @classmethod
    def load(cls, path, job=None):
        path = Path(path).resolve()
        jobs = load_generation_jobs(path)
        if job is None:
            if len(jobs) != 1:
                raise ValueError(
                    f"{path} has named generation jobs: "
                    + ", ".join(jobs)
                    + "; choose one")
            document = next(iter(jobs.values()))
        else:
            try:
                document = jobs[str(job)]
            except KeyError as exc:
                raise ValueError(
                    f"{path} has no generation job {job!r}; choose "
                    + ", ".join(name for name in jobs if name is not None)) \
                    from exc
        return cls.from_dict(document, path=str(path))

    @property
    def base_dir(self):
        return str(Path(self.path).resolve().parent) if self.path else os.getcwd()

    def validate_components(self):
        return tuple((name, component.resolve()) for name, component in (
            ("sampler", self.sampler), ("generator", self.generator),
            ("executor", self.executor), ("writer", self.writer)))

    def execute(self, *, progress=None):
        grouped = self.worlds is not None
        requested_worlds = self.worlds if grouped else None
        attempts = ((self.max_world_attempts or requested_worlds)
                    if grouped else (self.max_attempts or self.count))
        sampler = self.sampler.create(base_dir=self.base_dir,
                                      metadata=dict(self.metadata))
        if grouped:
            sample_worlds = getattr(sampler, "sample_worlds", None)
            if sample_worlds is None:
                raise TypeError(
                    f"sampler {self.sampler.type!r} does not implement "
                    "sample_worlds(count, samples_per_world, seed), which "
                    "world-group generation requires")
            inputs = _validated_world_groups(sample_worlds(
                attempts, self.samples_per_world, self.seed),
                self.samples_per_world)
        else:
            inputs = sampler.sample(attempts, self.seed)
        if self.unique_worlds and not grouped:
            # This preflight retains only distinct integers, not Case objects.
            # Re-sampling is deterministic and keeps the execution stream
            # lazy while preserving the old fail-before-generator behavior.
            available = len({case.seed for case in inputs})
            if available < self.count:
                raise ValueError(
                    f"unique_worlds requests {self.count} accepted worlds, "
                    f"but the sampler produced only {available} distinct "
                    "world seeds")
            inputs = sampler.sample(attempts, self.seed)
        executor = self.executor.create()
        if grouped:
            run_groups = getattr(executor, "run_groups", None)
            if run_groups is None:
                raise TypeError(
                    f"executor {self.executor.type!r} does not implement "
                    "run_groups; grouped worlds must stay on one worker so "
                    "their snapshot is built once and acceptance is atomic")
            run_kwargs = {"base_dir": self.base_dir,
                          "metadata": self.metadata}
            executor_heartbeat = None
            if progress is not None and isinstance(executor, ProcessExecutor):
                executor_heartbeat = lambda state: report_progress(
                    force=True, executor_state=state)
                run_kwargs["heartbeat"] = executor_heartbeat
            outcomes = run_groups(inputs, self.generator, **run_kwargs)
        else:
            run_kwargs = {"base_dir": self.base_dir,
                          "metadata": self.metadata}
            if progress is not None and isinstance(executor, ProcessExecutor):
                run_kwargs["heartbeat"] = lambda state: report_progress(
                    force=True, executor_state=state)
            outcomes = executor.run(inputs, self.generator, **run_kwargs)
        stats = {"attempted": 0, "accepted": 0, "rejected": 0}
        world_stats = {"worlds_attempted": 0, "worlds_accepted": 0,
                       "worlds_rejected": 0, "samples_executed": 0}
        failure_counts = {}
        failure_examples = {}
        failure_details = {}
        accepted_worlds = set()
        emitted = 0
        progress_started = time.monotonic()
        progress_last_at = progress_started
        progress_last_attempts = 0
        progress_max_completed = 0
        progress_interval = 10.0
        rejection_stream = None
        rejection_temporary = None
        if self.rejections_output is not None:
            rejection_target = Path(self.rejections_output)
            rejection_target.parent.mkdir(parents=True, exist_ok=True)
            rejection_stream = tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", delete=False,
                dir=rejection_target.parent,
                prefix=f".{rejection_target.name}.", suffix=".tmp")
            rejection_temporary = Path(rejection_stream.name)

        def record_rejection(outcome):
            if rejection_stream is None:
                return
            row = {
                "schema": "bedrock.rejection.v1",
                "case": outcome.get("case"),
                **({"cases": outcome["cases"]}
                   if len(outcome.get("cases", ())) > 1 else {}),
                "code": outcome.get("code", "case_rejected"),
                "reason": outcome.get("reason", "case rejected"),
                **({"details": outcome["details"]}
                   if outcome.get("details") else {}),
                **({"fatal": True} if outcome.get("fatal") else {}),
                **({"attempted_samples": int(outcome["attempted"])}
                   if "attempted" in outcome else {}),
                **({"discarded_samples": int(outcome["discarded"])}
                   if outcome.get("discarded") else {}),
            }
            rejection_stream.write(json.dumps(
                json_value(row, what="rejection ledger row"),
                separators=(",", ":"), sort_keys=True) + "\n")

        def report_progress(*, force=False, executor_state=None):
            nonlocal progress_last_at, progress_last_attempts
            nonlocal progress_max_completed
            if progress is None:
                return
            now = time.monotonic()
            committed = (world_stats["worlds_attempted"] if grouped
                         else stats["attempted"])
            completed = max(
                committed, progress_max_completed,
                int((executor_state or {}).get("completed", committed)))
            progress_max_completed = completed
            if (not force and completed == progress_last_attempts
                    and now - progress_last_at < progress_interval):
                return
            if (not force and now - progress_last_at < progress_interval
                    and completed - progress_last_attempts < 10):
                return
            elapsed = max(0.0, now - progress_started)
            accepted = (world_stats["worlds_accepted"] if grouped
                        else stats["accepted"])
            wanted = requested_worlds if grouped else self.count
            rate = completed / elapsed if elapsed else 0.0
            acceptance = accepted / completed if completed else 0.0
            eta = (None if rate <= 0 or acceptance <= 0 else
                   max(0, wanted - accepted) / (rate * acceptance))
            top = ",".join(
                f"{code}:{count}" for code, count in sorted(
                    failure_counts.items(),
                    key=lambda item: (-item[1], item[0]))[:3]) or "none"
            progress({
                "completed": completed,
                "committed": committed,
                "accepted": accepted,
                "wanted": wanted,
                "attempt_limit": attempts,
                "rejected": (world_stats["worlds_rejected"] if grouped
                             else stats["rejected"]),
                # Keep a compact display field and expose the complete counts
                # as structured data for long-running seed scouts.
                "rejection_counts": {
                    code: count for code, count in sorted(
                        failure_counts.items(),
                        key=lambda item: (-item[1], item[0]))
                },
                "top_rejections": top,
                "elapsed_seconds": elapsed,
                "rate": rate,
                "eta_seconds": eta,
                "grouped": grouped,
            })
            progress_last_at = now
            progress_last_attempts = completed

        def accepted_records():
            nonlocal emitted
            for outcome in outcomes:
                if outcome.get("fatal"):
                    _discard_group_outcome(outcome)
                    record_rejection(outcome)
                    case = outcome.get("case") or {}
                    raise RuntimeError(
                        f"{outcome.get('code', 'generator_error')} for "
                        f"world seed {case.get('seed', 'unknown')}: "
                        f"{outcome['reason']}")
                if grouped:
                    world_stats["worlds_attempted"] += 1
                    stats["attempted"] += self.samples_per_world
                    world_stats["samples_executed"] += int(
                        outcome.get("attempted", 0))
                    world = int(outcome["case"]["seed"])
                    if world in accepted_worlds:
                        _discard_group_outcome(outcome)
                        outcome = {**outcome, "ok": False,
                                   "code": "duplicate_world",
                                   "reason": (f"world seed {world} was "
                                              "already accepted")}
                    if outcome["ok"]:
                        value_count, values = _group_outcome_values(outcome)
                        if value_count != self.samples_per_world:
                            _discard_group_outcome(outcome)
                            close_values = getattr(values, "close", None)
                            if close_values is not None:
                                close_values()
                            raise RuntimeError(
                                f"executor returned {value_count} samples "
                                f"for world {world}; expected "
                                f"{self.samples_per_world}")
                        accepted_worlds.add(world)
                        world_stats["worlds_accepted"] += 1
                        stats["accepted"] += value_count
                        try:
                            for value in values:
                                emitted += 1
                                yield value
                        finally:
                            close_values = getattr(values, "close", None)
                            if close_values is not None:
                                close_values()
                        if world_stats["worlds_accepted"] == requested_worlds:
                            report_progress(force=True)
                            return
                        report_progress()
                        continue
                    _discard_group_outcome(outcome)
                    world_stats["worlds_rejected"] += 1
                    # A failed sibling rejects the complete world group.
                    stats["rejected"] += self.samples_per_world
                    code = outcome.get("code", "case_rejected")
                    reason = outcome["reason"]
                    record_rejection(outcome)
                    failure_counts[code] = failure_counts.get(code, 0) + 1
                    failure_examples.setdefault(code, reason)
                    if outcome.get("details"):
                        failure_details.setdefault(code, outcome["details"])
                    report_progress()
                    continue
                stats["attempted"] += 1
                if outcome["ok"]:
                    world = int(outcome["case"]["seed"])
                    if self.unique_worlds and world in accepted_worlds:
                        stats["rejected"] += 1
                        code = "duplicate_world"
                        duplicate = {
                            **outcome, "ok": False, "code": code,
                            "reason": f"world seed {world} was already accepted",
                        }
                        record_rejection(duplicate)
                        failure_counts[code] = failure_counts.get(code, 0) + 1
                        failure_examples.setdefault(
                            code, f"world seed {world} was already accepted")
                        continue
                    accepted_worlds.add(world)
                    stats["accepted"] += 1
                    emitted += 1
                    yield outcome["value"]
                    if stats["accepted"] == self.count:
                        report_progress(force=True)
                        return
                    report_progress()
                    continue
                stats["rejected"] += 1
                code = outcome.get("code", "case_rejected")
                reason = outcome["reason"]
                record_rejection(outcome)
                failure_counts[code] = failure_counts.get(code, 0) + 1
                failure_examples.setdefault(code, reason)
                if outcome.get("details"):
                    failure_details.setdefault(code, outcome["details"])
                report_progress()
            complete = (world_stats["worlds_accepted"] >= requested_worlds
                        if grouped else stats["accepted"] >= self.count)
            if complete:
                return
            shown = "; ".join(
                f"{number}x {code}: {failure_examples[code]}"
                for code, number in
                sorted(failure_counts.items(),
                       key=lambda item: -item[1])[:5])
            made = (f"{world_stats['worlds_accepted']} complete worlds / "
                    f"{stats['accepted']} records" if grouped else
                    f"{stats['accepted']} usable records")
            wanted = (f"{requested_worlds} worlds x "
                      f"{self.samples_per_world} samples" if grouped else
                      str(self.count))
            raise RuntimeError(f"generated {made}, wanted {wanted} after "
                               f"{attempts} attempts"
                               + (f": {shown}" if shown else ""))
        records = None
        try:
            writer = self.writer.create(
                output=self.output, base_dir=self.base_dir,
                metadata=dict(self.metadata))
            records = accepted_records()
            written = writer.write(records)
            if emitted != self.count:
                raise RuntimeError(
                    f"writer {self.writer.type!r} stopped after consuming "
                    f"{emitted} of {self.count} accepted records; a writer "
                    "must exhaust its input before reporting success")
        finally:
            if records is not None:
                records.close()
            close = getattr(outcomes, "close", None)
            if close is not None:
                close()
            if rejection_stream is not None:
                rejection_stream.flush()
                os.fsync(rejection_stream.fileno())
                rejection_stream.close()
                os.replace(rejection_temporary, self.rejections_output)
        ranked_failures = sorted(
            failure_counts.items(), key=lambda item: (-item[1], item[0]))
        result = {"schema": "bedrock.generation.v1", "output": written,
                "requested": self.count, **stats,
                "acceptance_rate": (stats["accepted"] / stats["attempted"]
                                    if stats["attempted"] else 0.0),
                "rejection_reasons": [
                    {"code": code, "count": count,
                     "reason": failure_examples[code],
                     **({"details": failure_details[code]}
                        if code in failure_details else {})}
                    for code, count in ranked_failures
                ],
                "unique_worlds": self.unique_worlds,
                "seed": self.seed,
                "components": {"sampler": self.sampler.type,
                               "generator": self.generator.type,
                               "executor": self.executor.type,
                               "writer": self.writer.type}}
        if self.rejections_output is not None:
            result["rejections_output"] = self.rejections_output
        if grouped:
            result.update(world_stats)
            result["worlds"] = requested_worlds
            result["samples_per_world"] = self.samples_per_world
            result["world_acceptance_rate"] = (
                world_stats["worlds_accepted"] /
                world_stats["worlds_attempted"]
                if world_stats["worlds_attempted"] else 0.0)
        return result
