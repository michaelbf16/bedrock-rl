"""Modal application for ``bedrock train CONFIG --modal``.

The local CLI supplies resource choices through ``BRL_MODAL_*`` variables
before Modal imports this module.  They are launch settings rather than a
fourth YAML file: the remote leader ultimately invokes the ordinary
``bedrock train`` command with the same config.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import modal
import modal.experimental

from bedrock_rl.launch.modal import modal_gpu_dockerfile, task_turn_budget


ROOT = Path(__file__).resolve().parents[2]
REMOTE_ROOT = Path("/opt/bedrock-rl")
MOUNT = Path("/mnt/bedrock")
NODES = int(os.environ.get("BRL_MODAL_NODES", "1"))
GPU = os.environ.get("BRL_MODAL_GPU", "H100")
GPUS_PER_NODE = int(os.environ.get("BRL_MODAL_GPUS_PER_NODE", "1"))
TIMEOUT = int(os.environ.get("BRL_MODAL_TIMEOUT", "86400"))
RDMA = os.environ.get("BRL_MODAL_RDMA", "0") == "1" and NODES > 1
SMOKE = os.environ.get("BRL_MODAL_MODE", "train") == "smoke"
VOLUME_NAME = os.environ.get("BRL_MODAL_VOLUME", "bedrock-rl-runs")
SECRET_NAMES = tuple(
    name.strip() for name in os.environ.get("BRL_MODAL_SECRETS", "").split(",")
    if name.strip()
)
SETS = tuple(json.loads(os.environ.get("BRL_MODAL_SETS", "[]")))
TRAIN_ENV = tuple(json.loads(os.environ.get("BRL_MODAL_TRAIN_ENV", "[]")))
REMOTE_ENV = {
    key: value for key, value in os.environ.items()
    if key.startswith("BRL_MODAL_")
}

app = modal.App("bedrock-rl")

# Register only the selected mode because Modal builds every image referenced
# by an app. Smoke tests use a minimal image; training uses the project's GPU
# Docker target with the pinned Netherite engine and verl runtime.
if SMOKE:
    if modal.is_local():
        smoke_image = (
            modal.Image.debian_slim(python_version="3.12")
            .uv_pip_install("torch==2.9.0", "ray[default]", "numpy")
            .add_local_python_source("bedrock_rl")
        )
    else:
        # Function metadata already names the built image. The placeholder
        # only lets decorators re-run when Modal imports this mounted module.
        smoke_image = modal.Image.debian_slim()
else:
    if modal.is_local():
        pin = (
            ROOT / "patches" / "netherite" / "PINNED_SHA"
        ).read_text().strip()
        _dockerfile_source = modal_gpu_dockerfile(
            (ROOT / "Dockerfile").read_text())
        _dockerfile_digest = hashlib.sha256(
            _dockerfile_source.encode()).hexdigest()[:12]
        _dockerfile = Path(tempfile.gettempdir()) / (
            f"bedrock-rl-modal-{_dockerfile_digest}.Dockerfile")
        _dockerfile.write_text(_dockerfile_source)
        train_image = modal.Image.from_dockerfile(
            _dockerfile,
            context_dir=ROOT,
            build_args={"NETHERITE_SHA": pin},
        )
        if (ROOT / "data").is_dir():
            train_image = train_image.add_local_dir(
                ROOT / "data", remote_path=str(REMOTE_ROOT / "data"),
                copy=True)
    else:
        train_image = modal.Image.debian_slim()
    volume = modal.Volume.from_name(
        VOLUME_NAME, create_if_missing=True, version=2)
    secrets = [modal.Secret.from_name(name) for name in SECRET_NAMES]


def _clustered(function):
    if NODES == 1:
        return function
    return modal.experimental.clustered(size=NODES, rdma=RDMA)(function)


def _cluster_info():
    if NODES == 1:
        ip = socket.gethostbyname(socket.gethostname())
        return 0, [ip], "single-node"
    info = modal.experimental.get_cluster_info()
    # Ray currently has a substantially better-tested IPv4 control plane.
    # Modal exposes private IPv4 addresses for exactly this case.
    return info.rank, list(info.container_ipv4_ips), info.cluster_id


def _wait_port(host: str, port: int, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"timed out waiting for {host}:{port}")


def _send(stream, value) -> None:
    stream.write(json.dumps(value, sort_keys=True) + "\n")
    stream.flush()


def _recv(stream):
    line = stream.readline()
    if not line:
        raise RuntimeError("a Modal cluster peer disconnected")
    return json.loads(line)


def _leader_peers(head: str, port: int = 17391):
    if NODES == 1:
        return None, {}
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((head, port))
    server.listen(NODES - 1)
    server.settimeout(300)
    peers = {}
    while len(peers) < NODES - 1:
        connection, _ = server.accept()
        stream = connection.makefile("rw", encoding="utf-8", newline="\n")
        hello = _recv(stream)
        peers[int(hello["rank"])] = (connection, stream)
    return server, peers


def _worker_peer(rank: int, head: str, port: int = 17391):
    deadline = time.monotonic() + 300
    while True:
        try:
            connection = socket.create_connection((head, port), timeout=5)
            break
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.5)
    # The timeout above bounds only the connection attempt. A worker may spend
    # hours inside training after it joins the leader's control stream.
    connection.settimeout(None)
    stream = connection.makefile("rw", encoding="utf-8", newline="\n")
    _send(stream, {"rank": rank})
    return connection, stream


def _broadcast(peers, value) -> None:
    for _, stream in peers.values():
        _send(stream, value)


def _responses(peers, status: str):
    values = []
    for rank, (_, stream) in sorted(peers.items()):
        value = _recv(stream)
        if value.get("status") != status:
            raise RuntimeError(f"rank {rank} returned {value}, wanted {status}")
        values.append(value)
    return values


def _drain_until(peers, status: str):
    """Read through a prior reply while shutting down after leader errors."""
    values = []
    for rank, (_, stream) in sorted(peers.items()):
        while True:
            value = _recv(stream)
            if value.get("status") == status:
                values.append(value)
                break
            if value.get("status") not in {
                "reloaded", "collective", "ray_ready",
            }:
                raise RuntimeError(
                    f"rank {rank} returned {value}, wanted {status}")
    return values


def _json_result(result: subprocess.CompletedProcess, label: str) -> dict:
    """Find our final JSON object around stdout emitted by CUDA/Ray libs."""
    for line in reversed(result.stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError(
        f"{label} emitted no JSON result\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}")


def _ray_binary() -> str:
    candidates = [shutil.which("ray"), "/opt/verl-env/bin/ray"]
    for path in candidates:
        if path and os.access(path, os.X_OK):
            return path
    raise RuntimeError("Ray executable is missing from the Modal image")


def _ray_python() -> str:
    path = "/opt/verl-env/bin/python"
    return path if os.access(path, os.X_OK) else sys.executable


def _set_visible_devices() -> str:
    """Normalize Modal's full-host exposure for Ray and the shell launchers."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                text=True,
            )
            visible = ",".join(
                line.strip() for line in output.splitlines() if line.strip())
        except (OSError, subprocess.CalledProcessError):
            import torch
            visible = ",".join(str(i) for i in range(torch.cuda.device_count()))
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    count = len([item for item in visible.split(",") if item.strip()])
    if count != GPUS_PER_NODE:
        raise RuntimeError(
            f"Modal exposed CUDA_VISIBLE_DEVICES={visible!r}, expected "
            f"{GPUS_PER_NODE} devices from {GPU}")
    return visible


def _start_ray(rank: int, ips: list[str]) -> None:
    ray = _ray_binary()
    common = [
        ray,
        "start",
        f"--node-ip-address={ips[rank]}",
        f"--num-gpus={GPUS_PER_NODE}",
        f"--num-cpus={max(4, GPUS_PER_NODE * 4)}",
        "--disable-usage-stats",
    ]
    if rank == 0:
        command = common + ["--head", "--port=6379"]
    else:
        _wait_port(ips[0], 6379)
        command = common + [f"--address={ips[0]}:6379"]
    subprocess.run(command, check=True)


def _stop_ray() -> None:
    subprocess.run([_ray_binary(), "stop", "--force"], check=False)


_COLLECTIVE = r"""
import json, os
import torch
import torch.distributed as dist
rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(0)
dist.init_process_group(
    "nccl",
    init_method="tcp://%s:%s" % (os.environ["MASTER_ADDR"],
                                  os.environ["MASTER_PORT"]),
    rank=rank,
    world_size=world,
)
value = torch.tensor([rank + 1.0], device="cuda")
dist.all_reduce(value)
expected = world * (world + 1) / 2
assert value.item() == expected, (value.item(), expected)
print(json.dumps({
    "rank": rank,
    "sum": value.item(),
    "cuda": torch.cuda.get_device_name(0),
    "visible": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
}))
dist.destroy_process_group()
"""


def _collective(rank: int, head: str) -> dict:
    env = dict(os.environ)
    env.update({
        "RANK": str(rank),
        "WORLD_SIZE": str(NODES),
        "MASTER_ADDR": head,
        "MASTER_PORT": "29571",
        "NCCL_DEBUG": "WARN",
    })
    result = subprocess.run(
        [sys.executable, "-c", _COLLECTIVE],
        env=env,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"NCCL collective failed with {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return _json_result(result, "NCCL collective")


_RAY_REPORT = r"""
import json, os, socket, sys, time
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
address, expected_nodes, expected_gpus = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ray.init(address=address)
deadline = time.monotonic() + 180
while True:
    nodes = [node for node in ray.nodes() if node.get("Alive")]
    if len(nodes) >= expected_nodes:
        break
    if time.monotonic() >= deadline:
        raise RuntimeError("Ray saw %d/%d nodes" % (len(nodes), expected_nodes))
    time.sleep(1)
@ray.remote(num_gpus=1)
def probe():
    import torch
    import ray
    from ray.util import get_node_ip_address
    return {
        "host": socket.gethostname(),
        "ip": get_node_ip_address(),
        "node_id": ray.get_runtime_context().get_node_id(),
        "visible": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cuda": torch.cuda.get_device_name(0),
        "cuda_ok": torch.cuda.is_available(),
    }
refs = []
for node in nodes:
    refs.append(probe.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
        node["NodeID"], soft=False)).remote())
probes = ray.get(refs)
resources = ray.cluster_resources()
assert int(resources.get("GPU", 0)) == expected_nodes * expected_gpus, resources
assert len({p["node_id"] for p in probes}) == expected_nodes, probes
assert len({p["ip"] for p in probes}) == expected_nodes, probes
assert all(p["cuda_ok"] for p in probes), probes
print(json.dumps({"nodes": len(nodes), "resources": resources, "probes": probes}))
ray.shutdown()
"""


def _ray_report(head: str) -> dict:
    result = subprocess.run(
        [_ray_python(), "-c", _RAY_REPORT, f"{head}:6379", str(NODES),
         str(GPUS_PER_NODE)],
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"Ray probe failed with {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return _json_result(result, "Ray probe")


def _remote_config(config: str) -> Path:
    path = (REMOTE_ROOT / config).resolve()
    if REMOTE_ROOT not in path.parents or not path.is_file():
        raise ValueError(f"training config is not in the image: {path}")
    return path


def _training_plan(config: str, preset: str):
    from bedrock_rl.config import training as train_config

    path = _remote_config(config)
    return train_config.train_plan(
        train_config.load(path, preset or None),
        sets=SETS,
        resolve=lambda value: str((path.parent / str(value)).resolve()),
        origin=config + (f":{preset}" if preset else ""),
    )


def _prefetch_model(config: str, preset: str) -> str:
    """Put pinned policy weights on the shared volume before workers load."""
    from bedrock_rl import models
    from bedrock_rl.config.paths import DEFAULT_MODEL, model_arg

    path = _remote_config(config)
    base = str(path.parent)
    plan = _training_plan(config, preset)
    model = model_arg(plan.model or DEFAULT_MODEL, base_dir=base)
    # The launcher's later --snapshot resolves to this same deterministic
    # directory, but doing it before workers join lets every node reload the
    # committed volume and see the identical path.
    block = models.shell_env(
        model,
        max_turns=task_turn_budget(plan.task, base),
        snapshot=True,
    )
    model_path = None
    for line in block.splitlines():
        if line.startswith("BRL_MODEL_PATH="):
            model_path = shlex.split(line)[0].split("=", 1)[1]
            break
    if model_path is None:
        raise RuntimeError("model resolver did not emit BRL_MODEL_PATH")
    if plan.target in {"opd", "mopd"}:
        from bedrock_rl.adapters.verl.distillation import stage_teachers
        stage_teachers(plan.env["BRL_TEACHERS_JSON"], plan.target)
    return model_path


def _train_command(config: str, preset: str, run_id: str, head: str,
                   target: str, *, resume_configured: bool = False):
    run_dir = MOUNT / "runs" / run_id
    command = ["bedrock", "train", str(_remote_config(config))]
    if preset:
        command.append(preset)
    for value in SETS:
        command += ["--set", value]
    for value in TRAIN_ENV:
        command += ["-e", value]
    command += [
        "--set", "cuda_visible_devices=" + os.environ["CUDA_VISIBLE_DEVICES"],
        "-e", f"OUT={run_dir / 'checkpoints'}",
        "-e", f"METRICS={run_dir / 'metrics.jsonl'}",
    ]
    # Both auto modes start fresh when no complete checkpoint exists. Passing
    # them on the first launch as well as a relaunch removes a filesystem race
    # between checking the mounted Volume and the trainer opening it.
    if target == "sft" and not resume_configured:
        command += ["--set", "resume=auto"]
    elif target in {"rl", "opd", "mopd"} and not resume_configured:
        command += ["-e", "BRL_RESUME_MODE=auto"]
    if target in {"rl", "opd", "mopd"}:
        command += [
            "--set", f"nnodes={NODES}",
            "--set", f"gpus_per_node={GPUS_PER_NODE}",
            "-e", f"RAY_ADDRESS={head}:6379",
            "-e", f"BRL_TRAJECTORY_DIR={run_dir / 'trajectories'}",
        ]
    return command, run_dir


def _worker(rank: int, ips: list[str], use_volume: bool):
    connection, stream = _worker_peer(rank, ips[0])
    ray_started = False
    try:
        while True:
            message = _recv(stream)
            command = message["command"]
            if command == "reload":
                volume.reload()
                path = message["model_path"]
                if not os.path.isdir(path):
                    raise RuntimeError(
                        f"rank {rank} cannot see staged model {path}")
                _send(stream, {"status": "reloaded", "rank": rank})
            elif command == "collective":
                result = _collective(rank, ips[0])
                _send(stream, {"status": "collective", "result": result})
            elif command == "ray_start":
                _start_ray(rank, ips)
                ray_started = True
                _send(stream, {"status": "ray_ready", "rank": rank})
            elif command == "stop":
                if use_volume:
                    volume.commit()
                _send(stream, {"status": "stopped", "rank": rank})
                break
            else:
                raise RuntimeError(f"unknown cluster command {command!r}")
    finally:
        if ray_started:
            _stop_ray()
        stream.close()
        connection.close()


def _leader(config: str, preset: str, run_id: str, smoke: bool,
            ips: list[str], cluster_id: str, model_path: str | None = None):
    server, peers = _leader_peers(ips[0])
    ray_started = False
    try:
        if not smoke:
            os.environ["HF_HOME"] = str(MOUNT / "huggingface")
            volume.reload()
            if not model_path or not os.path.isdir(model_path):
                raise RuntimeError(
                    f"prefetched model is missing from the Volume: "
                    f"{model_path}")
            _broadcast(peers, {"command": "reload", "model_path": model_path})
            _responses(peers, "reloaded")

        collectives = []
        if smoke:
            _broadcast(peers, {"command": "collective"})
            collectives.append(_collective(0, ips[0]))
            collectives.extend(
                value["result"] for value in _responses(peers, "collective"))

        _start_ray(0, ips)
        ray_started = True
        _broadcast(peers, {"command": "ray_start"})
        _responses(peers, "ray_ready")
        report = _ray_report(ips[0])

        result = {
            "ok": True,
            "cluster_id": cluster_id,
            "nodes": NODES,
            "gpu": GPU,
            "gpus_per_node": GPUS_PER_NODE,
            "rdma": RDMA,
            "ray": report,
        }
        if smoke:
            result["nccl"] = sorted(collectives, key=lambda item: item["rank"])
            print("BRL_MODAL_RESULT=" + json.dumps(result, sort_keys=True))
            return result

        plan = _training_plan(config, preset)
        target = plan.target
        resume_key = "RESUME" if target == "sft" else "BRL_RESUME_MODE"
        resume_configured = bool(plan.env.get(resume_key)) or any(
            value.split("=", 1)[0] == resume_key for value in TRAIN_ENV)
        command, run_dir = _train_command(
            config, preset, run_id, ips[0], target,
            resume_configured=resume_configured)
        result.update({
            "run_id": run_id,
            "volume": VOLUME_NAME,
            "output": str(run_dir),
            "model_path": model_path,
        })
        try:
            subprocess.run(command, cwd=REMOTE_ROOT, check=True)
        except BaseException:
            # Checkpoints and metrics written before a trainer failure are
            # still the expensive output of the run. Make them durable before
            # preserving the original exception for Modal.
            try:
                volume.commit()
            except Exception as commit_error:
                print(f"warning: failed to commit training volume after "
                      f"trainer error: {commit_error}", file=sys.stderr)
            raise
        volume.commit()
        print("BRL_MODAL_RESULT=" + json.dumps(result, sort_keys=True))
        return result
    finally:
        if ray_started:
            _stop_ray()
        _broadcast(peers, {"command": "stop"})
        if peers:
            _drain_until(peers, "stopped")
        for connection, stream in peers.values():
            stream.close()
            connection.close()
        if server is not None:
            server.close()


def _run(config: str, preset: str, run_id: str, smoke: bool,
         model_path: str | None = None):
    rank, ips, cluster_id = _cluster_info()
    _set_visible_devices()
    if rank:
        return _worker(rank, ips, use_volume=not smoke)
    return _leader(
        config, preset, run_id, smoke, ips, cluster_id, model_path=model_path)


if SMOKE:
    @app.function(
        image=smoke_image,
        gpu=GPU,
        timeout=TIMEOUT,
        env=REMOTE_ENV,
    )
    @_clustered
    def cluster_smoke(config: str, preset: str, run_id: str):
        return _run(config, preset, run_id, smoke=True)
else:
    @app.function(
        image=train_image,
        timeout=TIMEOUT,
        env=REMOTE_ENV,
        secrets=secrets,
        volumes={MOUNT: volume},
    )
    def prefetch(config: str, preset: str):
        os.environ["HF_HOME"] = str(MOUNT / "huggingface")
        model_path = _prefetch_model(config, preset)
        volume.commit()
        return model_path

    @app.function(
        image=train_image,
        gpu=GPU,
        timeout=TIMEOUT,
        env=REMOTE_ENV,
        secrets=secrets,
        volumes={MOUNT: volume},
    )
    @_clustered
    def train(config: str, preset: str, run_id: str, model_path: str):
        os.environ["HF_HOME"] = str(MOUNT / "huggingface")
        return _run(
            config, preset, run_id, smoke=False, model_path=model_path)


@app.local_entrypoint()
def main(config: str, preset: str = "", smoke: bool = False,
         run_id: str = ""):
    if smoke != SMOKE:
        raise RuntimeError("Modal launch mode did not reach the remote app")
    run_id = run_id or "%s-%s" % (
        Path(config).stem,
        time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
    )
    if SMOKE:
        result = cluster_smoke.remote(config, preset, run_id)
    else:
        model_path = prefetch.remote(config, preset)
        result = train.remote(config, preset, run_id, model_path)
    print(json.dumps(result, indent=2, sort_keys=True))
