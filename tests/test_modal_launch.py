"""Modal is an optional launch adapter, not a second training schema."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from bedrock_rl.launch.modal import (
    ModalLaunchError,
    build_launch,
    build_scout_launch,
    modal_cpu_dockerfile,
    modal_gpu_dockerfile,
    normalize_gpu,
    task_turn_budget,
)


def _load_sdg_app():
    modal = ModuleType("modal")

    class App:
        def __init__(self, name):
            self.name = name

        def function(self, **kwargs):
            del kwargs
            return lambda function: function

        def local_entrypoint(self):
            return lambda function: function

    class Dict:
        @staticmethod
        def from_name(name, create_if_missing=False):
            del name, create_if_missing
            return {}

    class Image:
        @staticmethod
        def debian_slim():
            return object()

    modal.App = App
    modal.Dict = Dict
    modal.Image = Image
    modal.is_local = lambda: False
    path = (Path(__file__).parents[1] / "bedrock_rl" / "launch" /
            "sdg_app.py")
    spec = importlib.util.spec_from_file_location(
        "_bedrock_test_sdg_app", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"modal": modal}):
        spec.loader.exec_module(module)
    return module


class ModalLaunchTests(unittest.TestCase):
    def test_scout_identity_includes_task_and_source_bytes(self):
        module = _load_sdg_app()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bedrock_rl").mkdir()
            (root / "examples").mkdir()
            config = root / "examples" / "data.yaml"
            task = root / "examples" / "task.yaml"
            source = root / "bedrock_rl" / "policy.py"
            config.write_text("worlds: 1\n")
            task.write_text("name: task\n")
            source.write_text("VERSION = 1\n")
            with (patch.object(module, "REMOTE_ROOT", root),
                  patch.object(module, "CONFIG", config)):
                first = module._configuration_identity()
                task.write_text("name: changed\n")
                second = module._configuration_identity()
                source.write_text("VERSION = 2\n")
                third = module._configuration_identity()

        self.assertNotEqual(first, second)
        self.assertNotEqual(second, third)

    def test_scout_drains_completed_rows_after_one_worker_failure(self):
        module = _load_sdg_app()
        module.results = {}
        module._configuration_identity = lambda: "identity"
        module.uuid4 = lambda: SimpleNamespace(hex="attempt")

        class Case:
            def __init__(self, ordinal):
                self.ordinal = ordinal

            def to_dict(self):
                return {"index": self.ordinal, "seed": self.ordinal,
                        "decision_seed": self.ordinal}

        class Sampler:
            def sample_worlds(self, count, samples_per_world, seed):
                del samples_per_world, seed
                return ((Case(index),) for index in range(count))

        module._components = lambda with_generator=False: (
            SimpleNamespace(worlds=3, samples_per_world=1, seed=0),
            Sampler(), None)
        invocation = "invocation"
        module._invocation_id = lambda run_id, identity=None: invocation
        module.results[module._attempt_key(
            "run", invocation, "attempt", 0)] = {
                "ok": False, "fatal": True, "ordinal": 0,
                "code": "generator_error", "reason": "boom"}
        module.results[module._attempt_key(
            "run", invocation, "attempt", 1)] = {
                "ok": True, "ordinal": 1,
                "accepted": [{"case": {"seed": 1}}]}
        module.results[module._attempt_key(
            "run", invocation, "attempt", 2)] = {
                "ok": False, "ordinal": 2,
                "code": "policy_veto", "reason": "no route"}

        calls = SimpleNamespace(cancelled=False)
        calls.cancel = lambda: setattr(calls, "cancelled", True)
        module.scout = SimpleNamespace(
            experimental_spawn_map=lambda *streams: calls)

        manifest = module.coordinate("run", invocation, 0, 3, 2)

        self.assertFalse(manifest["complete"])
        self.assertEqual(manifest["fatal_count"], 1)
        self.assertEqual(manifest["infrastructure_failure_count"], 1)
        self.assertIn(module._result_key("run", invocation, 1),
                      module.results)
        self.assertIn(module._result_key("run", invocation, 2),
                      module.results)
        self.assertTrue(calls.cancelled)

    def test_scout_launch_reports_the_config_scoped_namespace(self):
        module = _load_sdg_app()
        module._configuration_identity = lambda root, config: "identity"
        module._invocation_id = lambda run_id, identity=None: (
            f"{run_id}-{identity}")
        call = SimpleNamespace(object_id="call-1")
        module.coordinate = SimpleNamespace(spawn=lambda *args: call)
        module.ROOT = Path("/checkout")
        module.REMOTE_ROOT = Path("/opt/bedrock-rl")
        module.CONFIG = Path("/opt/bedrock-rl/examples/data.yaml")

        output = io.StringIO()
        with redirect_stdout(output):
            module.main(run_id="run")

        launch = json.loads(output.getvalue())
        self.assertEqual(launch["invocation_id"], "run-identity")

    def test_scout_refuses_local_remote_identity_drift(self):
        module = _load_sdg_app()
        module._configuration_identity = lambda: "remote"
        with self.assertRaisesRegex(ValueError, "local and remote"):
            module.coordinate("run", "run-local", 0, 1, 1)

    def test_modal_build_selects_gpu_without_changing_docker_default(self):
        source = "ARG FINAL_STAGE=cpu\nFROM ${FINAL_STAGE} AS default\n"
        rendered = modal_gpu_dockerfile(source)
        self.assertIn("ARG FINAL_STAGE=gpu", rendered)
        self.assertNotIn("ARG FINAL_STAGE=cpu", rendered)

    def test_modal_cpu_build_drops_the_cuda_training_stage(self):
        source = (
            "FROM base AS cpu\n"
            "# ── gpu: the training stack on top\n"
            "FROM cuda AS gpu\n"
            "FROM ${FINAL_STAGE} AS default\n"
        )
        rendered = modal_cpu_dockerfile(source)
        self.assertIn("FROM base AS cpu", rendered)
        self.assertIn("FROM cpu AS default", rendered)
        self.assertNotIn("FROM cuda AS gpu", rendered)

    def test_modal_cpu_build_requires_one_documented_stage_boundary(self):
        with self.assertRaisesRegex(ModalLaunchError, "exactly one"):
            modal_cpu_dockerfile("FROM base AS cpu\n")

    def test_sdg_scout_launch_is_cpu_only_detached_and_durable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            config = root / "data.yaml"
            config.write_text("worlds: 500\n")
            launch = build_scout_launch(
                root=root, config=config, job=None,
                sets=["worlds=550"], target=550,
                max_candidates=150000, containers=1000,
                results="scout-results", run_id="log-farm-500")

        self.assertEqual(launch.config, "data.yaml")
        self.assertEqual(launch.environment()["BRL_MODAL_SDG_CONTAINERS"],
                         "1000")
        self.assertEqual(launch.environment()["BRL_MODAL_SDG_SETS"],
                         '["worlds=550"]')
        self.assertEqual(launch.command()[:3], ["modal", "run", "--detach"])
        self.assertIn("bedrock_rl.launch.sdg_app", launch.command())
        self.assertNotIn("--gpu", launch.command())

    def test_sdg_scout_launch_bounds_target_attempts_and_containers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            config = root / "data.yaml"
            config.write_text("worlds: 1\n")
            kwargs = dict(
                root=root, config=config, job=None, sets=(), target=2,
                max_candidates=1, containers=1, results="results")
            with self.assertRaisesRegex(ModalLaunchError, "cannot be smaller"):
                build_scout_launch(**kwargs)
            kwargs.update(max_candidates=2, containers=0)
            with self.assertRaisesRegex(ModalLaunchError, "between 1 and"):
                build_scout_launch(**kwargs)

    def test_single_node_defaults_to_one_gpu(self):
        self.assertEqual(normalize_gpu(None, 1), ("H100", 1))

    def test_multi_node_expands_to_full_host(self):
        self.assertEqual(normalize_gpu("H100", 2), ("H100:8", 8))

    def test_partial_multi_node_host_is_refused(self):
        with self.assertRaisesRegex(ModalLaunchError, "full H100 host"):
            normalize_gpu("H100:2", 2)

    def test_cuda_13_gpu_families_are_refused_by_cuda_12_image(self):
        for gpu in ("B300", "B200+"):
            with self.subTest(gpu=gpu), self.assertRaisesRegex(
                    ModalLaunchError, "CUDA 12.8"):
                normalize_gpu(gpu, 1)
        self.assertEqual(normalize_gpu("B200", 1), ("B200", 1))

    def make_launch(self, target="rl", nodes=2, config_inside=True):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name) / "repo"
        root.mkdir()
        config = (root if config_inside else Path(temp.name)) / "train.yaml"
        config.write_text("trainer: rl\n")
        return build_launch(
            root=root,
            config=config,
            preset=None,
            target=target,
            nodes=nodes,
            gpu="H100",
            volume="bedrock-rl-runs",
            secrets=["huggingface"],
            sets=["steps=2"],
            train_env=["WANDB=1"],
            timeout=3600,
            rdma=True,
            smoke=True,
            detach=False,
        )

    def test_same_config_is_forwarded_to_modal_module(self):
        launch = self.make_launch()
        self.assertEqual(launch.config, "train.yaml")
        self.assertEqual(launch.gpu, "H100:8")
        self.assertEqual(launch.gpus_per_node, 8)
        self.assertIn("bedrock_rl.launch.training_app", launch.command())
        self.assertIn("--smoke", launch.command())
        self.assertEqual(launch.environment()["BRL_MODAL_SETS"],
                         '["steps=2"]')
        self.assertEqual(launch.environment()["BRL_MODAL_MODE"], "smoke")

    def test_modal_run_id_reaches_remote_entrypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            config = root / "train.yaml"
            config.write_text("trainer: sft\n")
            launch = build_launch(
                root=root, config=config, preset=None, target="sft",
                nodes=1, gpu="H100", volume="runs", secrets=(), sets=(),
                train_env=(), timeout=3600, rdma=False, smoke=False,
                detach=True, run_id="farm-log-17")

        self.assertEqual(launch.command()[-2:], ["--run-id", "farm-log-17"])

    def test_modal_run_id_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            config = root / "train.yaml"
            config.write_text("trainer: sft\n")
            with self.assertRaisesRegex(ModalLaunchError, "modal-run"):
                build_launch(
                    root=root, config=config, preset=None, target="sft",
                    nodes=1, gpu="H100", volume="runs", secrets=(), sets=(),
                    train_env=(), timeout=3600, rdma=False, smoke=False,
                    detach=True, run_id="../escape")

    def test_multi_node_is_only_exposed_for_distributed_trainer(self):
        with self.assertRaisesRegex(ModalLaunchError, "rl/opd/mopd"):
            self.make_launch(target="sft")

    def test_multi_node_opd_and_mopd_use_verl_distributed_launcher(self):
        self.assertEqual(self.make_launch(target="opd").nodes, 2)
        self.assertEqual(self.make_launch(target="mopd").nodes, 2)

    def test_config_must_be_in_image_context(self):
        with self.assertRaisesRegex(ModalLaunchError, "inside this checkout"):
            self.make_launch(config_inside=False)

    def test_modal_prefetch_reads_turn_budget_from_task_not_removed_knob(self):
        root = Path(__file__).parents[1]
        task = root / "examples" / "equip_pickaxe_grpo" / "task.yaml"
        self.assertEqual(task_turn_budget(str(task), root), 1)
        source = (root / "bedrock_rl" / "launch" /
                  "training_app.py").read_text()
        self.assertIn("max_turns=task_turn_budget(plan.task, base)", source)
        self.assertNotIn('plan.env.get("MAX_TURNS"', source)

    def test_modal_prefetch_stages_distillation_teachers(self):
        root = Path(__file__).parents[1]
        source = (root / "bedrock_rl" / "launch" /
                  "training_app.py").read_text()
        self.assertIn('if plan.target in {"opd", "mopd"}', source)
        self.assertIn("stage_teachers(plan.env", source)

    def test_modal_resume_uses_a_launcher_input_model_resolution_cannot_reset(self):
        root = Path(__file__).parents[1]
        app_source = (root / "bedrock_rl" / "launch" /
                      "training_app.py").read_text()
        launcher_source = (root / "bin" / "rl.sh").read_text()
        self.assertIn('"BRL_RESUME_MODE=auto"', app_source)
        self.assertIn('trainer.resume_mode="${BRL_RESUME_MODE:-auto}"',
                      launcher_source)
        self.assertNotIn(
            '"BRL_VERL_OVERRIDES=trainer.resume_mode=auto"', app_source)

    def test_modal_injects_verl_only_knobs_only_for_verl_targets(self):
        root = Path(__file__).parents[1]
        source = (root / "bedrock_rl" / "launch" /
                  "training_app.py").read_text()
        self.assertIn('if target in {"rl", "opd", "mopd"}:', source)
        conditional = source.split(
            'if target in {"rl", "opd", "mopd"}:', 1)[1]
        self.assertIn('f"nnodes={NODES}"', conditional)
        self.assertIn('f"gpus_per_node={GPUS_PER_NODE}"', conditional)
        self.assertIn('f"RAY_ADDRESS={head}:6379"', conditional)


if __name__ == "__main__":
    unittest.main()
