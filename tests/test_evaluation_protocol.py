import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import yaml

from bedrock_rl.env.task import Task
from bedrock_rl.eval.protocol import (digest, environment_identity,
                                     processor_identity)
from bedrock_rl.eval.summary import EvaluationError, PointSpec, analyze
from bedrock_rl.eval.trials import evaluation_protocol, evaluation_task
from bedrock_rl.eval import trials


DOCUMENT = {"name": "pickaxe", "goal": "Select the pickaxe", "max_turns": 2,
            "success": {"type": "selected_item", "items": ["iron_pickaxe"]}}
ARGS = SimpleNamespace(hint_level="procedure", finish_level="off",
                       keep_latest_images=0, blank_frames=False,
                       max_model_len=8192, tp=1)


def frozen_row(task, *, seed=1):
    return {"prompt": [{"role": "user", "content": "<image>\nSelect it"}],
            "images": [Image.new("RGB", (2, 2), "black")],
            "extra_info": {"task_yaml": str(task.path), "seed": seed,
                           "init_yaw": 0.0, "init_pitch": 0.0,
                           "task_digest": task.definition()["sha256"],
                           "task_definition": json.dumps(task.definition())}}


def engine_episode(home):
    (home / "magma").mkdir(parents=True)
    (home / "magma/game").write_bytes(b"engine-binary")
    (home / ".bedrock-rl-patches").write_text("pinned patch bytes")
    (home / ".bedrock-rl-assets").write_text("kind=procedural")
    (home / "state.bsnp").write_bytes(b"frozen-world")
    return SimpleNamespace(view_provenance={"type": "netherite-procedural"},
                           env=SimpleNamespace(
        binary=home / "magma/game", snapshot_in=home / "state.bsnp",
        raster="cpu", frame_w=428, frame_h=240, view_distance=4,
        journal_on=True, mobs=False, world_time=1000))


def contract(task, rows, episode, hint="hint"):
    protocol, inputs = evaluation_protocol(task, hint, rows, ARGS, {"temperature": 0})
    return dict(protocol, input_sha256=inputs[0],
                initial_prompt_sha256=digest("rendered prompt"),
                processor_sha256=digest("same processor"),
                environment=environment_identity(episode))


def record(protocol, success=False):
    return {"trial_id": "0:0", "row_index": 0, "sample_index": 0,
            "seed": 1, "init_yaw": 0, "init_pitch": 0,
            "success": success, "opened": False, "actions": None,
            "reward": float(success), "turns": 1, "protocol": protocol}


class EvaluationProtocolTests(unittest.TestCase):
    def test_real_driver_emits_the_same_contract_for_each_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "task.yaml"
            path.write_text(yaml.safe_dump(DOCUMENT))
            task = Task.load(path)
            rows = [frozen_row(task, seed=seed) for seed in (1, 2)]
            environment = engine_episode(root / "engine")
            observed_prompts = []

            class Sampler:
                def __init__(self, **_kwargs):
                    pass

                def generate(self, requests, *_args, **_kwargs):
                    observed_prompts.extend(request["prompt"] for request in requests)
                    return [SimpleNamespace(outputs=[SimpleNamespace(text="No action")])
                            for _ in requests]

            processor = SimpleNamespace(
                to_dict=lambda: {}, chat_template="stable template",
                tokenizer=SimpleNamespace(get_vocab=lambda: {"token": 0},
                                          special_tokens_map={}),
                apply_chat_template=lambda messages, **_: json.dumps(messages,
                                                                     sort_keys=True))

            def open_episode(*_args, **_kwargs):
                episode = copy.deepcopy(environment)
                episode.done = episode.success = episode.failed = False
                episode.turns = 0
                episode.env.journal = []
                episode.final_reward = lambda: 0.0
                episode.close = lambda: None
                return episode

            def run(name, shard=None):
                output = root / f"{name}.json"
                argv = ["trials", "--data", "unused.parquet", "--task", str(path),
                        "--model", "qwen3-vl-2b", "--n", "2",
                        "--records-out", str(output)]
                if shard:
                    argv.extend(["--shard", shard])
                modules = {
                    "datasets": SimpleNamespace(load_dataset=lambda *_a, **_k: {"train": rows}),
                    "vllm": SimpleNamespace(LLM=Sampler, SamplingParams=lambda **kw: kw),
                    "vllm.lora.request": SimpleNamespace(LoRARequest=object),
                }
                with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv), \
                        patch.object(trials, "PREFIX", ""), \
                        patch.object(trials.reg, "resolve_policy", return_value=SimpleNamespace(
                            base="base", adapter=None, base_revision="pinned-revision")), \
                        patch.object(trials.reg, "weights_path", return_value="base") as weights, \
                        patch.object(trials.reg, "resolve", return_value=SimpleNamespace(
                            vit_attn_backend=None)), \
                        patch.object(trials.reg, "load_processor", return_value=processor), \
                        patch.object(trials.Episode, "from_spec", side_effect=open_episode), \
                        redirect_stdout(io.StringIO()):
                    trials.main()
                weights.assert_called_once_with("base", revision="pinned-revision")
                return json.loads(output.read_text())

            whole = run("whole")
            sharded = run("first", "0/2") + run("second", "1/2")
            self.assertEqual(whole, sharded)
            self.assertEqual(whole[0]["protocol"]["initial_prompt_sha256"],
                             digest(observed_prompts[0]))
            self.assertEqual(whole[0]["protocol"]["environment"],
                             environment_identity(environment))
            (root / "merged.json").write_text(json.dumps(sharded))
            summary = analyze([PointSpec("whole", 0, root / "whole.json"),
                               PointSpec("sharded", 1, root / "merged.json")])
            self.assertEqual(summary["contract_verification"], "content-fingerprinted")

    @unittest.skipUnless(importlib.util.find_spec("transformers"),
                         "training extra is not installed")
    def test_real_qwen_processor_metadata_keeps_identity_after_saving(self):
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from transformers import (Qwen3VLProcessor, Qwen2VLImageProcessorFast,
                                  Qwen2TokenizerFast, Qwen3VLVideoProcessor)

        # The full processor requires torchvision even for image-only tasks.
        # A tiny tokenizer exercises that dependency and real serialization
        # without downloading a model or loading any model weights.
        original = Qwen3VLProcessor(
            image_processor=Qwen2VLImageProcessorFast(),
            tokenizer=Qwen2TokenizerFast(
                tokenizer_object=Tokenizer(BPE(vocab={"a": 0}, merges=[]))),
            video_processor=Qwen3VLVideoProcessor(),
            chat_template="{{ messages }}")
        # A family-specific processor template can replace the tokenizer's
        # inherited template, which is not used to render evaluation prompts.
        original.tokenizer.chat_template = "unused tokenizer template"
        before = processor_identity(original)
        with tempfile.TemporaryDirectory() as directory:
            original.save_pretrained(directory)
            restored = Qwen3VLProcessor.from_pretrained(
                directory, local_files_only=True)
            self.assertEqual(processor_identity(restored), before)

    def test_processor_location_is_ignored_but_template_and_pixels_are_bound(self):
        config = {"processor_class": "Test", "_name_or_path": "/old/cache",
                  "init_kwargs": {"name_or_path": "/old/cache"}}
        image_config = {"size": {"max_pixels": 1024}, "name_or_path": "/old/cache"}
        processor = SimpleNamespace(
            to_dict=lambda: config, chat_template="initial template",
            image_processor=SimpleNamespace(to_dict=lambda: image_config),
            tokenizer=SimpleNamespace(get_vocab=lambda: {"token": 1},
                                      special_tokens_map={"pad_token": "pad"}))
        before = processor_identity(processor)
        config["_name_or_path"] = "/new/cache"
        config["init_kwargs"]["name_or_path"] = "/new/cache"
        image_config["name_or_path"] = "/new/cache"
        self.assertEqual(processor_identity(processor), before)
        processor.chat_template = "changed template"
        self.assertNotEqual(processor_identity(processor), before)
        processor.chat_template = "initial template"
        image_config["size"]["max_pixels"] = 2048
        self.assertNotEqual(processor_identity(processor), before)

    def test_frozen_definition_survives_edits_but_explicit_changed_verifier_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.yaml"
            path.write_text(yaml.safe_dump(DOCUMENT))
            original = Task.load(path)
            rows = [frozen_row(original)]
            changed = copy.deepcopy(DOCUMENT)
            changed["success"]["items"] = ["dirt"]
            path.write_text(yaml.safe_dump(changed))
            # Reconstruct the frozen definition, not today's edited template.
            self.assertEqual(evaluation_task(rows).success.spec["items"],
                             ["iron_pickaxe"])
            with self.assertRaisesRegex(SystemExit, "refusing to score"):
                evaluation_task(rows, str(path))
            rows[0]["extra_info"].pop("task_definition")
            with self.assertRaisesRegex(SystemExit, "task digest differs"):
                evaluation_task(rows)

    def test_legacy_frozen_rows_require_explicit_opt_in_and_remain_unverified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.yaml"
            path.write_text(yaml.safe_dump(DOCUMENT))
            task = Task.load(path)
            rows = [frozen_row(task)]
            rows[0]["extra_info"].pop("task_definition")
            rows[0]["extra_info"].pop("task_digest")
            with self.assertRaisesRegex(SystemExit, "allow-legacy-inputs"):
                evaluation_task(rows)
            restored = evaluation_task(rows, allow_legacy=True)
            protocol, _ = evaluation_protocol(restored, None, rows, ARGS, {})
            self.assertFalse(protocol["frozen_task_verified"])

    def test_changed_task_hint_prompt_pixels_and_environment_cannot_be_paired(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = Task(copy.deepcopy(DOCUMENT), path=str(root / "task.yaml"))
            rows = [frozen_row(task)]
            episode = engine_episode(root / "engine")
            before = contract(task, rows, episode)
            variants = {}
            document = copy.deepcopy(DOCUMENT)
            document["success"]["items"] = ["dirt"]
            changed_task = Task(document, path=task.path)
            variants["task"] = contract(changed_task, [frozen_row(changed_task)], episode)
            variants["hint"] = contract(task, rows, episode, hint="changed hint")
            text_rows = copy.deepcopy(rows)
            text_rows[0]["prompt"][0]["content"] += " changed"
            variants["prompt"] = contract(task, text_rows, episode)
            pixel_rows = copy.deepcopy(rows)
            pixel_rows[0]["images"][0].putpixel((0, 0), (1, 2, 3))
            variants["image"] = contract(task, pixel_rows, episode)
            for name, path in (
                    ("snapshot", episode.env.snapshot_in),
                    ("binary", episode.env.binary),
                    ("patches", root / "engine/.bedrock-rl-patches"),
                    ("assets", root / "engine/.bedrock-rl-assets")):
                original = path.read_bytes()
                path.write_bytes(original + b" changed")
                variants[name] = contract(task, rows, episode)
                path.write_bytes(original)
            base = root / "base.json"
            base.write_text(json.dumps([record(before)]))
            for name, after in variants.items():
                with self.subTest(changed=name):
                    final = root / f"{name}.json"
                    final.write_text(json.dumps([record(after, success=True)]))
                    with self.assertRaisesRegex(EvaluationError, "changed evaluation protocol"):
                        analyze([PointSpec("base", 0, base), PointSpec("final", 1, final)])

    def test_custom_verifier_helper_change_invalidates_paired_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "bedrock_test_reward"
            package.mkdir()
            (package / "__init__.py").write_text(
                "from .rules import holds\n")
            (package / "rules.py").write_text(
                "from .threshold import verdict\n"
                "def holds(env, start):\n    return verdict()\n")
            helper = package / "threshold.py"
            module_names = ("bedrock_test_reward", "bedrock_test_reward.rules",
                            "bedrock_test_reward.threshold")
            task_path = root / "task.yaml"
            doc = {**DOCUMENT, "success": {"type": "bedrock_test_reward:holds"},
                   "verifier_dependencies": {}}
            environment = engine_episode(root / "engine")
            points = []
            with patch.object(sys, "path", [str(root), *sys.path]):
                try:
                    for idx, expected in enumerate((False, True)):
                        for name in module_names:
                            sys.modules.pop(name, None)
                        helper.write_text(f"def verdict():\n    return {expected!r}\n")
                        importlib.invalidate_caches()
                        task = Task(doc, path=task_path)
                        success = task.success.holds(None, {})
                        self.assertIs(success, expected)
                        protocol = contract(task, [frozen_row(task)], environment)
                        records = root / f"records{idx}.json"
                        records.write_text(json.dumps([record(protocol, success)]))
                        points.append(PointSpec(str(idx), idx, records))
                    with self.assertRaisesRegex(EvaluationError, "changed evaluation protocol"):
                        analyze(points)
                finally:
                    for name in module_names:
                        sys.modules.pop(name, None)

    def test_external_verifier_dependencies_and_task_relative_assets(self):
        from bedrock_rl.eval.protocol import verifier_identity

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            modules = root / "modules"
            package = modules / "audit_scores"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("score = 1\n")
            helper = modules / "audit_helper.py"
            helper.write_text("score = 1\n")
            (modules / "audit_predicate.py").write_text(
                "import audit_helper, audit_scores\n"
                "def holds(env, start):\n"
                "    return audit_helper.score == audit_scores.score\n")
            doc = {**DOCUMENT, "success": {"type": "audit_predicate:holds"}}
            original, moved = root / "original", root / "moved"
            original.mkdir()
            moved.mkdir()
            for base in (original, moved):
                (base / "scoring.json").write_text("1")
            environment = engine_episode(root / "engine")
            with patch.object(sys, "path", [str(modules), *sys.path]):
                try:
                    task = Task(doc, path=str(original / "task.yaml"))
                    protocol = contract(task, [frozen_row(task)], environment)
                    self.assertFalse(protocol["verifier_dependencies_verified"])
                    output = root / "records.json"
                    output.write_text(json.dumps([record(protocol)]))
                    points = [PointSpec("a", 0, output), PointSpec("b", 1, output)]
                    with self.assertRaisesRegex(EvaluationError, "undeclared"):
                        analyze(points)
                    self.assertEqual(analyze(points, allow_legacy_protocol=True)
                                     ["contract_verification"], "undeclared-verifier-dependencies")

                    doc["verifier_dependencies"] = {
                        "modules": ["audit_helper", "audit_scores"], "files": ["scoring.json"]}
                    task = Task(doc, path=str(original / "task.yaml"))
                    relocated = Task(doc, path=str(moved / "task.yaml"))
                    (moved / "task.yaml").write_text(yaml.safe_dump(doc))
                    frozen = Task.from_definition(task.definition())
                    before, verified = verifier_identity(frozen)
                    self.assertTrue(verified)
                    self.assertEqual((before, True), verifier_identity(relocated))
                    for artifact in (helper, package / "__init__.py", original / "scoring.json"):
                        data = artifact.read_bytes()
                        artifact.write_bytes(data + b" ")
                        self.assertNotEqual(before, verifier_identity(frozen)[0])
                        artifact.write_bytes(data)
                    frozen_row_before_move = frozen_row(task)
                    shutil.rmtree(original)
                    selected = evaluation_task([frozen_row_before_move], str(moved / "task.yaml"))
                    self.assertEqual(Path(selected.path).resolve(), (moved / "task.yaml").resolve())
                    # Confirm that the actual Episode constructor is given the
                    # attested new base, rather than silently reverting to the
                    # disappeared path in the frozen definition.
                    from bedrock_rl.env.episode import Episode
                    with patch.object(Episode, "__init__", return_value=None) as opened, \
                            patch.object(Episode, "spec", {}, create=True):
                        episode = Episode.from_spec(frozen_row_before_move["extra_info"],
                                                    lambda *_: None, task=selected)
                        self.assertEqual(Path(opened.call_args.args[0].path).resolve(),
                                         (moved / "task.yaml").resolve())
                        self.assertEqual(episode.spec["task_definition"]["verifier_files_sha256"],
                                         selected.verifier_file_digests())
                    legacy = copy.deepcopy(frozen_row_before_move)
                    definition = json.loads(legacy["extra_info"]["task_definition"])
                    definition.pop("verifier_files_sha256")
                    legacy["extra_info"]["task_definition"] = json.dumps(definition)
                    with self.assertRaisesRegex(SystemExit, "cannot relocate"):
                        evaluation_task([legacy], str(moved / "task.yaml"))
                    (moved / "scoring.json").write_text("2")
                    with self.assertRaisesRegex(SystemExit, "dependency files"):
                        evaluation_task([frozen_row_before_move], str(moved / "task.yaml"))
                finally:
                    for name in ("audit_scores", "audit_helper", "audit_predicate"):
                        sys.modules.pop(name, None)

    def test_v2_evidence_requires_explicit_historical_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = Task(copy.deepcopy(DOCUMENT), path=str(root / "task.yaml"))
            protocol = contract(task, [frozen_row(task)], engine_episode(root / "engine"))
            protocol["schema"] = "bedrock.eval-protocol.v2"
            protocol.pop("verifier_sha256")
            path = root / "historical.json"
            path.write_text(json.dumps([record(protocol)]))
            points = [PointSpec("a", 0, path), PointSpec("b", 1, path)]
            with self.assertRaisesRegex(EvaluationError, "allow-legacy-protocol"):
                analyze(points)
            self.assertEqual(analyze(points, allow_legacy_protocol=True)
                             ["contract_verification"], "v2-unverified-verifier")

    def test_content_identity_is_stable_across_hosts_and_shard_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = Task(copy.deepcopy(DOCUMENT), path=str(root / "host1/task.yaml"))
            other_task = Task(copy.deepcopy(DOCUMENT), path=str(root / "host2/task.yaml"))
            rows = [frozen_row(task, seed=seed) for seed in (1, 2)]
            moved_rows = [frozen_row(other_task, seed=seed) for seed in (1, 2)]
            episode = engine_episode(root / "engine1")
            other = copy.deepcopy(episode)
            shutil.copytree(root / "engine1", root / "engine2")
            other.env.binary = root / "engine2/magma/game"
            other.env.snapshot_in = root / "engine2/state.bsnp"
            self.assertEqual(contract(task, rows, episode),
                             contract(other_task, moved_rows, other))
            whole, identities = evaluation_protocol(task, None, rows, ARGS, {})
            # Every shard fingerprints the full selected rows before taking
            # its trial range, and uses the original row index for its input.
            for index in range(2):
                shard, shard_ids = evaluation_protocol(other_task, None, moved_rows, ARGS, {})
                self.assertEqual(dict(whole, input_sha256=identities[index]),
                                 dict(shard, input_sha256=shard_ids[index]))

    def test_legacy_record_opt_in_is_marked_and_cannot_mix_with_v2(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = {"task": "old.yaml", "hint": False, "max_turns": 2}
            paths = [root / "base.json", root / "final.json"]
            for path in paths:
                path.write_text(json.dumps([record(legacy)]))
            points = [PointSpec("base", 0, paths[0]), PointSpec("final", 1, paths[1])]
            with self.assertRaisesRegex(EvaluationError, "allow-legacy-protocol"):
                analyze(points)
            result = analyze(points, allow_legacy_protocol=True)
            self.assertEqual(result["contract_verification"], "legacy-unverified")
            self.assertTrue(result["warnings"])
            task = Task(copy.deepcopy(DOCUMENT), path=str(root / "task.yaml"))
            protocol = contract(task, [frozen_row(task)], engine_episode(root / "engine"))
            paths[1].write_text(json.dumps([record(protocol)]))
            with self.assertRaisesRegex(EvaluationError, "cannot mix"):
                analyze(points, allow_legacy_protocol=True)


if __name__ == "__main__":
    unittest.main()
