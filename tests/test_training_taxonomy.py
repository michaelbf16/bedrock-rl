import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

from bedrock_rl.config import training as config
from bedrock_rl.sdg import dataset_path


class TrainingTaxonomyTests(unittest.TestCase):
    def test_named_training_presets_are_independent_configs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.yaml"
            path.write_text("""
defaults:
  task: task.yaml
  model: qwen3-vl-2b
presets:
  warmup: {trainer: sft, steps: 10}
  sdft: {trainer: sdft, steps: 5, demonstrations: demos.parquet}
""")
            warmup = config.train_plan(config.load(path, "warmup"))
            sdft = config.train_plan(config.load(path, "sdft"))
        self.assertEqual(warmup.target, "sft")
        self.assertEqual(sdft.target, "sdft")

    def test_paths_inside_training_configs_are_config_relative(self):
        from bedrock_rl.cli.main import config_relative

        self.assertEqual(
            config_relative("task.yaml", "/tmp/example"),
            "/tmp/example/task.yaml",
        )
        self.assertEqual(config_relative("", "/tmp/example"), "")

    def test_existing_model_and_teacher_paths_are_config_relative(self):
        import json
        from bedrock_rl.cli.main import config_relative, model_arg

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            student = root / "checkpoints" / "student"
            teacher = root / "checkpoints" / "teacher"
            student.mkdir(parents=True)
            teacher.mkdir(parents=True)
            self.assertEqual(
                model_arg("checkpoints/student", directory), str(student))
            plan = config.train_plan({
                "trainer": "opd", "task": "equip_pickaxe_grpo",
                "model": "qwen3-vl", "steps": 3,
                "teachers": {"expert": {
                    "model": "checkpoints/teacher"}},
            }, resolve=lambda value: config_relative(value, directory))
            teachers = json.loads(plan.env["BRL_TEACHERS_JSON"])
            self.assertEqual(teachers["expert"]["model"], str(teacher))

    def test_huggingface_ids_are_not_rewritten_as_config_paths(self):
        from bedrock_rl.cli.main import model_arg

        model = "Qwen/Qwen3-VL-2B-Instruct"
        self.assertEqual(model_arg(model, "/tmp/example"), model)

    def plan(self, **values):
        base = {
            "task": "equip_pickaxe_grpo",
            "model": "qwen3-vl",
            "steps": 3,
        }
        return config.train_plan({**base, **values})

    def error(self, **values):
        with self.assertRaises(config.ConfigError) as caught:
            self.plan(**values)
        return str(caught.exception)

    def test_public_names_separate_trainers_and_workflows(self):
        self.assertEqual(tuple(config.TRAINERS),
                         ("rl", "opd", "mopd", "sft", "sdft", "sdpo"))
        self.assertEqual(tuple(config.WORKFLOWS), ("self_distill",))

    def test_sft_data_has_an_sft_name(self):
        task = SimpleNamespace(name="example_task")
        self.assertEqual(dataset_path(task, "sft"),
                         "data/example_task_cu/sft.parquet")

    def test_rl_prompt_paths_refuse_unremovable_static_hints(self):
        task = SimpleNamespace(name="example_task")
        with self.assertRaisesRegex(ValueError, "must be hint-free"):
            dataset_path(task, "rl", "procedure")

    def test_rl_algorithm_maps_to_adv(self):
        plan = self.plan(trainer="rl", algorithm="rloo")
        self.assertEqual(plan.kind, "trainer")
        self.assertEqual(plan.target, "rl")
        self.assertEqual(plan.script, "rl.sh")
        self.assertEqual(plan.env["ADV"], "rloo")

    def test_rl_agent_loop_is_replaceable(self):
        plan = self.plan(
            trainer="rl", agent_loop_config="custom_agent.yaml",
            agent_loop="custom")
        self.assertEqual(plan.env["AGENT_LOOP_CONFIG"],
                         "custom_agent.yaml")
        self.assertEqual(plan.env["AGENT_LOOP"], "custom")

    def test_rl_agent_worker_count_is_configurable(self):
        plan = self.plan(trainer="rl", agent_workers=4)
        self.assertEqual(plan.env["AGENT_WORKERS"], "4")

    def test_agent_workers_must_partition_the_rollout_batch(self):
        message = self.error(
            trainer="rl", train_bs=32, rollout_n=8, agent_workers=24)
        self.assertIn(
            "agent_workers (24) must divide train_bs * rollout_n "
            "(32 * 8 = 256)", message)

    def test_standalone_distillation_sampling_knobs_reach_launchers(self):
        for trainer in ("sdft", "sdpo"):
            with self.subTest(trainer=trainer):
                values = {
                    "trainer": trainer, "temperature": 0.7,
                    "max_new_tokens": 123, "workers": 6,
                }
                if trainer == "sdft":
                    values["demonstrations"] = "demos.parquet"
                plan = self.plan(**values)
                self.assertEqual(plan.env["TEMPERATURE"], "0.7")
                self.assertEqual(plan.env["MAX_NEW_TOKENS"], "123")
                self.assertEqual(plan.env["WORKERS"], "6")

    def test_verl_validation_inputs_are_not_silent_training_knobs(self):
        for key in ("val_data", "val_files"):
            with self.subTest(key=key):
                self.assertIn(f"unknown key '{key}'", self.error(
                    trainer="rl", **{key: "held-out.parquet"}))

    def test_rl_exposes_verl_tensorboard(self):
        plan = self.plan(
            trainer="rl", tensorboard=True,
            tensorboard_dir="curves/tensorboard")
        self.assertEqual(plan.env["TENSORBOARD"], "1")
        self.assertEqual(plan.env["TENSORBOARD_DIR"],
                         "curves/tensorboard")

    def test_tensorboard_is_not_claimed_by_non_verl_trainers(self):
        message = self.error(trainer="sft", tensorboard=True)
        self.assertIn("bin/sft.sh does not read", message)

    def test_sft_is_the_trainer_name_and_launcher(self):
        plan = self.plan(trainer="sft", deepspeed="z3")
        self.assertEqual(plan.kind, "trainer")
        self.assertEqual(plan.target, "sft")
        self.assertEqual(plan.script, "sft.sh")
        self.assertEqual(plan.env["DEEPSPEED"], "z3")

    def test_sdft_is_a_trainer(self):
        plan = self.plan(
            trainer="sdft", demonstrations="demos.jsonl",
            demonstration_match="seed")
        self.assertEqual(plan.kind, "trainer")
        self.assertEqual(plan.target, "sdft")
        self.assertEqual(plan.script, "sdft.sh")
        self.assertEqual(plan.env["DEMONSTRATIONS"], "demos.jsonl")
        self.assertEqual(plan.env["DEMONSTRATION_MATCH"], "seed")

    def test_sdpo_exposes_ema_teacher_update_rate(self):
        plan = self.plan(trainer="sdpo", teacher_update_rate=0.05)

        self.assertEqual(plan.env["TEACHER_UPDATE_RATE"], "0.05")

    def test_opd_serializes_one_structured_teacher(self):
        plan = self.plan(
            trainer="opd",
            teachers={"expert": {
                "model": "Qwen/Qwen3-VL-32B-Instruct",
                "tensor_parallel": 2,
            }},
            teacher_gpus_per_node=2,
        )
        self.assertEqual(plan.script, "rl.sh")
        self.assertEqual(plan.env["BRL_DISTILLATION_MODE"], "opd")
        self.assertIn('"expert"', plan.env["BRL_TEACHERS_JSON"])
        self.assertIn("Qwen/Qwen3-VL-32B-Instruct",
                      plan.env["BRL_TEACHERS_JSON"])
        self.assertEqual(plan.env["TEACHER_GPUS_PER_NODE"], "2")

    def test_mopd_needs_unique_named_teacher_configs(self):
        plan = self.plan(
            trainer="mopd",
            teachers={
                "mining": {"model": "teacher-a", "route": "mining"},
                "crafting": {"model": "teacher-b", "route": "crafting"},
            },
            teacher_key="data_source",
        )
        self.assertEqual(plan.script, "rl.sh")
        self.assertEqual(plan.env["BRL_DISTILLATION_MODE"], "mopd")
        self.assertEqual(plan.env["TEACHER_KEY"], "data_source")
        self.assertIn('"route":"crafting"', plan.env["BRL_TEACHERS_JSON"])

    def test_distillation_teacher_counts_are_explicit(self):
        self.assertIn("exactly 1 teacher", self.error(
            trainer="opd", teachers={
                "a": {"model": "a"}, "b": {"model": "b"}}))
        self.assertIn("at least 2 teachers", self.error(
            trainer="mopd", teachers={"a": {"model": "a"}}))
        self.assertIn("needs teachers", self.error(trainer="opd"))

    def test_self_distill_is_a_workflow(self):
        plan = self.plan(workflow="self_distill", epochs=2)
        self.assertEqual(plan.kind, "workflow")
        self.assertEqual(plan.target, "self_distill")
        self.assertEqual(plan.script, "self_distill.sh")
        self.assertEqual(plan.env["EPOCHS"], "2")

    def test_qlora_only_reaches_sft_based_targets(self):
        self.assertEqual(
            self.plan(trainer="sft", lora_rank=16,
                      lora_4bit=True).env["LORA_4BIT"], "1")
        self.assertEqual(
            self.plan(workflow="self_distill",
                      lora_rank=16, lora_4bit=True).env["LORA_4BIT"], "1")
        self.assertIn("does not read",
                      self.error(trainer="sdft", lora_4bit=True))
        self.assertIn("does not read",
                      self.error(trainer="sdpo", lora_4bit=True))

    def test_selection_is_exclusive(self):
        message = self.error(trainer="rl", workflow="self_distill")
        self.assertIn("both trainer: and workflow:", message)

    def test_algorithm_only_applies_to_rl(self):
        message = self.error(trainer="sft", algorithm="grpo")
        self.assertIn("does not apply to trainer: sft", message)

    def test_old_names_have_explicit_migrations(self):
        self.assertIn("trainer: sft", self.error(trainer="warmstart"))
        self.assertIn("workflow: self_distill", self.error(workflow="star"))
        self.assertIn("whole-loop runner was removed",
                      self.error(trainer="loop"))

    def test_old_rl_and_loop_keys_have_explicit_migrations(self):
        self.assertIn("algorithm: grpo",
                      self.error(trainer="rl", adv="grpo"))
        self.assertIn("removed whole-loop runner",
                      self.error(trainer="rl", mode="warmstart"))

    def test_synthetic_steps_do_not_expose_a_pipeline_command(self):
        from bedrock_rl.cli.main import GROUP_NAMES
        self.assertIn("generate", GROUP_NAMES)
        self.assertNotIn("pipeline", GROUP_NAMES)

    def test_evaluation_exposes_trials_and_paired_summaries(self):
        from bedrock_rl.cli.main import build_parser
        _, groups = build_parser()
        choices = groups["eval"].choices
        self.assertEqual(tuple(choices), ("trials", "summarize"))

    def test_numeric_switches_are_actual_booleans(self):
        knob = next(item for item in config.KNOBS
                    if item.env == "HARVEST_ONLY")
        self.assertEqual(knob.render(0), knob.off)
        self.assertEqual(knob.render(1), knob.on)
        with self.assertRaisesRegex(ValueError, r"0 \(off\) or 1 \(on\)"):
            knob.render(2)

    def test_evaluation_requires_an_explicit_dataset(self):
        from bedrock_rl.cli.main import build_parser
        parser, _ = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["eval", "trials", "task.yaml",
                               "--model", "checkpoint"])
        args = parser.parse_args([
            "eval", "trials", "task.yaml", "--model", "checkpoint",
            "--data", "dev.parquet", "--keep-latest-images", "2"])
        self.assertEqual(args.data, "dev.parquet")
        self.assertEqual(args.keep_latest_images, 2)

    def test_turn_budget_belongs_only_to_the_task(self):
        self.assertIn("unknown key 'max_turns'", self.error(
            trainer="rl", max_turns=3))


if __name__ == "__main__":
    unittest.main()
