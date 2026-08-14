import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

from bedrock_rl.config import training as config
from bedrock_rl.sdg.generation import GenerationSpec, load_generation_jobs
from bedrock_rl.env.catalog import resolve_items
from bedrock_rl.env.task import Task


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "equip_pickaxe_grpo"


class AcceptedExampleTests(unittest.TestCase):
    def test_public_example_has_only_task_data_and_train_yaml(self):
        self.assertEqual(
            sorted(path.name for path in EXAMPLE.glob("*.yaml")),
            ["data.yaml", "task.yaml", "train.yaml"],
        )

    def test_task_uses_live_selected_item_reward(self):
        task = Task.load(EXAMPLE / "task.yaml")
        self.assertEqual(task.name, "equip_pickaxe_grpo")
        self.assertEqual(task.success.type, "selected_item")
        self.assertEqual(task.success.spec["items"], ["iron_pickaxe"])
        self.assertEqual(task.max_turns, 1)

    def test_data_jobs_are_disjoint_and_slot_balanced(self):
        jobs = load_generation_jobs(EXAMPLE / "data.yaml")
        self.assertEqual(tuple(jobs), ("train", "dev", "test"))
        train = jobs["train"]["sampler"]["seeds"]
        dev = jobs["dev"]["sampler"]["seeds"]
        self.assertTrue(set(train).isdisjoint(dev))

        task = Task.load(EXAMPLE / "task.yaml")
        target = resolve_items(["iron_pickaxe"])[0]
        for seeds in (train, dev):
            slots = []
            for seed in seeds:
                inventory = task.initial_state.inventory_slots(seed)
                slots.append(next(
                    slot for slot, (item, _count, _meta) in inventory.items()
                    if item == target))
            self.assertEqual(Counter(slots), Counter({slot: 4
                                                      for slot in range(9)}))

    def test_every_data_job_resolves(self):
        for job in ("train", "dev", "test"):
            spec = GenerationSpec.load(EXAMPLE / "data.yaml", job)
            self.assertEqual(
                {name for name, _ in spec.validate_components()},
                {"sampler", "generator", "executor", "writer"},
            )

    def test_train_config_is_grpo(self):
        plan = config.train_plan(config.load(EXAMPLE / "train.yaml"))
        self.assertEqual(plan.target, "rl")
        self.assertEqual(plan.algorithm, "grpo")
        self.assertEqual(plan.steps, 10)
        self.assertEqual(plan.env["TRAIN_BS"], "32")
        self.assertEqual(plan.env["ROLLOUT_N"], "8")
        self.assertEqual(plan.env["AGENT_WORKERS"], "32")
        self.assertEqual(plan.env["TEMP"], "2.0")
        self.assertEqual(plan.env["ENTROPY_COEFF"], "0.1")
        self.assertEqual(plan.env["RESPONSE_LEN"], "256")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", plan.env)

    def test_ten_step_smoke_evidence_is_complete(self):
        curves = EXAMPLE / "curves"
        summary = json.loads((curves / "dev_summary.json").read_text())
        self.assertEqual(summary["trials_per_point"], 288)
        self.assertEqual(
            [(point["step"], point["metrics"]["successes"])
             for point in summary["points"]],
            [(0.0, 27), (5.0, 33), (10.0, 39)],
        )
        for name in ("dev_base.records.json", "dev_step5.records.json",
                     "dev_step10.records.json"):
            records = json.loads((curves / name).read_text())
            self.assertEqual(len(records), 288)
            self.assertFalse(any(record["broken"] for record in records))
        metrics = [json.loads(line) for line in
                   (curves / "train_metrics.jsonl").read_text().splitlines()]
        self.assertEqual([row["step"] for row in metrics], list(range(1, 11)))
        curve = (curves / "learning_curve.svg").read_text()
        self.assertTrue(curve.startswith("<?xml"))
        self.assertIn("held-out eval", curve)
        self.assertIn(">train reward (each step)</text>", curve)
        self.assertIn(">20%</text>", curve)

    def test_pass3_evidence_has_three_complete_samples_per_prompt(self):
        curves = EXAMPLE / "curves"
        summary = json.loads((curves / "pass3_summary.json").read_text())
        self.assertEqual(summary["samples_per_prompt"], 3)
        self.assertEqual(
            [(point["label"], point["pass_successes"])
             for point in summary["points"]],
            [("Base", 31), ("GRPO-10", 64)],
        )
        self.assertEqual(summary["paired_transitions"], {
            "both_failure": 220,
            "both_success": 27,
            "failure_to_success": 37,
            "success_to_failure": 4,
        })
        self.assertEqual(summary["broken_trials"], 0)
        for point in summary["points"]:
            name = point["records"]
            records = json.loads((curves / name).read_text())
            self.assertEqual(len(records), 864)
            self.assertEqual(
                {record["sample_index"] for record in records}, {0, 1, 2})
            self.assertFalse(any(record["broken"] for record in records))
            self.assertEqual(
                hashlib.sha256((curves / name).read_bytes()).hexdigest(),
                point["sha256"],
            )

if __name__ == "__main__":
    unittest.main()
