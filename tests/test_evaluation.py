import json
import importlib.util
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from bedrock_rl.eval import (EvaluationError, PointSpec, analyze,
                             combined_svg, load_training_scores,
                             seed_clustered_interval, wilson_interval,
                             write_summary, write_svg)


def trial(identity, seed, success, reward, turns, actions=None,
          broken=None):
    return {
        "trial_id": identity,
        "seed": seed,
        "success": success,
        "failed": False,
        "actions": actions,
        "turns": turns,
        "reward": reward,
        "opened": False,
        "broken": broken,
        "protocol": {
            "task": "task.yaml", "hint": False, "hint_level": "none",
            "finish_level": "off", "keep_latest_images": 0,
            "blank_frames": False,
            "sampling": {"temperature": 1.0, "top_p": 1.0,
                         "top_k": 0, "max_new_tokens": 256},
            "max_model_len": 8192,
            "max_turns": 6,
            "tensor_parallel_size": 1,
        },
    }


class EvaluationTests(unittest.TestCase):
    def write(self, directory, name, records):
        path = Path(directory) / name
        path.write_text(json.dumps(records))
        return path

    def test_wilson_interval_handles_boundary_rates(self):
        low, high = wilson_interval(0, 10)
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, 0.2775328, places=6)
        low, high = wilson_interval(10, 10)
        self.assertAlmostEqual(low, 0.7224672, places=6)
        self.assertEqual(high, 1.0)

    def test_analysis_pairs_by_explicit_identity_and_computes_deltas(self):
        with tempfile.TemporaryDirectory() as directory:
            base = self.write(directory, "base.json", [
                trial("a", 11, False, -0.2, 3, None),
                trial("b", 12, True, 1.0, 2, [{"action": "key"}]),
            ])
            # Deliberately reversed: explicit identities, not file position,
            # own the pairing.
            final_rows = [
                trial("b", 12, True, 1.0, 1, [{"action": "key"}]),
                trial("a", 11, True, 0.8, 2, [{"action": "key"}]),
            ]
            final_rows[1]["opened"] = True
            final = self.write(directory, "final.json", final_rows)
            result = analyze([
                PointSpec("base", 0, base),
                PointSpec("final", 30, final),
            ], base="base")

        self.assertEqual(result["schema"], "bedrock.eval-curve.v1")
        self.assertEqual(result["trials_per_point"], 2)
        self.assertEqual(result["identity_mode"], "trial_id")
        first, last = result["points"]
        self.assertEqual(first["metrics"]["success_rate"], 0.5)
        self.assertEqual(last["metrics"]["success_rate"], 1.0)
        self.assertEqual(last["metrics"]["tool_call_rate"], 1.0)
        self.assertAlmostEqual(last["delta_vs_base"]["success_rate"], 0.5)
        self.assertAlmostEqual(last["delta_vs_base"]["reward_mean"], 0.5)
        self.assertEqual(last["metrics"]["workstation_open_rate"], 0.5)
        self.assertEqual(
            last["delta_vs_base"]["workstation_open_rate"], 0.5)
        self.assertEqual(
            last["delta_vs_base"]["transitions"]["failure_to_success"], 1)
        self.assertEqual(
            last["delta_vs_base"]["transitions"]["success_to_failure"], 0)

    def test_broken_trial_refuses_the_whole_curve(self):
        with tempfile.TemporaryDirectory() as directory:
            base = self.write(directory, "base.json", [
                trial("a", 1, False, 0.0, 1),
            ])
            broken = self.write(directory, "broken.json", [
                trial("a", 1, False, None, 0,
                      broken="engine stopped answering"),
            ])
            with self.assertRaisesRegex(EvaluationError, "partial"):
                analyze([PointSpec("base", 0, base),
                         PointSpec("final", 1, broken)])

    def test_mismatched_trial_identities_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            base = self.write(directory, "base.json", [
                trial("a", 1, False, 0.0, 1),
                trial("b", 2, False, 0.0, 1),
            ])
            final = self.write(directory, "final.json", [
                trial("a", 1, True, 1.0, 1),
                trial("c", 3, True, 1.0, 1),
            ])
            with self.assertRaisesRegex(EvaluationError, "not paired"):
                analyze([PointSpec("base", 0, base),
                         PointSpec("final", 1, final)])

    def test_same_trial_id_with_different_world_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            base = self.write(directory, "base.json", [
                trial("a", 1, False, 0.0, 1),
            ])
            final = self.write(directory, "final.json", [
                trial("a", 2, True, 1.0, 1),
            ])
            with self.assertRaisesRegex(EvaluationError, "changed seed"):
                analyze([PointSpec("base", 0, base),
                         PointSpec("final", 1, final)])

    def test_mismatched_sampling_protocol_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            base_row = trial("a", 1, False, 0.0, 1)
            final_row = trial("a", 1, True, 1.0, 1)
            final_row["protocol"]["keep_latest_images"] = 2
            base = self.write(directory, "base.json", [base_row])
            final = self.write(directory, "final.json", [final_row])
            with self.assertRaisesRegex(EvaluationError,
                                        "changed evaluation protocol"):
                analyze([PointSpec("base", 0, base),
                         PointSpec("final", 1, final)])

    def test_success_interval_clusters_repeated_trials_by_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = ([trial(f"a{i}", 1, True, 1.0, 1) for i in range(8)]
                    + [trial(f"b{i}", 2, False, 0.0, 1)
                       for i in range(8)])
            base = self.write(directory, "base.json", rows)
            final = self.write(directory, "final.json", rows)
            result = analyze([PointSpec("base", 0, base),
                              PointSpec("final", 1, final)])

        metrics = result["points"][0]["metrics"]
        self.assertEqual(metrics["independent_seed_clusters"], 2)
        self.assertEqual(metrics["success_ci95_method"], "seed-clustered")
        low, high = metrics["success_ci95"]
        self.assertLess(low, 0.1)
        self.assertGreater(high, 0.9)

    def test_seed_interval_handles_unbalanced_cluster_sizes(self):
        records = ([{"seed": 1, "success": True}] * 9
                   + [{"seed": 2, "success": False}])

        low, high = seed_clustered_interval(records)

        self.assertLessEqual(low, 0.9)
        self.assertGreaterEqual(high, 0.9)
        self.assertGreater(high - low, 0.25)

    def test_summary_counts_tool_attempts_from_any_turn(self):
        with tempfile.TemporaryDirectory() as directory:
            base_row = trial("a", 1, False, -0.2, 2, None)
            base_row["emitted_tool_call"] = True
            final_row = dict(base_row)
            final_row["protocol"] = dict(base_row["protocol"])
            base = self.write(directory, "base.json", [base_row])
            final = self.write(directory, "final.json", [final_row])

            result = analyze([PointSpec("base", 0, base),
                              PointSpec("final", 1, final)])

        self.assertEqual(
            result["points"][0]["metrics"]["tool_call_rate"], 1.0)

    def test_legacy_run_trials_records_pair_by_position_and_seed(self):
        def legacy(seed):
            row = trial("unused", seed, False, 0.0, 1)
            row.pop("trial_id")
            return row

        with tempfile.TemporaryDirectory() as directory:
            base = self.write(directory, "base.json",
                              [legacy(10), legacy(20)])
            reordered = self.write(directory, "reordered.json",
                                   [legacy(20), legacy(10)])
            with self.assertRaisesRegex(EvaluationError, "not paired"):
                analyze([PointSpec("base", 0, base),
                         PointSpec("final", 1, reordered)])

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"),
                         "plot extra is not installed")
    def test_summary_and_svg_are_valid_files(self):
        with tempfile.TemporaryDirectory() as directory:
            base = self.write(directory, "base.json", [
                trial("a", 1, False, 0.0, 2),
                trial("b", 2, False, 0.0, 2),
            ])
            final = self.write(directory, "final.json", [
                trial("a", 1, True, 1.0, 1, []),
                trial("b", 2, False, 0.0, 2),
            ])
            result = analyze([PointSpec("base", 0, base),
                              PointSpec("final", 10, final)])
            summary = Path(directory) / "summary.json"
            curve = Path(directory) / "success.svg"
            write_summary(result, summary)
            write_svg(result, curve, "Test held-out curve")
            reread = json.loads(summary.read_text())
            ET.parse(curve)
            curve_text = curve.read_text()

        self.assertEqual(reread["base"], "base")
        self.assertIn("held-out eval", curve_text)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"),
                         "plot extra is not installed")
    def test_matplotlib_curve_is_heldout_only(self):
        from bedrock_rl.eval.plotting import heldout_svg

        rendered = heldout_svg({
            "points": [
                {"step": 0, "metrics": {"success_rate": 0.1}},
                {"step": 40, "metrics": {"success_rate": 0.3}},
            ],
        }, success_max=0.3)
        ET.fromstring(rendered)
        self.assertIn("held-out eval", rendered)
        self.assertIn("HELD-OUT SUCCESS", rendered)
        self.assertIn(">30%</text>", rendered)
        self.assertNotIn("BEDROCK-RL", rendered)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"),
                         "plot extra is not installed")
    def test_combined_curve_keeps_every_training_step_and_heldout_mark(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics = Path(directory) / "metrics.jsonl"
            metrics.write_text(
                '{"step": 1, "critic/score/mean": -0.5}\n'
                '{"step": 2, "critic/score/mean": 0.25}\n')
            training = load_training_scores(metrics)
            base = self.write(directory, "base.json", [
                trial("a", 1, False, -0.5, 2),
                trial("b", 2, False, -0.5, 2),
            ])
            final = self.write(directory, "final.json", [
                trial("a", 1, True, 1.0, 1, []),
                trial("b", 2, False, -0.5, 2),
            ])
            heldout = analyze([PointSpec("base", 0, base),
                               PointSpec("final", 2, final)])
            rendered = combined_svg(
                training, [("held-out eval", heldout)], "Combined")
            ET.fromstring(rendered)

        self.assertEqual([p["step"] for p in training["points"]], [1, 2])
        self.assertIn(">train reward (each step)</text>", rendered)
        self.assertIn("TRAIN MEAN REWARD", rendered)
        self.assertIn("held-out eval", rendered)


if __name__ == "__main__":
    unittest.main()
