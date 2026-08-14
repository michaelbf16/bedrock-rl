import unittest

from bedrock_rl.train.sft_data import LazyEncodedDataset, validate_columns


class SFTDataTests(unittest.TestCase):
    def test_dataset_encodes_only_requested_rows(self):
        calls = []
        dataset = LazyEncodedDataset([{"value": 1}, {"value": 2}],
                                     lambda row: calls.append(row["value"])
                                     or row["value"] * 10)

        self.assertEqual(len(dataset), 2)
        self.assertEqual(calls, [])
        self.assertEqual(dataset[1], 20)
        self.assertEqual(calls, [2])

    def test_control_schema_rejects_partial_and_misspelled_columns(self):
        validate_columns([
            "messages", "source_trajectory_id", "curriculum_stage",
            "loss_last_assistant_only",
        ])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_columns(["messages", "source_trajectory_id"])
        with self.assertRaisesRegex(ValueError, "loss_last_asistant_only"):
            validate_columns(["messages", "loss_last_asistant_only"])


if __name__ == "__main__":
    unittest.main()
