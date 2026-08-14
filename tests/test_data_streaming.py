import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from bedrock_rl.data import (disabled_validation_table, first_data_file,
                             write_rows)


class DataFileSelectionTests(unittest.TestCase):
    def test_first_data_file_accepts_scalar_and_serialized_list(self):
        self.assertEqual(first_data_file("/tmp/train.parquet"),
                         "/tmp/train.parquet")
        self.assertEqual(
            first_data_file('["/mnt/custom.parquet", "/mnt/other.parquet"]'),
            "/mnt/custom.parquet")

    def test_first_data_file_rejects_empty_or_malformed_lists(self):
        for value in ("", "[]", "[not-json]", []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                first_data_file(value)

    def test_disabled_validation_table_is_valid_but_not_training_data(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is optional")
        row = {
            "data_source": "bedrock",
            "prompt": [{"role": "user", "content": "<image>\nreal row"}],
            "extra_info": {
                "seed": 41,
                "init_yaw": 12.0,
                "init_pitch": 3.0,
                "tools_kwargs": {"computer": {"create_kwargs": {
                    "seed": 41, "init_yaw": 12.0, "init_pitch": 3.0,
                }}},
            },
            "images": [{"bytes": b"training-image", "path": None}],
        }
        source_table = pa.Table.from_pylist([row])
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.parquet"
            pq.write_table(source_table, source)
            table = disabled_validation_table(source)

        sentinel = table.to_pylist()[0]
        self.assertEqual(table.schema, source_table.schema)
        self.assertEqual(table.num_rows, 1)
        self.assertEqual(sentinel["data_source"],
                         "bedrock-validation-disabled")
        self.assertNotEqual(sentinel["prompt"], row["prompt"])
        self.assertNotEqual(sentinel["images"], row["images"])
        self.assertEqual(sentinel["extra_info"]["seed"], -1)
        self.assertEqual(
            sentinel["extra_info"]["tools_kwargs"]["computer"]
            ["create_kwargs"]["seed"], -1)


class _Features:
    arrow_schema = "test-schema"

    @staticmethod
    def encode_example(row):
        if row.get("explode"):
            raise RuntimeError("bad late row")
        return dict(row)


class _DatasetValue:
    def __init__(self, rows, observed):
        observed.append(len(rows))
        self.column_names = list(rows[0])
        self.features = _Features()

    def cast_column(self, _name, _feature):
        return self


def _fake_modules(observed, batches):
    datasets = types.ModuleType("datasets")
    datasets.Dataset = types.SimpleNamespace(
        from_list=lambda rows: _DatasetValue(rows, observed))
    datasets.Sequence = lambda feature: ("sequence", feature)
    datasets.Image = lambda: "image"

    pyarrow = types.ModuleType("pyarrow")
    pyarrow.__path__ = []
    pyarrow.Table = types.SimpleNamespace(
        from_pylist=lambda rows, schema: (list(rows), schema))
    parquet = types.ModuleType("pyarrow.parquet")

    class Writer:
        def __init__(self, path, schema):
            self.path = Path(path)
            self.path.write_bytes(b"")
            self.schema = schema

        def write_table(self, table):
            rows, schema = table
            if schema != self.schema:
                raise AssertionError("schema changed between batches")
            batches.append(len(rows))
            with self.path.open("ab") as stream:
                stream.write(bytes([len(rows)]))

        def close(self):
            pass

    parquet.ParquetWriter = Writer
    pyarrow.parquet = parquet
    return {"datasets": datasets, "pyarrow": pyarrow,
            "pyarrow.parquet": parquet}


class StreamingParquetTests(unittest.TestCase):
    def test_iterable_is_written_in_bounded_batches(self):
        observed, batches, consumed = [], [], []

        def rows():
            for index in range(10):
                consumed.append(index)
                yield {"index": index, "images": []}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rows.parquet"
            with patch.dict(sys.modules,
                            _fake_modules(observed, batches)):
                count = write_rows(rows(), output, batch_size=3)

            self.assertEqual(count, 10)
            self.assertEqual(consumed, list(range(10)))
            self.assertEqual(observed, [3])
            self.assertEqual(batches, [3, 3, 3, 1])
            self.assertEqual(output.read_bytes(), bytes([3, 3, 3, 1]))

    def test_late_failure_preserves_existing_output(self):
        observed, batches = [], []
        rows = ({"index": index, "explode": index == 4}
                for index in range(7))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rows.parquet"
            output.write_bytes(b"previous-complete-dataset")
            with patch.dict(sys.modules,
                            _fake_modules(observed, batches)):
                with self.assertRaisesRegex(RuntimeError, "bad late row"):
                    write_rows(rows, output, batch_size=3)

            self.assertEqual(output.read_bytes(),
                             b"previous-complete-dataset")
            self.assertEqual(
                list(output.parent.glob(f".{output.name}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
