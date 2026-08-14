"""Lightweight SFT row preparation, independent of torch and trainers."""

from __future__ import annotations

from collections.abc import Callable, Sequence

CONTROL_COLUMNS = frozenset({
    "source_trajectory_id", "curriculum_stage",
    "loss_last_assistant_only",
})
def validate_columns(columns: Sequence[str]):
    """Reject partial or misspelled trajectory-control schemas."""
    names = set(columns)
    source = {"source_trajectory_id", "curriculum_stage"}
    if names.intersection(source) and not source.issubset(names):
        missing = sorted(source - names)
        raise ValueError(
            "SFT curriculum schema is incomplete; missing "
            + ", ".join(missing))
    suspicious = sorted(
        name for name in names
        if (name.startswith("loss_") or name.startswith("curriculum_"))
        and name not in CONTROL_COLUMNS)
    if suspicious:
        raise ValueError("unknown SFT control column(s): "
                         + ", ".join(suspicious))


class LazyEncodedDataset:
    """Indexable dataset that encodes only the rows a batch requests."""

    def __init__(self, rows, encode: Callable):
        self.rows = rows
        self.encode = encode

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        example = self.encode(self.rows[int(index)])
        if example is None:
            raise ValueError(
                f"SFT row {index} has assistant spans that cannot be "
                "located exactly")
        return example
