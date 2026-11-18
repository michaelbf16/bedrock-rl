"""Lightweight SFT row preparation, independent of torch and trainers."""

from __future__ import annotations

import json
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


class CausalSFTDataset:
    """Index bounded full conversations by action without loading their pixels.

    Rows already carrying a final-action loss mask retain that mask. Other
    rows become one prefix per action, so selecting the latest images cannot
    remove the observation used by an earlier supervised action.
    """

    def __init__(self, rows):
        self.rows = rows
        self.index = []
        # HuggingFace's column view avoids decoding/copying every image just
        # to find assistant boundaries. Plain sequences remain useful to tools.
        if hasattr(rows, "column_names"):
            columns = ["messages"]
            if "loss_last_assistant_only" in rows.column_names:
                columns.append("loss_last_assistant_only")
            metadata = rows.select_columns(columns)
        else:
            metadata = rows
        for row_index, row in enumerate(metadata):
            if row.get("loss_last_assistant_only", False):
                self.index.append((row_index, None))
                continue
            targets = [index for index, message in enumerate(
                json.loads(row["messages"]))
                if message.get("role") == "assistant"]
            if not targets:
                raise ValueError(f"SFT row {row_index} has no assistant")
            self.index.extend((row_index, target) for target in targets)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        from bedrock_rl.adapters.netherite.chat import sft_action_prefix

        row_index, target = self.index[int(index)]
        row = dict(self.rows[row_index])
        if target is not None:
            messages, images = sft_action_prefix(
                json.loads(row["messages"]), row["images"], target)
            row.update(messages=json.dumps(messages), images=images,
                       loss_last_assistant_only=True)
        return row


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
