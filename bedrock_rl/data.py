"""Small serialization helpers shared by dataset and rollout writers."""
import io
import json
import os
from pathlib import Path
from uuid import uuid4


def first_data_file(files):
    """Return the first path from a scalar or JSON-encoded file list.

    Training configs serialize list-valued knobs as JSON before handing them
    to shell launchers. Keeping that boundary here lets launch-time parquet
    checks inspect the dataset the user actually selected instead of falling
    back to a task-derived default.
    """
    value = files
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("data files cannot be empty")
        if text.startswith("["):
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "data files must be a path or a JSON list of paths") from exc
        else:
            value = text
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("data files cannot be empty")
        value = value[0]
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise ValueError("the first data file must be a non-empty path")
    return str(value)


def disabled_validation_table(source):
    """A valid one-row sentinel shaped like an RL parquet.

    Verl requires a non-empty ``val_files`` dataset even when validation is
    disabled. Build a loadable image/prompt row from the selected training
    file's schema, but replace the sample content and episode identity so no
    real training example is also presented as validation data.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(source).slice(0, 1)
    if table.num_rows != 1:
        raise ValueError("RL training data cannot be empty")
    row = table.to_pylist()[0]
    required = {"data_source", "prompt", "extra_info", "images"}
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(
            "RL training data is missing columns needed for validation: "
            + ", ".join(missing))

    row["data_source"] = "bedrock-validation-disabled"
    row["prompt"] = [{
        "role": "user",
        "content": "<image>\nValidation is disabled for this training run.",
    }]
    row["images"] = [{
        "bytes": png_bytes(Image.new("RGB", (16, 16), (0, 0, 0))),
        "path": None,
    }]
    extra = row.get("extra_info") or {}
    extra.update({"seed": -1, "init_yaw": 0.0, "init_pitch": 0.0})
    create = (((extra.get("tools_kwargs") or {}).get("computer") or {})
              .get("create_kwargs"))
    if isinstance(create, dict):
        create.update({"seed": -1, "init_yaw": 0.0, "init_pitch": 0.0})
    row["extra_info"] = extra
    return pa.Table.from_pylist([row], schema=table.schema)


def png_bytes(frame):
    """A rendered frame as PNG bytes, whatever the producer handed over.

    Engine frames arrive already encoded (env.frame) and episode frames
    arrive as PIL (Episode.frame), and one `images` list holds both: the
    t0 frame comes off the env and the per-turn frames come off the
    episode. Three encoders existed and only one of them knew that, so
    the other two would have raised AttributeError on bytes and survived
    only by never being handed any.
    """
    if frame is None or isinstance(frame, (bytes, bytearray)):
        return frame
    buf = io.BytesIO()
    frame.save(buf, format="PNG")
    return buf.getvalue()


def write_rows(rows, out, *, batch_size=32):
    """Rows with an `images` column -> a parquet the trainers can read.

    The cast is the load-bearing part and the reason this is one
    function. Written through HF datasets with an Image() feature, verl's
    RLHFDataset decodes the column back to PIL; a plain pandas parquet
    keeps {bytes, path} dicts, which qwen_vl_utils receives as BytesIO
    and crashes on. Four writers each carried their own copy of the cast,
    which is four chances to leave it out.

    The input is an iterable rather than necessarily a list. Only one bounded
    Arrow batch is resident at a time, which matters for visual trajectories:
    a single row may contain dozens of PNGs. The destination is replaced only
    after every batch succeeds, preserving the generation runner's exact-count
    all-or-nothing contract.
    """
    import datasets
    import pyarrow as pa
    import pyarrow.parquet as pq

    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("parquet batch_size must be at least one")
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f".{output.name}.{os.getpid()}.{uuid4().hex}.tmp")
    iterator = iter(rows)
    writer = None
    count = 0

    def take_batch(first=None):
        batch = [] if first is None else [first]
        for _ in range(batch_size - len(batch)):
            try:
                batch.append(next(iterator))
            except StopIteration:
                break
        return batch

    try:
        try:
            first = next(iterator)
        except StopIteration as exc:
            raise ValueError("cannot write an empty parquet dataset") from exc
        batch = take_batch(first)
        inferred = datasets.Dataset.from_list(batch)
        if "images" in inferred.column_names:
            inferred = inferred.cast_column(
                "images", datasets.Sequence(datasets.Image()))
        features = inferred.features
        schema = features.arrow_schema
        writer = pq.ParquetWriter(str(temporary), schema)
        while batch:
            encoded = [features.encode_example(row) for row in batch]
            writer.write_table(pa.Table.from_pylist(encoded, schema=schema))
            count += len(batch)
            batch = take_batch()
        writer.close()
        writer = None
        os.replace(temporary, output)
    except BaseException:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise
    return count
