"""Content identities for evaluation inputs, independent of host paths."""

from functools import lru_cache
import hashlib
import io
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from bedrock_rl.env.engine import PATCH_MANIFEST_NAME
from bedrock_rl.env.views import ASSET_MANIFEST_NAME


SCHEMA = "bedrock.eval-protocol.v2"


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, allow_nan=False,
        separators=(",", ":")).encode()).hexdigest()


@lru_cache(maxsize=256)
def _file_digest(path, signature):
    del signature
    checksum = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def file_digest(path):
    """Hash bytes once per file version, including in-place replacements."""
    path = Path(path).resolve()
    stat = path.stat()
    return _file_digest(str(path), (stat.st_dev, stat.st_ino, stat.st_size,
                                    stat.st_mtime_ns, stat.st_ctime_ns))


def image_identity(image):
    """Hash decoded pixels rather than an image filename or PNG metadata."""
    from PIL import Image
    if isinstance(image, dict):
        image = image.get("bytes") or image.get("path")
    if isinstance(image, (bytes, bytearray, memoryview, str, Path)):
        source = (io.BytesIO(bytes(image)) if not isinstance(image, (str, Path))
                  else image)
        with Image.open(source) as opened:
            return image_identity(opened)
    # Palette contents matter too, and the model processors consume RGB.
    pixels = image.convert("RGB")
    return {"size": list(pixels.size),
            "pixels_sha256": hashlib.sha256(pixels.tobytes()).hexdigest()}


def input_identity(row, task_sha256):
    extra = row["extra_info"]
    return digest({
        "prompt": row["prompt"],
        "images": [image_identity(image) for image in row["images"]],
        "task_sha256": task_sha256,
        "seed": int(extra["seed"]),
        "init_yaw": float(extra["init_yaw"]),
        "init_pitch": float(extra["init_pitch"]),
        "view": extra.get("view"),
    })


def runtime_identity():
    """Fingerprint the Python implementation of transitions and scoring."""
    root = Path(__file__).resolve().parents[1]
    files = [*root.joinpath("env").glob("*.py"),
             *root.joinpath("adapters/netherite").glob("*.py"),
             root / "core/context.py", root / "core/sampling.py",
             root / "sdg/guidance.py", root / "eval/trials.py",
             root / "eval/protocol.py"]
    dependencies = {}
    for package in ("transformers", "tokenizers", "vllm", "torch", "Pillow"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    return digest({"sources": {str(path.relative_to(root)): file_digest(path)
                               for path in sorted(files)},
                   "dependencies": dependencies})


def _without_source_paths(value):
    """Loading locations/owner annotations do not change the pixel transform."""
    if isinstance(value, dict):
        location_keys = {"_name_or_path", "name_or_path", "cache_dir",
                         "_processor_class",
                         "tokenizer_file", "vocab_file", "merges_file",
                         "special_tokens_map_file", "tokenizer_config_file"}
        return {key: _without_source_paths(item) for key, item in value.items()
                if key not in location_keys}
    if isinstance(value, (list, tuple)):
        return [_without_source_paths(item) for item in value]
    return value


def processor_identity(processor):
    tokenizer = processor.tokenizer
    backend = getattr(tokenizer, "backend_tokenizer", None)
    return digest({
        "processor": _without_source_paths(processor.to_dict()),
        "chat_template": getattr(processor, "chat_template",
                                 getattr(tokenizer, "chat_template", None)),
        "image_processor": (_without_source_paths(processor.image_processor.to_dict())
                            if hasattr(processor, "image_processor") else None),
        "tokenizer": (json.loads(backend.to_str()) if backend is not None
                      else tokenizer.get_vocab()),
        "special_tokens": tokenizer.special_tokens_map,
    })


def environment_identity(episode):
    """Identify the actual binary, patch/asset stamps and restored snapshot."""
    env = episode.env
    binary = Path(env.binary)
    home = binary.parent.parent
    if home.name == "c":  # Older engine checkouts use c/magma/<binary>.
        home = home.parent

    def manifest(name):
        path = home / name
        return file_digest(path) if path.is_file() else None

    return {
        "binary_sha256": file_digest(binary),
        "patch_manifest_sha256": manifest(PATCH_MANIFEST_NAME),
        "asset_manifest_sha256": manifest(ASSET_MANIFEST_NAME),
        "snapshot_sha256": (file_digest(env.snapshot_in)
                            if env.snapshot_in else None),
        "view": episode.view_provenance,
        "settings": {name: getattr(env, name) for name in (
            "raster", "frame_w", "frame_h", "view_distance", "journal_on",
            "mobs", "world_time")},
    }
