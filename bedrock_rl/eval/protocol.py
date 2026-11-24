"""Content identities for evaluation inputs, independent of host paths."""

import hashlib
from importlib import import_module
from importlib.machinery import EXTENSION_SUFFIXES
import inspect
import io
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from bedrock_rl.resources import file_digest
from bedrock_rl.env.engine import PATCH_MANIFEST_NAME
from bedrock_rl.env.views import ASSET_MANIFEST_NAME


SCHEMA = "bedrock.eval-protocol.v3"


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, allow_nan=False,
        separators=(",", ":")).encode()).hexdigest()


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


def verifier_identity(task):
    """Bind custom scoring code, including helpers shipped in its package."""
    from bedrock_rl.core.components import load_object

    packages = set()
    for check in (task.success, task.fail, *task.shaping):
        if check is None:
            continue
        if check.custom is None:
            # The built-in modules are covered by runtime_identity. Extension
            # modules may also install a named predicate using @check_type.
            if check.fn.__module__ in ("bedrock_rl.env.task", "bedrock_rl.env.journal"):
                continue
            implementations = (check.fn,)
        else:
            packages.add(check.type.partition(":")[0].partition(".")[0])
            implementations = (load_object(check.type),
                               getattr(check.custom, "holds", check.custom))
        for obj in implementations:
            module = inspect.getmodule(obj) or inspect.getmodule(type(obj))
            if module is None:
                raise ValueError(f"cannot identify verifier {check.type!r}")
            packages.add(module.__name__.split(".")[0])

    declared = task.verifier_dependencies
    verified = not packages or declared is not None
    dependencies = declared or {}
    packages.update(dependencies.get("modules", []))
    sources = {}
    for name in sorted(packages):
        package = import_module(name)
        locations = getattr(package, "__path__", None)
        if locations is None:
            origin = getattr(package, "__file__", None)
            if not origin or not Path(origin).is_file():
                raise ValueError(f"verifier module {name!r} has no hashable file")
            sources[name] = file_digest(origin)
            continue
        files = {}
        for location in locations:
            root = Path(location)
            for path in root.rglob("*"):
                if path.name.endswith((".py", *EXTENSION_SUFFIXES)) and path.is_file():
                    key = path.relative_to(root).as_posix()
                    if key in files:
                        raise ValueError(f"ambiguous verifier package file {name}/{key}")
                    files[key] = file_digest(path)
        if not files:
            raise ValueError(f"verifier package {name!r} has no hashable code")
        sources[name] = files
    return digest({"modules": sources, "files": task.verifier_file_digests()}), verified


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
