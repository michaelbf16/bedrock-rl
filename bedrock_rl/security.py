"""Narrow security backport for the Transformers 4.x training stack.

Upstream fix: huggingface/transformers@eaaaf8494dd5386634ae37d1d122212fdc315be5
(CVE-2026-9856). vLLM requires Transformers <5; the final 4.x release lacks
the fix. Native/Docker setup patches the installed files. The save guards
also protect standalone ``uv run --extra train`` installations and Trainer
callbacks, without changing the lightweight package import contract.
"""

from functools import wraps
from pathlib import Path


def validate_chat_templates(owner, save_directory, filename_prefix=None):
    """Reject untrusted template names before any checkpoint files are saved."""
    directory = Path(save_directory) / (
        (str(filename_prefix) + "-" if filename_prefix else "")
        + "additional_chat_templates")
    templates = getattr(owner, "chat_template", None)
    if isinstance(templates, list):
        templates = {item["name"]: item["template"] for item in templates}
    if isinstance(templates, dict):
        for name in templates:
            if name == "default":
                continue
            if (not isinstance(name, str) or "/" in name or "\\" in name
                    or (directory / f"{name}.jinja").resolve().parent
                    != directory.resolve()):
                raise ValueError(f"Invalid chat template name: {name!r}")
    tokenizer = getattr(owner, "tokenizer", None)
    if tokenizer is not None and tokenizer is not owner:
        validate_chat_templates(tokenizer, save_directory, filename_prefix)


def guard_transformers_saves():
    """Apply the upstream validation to both save APIs, once per process."""
    from transformers import PreTrainedTokenizerBase, ProcessorMixin

    def guarded(original):
        @wraps(original)
        def save(self, save_directory, *args, **kwargs):
            validate_chat_templates(
                self, save_directory, kwargs.get("filename_prefix"))
            return original(self, save_directory, *args, **kwargs)
        save._bedrock_template_guard = True
        return save

    for cls in (PreTrainedTokenizerBase, ProcessorMixin):
        if not getattr(cls.save_pretrained, "_bedrock_template_guard", False):
            cls.save_pretrained = guarded(cls.save_pretrained)


def patch_installed_transformers():
    """Persist the upstream checks for third-party verl save paths too.

    Match the pinned source exactly, fail on drift, and accept an already
    patched environment. No network fetch or unverified patch is performed.
    """
    import importlib.metadata
    import importlib.util

    version = importlib.metadata.version("transformers")
    if version != "4.57.6":
        raise RuntimeError(f"review the Transformers backport for {version}")
    root = Path(importlib.util.find_spec("transformers").origin).parent
    marker = 'template_filepath = os.path.join(chat_template_dir, f"{template_name}.jinja")'
    check = ('if Path(template_filepath).resolve().parent '
             '!= Path(chat_template_dir).resolve():')
    changes = []
    for name in ("processing_utils.py", "tokenization_utils_base.py"):
        target = root / name
        source = target.read_text()
        if check in source:
            continue
        if source.count(marker) != 1 or "from pathlib import Path" not in source:
            raise RuntimeError(f"unrecognized Transformers source: {target}")
        lines = source.splitlines(keepends=True)
        index = next(i for i, line in enumerate(lines) if marker in line)
        indent = lines[index][:len(lines[index]) - len(lines[index].lstrip())]
        lines[index + 1:index + 1] = [
            indent + "# CVE-2026-9856: upstream template-name validation.\n",
            indent + check + "\n",
            indent + '    raise ValueError(f"Invalid chat template name: {template_name!r}")\n',
        ]
        patched = "".join(lines)
        compile(patched, str(target), "exec")
        changes.append((target, patched))
    for target, patched in changes:
        target.write_text(patched)
    print(f"Transformers {version}: template path validation verified")


if __name__ == "__main__":
    patch_installed_transformers()
