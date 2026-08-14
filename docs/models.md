# Models

`model:` accepts a registry alias, a Hugging Face model ID from a supported
family, or a local checkpoint:

```yaml
model: qwen3-vl-2b
# model: Qwen/Qwen3-VL-8B-Instruct
# model: /checkpoints/my-agent
```

Inspect registry selection without using a GPU or a training environment:

```bash
uv run bedrock models list
uv run bedrock models resolve qwen3-vl-2b
```

The deeper check loads the real processor and verl tool parser. It still uses
no GPU, but it runs in the pinned training environment created by full setup:

```bash
uv run bedrock setup
uv run bedrock models check qwen3-vl-2b
```

Local full checkpoints are matched by `model_type` in `config.json`. LoRA
adapter directories resolve through their recorded base model, so an SFT
adapter can be passed directly to another trainer.

## Add a model family

Add one `ModelSpec` to `MODELS` in `bedrock_rl/models.py`. Do not add
model-name conditionals to trainers.

```python
ModelSpec(
    key="my-vlm",
    display="My VLM",
    provider="Example",
    hub="https://huggingface.co/example",
    ids={"3B": "example/my-vlm-3b"},
    revisions={"example/my-vlm-3b": "FULL_HUB_COMMIT"},
    model_types=("my_vlm",),
    id_prefixes=("example/my-vlm-",),
    aliases=("myvlm",),
    image_span=("<vision>", "<image>", "</vision>"),
    vision_patterns=(".vision_tower.",),
    chat_template="my_vlm_tools.jinja",  # or None for the model's template
    status="untested",
)
```

The entry must define:

- how the processor represents an image (`image_span`);
- a chat template that renders the `computer` schema, assistant tool calls,
  tool results, and images exactly as rollout does;
- the model's Transformers `model_type` and vision-module patterns;
- immutable revisions for shipped Hub aliases.

If the model needs a custom template, put it in
`bedrock_rl/templates/chat_template/`. If it needs a predictable verl
override, use `verl_overrides` or `verl_override_fn` on the registry entry.

Then run:

```bash
uv run bedrock models check my-vlm
uv run python -m unittest tests.test_package_architecture tests.test_consistency_regressions
uv run bedrock train path/to/learner.yaml --set model=my-vlm --dry-run
```

Mark a family `tested` only after a real multimodal tool-call rollout and
training step succeed. Registry validation alone is not an end-to-end model
verification.
