"""Verify one registry entry against the real model, no GPU needed.

  bedrock models check qwen3-vl
  bedrock models check /path/to/checkpoint

Run it through the CLI rather than directly: this file imports
transformers, which lives in the TRAINING venv, and `bedrock models
check` is what runs it there. `uv run` is the CPU project environment and
the import fails.

Checks the things that silently ruin a multi-turn computer-use run:
the chat template must render the `computer` tool schema, must render an
assistant tool call in the format the configured parser reads back, and
must turn image content in a tool response into a real image placeholder.
It also measures what one 428x240 frame costs in tokens, and checks the
registry's vision-tower patterns against the checkpoint's parameter
names. Run this whenever you add a ModelSpec.
"""
import json
import os
import re
import sys

from bedrock_rl import lora, models as reg
from bedrock_rl.adapters.netherite.chat import (
    template_messages, tool_schema,
)
from bedrock_rl.core.messages import (
    image_url_part, text_part, validate_transcript,
)
from bedrock_rl.core.tools import ToolCall, ToolResult
from bedrock_rl.env.episode import FRAME_H, FRAME_W

ok = fail = warn = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS {name}")
    else:
        fail += 1
        print(f"  FAIL {name} {detail}")


def note(name, detail=""):
    global warn
    warn += 1
    print(f"  SKIP {name} {detail}")


TOOLS = tool_schema()

CALL = {"actions": [{"action": "mouse_move", "coordinate": [500, 350]},
                    {"action": "key", "k": "w", "ticks": 16}]}
_FRAME_URL = "https://bedrock-rl.invalid/model-check.png"
_CALL_ID = "model_check_0"
CHAIN = validate_transcript([
    {"role": "user", "content": [image_url_part(_FRAME_URL),
                                 text_part("GOAL_MARKER")]},
    {"role": "assistant", "content": None,
     "tool_calls": [ToolCall(_CALL_ID, "computer", CALL).to_openai()]},
    *ToolResult([text_part("OBS_MARKER"),
                 image_url_part(_FRAME_URL)]).messages(
                     _CALL_ID, name="computer"),
])


def _drop_images(chain):
    out = []
    for m in chain:
        if isinstance(m.get("content"), list):
            m = dict(m, content=[p for p in m["content"]
                                 if p.get("type") != "image_url"])
        out.append(m)
    return out


def param_names(path):
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
        return list(json.load(open(idx))["weight_map"].keys())
    one = os.path.join(path, "model.safetensors")
    if os.path.exists(one):
        from safetensors import safe_open
        with safe_open(one, framework="pt") as f:
            return list(f.keys())
    return None


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL")
    if not name:
        raise SystemExit("usage: bedrock models check MODEL\n\n"
                         + reg.supported_table())
    spec = reg.resolve(name)
    path = reg.model_path(name)
    print(f"model    {name}")
    print(f"entry    {spec.key} ({spec.display}, {spec.provider}) "
          f"status={spec.status}")
    print(f"path     {path}")
    print(f"template {spec.chat_template or 'shipped with the model'}")
    print(f"parser   {spec.tool_parser}\n")

    from transformers import AutoProcessor
    # the same gate the loaders use: a family that executes its own repo's
    # python does not do it because a gate script asked politely
    reg.require_remote_code(spec, "check this family")
    proc = AutoProcessor.from_pretrained(
        path, revision=reg.model_revision(path),
        trust_remote_code=reg.trust_remote_code(spec))
    print(f"== processor {type(proc).__name__} ==")
    custom = spec.read_chat_template()
    owner = proc if getattr(proc, "chat_template", None) else proc.tokenizer
    if custom is not None:
        owner.chat_template = custom
    check("has a chat template", getattr(owner, "chat_template", None))

    print("== chat template renders the computer tool ==")
    rendered = owner.apply_chat_template(template_messages(CHAIN), tokenize=False,
                                         add_generation_prompt=True,
                                         tools=TOOLS)
    check("tool schema in prompt", '"name": "computer"' in rendered
          or '"name":"computer"' in rendered)
    check("hermes tool_call block", "<tool_call>" in rendered
          and "</tool_call>" in rendered)
    check("assistant tool call rendered", '"mouse_move"' in rendered)
    check("goal text kept", "GOAL_MARKER" in rendered)
    check("tool response text kept", "OBS_MARKER" in rendered)
    check("no stringified content dicts", "'type': 'image'" not in rendered)

    print("== every turn's frame reaches the model ==")
    from PIL import Image

    from bedrock_rl.models import fit_pixels
    frames = [fit_pixels(Image.new("RGB", (FRAME_W, FRAME_H), (20, 110, 40)),
                         spec),
              fit_pixels(Image.new("RGB", (FRAME_W, FRAME_H), (30, 60, 120)),
                         spec)]
    print(f"  frame {FRAME_W}x{FRAME_H} enters the processor at "
          f"{frames[0].width}x{frames[0].height} "
          f"(band {spec.image_min_pixels}-{spec.image_max_pixels} pixels)")
    try:
        enc = proc(text=[rendered], images=frames, return_tensors="pt")
        n_with = int(enc["input_ids"].shape[1])
        # the same two turns with the image content dropped: the token
        # difference is what the frames themselves cost
        textonly = owner.apply_chat_template(
            template_messages(_drop_images(CHAIN)), tokenize=False,
            add_generation_prompt=True,
            tools=TOOLS)
        n_without = int(proc(text=[textonly],
                             return_tensors="pt")["input_ids"].shape[1])
        per_frame = (n_with - n_without) // len(frames)
        check("processor accepts one frame per turn in one context",
              n_with > n_without)
        checked_images = len(frames)
        image_tokens = per_frame * checked_images
        print(f"  one {FRAME_W}x{FRAME_H} frame costs {per_frame} tokens; "
              f"this {checked_images}-image check costs {image_tokens}")
        check("max_prompt_length leaves room for the checked frames",
              spec.max_prompt_length > image_tokens,
              f"{spec.max_prompt_length} vs {image_tokens} image tokens")
    except Exception as e:
        check("processor accepts one frame per turn in one context", False,
              f"{type(e).__name__}: {e}")

    print(f"== {spec.tool_parser} parser reads the call back ==")
    try:
        import asyncio

        from verl.experimental.agent_loop.tool_parser import ToolParser
        parser = ToolParser.get_tool_parser(spec.tool_parser, proc.tokenizer)
        assistant = ('<tool_call>\n{"name": "computer", "arguments": '
                     + json.dumps(CALL) + "}\n</tool_call>")
        ids = proc.tokenizer(assistant, add_special_tokens=False)["input_ids"]
        _, calls = asyncio.run(parser.extract_tool_calls(ids))
        check("one call parsed", len(calls) == 1, str(calls))
        if calls:
            check("call names the computer tool", calls[0].name == "computer")
            check("arguments round trip",
                  json.loads(calls[0].arguments) == CALL)
    except ImportError:
        note("parser round trip", "(verl not importable in this venv)")

    print("== vision tower patterns ==")
    names = param_names(path) if os.path.isdir(path) else None
    # peft matches a module name, which is a parameter name without its
    # trailing .weight or .bias, and it matches with re.fullmatch.
    def module_of(name):
        return re.sub(r"\.(weight|bias)$", "", name)

    if names is None:
        note("vision_patterns match real parameters",
             "(no local weights; pass a downloaded path to check)")
    else:
        hit = [n for n in names
               if any(p in n for p in spec.vision_patterns)]
        check("vision_patterns match real parameters", hit,
              f"patterns {spec.vision_patterns} matched none of "
              f"{len(names)} parameters")
        if hit:
            print(f"  {len(hit)}/{len(names)} parameters would be frozen")

    # The same patterns, as the regex a LoRA run keeps adapter-free. It
    # is derived rather than declared, so this is where a family whose
    # tower names moved finds out. An exclusion that matched nothing
    # would put adapters on the vision tower, which the rollout engine
    # drops without a word, so the run would train weights it never
    # samples from.
    print("== lora exclusion ==")
    rx = lora.exclude_regex(spec)
    check("the family declares a vision tower to exclude", rx,
          "vision_patterns is empty, so LoRA would adapt the tower")
    if rx and names is not None:
        excluded = [n for n in names if re.fullmatch(rx, module_of(n))]
        check("the exclusion matches real modules", excluded,
              f"{rx} full-matched none of {len(names)} module names")
        if excluded:
            print(f"  {rx}")
            print(f"  {len(excluded)}/{len(names)} parameters would carry "
                  f"no adapter")
    elif rx:
        note("the exclusion matches real modules",
             f"(no local weights; the regex is {rx})")

    print(f"\n{ok} passed, {fail} failed, {warn} skipped")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
