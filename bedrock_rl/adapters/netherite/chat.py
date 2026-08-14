"""Chat-template tokenization for multi-turn computer-use trajectories.

The self-distillation, SFT, and RL paths share this tool schema and renderer so
training and rollout prompts remain token-identical.

Images are expanded to their real pad-token count before tokenizing, so
plain tokenizer calls on message prefixes stay token prefixes of the full
sequence. That is what makes assistant spans exact without re-running
image preprocessing once per prefix.
"""

import copy

from bedrock_rl.adapters.netherite.computer import ACTIONS, TOOL_NAME, cu_help

IMAGE_TAG = "<image>"
# Multi-turn SFT rows keep image bytes in a separate Arrow column.  Their
# message JSON uses this unambiguous internal placeholder so ordinary prose is
# free to contain the model-facing spelling above.
SFT_IMAGE_TAG = "<|netherite_image|>"
_LITERAL_IMAGE_TAG = "NETHERITE_LITERAL_IMAGE_TAG_7CFA2E"
_LITERAL_SFT_IMAGE_TAG = "NETHERITE_LITERAL_SFT_IMAGE_TAG_7CFA2E"


def normalize_legacy_sft_markers(messages, image_count):
    """Normalize image markers from the earlier SFT row schema.

    Earlier rows represented an extracted image as either a leading
    ``<image>`` in string content or an exact ``<image>`` text part. Current
    rows use :data:`SFT_IMAGE_TAG`, leaving ``<image>`` available as literal
    prose. Conversion occurs only when all legacy markers match the separate
    image-column count, making the interpretation unambiguous.
    """
    out = copy.deepcopy(list(messages))
    exact = []
    string_prefixes = []
    has_new = False
    for message in out:
        content = message.get("content")
        if isinstance(content, str):
            if message.get("role") not in ("user", "tool"):
                continue
            count = 0
            offset = 0
            while content.startswith(IMAGE_TAG, offset):
                count += 1
                offset += len(IMAGE_TAG)
            if count:
                string_prefixes.append((message, count, content[offset:]))
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text":
                continue
            if part.get("text") == SFT_IMAGE_TAG:
                has_new = True
            elif part.get("text") == IMAGE_TAG:
                exact.append(part)
    legacy_count = len(exact) + sum(count for _, count, _ in string_prefixes)
    if not has_new and legacy_count == int(image_count):
        for part in exact:
            part["text"] = SFT_IMAGE_TAG
        for message, count, remainder in string_prefixes:
            # Non-writer SFT producers use the same leading ``<image>`` string
            # convention as online rollouts while keeping pixels in a separate
            # column. Convert only after the total marker count proves that
            # interpretation unambiguous. Keeping the remainder as its own text
            # part preserves the byte immediately after the vision span.
            parts = [{"type": "text", "text": SFT_IMAGE_TAG}
                     for _ in range(count)]
            if remainder:
                parts.append({"type": "text", "text": remainder})
            message["content"] = parts
    return out


def _sft_marker_count(messages):
    return sum(
        1
        for message in messages
        for part in (message.get("content")
                     if isinstance(message.get("content"), list) else ())
        if (isinstance(part, dict) and part.get("type") == "text"
            and part.get("text") == SFT_IMAGE_TAG)
    )


def prepare_sft_multiturn(messages, images, *, last_assistant_only=False,
                          keep_latest_images=0):
    """Build a causal, bounded message/image view for one SFT target."""
    if isinstance(keep_latest_images, bool):
        raise TypeError("keep_latest_images must be an integer")
    keep_latest_images = int(keep_latest_images)
    if keep_latest_images < 0:
        raise ValueError("keep_latest_images cannot be negative")
    view = normalize_legacy_sft_markers(messages, len(images))
    if _sft_marker_count(view) != len(images):
        raise ValueError(
            "SFT image markers do not match the separate images column")

    if last_assistant_only:
        try:
            target = next(index for index in range(len(view) - 1, -1, -1)
                          if view[index].get("role") == "assistant")
        except StopIteration as exc:
            raise ValueError("last-assistant-only SFT row has no assistant") \
                from exc
        view = view[:target + 1]

    causal_images = list(images[:_sft_marker_count(view)])
    if keep_latest_images and len(causal_images) > keep_latest_images:
        drop = len(causal_images) - keep_latest_images
        seen = 0
        filtered = copy.deepcopy(view)
        for message in filtered:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            kept = []
            for part in content:
                marker = (isinstance(part, dict)
                          and part.get("type") == "text"
                          and part.get("text") == SFT_IMAGE_TAG)
                if marker:
                    seen += 1
                    if seen <= drop:
                        continue
                kept.append(part)
            message["content"] = kept
        view = filtered
        causal_images = causal_images[-keep_latest_images:]
    return view, causal_images


def template_messages(messages):
    """Normalize canonical OpenAI messages for model chat templates.

    Online rollouts reserve leading ``<image>`` prefixes. SFT rows use a
    dedicated whole-part marker, leaving ordinary text parts unambiguous.
    OpenAI tool-call turns conventionally carry ``content=null``; Qwen3-VL's
    shipped template supports tool calls but tries to iterate null content,
    so the model view uses equivalent empty text while the saved trajectory
    remains canonical.
    """
    out = copy.deepcopy(list(messages))
    for message in out:
        content = message.get("content")
        if isinstance(content, str):
            offset = 0
            if message.get("role") in ("user", "tool"):
                while content.startswith(IMAGE_TAG, offset):
                    offset += len(IMAGE_TAG)
            prefix, remainder = content[:offset], content[offset:]
            remainder = remainder.replace(IMAGE_TAG, _LITERAL_IMAGE_TAG)
            remainder = remainder.replace(
                SFT_IMAGE_TAG, _LITERAL_SFT_IMAGE_TAG)
            message["content"] = prefix + remainder
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "text":
                    continue
                value = str(part.get("text", ""))
                if value == SFT_IMAGE_TAG:
                    continue
                part["text"] = value.replace(
                    IMAGE_TAG, _LITERAL_IMAGE_TAG).replace(
                        SFT_IMAGE_TAG, _LITERAL_SFT_IMAGE_TAG)
        if (message.get("role") == "assistant"
                and message.get("content") is None
                and message.get("tool_calls")):
            message["content"] = ""
    return out


def tool_function():
    """Build the computer schema from the shipped action vocabulary."""
    forms = "; ".join(example for example, _ in ACTIONS)
    return {
        "name": TOOL_NAME,
        "description": (cu_help(20) + "\nCall this repeatedly until the "
                        "task is complete."),
        "parameters": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Actions executed in order: " + forms,
                },
            },
            "required": ["actions"],
        },
    }


def tool_schema():
    """The one `computer` schema every model adapter advertises, in the
    OpenAI tools shape. Rendering a different tools block at training time
    moves the imprint onto a prompt the rollout never sees."""
    return [{"type": "function", "function": tool_function()}]


class Coder:
    def __init__(self, processor, tools=None, chat_template=None, spec=None):
        self.proc = processor
        self.tok = processor.tokenizer
        self.ip = processor.image_processor
        self.tools = tools if tools is not None else tool_schema()
        # the family's tool-call template from bedrock_rl/models.py, so
        # every consumer renders the bytes the rollout renders
        self.chat_template = chat_template
        self.spec = spec
        # How one image expands before tokenization. Each family writes
        # its own span, so a hardcoded one produces a training view with
        # no image in it for every other family and says nothing.
        # Refusing is the only safe default: a wrong span is silent and a
        # missing one is not.
        self.image_span = tuple(getattr(spec, "image_span", ()) or ())
        if spec is not None and len(self.image_span) != 3:
            raise ValueError(
                f"model family {spec.key!r} declares no image_span, so an "
                "<image> marker cannot be expanded into its vision tokens. "
                "Add image_span=(prefix, pad, suffix) to its ModelSpec in "
                "bedrock_rl/models.py; the family's own chat template "
                "under bedrock_rl/templates/chat_template/ shows the "
                "three tokens.")

    def _family(self):
        """What to call this model in an error, from the registry when
        there is one, so a refusal names the entry a reader has to edit
        rather than a class name they then have to map back."""
        key = getattr(self.spec, "key", None)
        return key or type(self.proc).__name__

    def encode_images(self, images):
        """Pixel values, grid, and the pad-token count per image.

        Supported processors provide ``pixel_values``, one ``image_grid_thw``
        row per image, and an integer ``merge_size`` used to calculate vision
        pad tokens. Validation happens here so every trainer rejects an
        unsupported model family with the same actionable error.
        """
        if not images:
            return None, None, []
        if self.spec is not None:
            from bedrock_rl import models as reg
            images = [reg.fit_pixels(im, self.spec) for im in images]
        out = self.ip(images=images, return_tensors="pt")
        merge = getattr(self.ip, "merge_size", None)
        missing = [k for k in ("pixel_values", "image_grid_thw")
                   if k not in out]
        # An int, because it squares into a token count below. A family
        # that spells its merge factor as a pair belongs in the same
        # refusal as one that has none: both mean the pad-token count
        # cannot be derived the way every consumer here expects.
        if not isinstance(merge, int) or isinstance(merge, bool):
            missing.append("merge_size")
        if missing:
            raise SystemExit(
                f"model family {self._family()!r} does not produce the "
                f"image encoding this repo's trainers read: "
                f"{type(self.proc).__name__} returned no "
                f"{', '.join(repr(m) for m in missing)}. Every consumer of "
                f"a rendered trajectory (the SFT collator, the SDPO "
                f"trainer, the self-distillation harvester) reads pixel_values, "
                f"image_grid_thw and merge_size out of one call, so this "
                f"family cannot run multimodal SFT or self-distil until that call "
                f"is taught to answer for it. RL from base is unaffected "
                f"and runs through verl (bin/rl.sh).")
        grid = out["image_grid_thw"]
        n = (grid.prod(-1) // (merge ** 2)).tolist()
        return out["pixel_values"], grid, n

    def _expand(self, text, npads):
        """Replace each <image> marker with the family's real vision span.

        Doing it before tokenizing is what makes a rendered prefix a TOKEN
        prefix of the full sequence, which is what makes the assistant
        mask exact without re-running image preprocessing per prefix."""
        if (IMAGE_TAG in text or SFT_IMAGE_TAG in text) and not self.image_span:
            raise ValueError(
                "no image_span for this model, so <image> cannot be "
                "expanded; construct Coder with spec=<ModelSpec>")
        i = 0
        while IMAGE_TAG in text or SFT_IMAGE_TAG in text:
            positions = [(text.find(tag), tag)
                         for tag in (IMAGE_TAG, SFT_IMAGE_TAG)
                         if tag in text]
            position, tag = min(positions)
            if i >= len(npads):
                raise ValueError("rendered prompt contains more image "
                                 "placeholders than supplied images")
            start, pad, end = self.image_span
            span = start + pad * npads[i] + end
            text = text[:position] + span + text[position + len(tag):]
            i += 1
        return text

    def render(self, messages, npads, add_generation_prompt=False):
        messages = template_messages(messages)
        kw = {} if self.chat_template is None else {
            "chat_template": self.chat_template}
        rendered = self._expand(self.tok.apply_chat_template(
            messages, tools=self.tools, tokenize=False,
            add_generation_prompt=add_generation_prompt, **kw), npads)
        return rendered.replace(_LITERAL_IMAGE_TAG, IMAGE_TAG).replace(
            _LITERAL_SFT_IMAGE_TAG, SFT_IMAGE_TAG)

    def ids(self, messages, npads, add_generation_prompt=False):
        return self.tok(self.render(messages, npads, add_generation_prompt),
                        add_special_tokens=False).input_ids

    def encode_trajectory(self, messages, npads, assistant_indices=None):
        """Full token ids plus a mask marking assistant-generated tokens.

        The mask spans from the end of the generation prompt to the turn
        terminator, which is exactly what the model sampled. Returns
        (None, None) when a rendered prefix is not a token prefix of the
        full sequence, because the mask would then be a lie."""
        full = self.ids(messages, npads)
        selected = (None if assistant_indices is None
                    else {int(index) for index in assistant_indices})
        mask = [0] * len(full)
        for k, m in enumerate(messages):
            if (m["role"] != "assistant"
                    or (selected is not None and k not in selected)):
                continue
            pre = self.ids(messages[:k], npads, add_generation_prompt=True)
            upto = self.ids(messages[:k + 1], npads)
            if full[:len(pre)] != pre or full[:len(upto)] != upto:
                return None, None
            # the template closes the turn with the terminator then a
            # newline; the newline is template, not sampled
            end = len(upto)
            while (end > len(pre)
                   and self.tok.decode([full[end - 1]]).strip() == ""):
                end -= 1
            for j in range(len(pre), end):
                mask[j] = 1
        return full, mask

    def sampled_view(self, prompt_ids, response_ids, teacher_messages,
                     image_pads):
        """Student/teacher sequences over exact sampled response IDs.

        ``encode_trajectory`` is for imported OpenAI transcripts whose only
        source of tokens is rendering.  Online distillation has a stronger
        source: the sampler's token IDs.  Appending those IDs to each prompt
        avoids parsing and re-serializing assistant prose/tool JSON between
        rollout and loss.
        """
        response = [int(token) for token in response_ids]
        if not response:
            return None
        student_prompt = [int(token) for token in prompt_ids]
        teacher_prompt = self.ids(
            list(teacher_messages), list(image_pads),
            add_generation_prompt=True)
        student = student_prompt + response
        teacher = teacher_prompt + response
        return {
            "student": (student, [0] * len(student_prompt)
                        + [1] * len(response)),
            "teacher": (teacher, [0] * len(teacher_prompt)
                        + [1] * len(response)),
        }
