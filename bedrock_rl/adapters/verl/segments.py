"""Token-exact context segments for dynamic views in verl rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenSegment:
    kind: str
    token_ids: list[int]
    image_ids: tuple[str, ...] = ()
    images: tuple[Any, ...] = ()
    without_images: list[int] | None = None
    sampled_image_ids: tuple[str, ...] = ()
    logprobs: list[float] = field(default_factory=list)
    image_variants: dict[tuple[str, ...], list[int]] = field(
        default_factory=dict)

    def __post_init__(self):
        self.token_ids = list(self.token_ids)
        self.image_ids = tuple(self.image_ids)
        self.images = tuple(self.images)
        self.sampled_image_ids = tuple(self.sampled_image_ids)
        self.logprobs = list(self.logprobs)
        self.image_variants = {
            tuple(ids): list(tokens)
            for ids, tokens in self.image_variants.items()
        }
        if len(self.image_ids) != len(self.images):
            raise ValueError("one image object is required per image id")
        if self.without_images is not None:
            self.without_images = list(self.without_images)
        if self.kind == "assistant" and self.logprobs and (
                len(self.logprobs) != len(self.token_ids)):
            raise ValueError("assistant logprobs must align with token ids")

    def ids_for(self, selected: set[str]) -> list[int]:
        kept = tuple(image for image in self.image_ids if image in selected)
        if not self.image_ids or kept == self.image_ids:
            return self.token_ids
        if not kept:
            if self.without_images is None:
                raise ValueError(f"{self.kind} segment has removable images "
                                 "but no image-free tokenization")
            return self.without_images
        variant = self.image_variants.get(kept)
        if variant is None:
            raise ValueError(
                f"{self.kind} segment has no exact tokenization for retained "
                f"images {kept!r}; available partial variants are "
                f"{sorted(self.image_variants)!r}")
        return variant


def selected_image_ids(segments: list[TokenSegment],
                       keep_last: int | None) -> tuple[str, ...]:
    ordered = [image_id for segment in segments
               for image_id in segment.image_ids]
    if keep_last is None:
        return tuple(ordered)
    keep_last = int(keep_last)
    if keep_last < 0:
        raise ValueError("keep_last must be non-negative or None")
    return tuple(ordered[-keep_last:]) if keep_last else ()


def render_segments(segments: list[TokenSegment],
                    keep_last: int | None):
    selected_ids = selected_image_ids(segments, keep_last)
    selected = set(selected_ids)
    token_ids = [token for segment in segments
                 for token in segment.ids_for(selected)]
    image_by_id = {image_id: image for segment in segments
                   for image_id, image in zip(segment.image_ids,
                                              segment.images)}
    return token_ids, [image_by_id[image_id] for image_id in selected_ids], selected_ids


def finalize_segments(segments: list[TokenSegment],
                      keep_last: int | None):
    """Return prompt, response, mask, logprobs, images, and masked turns.

    When a dynamic image window changes an earlier assistant's causal
    context, that assistant remains useful history but is masked out of the
    loss.  Tokens are trained only when the images before them in the final
    sequence exactly match the images present when those tokens were sampled.
    """
    if not segments or segments[0].kind != "initial":
        raise ValueError("token stream must begin with an initial segment")
    final_ids = selected_image_ids(segments, keep_last)
    if keep_last is not None:
        # A terminal tool result is observed after the final assistant action.
        # It was never part of an assistant's sampling context, so allowing its
        # frame to displace that context can mask every trainable token in a
        # successful episode.  Anchor a bounded training window to the most
        # recent context that was actually sampled instead.
        final_assistant = next(
            (segment for segment in reversed(segments)
             if segment.kind == "assistant"), None)
        if final_assistant is not None:
            final_ids = final_assistant.sampled_image_ids
    selected = set(final_ids)
    prompt_ids = segments[0].ids_for(selected)
    response_ids = []
    response_mask = []
    response_logprobs = []
    masked_assistants = 0
    prior_images = []
    prior_images.extend(image for image in segments[0].image_ids
                        if image in selected)
    for segment in segments[1:]:
        ids = segment.ids_for(selected)
        response_ids.extend(ids)
        if segment.kind == "assistant":
            exact = tuple(prior_images) == segment.sampled_image_ids
            response_mask.extend([1 if exact else 0] * len(ids))
            if segment.logprobs and exact:
                response_logprobs.extend(segment.logprobs)
            else:
                response_logprobs.extend([0.0] * len(ids))
            if not exact:
                masked_assistants += 1
        else:
            response_mask.extend([0] * len(ids))
            response_logprobs.extend([0.0] * len(ids))
        prior_images.extend(image for image in segment.image_ids
                            if image in selected)
    image_by_id = {image_id: image for segment in segments
                   for image_id, image in zip(segment.image_ids,
                                              segment.images)}
    images = [image_by_id[image_id] for image_id in final_ids]
    assistant_tokens = sum(len(segment.ids_for(selected))
                           for segment in segments
                           if segment.kind == "assistant")
    if assistant_tokens and not any(response_mask):
        raise ValueError(
            "all assistant tokens were masked by the final image window")
    return (prompt_ids, response_ids, response_mask, response_logprobs,
            images, masked_assistants)
