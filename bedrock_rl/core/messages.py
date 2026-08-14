"""OpenAI-chat message helpers used at every framework boundary."""

from __future__ import annotations

import base64
import binascii
import copy
import json
from typing import Any, Iterable, Mapping

ROLES = frozenset(("system", "developer", "user", "assistant", "tool"))
IMAGE_PART_TYPES = frozenset(("image", "image_url", "input_image"))
EMBEDDED_IMAGE_TYPES = frozenset(
    ("image/png", "image/jpeg", "image/gif", "image/webp"))


def copy_messages(messages: Iterable[Mapping[str, Any]]) -> list[dict]:
    return copy.deepcopy([dict(message) for message in messages])


def is_image_part(part: Any) -> bool:
    return isinstance(part, Mapping) and part.get("type") in IMAGE_PART_TYPES


def image_url_part(url: str, *, detail: str | None = None) -> dict:
    image = {"url": str(url)}
    if detail is not None:
        if detail not in ("auto", "low", "high"):
            raise ValueError("image detail must be auto, low, or high")
        image["detail"] = detail
    return {"type": "image_url", "image_url": image}


def _validate_image_bytes(data: bytes, mime_type: str, where: str) -> bytes:
    mime = str(mime_type).lower()
    if mime == "image/jpg":
        mime = "image/jpeg"
    if mime not in EMBEDDED_IMAGE_TYPES:
        raise ValueError(
            f"{where} uses unsupported embedded image type {mime_type!r}; "
            f"use one of {sorted(EMBEDDED_IMAGE_TYPES)}")
    valid = {
        "image/png": data.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/jpeg": data.startswith(b"\xff\xd8\xff"),
        "image/gif": data.startswith((b"GIF87a", b"GIF89a")),
        "image/webp": (len(data) >= 12 and data[:4] == b"RIFF"
                        and data[8:12] == b"WEBP"),
    }[mime]
    if not valid:
        raise ValueError(
            f"{where} payload does not match its declared {mime} type")
    return data


def decode_data_image_url(url: str, where: str = "image") -> bytes:
    """Decode and validate a portable model image data URL."""
    if not isinstance(url, str) or not url.startswith("data:"):
        raise ValueError(f"{where} must be an embedded image data URL")
    try:
        header, payload = url.split(",", 1)
    except ValueError as exc:
        raise ValueError(f"{where} data URL has no payload") from exc
    fields = header[5:].split(";")
    mime = fields[0].lower()
    if "base64" not in {value.lower() for value in fields[1:]}:
        raise ValueError(f"{where} must use a base64 image data URL")
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{where} has invalid base64 image data") from exc
    if not decoded:
        raise ValueError(f"{where} has an empty image payload")
    return _validate_image_bytes(decoded, mime, where)


def data_image_part(data: bytes, mime_type: str = "image/png",
                    *, detail: str | None = None) -> dict:
    data = _validate_image_bytes(bytes(data), mime_type, "image")
    encoded = base64.b64encode(data).decode("ascii")
    return image_url_part(f"data:{mime_type};base64,{encoded}", detail=detail)


def text_part(text: str) -> dict:
    return {"type": "text", "text": str(text)}


def validate_message(message: Mapping[str, Any], index: int | None = None):
    where = f"message {index}" if index is not None else "message"
    if not isinstance(message, Mapping):
        raise TypeError(f"{where} must be a mapping")
    role = message.get("role")
    if role not in ROLES:
        raise ValueError(f"{where} has role {role!r}; expected one of "
                         f"{sorted(ROLES)}")
    content = message.get("content")
    if content is not None and not isinstance(content, (str, list)):
        raise TypeError(f"{where}.content must be text, a content-part list, "
                        "or null")
    if isinstance(content, list):
        for part_index, part in enumerate(content):
            if not isinstance(part, Mapping) or not part.get("type"):
                raise TypeError(f"{where}.content[{part_index}] must be a "
                                "typed content-part mapping")
            part_where = f"{where}.content[{part_index}]"
            if part.get("type") in ("image", "input_image"):
                raise ValueError(
                    f"{part_where} uses runtime part type "
                    f"{part.get('type')!r}; canonical OpenAI chat uses "
                    "image_url with an image_url.url")
            if part.get("type") == "text" and not isinstance(
                    part.get("text"), str):
                raise TypeError(f"{part_where}.text must be text")
            if part.get("type") == "image_url":
                if role != "user":
                    raise ValueError(
                        f"{part_where} is on role {role!r}; OpenAI Chat "
                        "Completions accepts image_url input only on user "
                        "messages")
                image = part.get("image_url")
                if not isinstance(image, Mapping):
                    raise TypeError(f"{part_where}.image_url must be a "
                                    "mapping with a url")
                url = image.get("url")
                if not isinstance(url, str) or not url:
                    raise TypeError(f"{part_where}.image_url.url must be "
                                    "non-empty text")
                if url.startswith("data:"):
                    decode_data_image_url(url, part_where)
                elif not url.startswith(("https://", "http://")):
                    raise ValueError(
                        f"{part_where}.image_url.url must be an http(s) URL "
                        "or embedded image data URL")
                detail = image.get("detail")
                if detail is not None and detail not in ("auto", "low",
                                                          "high"):
                    raise ValueError(f"{part_where}.image_url.detail must be "
                                     "auto, low, or high")
    if role == "tool" and not message.get("tool_call_id"):
        raise ValueError(f"{where} is a tool result without tool_call_id")
    # OpenAI clients commonly materialize an absent optional field as null.
    # Treat that exactly like an omitted field instead of trying to iterate
    # None and killing a whole batch at a validation boundary.
    calls = message.get("tool_calls") or ()
    if calls and role != "assistant":
        raise ValueError(f"{where} carries tool_calls but is not assistant")
    for call_index, call in enumerate(calls):
        if not isinstance(call, Mapping):
            raise TypeError(f"{where}.tool_calls[{call_index}] must be a "
                            "mapping")
        fn = call.get("function")
        if (call.get("type", "function") != "function"
                or not call.get("id") or not isinstance(fn, Mapping)
                or not fn.get("name") or "arguments" not in fn):
            raise ValueError(f"{where}.tool_calls[{call_index}] is not an "
                             "OpenAI function call")
        arguments = fn["arguments"]
        if not isinstance(arguments, (str, Mapping)):
            raise TypeError(f"{where}.tool_calls[{call_index}].function."
                            "arguments must be JSON text or a mapping")
        if isinstance(arguments, str):
            try:
                decoded = json.loads(arguments)
            except ValueError as exc:
                raise ValueError(
                    f"{where}.tool_calls[{call_index}] has invalid JSON "
                    f"arguments: {exc}") from exc
            if not isinstance(decoded, Mapping):
                raise ValueError(f"{where}.tool_calls[{call_index}] JSON "
                                 "arguments must decode to an object")
    return message


def validate_messages(messages: Iterable[Mapping[str, Any]]) -> list[dict]:
    out = copy_messages(messages)
    for index, message in enumerate(out):
        validate_message(message, index)
    return out


def validate_transcript(messages: Iterable[Mapping[str, Any]]) -> list[dict]:
    """Validate messages plus OpenAI tool-call/result relationships."""
    out = validate_messages(messages)
    pending: dict[str, str] = {}
    seen = set()
    for index, message in enumerate(out):
        role = message["role"]
        if role != "tool" and pending:
            raise ValueError(
                f"message {index} arrives before tool results for "
                f"{sorted(pending)}")
        if role == "assistant":
            for call in (message.get("tool_calls") or ()):
                call_id = str(call["id"])
                if call_id in seen:
                    raise ValueError(f"duplicate tool call id {call_id!r}")
                seen.add(call_id)
                pending[call_id] = str(call["function"]["name"])
        elif role == "tool":
            call_id = str(message["tool_call_id"])
            if call_id not in pending:
                raise ValueError(
                    f"message {index} is a tool result for unknown or "
                    f"already-resolved call {call_id!r}")
            name = message.get("name")
            if name is not None and str(name) != pending[call_id]:
                raise ValueError(
                    f"message {index} names tool {name!r}, but call "
                    f"{call_id!r} requested {pending[call_id]!r}")
            del pending[call_id]
    if pending:
        raise ValueError(
            f"trajectory ends without tool results for {sorted(pending)}")
    return out


def image_locations(messages: Iterable[Mapping[str, Any]]) -> list[tuple[int, int]]:
    out = []
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part_index, part in enumerate(content):
            if is_image_part(part):
                out.append((message_index, part_index))
    return out
