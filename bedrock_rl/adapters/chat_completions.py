"""Minimal model adapter for OpenAI-compatible chat-completions servers."""

from __future__ import annotations

import asyncio
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


class ChatCompletionsModel:
    """Call a local vLLM server or any compatible remote endpoint.

    The adapter deliberately depends only on the standard library.
    Authentication, model identity, and sampling remain component
    configuration rather than framework globals.
    """

    def __init__(self, model, base_url="http://127.0.0.1:8000/v1",
                 api_key=None, api_key_env=None, headers=None,
                 timeout=300):
        self.model = str(model)
        base = str(base_url).rstrip("/")
        self.url = (base if base.endswith("/chat/completions") else
                    base + "/chat/completions")
        if api_key is not None:
            self.api_key = str(api_key)
        elif api_key_env:
            # An explicitly named environment variable is an explicit trust
            # decision for this endpoint.  Never send OPENAI_API_KEY to an
            # arbitrary OpenAI-compatible URL merely because it exists.
            self.api_key = os.environ.get(str(api_key_env))
        else:
            parsed = urlsplit(base)
            self.api_key = (os.environ.get("OPENAI_API_KEY")
                            if parsed.scheme == "https"
                            and parsed.hostname == "api.openai.com"
                            else None)
        self.headers = {str(key): str(value)
                        for key, value in (headers or {}).items()}
        self.timeout = float(timeout)

    def _complete(self, messages, tools, sampling):
        blocked = {"model", "messages", "tools"} & set(sampling)
        if blocked:
            raise ValueError("sampling cannot override "
                             + ", ".join(sorted(blocked)))
        payload = {"model": self.model, "messages": messages, **sampling}
        if tools:
            payload["tools"] = tools
        headers = {"Content-Type": "application/json", **self.headers}
        if self.api_key:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")
        request = Request(self.url,
                          data=json.dumps(payload).encode("utf-8"),
                          headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"chat-completions server returned HTTP "
                               f"{exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"cannot reach chat-completions server at "
                               f"{self.url}: {exc.reason}") from exc

    async def complete(self, messages, tools, sampling):
        return await asyncio.to_thread(
            self._complete, messages, tools, dict(sampling or {}))
