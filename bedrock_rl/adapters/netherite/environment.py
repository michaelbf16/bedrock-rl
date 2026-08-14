"""The existing deterministic Episode exposed as a core environment session."""

from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import Any, Mapping

from bedrock_rl.adapters.netherite.chat import IMAGE_TAG
from bedrock_rl.adapters.netherite.computer import cu_parser, cu_user_prompt
from bedrock_rl.core.messages import data_image_part, text_part
from bedrock_rl.core.state import StateEvent
from bedrock_rl.env.episode import Episode
from bedrock_rl.env.task import Task as NetheriteTask


class NetheriteSession:
    def __init__(self, episode: Episode, task: NetheriteTask,
                 *, use_task_prompt: bool = True):
        self.episode = episode
        self.task = task
        self.use_task_prompt = bool(use_task_prompt)
        self._event_index = 0

    @property
    def done(self):
        return self.episode.done

    def initial_messages(self, task):
        if self.use_task_prompt:
            prompt = cu_user_prompt(self.task, self.episode.env)
            if prompt.startswith(IMAGE_TAG):
                prompt = prompt[len(IMAGE_TAG):]
        else:
            prompt = task.instruction
        content = []
        if self.episode.t0_png is not None:
            content.append(data_image_part(self.episode.t0_png))
        content.append(text_part(prompt))
        return [{"role": "user", "content": content}]

    def drain_events(self):
        raw = self.episode.env.journal[self._event_index:]
        self._event_index += len(raw)
        events = []
        for event in raw:
            value = event.as_dict()
            name = value.pop("event")
            tick = value.pop("tick")
            events.append(StateEvent(name, value, int(tick)))
        return events

    async def close(self):
        if self.episode is not None:
            episode, self.episode = self.episode, None
            await asyncio.to_thread(episode.close)


class NetheriteEnvironment:
    """Open a Netherite task behind the generic session contract.

    ``episode`` may carry the four deterministic coordinates of an existing
    trial. Without it, the adapter draws a legal pose from the configured
    seed exactly as the existing data builders do.
    """

    def __init__(self, task_yaml: str, *, episode: Mapping[str, Any] | None = None,
                 seed: int | None = None, random_seed: int = 0,
                 netherite_home: str | None = None,
                 frames_root: str | None = None, raster: str | None = None,
                 view_distance: int | None = None,
                 use_task_prompt: bool = True):
        self.task_yaml = str(task_yaml)
        self.episode_spec = None if episode is None else dict(episode)
        self.seed = seed
        self.random_seed = int(random_seed)
        self.netherite_home = netherite_home
        self.frames_root = frames_root
        self.raster = raster
        self.view_distance = view_distance
        self.use_task_prompt = bool(use_task_prompt)

    def _open_blocking(self, task_yaml):
        task = NetheriteTask.load(task_yaml)
        parser = cu_parser(task)
        options = {"netherite_home": self.netherite_home,
                   "frames_root": self.frames_root, "raster": self.raster,
                   "view_distance": self.view_distance}
        options = {key: value for key, value in options.items()
                   if value is not None}
        if self.episode_spec is not None:
            spec = dict(self.episode_spec)
            spec.setdefault("task_yaml", task.path)
            episode = Episode.from_spec(spec, parser, task=task, **options)
        else:
            seed = int(self.seed if self.seed is not None else task.seeds[0])
            episode = Episode.draw(task, seed, random.Random(self.random_seed),
                                   parser, **options)
        return NetheriteSession(episode, task,
                                use_task_prompt=self.use_task_prompt)

    async def open(self, task):
        task_yaml = Path(self.task_yaml)
        if not task_yaml.is_absolute() and getattr(task, "path", None):
            task_yaml = Path(task.path).resolve().parent / task_yaml
        return await asyncio.to_thread(self._open_blocking, str(task_yaml))
