"""Small, deterministic media renderers for saved Netherite trajectories."""

from __future__ import annotations

import base64
import copy
import io
import math
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from bedrock_rl.env.semantic import semantic_image

_FRAME_NAME = re.compile(r"^frame_(\d+)\.ppm$")
DEFAULT_REPLAY_VIEWS = (
    "semantic", "netherite-procedural", "minecraft-official",
)
DEFAULT_REPLAY_LABELS = ("Semantic", "Netherite", "Minecraft")
_DEFAULT_GITHUB_GIF_BYTES = 90_000_000
_MAX_REPLAY_FRAMES = 10_000
_MAX_REPLAY_SPOOL_BYTES = 640 * 1024**2
_INITIAL_GIF_FRAMES = 1_200


class _GifTooLarge(RuntimeError):
    """Private signal used to retry an animation at a lower cadence."""


def _gif_delay(milliseconds):
    """Normalize a delay to GIF's integer centisecond resolution."""
    return max(20, int(milliseconds)) // 10 * 10


def trajectory_image_frames(trajectory, *, width=None):
    """Decode model-visible OpenAI image parts in chronological order."""
    from PIL import Image

    frames = []
    for message_index, message in enumerate(trajectory.messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part_index, part in enumerate(content):
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else None
            where = f"message {message_index}.content[{part_index}]"
            if not isinstance(url, str) or not url.startswith("data:image/"):
                raise ValueError(f"{where} is not an embedded image data URL")
            try:
                header, payload = url.split(",", 1)
                if ";base64" not in header.lower():
                    raise ValueError
                data = base64.b64decode(payload, validate=True)
                with Image.open(io.BytesIO(data)) as source:
                    source.load()
                    frame = source.convert("RGB")
            except Exception as exc:
                raise ValueError(f"{where} is not a decodable base64 image") \
                    from exc
            if width is not None and frame.width != int(width):
                target_width = int(width)
                if target_width < 1:
                    raise ValueError("GIF width must be positive")
                target_height = round(frame.height * target_width / frame.width)
                frame = frame.resize((target_width, target_height),
                                     Image.Resampling.NEAREST)
            frames.append(frame)
    if not frames:
        raise ValueError("trajectory has no embedded model-view images")
    return frames


def _gif_frame(value):
    """Own one RGB frame without leaving an input file open."""
    from PIL import Image

    if isinstance(value, (str, os.PathLike)):
        with Image.open(value) as source:
            source.load()
            return source.convert("RGB")
    return value.convert("RGB")


def _write_streaming_gif(frames, stream, base_duration, durations,
                         final_hold_ms, colors, max_bytes):
    """Write GIF blocks with one full frame of working memory.

    Pillow's public multi-frame writer first copies every input frame. That is
    convenient for short previews but several gigabytes for a dense portal
    replay. Its GIF block helpers are deliberately used here one frame at a
    time; later frames carry local palettes and update only their changed
    bounding box.
    """
    from PIL import GifImagePlugin, Image, ImageChops

    iterator = iter(frames)
    try:
        pending = _gif_frame(next(iterator))
    except StopIteration as exc:
        raise ValueError("cannot write a GIF with no frames") from exc
    size = pending.size
    duration_iterator = (None if durations is None else iter(durations))

    def next_duration():
        if duration_iterator is None:
            return base_duration
        try:
            return _gif_delay(next(duration_iterator))
        except StopIteration as exc:
            raise ValueError("GIF durations have fewer values than frames") \
                from exc

    pending_duration = next_duration()
    previous = None
    wrote_header = False

    def write_frame(frame, duration):
        nonlocal previous, wrote_header
        if previous is None:
            offset = (0, 0)
            output = frame
        else:
            box = ImageChops.difference(previous, frame).getbbox()
            if box is None:
                raise AssertionError("identical GIF frame was not coalesced")
            offset = box[:2]
            output = frame.crop(box)
        output = output.quantize(
            colors=colors, method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE)
        if not wrote_header:
            for block in GifImagePlugin._get_global_header(
                    output, {"loop": 0}):
                stream.write(block)
            include_color_table = False
            wrote_header = True
        else:
            include_color_table = True
        GifImagePlugin._write_frame_data(
            stream, output, offset,
            {"duration": duration, "disposal": 1,
             "include_color_table": include_color_table})
        if max_bytes is not None and stream.tell() > max_bytes:
            raise _GifTooLarge
        previous = frame

    for raw in iterator:
        frame = _gif_frame(raw)
        if frame.size != size:
            raise ValueError(
                "GIF frames do not share one size: "
                f"{size!r} and {frame.size!r}")
        frame_duration = next_duration()
        if ImageChops.difference(pending, frame).getbbox() is None:
            pending_duration += frame_duration
            continue
        write_frame(pending, pending_duration)
        pending = frame
        pending_duration = frame_duration
    if duration_iterator is not None:
        try:
            next(duration_iterator)
        except StopIteration:
            pass
        else:
            raise ValueError("GIF durations have more values than frames")
    write_frame(pending, max(pending_duration, int(final_hold_ms)))
    stream.write(b";")
    if max_bytes is not None and stream.tell() > max_bytes:
        raise _GifTooLarge


def save_frames_gif(frames, path, *, duration_ms=55, final_hold_ms=900,
                    durations_ms=None, colors=256, max_bytes=None):
    """Atomically stream same-sized frames to a looping GIF.

    ``frames`` may be a one-shot iterator or paths to spooled images. Memory
    use is independent of frame count.
    """
    base_duration = _gif_delay(duration_ms)
    colors = int(colors)
    if not 2 <= colors <= 256:
        raise ValueError("GIF colors must be in 2..256")
    durations = None if durations_ms is None else list(durations_ms)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
        delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            _write_streaming_gif(
                frames, handle, base_duration, durations, final_hold_ms,
                colors, max_bytes)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def save_trajectory_gif(trajectory, path, *, width=None, duration_ms=55,
                        final_hold_ms=900):
    """Save the exact view images carried by a canonical trajectory."""
    return save_frames_gif(
        trajectory_image_frames(trajectory, width=width), path,
        duration_ms=duration_ms, final_hold_ms=final_hold_ms)


def recording_frames(directory, *, width=None):
    """Load dense PPM captures left by ``MagmaEnv(capture_every=...)``."""
    from PIL import Image

    directory = Path(directory)
    paths = []
    # MagmaEnv writes one seed/run directory below Episode's unique capture
    # root. Accept direct frames as well for custom recorders and tests.
    for path in directory.rglob("frame_*.ppm"):
        match = _FRAME_NAME.match(path.name)
        if match:
            paths.append((int(match.group(1)), str(path.relative_to(directory)),
                          path))
    frames = []
    for _index, _relative, path in sorted(paths):
        with Image.open(path) as source:
            source.load()
            frame = source.convert("RGB")
        if width is not None and frame.width != int(width):
            target_width = int(width)
            target_height = round(frame.height * target_width / frame.width)
            frame = frame.resize((target_width, target_height),
                                 Image.Resampling.NEAREST)
        frames.append(frame)
    if not frames:
        raise ValueError(f"recording directory {directory} has no frames")
    return frames


def trajectory_semantic_frames(trajectory, *, width=800):
    """Render every semantic-camera state stored by a trajectory."""
    frames = []
    for sample in trajectory.state.samples:
        observation = sample.values.get("observation")
        if not isinstance(observation, dict):
            continue
        if not all(name in observation for name in ("cam", "depth", "edge")):
            continue
        frames.append(semantic_image(
            observation["cam"], observation["depth"], observation["edge"],
            width=width))
    if not frames:
        raise ValueError(
            "trajectory has no semantic frames; construct the scripted "
            "generator with semantic_camera: true")
    return frames


def save_semantic_gif(trajectory, path, *, width=800, duration_ms=55,
                      final_hold_ms=900):
    """Write a fast looping semantic-camera GIF from an accepted run."""
    frames = trajectory_semantic_frames(trajectory, width=width)
    return save_frames_gif(frames, path, duration_ms=duration_ms,
                           final_hold_ms=final_hold_ms)


def align_frame_sequences(sequences):
    """Pad shorter frame sequences by holding their final frame.

    Reusing the last frame is deterministic and keeps all panels present when
    one renderer writes its final periodic capture one cadence earlier.
    """
    values = [list(sequence) for sequence in sequences]
    if not values or any(not sequence for sequence in values):
        raise ValueError("every replay view must contain at least one frame")
    length = max(map(len, values))
    return [sequence + [sequence[-1]] * (length - len(sequence))
            for sequence in values]


def compose_labeled_panels(sequences, labels, *, panel_width=320):
    """Compose synchronized, labeled frames into a horizontal strip."""
    sequences = align_frame_sequences(sequences)
    labels = [str(label) for label in labels]
    if len(sequences) != len(labels):
        raise ValueError("replay labels must match replay views")
    width = int(panel_width)
    if width < 1:
        raise ValueError("replay panel_width must be positive")
    return list(_iter_labeled_panels(sequences, labels, width=width))


def _iter_labeled_panels(sequences, labels, *, width, positions=None):
    """Yield composed rows while owning only the current three inputs."""
    from PIL import Image, ImageDraw, ImageFont

    if positions is None:
        positions = range(len(sequences[0]))
    font = ImageFont.load_default()
    output_size = None
    for position in positions:
        row = [_gif_frame(sequence[position]) for sequence in sequences]
        source = row[0]
        height = max(1, source.height * width // source.width)
        size = (width, height)
        if output_size is None:
            output_size = size
        else:
            # A replay deliberately switches the semantic panel to the
            # engine raster while a GUI is open. Hold the geometry selected
            # by the first sampled tick across those view transitions; every
            # panel is resized below and all three tick streams remain exact.
            size = output_size
            height = output_size[1]
        canvas = Image.new("RGB", (width * len(row), height), "black")
        draw = ImageDraw.Draw(canvas)
        for index, (frame, label) in enumerate(zip(row, labels)):
            panel = frame
            if panel.size != size:
                panel = panel.resize(size, Image.Resampling.NEAREST)
            x = index * width
            canvas.paste(panel, (x, 0))
            box = draw.textbbox((0, 0), label, font=font)
            label_width = box[2] - box[0]
            label_height = box[3] - box[1]
            pad = 4
            draw.rectangle(
                (x, 0, x + label_width + 2 * pad,
                 label_height + 2 * pad), fill=(11, 12, 18))
            draw.text((x + pad, pad), label, fill=(244, 240, 255),
                      font=font)
        yield canvas


def _assistant_programs(trajectory):
    """The exact computer action payloads saved in assistant messages."""
    from bedrock_rl.adapters.netherite.computer import cu_payload
    from bedrock_rl.core.tools import ToolCall

    programs = []
    for message_index, message in enumerate(trajectory.messages):
        if message.get("role") != "assistant":
            continue
        for raw in message.get("tool_calls") or ():
            call = ToolCall.from_openai(raw)
            if call.name != "computer":
                raise ValueError(
                    f"assistant message {message_index} calls {call.name!r}; "
                    "renderer replay accepts canonical computer trajectories")
            arguments = raw["function"].get("arguments", {})
            # Match the live tool boundary exactly. OpenAI tool arguments are
            # the {"actions": ...} wrapper; Episode's parser consumes the
            # value under actions. cu_payload is the one production reader of
            # that boundary and preserves the saved action order and values.
            programs.append(cu_payload(arguments))
    if not programs:
        raise ValueError("trajectory has no assistant computer calls to replay")
    return programs


def _recorded_episode(trajectory):
    episode = trajectory.state.latest.get("episode")
    if not isinstance(episode, Mapping):
        raise ValueError("trajectory has no recorded terminal episode state")
    provenance = trajectory.provenance
    spec = (provenance.get("episode_spec")
            if isinstance(provenance, Mapping) else None)
    if not isinstance(spec, Mapping):
        # Earlier artifacts stored the reconstruction spec in terminal state
        # instead of provenance.
        spec = episode.get("spec")
    if not isinstance(spec, Mapping):
        raise ValueError("trajectory has no recorded episode spec")
    required = ("task_yaml", "seed", "init_yaw", "init_pitch")
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError("trajectory episode spec is missing "
                         + ", ".join(missing))
    if not episode.get("done") or not episode.get("success") \
            or episode.get("failed"):
        raise ValueError(
            "renderer replay needs an accepted successful trajectory")
    return episode, spec


def _load_recorded_task(trajectory, spec):
    from bedrock_rl.env.task import Task

    candidates = [spec["task_yaml"]]
    task_path = trajectory.task.get("path")
    if task_path and task_path not in candidates:
        candidates.append(task_path)
    errors = []
    for path in candidates:
        try:
            return Task.load(path)
        except FileNotFoundError as exc:
            errors.append(str(exc))
    raise FileNotFoundError(
        "cannot replay trajectory because its task YAML is unavailable; "
        + " | ".join(errors))


def _event_contract(events):
    values = []
    for event in events:
        if hasattr(event, "as_dict"):
            raw = dict(event.as_dict())
            name = raw.pop("event")
            tick = raw.pop("tick")
            values.append((str(name), raw, int(tick)))
        else:
            values.append((str(event.name), copy.deepcopy(event.payload),
                           None if event.tick is None else int(event.tick)))
    return values


def _verify_replay(trajectory, recorded, episode, view):
    actual = {
        "done": bool(episode.done), "success": bool(episode.success),
        "failed": bool(episode.failed), "turns": int(episode.turns),
        "actions": int(episode.nlines),
        "reward": float(episode.final_reward()),
    }
    expected = {key: recorded.get(key) for key in actual}
    scalar_mismatch = []
    for key in ("done", "success", "failed", "turns", "actions"):
        if actual[key] != expected[key]:
            scalar_mismatch.append(
                f"{key} expected {expected[key]!r}, got {actual[key]!r}")
    expected_reward = float(expected["reward"])
    if not math.isclose(actual["reward"], expected_reward,
                        rel_tol=0.0, abs_tol=1e-9):
        scalar_mismatch.append(
            f"reward expected {expected_reward!r}, got {actual['reward']!r}")
    expected_events = _event_contract(trajectory.state.events)
    actual_events = _event_contract(episode.env.journal)
    if expected_events and actual_events != expected_events:
        scalar_mismatch.append(
            f"event stream expected {len(expected_events)} records, got "
            f"{len(actual_events)} or different event content/order")
    if scalar_mismatch:
        raise RuntimeError(
            f"replay under view {view!r} did not reproduce the accepted "
            "terminal contract: " + "; ".join(scalar_mismatch))
    return actual


def _await_periodic_capture(env, index, *, load=True):
    """Load one exact periodic capture, however directory writes arrive."""
    from PIL import Image

    from bedrock_rl.env.engine import EngineTimeout

    root = getattr(env, "_frames_path", None)
    if root is None:
        raise RuntimeError("periodic replay capture has no frames directory")
    index = int(index)
    path = Path(root) / f"frame_{index:06d}.ppm"
    header = f"P6\n{env.frame_w} {env.frame_h}\n255\n"
    expected_size = len(header) + env.frame_w * env.frame_h * 3
    deadline = time.monotonic() + float(env.timeout)
    while True:
        try:
            complete = path.stat().st_size >= expected_size
        except OSError:
            complete = False
        if complete:
            # Load while the episode still owns its private capture tree.
            frame = None
            if load:
                with Image.open(path) as source:
                    source.load()
                    frame = source.convert("RGB")
            path.unlink(missing_ok=True)
            return frame
        if time.monotonic() > deadline:
            env.failed = True
            raise EngineTimeout(
                f"{env.describe()}: periodic capture index {index} did not "
                f"complete in {root} within {float(env.timeout):g}s")
        time.sleep(0.002)


class _ReplayFrameCollector:
    """Collect the observation and pixels produced by the same engine tick.

    With ``capture_every`` set, lazy capture is disabled and the engine calls
    its capture writer once per tick. ``reset_pose`` consumes capture index 0
    and then resets ``env.ticks``; Episode's unrecorded t0 tick is therefore
    index 1, and every later ``env.ticks`` value is the exact PPM filename
    index. Reading directory deltas loses that relationship when a write is
    discovered late or several names appear together, so cadence is derived
    from the tick and pixel views wait for that one filename explicitly.
    """

    def __init__(self, view, recording_every, *, spool_dir=None,
                 frame_width=None):
        self.view = str(view)
        self.recording_every = int(recording_every)
        self.frames = []
        self.frame_indices = []
        self.spool_dir = None if spool_dir is None else Path(spool_dir)
        if self.spool_dir is not None:
            self.spool_dir.mkdir(parents=True, exist_ok=True)
        self.frame_width = (None if frame_width is None
                            else int(frame_width))
        self.frame_height = None
        self.spool_bytes = 0
        self.gui_open = False
        self.event_index = 0
        self.last_tick = None

    def _store(self, frame, tick):
        from PIL import Image

        if self.frame_width is not None:
            if self.frame_height is None:
                self.frame_height = max(
                    1, round(frame.height * self.frame_width / frame.width))
            size = (self.frame_width, self.frame_height)
            if frame.size != size:
                # Semantic replay uses the engine frame while a GUI is open
                # and the 64x36 semantic camera otherwise. The engine's
                # 428x240 raster rounds to one pixel shorter at common panel
                # widths, so normalize every sampled tick to the first 16:9
                # panel instead of treating normal GUI transitions as a
                # changing replay aspect ratio.
                frame = frame.resize(size, Image.Resampling.NEAREST)
        if self.spool_dir is None:
            self.frames.append(frame)
            return
        if len(self.frames) >= _MAX_REPLAY_FRAMES:
            raise RuntimeError(
                "replay exceeds the bounded frame spool; raise "
                "recording_every for this trajectory")
        output = self.spool_dir / f"frame_{tick:06d}.png"
        palette = frame.quantize(
            colors=256, method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE)
        palette.save(output, format="PNG", compress_level=1)
        self.spool_bytes += output.stat().st_size
        if self.spool_bytes > _MAX_REPLAY_SPOOL_BYTES:
            raise RuntimeError(
                "replay view exceeds the bounded 640 MiB frame spool; "
                "raise recording_every or lower panel_width")
        self.frames.append(output)

    def __call__(self, env):
        for event in env.journal[self.event_index:]:
            if event.name == "container_opened":
                self.gui_open = True
            elif event.name == "container_closed":
                self.gui_open = False
        self.event_index = len(env.journal)
        tick = int(env.ticks)
        if self.last_tick is None:
            # Episode invokes the observer for its t0 transition. It belongs
            # to episode setup rather than the saved computer action stream.
            if tick != 1:
                raise RuntimeError(
                    "replay frame observer must begin on Episode's t0 tick "
                    f"1, got {tick}")
            self.last_tick = tick
            return
        if tick != self.last_tick + 1:
            raise RuntimeError(
                "replay frame observer lost engine tick(s): expected "
                f"{self.last_tick + 1}, got {tick}")
        self.last_tick = tick
        if tick % self.recording_every:
            return
        if self.view == "semantic" and not self.gui_open:
            # Preserve the observation now. Filesystem discovery may lag into
            # a later tick, whose semantic observation is no longer this one.
            frame = semantic_image(
                env.obs["cam"], env.obs["depth"], env.obs["edge"], width=64)
            # Periodic rendering still wrote this tick. It is not the source
            # of the semantic panel, but waiting and unlinking keeps capture
            # disk use constant during a long replay.
            if getattr(env, "_frames_path", None) is not None:
                _await_periodic_capture(env, tick, load=False)
        else:
            frame = _await_periodic_capture(env, tick)
        self._store(frame, tick)
        self.frame_indices.append(tick)


@dataclass(frozen=True)
class ReplayViewResult:
    view: str
    frames: list
    terminal: dict
    frame_indices: tuple = ()
    spool_bytes: int = 0


def replay_trajectory_view(trajectory, view, *, recording_every=5,
                           renderer_home=None, frames_root=None,
                           episode_class=None, spool_dir=None,
                           frame_width=None):
    """Replay one canonical computer trajectory without rewriting it."""
    from bedrock_rl.adapters.netherite.computer import cu_parser
    from bedrock_rl.env.episode import Episode
    from bedrock_rl.env.task import Task

    every = int(recording_every)
    if every < 1:
        raise ValueError("replay recording_every must be positive")
    recorded, spec = _recorded_episode(trajectory)
    task = _load_recorded_task(trajectory, spec)
    raw = copy.deepcopy(task.raw)
    raw["view"] = str(view)
    replay_task = Task(raw, path=task.path)
    collector_kwargs = {}
    if spool_dir is not None:
        collector_kwargs["spool_dir"] = spool_dir
    if frame_width is not None:
        collector_kwargs["frame_width"] = frame_width
    collector = _ReplayFrameCollector(view, every, **collector_kwargs)
    own_root = None
    if frames_root is None:
        own_root = tempfile.TemporaryDirectory(prefix="bedrock-replay-")
        frames_root = own_root.name
    episode_type = episode_class or Episode
    episode = None
    try:
        try:
            episode = episode_type(
                replay_task, int(spec["seed"]), float(spec["init_yaw"]),
                float(spec["init_pitch"]), parser=cu_parser(replay_task),
                netherite_home=renderer_home, frames_root=frames_root,
                capture_every=every, capture_frames=False,
                tick_observer=collector)
        except Exception as exc:
            variables = {
                "semantic": "NETHERITE_HOME (or either renderer home)",
                "netherite-procedural": "NETHERITE_HOME_PROCEDURAL",
                "minecraft-official": "NETHERITE_HOME_JAR",
            }
            hint = variables.get(str(view), "the view's renderer home")
            raise RuntimeError(
                f"could not start replay view {view!r}; set {hint} to a "
                f"matching built Netherite checkout. Cause: {exc}") from exc
        programs = _assistant_programs(trajectory)
        for index, program in enumerate(programs):
            if episode.done:
                raise RuntimeError(
                    f"replay view {view!r} terminated before saved computer "
                    f"call {index} of {len(programs)}")
            episode.step(program)
        terminal = _verify_replay(trajectory, recorded, episode, view)
        if not collector.frames:
            raise RuntimeError(
                f"replay view {view!r} produced no periodic frames; lower "
                f"recording_every from {every}")
        return ReplayViewResult(
            str(view), collector.frames, terminal,
            tuple(getattr(collector, "frame_indices", ())),
            int(getattr(collector, "spool_bytes", 0)))
    finally:
        if episode is not None:
            episode.close()
        if own_root is not None:
            own_root.cleanup()


def save_replayed_views_gif(
        trajectory, path, *, views=DEFAULT_REPLAY_VIEWS,
        labels=DEFAULT_REPLAY_LABELS, recording_every=5, panel_width=320,
        renderer_homes=None, duration_ms=55, final_hold_ms=900,
        episode_class=None, max_bytes=_DEFAULT_GITHUB_GIF_BYTES,
        replay_workers="auto"):
    """Verify and render one accepted action stream under several views.

    All views replay the same saved actions independently and concurrently,
    then must report exactly the same sampled engine tick IDs. The engine
    processes already isolate their worlds; threads only orchestrate those
    children and avoid making trajectories, custom episode classes, or test
    doubles cross a pickle boundary. ``replay_workers=1`` retains serial
    execution. Results remain in ``views`` order regardless of completion
    order.

    Frames are spooled at their final panel size. Long animations are
    uniformly sampled at one shared set of positions until the file fits;
    each emitted frame's delay covers the skipped intervals, preserving
    trajectory time as well as panel sync.
    """
    views = tuple(str(view) for view in views)
    labels = tuple(str(label) for label in labels)
    if not views:
        raise ValueError("replay views cannot be empty")
    if len(views) != len(labels):
        raise ValueError("replay labels must match replay views")
    width = int(panel_width)
    if width < 1:
        raise ValueError("replay panel_width must be positive")
    limit = None if max_bytes is None else int(max_bytes)
    if limit is not None and limit < 1:
        raise ValueError("replay max_bytes must be positive")
    if replay_workers == "auto":
        worker_count = min(3, len(views))
    else:
        if isinstance(replay_workers, bool):
            raise TypeError("replay_workers must be 'auto' or an integer")
        worker_count = int(replay_workers)
        if worker_count < 1:
            raise ValueError("replay_workers must be positive")
        worker_count = min(worker_count, len(views))
    homes = dict(renderer_homes or {})
    with tempfile.TemporaryDirectory(prefix="bedrock-replay-spool-") as root:
        def replay(request):
            index, view = request
            return replay_trajectory_view(
                trajectory, view, recording_every=recording_every,
                renderer_home=homes.get(view), episode_class=episode_class,
                spool_dir=Path(root) / f"view-{index}", frame_width=width)

        requests = tuple(enumerate(views))
        if worker_count == 1:
            results = [replay(request) for request in requests]
        else:
            with ThreadPoolExecutor(
                    max_workers=worker_count,
                    thread_name_prefix="bedrock-replay") as pool:
                # map returns in input order. Completion order therefore
                # cannot change panel order or the cross-view verification.
                results = list(pool.map(replay, requests))
        ticks = results[0].frame_indices
        if not ticks:
            # Test doubles and custom collectors predating exact-tick replay
            # still compose positionally. Production collectors always carry
            # tick IDs and take the stricter branch below.
            lengths = {len(result.frames) for result in results}
            if len(lengths) != 1:
                raise RuntimeError(
                    "replay views have no tick IDs and unequal frame counts")
            ticks = tuple(range(next(iter(lengths))))
        elif any(result.frame_indices != ticks for result in results[1:]):
            detail = ", ".join(
                f"{result.view}={len(result.frame_indices)} frames, "
                f"first={result.frame_indices[:3]!r}, "
                f"last={result.frame_indices[-3:]!r}"
                for result in results)
            raise RuntimeError(
                "replay views did not capture identical engine ticks: "
                + detail)
        sequences = [result.frames for result in results]
        trajectory_frames = _trajectory_image_count(trajectory)
        minimum_frames = trajectory_frames + 1
        if len(ticks) < minimum_frames:
            raise RuntimeError(
                f"replay captured {len(ticks)} periodic frames but the "
                f"trajectory has {trajectory_frames} model-view frames; "
                "lower recording_every so replay media has a higher "
                "cadence than trajectory images")
        positions = _sample_positions(
            len(ticks), max(_INITIAL_GIF_FRAMES, minimum_frames))
        encoded = 0
        while True:
            durations = _sample_durations(
                positions, len(ticks), duration_ms)
            frames = _iter_labeled_panels(
                sequences, labels, width=width, positions=positions)
            try:
                save_frames_gif(
                    frames, path, duration_ms=duration_ms,
                    durations_ms=durations, final_hold_ms=final_hold_ms,
                    max_bytes=limit)
                encoded = len(positions)
                break
            except _GifTooLarge:
                next_count = max(
                    minimum_frames, len(positions) * 3 // 4)
                if next_count >= len(positions):
                    raise RuntimeError(
                        f"replay GIF cannot fit the {limit}-byte limit "
                        "while remaining denser than trajectory images") \
                        from None
                positions = _sample_positions(
                    len(ticks), next_count)
        return {
            "gif": str(Path(path)),
            "views": [result.view for result in results],
            "frames": encoded,
            "source_frames": len(ticks),
            "trajectory_frames": trajectory_frames,
            "first_tick": ticks[0],
            "last_tick": ticks[-1],
            "sampled_ticks": [ticks[index] for index in positions],
            "bytes": Path(path).stat().st_size,
            "spool_bytes": sum(result.spool_bytes for result in results),
            "terminal": [result.terminal for result in results],
        }


def _trajectory_image_count(trajectory):
    """Count model-view images without decoding their embedded payloads."""
    count = 0
    for message in trajectory.messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        count += sum(
            isinstance(part, Mapping) and part.get("type") == "image_url"
            for part in content)
    return count


def _sample_positions(length, count):
    """Return deterministic, distinct, endpoint-preserving positions."""
    length = int(length)
    count = min(length, max(1, int(count)))
    if count == length:
        return list(range(length))
    if count == 1:
        return [length - 1]
    return [(index * (length - 1)) // (count - 1)
            for index in range(count)]


def _sample_durations(positions, source_length, duration_ms):
    """Assign skipped source intervals to the preceding displayed frame."""
    base = _gif_delay(duration_ms)
    return [
        ((positions[index + 1] - position) * base
         if index + 1 < len(positions)
         else (source_length - position) * base)
        for index, position in enumerate(positions)
    ]
