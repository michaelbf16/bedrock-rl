"""Built-in Netherite state probes; custom probes can inspect the same session."""

from __future__ import annotations

import copy


# Enough state to replay reward/debugging decisions without serializing the
# 100k-cell camera, depth, and edge buffers after every turn. Authors can pass
# ``fields: null`` to capture the complete observation or name any subset.
DEFAULT_OBSERVATION_FIELDS = (
    "tick", "x", "y", "z", "yaw", "pitch", "dead", "dimension",
    "health", "food",
    "hotbar_ids", "hotbar_counts", "hotbar_meta", "hotbar_sel", "container",
    "inventory_ids", "inventory_counts", "inventory_meta",
    "inv_counts", "blocks", "logs", "coal",
)

# Engines built before the full-inventory observation extension expose the
# hotbar and public slot journal but not these exact 36-slot arrays.  Keep the
# default probe usable against those recordings while continuing to reject a
# genuinely misspelled required field.
OPTIONAL_OBSERVATION_FIELDS = frozenset((
    "inventory_ids", "inventory_counts", "inventory_meta",
))


def _snapshot(value):
    """Detach one engine value into a JSON-safe state snapshot.

    The semantic camera, depth, and edge arrays are zero-copy memoryviews
    backed by the packed observation record. ``deepcopy`` cannot pickle a
    memoryview, and retaining the view would also let the engine buffer's
    lifetime leak into a trajectory. ``tolist`` preserves every numeric cell
    while making the state portable and directly usable by rewards.
    """
    if isinstance(value, memoryview):
        return value.tolist()
    if isinstance(value, dict):
        return {copy.deepcopy(key): _snapshot(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    return copy.deepcopy(value)


def _vital(episode, field):
    """Current journaled vital when the compact observation omits it."""
    defaults = {"health": 20.0, "food": 20}
    task = getattr(episode, "task", None)
    initial_state = getattr(task, "initial_state", None)
    player = getattr(initial_state, "player", {})
    value = player.get(field, defaults[field])
    journal = getattr(getattr(episode, "env", None), "journal", ())
    for event in reversed(journal):
        if getattr(event, "name", None) == "vitals_changed":
            changed = event.get(field)
            if changed is not None:
                value = changed
            break
    return float(value) if field == "health" else int(value)


def capture_observation(episode, fields=DEFAULT_OBSERVATION_FIELDS):
    """Detach bounded state and derive the two journaled player vitals."""
    observation = episode.env.obs
    selected = list(observation) if fields is None else list(fields)
    for field in ("health", "food"):
        if fields is None and field not in selected:
            selected.append(field)
    out = {}
    for field in selected:
        if (field not in observation
                and field in OPTIONAL_OBSERVATION_FIELDS):
            continue
        value = (observation[field] if field in observation
                 else _vital(episode, field) if field in ("health", "food")
                 else observation[field])
        out[field] = _snapshot(value)
    return out


class ObservationProbe:
    name = "observation"

    def __init__(self, fields=DEFAULT_OBSERVATION_FIELDS, *,
                 name: str | None = None):
        self.fields = None if fields is None else tuple(fields)
        if name is not None:
            self.name = str(name)

    def capture(self, session):
        return capture_observation(session.episode, self.fields)


class EpisodeProbe:
    name = "episode"

    def capture(self, session):
        episode = session.episode
        return {
            "done": bool(episode.done), "success": bool(episode.success),
            "failed": bool(episode.failed), "turns": int(episode.turns),
            "actions": int(episode.nlines),
            "reward": float(episode.final_reward()),
            "spec": copy.deepcopy(episode.spec),
            "view": copy.deepcopy(episode.view_provenance),
        }
