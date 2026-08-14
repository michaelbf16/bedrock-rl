"""Read-only SDG policy queries over a Netherite start-state snapshot.

``SnapshotWorldMap`` intentionally reads the same compact BSNP v1 grid that
the engine restores.  It does not turn the world into Python objects: the
snapshot stays memory-mapped, searches stream in file order, and resource
clusters retain only a bounded sample of their cells.

The map describes the captured start state.  Policies must still use live
observations and journal events after they begin changing the world.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import hashlib
import heapq
import mmap
import os
from pathlib import Path
import struct
import sys
from typing import Iterable, Iterator

from bedrock_rl.env.catalog import resolve_blocks
from bedrock_rl.env.snapshots import (
    ANGLE_OFF,
    HEAD_SIZE,
    INV_OFF,
    INV_SLOTS,
    ITEM_SIZE,
    POS_OFF,
    _check_layout,
)


class SnapshotFormatError(ValueError):
    """The file is not a complete, exact BSNP v1 snapshot."""


class SnapshotBoundsError(IndexError):
    """A planning query addressed a cell outside the captured snapshot."""


@dataclass(frozen=True, slots=True)
class PlayerPose:
    x: float
    y: float
    z: float
    yaw: float
    pitch: float


@dataclass(frozen=True, slots=True)
class Bounds:
    """Inclusive integer bounds of a captured block region."""

    min_x: int
    min_y: int
    min_z: int
    max_x: int
    max_y: int
    max_z: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.max_x - self.min_x + 1,
                self.max_y - self.min_y + 1,
                self.max_z - self.min_z + 1)

    @property
    def volume(self) -> int:
        nx, ny, nz = self.shape
        return nx * ny * nz

    def contains(self, x: int, y: int, z: int) -> bool:
        return (self.min_x <= x <= self.max_x
                and self.min_y <= y <= self.max_y
                and self.min_z <= z <= self.max_z)

    def intersection(self, other: Bounds) -> Bounds | None:
        values = Bounds(
            max(self.min_x, other.min_x),
            max(self.min_y, other.min_y),
            max(self.min_z, other.min_z),
            min(self.max_x, other.max_x),
            min(self.max_y, other.max_y),
            min(self.max_z, other.max_z),
        )
        if (values.min_x > values.max_x
                or values.min_y > values.max_y
                or values.min_z > values.max_z):
            return None
        return values


@dataclass(frozen=True, slots=True)
class SnapshotHeader:
    version: int
    seed: int
    tick: int
    origin_x: int
    origin_z: int
    player: PlayerPose
    bounds: Bounds
    snapshot_sha256: str


@dataclass(frozen=True, slots=True)
class BlockState:
    block_id: int
    meta: int


@dataclass(frozen=True, slots=True)
class BlockCell:
    x: int
    y: int
    z: int
    block_id: int
    meta: int

    @property
    def position(self) -> tuple[int, int, int]:
        return self.x, self.y, self.z


@dataclass(frozen=True, slots=True)
class ResourceCluster:
    """One face-connected block cluster with a bounded cell sample.

    ``size`` is always exact.  ``cells`` contains every coordinate when
    ``cells_complete`` is true; otherwise it contains the first bounded,
    deterministic sample encountered by the traversal.
    """

    block_ids: tuple[int, ...]
    size: int
    bounds: Bounds
    anchor: tuple[int, int, int]
    cells: tuple[tuple[int, int, int], ...]
    cells_complete: bool


@dataclass(frozen=True, slots=True)
class ResourceTargetPool:
    """Exact population summary with a bounded planning-target sample.

    Unlike :class:`ResourceCluster`, this deliberately does not compute
    connected components.  ``available_count``, ``bounds``, ``anchor``, and
    ``block_ids`` describe the complete matching population; ``cells`` is a
    deterministic bounded sample suitable for a planner that only needs a
    handful of actual targets.  Every sampled coordinate is still read from
    the immutable snapshot and can therefore be certified before execution.
    """

    block_ids: tuple[int, ...]
    available_count: int
    bounds: Bounds | None
    anchor: tuple[int, int, int] | None
    cells: tuple[tuple[int, int, int], ...]
    cells_complete: bool


_FORMAT_PREFIX_SIZE = struct.calcsize("<4sI")
_SEED_TICK_ORIGIN_OFF = _FORMAT_PREFIX_SIZE
_ITEM_COUNT_OFF = INV_OFF + INV_SLOTS * 3 * 4
_REGION_HEADER_OFF = _ITEM_COUNT_OFF + struct.calcsize("<I")

_AIR = frozenset((0,))
_WATER = frozenset((8, 9))
_LAVA = frozenset((10, 11))
_FLUID = _WATER | _LAVA

# Collision-free blocks relevant to Minecraft 1.11.2 route planning.  Unknown
# blocks are conservatively treated as solid.
_NON_SOLID = frozenset((
    0, 6, 27, 28, 31, 32, 37, 38, 39, 40, 50, 51, 55, 59, 63, 65,
    66, 68, 69, 70, 72, 75, 76, 77, 78, 83, 90, 93, 94, 104, 105,
    106, 111, 115, 127, 131, 132, 141, 142, 143, 147, 148, 149, 150,
    151, 157, 171, 175, 176, 177, 198, 207,
)) | _FLUID

_HAZARDS = {
    10: "lava",
    11: "lava",
    51: "fire",
    81: "cactus",
    213: "magma",
}


def _block_ids(values: int | str | Iterable[int | str]) -> tuple[int, ...]:
    if isinstance(values, (int, str)):
        values = (values,)
    resolved: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise TypeError("block ids must be integers or block names")
        if isinstance(value, int):
            resolved.append(value)
        else:
            resolved.extend(resolve_blocks(value))
    if not resolved:
        raise ValueError("at least one block id is required")
    if any(value < 0 or value > 4095 for value in resolved):
        raise ValueError("block ids must fit the BSNP 12-bit block field")
    return tuple(dict.fromkeys(resolved))


def _metadata_values(
    values: int | Iterable[int] | None,
) -> tuple[int, ...] | None:
    """Normalize an optional BSNP metadata predicate."""

    if values is None:
        return None
    if isinstance(values, bool):
        raise TypeError("metadata values must be integers in 0..15")
    if isinstance(values, int):
        values = (values,)
    resolved = tuple(dict.fromkeys(values))
    if not resolved:
        raise ValueError("metadata cannot be empty")
    if any(isinstance(value, bool) or not isinstance(value, int)
           or value < 0 or value > 15 for value in resolved):
        raise ValueError("metadata values must be integers in 0..15")
    return resolved


class SnapshotWorldMap:
    """Memory-mapped, read-only view of one exact BSNP v1 snapshot."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self._stream = None
        self._map = None
        # Built lazily because tiny task-authoring checks may only ask for a
        # handful of individual cells.  Once built, this is the one immutable
        # block-id pass shared by every vectorized population query.
        self._voxel_index_data = None
        self._matching_indices = {}
        self._target_pool_cache = {}
        self._cluster_cache = {}
        try:
            self._stream = self.path.open("rb")
            size = os.fstat(self._stream.fileno()).st_size
            if size < _FORMAT_PREFIX_SIZE:
                raise SnapshotFormatError(
                    f"snapshot is {size} bytes, shorter than its prefix")
            self._map = mmap.mmap(
                self._stream.fileno(), length=0, access=mmap.ACCESS_READ)
            try:
                _check_layout(self._map)
            except (RuntimeError, struct.error) as exc:
                raise SnapshotFormatError(str(exc)) from exc

            magic, version = struct.unpack_from("<4sI", self._map, 0)
            if magic != b"BSNP" or version != 1:
                # _check_layout already catches this; retain a local guard so
                # the public parser remains explicit about the accepted form.
                raise SnapshotFormatError(
                    f"not an exact BSNP v1 snapshot ({magic!r} v{version})")
            seed, tick, origin_x, origin_z = struct.unpack_from(
                "<qqii", self._map, _SEED_TICK_ORIGIN_OFF)
            x, y, z = struct.unpack_from("<3d", self._map, POS_OFF)
            yaw, pitch = struct.unpack_from("<2f", self._map, ANGLE_OFF)
            self._item_count = struct.unpack_from(
                "<I", self._map, _ITEM_COUNT_OFF)[0]
            rx, ry, rz, nx, ny, nz = struct.unpack_from(
                "<6i", self._map, _REGION_HEADER_OFF)
            self._nx, self._ny, self._nz = nx, ny, nz
            self._stride_x = ny * nz
            self._cells_at = HEAD_SIZE + self._item_count * ITEM_SIZE
            bounds = Bounds(rx, ry, rz,
                            rx + nx - 1, ry + ny - 1, rz + nz - 1)
            digest = hashlib.sha256(self._map).hexdigest()
            # BSNP deliberately stores the player in the engine's moving
            # window coordinate system so restoring preserves the exact
            # double bits.  Region cells and every public environment
            # observation use world coordinates.  Convert once here so a
            # planner cannot accidentally combine the two coordinate spaces.
            self.header = SnapshotHeader(
                version=version, seed=seed, tick=tick,
                origin_x=origin_x, origin_z=origin_z,
                player=PlayerPose(
                    x + origin_x, y, z + origin_z, yaw, pitch),
                bounds=bounds, snapshot_sha256=digest)
        except BaseException:
            self.close()
            raise

    @property
    def seed(self) -> int:
        return self.header.seed

    @property
    def tick(self) -> int:
        return self.header.tick

    @property
    def player(self) -> PlayerPose:
        return self.header.player

    @property
    def bounds(self) -> Bounds:
        return self.header.bounds

    @property
    def snapshot_sha256(self) -> str:
        return self.header.snapshot_sha256

    @property
    def closed(self) -> bool:
        return self._map is None

    def close(self) -> None:
        mapping, stream = self._map, self._stream
        self._map = None
        self._stream = None
        # NumPy views retain an exported pointer into mmap. Drop every cached
        # view before closing it or mmap.close() raises BufferError.
        self._cluster_cache.clear()
        self._target_pool_cache.clear()
        self._matching_indices.clear()
        self._voxel_index_data = None
        if mapping is not None:
            mapping.close()
        if stream is not None:
            stream.close()

    def __enter__(self) -> SnapshotWorldMap:
        if self.closed:
            raise RuntimeError("snapshot world map is closed")
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def _require_open(self):
        if self._map is None:
            raise RuntimeError("snapshot world map is closed")
        return self._map

    def _voxel_index(self):
        """One immutable NumPy index over the packed snapshot, or ``None``.

        The packed states remain a zero-copy mmap view. One stable ordering by
        block id plus two 4096-entry histograms are computed once; every later
        resource query slices that ordering instead of rescanning 13 million
        voxels. Positions use uint32 because BSNP's native volume cap is 2^24,
        keeping the index four times smaller than ``argwhere``'s three int64
        columns. Stable sorting preserves original x/y/z file order per id.
        """

        if self._voxel_index_data is not None:
            return self._voxel_index_data
        try:
            import numpy as np
        except ImportError:
            return None
        states = np.frombuffer(
            self._require_open(), dtype="<u2", count=self.bounds.volume,
            offset=self._cells_at)
        block_ids = states >> 4
        histogram = np.bincount(block_ids, minlength=4096)
        source_histogram = np.bincount(
            block_ids[(states & 15) == 0], minlength=4096)
        offsets = np.empty(4097, dtype=np.uint32)
        offsets[0] = 0
        np.cumsum(histogram, dtype=np.uint32, out=offsets[1:])
        # NumPy's argsort returns platform indexes (int64 here); narrow after
        # the stable ordering is built. The temporary is released before any
        # planner stage begins.
        ordered = np.argsort(block_ids, kind="stable").astype(
            np.uint32, copy=False)
        self._voxel_index_data = (
            np, states, histogram, source_histogram, ordered, offsets)
        return self._voxel_index_data

    def _matching_flat_indices(
        self,
        block_ids: Iterable[int],
        metadata: tuple[int, ...] | None = None,
    ):
        """Stable x/y/z-ordered uint32 positions for one state predicate."""

        index = self._voxel_index()
        if index is None:
            return None
        np, states, _histogram, _source_histogram, ordered, offsets = index
        ids = tuple(sorted(frozenset(block_ids)))
        metas = None if metadata is None else tuple(sorted(metadata))
        key = ids, metas
        cached = self._matching_indices.get(key)
        if cached is not None:
            return cached
        parts = [ordered[int(offsets[block_id]):int(offsets[block_id + 1])]
                 for block_id in ids]
        if not parts:
            matches = ordered[:0]
        elif len(parts) == 1:
            matches = parts[0]
        else:
            # Concatenating id buckets groups by id. Sort flat positions back
            # into native x/y/z order, exactly matching the legacy scan.
            matches = np.sort(np.concatenate(parts))
        if metas is not None:
            matches = matches[np.isin(states[matches] & 15, metas)]
        self._matching_indices[key] = matches
        return matches

    def _indices_in_region(self, indices, region: Bounds):
        """Filter stable flat indexes to an arbitrary clipped Bounds."""

        if region == self.bounds:
            return indices
        index = self._voxel_index()
        if index is None:
            return None
        np = index[0]
        flat = indices.astype(np.int64, copy=False)
        ix = flat // self._stride_x
        remainder = flat % self._stride_x
        iy = remainder // self._nz
        iz = remainder % self._nz
        keep = (
            (ix >= region.min_x - self.bounds.min_x)
            & (ix <= region.max_x - self.bounds.min_x)
            & (iy >= region.min_y - self.bounds.min_y)
            & (iy <= region.max_y - self.bounds.min_y)
            & (iz >= region.min_z - self.bounds.min_z)
            & (iz <= region.max_z - self.bounds.min_z)
        )
        return indices[keep]

    @staticmethod
    def _nearest_prefix(np, flat, distances, limit):
        """Exact ``(distance, x, y, z)`` prefix without sorting all cells.

        Flat BSNP indexes are already ordered x, then y, then z, so they are
        the legacy lexicographic tiebreak. ``argpartition`` finds the boundary
        in linear time; explicitly taking the first flat-index ties makes the
        result identical to a full stable lexsort, including a tie that
        crosses the requested limit.
        """

        if limit >= int(flat.size):
            return flat
        partition = np.argpartition(distances, limit - 1)
        threshold = distances[partition[limit - 1]]
        closer = np.flatnonzero(distances < threshold)
        needed = limit - int(closer.size)
        tied = np.flatnonzero(distances == threshold)
        if int(tied.size) > needed:
            tied = tied[np.argsort(flat[tied], kind="stable")[:needed]]
        selected = np.concatenate((closer, tied))
        order = np.lexsort((flat[selected], distances[selected]))
        return flat[selected[order]]

    def in_bounds(self, x: int, y: int, z: int) -> bool:
        return self.bounds.contains(x, y, z)

    def _index(self, x: int, y: int, z: int) -> int:
        if not self.in_bounds(x, y, z):
            raise SnapshotBoundsError(
                f"block ({x}, {y}, {z}) is outside {self.bounds}")
        return ((x - self.bounds.min_x) * self._stride_x
                + (y - self.bounds.min_y) * self._nz
                + (z - self.bounds.min_z))

    def _coordinates(self, index: int) -> tuple[int, int, int]:
        ix, remainder = divmod(index, self._stride_x)
        iy, iz = divmod(remainder, self._nz)
        return (self.bounds.min_x + ix,
                self.bounds.min_y + iy,
                self.bounds.min_z + iz)

    def _packed_at_index(self, index: int) -> int:
        return struct.unpack_from(
            "<H", self._require_open(), self._cells_at + index * 2)[0]

    def block(self, x: int, y: int, z: int) -> BlockState:
        packed = self._packed_at_index(self._index(x, y, z))
        return BlockState(packed >> 4, packed & 15)

    state = block

    def block_id(self, x: int, y: int, z: int) -> int:
        return self._packed_at_index(self._index(x, y, z)) >> 4

    def metadata(self, x: int, y: int, z: int) -> int:
        return self._packed_at_index(self._index(x, y, z)) & 15

    meta = metadata

    def _query_bounds(self, bounds: Bounds | None) -> Bounds | None:
        if bounds is None:
            return self.bounds
        if not isinstance(bounds, Bounds):
            raise TypeError("bounds must be a Bounds instance")
        return self.bounds.intersection(bounds)

    def find(
        self,
        blocks: int | str | Iterable[int | str],
        *,
        bounds: Bounds | None = None,
        limit: int | None = None,
        metadata: int | Iterable[int] | None = None,
    ) -> Iterator[BlockCell]:
        """Yield matching cells in deterministic x, y, z order."""
        ids = frozenset(_block_ids(blocks))
        metas = _metadata_values(metadata)
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0:
            return
        region = self._query_bounds(bounds)
        if region is None:
            return
        indexed = self._matching_flat_indices(ids, metas)
        if indexed is not None:
            indexed = self._indices_in_region(indexed, region)
            if limit is not None:
                indexed = indexed[:limit]
            states = self._voxel_index()[1]
            for value in indexed:
                index = int(value)
                packed = int(states[index])
                x, y, z = self._coordinates(index)
                yield BlockCell(x, y, z, packed >> 4, packed & 15)
            return
        emitted = 0
        for x in range(region.min_x, region.max_x + 1):
            for y in range(region.min_y, region.max_y + 1):
                index = self._index(x, y, region.min_z)
                for z in range(region.min_z, region.max_z + 1):
                    packed = self._packed_at_index(index)
                    block_id = packed >> 4
                    if (block_id in ids
                            and (metas is None or packed & 15 in metas)):
                        yield BlockCell(x, y, z, block_id, packed & 15)
                        emitted += 1
                        if limit is not None and emitted >= limit:
                            return
                    index += 1

    find_blocks = find

    def resource_target_pool(
        self,
        blocks: int | str | Iterable[int | str],
        *,
        bounds: Bounds | None = None,
        sample_limit: int = 4096,
        sample_origin: tuple[float, float, float] | None = None,
        prefer_exposed: bool = False,
        metadata: int | Iterable[int] | None = None,
    ) -> ResourceTargetPool:
        """Summarize resources without a connected-component flood fill.

        This is the planning query for abundant blocks such as stone.  It
        performs one packed-grid scan, reports an exact population count,
        and retains only a bounded deterministic target sample.  Callers
        needing exact face-connected cluster boundaries must continue to use
        :meth:`resource_clusters`; that public API and its semantics are
        intentionally unchanged.
        """

        ids = frozenset(_block_ids(blocks))
        metas = _metadata_values(metadata)
        if sample_limit < 0:
            raise ValueError("sample_limit must be non-negative")
        region = self._query_bounds(bounds)
        if region is None:
            return ResourceTargetPool((), 0, None, None, (), True)
        key = (tuple(sorted(ids)), metas, region, sample_limit,
               sample_origin, bool(prefer_exposed))
        cached = self._target_pool_cache.get(key)
        if cached is not None:
            return cached

        indexed = self._matching_flat_indices(ids, metas)
        if indexed is not None:
            indexed = self._indices_in_region(indexed, region)
            np, states, _histogram, _sources, _ordered, _offsets = \
                self._voxel_index()
            available = int(indexed.size)
            if not available:
                result = ResourceTargetPool((), 0, None, None, (), True)
                self._target_pool_cache[key] = result
                return result

            flat = indexed.astype(np.int64, copy=False)
            ix = flat // self._stride_x
            remainder = flat % self._stride_x
            iy = remainder // self._nz
            iz = remainder % self._nz
            world_x = ix + self.bounds.min_x
            world_y = iy + self.bounds.min_y
            world_z = iz + self.bounds.min_z
            population_bounds = Bounds(
                int(world_x.min()), int(world_y.min()), int(world_z.min()),
                int(world_x.max()), int(world_y.max()), int(world_z.max()))
            anchor = (int(world_x[0]), int(world_y[0]), int(world_z[0]))
            actual_ids = tuple(sorted(
                int(value) for value in np.unique(states[flat] >> 4)))

            if not sample_limit:
                selected = flat[:0]
            elif sample_origin is None:
                selected = flat[:sample_limit]
            else:
                ox, oy, oz = sample_origin
                distances = ((world_x + 0.5 - ox) ** 2
                             + (world_y + 0.5 - oy) ** 2
                             + (world_z + 0.5 - oz) ** 2)
                retained_limit = min(
                    available,
                    (max(4096, sample_limit * 64)
                     if prefer_exposed else sample_limit))
                retained = self._nearest_prefix(
                    np, flat, distances, retained_limit)
                if prefer_exposed:
                    exposed = np.zeros(retained.size, dtype=np.bool_)
                    for offset, value in enumerate(retained):
                        x, y, z = self._coordinates(int(value))
                        exposed[offset] = any(
                            self.bounds.contains(x + dx, y + dy, z + dz)
                            and int(states[self._index(
                                x + dx, y + dy, z + dz)]) >> 4 == 0
                            for dx, dy, dz in (
                                (0, 1, 0), (-1, 0, 0), (1, 0, 0),
                                (0, 0, -1), (0, 0, 1)))
                    retained_flat = retained.astype(np.int64, copy=False)
                    retained_x = retained_flat // self._stride_x \
                        + self.bounds.min_x
                    retained_rem = retained_flat % self._stride_x
                    retained_y = retained_rem // self._nz \
                        + self.bounds.min_y
                    retained_z = retained_rem % self._nz \
                        + self.bounds.min_z
                    retained_distance = (
                        (retained_x + 0.5 - ox) ** 2
                        + (retained_y + 0.5 - oy) ** 2
                        + (retained_z + 0.5 - oz) ** 2)
                    order = np.lexsort((
                        retained_z, retained_y, retained_x,
                        retained_distance, ~exposed))
                    retained = retained[order]
                selected = retained[:sample_limit]

            cells = tuple(sorted(
                self._coordinates(int(value)) for value in selected))
            result = ResourceTargetPool(
                actual_ids, available, population_bounds, anchor, cells,
                available <= sample_limit)
            self._target_pool_cache[key] = result
            return result

        actual_ids: set[int] = set()
        available = 0
        anchor = None
        min_x = min_y = min_z = None
        max_x = max_y = max_z = None
        samples: list[tuple[int, int, int]] = []
        nearest: list[tuple[float, int, int, int]] = []
        retained_limit = (
            max(4096, sample_limit * 64)
            if prefer_exposed and sample_limit else sample_limit)
        for cell in self.find(ids, bounds=region, metadata=metas):
            position = cell.position
            if anchor is None:
                anchor = position
            actual_ids.add(cell.block_id)
            available += 1
            min_x = cell.x if min_x is None else min(min_x, cell.x)
            min_y = cell.y if min_y is None else min(min_y, cell.y)
            min_z = cell.z if min_z is None else min(min_z, cell.z)
            max_x = cell.x if max_x is None else max(max_x, cell.x)
            max_y = cell.y if max_y is None else max(max_y, cell.y)
            max_z = cell.z if max_z is None else max(max_z, cell.z)
            if sample_origin is None:
                if len(samples) < sample_limit:
                    samples.append(position)
            elif retained_limit:
                ox, oy, oz = sample_origin
                distance = ((cell.x + 0.5 - ox) ** 2
                            + (cell.y + 0.5 - oy) ** 2
                            + (cell.z + 0.5 - oz) ** 2)
                candidate = (-distance, -cell.x, -cell.y, -cell.z)
                if len(nearest) < retained_limit:
                    heapq.heappush(nearest, candidate)
                elif candidate > nearest[0]:
                    heapq.heapreplace(nearest, candidate)
        if not available:
            result = ResourceTargetPool((), 0, None, None, (), True)
            self._target_pool_cache[key] = result
            return result
        if sample_origin is not None:
            candidates = [(-x, -y, -z)
                          for _distance, x, y, z in nearest]
            ox, oy, oz = sample_origin

            def rank(cell):
                x, y, z = cell
                exposed = prefer_exposed and any(
                    region.contains(x + dx, y + dy, z + dz)
                    and self.block_id(x + dx, y + dy, z + dz) == 0
                    for dx, dy, dz in (
                        (0, 1, 0), (-1, 0, 0), (1, 0, 0),
                        (0, 0, -1), (0, 0, 1)))
                distance = ((x + 0.5 - ox) ** 2
                            + (y + 0.5 - oy) ** 2
                            + (z + 0.5 - oz) ** 2)
                return (not exposed if prefer_exposed else False,
                        distance, cell)

            samples = sorted(candidates, key=rank)[:sample_limit]
        result = ResourceTargetPool(
            tuple(sorted(actual_ids)), available,
            Bounds(min_x, min_y, min_z, max_x, max_y, max_z),
            anchor, tuple(sorted(samples)), available <= sample_limit,
        )
        self._target_pool_cache[key] = result
        return result

    def find_source_fluids(
        self,
        blocks: int | str | Iterable[int | str],
        *,
        bounds: Bounds | None = None,
        limit: int | None = None,
    ) -> Iterator[BlockCell]:
        """Yield collectible source-fluid cells in stable coordinate order."""

        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0:
            return
        region = self._query_bounds(bounds)
        if region is None:
            return
        yield from self.find(
            blocks, bounds=region, limit=limit, metadata=0)

    def count_blocks_many(
        self,
        groups: dict[str, int | str | Iterable[int | str]],
        *,
        limits: dict[str, int] | None = None,
    ) -> dict[str, int]:
        """Count many resource groups in one packed-voxel traversal.

        Without ``limits`` counts are exact. With positive per-group limits,
        counts saturate at those minima and traversal stops once every group
        is satisfied; this is the efficient feasibility form used by seed
        scouts before any episode action.
        """

        if not isinstance(groups, dict) or not groups:
            raise ValueError("groups must be a nonempty mapping")
        wanted = {str(name): frozenset(_block_ids(blocks))
                  for name, blocks in groups.items()}
        caps = None
        if limits is not None:
            if set(limits) != set(wanted):
                raise ValueError("limits must name every resource group")
            caps = {str(name): int(value)
                    for name, value in limits.items()}
            if any(value < 1 for value in caps.values()):
                raise ValueError("resource count limits must be positive")
        index = self._voxel_index()
        if index is not None:
            histogram = index[2]
            counts = {
                name: sum(int(histogram[block_id]) for block_id in ids)
                for name, ids in wanted.items()
            }
            if caps is not None:
                counts = {name: min(count, caps[name])
                          for name, count in counts.items()}
            return counts

        by_id: dict[int, list[str]] = {}
        for name, ids in wanted.items():
            for block_id in ids:
                by_id.setdefault(block_id, []).append(name)
        counts = {name: 0 for name in wanted}
        packed = array("H")
        end = self._cells_at + self.bounds.volume * 2
        packed.frombytes(self._require_open()[self._cells_at:end])
        if sys.byteorder != "little":
            packed.byteswap()
        remaining = set(wanted)
        for value in packed:
            names = by_id.get(int(value) >> 4)
            if names is None:
                continue
            for name in names:
                if caps is None or counts[name] < caps[name]:
                    counts[name] += 1
                    if caps is not None and counts[name] >= caps[name]:
                        remaining.discard(name)
            if caps is not None and not remaining:
                break
        return counts

    def count_source_fluids_many(
        self,
        groups: dict[str, int | str | Iterable[int | str]],
        *,
        limits: dict[str, int] | None = None,
    ) -> dict[str, int]:
        """Count source-level fluids (metadata zero) in one traversal.

        Vanilla buckets can collect only source blocks.  Ordinary block-id
        counts deliberately include flowing states, so fluid-consuming
        planners use this stricter query instead of qualifying worlds that
        contain enough liquid visually but not enough collectible sources.
        Positive ``limits`` have the same saturating feasibility semantics as
        :meth:`count_blocks_many`.
        """

        if not isinstance(groups, dict) or not groups:
            raise ValueError("groups must be a nonempty mapping")
        wanted = {str(name): frozenset(_block_ids(blocks))
                  for name, blocks in groups.items()}
        caps = None
        if limits is not None:
            if set(limits) != set(wanted):
                raise ValueError("limits must name every source-fluid group")
            caps = {str(name): int(value)
                    for name, value in limits.items()}
            if any(value < 1 for value in caps.values()):
                raise ValueError(
                    "source-fluid count limits must be positive")
        index = self._voxel_index()
        if index is not None:
            histogram = index[3]
            counts = {
                name: sum(int(histogram[block_id]) for block_id in ids)
                for name, ids in wanted.items()
            }
            if caps is not None:
                counts = {name: min(count, caps[name])
                          for name, count in counts.items()}
            return counts

        by_id: dict[int, list[str]] = {}
        for name, ids in wanted.items():
            for block_id in ids:
                by_id.setdefault(block_id, []).append(name)
        counts = {name: 0 for name in wanted}
        packed = array("H")
        end = self._cells_at + self.bounds.volume * 2
        packed.frombytes(self._require_open()[self._cells_at:end])
        if sys.byteorder != "little":
            packed.byteswap()
        remaining = set(wanted)
        for value in packed:
            if int(value) & 15:
                continue
            names = by_id.get(int(value) >> 4)
            if names is None:
                continue
            for name in names:
                if caps is None or counts[name] < caps[name]:
                    counts[name] += 1
                    if caps is not None and counts[name] >= caps[name]:
                        remaining.discard(name)
            if caps is not None and not remaining:
                break
        return counts

    def nearest(
        self,
        blocks: int | str | Iterable[int | str],
        origin: tuple[float, float, float],
        *,
        limit: int = 1,
        bounds: Bounds | None = None,
    ) -> tuple[BlockCell, ...]:
        """Return nearest matches with O(``limit``) auxiliary memory."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        ox, oy, oz = origin

        def distance_squared(cell: BlockCell) -> float:
            # A block occupies a unit cube.  Distances exposed by task and
            # planning APIs are to its center, matching live observations.
            return ((cell.x + 0.5 - ox) ** 2
                    + (cell.y + 0.5 - oy) ** 2
                    + (cell.z + 0.5 - oz) ** 2)

        heap: list[tuple[float, int, int, int, BlockCell]] = []
        for cell in self.find(blocks, bounds=bounds):
            distance = distance_squared(cell)
            # A max heap represented with negated distance and coordinates.
            item = (-distance, -cell.x, -cell.y, -cell.z, cell)
            if len(heap) < limit:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
        return tuple(sorted(
            (item[-1] for item in heap),
            key=lambda cell: (
                distance_squared(cell),
                cell.x, cell.y, cell.z),
        ))

    def is_fluid(self, x: int, y: int, z: int) -> bool:
        return self.block_id(x, y, z) in _FLUID

    def fluid_at(self, x: int, y: int, z: int) -> str | None:
        block_id = self.block_id(x, y, z)
        if block_id in _WATER:
            return "water"
        if block_id in _LAVA:
            return "lava"
        return None

    def hazard_at(self, x: int, y: int, z: int) -> str | None:
        return _HAZARDS.get(self.block_id(x, y, z))

    def is_hazard(self, x: int, y: int, z: int) -> bool:
        return self.hazard_at(x, y, z) is not None

    def near_hazard(
        self, x: int, y: int, z: int, *, radius: int = 1,
    ) -> bool:
        """Whether a captured hazard is within Chebyshev ``radius``."""
        if radius < 0:
            raise ValueError("radius must be non-negative")
        for xx in range(max(self.bounds.min_x, x - radius),
                        min(self.bounds.max_x, x + radius) + 1):
            for yy in range(max(self.bounds.min_y, y - radius),
                            min(self.bounds.max_y, y + radius) + 1):
                for zz in range(max(self.bounds.min_z, z - radius),
                                min(self.bounds.max_z, z + radius) + 1):
                    if self.is_hazard(xx, yy, zz):
                        return True
        return False

    def is_solid(self, x: int, y: int, z: int) -> bool:
        return self.block_id(x, y, z) not in _NON_SOLID

    def has_headroom(
        self, x: int, y: int, z: int, *, height: int = 2,
        allow_fluid: bool = False,
    ) -> bool:
        """Whether ``height`` cells from a prospective feet cell are clear."""
        if height <= 0:
            raise ValueError("height must be positive")
        if not self.in_bounds(x, y, z) or not self.in_bounds(
                x, y + height - 1, z):
            return False
        for yy in range(y, y + height):
            block_id = self.block_id(x, yy, z)
            if block_id not in _NON_SOLID:
                return False
            if not allow_fluid and block_id in _FLUID:
                return False
            if block_id in _HAZARDS:
                return False
        return True

    def is_standable(
        self, x: int, y: int, z: int, *, height: int = 2,
        hazard_radius: int = 0,
    ) -> bool:
        """Whether a player can stand safely with feet at ``(x, y, z)``."""
        if not self.in_bounds(x, y - 1, z):
            return False
        return (self.is_solid(x, y - 1, z)
                and not self.is_hazard(x, y - 1, z)
                and self.has_headroom(x, y, z, height=height)
                and not self.near_hazard(
                    x, y, z, radius=hazard_radius))

    def surface_y(
        self, x: int, z: int, *, headroom: int = 2,
        min_y: int | None = None, max_y: int | None = None,
    ) -> int | None:
        """Highest solid support with requested clear space above it."""
        if not (self.bounds.min_x <= x <= self.bounds.max_x
                and self.bounds.min_z <= z <= self.bounds.max_z):
            raise SnapshotBoundsError(
                f"column ({x}, {z}) is outside {self.bounds}")
        if headroom < 0:
            raise ValueError("headroom must be non-negative")
        low = self.bounds.min_y if min_y is None else max(
            self.bounds.min_y, min_y)
        high = self.bounds.max_y if max_y is None else min(
            self.bounds.max_y, max_y)
        if low > high:
            return None
        for y in range(high, low - 1, -1):
            if not self.is_solid(x, y, z) or self.is_hazard(x, y, z):
                continue
            if headroom == 0:
                return y
            if self.has_headroom(x, y + 1, z, height=headroom):
                return y
        return None

    def resource_clusters(
        self,
        blocks: int | str | Iterable[int | str],
        *,
        bounds: Bounds | None = None,
        min_size: int = 1,
        sample_limit: int = 4096,
        sample_origin: tuple[float, float, float] | None = None,
        prefer_exposed: bool = False,
        metadata: int | Iterable[int] | None = None,
    ) -> Iterator[ResourceCluster]:
        """Yield exact six-neighbor clusters without expanding the world.

        The traversal uses one byte per snapshot cell plus a compact uint32
        queue.  This keeps even deliberately broad queries bounded compared
        with Python coordinate tuples.
        """
        ids = frozenset(_block_ids(blocks))
        metas = _metadata_values(metadata)
        if min_size <= 0:
            raise ValueError("min_size must be positive")
        if sample_limit < 0:
            raise ValueError("sample_limit must be non-negative")
        region = self._query_bounds(bounds)
        if region is None:
            return
        key = (tuple(sorted(ids)), metas, region, min_size, sample_limit,
               sample_origin, bool(prefer_exposed))
        cached = self._cluster_cache.get(key)
        if cached is not None:
            yield from cached
            return
        results = []
        visited = bytearray(self.bounds.volume)
        neighbors = (
            (-1, 0, 0), (0, -1, 0), (0, 0, -1),
            (0, 0, 1), (0, 1, 0), (1, 0, 0),
        )
        for first in self.find(ids, bounds=region, metadata=metas):
            start = self._index(first.x, first.y, first.z)
            if visited[start]:
                continue
            anchor = first.position
            visited[start] = 1
            queue = array("I", (start,))
            cursor = 0
            samples: list[tuple[int, int, int]] = []
            nearest: list[tuple[float, int, int, int]] = []
            actual_ids: set[int] = set()
            min_x = max_x = first.x
            min_y = max_y = first.y
            min_z = max_z = first.z
            while cursor < len(queue):
                index = queue[cursor]
                cursor += 1
                x, y, z = self._coordinates(index)
                packed = self._packed_at_index(index)
                actual_ids.add(packed >> 4)
                if sample_origin is None:
                    if len(samples) < sample_limit:
                        samples.append((x, y, z))
                elif sample_limit:
                    ox, oy, oz = sample_origin
                    distance = ((x + 0.5 - ox) ** 2
                                + (y + 0.5 - oy) ** 2
                                + (z + 0.5 - oz) ** 2)
                    candidate = (-distance, -x, -y, -z)
                    retained_limit = (max(4096, sample_limit * 64)
                                      if prefer_exposed else sample_limit)
                    if len(nearest) < retained_limit:
                        heapq.heappush(nearest, candidate)
                    elif candidate > nearest[0]:
                        heapq.heapreplace(nearest, candidate)
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)
                min_z, max_z = min(min_z, z), max(max_z, z)
                for dx, dy, dz in neighbors:
                    xx, yy, zz = x + dx, y + dy, z + dz
                    if not region.contains(xx, yy, zz):
                        continue
                    neighbor = self._index(xx, yy, zz)
                    if visited[neighbor]:
                        continue
                    neighbor_packed = self._packed_at_index(neighbor)
                    if (neighbor_packed >> 4 not in ids
                            or (metas is not None
                                and neighbor_packed & 15 not in metas)):
                        continue
                    visited[neighbor] = 1
                    queue.append(neighbor)
            size = len(queue)
            if size >= min_size:
                if sample_origin is not None:
                    candidates = [(-x, -y, -z)
                                  for _distance, x, y, z in nearest]
                    if prefer_exposed:
                        ox, oy, oz = sample_origin

                        def sample_rank(cell):
                            x, y, z = cell
                            exposed = any(
                                self.bounds.contains(
                                    x + dx, y + dy, z + dz)
                                and self._packed_at_index(self._index(
                                    x + dx, y + dy, z + dz)) >> 4 == 0
                                for dx, dy, dz in (
                                    (0, 1, 0), (-1, 0, 0), (1, 0, 0),
                                    (0, 0, -1), (0, 0, 1)))
                            distance = ((x + 0.5 - ox) ** 2
                                        + (y + 0.5 - oy) ** 2
                                        + (z + 0.5 - oz) ** 2)
                            return (not exposed, distance, cell)

                        candidates.sort(key=sample_rank)
                    samples = candidates[:sample_limit]
                cells = tuple(sorted(samples))
                result = ResourceCluster(
                    block_ids=tuple(sorted(actual_ids)), size=size,
                    bounds=Bounds(min_x, min_y, min_z,
                                  max_x, max_y, max_z),
                    anchor=anchor,
                    cells=cells, cells_complete=size <= sample_limit,
                )
                results.append(result)
                yield result
        self._cluster_cache[key] = tuple(results)


# Short name for policy code, while the full name documents the data source.
WorldMap = SnapshotWorldMap


__all__ = [
    "BlockCell", "BlockState", "Bounds", "PlayerPose", "ResourceCluster",
    "ResourceTargetPool", "SnapshotFormatError", "SnapshotHeader",
    "SnapshotWorldMap", "WorldMap",
]
