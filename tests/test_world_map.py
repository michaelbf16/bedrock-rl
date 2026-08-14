import hashlib
import builtins
from pathlib import Path
import struct
import tempfile
import time
import unittest
from unittest.mock import patch

from bedrock_rl.sdg.policy.world import (
    BlockState,
    Bounds,
    SnapshotFormatError,
    SnapshotWorldMap,
)
from bedrock_rl.env.snapshots import (
    ANGLE_OFF,
    HEAD_SIZE,
    INV_OFF,
    INV_SLOTS,
    POS_OFF,
    patch_initial_state,
)
from bedrock_rl.env.initial_state import InitialStateSpec


ITEM_COUNT_OFF = INV_OFF + INV_SLOTS * 3 * 4
REGION_HEADER_OFF = ITEM_COUNT_OFF + struct.calcsize("<I")


def write_snapshot(path, *, truncate=0, version=1):
    seed, tick = -37, 8123
    rx, ry, rz, nx, ny, nz = 10, 60, -2, 3, 5, 3
    buf = bytearray(HEAD_SIZE + nx * ny * nz * 2 + 4)
    struct.pack_into("<4sIqqii", buf, 0, b"BSNP", version,
                     seed, tick, 2, -4)
    struct.pack_into("<3d", buf, POS_OFF, 11.5, 61.0, -0.5)
    struct.pack_into("<2f", buf, ANGLE_OFF, 90.0, -12.5)
    struct.pack_into("<I", buf, ITEM_COUNT_OFF, 0)
    struct.pack_into("<6i", buf, REGION_HEADER_OFF,
                     rx, ry, rz, nx, ny, nz)

    def set_block(x, y, z, block_id, meta=0):
        index = ((x - rx) * ny + (y - ry)) * nz + (z - rz)
        struct.pack_into("<H", buf, HEAD_SIZE + index * 2,
                         block_id << 4 | meta)

    for x in range(rx, rx + nx):
        for z in range(rz, rz + nz):
            set_block(x, 60, z, 1)
    set_block(10, 61, -2, 56, 3)
    set_block(11, 61, -2, 56)
    set_block(12, 61, 0, 56, 7)
    set_block(10, 61, -1, 9)
    set_block(11, 61, -1, 10)
    set_block(12, 61, -1, 51)
    # Exact empty coal-coordinate trailer.
    struct.pack_into("<I", buf, len(buf) - 4, 0)
    path.write_bytes(buf[:-truncate] if truncate else buf)
    return bytes(buf)


def write_dense_stone_snapshot(path, *, nx=129, ny=24, nz=129):
    """Write a moderate dense component for a planner-query time bound."""

    volume = nx * ny * nz
    buf = bytearray(HEAD_SIZE + volume * 2 + 4)
    struct.pack_into("<4sIqqii", buf, 0, b"BSNP", 1, 777, 0, 0, 0)
    struct.pack_into("<3d", buf, POS_OFF, 0.5, 20.0, 0.5)
    struct.pack_into("<2f", buf, ANGLE_OFF, 0.0, 0.0)
    struct.pack_into("<I", buf, ITEM_COUNT_OFF, 0)
    struct.pack_into("<6i", buf, REGION_HEADER_OFF,
                     -(nx // 2), 0, -(nz // 2), nx, ny, nz)
    buf[HEAD_SIZE:HEAD_SIZE + volume * 2] = \
        struct.pack("<H", 1 << 4) * volume
    struct.pack_into("<I", buf, len(buf) - 4, 0)
    path.write_bytes(buf)


class SnapshotWorldMapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "world.bsnp"
        self.raw = write_snapshot(self.path)

    def tearDown(self):
        self.temporary.cleanup()

    def test_header_hash_bounds_and_exact_block_lookup(self):
        with SnapshotWorldMap(self.path) as world:
            self.assertEqual(world.seed, -37)
            self.assertEqual(world.tick, 8123)
            self.assertEqual(world.header.origin_x, 2)
            self.assertEqual(world.header.origin_z, -4)
            self.assertEqual(world.player.x, 13.5)
            self.assertEqual(world.player.z, -4.5)
            self.assertAlmostEqual(world.player.pitch, -12.5)
            self.assertEqual(world.bounds, Bounds(10, 60, -2, 12, 64, 0))
            self.assertEqual(world.bounds.shape, (3, 5, 3))
            self.assertEqual(world.bounds.volume, 45)
            self.assertEqual(world.snapshot_sha256,
                             hashlib.sha256(self.raw).hexdigest())
            self.assertTrue(world.in_bounds(12, 64, 0))
            self.assertFalse(world.in_bounds(13, 64, 0))
            self.assertEqual(world.block(10, 61, -2), BlockState(56, 3))
            self.assertEqual(world.block_id(10, 61, -2), 56)
            self.assertEqual(world.metadata(10, 61, -2), 3)
            with self.assertRaises(IndexError):
                world.block_id(9, 61, -2)
        self.assertTrue(world.closed)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            world.block_id(10, 61, -2)

    def test_designated_spawn_round_trips_between_world_and_local_coords(self):
        spec = InitialStateSpec.parse({
            "spawn": {"points": [{
                "x": 12.25, "y": 62.0, "z": -1.75,
                "yaw": 30, "pitch": -15,
            }]},
        })
        patch_initial_state(str(self.path), spec, seed=-37)

        with SnapshotWorldMap(self.path) as world:
            self.assertEqual(
                (world.player.x, world.player.y, world.player.z),
                (12.25, 62.0, -1.75),
            )
            self.assertEqual(
                (world.player.yaw, world.player.pitch), (30.0, -15.0))
        # The binary form remains explicitly local to its recorded window.
        raw = self.path.read_bytes()
        self.assertEqual(
            struct.unpack_from("<3d", raw, POS_OFF),
            (10.25, 62.0, 2.25),
        )

    def test_find_and_nearest_are_deterministic_and_bounded(self):
        with SnapshotWorldMap(self.path) as world:
            cells = list(world.find("diamond_ore"))
            self.assertEqual(
                [(cell.x, cell.y, cell.z, cell.meta) for cell in cells],
                [(10, 61, -2, 3), (11, 61, -2, 0),
                 (12, 61, 0, 7)])
            self.assertEqual(
                [cell.position for cell in world.find(56, limit=2)],
                [(10, 61, -2), (11, 61, -2)])
            self.assertEqual(list(world.find(56, limit=0)), [])
            self.assertEqual(
                [cell.position for cell in world.find(
                    56, bounds=Bounds(11, 60, -2, 12, 64, -2))],
                [(11, 61, -2)])
            self.assertEqual(
                [cell.position for cell in world.nearest(
                    56, (12, 61, 0), limit=2)],
                [(12, 61, 0), (11, 61, -2)])

    def test_many_resource_counts_share_one_packed_traversal(self):
        with SnapshotWorldMap(self.path) as world:
            self.assertEqual(world.count_blocks_many({
                "diamond": "diamond_ore",
                "hazards": ("flowing_lava", "lava", "fire"),
                "stone": "stone",
            }), {"diamond": 3, "hazards": 2, "stone": 9})
            # Feasibility counts saturate and stop once every minimum is met.
            self.assertEqual(world.count_blocks_many(
                {"diamond": "diamond_ore", "stone": "stone"},
                limits={"diamond": 2, "stone": 4}),
                {"diamond": 2, "stone": 4})
            with self.assertRaisesRegex(ValueError, "every resource"):
                world.count_blocks_many(
                    {"diamond": "diamond_ore"}, limits={"stone": 1})

    def test_source_fluid_counts_exclude_flowing_metadata(self):
        with SnapshotWorldMap(self.path) as world:
            self.assertEqual(
                [(cell.position, cell.block_id, cell.meta)
                 for cell in world.find_source_fluids(
                     ("flowing_water", "water"))],
                [((10, 61, -1), 9, 0)],
            )
            self.assertEqual(world.count_source_fluids_many({
                "water": ("flowing_water", "water"),
                "lava": ("flowing_lava", "lava"),
            }), {"water": 1, "lava": 1})
            self.assertEqual(world.count_source_fluids_many(
                {"water": ("flowing_water", "water"),
                 "lava": ("flowing_lava", "lava")},
                limits={"water": 1, "lava": 1}),
                {"water": 1, "lava": 1})
            with self.assertRaisesRegex(ValueError, "every source-fluid"):
                world.count_source_fluids_many(
                    {"water": "water"}, limits={"lava": 1})

    def test_surface_headroom_fluid_and_hazard_queries(self):
        with SnapshotWorldMap(self.path) as world:
            self.assertEqual(world.surface_y(10, 0), 60)
            self.assertTrue(world.has_headroom(10, 61, 0))
            self.assertTrue(world.is_standable(10, 61, 0))
            self.assertEqual(world.fluid_at(10, 61, -1), "water")
            self.assertEqual(world.fluid_at(11, 61, -1), "lava")
            self.assertIsNone(world.fluid_at(10, 61, 0))
            self.assertTrue(world.is_fluid(10, 61, -1))
            self.assertFalse(world.has_headroom(10, 61, -1))
            self.assertTrue(world.has_headroom(
                10, 61, -1, allow_fluid=True))
            self.assertEqual(world.hazard_at(11, 61, -1), "lava")
            self.assertEqual(world.hazard_at(12, 61, -1), "fire")
            self.assertTrue(world.near_hazard(10, 61, 0, radius=1))
            self.assertFalse(world.is_standable(
                10, 61, 0, hazard_radius=1))
            self.assertIsNone(world.surface_y(12, -1))
            self.assertEqual(world.surface_y(12, -1, headroom=0), 60)

    def test_resource_clusters_are_exact_compact_and_deterministic(self):
        with SnapshotWorldMap(self.path) as world:
            clusters = list(world.resource_clusters(
                "diamond_ore", sample_limit=1))
            self.assertEqual([cluster.size for cluster in clusters], [2, 1])
            self.assertEqual(clusters[0].anchor, (10, 61, -2))
            self.assertEqual(clusters[0].bounds,
                             Bounds(10, 61, -2, 11, 61, -2))
            self.assertEqual(clusters[0].block_ids, (56,))
            self.assertEqual(clusters[0].cells, ((10, 61, -2),))
            self.assertFalse(clusters[0].cells_complete)
            self.assertTrue(clusters[1].cells_complete)
            self.assertEqual(
                [cluster.anchor for cluster in world.resource_clusters(
                    56, min_size=2)],
                [(10, 61, -2)])
            summary = next(world.resource_clusters(56, sample_limit=0))
            self.assertEqual(summary.anchor, (10, 61, -2))
            self.assertEqual(summary.cells, ())
            self.assertFalse(summary.cells_complete)

    def test_cluster_sampling_can_prefer_cells_nearest_an_origin(self):
        with SnapshotWorldMap(self.path) as world:
            cluster = next(world.resource_clusters(
                "diamond_ore", sample_limit=1,
                sample_origin=(11.5, 61.5, -1.5)))
            self.assertEqual(cluster.size, 2)
            self.assertEqual(cluster.cells, ((11, 61, -2),))
            # Exact cluster accounting and its stable file-order anchor do
            # not change merely because planning asks for a local sample.
            self.assertEqual(cluster.anchor, (10, 61, -2))

    def test_resource_target_pool_is_exact_bounded_and_deterministic(self):
        with SnapshotWorldMap(self.path) as world:
            first = world.resource_target_pool(
                "diamond_ore", sample_limit=2,
                sample_origin=(11.5, 61.5, -1.5))
            again = world.resource_target_pool(
                "diamond_ore", sample_limit=2,
                sample_origin=(11.5, 61.5, -1.5))
            complete = world.resource_target_pool(
                "diamond_ore", sample_limit=8)

        self.assertEqual(first, again)
        self.assertEqual(first.available_count, 3)
        self.assertEqual(first.block_ids, (56,))
        self.assertEqual(first.anchor, (10, 61, -2))
        self.assertEqual(first.bounds, Bounds(10, 61, -2, 12, 61, 0))
        self.assertEqual(first.cells,
                         ((10, 61, -2), (11, 61, -2)))
        self.assertFalse(first.cells_complete)
        self.assertEqual(
            complete.cells,
            ((10, 61, -2), (11, 61, -2), (12, 61, 0)),
        )
        self.assertTrue(complete.cells_complete)

    def test_indexed_queries_match_legacy_scan_in_order_and_metadata(self):
        """The acceleration cannot alter planner targets or cluster order."""

        def queries(world):
            return {
                "find": tuple(world.find(
                    (1, 56),
                    bounds=Bounds(10, 60, -2, 12, 62, 0))),
                "find_meta": tuple(world.find(
                    56, metadata=(0, 7))),
                "counts": world.count_blocks_many({
                    "diamond": 56, "solid": (1, 56),
                }),
                "source_counts": world.count_source_fluids_many({
                    "water": (8, 9), "lava": (10, 11),
                }),
                "pool": world.resource_target_pool(
                    (1, 56), bounds=Bounds(10, 60, -2, 12, 62, 0),
                    sample_limit=4, sample_origin=(11.5, 61.5, -1.5),
                    prefer_exposed=True),
                "pool_meta": world.resource_target_pool(
                    56, metadata=0, sample_limit=4),
                "clusters": tuple(world.resource_clusters(
                    56, metadata=(0, 3), sample_limit=2,
                    sample_origin=(11.5, 61.5, -1.5))),
            }

        with SnapshotWorldMap(self.path) as accelerated:
            indexed = queries(accelerated)

        real_import = builtins.__import__

        def without_numpy(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError("forced legacy packed scan")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=without_numpy), \
                SnapshotWorldMap(self.path) as legacy:
            scanned = queries(legacy)

        self.assertEqual(indexed, scanned)
        self.assertEqual(
            [cell.position for cell in indexed["find_meta"]],
            [(11, 61, -2), (12, 61, 0)],
        )
        self.assertEqual(indexed["pool_meta"].available_count, 1)

    def test_one_immutable_index_and_query_results_are_reused(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy planner acceleration is not installed")
        with SnapshotWorldMap(self.path) as world:
            self.assertIsNone(world._voxel_index_data)
            world.count_blocks_many({"diamond": 56})
            index = world._voxel_index_data
            self.assertIsNotNone(index)
            index_id = id(index)
            # Counts, target pools, and clusters all retain this exact index.
            first = world.resource_target_pool(56, metadata=0)
            self.assertIs(first, world.resource_target_pool(56, metadata=0))
            clusters = tuple(world.resource_clusters(56, metadata=0))
            self.assertEqual(
                clusters, tuple(world.resource_clusters(56, metadata=0)))
            world.count_source_fluids_many({"water": (8, 9)})
            self.assertEqual(index_id, id(world._voxel_index_data))
            # One stable position ordering serves every queried block id.
            self.assertEqual(index[4].dtype.itemsize, 4)
            self.assertEqual(index[4].size, world.bounds.volume)
            del index

    def test_metadata_query_validation_is_strict(self):
        with SnapshotWorldMap(self.path) as world:
            for value in (True, -1, 16, (), (0, "1")):
                with self.subTest(metadata=value):
                    with self.assertRaises((TypeError, ValueError)):
                        world.resource_target_pool(1, metadata=value)

    def test_nearest_prefix_preserves_boundary_ties_exactly(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy planner acceleration is not installed")
        # Positions 4/5/6 tie at the selection boundary. Native flat order
        # must decide which two survive, matching the former full lexsort.
        flat = np.asarray([9, 4, 6, 5, 2, 8], dtype=np.uint32)
        distances = np.asarray([4.0, 2.0, 2.0, 2.0, 1.0, 3.0])
        expected_order = np.lexsort((flat, distances))[:3]
        expected = flat[expected_order]
        actual = SnapshotWorldMap._nearest_prefix(
            np, flat, distances, 3)
        np.testing.assert_array_equal(actual, expected)

    def test_dense_resource_target_pool_has_a_wall_time_bound(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy planner acceleration is not installed")
        dense = Path(self.temporary.name) / "dense.bsnp"
        write_dense_stone_snapshot(dense)
        with SnapshotWorldMap(dense) as world:
            started = time.perf_counter()
            pool = world.resource_target_pool(
                "stone", sample_limit=64,
                sample_origin=(0.5, 20.0, 0.5),
                prefer_exposed=True)
            elapsed = time.perf_counter() - started
        self.assertEqual(pool.available_count, 129 * 24 * 129)
        self.assertEqual(len(pool.cells), 64)
        self.assertLess(
            elapsed, 0.75,
            f"dense resource target query regressed to {elapsed:.3f}s",
        )

    def test_rejects_truncation_bad_version_and_trailing_bytes(self):
        truncated = Path(self.temporary.name) / "truncated.bsnp"
        write_snapshot(truncated, truncate=1)
        with self.assertRaises(SnapshotFormatError):
            SnapshotWorldMap(truncated)

        bad_version = Path(self.temporary.name) / "bad-version.bsnp"
        write_snapshot(bad_version, version=2)
        with self.assertRaisesRegex(SnapshotFormatError, "v2"):
            SnapshotWorldMap(bad_version)

        trailing = Path(self.temporary.name) / "trailing.bsnp"
        trailing.write_bytes(self.raw + b"x")
        with self.assertRaisesRegex(SnapshotFormatError, "size mismatch"):
            SnapshotWorldMap(trailing)

        empty = Path(self.temporary.name) / "empty.bsnp"
        empty.write_bytes(b"")
        with self.assertRaisesRegex(SnapshotFormatError, "shorter"):
            SnapshotWorldMap(empty)


if __name__ == "__main__":
    unittest.main()
