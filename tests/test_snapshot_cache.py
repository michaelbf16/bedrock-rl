import tempfile
import unittest
import builtins
from pathlib import Path
import struct
from unittest.mock import patch

from bedrock_rl.sdg.policy.world import SnapshotWorldMap
from bedrock_rl.sdg.policy.spatial import is_surface_stand
from bedrock_rl.env.initial_state import InitialStateSpec
from bedrock_rl.env.snapshots import (
    ANGLE_OFF,
    BOX_OFF,
    HEAD_SIZE,
    INV_OFF,
    INV_SLOTS,
    POS_OFF,
    SnapshotRejected,
    _check_layout,
    _constraint_index,
    _nearest_in_band,
    _validate_snapshot_constraints,
    build_snapshot,
    engine_cache_key,
    patch_initial_state,
    select_spawn_point,
    snapshot_path,
)
from bedrock_rl.env.task import Task
from tests.test_world_map import write_snapshot


_ITEM_COUNT_OFF = INV_OFF + INV_SLOTS * 3 * 4
_REGION_HEADER_OFF = _ITEM_COUNT_OFF + struct.calcsize("<I")


def write_spawn_snapshot(path, *, resources=(), unsafe=(),
                         player=(4.5, 61.0, 4.5)):
    """Small natural region for exact constraint-spawn contracts."""
    rx, ry, rz, nx, ny, nz = 0, 60, 0, 9, 6, 9
    buf = bytearray(HEAD_SIZE + nx * ny * nz * 2 + 4)
    struct.pack_into("<4sIqqii", buf, 0, b"BSNP", 1, 19, 20, 0, 0)
    struct.pack_into("<3d", buf, POS_OFF, *player)
    struct.pack_into("<6d", buf, BOX_OFF,
                     player[0] - 0.3, player[1], player[2] - 0.3,
                     player[0] + 0.3, player[1] + 1.8, player[2] + 0.3)
    struct.pack_into("<2f", buf, ANGLE_OFF, 0.0, 0.0)
    struct.pack_into("<I", buf, _ITEM_COUNT_OFF, 0)
    struct.pack_into("<6i", buf, _REGION_HEADER_OFF,
                     rx, ry, rz, nx, ny, nz)

    def set_block(x, y, z, block_id):
        index = ((x - rx) * ny + (y - ry)) * nz + (z - rz)
        struct.pack_into("<H", buf, HEAD_SIZE + index * 2,
                         int(block_id) << 4)

    for x in range(nx):
        for z in range(nz):
            set_block(x, 60, z, 1)
    for x, y, z, block_id in resources:
        set_block(x, y, z, block_id)
    for x, y, z, block_id in unsafe:
        set_block(x, y, z, block_id)
    struct.pack_into("<I", buf, len(buf) - 4, 0)
    path.write_bytes(buf)


def spawn_spec(*constraints, points=(), hazard_clearance=0):
    return InitialStateSpec.parse({"spawn": {
        "points": list(points),
        "constraints": list(constraints),
        "hazard_clearance": hazard_clearance,
    }})


def task_spec(directory=None, radius=None):
    spec = {
        "name": "snapshot-cache",
        "goal": "stand still",
        "initial_state": {"world": {"time": 0}},
        "success": {"type": "pitch_below", "value": 90},
    }
    if directory is not None:
        spec["snapshots_dir"] = str(directory)
    if radius is not None:
        spec["snapshot_radius"] = radius
    return spec


class SnapshotCacheTests(unittest.TestCase):
    def test_nearest_band_uses_the_nearest_not_any_matching_block(self):
        constraint = InitialStateSpec.parse({"spawn": {"constraints": [{
            "block": "log", "min_distance": 4, "max_distance": 8,
        }]}}).constraints[0]
        # One match is in-band, but a second is too close. The declaration is
        # about the nearest matching block, so this point must be rejected.
        index = _constraint_index(((0, 0, 0), (6, 0, 0)))
        self.assertFalse(_nearest_in_band(
            (1.5, 0.5, 0.5), constraint, index))

    def test_constraint_spawn_relocates_for_the_nearest_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "world.bsnp"
            write_spawn_snapshot(path, resources=(
                (1, 60, 4, 17), (7, 60, 4, 17),
            ))
            spec = spawn_spec({
                "block": "log", "min_distance": 2.0,
                "max_distance": 2.1,
            })
            point = select_spawn_point(path, spec, 91)

            self.assertNotEqual((point.x, point.y, point.z),
                                (4.5, 61.0, 4.5))
            patch_initial_state(path, spec, 91, selected_point=point)
            _validate_snapshot_constraints(path, spec)

    def test_constraint_spawn_satisfies_multiple_bands_deterministically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "world.bsnp"
            write_spawn_snapshot(path, resources=(
                (1, 60, 4, 17), (7, 60, 4, 56),
            ))
            spec = spawn_spec(
                {"block": "log", "min_distance": 3.0,
                 "max_distance": 3.1},
                {"block": "diamond_ore", "min_distance": 3.0,
                 "max_distance": 3.1},
            )
            first = select_spawn_point(path, spec, 1234)
            second = select_spawn_point(path, spec, 1234)

            self.assertEqual(first, second)
            patch_initial_state(path, spec, 1234, selected_point=first)
            _validate_snapshot_constraints(path, spec)

    def test_constraint_spawn_requires_dry_headroom_and_safe_support(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "world.bsnp"
            write_spawn_snapshot(
                path, resources=((1, 60, 4, 17),),
                unsafe=((4, 61, 4, 9), (3, 62, 4, 1),
                        (4, 60, 3, 11)))
            spec = spawn_spec({
                "block": "log", "min_distance": 0,
                "max_distance": 8,
            })
            point = select_spawn_point(path, spec, 5)

            self.assertNotEqual((point.x, point.y, point.z),
                                (4.5, 61.0, 4.5))
            # The chosen point remains viable after the ordinary patch and
            # the public world-map parser observes the same world position.
            patch_initial_state(path, spec, 5, selected_point=point)
            with SnapshotWorldMap(path) as world:
                self.assertEqual(
                    (world.player.x, world.player.y, world.player.z),
                    (point.x, point.y, point.z))
                self.assertTrue(world.is_standable(
                    int(point.x - 0.5), int(point.y),
                    int(point.z - 0.5)))

    def test_constraint_spawn_enforces_declared_hazard_clearance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "world.bsnp"
            write_spawn_snapshot(
                path, resources=((1, 60, 4, 17),),
                unsafe=((5, 61, 4, 11),))
            spec = InitialStateSpec.parse({"spawn": {
                "hazard_clearance": 1,
                "constraints": [{
                    "block": "log", "min_distance": 0,
                    "max_distance": 8,
                }],
            }})

            point = select_spawn_point(path, spec, 5)

            self.assertNotEqual(
                (point.x, point.y, point.z), (4.5, 61.0, 4.5))
            with SnapshotWorldMap(path) as world:
                self.assertTrue(is_surface_stand(
                    world,
                    (int(point.x - 0.5), int(point.y),
                     int(point.z - 0.5)),
                    hazard_clearance=1))

    def test_optional_numpy_and_python_spawn_paths_are_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "world.bsnp"
            write_spawn_snapshot(
                path, resources=((1, 60, 4, 17), (7, 60, 4, 17)),
                unsafe=((4, 61, 4, 9), (3, 62, 4, 1),
                        (4, 60, 3, 11)))
            spec = spawn_spec({
                "block": "log", "min_distance": 2,
                "max_distance": 6,
            }, hazard_clearance=1)
            accelerated = select_spawn_point(path, spec, 810)
            real_import = builtins.__import__

            def without_numpy(name, *args, **kwargs):
                if name == "numpy":
                    raise ImportError("forced minimal install")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=without_numpy):
                fallback = select_spawn_point(path, spec, 810)
            self.assertEqual(fallback, accelerated)

    def test_explicit_spawn_precedes_constraints(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "world.bsnp"
            write_spawn_snapshot(path)
            spec = spawn_spec(
                {"block": "log", "min_distance": 0,
                 "max_distance": 1},
                points=({"x": 2.5, "y": 61, "z": 2.5,
                         "yaw": 90, "pitch": -10},),
            )

            point = select_spawn_point(path, spec, 7)
            self.assertEqual((point.x, point.y, point.z),
                             (2.5, 61.0, 2.5))

    def test_constraint_spawn_rejections_distinguish_missing_and_unsafe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "world.bsnp"
            write_spawn_snapshot(path)
            missing = spawn_spec({
                "block": "log", "min_distance": 0,
                "max_distance": 8,
            })
            with self.assertRaisesRegex(
                    SnapshotRejected, "no constraint block 'log'.*bounds"):
                select_spawn_point(path, missing, 1)

            write_spawn_snapshot(
                path, resources=((1, 60, 4, 17),),
                unsafe=tuple((x, 61, z, 9)
                             for x in range(1, 8)
                             for z in range(1, 8)))
            impossible = spawn_spec({
                "block": "log", "min_distance": 0,
                "max_distance": 8,
            })
            with self.assertRaisesRegex(
                    SnapshotRejected,
                    "no safe surface spawn.*bounds.*captured radius"):
                select_spawn_point(path, impossible, 1)

    def test_constraints_scan_complete_snapshot_and_use_block_centers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "world.bsnp"
            write_snapshot(path)
            accepted = Task({
                **task_spec(directory),
                "initial_state": {
                    "spawn": {"constraints": [{
                        "block": "diamond_ore",
                        "min_distance": 3.6,
                        "max_distance": 3.7,
                    }]},
                },
            })
            # Player is (13.5, 61, -4.5); nearest diamond block center is
            # (11.5, 61.5, -1.5), whose distance is sqrt(13.25).
            _validate_snapshot_constraints(path, accepted.initial_state)
            refused = Task({
                **task_spec(directory),
                "initial_state": {
                    "spawn": {"constraints": [{
                        "block": "diamond_ore",
                        "min_distance": 4.0,
                        "max_distance": 8.0,
                    }]},
                },
            })
            with self.assertRaisesRegex(ValueError, "nearest diamond_ore"):
                _validate_snapshot_constraints(path, refused.initial_state)

    def test_radius_is_bounded_and_only_belongs_to_initial_state(self):
        self.assertEqual(Task(task_spec()).snapshot_radius, 32)
        self.assertEqual(Task(task_spec(radius=160)).snapshot_radius, 160)

        for radius in (True, 32.0, "32"):
            with self.subTest(radius=radius):
                with self.assertRaisesRegex(TypeError, "integer in 1..160"):
                    Task(task_spec(radius=radius))
        for radius in (0, 161):
            with self.subTest(radius=radius):
                with self.assertRaisesRegex(ValueError, "in 1..160"):
                    Task(task_spec(radius=radius))
        with self.assertRaisesRegex(ValueError, "needs initial_state"):
            Task({"name": "plain", "goal": "wait", "snapshot_radius": 8,
                  "success": {"type": "pitch_below", "value": 90}})

    def test_layout_uses_native_volume_cap_not_old_256_axis_cap(self):
        # A ten-chunk native capture has 320-cell horizontal axes.  This
        # compact fixture proves the layout parser accepts that axis width;
        # the engine integration test/build proves the full 320x128x320 file.
        nx, ny, nz = 320, 1, 1
        buf = bytearray(HEAD_SIZE + nx * ny * nz * 2 + 4)
        struct.pack_into("<4sI", buf, 0, b"BSNP", 1)
        struct.pack_into("<I", buf, _ITEM_COUNT_OFF, 0)
        struct.pack_into("<6i", buf, _REGION_HEADER_OFF,
                         0, 0, 0, nx, ny, nz)
        struct.pack_into("<I", buf, len(buf) - 4, 0)
        _check_layout(buf)

        struct.pack_into("<3i", buf, _REGION_HEADER_OFF + 3 * 4,
                         512, 256, 129)
        with self.assertRaisesRegex(RuntimeError, "implausible region"):
            _check_layout(buf)

    def test_path_preserves_state_key_and_separates_radius(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("bedrock_rl.env.snapshots.engine_cache_key",
                      return_value="engine"):
            small = Task(task_spec(directory, radius=16))
            large = Task(task_spec(directory, radius=64))
            small_path = snapshot_path(small, 7)
            large_path = snapshot_path(large, 7)

        self.assertNotEqual(small_path, large_path)
        self.assertIn(f"v3.s7.{small.initial_state.cache_key}.", small_path)
        self.assertIn(".r16.engine.bsnp", small_path)
        self.assertIn(".r64.engine.bsnp", large_path)

    def test_engine_key_separates_actual_and_expected_identity(self):
        fake_patch = Path("0001-test.diff")
        with patch("bedrock_rl.env.snapshots.engine.shipped_patches",
                   return_value=[fake_patch]), \
                patch("bedrock_rl.env.snapshots.engine.patch_digest",
                      return_value="expected-patch") as digest, \
                patch("bedrock_rl.env.snapshots.engine."
                      "read_patch_manifest",
                      return_value={fake_patch.name: "actual-patch"}) \
                as manifest, \
                patch("bedrock_rl.env.snapshots.engine.head_sha",
                      return_value="actual-head") as head, \
                patch("bedrock_rl.env.snapshots.engine.pinned_sha",
                      return_value="expected-head") as pinned:
            baseline = engine_cache_key("/engine")
            head.return_value = "other-actual-head"
            self.assertNotEqual(engine_cache_key("/engine"), baseline)
            head.return_value = "actual-head"
            manifest.return_value = {fake_patch.name: "other-actual-patch"}
            self.assertNotEqual(engine_cache_key("/engine"), baseline)
            manifest.return_value = {fake_patch.name: "actual-patch"}
            pinned.return_value = "other-expected-head"
            self.assertNotEqual(engine_cache_key("/engine"), baseline)
            pinned.return_value = "expected-head"
            digest.return_value = "other-expected-patch"
            self.assertNotEqual(engine_cache_key("/engine"), baseline)

    def test_build_passes_task_radius_without_starting_real_engine(self):
        radii = []

        class FakeEnv:
            def __init__(self, **kwargs):
                del kwargs
                self.obs = {"x": 0.5, "y": 64.0, "z": 0.5}

            def act(self):
                return self.obs

            def snapshot(self, path, radius):
                radii.append(radius)
                Path(path).write_bytes(b"snapshot")

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory, \
                patch("bedrock_rl.env.snapshots.MagmaEnv", FakeEnv), \
                patch("bedrock_rl.env.snapshots."
                      "_select_spawn_point_with_matches",
                      return_value=(None, None)), \
                patch("bedrock_rl.env.snapshots.patch_initial_state"), \
                patch("bedrock_rl.env.snapshots."
                      "_validate_snapshot_constraints"), \
                patch("bedrock_rl.env.snapshots.engine_cache_key",
                      return_value="engine"):
            task = Task(task_spec(directory, radius=73))
            result = build_snapshot(task, 11, netherite_home="/not-an-engine")
            self.assertTrue(Path(result).is_file())

        self.assertEqual(radii, [73])


if __name__ == "__main__":
    unittest.main()
