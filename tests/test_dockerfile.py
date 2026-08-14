"""Static contracts shared by Docker and the native setup script."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "Dockerfile").read_text()
SETUP = (ROOT / "bin" / "setup_deps.sh").read_text()
CI = (ROOT / ".github" / "workflows" / "ci.yml").read_text()


def _value(source: str, pattern: str) -> str:
    match = re.search(pattern, source, re.MULTILINE)
    if match is None:
        raise AssertionError(f"missing pattern {pattern!r}")
    return match.group(1)


class DockerfileContractTests(unittest.TestCase):
    def test_runtime_python_is_managed_outside_uv_cache(self):
        self.assertIn("UV_PYTHON_INSTALL_DIR=/opt/uv-python", DOCKERFILE)
        self.assertIn("/opt/bedrock-env/bin/python -V", DOCKERFILE)

    def test_engine_pin_is_required_and_both_runtime_targets_exist(self):
        self.assertRegex(DOCKERFILE, r"(?m)^ARG NETHERITE_REPO=")
        self.assertRegex(DOCKERFILE, r"(?m)^ARG NETHERITE_SHA$")
        self.assertNotRegex(DOCKERFILE, r"(?m)^ARG NETHERITE_SHA=")
        self.assertIn("FROM debian:bookworm-slim AS cpu", DOCKERFILE)
        self.assertIn("FROM cpu AS gpu", DOCKERFILE)
        self.assertIn("ARG FINAL_STAGE=cpu", DOCKERFILE)
        self.assertIn("FROM ${FINAL_STAGE} AS default", DOCKERFILE)

    def test_every_patch_directory_is_applied(self):
        self.assertIn("COPY patches/netherite /opt/patches/netherite", DOCKERFILE)
        self.assertIn("set -- /opt/patches/netherite/*.diff", DOCKERFILE)
        self.assertIn("/opt/netherite/.bedrock-rl-patches", DOCKERFILE)
        self.assertIn('sha256sum "$d"', DOCKERFILE)
        self.assertIn("COPY patches/verl /opt/patches/verl", DOCKERFILE)
        self.assertIn("set -- /opt/patches/verl/*.diff", DOCKERFILE)
        self.assertIn('git -C /opt/verl apply --reverse --check "$d"',
                      DOCKERFILE)

    def test_native_setup_discovers_the_same_engine_patch_glob(self):
        self.assertIn(
            'for NETHERITE_PATCH in "$ROOT"/patches/netherite/*.diff',
            SETUP)
        for patch in (ROOT / "patches" / "netherite").glob("*.diff"):
            self.assertEqual(SETUP.count(patch.stem), 0)

    def test_ci_patch_paths_are_anchored_outside_target_checkouts(self):
        self.assertIn(
            '"$GITHUB_WORKSPACE"/patches/netherite/*.diff', CI)
        self.assertIn('"$GITHUB_WORKSPACE"/patches/verl/*.diff', CI)
        self.assertNotIn("for d in patches/netherite/*.diff", CI)
        self.assertNotIn("for d in patches/verl/*.diff", CI)

    def test_runtime_identifies_itself_as_a_container_image(self):
        self.assertIn("BRL_CONTAINER_IMAGE=1", DOCKERFILE)

    def test_injected_container_users_can_check_provenance(self):
        self.assertIn(
            "git config --system --add safe.directory /opt/netherite",
            DOCKERFILE,
        )
        self.assertIn(
            "git config --system --add safe.directory /opt/verl",
            DOCKERFILE,
        )

    def test_gpu_versions_match_native_setup(self):
        docker_verl = _value(DOCKERFILE, r"^ARG VERL_SHA=([0-9a-f]{40})$")
        setup_verl = _value(SETUP, r"^VERL_SHA=([0-9a-f]{40})$")
        self.assertEqual(docker_verl, setup_verl)

        docker_flash = _value(
            DOCKERFILE, r"^ARG FLASH_ATTN_VERSION=([0-9.]+)$")
        setup_flash = _value(
            SETUP, r"^FLASH_ATTN_VERSION=([0-9.]+)$")
        self.assertEqual(docker_flash, setup_flash)

        docker_python = _value(DOCKERFILE, r"^ARG PYTHON_VERSION=([0-9.]+)$")
        setup_python = _value(
            SETUP, r'^VERL_PYTHON="\$\{VERL_PYTHON:-([0-9.]+)\}"$')
        self.assertEqual(docker_python, setup_python)

    def test_gpu_runtime_dependencies_have_one_pinned_manifest(self):
        requirements = (ROOT / "requirements" /
                        "verl-runtime.txt").read_text()
        lock = (ROOT / "requirements" / "verl-cu128.lock").read_text()
        self.assertIn(
            '--requirement "$ROOT/requirements/verl-cu128.lock"', SETUP)
        self.assertIn(
            "--requirement /opt/bedrock-rl/requirements/verl-cu128.lock",
            DOCKERFILE)
        for line in requirements.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                self.assertRegex(line, r"^[A-Za-z0-9_.-]+(?:\[[^]]+\])?==")
        self.assertIn("torch==2.9.0+cu128", lock)
        self.assertIn("vllm==0.12.0", lock)
        for line in lock.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                self.assertRegex(line, r"^[A-Za-z0-9_.-]+==")

    def test_gpu_runtime_can_compile_triton_kernels(self):
        gpu_stage = DOCKERFILE.split("FROM cpu AS gpu", 1)[1]
        gpu_stage = gpu_stage.split("FROM ${FINAL_STAGE} AS default", 1)[0]
        self.assertRegex(
            gpu_stage,
            r"apt-get install[^\n]*-y --no-install-recommends\s+"
            r"(?:\\\n\s*)?build-essential(?:\s|\\)",
        )

    def test_texture_pack_build_is_verified_and_provenanced(self):
        # A valueless ARG is still optional in Docker, and is accepted by
        # Modal's Dockerfile parser (which rejects an explicitly empty
        # ``ARG NAME=`` default).
        self.assertIn("ARG TEXTURE_PACK_URL\n", DOCKERFILE)
        self.assertIn("ARG TEXTURE_PACK_SHA256\n", DOCKERFILE)
        self.assertIn("TEXTURE_PACK_SHA256 is required", DOCKERFILE)
        self.assertIn("compose_texture_pack.py", DOCKERFILE)
        self.assertIn("texture_pack_sha256=", DOCKERFILE)
        self.assertIn("TEXTURE_PACK needs MC_JAR", SETUP)


if __name__ == "__main__":
    unittest.main()
