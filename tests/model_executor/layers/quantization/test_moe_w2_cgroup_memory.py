# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

ROOT = Path(__file__).resolve().parents[4]
STORE_PATH = ROOT / "vllm/model_executor/layers/quantization/utils/moe_w2_store.py"
GIB = 1 << 30
SAFETY_ENV = {
    "VLLM_MOE_W2_MIN_MEM_AVAILABLE_GB": "16",
    "VLLM_MOE_W2_MIN_CGROUP_HEADROOM_GB": "4",
}


def _load_store_module():
    """Load the store in isolation; these checks need only CPU torch."""
    vllm_module = ModuleType("vllm")
    vllm_module.__path__ = [str(ROOT / "vllm")]
    logger_module = ModuleType("vllm.logger")
    logger_module.init_logger = logging.getLogger
    spec = importlib.util.spec_from_file_location("_test_moe_w2_store", STORE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {"vllm": vllm_module, "vllm.logger": logger_module},
    ):
        spec.loader.exec_module(module)
    return module


def _cgroup_status(max_available: int, high_available: int | None) -> dict:
    return {
        "known": True,
        "version": 2,
        "path": "/test",
        "limited": True,
        "max_available": max_available,
        "high_available": high_available,
        "current": 1 * GIB,
        "events": {},
    }


class TestMoeW2CgroupMemory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = _load_store_module()

    def _v2_status(self, current: int, high: str, maximum: str) -> dict:
        with tempfile.TemporaryDirectory() as root:
            files = {
                "memory.current": str(current),
                "memory.high": high,
                "memory.max": maximum,
                "memory.stat": "anon 1\nfile 2\nfile_mapped 3\n",
                "memory.events": "high 0\nmax 0\noom 0\n",
                "memory.swap.current": "0",
                "memory.swap.max": "max",
            }
            for name, value in files.items():
                Path(root, name).write_text(value)
            with mock.patch.object(
                self.store, "_active_cgroup_v2_dirs", return_value=[root]
            ):
                return self.store._cgroup_memory_status()

    def test_crossed_soft_high_is_separate_from_hard_max_headroom(self):
        status = self._v2_status(11 * GIB, str(10 * GIB), str(20 * GIB))

        self.assertTrue(status["limited"])
        self.assertEqual(status["max_available"], 9 * GIB)
        self.assertEqual(status["high_available"], -1 * GIB)

    def test_crossed_soft_high_does_not_refuse_safe_hard_headroom(self):
        with (
            mock.patch.dict(os.environ, SAFETY_ENV, clear=False),
            mock.patch.object(
                self.store, "_mem_available_bytes", return_value=64 * GIB
            ),
            mock.patch.object(
                self.store,
                "_cgroup_memory_status",
                return_value=_cgroup_status(8 * GIB, -1 * GIB),
            ),
        ):
            report = self.store._memory_preflight("soft-high", 2 * GIB)

        self.assertEqual(report["cgroup_max_available"], 8 * GIB)
        self.assertEqual(report["cgroup_high_available"], -1 * GIB)

    def test_hard_max_headroom_still_refuses_unsafe_transient(self):
        with (
            mock.patch.dict(os.environ, SAFETY_ENV, clear=False),
            mock.patch.object(
                self.store, "_mem_available_bytes", return_value=64 * GIB
            ),
            mock.patch.object(
                self.store,
                "_cgroup_memory_status",
                return_value=_cgroup_status(6 * GIB, 20 * GIB),
            ),
            self.assertRaisesRegex(RuntimeError, "memory.max headroom"),
        ):
            self.store._memory_preflight("hard-max", 3 * GIB)

    def test_finite_soft_high_does_not_make_unlimited_max_limited(self):
        status = self._v2_status(5 * GIB, str(10 * GIB), "max")

        self.assertFalse(status["limited"])
        self.assertIsNone(status["max_available"])
        self.assertEqual(status["high_available"], 5 * GIB)


if __name__ == "__main__":
    unittest.main()
