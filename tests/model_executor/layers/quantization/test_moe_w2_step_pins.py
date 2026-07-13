# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import contextlib
import importlib.util
import logging
import sys
import threading
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[4]
DELTA_PATH = ROOT / "vllm/model_executor/layers/quantization/utils/moe_w2_delta.py"
GATE_PATH = ROOT / "vllm/model_executor/layers/quantization/utils/moe_w2_gate.py"
RUNNER_PATH = ROOT / "vllm/v1/worker/gpu_model_runner.py"
WORKER_PATH = ROOT / "vllm/v1/worker/gpu_worker.py"
SPEC_CONFIG_PATH = ROOT / "vllm/config/speculative.py"


def _load_delta_module():
    """Load the tier in isolation; this regression needs only CPU torch."""
    vllm_module = ModuleType("vllm")
    vllm_module.__path__ = [str(ROOT / "vllm")]
    logger_module = ModuleType("vllm.logger")
    logger_module.init_logger = logging.getLogger
    spec = importlib.util.spec_from_file_location("_test_moe_w2_delta", DELTA_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with (
        mock.patch.dict(
            sys.modules,
            {"vllm": vllm_module, "vllm.logger": logger_module},
        ),
        mock.patch.dict("os.environ", {"VLLM_MOE_W2_DELTA_TRACE": "0"}, clear=False),
    ):
        spec.loader.exec_module(module)
    return module


class TestMoeW2StepPins(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.delta = _load_delta_module()

    def _saturated_lru_tier(self):
        tier = object.__new__(self.delta.DeltaTier)
        tier._lock = threading.Lock()
        tier.n_slots = 3
        tier._free = []
        tier._alloc_owner(tier.n_slots)
        tier._owner_li[:] = torch.tensor([0, 0, 0])
        tier._owner_ei[:] = torch.tensor([0, 1, 2])
        tier._owner_tick[:] = torch.tensor([0, 1, 2])
        tier._policy = "lru"
        tier._tick = 10
        tier._step_pins = {0, 1, 2}
        tier._coupled_fp4 = None
        tier._seen_host = torch.zeros((1, 32), dtype=torch.uint8)
        tier._freq = torch.zeros((1, 32), dtype=torch.float32)
        tier._need = torch.zeros((1, 32), dtype=torch.float32)
        tier.slot_table = torch.full((1, 32), -1, dtype=torch.int32)
        tier._mirror = torch.full((1, 32), -1, dtype=torch.int32)
        for slot, (_, expert, _) in enumerate(tier._owner):
            tier.slot_table[0, expert] = slot
            tier._mirror[0, expert] = slot
        tier._n_evicted = 0
        tier._win_evicted = 0
        return tier

    def test_step_scope_reset_keeps_saturated_lru_evicting(self):
        tier = self._saturated_lru_tier()

        # This is the production failure mode before the runner reset: once
        # every saturated slot has accumulated in _step_pins, even emergency
        # promotion cannot find a victim.
        self.assertEqual(tier._take_slots_batch(1, emergency=True), [])

        for expert in range(3, 19):
            tier._tick += 3
            tier.step_begin()
            expected = min(range(tier.n_slots), key=lambda slot: tier._owner[slot][2])
            self.assertEqual(tier._take_slots_batch(1, emergency=True), [expected])

            # Mirror force_promote's ownership handoff and current-step pin.
            tier._own(expected, 0, expert)
            tier.slot_table[0, expert] = expected
            tier._mirror[0, expert] = expected
            tier._step_pins.add(expected)
            self.assertEqual(tier._free, [])

        self.assertEqual(tier._n_evicted, 16)

    def test_explicit_seen_snapshot_wins_over_shared_snapshot_overwrite(self):
        tier = self._saturated_lru_tier()
        tier._step_pins.clear()
        # Simulate another caller overwriting the shared host buffer after this
        # caller captured expert 0. The immutable set must still protect slot 0.
        tier._seen_host.zero_()
        tier._seen_host[0, 2] = 1
        self.assertEqual(tier._take_slot({(0, 0)}), 1)

    def test_emergency_third_pass_drops_seen_but_keeps_step_pins(self):
        seen_set = {(0, 0), (0, 1), (0, 2)}
        tier = self._saturated_lru_tier()
        tier._step_pins.clear()

        # A verify step can mark the whole saturated pool. Once that step has
        # drained, its immutable seen snapshot is recency information, not an
        # in-flight reader, so the emergency restore must still find a victim.
        self.assertEqual(
            tier._take_slots_batch(1, emergency=True, seen_set=seen_set),
            [0],
        )

        # Current-step pins remain a correctness exclusion in every pass.
        tier = self._saturated_lru_tier()
        self.assertEqual(
            tier._take_slots_batch(1, emergency=True, seen_set=seen_set),
            [],
        )

    def test_step_end_clears_seen_but_preserves_pins(self):
        tier = object.__new__(self.delta.DeltaTier)
        tier.dev = torch.device("cpu")
        tier._snap_lock = threading.Lock()
        tier._stream = mock.Mock()
        tier.seen = torch.ones((2, 4), dtype=torch.uint8)
        tier._step_pins = {1, 2}
        main = object()

        with (
            mock.patch.object(
                self.delta.torch.cuda, "current_stream", return_value=main
            ),
            mock.patch.object(
                self.delta.torch.cuda, "stream", return_value=contextlib.nullcontext()
            ),
        ):
            tier.step_end()

        self.assertEqual(tier.seen.count_nonzero().item(), 0)
        self.assertEqual(tier._step_pins, {1, 2})
        tier._stream.wait_stream.assert_called_once_with(main)

    def test_module_boundaries_reset_both_tiers(self):
        base = mock.Mock()
        fp4 = mock.Mock()
        events = []
        base.step_end.side_effect = lambda: events.append("base.end")
        fp4.step_end.side_effect = lambda: events.append("fp4.end")
        base.wake.side_effect = lambda: events.append("base.wake")
        fp4.wake.side_effect = lambda: events.append("fp4.wake")
        with (
            mock.patch.object(self.delta, "_BASE_TIER", base),
            mock.patch.object(self.delta, "_TIER", fp4),
        ):
            self.delta.begin_target_step()
            self.delta.begin_replay_step()
            self.delta.finish_forward_step()

        base.pause_for_forward.assert_called_once_with()
        fp4.pause_for_forward.assert_called_once_with()
        base.routing_step_begin.assert_called_once_with()
        fp4.routing_step_begin.assert_called_once_with()
        base.step_begin.assert_called_once_with()
        fp4.step_begin.assert_called_once_with()
        base.step_end.assert_called_once_with()
        fp4.step_end.assert_called_once_with()
        base.wake.assert_called_once_with()
        fp4.wake.assert_called_once_with()
        self.assertEqual(
            events,
            ["base.end", "fp4.end", "base.wake", "fp4.wake"],
        )

    def test_manager_pass_cannot_overlap_forward_window(self):
        tier = object.__new__(self.delta.DeltaTier)
        tier._forward_lock = threading.Lock()
        tier._forward_paused = False
        tier._wake = mock.Mock()
        tier._wake_driven = False

        tier.pause_for_forward()
        entered = threading.Event()

        def manager_pass():
            with tier._forward_lock:
                entered.set()

        thread = threading.Thread(target=manager_pass)
        thread.start()
        self.assertFalse(entered.wait(0.05))
        tier.wake()
        self.assertTrue(entered.wait(1.0))
        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())
        self.assertFalse(tier._forward_paused)
        self.assertTrue(tier._wake_driven)
        tier._wake.set.assert_called_once_with()

    def test_stop_joins_idle_manager_and_releases_forward_pause(self):
        tier = object.__new__(self.delta.DeltaTier)
        tier._stop = False
        tier._wake = threading.Event()
        tier._forward_lock = threading.Lock()
        tier._forward_lock.acquire()
        tier._forward_paused = True
        tier._thread = threading.Thread(target=tier._wake.wait, daemon=True)
        tier._thread.start()

        tier.stop()

        self.assertTrue(tier._stop)
        self.assertFalse(tier._forward_paused)
        self.assertIsNone(tier._thread)
        self.assertTrue(tier._forward_lock.acquire(blocking=False))
        tier._forward_lock.release()

    def test_shutdown_all_stops_each_unique_tier_once(self):
        tier = mock.Mock()
        with (
            mock.patch.object(self.delta, "_BASE_TIER", tier),
            mock.patch.object(self.delta, "_TIER", tier),
        ):
            self.delta.shutdown_all()

        tier.stop.assert_called_once_with()

    def test_ensure_resident_drains_then_uses_exact_layer_snapshot(self):
        tree = ast.parse(DELTA_PATH.read_text())
        ensure = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "ensure_resident"
        )
        snap_scope = next(
            node
            for node in ast.walk(ensure)
            if isinstance(node, ast.With)
            and any(
                isinstance(item.context_expr, ast.Attribute)
                and item.context_expr.attr == "_snap_lock"
                for item in node.items
            )
        )
        lock_scope = next(
            node
            for node in ast.walk(snap_scope)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "synchronize"
        )
        pool_lock = next(
            node
            for node in ast.walk(ensure)
            if isinstance(node, ast.With)
            and any(
                isinstance(item.context_expr, ast.Attribute)
                and item.context_expr.attr == "_lock"
                for item in node.items
            )
        )
        self.assertLess(lock_scope.lineno, pool_lock.lineno)
        layer_snapshot = next(
            node
            for node in ast.walk(ensure)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "layer_seen_set"
                for target in node.targets
            )
        )
        self.assertIsInstance(layer_snapshot.value, ast.SetComp)
        take = next(
            node
            for node in ast.walk(pool_lock)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_take_slots_batch"
        )
        seen_kw = next(
            keyword for keyword in take.keywords if keyword.arg == "seen_set"
        )
        self.assertIsInstance(seen_kw.value, ast.Name)
        self.assertEqual(seen_kw.value.id, "layer_seen_set")
        pin_clears = [
            node
            for node in ast.walk(pool_lock)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "clear"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "_step_pins"
        ]
        pin_mutations = [
            node
            for node in ast.walk(pool_lock)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"add", "update"}
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "_step_pins"
        ]
        self.assertEqual(len(pin_clears), 1)
        self.assertGreaterEqual(len(pin_mutations), 2)
        self.assertLess(pin_clears[0].lineno, take.lineno)

    def test_manager_and_force_promote_pass_immutable_seen_sets(self):
        tree = ast.parse(DELTA_PATH.read_text())
        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        tick_take = next(
            node
            for node in ast.walk(functions["_tick_once"])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_take_slot"
        )
        self.assertEqual(len(tick_take.args), 1)
        self.assertIsInstance(tick_take.args[0], ast.Name)
        self.assertEqual(tick_take.args[0].id, "seen_set")

        force_take = next(
            node
            for node in ast.walk(functions["force_promote"])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_take_slots_batch"
        )
        seen_kw = next(
            keyword for keyword in force_take.keywords if keyword.arg == "seen_set"
        )
        self.assertIsInstance(seen_kw.value, ast.Name)
        self.assertEqual(seen_kw.value.id, "seen_set")

        wrapper_take = next(
            node
            for node in ast.walk(functions["_take_slot"])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_take_slots_batch"
        )
        wrapper_kw = next(
            keyword for keyword in wrapper_take.keywords if keyword.arg == "seen_set"
        )
        self.assertIsInstance(wrapper_kw.value, ast.Name)
        self.assertEqual(wrapper_kw.value.id, "seen_set")

        for name in ("_tick_once", "force_promote"):
            assignments = [
                node
                for node in ast.walk(functions[name])
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "seen_set"
                    for target in node.targets
                )
            ]
            self.assertEqual(len(assignments), 1, name)

    def test_force_promote_only_repins_when_replay_follows(self):
        for pin, expected_pins in ((True, {0}), (False, set())):
            with self.subTest(pin=pin):
                tier = object.__new__(self.delta.DeltaTier)
                tier.dev = torch.device("cpu")
                tier._store = {0: object()}
                tier.n_layers = 1
                tier._store_mask_cache = None
                tier._store_mask_n = -1
                tier._snap_lock = threading.Lock()
                tier._lock = threading.Lock()
                tier._stream = mock.Mock()
                tier.seen = torch.zeros((1, 4), dtype=torch.uint8)
                tier.seen[0, 2] = 1
                tier._seen_host = torch.zeros_like(tier.seen)
                tier._mirror = torch.full((1, 4), -1, dtype=torch.int32)
                tier._mirror[0, 2] = 0
                tier._alloc_owner(1)
                tier._owner_li[0] = 0
                tier._owner_ei[0] = 2
                tier._owner_tick[0] = 3
                tier._tick = 10
                tier._step_pins = {0}
                tier._need = torch.zeros((1, 4), dtype=torch.float32)
                main = object()
                event = mock.Mock()

                tier.step_begin()
                self.assertEqual(tier._step_pins, set())
                with (
                    mock.patch.object(
                        self.delta.torch.cuda, "current_stream", return_value=main
                    ),
                    mock.patch.object(
                        self.delta.torch.cuda,
                        "stream",
                        return_value=contextlib.nullcontext(),
                    ),
                    mock.patch.object(
                        self.delta.torch.cuda, "Event", return_value=event
                    ),
                ):
                    self.assertEqual(tier.force_promote(pin=pin), 0)

                self.assertEqual(tier._step_pins, expected_pins)
                self.assertEqual(tier._owner[0], (0, 2, tier._tick))
                tier._stream.wait_stream.assert_called_once_with(main)
                event.synchronize.assert_called_once_with()

    def test_final_no_replay_fetch_does_not_pin_new_slot(self):
        class Store(dict):
            def rows_for(self, pairs):
                self.requested = pairs
                return [torch.tensor([7.0])]

        tier = object.__new__(self.delta.DeltaTier)
        tier.dev = torch.device("cpu")
        tier._store = Store({0: object()})
        tier.n_layers = 1
        tier._store_mask_cache = None
        tier._store_mask_n = -1
        tier._snap_lock = threading.Lock()
        tier._lock = threading.Lock()
        tier._stream = mock.Mock()
        tier.seen = torch.zeros((1, 4), dtype=torch.uint8)
        tier.seen[0, 2] = 1
        tier._seen_host = torch.zeros_like(tier.seen)
        tier._mirror = torch.full((1, 4), -1, dtype=torch.int32)
        tier.slot_table = torch.full((1, 4), -1, dtype=torch.int32)
        tier.pool = torch.zeros((1, 1), dtype=torch.float32)
        tier.n_slots = 1
        tier._free = [0]
        tier._alloc_owner(1)
        tier._tick = 10
        tier._step_pins = set()
        tier._coupled_fp4 = None
        tier._need = torch.zeros((1, 4), dtype=torch.float32)
        tier._freq = torch.zeros((1, 4), dtype=torch.float32)
        tier._n_promoted = 0
        tier._win_promoted = 0
        tier._kpi_deferred = 0
        tier._kpi_unfixed = 0
        main = object()
        event = mock.Mock()

        with (
            mock.patch.object(
                self.delta.torch.cuda, "current_stream", return_value=main
            ),
            mock.patch.object(
                self.delta.torch.cuda,
                "stream",
                return_value=contextlib.nullcontext(),
            ),
            mock.patch.object(self.delta.torch.cuda, "Event", return_value=event),
        ):
            self.assertEqual(tier.force_promote(pin=False), 1)

        self.assertEqual(tier._store.requested, [(0, 2)])
        self.assertEqual(tier._step_pins, set())
        self.assertEqual(tier._mirror[0, 2].item(), 0)
        self.assertEqual(tier.pool[0, 0].item(), 7.0)

    def test_runner_pins_force_promote_only_when_fixed_point_continues(self):
        tree = ast.parse(RUNNER_PATH.read_text())
        execute_model = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "execute_model"
        )
        promotions = [
            node
            for node in ast.walk(execute_model)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "force_promote"
        ]
        self.assertEqual(len(promotions), 4)

        pin_args = []
        for promotion in promotions:
            pin_kw = next(
                keyword for keyword in promotion.keywords if keyword.arg == "pin"
            )
            self.assertIsInstance(pin_kw.value, ast.Call)
            self.assertIsInstance(pin_kw.value.func, ast.Attribute)
            self.assertIsInstance(pin_kw.value.func.value, ast.Name)
            self.assertEqual(pin_kw.value.func.value.id, "_w2d")
            self.assertEqual(pin_kw.value.func.attr, "fp_continue")
            pin_args.append(pin_kw.value.args[0])

        self.assertEqual(
            sum(isinstance(arg, ast.Constant) and arg.value == 0 for arg in pin_args),
            2,
        )
        self.assertEqual(
            sum(isinstance(arg, ast.Name) and arg.id == "_replays" for arg in pin_args),
            2,
        )

    def test_spec_guard_covers_ngram_and_native_mtp_k1_without_method_branch(self):
        runner_tree = ast.parse(RUNNER_PATH.read_text())
        take_drafts = next(
            node
            for node in ast.walk(runner_tree)
            if isinstance(node, ast.FunctionDef) and node.name == "take_draft_token_ids"
        )
        spec_guard = next(
            node
            for node in ast.walk(take_drafts)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "spec_suppressed"
        )
        schedule_drafts = next(
            node
            for node in ast.walk(take_drafts)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_get_draft_token_ids_cpu"
        )
        self.assertLess(spec_guard.lineno, schedule_drafts.lineno)
        self.assertFalse(
            any(
                isinstance(node, ast.Attribute) and node.attr == "method"
                for node in ast.walk(take_drafts)
            )
        )

        config_tree = ast.parse(SPEC_CONFIG_PATH.read_text())
        aliases = {
            node.targets[0].id: node
            for node in ast.walk(config_tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        }
        mtp_values = {
            node.value
            for node in ast.walk(aliases["MTPModelTypes"].value)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        spec_values = {
            node.value
            for node in ast.walk(aliases["SpeculativeMethod"].value)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        spec_names = {
            node.id
            for node in ast.walk(aliases["SpeculativeMethod"].value)
            if isinstance(node, ast.Name)
        }
        eagle_names = {
            node.id
            for node in ast.walk(aliases["EagleModelTypes"].value)
            if isinstance(node, ast.Name)
        }
        self.assertIn("mtp", mtp_values)
        self.assertIn("ngram", spec_values)
        self.assertIn("EagleModelTypes", spec_names)
        self.assertIn("MTPModelTypes", eagle_names)

        spec_config = next(
            node
            for node in ast.walk(config_tree)
            if isinstance(node, ast.ClassDef) and node.name == "SpeculativeConfig"
        )
        token_count = next(
            node
            for node in spec_config.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "num_speculative_tokens"
        )
        self.assertIsInstance(token_count.value, ast.Call)
        gt = next(
            keyword for keyword in token_count.value.keywords if keyword.arg == "gt"
        )
        self.assertIsInstance(gt.value, ast.Constant)
        self.assertEqual(gt.value.value, 0)

    def test_runner_uses_routing_then_post_forward_pin_boundaries(self):
        tree = ast.parse(RUNNER_PATH.read_text())
        execute_model = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "execute_model"
        )
        target_begins = [
            node
            for node in ast.walk(execute_model)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "begin_target_step"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "_w2d"
        ]
        replay_begins = [
            node
            for node in ast.walk(execute_model)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "begin_replay_step"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "_w2d"
        ]
        self.assertEqual(len(target_begins), 1)
        self.assertEqual(len(replay_begins), 1)

        swallowing_handlers = [
            node
            for node in ast.walk(execute_model)
            if isinstance(node, ast.Try)
            and (
                target_begins[0] in ast.walk(node) or replay_begins[0] in ast.walk(node)
            )
        ]
        self.assertEqual(swallowing_handlers, [])

        target_forwards = [
            node
            for node in ast.walk(execute_model)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_model_forward"
        ]
        first_forward = min(node.lineno for node in target_forwards)
        post_forward_reset = replay_begins[0]
        self.assertLess(target_begins[0].lineno, first_forward)
        self.assertLess(first_forward, post_forward_reset.lineno)

        direct_tier_resets = [
            node
            for node in ast.walk(execute_model)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "step_begin"
        ]
        self.assertEqual(direct_tier_resets, [])

    def test_worker_releases_manager_only_after_gate_barrier(self):
        tree = ast.parse(WORKER_PATH.read_text())
        execute_model = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "execute_model"
        )
        finishes = [
            node
            for node in ast.walk(execute_model)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_finish_w2_manager_step"
        ]
        barriers = [
            node
            for node in ast.walk(execute_model)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_gate_pp_barrier"
        ]
        self.assertEqual(len(finishes), 2)
        self.assertEqual(len(barriers), 2)
        finally_finishes = [
            node
            for node in ast.walk(execute_model)
            if isinstance(node, ast.Try)
            and any(finish in ast.walk(node) for finish in finishes)
            and any(
                finish in ast.walk(final)
                for final in node.finalbody
                for finish in finishes
            )
        ]
        self.assertEqual(len(finally_finishes), 2)

        gate_tree = ast.parse(GATE_PATH.read_text())
        should_reforward = next(
            node
            for node in ast.walk(gate_tree)
            if isinstance(node, ast.FunctionDef) and node.name == "should_reforward"
        )
        unsafe_wakes = [
            node
            for node in ast.walk(should_reforward)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "wake_all"
        ]
        self.assertEqual(unsafe_wakes, [])


if __name__ == "__main__":
    unittest.main()
