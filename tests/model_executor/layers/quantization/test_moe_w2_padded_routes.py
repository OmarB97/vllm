# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import importlib.util
import logging
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
CUBIT_PATH = ROOT / "vllm/model_executor/layers/quantization/utils/moe_w2_cubit.py"
DELTA_PATH = ROOT / "vllm/model_executor/layers/quantization/utils/moe_w2_delta.py"
FORWARD_CONTEXT_PATH = ROOT / "vllm/forward_context.py"
RUNNER_PATH = ROOT / "vllm/v1/worker/gpu_model_runner.py"
UBATCH_PATH = ROOT / "vllm/v1/worker/gpu_ubatch_wrapper.py"


def _load_function(path: Path, name: str):
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"_BULK_PREFILL_TOKENS": 96, "np": np, "torch": torch}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def _load_delta_module():
    vllm_module = ModuleType("vllm")
    vllm_module.__path__ = [str(ROOT / "vllm")]
    logger_module = ModuleType("vllm.logger")
    logger_module.init_logger = logging.getLogger
    spec = importlib.util.spec_from_file_location("_test_moe_w2_delta_pad", DELTA_PATH)
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


class TestMoeW2PaddedRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.route_metadata = staticmethod(
            _load_function(CUBIT_PATH, "_masked_route_metadata")
        )
        cls.get_slot_mapping = staticmethod(
            _load_function(CUBIT_PATH, "_get_token_slot_mapping")
        )
        cls.get_has_prefill = staticmethod(
            _load_function(CUBIT_PATH, "_get_has_prefill")
        )
        cls.batch_has_prefill = staticmethod(
            _load_function(RUNNER_PATH, "_batch_has_prefill")
        )
        cls.delta = _load_delta_module()

    def test_runner_prefill_classification_uses_prompt_progress(self):
        self.assertTrue(
            self.batch_has_prefill(
                np.array([0], dtype=np.int32), np.array([14], dtype=np.int32)
            )
        )
        self.assertTrue(
            self.batch_has_prefill(
                np.array([20, 7], dtype=np.int32),
                np.array([20, 8], dtype=np.int32),
            )
        )
        self.assertFalse(
            self.batch_has_prefill(
                np.array([14, 20], dtype=np.int32),
                np.array([14, 8], dtype=np.int32),
            )
        )

    def test_forward_context_prefill_overrides_bulk_fallback(self):
        vllm_module = ModuleType("vllm")
        vllm_module.__path__ = [str(ROOT / "vllm")]
        context_module = ModuleType("vllm.forward_context")
        context = SimpleNamespace(has_prefill=True)
        context_module.get_forward_context = lambda: context
        with mock.patch.dict(
            sys.modules,
            {"vllm": vllm_module, "vllm.forward_context": context_module},
        ):
            self.assertTrue(self.get_has_prefill(14))
            context.has_prefill = False
            self.assertFalse(self.get_has_prefill(128))
            context.has_prefill = None
            self.assertFalse(self.get_has_prefill(14))
            self.assertTrue(self.get_has_prefill(128))

    def test_fixed_shape_route_metadata_masks_padding(self):
        sorted_ids = torch.arange(8)
        slot_mapping = torch.tensor([20, 21, 22, -1])

        token_valid, route_valid, pair_live, rows = self.route_metadata(
            sorted_ids, slot_mapping, top_k=2, mblock=1, pad_row=4
        )

        torch.testing.assert_close(token_valid, torch.tensor([True, True, True, False]))
        torch.testing.assert_close(
            route_valid,
            torch.tensor([True, True, True, True, True, True, False, False]),
        )
        torch.testing.assert_close(pair_live, route_valid)
        torch.testing.assert_close(rows, torch.tensor([0, 0, 1, 1, 2, 2, 4, 4]))

        slot_mapping.copy_(torch.tensor([20, -1, -1, -1]))
        _, route_valid, pair_live, rows = self.route_metadata(
            sorted_ids, slot_mapping, top_k=2, mblock=1, pad_row=4
        )
        torch.testing.assert_close(
            route_valid,
            torch.tensor([True, True, False, False, False, False, False, False]),
        )
        torch.testing.assert_close(pair_live, route_valid)
        torch.testing.assert_close(rows, torch.tensor([0, 0, 4, 4, 4, 4, 4, 4]))

        _, route_valid, pair_live, _ = self.route_metadata(
            sorted_ids, slot_mapping, top_k=2, mblock=4, pad_row=4
        )
        torch.testing.assert_close(
            route_valid,
            torch.tensor([True, True, False, False, False, False, False, False]),
        )
        torch.testing.assert_close(pair_live, torch.tensor([True, False]))

    def test_seen_mask_excludes_disjoint_padding_experts_in_both_tiers(self):
        ids = torch.tensor([[0, 1], [2, 3], [6, 7]])
        token_valid = torch.tensor([True, True, False])
        base_seen = torch.zeros(8, dtype=torch.uint8)
        fp4_seen = torch.zeros_like(base_seen)

        self.delta.mark_seen(base_seen, ids, token_valid)
        self.delta.mark_seen(fp4_seen, ids, token_valid)

        expected = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.uint8)
        torch.testing.assert_close(base_seen, expected)
        torch.testing.assert_close(fp4_seen, expected)

    def test_slot_mapping_shape_mismatch_fails_loud(self):
        vllm_module = ModuleType("vllm")
        vllm_module.__path__ = [str(ROOT / "vllm")]
        context_module = ModuleType("vllm.forward_context")
        context_module.get_forward_context = lambda: SimpleNamespace(
            token_slot_mapping=torch.zeros(3, dtype=torch.int64)
        )
        with (
            mock.patch.dict(
                sys.modules,
                {"vllm": vllm_module, "vllm.forward_context": context_module},
            ),
            self.assertRaisesRegex(RuntimeError, "match padded T"),
        ):
            self.get_slot_mapping(4)

    def test_static_capture_and_topology_contracts(self):
        cubit_tree = ast.parse(CUBIT_PATH.read_text())
        functions = {
            node.name: node
            for node in cubit_tree.body
            if isinstance(node, ast.FunctionDef)
        }
        for name in (
            "_desc_build_kernel",
            "_desc_build_kernel_w4s",
            "_desc_build_kernel_basecache",
            "_desc_build_kernel_base_delta",
            "_desc_build_kernel_base_delta_split",
        ):
            function = functions[name]
            self.assertIn("pair_live_ptr", [arg.arg for arg in function.args.args])
            self.assertTrue(
                any(
                    isinstance(node, ast.Name) and node.id == "pair_live_ptr"
                    for node in ast.walk(function)
                )
            )

        real_args = [arg.arg for arg in functions["_moe_w2_forward"].args.args]
        fake_args = [arg.arg for arg in functions["_moe_w2_forward_fake"].args.args]
        self.assertEqual(real_args, fake_args)
        self.assertEqual(real_args, ["x", "topk_weights", "topk_ids", "layer_key"])

        runner_source = RUNNER_PATH.read_text()
        self.assertIn("force_eager=force_w2_prefill_eager", runner_source)
        self.assertIn("and not has_prefill", runner_source)
        self.assertIn("self._get_attention_kv_cache_gid()", runner_source)
        self.assertIn("decode_context_parallel_size != 1", runner_source)
        self.assertIn("token_slot_mapping[ubatch.token_slice]", runner_source)
        self.assertIn("_w2_profile_token_slot_mapping", runner_source)
        self.assertIn(
            "profile_mapping[num_tokens_unpadded:num_tokens_padded].fill_(-1)",
            runner_source,
        )
        runner_tree = ast.parse(runner_source)
        context_calls = [
            node
            for node in ast.walk(runner_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "set_forward_context"
            and any(keyword.arg == "slot_mapping" for keyword in node.keywords)
        ]
        self.assertGreaterEqual(len(context_calls), 5)
        for call in context_calls:
            self.assertIn(
                "token_slot_mapping", {keyword.arg for keyword in call.keywords}
            )
            self.assertIn("has_prefill", {keyword.arg for keyword in call.keywords})

        context_source = FORWARD_CONTEXT_PATH.read_text()
        self.assertIn("token_slot_mapping=token_slot_mapping", context_source)
        self.assertIn("has_prefill=has_prefill", context_source)
        ubatch_source = UBATCH_PATH.read_text()
        self.assertIn("token_slot_mapping[i]", ubatch_source)
        self.assertIn("has_prefill=has_prefill", ubatch_source)

        timed = functions["_moe_w2_forward_timed"]
        align_call = next(
            node
            for node in ast.walk(timed)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "moe_align_block_size"
        )
        pad_kw = next(
            keyword
            for keyword in align_call.keywords
            if keyword.arg == "pad_sorted_ids"
        )
        self.assertIsInstance(pad_kw.value, ast.Constant)
        self.assertIs(pad_kw.value.value, True)
        alignment_guard = next(
            node
            for node in ast.walk(timed)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.BinOp)
            and isinstance(node.test.op, ast.Mod)
        )
        self.assertTrue(
            any(isinstance(node, ast.Raise) for node in ast.walk(alignment_guard))
        )


@unittest.skipUnless(
    torch.cuda.is_available() and importlib.util.find_spec("triton") is not None,
    "CUDA and Triton are required",
)
class TestMoeW2PaddedRoutesCUDA(unittest.TestCase):
    def test_graph_replay_masks_seen_misses_and_output(self):
        from vllm.model_executor.layers.quantization.utils import (
            moe_w2_cubit,
            moe_w2_delta,
        )

        device = torch.device("cuda")
        T, top_k, mblock, n_experts = 4, 2, 1, 8
        sorted_ids = torch.arange(T * top_k, device=device, dtype=torch.int64)
        expert_blocks = torch.arange(n_experts, device=device, dtype=torch.int32)
        topk_ids = torch.arange(n_experts, device=device).view(T, top_k)
        slot_mapping = torch.tensor([10, 11, 12, -1], device=device)
        slot_row = torch.full((n_experts,), -1, dtype=torch.int32, device=device)
        num_post = torch.tensor([T * top_k], dtype=torch.int32, device=device)
        seen = torch.zeros(n_experts, dtype=torch.int32, device=device)
        miss = torch.zeros(1, dtype=torch.int32, device=device)
        desc = torch.zeros((2, n_experts, 6), dtype=torch.int64, device=device)
        scratch = torch.zeros(1, dtype=torch.uint8, device=device)
        routed_output = torch.zeros(T + 1, dtype=torch.float32, device=device)
        route_values = torch.arange(1, n_experts + 1, device=device).float()

        def run_masked_routes():
            seen.zero_()
            miss.zero_()
            routed_output.zero_()
            token_valid, route_valid, pair_live, rows = (
                moe_w2_cubit._masked_route_metadata(
                    sorted_ids, slot_mapping, top_k, mblock, T
                )
            )
            moe_w2_delta.mark_seen(seen, topk_ids, token_valid)
            moe_w2_cubit._desc_build_kernel_basecache[(1,)](
                expert_blocks,
                num_post,
                pair_live,
                slot_row,
                miss,
                desc,
                scratch.data_ptr(),
                scratch.data_ptr(),
                scratch.data_ptr(),
                scratch.data_ptr(),
                scratch.data_ptr(),
                scratch.data_ptr(),
                scratch.data_ptr(),
                1,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                n_experts,
                n_experts,
                n_experts * 6,
                mblock,
                BLOCK=256,
            )
            routed_output.index_add_(
                0, rows, torch.where(route_valid, route_values, 0.0)
            )

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            run_masked_routes()
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_masked_routes()

        cases = (
            ([10, 11, 12, -1], 6, [3.0, 7.0, 11.0, 0.0]),
            ([10, -1, -1, -1], 2, [3.0, 0.0, 0.0, 0.0]),
        )
        for mapping, expected_routes, expected_output in cases:
            slot_mapping.copy_(torch.tensor(mapping, device=device))
            graph.replay()
            torch.cuda.synchronize()

            self.assertEqual(miss.item(), expected_routes)
            self.assertEqual(seen.count_nonzero().item(), expected_routes)
            promotion_candidates = ((seen > 0) & (slot_row < 0)).count_nonzero()
            self.assertEqual(promotion_candidates.item(), expected_routes)
            torch.testing.assert_close(
                routed_output[:T], torch.tensor(expected_output, device=device)
            )
            self.assertEqual(routed_output[T].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
