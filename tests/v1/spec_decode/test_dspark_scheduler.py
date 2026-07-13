# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for the DSpark hardware-aware scheduling policy."""

import pytest
import torch

from vllm.v1.spec_decode.dynamic.utils import DSparkLiveRederivation
from vllm.v1.worker.gpu.spec_decode.dspark.scheduler import (
    allocate_widths,
    derive_dynamic_sd_table,
    schedule_uniform_length,
)

pytestmark = pytest.mark.cpu_test


def test_uniform_scheduler_falls_back_without_profile() -> None:
    assert schedule_uniform_length([0.0, 0.8, 1.2], 4, 2, None, None) == (2, 0.0)


def test_uniform_scheduler_uses_ceiling_request_bucket() -> None:
    # At R=5, dispatch pads to the R=8 graph. Interpolation would incorrectly
    # smooth over the deliberately expensive width-2 graph at that bucket.
    length, predicted = schedule_uniform_length(
        accept=[0.0, 0.9, 1.5],
        num_reqs=5,
        gamma=2,
        r_grid=[4, 8],
        times_by_l=[
            [0.010, 0.012],
            [0.012, 0.014],
            [0.013, 0.050],
        ],
    )
    assert length == 1
    assert predicted == pytest.approx(0.014)


def test_uniform_scheduler_hysteresis_keeps_close_incumbent() -> None:
    length, _ = schedule_uniform_length(
        accept=[0.0, 0.8, 1.3],
        num_reqs=8,
        gamma=2,
        r_grid=[8],
        times_by_l=[[0.010], [0.014], [0.017]],
        current=1,
        hysteresis=0.10,
    )
    assert length == 1


def test_allocate_widths_supports_threshold_and_global_budget() -> None:
    survival = torch.tensor([[0.9, 0.7, 0.4], [0.8, 0.3, 0.1]])
    calibration = torch.ones(3)

    thresholded = allocate_widths(
        survival, calibration, num_reqs=2, length=3, tau=0.5, budget_frac=1.0
    )
    assert thresholded.tolist() == [2, 1]

    budgeted = allocate_widths(
        survival, calibration, num_reqs=2, length=3, tau=0.0, budget_frac=0.25
    )
    assert budgeted.tolist() == [1, 1]


def test_derived_table_covers_full_runtime_range() -> None:
    table = derive_dynamic_sd_table(
        r_grid=[4, 8],
        times_by_l=[[0.010, 0.014], [0.012, 0.017], [0.020, 0.030]],
        gamma=2,
        max_num_reqs=16,
    )
    assert table[0][0] == 1
    assert table[-1][1] == 16
    assert all(0 <= width <= 2 for _, _, width in table)


def test_live_rederivation_returns_dense_bounded_lookup() -> None:
    policy = DSparkLiveRederivation(
        r_grid=[2, 4],
        times_by_l=[[0.010, 0.012], [0.012, 0.015], [0.020, 0.026]],
        max_num_seqs=8,
        num_spec_tokens=2,
    )
    policy.REDERIVE_DRAFTS = 4
    policy.MIN_POSITION_OBS = 1.0

    lookup = None
    for accepted in (2, 1, 0, 2):
        lookup = policy.observe(num_draft_tokens=2, num_accepted=accepted)

    assert lookup is not None
    assert len(lookup) == 9
    assert lookup[0] == 0
    assert all(0 <= width <= 2 for width in lookup[1:])
