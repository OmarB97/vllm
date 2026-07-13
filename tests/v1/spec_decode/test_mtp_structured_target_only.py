# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

from tests.v1.core.test_scheduler import create_requests, create_scheduler
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput


def test_structured_output_requests_use_target_only_decoding():
    scheduler = create_scheduler(num_speculative_tokens=1)
    (request,) = create_requests(num_requests=1, num_tokens=1)
    request.structured_output_request = Mock(grammar=Mock())
    scheduler.add_request(request)

    output = scheduler.schedule()
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[0]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )

    scheduler.update_draft_token_ids(
        DraftTokenIds([request.request_id], [[30]])
    )
    assert request.spec_token_ids == []

    request.spec_token_ids = [30]
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[request.request_id] == 1
    assert request.request_id not in output.scheduled_spec_decode_tokens

    output.scheduled_spec_decode_tokens[request.request_id] = [30]
    scheduler.update_draft_token_ids_in_output(
        DraftTokenIds([request.request_id], [[30]]), output
    )
    assert output.scheduled_spec_decode_tokens[request.request_id] == [-1]
