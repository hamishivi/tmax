import json
from unittest import mock

import numpy as np
import pytest

from open_instruct import data_types, model_utils, rl_utils


@pytest.mark.parametrize("keep_indices", [[], [1], [0, 1]])
def test_rollout_save_preserves_unfiltered_result(tmp_path, keep_indices):
    batch = model_utils.Batch(
        queries=[[10], [20]],
        ground_truths=[[11], [21]],
        datasets=["first", "second"],
        raw_queries=None,
        decoded_responses=None,
        indices=None,
        scores=[0.0, 1.0],
    )
    result = data_types.GenerationResult(
        responses=[[11, 12], [21]],
        finish_reasons=["length", "stop"],
        masks=[[1, 1], [1]],
        request_info=data_types.RequestInfo(
            num_calls=[1, 2],
            timeouts=[0, 0],
            tool_errors=["", ""],
            tool_outputs=["first output", "second output"],
            tool_runtimes=[0.1, 0.2],
            tool_calleds=[True, True],
        ),
        index=None,
        prompt_id=None,
        logprobs=[[-0.1, -0.2], [-0.3]],
    )
    # Hold the submitted job until after the actor masks completions. This
    # reproduces the race deterministically, without timing-dependent sleeps.
    with mock.patch.object(rl_utils, "_rollout_executor") as executor:
        rl_utils.save_rollouts_to_disk(str(tmp_path), "run", 7, batch, result, np.array([-1.0, 1.0]), 2, 0)
        executor.submit.assert_called_once()
        save, *args = executor.submit.call_args.args

    for name in ("responses", "masks", "finish_reasons", "logprobs"):
        original = getattr(result, name)
        setattr(result, name, [original[i] for i in keep_indices])
    save(*args)

    records = [json.loads(line) for line in (tmp_path / "run_rollouts_000000.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert [r["response_tokens"] for r in records] == [[11, 12], [21]]
    assert [r["finish_reason"] for r in records] == ["length", "stop"]
    assert [r["logprobs"] for r in records] == [[-0.1, -0.2], [-0.3]]
    assert [r["prompt_tokens"] for r in records] == [[10], [20]]
    assert [r["reward"] for r in records] == [0.0, 1.0]
    assert [r["advantage"] for r in records] == [-1.0, 1.0]
    assert [r["request_info"]["num_calls"] for r in records] == [1, 2]
    assert len(result.responses) == len(keep_indices)
