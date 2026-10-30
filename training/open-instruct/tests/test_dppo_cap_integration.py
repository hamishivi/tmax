"""CPU regression checks for capped DPPO integrated with padded batches."""

from types import SimpleNamespace

import pytest
import torch
from test_packed_batch_padding import production_namespace


@pytest.mark.parametrize("cap", [0.0, 10.0])
def test_dense_gradient_matches_explicit_importance_coefficient(cap):
    ns = production_namespace()
    new = torch.tensor([-2.0, -3.0, -1.0, -4.0], requires_grad=True)
    ratios = torch.tensor([0.5, 2.0, 50.0, 100.0])
    old = new.detach() - ratios.log()
    adv = torch.tensor([1.0, -1.0, 2.0, -2.0])
    config = SimpleNamespace(loss_fn=ns["GRPOLossType"].dppo, dppo_ratio_cap=cap)
    loss = ns["compute_grpo_loss"](new, ns["compute_dppo_ratio"](new - old), adv, None, config)[2]
    loss.sum().backward()
    coefficient = ratios.clamp(max=cap) if cap else ratios
    torch.testing.assert_close(new.grad, -adv * coefficient)


def test_extreme_ratios_are_finite_and_capped():
    ns = production_namespace()
    new = torch.tensor([1000.0, -1000.0], requires_grad=True)
    loss = ns["compute_dppo_policy_loss"](new, ns["compute_dppo_ratio"](new), torch.ones(2), 10.0)
    loss.sum().backward()
    assert torch.isfinite(loss).all() and torch.isfinite(new.grad).all()
    assert new.grad[0] == -10


def test_padding_does_not_enter_drift_metrics():
    ns = production_namespace()
    accumulator = ns["PolicyDriftAccumulator"]("cpu", max_staleness=4)
    accumulator.update(
        torch.tensor([[0.0, 1000.0]]), torch.zeros(1, 2), torch.tensor([[True, False]]), torch.ones(1, 2)
    )
    metrics = accumulator.compute_metrics()
    assert metrics["debug/vllm_vs_local_logprob_diff_max"] == 0.0
    assert metrics["policy_drift/effective_data_pct"] == 100.0
