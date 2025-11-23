from __future__ import annotations

import math
from pathlib import Path

import torch

from eval.eval import (
    ForwardCache,
    SampleSpec,
    compute_logit_agreement,
    compute_surprisal,
)


def _make_sample(sample_id: str = "demo") -> SampleSpec:
    return SampleSpec(sample_id=sample_id, path=Path(f"{sample_id}.txt"), text="stub")


def test_compute_surprisal_perfect_prediction() -> None:
    sample = _make_sample()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    vocab = 5
    logits = torch.full((1, input_ids.shape[-1], vocab), fill_value=-5.0, dtype=torch.float32)
    logits[0, 0, 2] = 8.0  # Predict token 2 after token 1.
    logits[0, 1, 3] = 7.0  # Predict token 3 after token 2.

    cache = ForwardCache(
        sample=sample,
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits=logits,
    )
    stats = compute_surprisal(cache)

    assert stats.token_count == input_ids.shape[-1] - 1
    assert stats.top1_accuracy == 1.0
    assert stats.mean_bits < 0.01  # Near-perfect low surprisal.
    assert math.isfinite(stats.perplexity)


def test_compute_logit_agreement_tracks_basic_metrics() -> None:
    sample = _make_sample()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    vocab = 6

    base_logits = torch.randn((1, input_ids.shape[-1], vocab), dtype=torch.float32)
    var_logits = base_logits + 0.1

    hidden = torch.randn((1, input_ids.shape[-1], 4), dtype=torch.float32)

    baseline = ForwardCache(
        sample=sample,
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits=base_logits,
        hidden_state=hidden,
    )
    variant = ForwardCache(
        sample=sample,
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits=var_logits,
        hidden_state=hidden + 0.05,
    )

    metrics = compute_logit_agreement(baseline, variant)

    assert metrics["sample_id"] == sample.sample_id
    assert metrics["logit_mse"] > 0.0
    assert 0.0 < metrics["logit_cosine"] <= 1.0
    assert metrics["prob_js"] >= 0.0
    assert metrics["hidden_state_cosine"] is not None

