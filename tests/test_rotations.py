from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from transport.permutations import cosine_similarity_matrix, hungarian_match
from transport.rotations import RotationConfig, RotationPipeline
from transport.weight_ops import (
    BlockLinearGroup,
    HeadPermutation,
    NeuronPermutation,
    WeightTransformSet,
    fold_rotation_into_block,
    unfold_rotation_from_block,
)


def random_orthogonal(dim: int) -> torch.Tensor:
    base = torch.randn(dim, dim)
    q, _ = torch.linalg.qr(base)
    return q


def build_block(embed_dim: int, ffn_dim: int) -> BlockLinearGroup:
    return BlockLinearGroup(
        q_proj=nn.Linear(embed_dim, embed_dim, bias=True),
        k_proj=nn.Linear(embed_dim, embed_dim, bias=True),
        v_proj=nn.Linear(embed_dim, embed_dim, bias=True),
        o_proj=nn.Linear(embed_dim, embed_dim, bias=True),
        gate_proj=nn.Linear(embed_dim, ffn_dim, bias=True),
        up_proj=nn.Linear(embed_dim, ffn_dim, bias=True),
        down_proj=nn.Linear(ffn_dim, embed_dim, bias=True),
    )


def snapshot_block(block: BlockLinearGroup) -> dict[str, torch.Tensor]:
    params = {}
    for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        module = getattr(block, name)
        params[f"{name}.weight"] = module.weight.detach().clone()
        if module.bias is not None:
            params[f"{name}.bias"] = module.bias.detach().clone()
    return params


def test_rotation_pipeline_recovers_synthetic_mapping() -> None:
    torch.manual_seed(0)
    tokens = 256
    dim = 32
    rotation_truth = random_orthogonal(dim)
    student = torch.randn(tokens, dim)
    teacher = student @ rotation_truth

    pipeline = RotationPipeline(RotationConfig(whiten=False, target_rank=dim))
    result = pipeline.solve(student, teacher)

    transported = result.student_to_teacher(student)
    torch.testing.assert_close(transported, teacher, atol=1e-3, rtol=1e-3)

    identity = result.rotation @ result.transpose
    torch.testing.assert_close(identity, torch.eye(dim), atol=1e-4, rtol=1e-4)


def test_hungarian_finds_identity_assignment() -> None:
    torch.manual_seed(1)
    features = torch.randn(6, 12)
    sims = cosine_similarity_matrix(features, features)
    rows, cols = hungarian_match(sims, maximize=True)
    sorter = torch.argsort(rows)
    ordered_cols = cols[sorter]
    torch.testing.assert_close(ordered_cols, torch.arange(features.shape[0]))


def test_fold_and_unfold_rotation_round_trip() -> None:
    torch.manual_seed(2)
    embed_dim = 16
    ffn_dim = 32
    block = build_block(embed_dim, ffn_dim)
    rotation = random_orthogonal(embed_dim)
    before = snapshot_block(block)
    fold_rotation_into_block(block, rotation)
    unfold_rotation_from_block(block, rotation)
    after = snapshot_block(block)
    for key in before:
        torch.testing.assert_close(after[key], before[key], atol=1e-5, rtol=1e-5)


def test_weight_transform_set_with_permutations_round_trip() -> None:
    torch.manual_seed(3)
    embed_dim = 8
    head_dim = 4
    num_heads = embed_dim // head_dim
    ffn_dim = 16
    block = build_block(embed_dim, ffn_dim)
    neuron_order = torch.randperm(ffn_dim)
    transforms = WeightTransformSet(
        head_permutation=HeadPermutation(order=torch.arange(num_heads - 1, -1, -1), head_dim=head_dim),
        neuron_permutation=NeuronPermutation(order=neuron_order),
    )
    before = snapshot_block(block)
    transforms.fold(block)
    transforms.unfold(block)
    after = snapshot_block(block)
    for key in before:
        torch.testing.assert_close(after[key], before[key], atol=1e-5, rtol=1e-5)


