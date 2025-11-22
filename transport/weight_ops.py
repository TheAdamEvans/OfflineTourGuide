from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import nn


@dataclass
class BlockLinearGroup:
    """
    Minimal view over the attention/MLP linears for a transformer block.
    """

    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    o_proj: nn.Linear
    gate_proj: nn.Linear
    up_proj: nn.Linear
    down_proj: nn.Linear

    @classmethod
    def from_block(cls, block: nn.Module) -> "BlockLinearGroup":
        needed = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        missing = [name for name in needed if not hasattr(block, name)]
        if missing:
            raise AttributeError(f"Block is missing expected linears: {', '.join(missing)}")
        return cls(
            q_proj=getattr(block, "q_proj"),
            k_proj=getattr(block, "k_proj"),
            v_proj=getattr(block, "v_proj"),
            o_proj=getattr(block, "o_proj"),
            gate_proj=getattr(block, "gate_proj"),
            up_proj=getattr(block, "up_proj"),
            down_proj=getattr(block, "down_proj"),
        )


@dataclass(frozen=True)
class HeadPermutation:
    order: torch.Tensor
    head_dim: int

    @property
    def num_heads(self) -> int:
        return int(self.order.numel())

    def inverse(self) -> "HeadPermutation":
        inv = torch.empty_like(self.order)
        inv[self.order] = torch.arange(self.order.numel())
        return HeadPermutation(order=inv, head_dim=self.head_dim)

    def apply_qkv(self, linear: nn.Linear) -> None:
        _permute_head_rows(linear, self.order, self.head_dim)

    def apply_output(self, linear: nn.Linear) -> None:
        _permute_head_columns(linear, self.order, self.head_dim)


@dataclass(frozen=True)
class NeuronPermutation:
    order: torch.Tensor

    def inverse(self) -> "NeuronPermutation":
        inv = torch.empty_like(self.order)
        inv[self.order] = torch.arange(self.order.numel())
        return NeuronPermutation(order=inv)

    def apply_up_proj(self, linear: nn.Linear) -> None:
        _permute_linear_rows(linear, self.order)

    def apply_gate_proj(self, linear: nn.Linear) -> None:
        _permute_linear_rows(linear, self.order)

    def apply_down_proj(self, linear: nn.Linear) -> None:
        _permute_linear_columns(linear, self.order)


@dataclass
class WeightTransformSet:
    rotation: Optional[torch.Tensor] = None
    head_permutation: Optional[HeadPermutation] = None
    neuron_permutation: Optional[NeuronPermutation] = None

    def fold(self, block: BlockLinearGroup) -> None:
        if self.head_permutation:
            for linear in (block.q_proj, block.k_proj, block.v_proj):
                self.head_permutation.apply_qkv(linear)
            self.head_permutation.apply_output(block.o_proj)
        if self.neuron_permutation:
            self.neuron_permutation.apply_gate_proj(block.gate_proj)
            self.neuron_permutation.apply_up_proj(block.up_proj)
            self.neuron_permutation.apply_down_proj(block.down_proj)
        if self.rotation is not None:
            fold_rotation_into_block(block, self.rotation)

    def unfold(self, block: BlockLinearGroup) -> None:
        if self.rotation is not None:
            unfold_rotation_from_block(block, self.rotation)
        if self.neuron_permutation:
            inverse = self.neuron_permutation.inverse()
            inverse.apply_gate_proj(block.gate_proj)
            inverse.apply_up_proj(block.up_proj)
            inverse.apply_down_proj(block.down_proj)
        if self.head_permutation:
            inverse = self.head_permutation.inverse()
            for linear in (block.q_proj, block.k_proj, block.v_proj):
                inverse.apply_qkv(linear)
            inverse.apply_output(block.o_proj)


def fold_rotation_into_block(block: BlockLinearGroup, rotation: torch.Tensor) -> None:
    """
    Right-multiply in-projections and left-multiply out-projections by ``rotation``.
    """
    with torch.no_grad():
        for linear in (block.q_proj, block.k_proj, block.v_proj, block.gate_proj, block.up_proj):
            linear.weight.copy_(linear.weight @ rotation)
        for linear in (block.o_proj, block.down_proj):
            linear.weight.copy_(rotation.T @ linear.weight)


def unfold_rotation_from_block(block: BlockLinearGroup, rotation: torch.Tensor) -> None:
    """
    Undo ``fold_rotation_into_block`` by applying the inverse transforms.
    """
    with torch.no_grad():
        for linear in (block.q_proj, block.k_proj, block.v_proj, block.gate_proj, block.up_proj):
            linear.weight.copy_(linear.weight @ rotation.T)
        for linear in (block.o_proj, block.down_proj):
            linear.weight.copy_(rotation @ linear.weight)


def _permute_linear_rows(linear: nn.Linear, order: torch.Tensor) -> None:
    with torch.no_grad():
        linear.weight.copy_(linear.weight[order])
        if linear.bias is not None:
            linear.bias.copy_(linear.bias[order])


def _permute_linear_columns(linear: nn.Linear, order: torch.Tensor) -> None:
    with torch.no_grad():
        linear.weight.copy_(linear.weight[:, order])


def _permute_head_rows(linear: nn.Linear, order: torch.Tensor, head_dim: int) -> None:
    out_dim, in_dim = linear.weight.shape
    if out_dim % head_dim != 0:
        raise ValueError("Output dimension must be divisible by head_dim for head permutations.")
    num_heads = out_dim // head_dim
    if num_heads != order.numel():
        raise ValueError("Permutation length does not match the number of heads.")
    weight = linear.weight.view(num_heads, head_dim, in_dim)
    permuted = weight[order]
    with torch.no_grad():
        linear.weight.copy_(permuted.view_as(linear.weight))
        if linear.bias is not None:
            bias = linear.bias.view(num_heads, head_dim)
            linear.bias.copy_(bias[order].view_as(linear.bias))


def _permute_head_columns(linear: nn.Linear, order: torch.Tensor, head_dim: int) -> None:
    out_dim, in_dim = linear.weight.shape
    if in_dim % head_dim != 0:
        raise ValueError("Input dimension must be divisible by head_dim for head permutations.")
    num_heads = in_dim // head_dim
    if num_heads != order.numel():
        raise ValueError("Permutation length does not match the number of heads.")
    weight = linear.weight.view(out_dim, num_heads, head_dim)
    permuted = weight[:, order, :]
    with torch.no_grad():
        linear.weight.copy_(permuted.view_as(linear.weight))


__all__ = [
    "BlockLinearGroup",
    "HeadPermutation",
    "NeuronPermutation",
    "WeightTransformSet",
    "fold_rotation_into_block",
    "unfold_rotation_from_block",
]


