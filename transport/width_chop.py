"""
Simple width-chop utility: take a large Qwen checkpoint, keep a subset of layers,
and truncate the hidden/MLP dimensions down to a smaller config (e.g., Qwen2.5-3B).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch
from torch import nn
from transformers import AutoConfig

from eval.eval import apply_layer_removals
from model.activation_recorder import load_model


@dataclass(frozen=True)
class WidthSpec:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int

    @property
    def q_proj_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_proj_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim


def _parse_int_list(payload: str) -> Sequence[int]:
    values = []
    for chunk in payload.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    return values


def _spec_from_config(config) -> WidthSpec:
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    return WidthSpec(
        hidden_size=int(config.hidden_size),
        intermediate_size=int(config.intermediate_size),
        num_attention_heads=int(config.num_attention_heads),
        num_key_value_heads=int(config.num_key_value_heads),
        head_dim=int(head_dim),
        max_position_embeddings=int(getattr(config, "max_position_embeddings", 0) or 0),
    )


def _resize_linear(linear: nn.Linear, out_dim: int, in_dim: int) -> None:
    weight = linear.weight[:out_dim, :in_dim].contiguous()
    linear.weight = nn.Parameter(weight)
    linear.out_features = out_dim
    linear.in_features = in_dim
    if linear.bias is not None:
        linear.bias = nn.Parameter(linear.bias[:out_dim].contiguous())


def _shrink_embedding(embedding: nn.Embedding, hidden_size: int) -> None:
    weight = embedding.weight[:, :hidden_size].contiguous()
    embedding.weight = nn.Parameter(weight)
    embedding.embedding_dim = hidden_size


def _shrink_norm(norm: nn.Module, hidden_size: int) -> None:
    if not hasattr(norm, "weight"):
        return
    weight = norm.weight[:hidden_size].contiguous()
    norm.weight = nn.Parameter(weight)


def _resize_decoder_layer(layer: nn.Module, spec: WidthSpec) -> None:
    attn = layer.self_attn
    mlp = layer.mlp

    _shrink_norm(layer.input_layernorm, spec.hidden_size)
    _shrink_norm(layer.post_attention_layernorm, spec.hidden_size)

    _resize_linear(attn.q_proj, spec.q_proj_dim, spec.hidden_size)
    _resize_linear(attn.k_proj, spec.kv_proj_dim, spec.hidden_size)
    _resize_linear(attn.v_proj, spec.kv_proj_dim, spec.hidden_size)
    _resize_linear(attn.o_proj, spec.hidden_size, spec.q_proj_dim)

    attn.head_dim = spec.head_dim
    attn.num_key_value_groups = spec.num_attention_heads // spec.num_key_value_heads
    attn.scaling = spec.head_dim ** -0.5

    _resize_linear(mlp.gate_proj, spec.intermediate_size, spec.hidden_size)
    _resize_linear(mlp.up_proj, spec.intermediate_size, spec.hidden_size)
    _resize_linear(mlp.down_proj, spec.hidden_size, spec.intermediate_size)


def apply_width_chop(model: nn.Module, *, target: WidthSpec) -> None:
    base = getattr(model, "model", None) or getattr(model, "transformer", None)
    if base is None or not hasattr(base, "layers"):
        raise RuntimeError("Unable to locate transformer backbone on model.")

    _shrink_embedding(base.embed_tokens, target.hidden_size)
    _shrink_norm(base.norm, target.hidden_size)

    for layer in base.layers:
        _resize_decoder_layer(layer, target)

    lm_head = getattr(model, "lm_head", None)
    if isinstance(lm_head, nn.Linear):
        _resize_linear(lm_head, lm_head.out_features, target.hidden_size)

    config = getattr(model, "config", None)
    if config:
        config.hidden_size = target.hidden_size
        config.intermediate_size = target.intermediate_size
        config.num_attention_heads = target.num_attention_heads
        config.num_key_value_heads = target.num_key_value_heads
        config.head_dim = target.head_dim
        config.max_position_embeddings = target.max_position_embeddings or config.max_position_embeddings
        config.layer_types = list(config.layer_types[: len(base.layers)])
        config.max_window_layers = len(base.layers)
        config.num_hidden_layers = len(base.layers)


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Layer+width chop a Qwen checkpoint down to a target config.")
    parser.add_argument("--source-model", default="Qwen/Qwen3-32B", help="HF repo or local path for the teacher model.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the chopped checkpoint.")
    parser.add_argument("--target-config", default="Qwen/Qwen2.5-3B", help="HF repo providing the width spec to match.")
    parser.add_argument(
        "--remove-layers",
        default=None,
        help="Comma-separated list of layer indices to drop before width pruning.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device hint passed to the model loader.")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=("float16", "bfloat16", "float32"),
        help="Precision to load the source checkpoint under.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    model, tokenizer = load_model(model_name=args.source_model, device=args.device, dtype=_resolve_dtype(args.dtype))

    base = getattr(model, "model", None) or getattr(model, "transformer", None)
    if base is None:
        raise RuntimeError("Loaded model does not expose a decoder backbone.")

    kept_layers: Optional[Sequence[int]] = None
    if args.remove_layers:
        remove_layers = _parse_int_list(args.remove_layers)
        kept_layers = apply_layer_removals(model, remove_layers)
        print(f"Removed {len(remove_layers)} layers, {len(kept_layers)} remain.")

    target_cfg = AutoConfig.from_pretrained(args.target_config, trust_remote_code=True)
    target_spec = _spec_from_config(target_cfg)
    apply_width_chop(model, target=target_spec)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if kept_layers is not None:
        (output_dir / "kept_layers.txt").write_text(",".join(map(str, kept_layers)), encoding="utf-8")

    print(f"Saved width-chopped checkpoint to {output_dir}")


if __name__ == "__main__":
    main()

