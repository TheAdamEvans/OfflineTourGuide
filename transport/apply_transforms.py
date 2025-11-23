"""
Fold previously solved rotations and optional permutations into a checkpoint.

Workflow:
1. Solve rotations (and optionally permutations) offline and serialize them to disk
   using ``torch.save``. Each tensor file should contain either a bare tensor or a
   dictionary with a ``"tensor"`` key pointing at the tensor.
2. Create a JSON manifest that lists the transformer block name plus the tensor
   paths to fold.
3. Run this module as a CLI to load the checkpoint, apply the transforms, and
   save the updated weights.

Example manifest (``runs/run_x/transforms.json``):
{
  "layers": [
    {
      "name": "model.layers.0",
      "rotation_path": "runs/run_x/rotations/model.layers.0.pt"
    },
    {
      "name": "model.layers.1",
      "rotation_path": "runs/run_x/rotations/model.layers.1.pt",
      "head_permutation_path": "runs/run_x/permutations/heads_1.pt",
      "neuron_permutation_path": "runs/run_x/permutations/neurons_1.pt",
      "head_dim": 128
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch

from model.activation_recorder import load_model
from transport.weight_ops import (
    BlockLinearGroup,
    HeadPermutation,
    NeuronPermutation,
    WeightTransformSet,
)


@dataclass(slots=True)
class LayerTransform:
    """
    Tensor bundle describing how to modify a single transformer block.
    """

    name: str
    rotation_path: Optional[Path] = None
    head_permutation_path: Optional[Path] = None
    neuron_permutation_path: Optional[Path] = None
    head_dim: Optional[int] = None


def _load_manifest(path: Path) -> List[LayerTransform]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries: Sequence[dict] = payload.get("layers", payload)
    if not isinstance(entries, Sequence):
        raise ValueError("Transform manifest must be a list or contain a 'layers' list.")

    transforms: List[LayerTransform] = []
    for item in entries:
        if "name" not in item:
            raise ValueError("Each transform entry needs a 'name' key.")
        transform = LayerTransform(
            name=item["name"],
            rotation_path=_resolve_optional_path(path, item.get("rotation_path")),
            head_permutation_path=_resolve_optional_path(path, item.get("head_permutation_path")),
            neuron_permutation_path=_resolve_optional_path(path, item.get("neuron_permutation_path")),
            head_dim=item.get("head_dim"),
        )
        if transform.head_permutation_path and transform.head_dim is None:
            raise ValueError(
                f"Layer '{transform.name}' supplies a head permutation without specifying head_dim."
            )
        transforms.append(transform)
    return transforms


def _resolve_optional_path(manifest: Path, value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (manifest.parent / candidate).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Transform tensor not found at '{candidate}'.")
    return candidate


def _load_tensor(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj.clone().to(torch.float32)
    if isinstance(obj, dict):
        if "tensor" in obj:
            tensor = obj["tensor"]
            if isinstance(tensor, torch.Tensor):
                return tensor.clone().to(torch.float32)
        if "rotation" in obj and isinstance(obj["rotation"], torch.Tensor):
            return obj["rotation"].clone().to(torch.float32)
    raise ValueError(f"Unsupported tensor payload stored in '{path}'.")


def _resolve_module(root: torch.nn.Module, dotted: str) -> torch.nn.Module:
    module = root
    for chunk in dotted.split("."):
        if not hasattr(module, chunk):
            raise AttributeError(f"Module '{module.__class__.__name__}' has no attribute '{chunk}'.")
        module = getattr(module, chunk)
    return module


def _build_transform(spec: LayerTransform) -> WeightTransformSet:
    rotation = _load_tensor(spec.rotation_path) if spec.rotation_path else None

    head_perm = None
    if spec.head_permutation_path:
        order = _load_tensor(spec.head_permutation_path).to(torch.long)
        head_perm = HeadPermutation(order=order, head_dim=int(spec.head_dim))

    neuron_perm = None
    if spec.neuron_permutation_path:
        order = _load_tensor(spec.neuron_permutation_path).to(torch.long)
        neuron_perm = NeuronPermutation(order=order)

    return WeightTransformSet(rotation=rotation, head_permutation=head_perm, neuron_permutation=neuron_perm)


def apply_transforms(
    model: torch.nn.Module,
    transforms: Sequence[LayerTransform],
) -> None:
    """
    Apply the requested rotations/permutations in-place.
    """

    for spec in transforms:
        block = _resolve_module(model, spec.name)
        linear_group = BlockLinearGroup.from_block(block)
        transform_set = _build_transform(spec)
        transform_set.fold(linear_group)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fold rotation/permutation tensors into a checkpoint.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="HuggingFace model id or path to a local checkpoint directory.",
    )
    parser.add_argument("--output-dir", required=True, help="Destination directory for the updated weights.")
    parser.add_argument("--spec", required=True, help="Path to the JSON manifest listing transforms per layer.")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to instantiate the model on (default: auto-detect).",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model load precision. Rotations are folded in float32 regardless.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    manifest_path = Path(args.spec)
    layer_specs = _load_manifest(manifest_path)
    if not layer_specs:
        raise ValueError("No layer transforms were provided.")

    model, tokenizer = load_model(model_name=args.checkpoint, device=args.device, dtype=_resolve_dtype(args.dtype))

    apply_transforms(model, layer_specs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved transformed checkpoint to {output_dir}")


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


if __name__ == "__main__":
    main()


