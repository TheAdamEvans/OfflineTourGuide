from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import torch

from .activation_cache import (
    ActivationCacheIndex,
    ActivationShardMetadata,
    ActivationShardReader,
)
from .rotations import RotationConfig, RotationPipeline


@dataclass
class RotationSummary:
    layer: str
    mean_cosine_before: float
    mean_cosine_after: float
    singular_values: List[float]
    sample_count: int
    token_count: int

    def to_json(self) -> str:
        return json.dumps(
            {
                "layer": self.layer,
                "mean_cosine_before": self.mean_cosine_before,
                "mean_cosine_after": self.mean_cosine_after,
                "singular_values": self.singular_values,
                "sample_count": self.sample_count,
                "token_count": self.token_count,
            },
            ensure_ascii=False,
        )


def load_metadata(path: str | Path) -> Sequence[ActivationShardMetadata]:
    index = ActivationCacheIndex(path)
    return index.load_all()


def gather_layer_activations(
    entries: Sequence[ActivationShardMetadata],
    layer_name: str,
) -> torch.Tensor:
    tensors: List[torch.Tensor] = []
    for entry in entries:
        payload = ActivationShardReader(entry).load()
        tensor = payload.activations.get(layer_name)
        if tensor is None:
            continue
        tensors.append(tensor.squeeze(0))
    if not tensors:
        raise ValueError(f"No activations found for layer '{layer_name}'.")
    return torch.cat(tensors, dim=0)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_norm = a / a.norm(dim=-1, keepdim=True).clamp_min(eps)
    b_norm = b / b.norm(dim=-1, keepdim=True).clamp_min(eps)
    return (a_norm * b_norm).sum(dim=-1)


def solve_layer_rotation(
    student_entries: Sequence[ActivationShardMetadata],
    teacher_entries: Sequence[ActivationShardMetadata],
    layer_name: str,
    config: RotationConfig,
) -> RotationSummary:
    student = gather_layer_activations(student_entries, layer_name)
    teacher = gather_layer_activations(teacher_entries, layer_name)
    if student.shape[0] != teacher.shape[0]:
        raise ValueError("Student and teacher activations must have the same number of tokens.")

    pipeline = RotationPipeline(config)
    result = pipeline.solve(student, teacher)
    aligned = result.student_to_teacher(student)

    before = cosine_similarity(student, teacher).mean().item()
    after = cosine_similarity(aligned, teacher).mean().item()

    return RotationSummary(
        layer=layer_name,
        mean_cosine_before=before,
        mean_cosine_after=after,
        singular_values=result.singular_values.tolist(),
        sample_count=len(student_entries),
        token_count=int(student.shape[0]),
    )


def append_summary(summary: RotationSummary, ledger_path: str | Path) -> None:
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(summary.to_json() + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solve rotations for cached activation shards.")
    parser.add_argument("--student-index", required=True, help="Path to student metadata JSONL.")
    parser.add_argument("--teacher-index", required=True, help="Path to teacher metadata JSONL.")
    parser.add_argument("--layer", action="append", required=True, help="Layer name to process.")
    parser.add_argument(
        "--ledger",
        default="runs/latest/rotations.jsonl",
        help="Output JSONL file for rotation summaries.",
    )
    parser.add_argument("--rank", type=int, default=None, help="Override PCA rank.")
    parser.add_argument("--shrinkage", type=float, default=1.0, help="Rank shrinkage factor.")
    parser.add_argument(
        "--whiten",
        dest="whiten",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable token whitening before PCA.",
    )
    parser.add_argument(
        "--force-pca",
        action="store_true",
        help="Force PCA even when dims match (skips the direct Procrustes shortcut).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    student_metadata = load_metadata(args.student_index)
    teacher_metadata = load_metadata(args.teacher_index)

    if len(student_metadata) != len(teacher_metadata):
        raise ValueError("Student/teacher metadata lists must be aligned 1:1.")

    config = RotationConfig(
        whiten=args.whiten,
        target_rank=args.rank,
        shrinkage=args.shrinkage,
        force_pca=args.force_pca,
    )

    for layer_name in args.layer:
        summary = solve_layer_rotation(student_metadata, teacher_metadata, layer_name, config)
        append_summary(summary, args.ledger)
        print(
            f"[{layer_name}] cosine before={summary.mean_cosine_before:.4f} "
            f"after={summary.mean_cosine_after:.4f} (singular values: "
            f"{', '.join(f'{sv:.3f}' for sv in summary.singular_values[:5])} ...)"
        )


if __name__ == "__main__":
    main()


