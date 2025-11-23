from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class RotationConfig:
    """
    Hyperparameters that control the PCA + Procrustes solve.
    """

    whiten: bool = True
    ridge: float = 1e-3
    target_rank: Optional[int] = None
    shrinkage: float = 1.0
    force_pca: bool = False

    def effective_rank(self, student_dim: int, teacher_dim: int) -> int:
        max_rank = min(student_dim, teacher_dim)
        if self.target_rank is not None:
            max_rank = min(max_rank, self.target_rank)
        shrunk = max(1, int(max_rank * self.shrinkage))
        return max(1, shrunk)


@dataclass
class RotationResult:
    """
    Output of the rotation solve. ``rotation`` maps student → teacher space.
    """

    rotation: torch.Tensor
    transpose: torch.Tensor
    singular_values: torch.Tensor
    student_components: torch.Tensor
    teacher_components: torch.Tensor

    def student_to_teacher(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor @ self.rotation

    def teacher_to_student(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor @ self.transpose


class RotationPipeline:
    """
    PCA + Procrustes rotations using cached activations.
    """

    def __init__(self, config: Optional[RotationConfig] = None) -> None:
        self.config = config or RotationConfig()

    def solve(self, student: torch.Tensor, teacher: torch.Tensor) -> RotationResult:
        if student.ndim != 2 or teacher.ndim != 2:
            raise ValueError("Expected [tokens, dim] tensors for student and teacher.")
        if student.shape[0] != teacher.shape[0]:
            raise ValueError("Student and teacher tensors must share the token dimension.")

        student_proc, student_stats = self._preprocess(student)
        teacher_proc, teacher_stats = self._preprocess(teacher)

        student_dim = student_proc.shape[1]
        teacher_dim = teacher_proc.shape[1]
        rank = self.config.effective_rank(student_dim, teacher_dim)

        if (
            not self.config.force_pca
            and student_dim == teacher_dim
            and rank == student_dim
        ):
            rotation, singular_values = solve_procrustes(student_proc, teacher_proc)
            student_basis = torch.eye(student_dim, dtype=student_proc.dtype, device=student_proc.device)
            teacher_basis = torch.eye(teacher_dim, dtype=teacher_proc.dtype, device=teacher_proc.device)
        else:
            student_basis = principal_components(student_proc, rank, ridge=self.config.ridge)
            teacher_basis = principal_components(teacher_proc, rank, ridge=self.config.ridge)
            student_proj = student_proc @ student_basis
            teacher_proj = teacher_proc @ teacher_basis
            rotation_core, singular_values = solve_procrustes(student_proj, teacher_proj)
            rotation = student_basis @ rotation_core @ teacher_basis.T

        rotation = _unwhiten_rotation(rotation, student_stats, teacher_stats)

        return RotationResult(
            rotation=rotation,
            transpose=rotation.T,
            singular_values=singular_values,
            student_components=student_basis,
            teacher_components=teacher_basis,
        )

    def _preprocess(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, "WhiteningStats"]:
        normalized = tensor.to(torch.float32)
        if self.config.whiten:
            normalized, stats = zscore_tokens_with_stats(normalized)
            return normalized, stats
        stats = WhiteningStats.identity(normalized.shape[1], normalized.device, normalized.dtype)
        return normalized, stats


def zscore_tokens(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    return (tensor - mean) / std


@dataclass
class WhiteningStats:
    mean: torch.Tensor
    std: torch.Tensor
    inv_std: torch.Tensor

    @classmethod
    def identity(cls, dim: int, device: torch.device, dtype: torch.dtype) -> "WhiteningStats":
        zeros = torch.zeros(dim, device=device, dtype=dtype)
        ones = torch.ones(dim, device=device, dtype=dtype)
        return cls(mean=zeros, std=ones, inv_std=ones)


def zscore_tokens_with_stats(
    tensor: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, WhiteningStats]:
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    inv_std = std.reciprocal()
    normalized = (tensor - mean) * inv_std
    stats = WhiteningStats(
        mean=mean.squeeze(0),
        std=std.squeeze(0),
        inv_std=inv_std.squeeze(0),
    )
    return normalized, stats


def _unwhiten_rotation(
    rotation: torch.Tensor,
    student_stats: WhiteningStats,
    teacher_stats: WhiteningStats,
) -> torch.Tensor:
    rot = rotation
    if student_stats.inv_std is not None:
        rot = rot * student_stats.inv_std.unsqueeze(1)
    if teacher_stats.std is not None:
        rot = rot * teacher_stats.std.unsqueeze(0)
    return rot


def principal_components(tensor: torch.Tensor, rank: int, ridge: float = 1e-3) -> torch.Tensor:
    cov = tensor.T @ tensor / max(1, tensor.shape[0] - 1)
    cov = cov + torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype) * ridge
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    eigvecs = eigvecs[:, order[:rank]]
    return eigvecs


def solve_procrustes(student: torch.Tensor, teacher: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if student.shape != teacher.shape:
        raise ValueError("Projected student/teacher tensors must share shape for Procrustes.")
    m = student.T @ teacher
    u, s, vt = torch.linalg.svd(m)
    return u @ vt, s


def _random_row_orthogonal(rows: int, cols: int) -> torch.Tensor:
    base = torch.randn(cols, cols, dtype=torch.float32)
    q, _ = torch.linalg.qr(base)
    return q[:rows, :].clone()


def demo_rotation(
    *,
    tokens: int = 512,
    student_dim: int = 32,
    teacher_dim: int = 48,
    noise: float = 1e-3,
    seed: int = 0,
) -> None:
    torch.manual_seed(seed)
    student = torch.randn(tokens, student_dim)
    rotation_truth = _random_row_orthogonal(student_dim, teacher_dim)
    teacher = student @ rotation_truth + noise * torch.randn(tokens, teacher_dim)

    pipeline = RotationPipeline(RotationConfig(whiten=True))
    result = pipeline.solve(student, teacher)

    transported = result.student_to_teacher(student)
    error = torch.norm(transported - teacher) / torch.norm(teacher)
    print(f"Relative alignment error: {error:.4f}")
    print(f"Singular values (top 5): {result.singular_values[:5].tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic rotation demo.")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--student-dim", type=int, default=32)
    parser.add_argument("--teacher-dim", type=int, default=48)
    parser.add_argument("--noise", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    demo_rotation(
        tokens=args.tokens,
        student_dim=args.student_dim,
        teacher_dim=args.teacher_dim,
        noise=args.noise,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()


