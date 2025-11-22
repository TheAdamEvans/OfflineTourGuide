from __future__ import annotations

from typing import Optional, Tuple

import torch


def cosine_similarity_matrix(
    student: torch.Tensor,
    teacher: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    student_norm = student / student.norm(dim=-1, keepdim=True).clamp_min(eps)
    teacher_norm = teacher / teacher.norm(dim=-1, keepdim=True).clamp_min(eps)
    return student_norm @ teacher_norm.T


def apply_permutation(tensor: torch.Tensor, permutation: torch.Tensor, dim: int = 0) -> torch.Tensor:
    indices = permutation.to(torch.long)
    return tensor.index_select(dim, indices)


def match_attention_heads(student_stats: torch.Tensor, teacher_stats: torch.Tensor) -> torch.Tensor:
    sims = cosine_similarity_matrix(student_stats, teacher_stats)
    rows, cols = hungarian_match(sims, maximize=True)
    permutation = torch.full((student_stats.shape[0],), -1, dtype=torch.long)
    permutation[rows] = cols
    return permutation


def match_neurons(student_stats: torch.Tensor, teacher_stats: torch.Tensor) -> torch.Tensor:
    sims = cosine_similarity_matrix(student_stats, teacher_stats)
    rows, cols = hungarian_match(sims, maximize=True)
    permutation = torch.full((student_stats.shape[0],), -1, dtype=torch.long)
    permutation[rows] = cols
    return permutation


def hungarian_match(matrix: torch.Tensor, maximize: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    solver = _HungarianSolver(matrix, maximize=maximize)
    return solver.solve()


class _HungarianSolver:
    def __init__(self, matrix: torch.Tensor, *, maximize: bool) -> None:
        cost = matrix.clone().to(torch.float64)
        if cost.ndim != 2:
            raise ValueError("Hungarian solver expects a 2D matrix.")
        if maximize:
            cost = cost.max() - cost

        self.orig_rows, self.orig_cols = cost.shape
        self.size = max(self.orig_rows, self.orig_cols)
        padded = torch.zeros((self.size, self.size), dtype=torch.float64)
        padded[: self.orig_rows, : self.orig_cols] = cost
        self.cost = padded
        self.mask = torch.zeros_like(self.cost, dtype=torch.int8)
        self.row_cover = torch.zeros(self.size, dtype=torch.bool)
        self.col_cover = torch.zeros(self.size, dtype=torch.bool)

    def solve(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self._reduce()
        self._star_initial_zeros()

        while True:
            covered_cols = self._cover_starred_columns()
            if covered_cols == self.size:
                break
            row, col = self._prime_uncovered_zero()
            while row is None:
                self._adjust_matrix()
                row, col = self._prime_uncovered_zero()
            if self._has_star_in_row(row):
                self.row_cover[row] = True
                star_col = self._starred_col_in_row(row)
                if star_col is not None:
                    self.col_cover[star_col] = False
            else:
                self._augment_path((row, col))
                self.row_cover.fill_(False)
                self.col_cover.fill_(False)
                self.mask[self.mask == 2] = 0

        return self._extract_assignment()

    def _reduce(self) -> None:
        self.cost -= self.cost.min(dim=1, keepdim=True).values
        self.cost -= self.cost.min(dim=0, keepdim=True).values

    def _star_initial_zeros(self) -> None:
        for row in range(self.size):
            for col in range(self.size):
                if self.cost[row, col] == 0 and not self.row_cover[row] and not self.col_cover[col]:
                    self.mask[row, col] = 1
                    self.row_cover[row] = True
                    self.col_cover[col] = True
        self.row_cover.fill_(False)
        self.col_cover.fill_(False)

    def _cover_starred_columns(self) -> int:
        for col in range(self.size):
            if torch.any(self.mask[:, col] == 1):
                self.col_cover[col] = True
        return int(self.col_cover.sum().item())

    def _prime_uncovered_zero(self) -> Tuple[Optional[int], Optional[int]]:
        while True:
            zero = self._find_uncovered_zero()
            if zero is None:
                return None, None
            row, col = zero
            self.mask[row, col] = 2
            star_col = self._starred_col_in_row(row)
            if star_col is not None:
                self.row_cover[row] = True
                self.col_cover[star_col] = False
            else:
                return row, col

    def _find_uncovered_zero(self) -> Optional[Tuple[int, int]]:
        for row in range(self.size):
            if self.row_cover[row]:
                continue
            for col in range(self.size):
                if self.col_cover[col]:
                    continue
                if self.cost[row, col] == 0:
                    return row, col
        return None

    def _has_star_in_row(self, row: int) -> bool:
        return torch.any(self.mask[row] == 1).item()

    def _starred_col_in_row(self, row: int) -> Optional[int]:
        cols = torch.where(self.mask[row] == 1)[0]
        if cols.numel() == 0:
            return None
        return int(cols[0].item())

    def _augment_path(self, zero: Tuple[int, int]) -> None:
        path = [zero]
        done = False
        while not done:
            star_row = self._starred_row_in_col(path[-1][1])
            if star_row is not None:
                path.append((star_row, path[-1][1]))
            else:
                done = True
                break

            prime_col = self._primed_col_in_row(path[-1][0])
            path.append((path[-1][0], prime_col))

        for row, col in path:
            if self.mask[row, col] == 1:
                self.mask[row, col] = 0
            elif self.mask[row, col] == 2:
                self.mask[row, col] = 1

    def _starred_row_in_col(self, col: int) -> Optional[int]:
        rows = torch.where(self.mask[:, col] == 1)[0]
        if rows.numel() == 0:
            return None
        return int(rows[0].item())

    def _primed_col_in_row(self, row: int) -> int:
        cols = torch.where(self.mask[row] == 2)[0]
        if cols.numel() == 0:
            raise RuntimeError("Expected a primed zero in the row while augmenting the path.")
        return int(cols[0].item())

    def _adjust_matrix(self) -> None:
        uncovered = self.cost[~self.row_cover][:, ~self.col_cover]
        if uncovered.numel() == 0:
            return
        min_val = uncovered.min()
        self.cost[~self.row_cover] -= min_val
        self.cost[:, self.col_cover] += min_val

    def _extract_assignment(self) -> Tuple[torch.Tensor, torch.Tensor]:
        rows: list[int] = []
        cols: list[int] = []
        for row in range(self.orig_rows):
            starred = torch.where(self.mask[row] == 1)[0]
            for col in starred:
                if col < self.orig_cols:
                    rows.append(row)
                    cols.append(int(col.item()))
        return torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long)


__all__ = [
    "apply_permutation",
    "cosine_similarity_matrix",
    "hungarian_match",
    "match_attention_heads",
    "match_neurons",
]


