"""
Tiny helper utilities to make recorded activations easier to inspect.

Instead of complex pruning heuristics we only compute a couple of visibility
metrics so you can eyeball which layers were more active.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping

import torch


def summarize_activation_file(path: str | Path) -> Dict[str, Dict[str, float]]:
    """
    Load a ``torch.save`` activation dump and compute simple stats per layer.
    """
    payload = torch.load(Path(path), map_location="cpu")
    activations: Mapping[str, torch.Tensor] = payload.get("activations", {})

    summary: Dict[str, Dict[str, float]] = {}
    for name, tensor in activations.items():
        flat = tensor.to(torch.float32).reshape(-1)
        if flat.numel() == 0:
            continue
        summary[name] = {
            "mean_abs": flat.abs().mean().item(),
            "max_abs": flat.abs().max().item(),
        }

    return summary


def format_summary_table(
    stats: Mapping[str, Mapping[str, float]],
    top_k: int | None = None,
) -> str:
    """
    Turn ``summarize_activation_file`` output into a quick text table.
    """
    if not stats:
        return "No activation data found."

    items = sorted(stats.items(), key=lambda item: item[0])
    if top_k:
        items = items[:top_k]

    name_width = max(len(name) for name, _ in items)
    lines = [f"{'Layer'.ljust(name_width)}  Mean|Abs  Max|Abs"]
    lines.append("-" * len(lines[0]))

    for name, metrics in items:
        mean_abs = metrics.get("mean_abs", 0.0)
        max_abs = metrics.get("max_abs", 0.0)
        lines.append(
            f"{name.ljust(name_width)}  {mean_abs:8.4f}  {max_abs:8.4f}",
        )

    return "\n".join(lines)


__all__ = ["summarize_activation_file", "format_summary_table"]

