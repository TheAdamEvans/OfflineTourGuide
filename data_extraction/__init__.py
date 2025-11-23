"""
Thin wrapper that lazily exposes the activation dumping CLI.
"""

from __future__ import annotations

from typing import Sequence


def build_parser():
    from .dump_activations import build_parser as _build

    return _build()


def main(argv: Sequence[str] | None = None):
    from .dump_activations import main as _main

    return _main(argv)


__all__ = ["build_parser", "main"]
