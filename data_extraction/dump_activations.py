"""
Command-line helper for dumping model activations against a few texts.

Despite the ``data_extraction`` package name this now serves a single purpose:
load a model, run a handful of sample prompts, and persist intermediate layer
outputs for quick inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

from model.activation_analyzer import format_summary_table, summarize_activation_file
from model.activation_recorder import ActivationDumper, load_model

DEFAULT_SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def _load_texts(
    samples_dir: Path,
    inline_texts: Sequence[str] | None,
    text_file: str | None,
) -> List[str]:
    texts: List[str] = []

    if inline_texts:
        for snippet in inline_texts:
            cleaned = snippet.strip()
            if cleaned:
                texts.append(cleaned)

    if text_file:
        file_path = Path(text_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Text file '{file_path}' does not exist.")
        for line in file_path.read_text(encoding="utf-8").splitlines():
            cleaned = line.strip()
            if cleaned:
                texts.append(cleaned)

    if not texts:
        for sample in sorted(samples_dir.glob("*.txt")):
            content = sample.read_text(encoding="utf-8").strip()
            if content:
                texts.append(content)

    if not texts:
        raise ValueError(
            "No texts were provided. Supply --text, --text-file, or add .txt files to "
            f"{samples_dir}."
        )

    return texts


def _parse_layer_names(raw: Sequence[str] | None) -> List[str] | None:
    if not raw:
        return None
    names: List[str] = []
    for value in raw:
        for chunk in value.split(","):
            trimmed = chunk.strip()
            if trimmed:
                names.append(trimmed)
    return names or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dump transformer activations to disk.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="HuggingFace model id.")
    parser.add_argument("--device", default=None, help="Torch device (default: auto-detect).")
    parser.add_argument("--output-dir", default="activations", help="Directory for .pt dumps.")
    parser.add_argument("--max-length", type=int, default=512, help="Token limit per sample.")
    parser.add_argument(
        "--text",
        default=None,
        help="Inline text to process (can be provided multiple times).",
        action="append",
    )
    parser.add_argument(
        "--text-file",
        default=None,
        help="Path to a newline-delimited text file.",
    )
    parser.add_argument(
        "--samples-dir",
        default=str(DEFAULT_SAMPLES),
        help="Directory of fallback .txt samples.",
    )
    parser.add_argument(
        "--layer",
        dest="layers",
        action="append",
        default=None,
        help="Explicit module name to hook. Repeat for multiple layers.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Print a simple summary table after dumping.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model weight dtype.",
    )
    return parser


def _resolve_dtype(name: str) -> "torch.dtype":
    import torch

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    samples_dir = Path(args.samples_dir)
    texts = _load_texts(
        samples_dir,
        inline_texts=args.text,
        text_file=args.text_file,
    )

    model, tokenizer = load_model(
        model_name=args.model,
        device=args.device,
        dtype=_resolve_dtype(args.dtype),
    )

    layer_names = _parse_layer_names(args.layers)
    dumper = ActivationDumper(model, layer_names=layer_names)
    try:
        saved_shards = dumper.dump_texts(
            texts=texts,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            max_length=args.max_length,
        )
    finally:
        dumper.close()

    if args.analyze:
        for shard in saved_shards:
            stats = summarize_activation_file(shard.path)
            print(f"\n{shard.path}:")
            print(format_summary_table(stats, top_k=None))


if __name__ == "__main__":
    main()

