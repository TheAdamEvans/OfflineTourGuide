"""
Command-line helper for dumping model activations against a few texts.

Despite the ``data_extraction`` package name this now serves a single purpose:
load a model, run a handful of sample prompts, and persist intermediate layer
outputs for quick inspection.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Sequence, Tuple

from model.activation_analyzer import format_summary_table, summarize_activation_file
from model.activation_recorder import ActivationDumper, load_model

WORKSPACE_HF_HOME = Path("/workspace/.hf_home")

DEFAULT_SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def _load_texts(
    samples_dir: Path,
    inline_texts: Sequence[str] | None,
    text_files: Sequence[str] | None,
) -> List[str]:
    texts: List[str] = []

    if inline_texts:
        for snippet in inline_texts:
            cleaned = snippet.strip()
            if cleaned:
                texts.append(cleaned)

    if text_files:
        for entry in text_files:
            file_path = Path(entry)
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
        action="append",
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
        choices=["float16", "bfloat16", "float32", "float8_e4m3fn", "float8_e5m2"],
        help="Model weight dtype.",
    )
    parser.add_argument(
        "--backend",
        default="torch",
        choices=["torch", "vllm"],
        help="Execution backend (PyTorch module vs. vLLM engine).",
    )
    parser.add_argument(
        "--hf-home",
        default=None,
        help="Override HF_HOME (default: /workspace/.hf_home).",
    )
    parser.add_argument(
        "--hf-hub-cache",
        default=None,
        help="Override HF_HUB_CACHE (default: <hf-home>/hub).",
    )
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel degree for the vLLM backend.",
    )
    parser.add_argument(
        "--vllm-gpu-mem-utilization",
        type=float,
        default=0.92,
        help="Fractional GPU memory budget vLLM is allowed to use.",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=8192,
        help="Max context length to request from vLLM.",
    )
    parser.add_argument(
        "--vllm-enforce-eager",
        action="store_true",
        help="Forward to vLLM to disable CUDA graph capture (fallback for older drivers).",
    )
    return parser


def _resolve_dtype(name: str) -> "torch.dtype":
    import torch

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float8_e4m3fn": getattr(torch, "float8_e4m3fn", None),
        "float8_e5m2": getattr(torch, "float8_e5m2", None),
    }
    dtype = mapping.get(name)
    if dtype is None:
        raise ValueError(
            f"Requested dtype '{name}' is unavailable in this torch build. "
            "Upgrade torch or pick a supported precision."
        )
    return dtype


def _ensure_hf_cache(
    hf_home: str | None,
    hf_hub_cache: str | None,
) -> Tuple[Path, Path]:
    """
    Guarantee HF cache env vars point at persistent workspace storage.
    """

    home = Path(hf_home or os.environ.get("HF_HOME") or WORKSPACE_HF_HOME)
    cache = Path(
        hf_hub_cache or os.environ.get("HF_HUB_CACHE") or home / "hub"
    )
    home.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(home)
    os.environ["HF_HUB_CACHE"] = str(cache)
    return home, cache


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    samples_dir = Path(args.samples_dir)
    _ensure_hf_cache(args.hf_home, args.hf_hub_cache)

    texts = _load_texts(
        samples_dir,
        inline_texts=args.text,
        text_files=args.text_file,
    )

    dtype = _resolve_dtype(args.dtype)

    layer_names = _parse_layer_names(args.layers)

    if args.backend == "vllm":
        if args.device not in {None, "auto"}:
            raise ValueError("--device is not supported when --backend vllm is selected.")
        if args.dtype.startswith("float8"):
            raise ValueError("The vLLM backend currently expects float16/bfloat16/float32 weights.")
        from model.vllm_activation_recorder import VLLMActivationRecorder

        recorder = VLLMActivationRecorder(
            model_name=args.model,
            dtype=args.dtype,
            layer_names=layer_names,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=args.vllm_gpu_mem_utilization,
            max_model_len=args.vllm_max_model_len,
            enforce_eager=args.vllm_enforce_eager,
        )
        try:
            saved_shards = recorder.dump_texts(
                texts=texts,
                output_dir=args.output_dir,
                max_length=args.max_length,
            )
        finally:
            recorder.close()
    else:
        model, tokenizer = load_model(
            model_name=args.model,
            device=args.device,
            dtype=dtype,
        )
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

