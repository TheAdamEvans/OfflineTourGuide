"""
vLLM-backed activation capture that reuses ActivationDumper hooks.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import List, Optional, Sequence

from torch import nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from model.activation_recorder import ActivationDumper
from transport.activation_cache import ActivationShardMetadata

try:
    from vllm import LLM
except ImportError as exc:  # pragma: no cover - surfaced at runtime
    raise RuntimeError(
        "The vLLM backend requires the `vllm` package. Install it via `uv pip install vllm`."
    ) from exc


def _resolve_worker_model(llm: "LLM") -> nn.Module:
    """
    Reach into the vLLM engine to grab the torch.nn.Module that runs each forward pass.
    """

    engine = getattr(llm, "llm_engine", None)
    if engine is None:
        raise RuntimeError("vLLM LLM instance is missing `llm_engine`.")

    executor = getattr(engine, "model_executor", None)
    if executor is None:
        raise RuntimeError("vLLM engine is missing `model_executor`.")

    worker = getattr(executor, "driver_worker", None)
    if worker is None:
        workers = getattr(executor, "driver_workers", None)
        if workers:
            worker = workers[0]
    if worker is None:
        raise RuntimeError("Could not locate a vLLM driver worker to attach hooks to.")

    runner = getattr(worker, "model_runner", None)
    if runner is None:
        raise RuntimeError("vLLM worker is missing `model_runner`.")

    model = getattr(runner, "model", None) or getattr(runner, "_model", None)
    if model is None:
        raise RuntimeError("Unable to resolve the torch model from vLLM's model runner.")

    return model


class VLLMActivationRecorder:
    """
    Wrapper that starts a vLLM engine and reuses ActivationDumper to write shards.
    """

    def __init__(
        self,
        model_name: str,
        *,
        dtype: str = "float16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.92,
        max_model_len: int = 8192,
        enforce_eager: bool = False,
        layer_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=enforce_eager,
        )
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        torch_model = _resolve_worker_model(self.llm)
        self._dumper = ActivationDumper(torch_model, layer_names=layer_names)

    def dump_texts(
        self,
        texts: Sequence[str],
        output_dir: str | Path,
        *,
        max_length: int = 512,
        show_progress: bool = True,
    ) -> List[ActivationShardMetadata]:
        return self._dumper.dump_texts(
            texts=texts,
            tokenizer=self.tokenizer,
            output_dir=output_dir,
            max_length=max_length,
            show_progress=show_progress,
        )

    def close(self) -> None:
        self._dumper.close()
        engine = getattr(self.llm, "llm_engine", None)
        if engine and hasattr(engine, "shutdown"):
            with contextlib.suppress(Exception):
                engine.shutdown()


__all__ = ["VLLMActivationRecorder"]

