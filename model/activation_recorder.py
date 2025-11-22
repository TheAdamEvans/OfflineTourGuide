"""
Lightweight utilities for recording transformer activations.

The original project included an end-to-end pruning and fine-tuning pipeline.
That code path has been replaced with a minimal tool whose only job is to run a
model, capture intermediate layer outputs, and persist them for quick
inspection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)


def load_model(
    model_name: str,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
) -> Tuple[nn.Module, PreTrainedTokenizerBase]:
    """
    Load a HuggingFace causal LM alongside its tokenizer.
    """
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    model.to(resolved_device)
    model.eval()
    return model, tokenizer


class ActivationDumper:
    """
    Register forward hooks on transformer blocks and persist their outputs.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.model = model.eval()
        self.device = next(self.model.parameters()).device
        self._module_lookup = dict(self.model.named_modules())
        self.layer_names = list(layer_names) if layer_names else self._infer_layer_names()
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._buffer: Dict[str, torch.Tensor] = {}

        self._register_hooks()

    def _infer_layer_names(self) -> List[str]:
        """
        Try to detect transformer block names (model.layers.N / transformer.h.N).
        """
        candidates: List[str] = []

        for name, _ in self.model.named_modules():
            parts = name.split(".")
            for idx in range(len(parts) - 1):
                current = parts[idx]
                nxt = parts[idx + 1]
                if current in {"layers", "h"} and nxt.isdigit():
                    base = ".".join(parts[: idx + 2])
                    candidates.append(base)
                    break

        if not candidates:
            # Fall back to any module that ends with "block"
            candidates = [
                name for name, module in self.model.named_modules() if name.endswith("block")
            ]

        # Deduplicate while preserving order
        seen = set()
        ordered = []
        for name in candidates:
            if name not in seen and name in self._module_lookup:
                ordered.append(name)
                seen.add(name)

        if not ordered:
            raise RuntimeError(
                "Could not infer transformer layers. Pass explicit layer names when "
                "constructing ActivationDumper."
            )

        return ordered

    def _register_hooks(self) -> None:
        for name in self.layer_names:
            module = self._module_lookup.get(name)
            if module is None:
                continue

            handle = module.register_forward_hook(self._capture(name))
            self._handles.append(handle)

    def _capture(self, layer_name: str):
        def hook(_: nn.Module, _inputs: Tuple[torch.Tensor, ...], output):
            tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(tensor, torch.Tensor):
                self._buffer[layer_name] = tensor.detach().to("cpu", torch.float32)

        return hook

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._buffer.clear()

    def dump_texts(
        self,
        texts: Sequence[str],
        tokenizer: PreTrainedTokenizerBase,
        output_dir: str | Path,
        max_length: int = 512,
        show_progress: bool = True,
    ) -> List[Path]:
        """
        Run each text through the model and save activations to disk.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: List[Path] = []

        iterable: Iterable[Tuple[int, str]]
        if show_progress and len(texts) > 1:
            try:
                from tqdm import tqdm

                iterable = enumerate(tqdm(texts, desc="Dumping activations"), start=1)
            except ImportError:
                iterable = enumerate(texts, start=1)
        else:
            iterable = enumerate(texts, start=1)

        for idx, text in iterable:
            cleaned = text.strip()
            if not cleaned:
                continue

            tokenized = tokenizer(
                cleaned,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

            self._buffer.clear()
            with torch.no_grad():
                outputs = self.model(**tokenized)

            logits = outputs.logits.detach().to("cpu", torch.float32)

            activations = {name: tensor.clone() for name, tensor in self._buffer.items()}

            sample_path = out_dir / f"sample_{idx:04d}.pt"
            attention_mask = tokenized.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.detach().to("cpu")

            payload = {
                "text": cleaned,
                "input_ids": tokenized["input_ids"].detach().to("cpu"),
                "attention_mask": attention_mask,
                "activations": activations,
                "logits": logits,
            }

            torch.save(payload, sample_path)
            saved_paths.append(sample_path)

        return saved_paths


__all__ = ["ActivationDumper", "load_model"]

