from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


@dataclass(frozen=True)
class SampleSpec:
    """Container for a single eval sample."""

    sample_id: str
    path: Path
    text: str


@dataclass
class ForwardCache:
    """Lightweight tensor cache for a single forward pass."""

    sample: SampleSpec
    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    logits: torch.Tensor
    hidden_state: Optional[torch.Tensor] = None


@dataclass
class SurprisalStats:
    """Token-level surprisal diagnostics."""

    sample_id: str
    token_count: int
    mean_nats: float
    mean_bits: float
    median_bits: float
    p95_bits: float
    max_bits: float
    perplexity: float
    top1_accuracy: float

    def to_dict(self) -> Dict[str, float]:
        payload = asdict(self)
        return payload


def load_samples(samples_dir: str | Path, limit: Optional[int] = None) -> List[SampleSpec]:
    """
    Load evaluation texts. Each *.txt file is treated as a single sample.
    """

    directory = Path(samples_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Samples directory '{directory}' does not exist.")

    samples: List[SampleSpec] = []
    for file_path in sorted(directory.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        samples.append(SampleSpec(sample_id=file_path.stem, path=file_path, text=text))
        if limit and len(samples) >= limit:
            break

    if not samples:
        raise ValueError(f"No .txt files with content found under '{directory}'.")
    return samples


def _resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg and device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return mapping[dtype_name]
    except KeyError as exc:  # pragma: no cover - argparse guards this
        raise ValueError(f"Unsupported dtype '{dtype_name}'.") from exc


def _prepare_tokenizer(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _parse_int_list(payload: str) -> List[int]:
    values: List[int] = []
    for chunk in payload.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError("No valid integer indices were provided.")
    return sorted(values)


def load_model_and_tokenizer(
    model_name: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = _prepare_tokenizer(tokenizer)
    load_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    device_map: Optional[str] = None
    if device.type == "cuda":
        device_map = "auto"
        load_kwargs["device_map"] = device_map
        load_kwargs["low_cpu_mem_usage"] = True

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **load_kwargs,
    )
    if device_map is None:
        model.to(device)
    model.eval()
    return model, tokenizer


def _ensure_attention_mask(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    mask = batch.get("attention_mask")
    if mask is None:
        mask = torch.ones_like(batch["input_ids"])
        batch["attention_mask"] = mask
    return mask


def run_forward_pass(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sample: SampleSpec,
    *,
    max_length: int,
    collect_hidden: bool,
) -> ForwardCache:
    encoded = tokenizer(
        sample.text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    _ensure_attention_mask(encoded)
    device = next(model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.inference_mode():
        outputs = model(
            **encoded,
            use_cache=False,
            output_hidden_states=collect_hidden,
        )

    logits = outputs.logits.detach().to(torch.float32).cpu()
    hidden_state = None
    if collect_hidden and outputs.hidden_states:
        hidden_state = outputs.hidden_states[-1].detach().to(torch.float32).cpu()

    return ForwardCache(
        sample=sample,
        input_ids=encoded["input_ids"].detach().cpu(),
        attention_mask=encoded.get("attention_mask", None).detach().cpu()
        if encoded.get("attention_mask") is not None
        else None,
        logits=logits,
        hidden_state=hidden_state,
    )


def _mask_tokens(mask: Optional[torch.Tensor], seq_len: int) -> torch.Tensor:
    if mask is None:
        return torch.ones((1, seq_len), dtype=torch.float32)
    trimmed = mask[:, 1:]
    return trimmed.to(torch.float32)


def compute_surprisal(cache: ForwardCache) -> SurprisalStats:
    logits = cache.logits[:, :-1, :]
    targets = cache.input_ids[:, 1:]
    mask = _mask_tokens(cache.attention_mask, targets.shape[-1])

    log_probs = F.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    surprisal_nats = (-target_log_probs) * mask

    valid = mask.sum().clamp_min(1.0)
    mean_nats = (surprisal_nats.sum() / valid).item()
    surprisal_bits = surprisal_nats / math.log(2)

    flat_bits = surprisal_bits.masked_select(mask.to(torch.bool))
    if flat_bits.numel() == 0:
        flat_bits = surprisal_bits.reshape(-1)

    median_bits = torch.median(flat_bits).item()
    p95_bits = torch.quantile(flat_bits, 0.95).item()
    max_bits = torch.max(flat_bits).item()

    predictions = logits.argmax(dim=-1)
    accuracy = (
        (predictions == targets).to(torch.float32) * mask
    ).sum() / valid

    return SurprisalStats(
        sample_id=cache.sample.sample_id,
        token_count=int(valid.item()),
        mean_nats=mean_nats,
        mean_bits=mean_nats / math.log(2),
        median_bits=median_bits,
        p95_bits=p95_bits,
        max_bits=max_bits,
        perplexity=math.exp(mean_nats),
        top1_accuracy=float(accuracy.item()),
    )


def aggregate_surprisal(stats: Sequence[SurprisalStats]) -> Dict[str, float]:
    total_tokens = sum(entry.token_count for entry in stats)
    if total_tokens == 0:
        return {}
    mean_nats = sum(entry.mean_nats * entry.token_count for entry in stats) / total_tokens
    mean_bits = mean_nats / math.log(2)
    perplexity = math.exp(mean_nats)
    accuracy = sum(entry.top1_accuracy * entry.token_count for entry in stats) / total_tokens
    return {
        "token_count": total_tokens,
        "mean_bits": mean_bits,
        "mean_nats": mean_nats,
        "perplexity": perplexity,
        "top1_accuracy": accuracy,
    }


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def compute_logit_agreement(
    reference: ForwardCache,
    candidate: ForwardCache,
) -> Dict[str, float]:
    if reference.sample.sample_id != candidate.sample.sample_id:
        raise ValueError("Sample mismatch when comparing logits.")

    ref_logits = reference.logits[:, :-1, :]
    cand_logits = candidate.logits[:, :-1, :]
    ref_mask = _mask_tokens(reference.attention_mask, ref_logits.shape[1])

    delta = (ref_logits - cand_logits).pow(2)
    mse = _weighted_mean(delta.mean(dim=-1), ref_mask).item()
    mae = _weighted_mean((ref_logits - cand_logits).abs().mean(dim=-1), ref_mask).item()

    logit_cos = F.cosine_similarity(ref_logits, cand_logits, dim=-1)
    cos_mean = _weighted_mean(logit_cos, ref_mask).item()

    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
    cand_log_probs = F.log_softmax(cand_logits, dim=-1)
    ref_probs = ref_log_probs.exp()
    cand_probs = cand_log_probs.exp()

    kl = _weighted_mean(
        (ref_probs * (ref_log_probs - cand_log_probs)).sum(dim=-1),
        ref_mask,
    ).item()
    reverse_kl = _weighted_mean(
        (cand_probs * (cand_log_probs - ref_log_probs)).sum(dim=-1),
        ref_mask,
    ).item()
    js = 0.5 * (kl + reverse_kl)

    topk_match = (
        (ref_logits.argmax(dim=-1) == cand_logits.argmax(dim=-1)).to(torch.float32) * ref_mask
    )
    topk_score = (topk_match.sum() / ref_mask.sum().clamp_min(1.0)).item()

    hidden_cos = None
    if reference.hidden_state is not None and candidate.hidden_state is not None:
        ref_hidden = reference.hidden_state[:, :-1, :]
        cand_hidden = candidate.hidden_state[:, :-1, :]
        hidden_cos = _weighted_mean(
            F.cosine_similarity(ref_hidden, cand_hidden, dim=-1),
            ref_mask,
        ).item()

    return {
        "sample_id": reference.sample.sample_id,
        "logit_mse": mse,
        "logit_mae": mae,
        "logit_cosine": cos_mean,
        "prob_kl": kl,
        "prob_reverse_kl": reverse_kl,
        "prob_js": js,
        "top1_match": topk_score,
        "hidden_state_cosine": hidden_cos,
    }


def apply_layer_stride_chop(
    model: PreTrainedModel,
    *,
    stride: int,
    offset: int = 0,
) -> List[int]:
    """
    Drop transformer blocks so that only layers matching (idx % stride) == offset remain.
    """

    if stride <= 1:
        raise ValueError("Stride must be greater than 1 when chopping layers.")

    base = getattr(model, "model", None) or getattr(model, "transformer", None)
    if base is None or not hasattr(base, "layers"):
        raise RuntimeError("Unable to locate transformer layers on the provided model.")

    layers: nn.ModuleList = base.layers  # type: ignore[assignment]
    keep_indices = [idx for idx in range(len(layers)) if idx % stride == offset]
    if not keep_indices:
        raise RuntimeError("Layer chop removed every block; adjust stride/offset.")

    base.layers = nn.ModuleList([layers[idx] for idx in keep_indices])  # type: ignore[assignment]
    if hasattr(base, "config"):
        base.config.num_hidden_layers = len(base.layers)
    if hasattr(model, "config"):
        model.config.num_hidden_layers = len(base.layers)
    return keep_indices


def apply_layer_removals(
    model: PreTrainedModel,
    remove_layers: Sequence[int],
) -> List[int]:
    """
    Remove the specified transformer blocks and return the indices that remain.
    """

    base = getattr(model, "model", None) or getattr(model, "transformer", None)
    if base is None or not hasattr(base, "layers"):
        raise RuntimeError("Unable to locate transformer layers on the provided model.")

    layers: nn.ModuleList = base.layers  # type: ignore[assignment]
    layer_count = len(layers)
    remove_set = {int(idx) for idx in remove_layers}
    for idx in remove_set:
        if idx < 0 or idx >= layer_count:
            raise ValueError(f"Layer index {idx} is out of range (0..{layer_count-1}).")
    keep_indices = [idx for idx in range(layer_count) if idx not in remove_set]
    if not keep_indices:
        raise RuntimeError("Layer removal eliminated the entire stack.")

    base.layers = nn.ModuleList([layers[idx] for idx in keep_indices])  # type: ignore[assignment]
    if hasattr(base, "config"):
        base.config.num_hidden_layers = len(base.layers)
    if hasattr(model, "config"):
        model.config.num_hidden_layers = len(base.layers)
    return keep_indices


def run_phase3_eval(args: argparse.Namespace) -> Dict[str, object]:
    if args.chop_every and args.remove_layers:
        raise ValueError("Choose either --chop-every or --remove-layers, not both.")

    samples = load_samples(args.samples_dir, limit=args.limit)
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype)
    model, tokenizer = load_model_and_tokenizer(args.model, dtype=dtype, device=device)

    baseline_forward: Dict[str, ForwardCache] = {}
    baseline_stats: List[SurprisalStats] = []

    for sample in samples:
        cache = run_forward_pass(
            model,
            tokenizer,
            sample,
            max_length=args.max_length,
            collect_hidden=not args.no_hidden,
        )
        stats = compute_surprisal(cache)
        baseline_forward[sample.sample_id] = cache
        baseline_stats.append(stats)

    report: Dict[str, object] = {
        "model": args.model,
        "samples": [sample.sample_id for sample in samples],
        "baseline": {
            "per_sample": [entry.to_dict() for entry in baseline_stats],
            "aggregate": aggregate_surprisal(baseline_stats),
        },
    }

    variant_config: Optional[Dict[str, object]] = None
    if args.chop_every:
        kept_layers = apply_layer_stride_chop(
            model,
            stride=args.chop_every,
            offset=args.chop_offset,
        )
        variant_config = {
            "mode": "stride",
            "stride": args.chop_every,
            "offset": args.chop_offset,
            "kept_layers": kept_layers,
        }
    elif args.remove_layers:
        remove_layers = _parse_int_list(args.remove_layers)
        kept_layers = apply_layer_removals(model, remove_layers)
        variant_config = {
            "mode": "remove_layers",
            "removed_layers": remove_layers,
            "kept_layers": kept_layers,
        }

    if variant_config:
        variant_stats: List[SurprisalStats] = []
        agreements: List[Dict[str, float]] = []

        for sample in samples:
            cache = run_forward_pass(
                model,
                tokenizer,
                sample,
                max_length=args.max_length,
                collect_hidden=not args.no_hidden,
            )
            stats = compute_surprisal(cache)
            variant_stats.append(stats)
            agreement = compute_logit_agreement(baseline_forward[sample.sample_id], cache)
            agreements.append(agreement)

        report["chopped"] = {
            **variant_config,
            "per_sample": [entry.to_dict() for entry in variant_stats],
            "aggregate": aggregate_surprisal(variant_stats),
            "logit_agreement": agreements,
        }

    if args.save_json:
        destination = Path(args.save_json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 3 eval harness (surprisal + logit agreement).")
    parser.add_argument(
        "--samples-dir",
        default=str(Path(__file__).resolve().parent / "samples"),
        help="Directory containing eval sample *.txt files.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-32B", help="Teacher model checkpoint.")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("float16", "bfloat16", "float32"),
        help="Model weight dtype.",
    )
    parser.add_argument("--device", default="auto", help="Execution device (cuda, cpu, mps, auto).")
    parser.add_argument("--max-length", type=int, default=1024, help="Max tokens per sample.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of samples.")
    parser.add_argument(
        "--no-hidden",
        action="store_true",
        help="Skip caching final hidden states (disables hidden cosine metric).",
    )
    parser.add_argument(
        "--chop-every",
        type=int,
        default=None,
        help="Keep only every Nth transformer block (set to 2 for every-other layer).",
    )
    parser.add_argument(
        "--chop-offset",
        type=int,
        default=0,
        help="Layer offset when chopping (0 keeps layers 0,N,2N...).",
    )
    parser.add_argument(
        "--remove-layers",
        type=str,
        default=None,
        help="Comma-separated list of transformer block indices to remove for the variant run.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional path to persist the evaluation report.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    run_phase3_eval(args)


if __name__ == "__main__":  # pragma: no cover
    main()

