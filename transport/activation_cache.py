from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch


class ActivationShardFormat(str, Enum):
    """Supported serialization containers for activation shards."""

    TORCH = "pt"
    NUMPY = "npz"


@dataclass(frozen=True)
class TokenSpan:
    """Inclusive-exclusive token index range."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"Invalid token span ({self.start}, {self.end}).")

    def to_dict(self) -> Dict[str, int]:
        return {"start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, payload: Mapping[str, int]) -> "TokenSpan":
        return cls(start=int(payload["start"]), end=int(payload["end"]))

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class ActivationShardMetadata:
    """Bookkeeping entry persisted alongside serialized tensors."""

    shard_id: str
    path: str
    format: ActivationShardFormat
    token_span: TokenSpan
    layer_names: Sequence[str]
    checksum: str
    tensor_dtype: str
    extras: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["format"] = self.format.value
        payload["token_span"] = self.token_span.to_dict()
        payload["layer_names"] = list(self.layer_names)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ActivationShardMetadata":
        return cls(
            shard_id=str(payload["shard_id"]),
            path=str(payload["path"]),
            format=ActivationShardFormat(str(payload["format"])),
            token_span=TokenSpan.from_dict(payload["token_span"]),
            layer_names=tuple(payload.get("layer_names", [])),
            checksum=str(payload["checksum"]),
            tensor_dtype=str(payload["tensor_dtype"]),
            extras=dict(payload.get("extras", {})),
        )


@dataclass
class ActivationShardPayload:
    """
    Structured bundle of tensors emitted by the capture job.
    """

    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    activations: Mapping[str, torch.Tensor]
    logits: Optional[torch.Tensor] = None
    metadata: Optional[Mapping[str, object]] = None

    def tensor_dtype(self) -> torch.dtype:
        return self.input_ids.dtype

    def clone(self) -> "ActivationShardPayload":
        def _clone_tensor(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return tensor.detach().clone() if tensor is not None else None

        return ActivationShardPayload(
            input_ids=_clone_tensor(self.input_ids) or torch.empty(0),
            attention_mask=_clone_tensor(self.attention_mask),
            activations={name: tensor.detach().clone() for name, tensor in self.activations.items()},
            logits=_clone_tensor(self.logits),
            metadata=dict(self.metadata) if self.metadata else {},
        )

    def as_torch_payload(self) -> Dict[str, object]:
        return {
            "input_ids": self.input_ids.detach().clone(),
            "attention_mask": (
                self.attention_mask.detach().clone() if self.attention_mask is not None else None
            ),
            "activations": {
                name: tensor.detach().clone() for name, tensor in self.activations.items()
            },
            "logits": self.logits.detach().clone() if self.logits is not None else None,
            "metadata": dict(self.metadata) if self.metadata else {},
        }


class ActivationCacheIndex:
    """
    Append-only JSONL index that tracks shard metadata and checksums.
    """

    def __init__(self, metadata_path: str | Path) -> None:
        self.metadata_path = Path(metadata_path)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.metadata_path.exists():
            self.metadata_path.write_text("", encoding="utf-8")

    def append(self, metadata: ActivationShardMetadata) -> None:
        with self.metadata_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metadata.to_dict(), ensure_ascii=False) + "\n")

    def load_all(self) -> Sequence[ActivationShardMetadata]:
        entries: list[ActivationShardMetadata] = []
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entries.append(ActivationShardMetadata.from_dict(json.loads(line)))
        return entries


class ActivationShardWriter:
    """
    Serialize activation payloads and register their metadata/checksum.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        shard_format: ActivationShardFormat = ActivationShardFormat.TORCH,
        index: Optional[ActivationCacheIndex] = None,
        metadata_path: Optional[str | Path] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_format = shard_format
        if index:
            self.index = index
        else:
            metadata_file = metadata_path or (self.output_dir / "metadata.jsonl")
            self.index = ActivationCacheIndex(metadata_file)

    def write_shard(
        self,
        payload: ActivationShardPayload,
        *,
        token_span: TokenSpan,
        layer_names: Sequence[str],
        shard_id: Optional[str] = None,
    ) -> ActivationShardMetadata:
        shard_identifier = shard_id or self._generate_shard_id(token_span)
        path = self.output_dir / f"{shard_identifier}.{self.shard_format.value}"
        if self.shard_format is ActivationShardFormat.TORCH:
            torch.save(payload.as_torch_payload(), path)
        else:
            self._serialize_npz(payload, path)
        checksum = sha256_path(path)
        metadata = ActivationShardMetadata(
            shard_id=shard_identifier,
            path=str(path),
            format=self.shard_format,
            token_span=token_span,
            layer_names=list(layer_names),
            checksum=checksum,
            tensor_dtype=str(payload.tensor_dtype()),
            extras=dict(payload.metadata) if payload.metadata else {},
        )
        self.index.append(metadata)
        return metadata

    def _serialize_npz(self, payload: ActivationShardPayload, path: Path) -> None:
        arrays: Dict[str, np.ndarray] = {}

        def _to_numpy(tensor: Optional[torch.Tensor]) -> Optional[np.ndarray]:
            if tensor is None:
                return None
            return tensor.detach().to("cpu").numpy()

        arrays["input_ids"] = _to_numpy(payload.input_ids)
        if payload.attention_mask is not None:
            arrays["attention_mask"] = _to_numpy(payload.attention_mask)
        if payload.logits is not None:
            arrays["logits"] = _to_numpy(payload.logits)
        for layer, tensor in payload.activations.items():
            arrays[f"activations::{layer}"] = _to_numpy(tensor)
        if payload.metadata:
            arrays["metadata::json"] = np.array(
                [json.dumps(payload.metadata, ensure_ascii=False)], dtype=object
            )

        np.savez(path, **arrays)

    @staticmethod
    def _generate_shard_id(span: TokenSpan) -> str:
        return f"tokens_{span.start:06d}_{span.end:06d}"


class ActivationShardReader:
    """
    Load serialized shards given their metadata entries.
    """

    def __init__(self, metadata: ActivationShardMetadata) -> None:
        self.metadata = metadata

    def load(self) -> ActivationShardPayload:
        path = Path(self.metadata.path)
        if self.metadata.format is ActivationShardFormat.TORCH:
            payload = torch.load(path, map_location="cpu")
            activations = payload.get("activations", {})
            activations = {
                name: tensor.detach().clone() for name, tensor in activations.items()
            }
            return ActivationShardPayload(
                input_ids=payload["input_ids"],
                attention_mask=payload.get("attention_mask"),
                activations=activations,
                logits=payload.get("logits"),
                metadata=payload.get("metadata"),
            )

        arrays = np.load(path, allow_pickle=True)
        activations: Dict[str, torch.Tensor] = {}
        metadata_json = arrays.get("metadata::json")
        metadata_payload = json.loads(metadata_json[0]) if metadata_json is not None else None

        for key in arrays.files:
            if key.startswith("activations::"):
                layer = key.split("::", 1)[1]
                activations[layer] = torch.from_numpy(arrays[key]).clone()

        return ActivationShardPayload(
            input_ids=torch.from_numpy(arrays["input_ids"]).clone(),
            attention_mask=(
                torch.from_numpy(arrays["attention_mask"]).clone()
                if "attention_mask" in arrays
                else None
            ),
            activations=activations,
            logits=torch.from_numpy(arrays["logits"]).clone() if "logits" in arrays else None,
            metadata=metadata_payload,
        )


def sha256_path(path: str | Path) -> str:
    """
    Stream a file to compute its SHA256 hex digest.
    """
    import hashlib

    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


__all__ = [
    "ActivationCacheIndex",
    "ActivationShardFormat",
    "ActivationShardMetadata",
    "ActivationShardPayload",
    "ActivationShardReader",
    "ActivationShardWriter",
    "TokenSpan",
    "sha256_bytes",
    "sha256_path",
]


