from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

try:  # Python 3.11+ ships tomllib in stdlib.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - safety for alternate runtimes.
    tomllib = None  # type: ignore[assignment]

ENV_CONFIG_PATH = "OFFLINE_TOUR_GUIDE_EVAL_CONFIG"


def _enabled_fields(dataclass_obj: Any) -> list[str]:
    return [
        name
        for name, value in asdict(dataclass_obj).items()
        if isinstance(value, bool) and value
    ]


@dataclass(slots=True)
class VariantToggles:
    """
    Switches that control which transport/QA modules participate in a run.
    """

    permutations: bool = True
    rotations: bool = True
    householder_refinement: bool = False
    prompt_kl: bool = False
    lm_head_ridge: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return asdict(self)

    def enabled(self) -> list[str]:
        return _enabled_fields(self)


@dataclass(slots=True)
class MetricToggles:
    """
    Switches for QA metrics (placeholders until full implementations land).
    """

    logit_kl: bool = True
    hidden_state_cosine: bool = True
    residual_rms: bool = True
    surprisal: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return asdict(self)

    def enabled(self) -> list[str]:
        return _enabled_fields(self)


@dataclass(slots=True)
class EvalConfig:
    """
    High-level configuration for QA notebooks + reporting scripts.
    """

    run_id: str = "dev"
    checkpoints: list[str] = field(default_factory=list)
    manifest_path: Optional[str] = None
    dataset_snapshot: Optional[str] = None
    metrics_output_path: str = "runs/dev/logs/metrics.jsonl"
    notes: str = ""
    variant: VariantToggles = field(default_factory=VariantToggles)
    metrics: MetricToggles = field(default_factory=MetricToggles)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def metrics_path(self) -> Path:
        return Path(self.metrics_output_path)

    @property
    def run_dir(self) -> Path:
        if self.manifest_path:
            return Path(self.manifest_path).parent
        return self.metrics_path.parent.parent

    def ensure_output_dirs(self) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def enabled_variant_labels(self) -> list[str]:
        return self.variant.enabled()

    def enabled_metric_labels(self) -> list[str]:
        return self.metrics.enabled()


def _resolve_path(path: str | Path | None) -> Optional[Path]:
    candidate = path or os.getenv(ENV_CONFIG_PATH)
    if candidate is None:
        return None
    resolved = Path(candidate).expanduser()
    return resolved if resolved.exists() else None


def load_eval_config(
    path: str | Path | None = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> EvalConfig:
    """
    Load config from JSON/TOML (if present) and apply optional overrides.
    """

    data: Dict[str, Any] = {}
    resolved = _resolve_path(path)
    if resolved:
        data = _read_config_file(resolved)

    if overrides:
        data = _deep_merge(data, overrides)

    return _build_config_from_mapping(data)


def save_eval_config(config: EvalConfig, path: str | Path) -> Path:
    """
    Persist the config as JSON (handy when templating new runs).
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _build_config_from_mapping(payload: Mapping[str, Any]) -> EvalConfig:
    variant = VariantToggles(**payload.get("variant", {}))
    metrics = MetricToggles(**payload.get("metrics", {}))
    checkpoints = payload.get("checkpoints", [])
    manifest_path = payload.get("manifest_path")
    dataset_snapshot = payload.get("dataset_snapshot")
    metrics_output_path = payload.get("metrics_output_path", "runs/dev/logs/metrics.jsonl")
    notes = payload.get("notes", "")
    run_id = payload.get("run_id", "dev")
    config = EvalConfig(
        run_id=run_id,
        checkpoints=list(checkpoints),
        manifest_path=manifest_path,
        dataset_snapshot=dataset_snapshot,
        metrics_output_path=metrics_output_path,
        notes=notes,
        variant=variant,
        metrics=metrics,
    )
    return config


def _read_config_file(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".json", ""}:
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".toml", ".tml"}:
        if tomllib is None:
            raise RuntimeError("tomllib is unavailable; cannot parse TOML config.")
        return tomllib.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported config extension '{suffix}'.")


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            merged[key] = _deep_merge(base[key], value)  # type: ignore[index]
        else:
            merged[key] = value
    return merged


__all__ = [
    "EvalConfig",
    "VariantToggles",
    "MetricToggles",
    "load_eval_config",
    "save_eval_config",
]


