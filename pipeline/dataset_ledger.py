"""
Dataset ledger helpers for freezing the ~200 tour blurbs before a run.

The ledger only touches ``samples/*.txt`` files so it can be executed on a local
machine without relying on any auxiliary manifests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from collections import Counter

from .utils import iso_timestamp, sha256_path


@dataclass
class SampleFileRecord:
    """Metadata about an individual ``samples/*.txt`` file."""

    path: str
    sha256: str
    num_bytes: int
    plus_code: Optional[str]
    language: Optional[str]

    def to_dict(self) -> Dict[str, str | int | None]:
        return asdict(self)


@dataclass
class SnapshotStats:
    """Simple aggregate counts attached to the snapshot."""

    total_samples: int
    language_breakdown: Mapping[str, int]
    total_bytes: int

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "total_samples": self.total_samples,
            "language_breakdown": dict(self.language_breakdown),
            "total_bytes": self.total_bytes,
        }
        return payload


@dataclass
class DatasetSnapshot:
    """
    Canonical structure that ends up serialized next to ``runs/<run_id>/``.
    """

    generated_at: str
    samples_dir: str
    samples: List[SampleFileRecord]
    stats: SnapshotStats

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "generated_at": self.generated_at,
            "samples_dir": self.samples_dir,
            "samples": [sample.to_dict() for sample in self.samples],
            "stats": self.stats.to_dict(),
        }
        return payload


class DatasetLedger:
    """
    Load + hash the static tour blurbs so runs can reference a frozen snapshot.
    """

    def __init__(
        self,
        samples_dir: str | Path,
    ) -> None:
        self.samples_dir = Path(samples_dir)

    # --------------------------------------------------------------------- I/O
    def build_snapshot(self) -> DatasetSnapshot:
        samples = self._collect_sample_files()
        stats = self._summarize(samples)
        return DatasetSnapshot(
            generated_at=iso_timestamp(),
            samples_dir=str(self.samples_dir),
            samples=samples,
            stats=stats,
        )

    def write_snapshot(self, path: str | Path) -> DatasetSnapshot:
        """
        Convenience helper that writes the snapshot JSON and returns it.
        """
        snapshot = self.build_snapshot()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        import json

        target.write_text(
            json.dumps(snapshot.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return snapshot

    # ----------------------------------------------------------------- helpers
    def _collect_sample_files(self) -> List[SampleFileRecord]:
        records: List[SampleFileRecord] = []
        for path in sorted(self.samples_dir.glob("*.txt")):
            data = path.read_bytes()
            sha = sha256_path(path)
            plus_code, language = self._parse_filename(path.stem)
            records.append(
                SampleFileRecord(
                    path=str(path.relative_to(self.samples_dir.parent)),
                    sha256=sha,
                    num_bytes=len(data),
                    plus_code=plus_code,
                    language=language,
                )
            )
        return records

    def _summarize(
        self,
        samples: Iterable[SampleFileRecord],
    ) -> SnapshotStats:
        language_counts = Counter(
            (record.language or "unknown") for record in samples
        )
        total_bytes = sum(record.num_bytes for record in samples)
        sample_paths = {record.path for record in samples}
        return SnapshotStats(
            total_samples=len(sample_paths),
            language_breakdown=dict(language_counts),
            total_bytes=total_bytes,
        )

    @staticmethod
    def _parse_filename(stem: str) -> tuple[Optional[str], Optional[str]]:
        if "_" not in stem:
            return stem or None, None
        plus_code, language = stem.rsplit("_", 1)
        return plus_code or None, language or None


__all__ = [
    "DatasetLedger",
    "DatasetSnapshot",
    "SampleFileRecord",
    "SnapshotStats",
]


