"""
Run scaffolding helpers for the Qwen32B → 3B transport experiments.

These utilities do not require the teacher or student checkpoints; they simply
prepare the ``runs/<run_id>`` directory, hash prompts, and record dataset
snapshots so that once GPUs are available the activation capture jobs can slot
in immediately.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .dataset_ledger import DatasetLedger
from .utils import iso_timestamp, sha256_path


@dataclass
class PromptHash:
    """Lightweight mapping of prompt/style files to their digests."""

    path: str
    sha256: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class ActivationShardInfo:
    """Placeholder metadata for activation shards recorded later."""

    path: str
    checksum: str
    layer_start: Optional[int] = None
    layer_end: Optional[int] = None
    token_count: Optional[int] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "checksum": self.checksum,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "token_count": self.token_count,
        }


@dataclass
class RunManifest:
    """
    The manifest that gets serialized to ``runs/<run_id>/manifest.json``.
    """

    run_id: str
    created_at: str
    prompt_hashes: List[PromptHash]
    dataset_snapshot_path: str
    tokenizer: Mapping[str, str]
    teacher_commit: Optional[str] = None
    student_commit: Optional[str] = None
    activation_shards: List[ActivationShardInfo] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "prompt_hashes": [entry.to_dict() for entry in self.prompt_hashes],
            "dataset_snapshot_path": self.dataset_snapshot_path,
            "tokenizer": dict(self.tokenizer),
            "activation_shards": [shard.to_dict() for shard in self.activation_shards],
        }
        if self.teacher_commit:
            payload["teacher_commit"] = self.teacher_commit
        if self.student_commit:
            payload["student_commit"] = self.student_commit
        return payload


@dataclass
class RunContext:
    """Returned from ``RunScaffolder.create_run`` for convenience."""

    run_id: str
    run_dir: Path
    manifest_path: Path
    dataset_snapshot_path: Path
    checkpoints_dir: Path


class RunScaffolder:
    """
    Create run directories, manifests, and placeholder logs ahead of GPU time.
    """

    def __init__(self, runs_root: str | Path = "runs") -> None:
        self.runs_root = Path(runs_root)

    # ----------------------------------------------------------------- creation
    def create_run(
        self,
        prompt_paths: Sequence[str | Path],
        dataset_snapshot: Optional[Mapping[str, object]] = None,
        dataset_snapshot_path: Optional[str | Path] = None,
        tokenizer_info: Optional[Mapping[str, str]] = None,
        teacher_commit: Optional[str] = None,
        student_commit: Optional[str] = None,
        activation_shards: Optional[Sequence[ActivationShardInfo]] = None,
        run_id: Optional[str] = None,
    ) -> RunContext:
        run_identifier = run_id or self._generate_run_id()
        run_dir = self.runs_root / run_identifier
        run_dir.mkdir(parents=True, exist_ok=False)
        checkpoints_dir = run_dir / "checkpoints"
        activations_dir = run_dir / "activations"
        logs_dir = run_dir / "logs"
        for subdir in (checkpoints_dir, activations_dir, logs_dir):
            subdir.mkdir(parents=True, exist_ok=True)

        dataset_snapshot_path = self._resolve_snapshot_path(
            run_dir, dataset_snapshot, dataset_snapshot_path
        )
        manifest = self._build_manifest(
            run_identifier,
            prompt_paths,
            dataset_snapshot_path,
            tokenizer_info or {"name": "unknown", "version": "unknown"},
            teacher_commit,
            student_commit,
            activation_shards or [],
        )
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        rotations_path = run_dir / "rotations.jsonl"
        rotations_path.write_text("", encoding="utf-8")
        report_path = run_dir / "report.md"
        if not report_path.exists():
            report_path.write_text(
                f"# Run {run_identifier} Report\n\n"
                "This is a placeholder. Populate once QA metrics are available.\n",
                encoding="utf-8",
            )

        return RunContext(
            run_id=run_identifier,
            run_dir=run_dir,
            manifest_path=manifest_path,
            dataset_snapshot_path=Path(dataset_snapshot_path),
            checkpoints_dir=checkpoints_dir,
        )

    # -------------------------------------------------------------- prompt hash
    def _prompt_hashes(self, prompt_paths: Sequence[str | Path]) -> List[PromptHash]:
        hashes: List[PromptHash] = []
        for path_like in prompt_paths:
            path = Path(path_like)
            if not path.exists():
                raise FileNotFoundError(f"Prompt file '{path}' does not exist.")
            hashes.append(
                PromptHash(
                    path=str(path),
                    sha256=sha256_path(path),
                )
            )
        return hashes

    # ----------------------------------------------------------- manifest core
    def _build_manifest(
        self,
        run_id: str,
        prompt_paths: Sequence[str | Path],
        dataset_snapshot_path: str | Path,
        tokenizer_info: Mapping[str, str],
        teacher_commit: Optional[str],
        student_commit: Optional[str],
        activation_shards: Sequence[ActivationShardInfo],
    ) -> RunManifest:
        prompt_hashes = self._prompt_hashes(prompt_paths)
        return RunManifest(
            run_id=run_id,
            created_at=iso_timestamp(),
            prompt_hashes=prompt_hashes,
            dataset_snapshot_path=str(dataset_snapshot_path),
            tokenizer=tokenizer_info,
            teacher_commit=teacher_commit,
            student_commit=student_commit,
            activation_shards=list(activation_shards),
        )

    def _resolve_snapshot_path(
        self,
        run_dir: Path,
        dataset_snapshot: Optional[Mapping[str, object]],
        dataset_snapshot_path: Optional[str | Path],
    ) -> Path:
        if dataset_snapshot_path:
            path = Path(dataset_snapshot_path)
        else:
            path = run_dir / "dataset_snapshot.json"

        if dataset_snapshot:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(dataset_snapshot, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        elif not path.exists():
            raise ValueError(
                "dataset_snapshot_path was not provided and no snapshot data was passed."
            )

        return path

    @staticmethod
    def _generate_run_id() -> str:
        return f"run_{iso_timestamp().replace(':', '').replace('-', '')}_{secrets.token_hex(2)}"


# ----------------------------------------------------------------------  CLI --
def _discover_prompt_files(prompts_dir: Path) -> List[Path]:
    extensions = {".md", ".txt", ".csv"}
    return sorted(
        path
        for path in prompts_dir.iterdir()
        if path.suffix.lower() in extensions and path.is_file()
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Create a run scaffold + dataset snapshot.")
    parser.add_argument("--run-id", default=None, help="Override the auto-generated run id.")
    parser.add_argument("--runs-root", default="runs", help="Directory for run artifacts.")
    parser.add_argument("--prompt", action="append", help="Prompt/style file to hash (repeatable).")
    parser.add_argument(
        "--prompts-dir",
        default="prompts",
        help="Directory to auto-discover prompt/style files when --prompt isn't supplied.",
    )
    parser.add_argument("--samples-dir", default="samples", help="Directory containing .txt blurbs.")
    parser.add_argument("--tokenizer-name", default="unknown", help="Tokenizer identifier.")
    parser.add_argument("--tokenizer-version", default="unknown", help="Tokenizer version string.")
    parser.add_argument("--teacher-commit", default=None, help="Teacher model git commit hash.")
    parser.add_argument("--student-commit", default=None, help="Student model git commit hash.")
    args = parser.parse_args(argv)

    prompts_dir = Path(args.prompts_dir)
    prompt_paths = (
        [Path(p) for p in args.prompt]
        if args.prompt
        else _discover_prompt_files(prompts_dir)
    )
    if not prompt_paths:
        raise ValueError("No prompt files were provided or discovered.")

    ledger = DatasetLedger(samples_dir=args.samples_dir)
    snapshot = ledger.build_snapshot().to_dict()

    scaffolder = RunScaffolder(runs_root=args.runs_root)
    context = scaffolder.create_run(
        prompt_paths=prompt_paths,
        dataset_snapshot=snapshot,
        tokenizer_info={
            "name": args.tokenizer_name,
            "version": args.tokenizer_version,
        },
        teacher_commit=args.teacher_commit,
        student_commit=args.student_commit,
        run_id=args.run_id,
    )

    print(f"Created run scaffold at {context.run_dir}")
    print(f"- manifest: {context.manifest_path}")
    print(f"- dataset snapshot: {context.dataset_snapshot_path}")


if __name__ == "__main__":
    main()


