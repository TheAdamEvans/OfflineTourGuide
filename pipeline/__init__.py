"""
Utility modules for planning and bookkeeping the offline tour-guide runs.

The package currently exposes helpers for:

- creating reproducible ``runs/<run_id>`` scaffolds with manifests and logs
- snapshotting the dataset/manifest inputs into a structured ledger
"""

from .dataset_ledger import DatasetLedger, DatasetSnapshot
from .run_scaffolder import (
    ActivationShardInfo,
    PromptHash,
    RunContext,
    RunManifest,
    RunScaffolder,
)

__all__ = [
    "ActivationShardInfo",
    "DatasetLedger",
    "DatasetSnapshot",
    "PromptHash",
    "RunContext",
    "RunManifest",
    "RunScaffolder",
]


