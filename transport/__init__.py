"""
Alignment and transport scaffolding utilities for activation-driven pruning.
"""

from .activation_cache import (
    ActivationCacheIndex,
    ActivationShardFormat,
    ActivationShardMetadata,
    ActivationShardPayload,
    ActivationShardReader,
    ActivationShardWriter,
    TokenSpan,
)
from .permutations import (
    apply_permutation,
    cosine_similarity_matrix,
    hungarian_match,
    match_attention_heads,
    match_neurons,
)
from .rotations import RotationConfig, RotationPipeline, RotationResult
from .weight_ops import (
    BlockLinearGroup,
    HeadPermutation,
    NeuronPermutation,
    WeightTransformSet,
    fold_rotation_into_block,
    unfold_rotation_from_block,
)

__all__ = [
    "ActivationCacheIndex",
    "ActivationShardFormat",
    "ActivationShardMetadata",
    "ActivationShardPayload",
    "ActivationShardReader",
    "ActivationShardWriter",
    "BlockLinearGroup",
    "HeadPermutation",
    "NeuronPermutation",
    "RotationConfig",
    "RotationPipeline",
    "RotationResult",
    "TokenSpan",
    "WeightTransformSet",
    "apply_permutation",
    "cosine_similarity_matrix",
    "fold_rotation_into_block",
    "hungarian_match",
    "match_attention_heads",
    "match_neurons",
    "unfold_rotation_from_block",
]


