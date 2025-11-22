"""
Model utilities for Qwen3-32B FP8 and activation-based pruning.

This module provides:
- Activation recording from transformer models
- Activation analysis for determining pruning targets
- Structured pruning (layer/neuron removal)
- Fine-tuning utilities for pruned models
"""

from .activation_recorder import (
    ActivationRecorder,
    load_model_for_activation_recording,
    record_activations_from_dataset
)

from .activation_analyzer import (
    ActivationAnalyzer,
    analyze_activations_from_file
)

from .pruner import (
    StructuredPruner,
    create_pruning_plan_from_analysis
)

from .finetune import (
    PrunedModelTrainer,
    prepare_training_data_from_jsonl
)

from .pruning_pipeline import run_pruning_pipeline

__all__ = [
    # Activation recording
    'ActivationRecorder',
    'load_model_for_activation_recording',
    'record_activations_from_dataset',
    # Activation analysis
    'ActivationAnalyzer',
    'analyze_activations_from_file',
    # Pruning
    'StructuredPruner',
    'create_pruning_plan_from_analysis',
    # Fine-tuning
    'PrunedModelTrainer',
    'prepare_training_data_from_jsonl',
    # Pipeline
    'run_pruning_pipeline'
]

