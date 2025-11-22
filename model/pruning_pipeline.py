"""
Complete activation-based pruning pipeline example.

This script demonstrates the full workflow:
1. Record activations from the model
2. Analyze activations to determine pruning targets
3. Prune the model based on analysis
4. Fine-tune the pruned model to recover performance
"""

import torch
from pathlib import Path
from typing import List, Optional

from .activation_recorder import (
    ActivationRecorder,
    load_model_for_activation_recording,
    record_activations_from_dataset
)
from .activation_analyzer import ActivationAnalyzer, analyze_activations_from_file
from .pruner import StructuredPruner, create_pruning_plan_from_analysis
from .finetune import PrunedModelTrainer, prepare_training_data_from_jsonl


def run_pruning_pipeline(
    activation_texts: List[str],
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",  # Use smaller model for testing
    training_data_file: Optional[str] = None,
    output_dir: str = "pruning_output",
    pruning_ratio: float = 0.3,
    num_epochs: int = 3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
):
    """
    Run the complete activation-based pruning pipeline.
    
    Args:
        model_name: HuggingFace model identifier
        activation_texts: List of texts to use for activation recording
        training_data_file: Optional JSONL file with training data for fine-tuning
        output_dir: Directory to save all outputs
        pruning_ratio: Fraction of model to prune (0.0 to 1.0)
        num_epochs: Number of fine-tuning epochs
        device: Device to run on
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Activation-Based Pruning Pipeline")
    print("=" * 60)
    
    # Step 1: Load model
    print("\n[Step 1] Loading model...")
    model, tokenizer = load_model_for_activation_recording(
        model_name=model_name,
        device=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit
    )
    
    # Step 2: Record activations
    print("\n[Step 2] Recording activations...")
    recorder = record_activations_from_dataset(
        model=model,
        tokenizer=tokenizer,
        texts=activation_texts,
        batch_size=1,
        max_length=512,
        save_dir=str(output_path / "activations")
    )
    
    activation_file = output_path / "activations" / "activations.pkl"
    print(f"Activations saved to {activation_file}")
    
    # Step 3: Analyze activations
    print("\n[Step 3] Analyzing activations...")
    analyzer = analyze_activations_from_file(
        activations_file=str(activation_file),
        output_file=str(output_path / "analysis.json")
    )
    
    # Get pruning candidates
    candidates = analyzer.get_pruning_candidates(
        pruning_ratio=pruning_ratio,
        strategy="importance"
    )
    print(f"\nIdentified {len(candidates)} layers/neurons for pruning:")
    for layer_name, score in candidates[:10]:  # Show top 10
        print(f"  - {layer_name}: score={score:.4f}")
    
    # Step 4: Prune model
    print("\n[Step 4] Pruning model...")
    pruner = StructuredPruner(model)
    
    # Get model size before pruning
    size_before = pruner.get_model_size(model)
    print(f"Model size before pruning: {size_before['total_parameters']:,} parameters "
          f"({size_before['size_mb']:.2f} MB)")
    
    # Create pruning plan (simplified - would need per-neuron analysis)
    # For now, we'll do layer-level pruning
    layer_names_to_prune = [name for name, _ in candidates[:int(len(candidates) * pruning_ratio)]]
    
    # Prune layers (this is model-specific)
    pruned_model = pruner.prune_layers(
        layers_to_prune=layer_names_to_prune,
        keep_ratio=1.0 - pruning_ratio
    )
    
    # Get model size after pruning
    size_after = pruner.get_model_size(pruned_model)
    print(f"Model size after pruning: {size_after['total_parameters']:,} parameters "
          f"({size_after['size_mb']:.2f} MB)")
    print(f"Reduction: {(1 - size_after['total_parameters'] / size_before['total_parameters']) * 100:.1f}%")
    
    # Save pruned model
    pruned_model_path = output_path / "pruned_model.pt"
    pruner.save_pruned_model(pruned_model, str(pruned_model_path))
    
    # Step 5: Fine-tune (if training data provided)
    if training_data_file:
        print("\n[Step 5] Fine-tuning pruned model...")
        
        # Load training data
        train_texts = prepare_training_data_from_jsonl(training_data_file)
        print(f"Loaded {len(train_texts)} training examples")
        
        # Create trainer
        trainer = PrunedModelTrainer(pruned_model, tokenizer, device=device)
        
        # Fine-tune
        history = trainer.fine_tune(
            train_texts=train_texts,
            num_epochs=num_epochs,
            learning_rate=2e-5,
            batch_size=4,
            max_length=512,
            save_dir=str(output_path / "checkpoints")
        )
        
        # Save fine-tuned model
        final_model_path = output_path / "fine_tuned_model.pt"
        trainer.save_checkpoint(str(final_model_path), num_epochs - 1, history['train_loss'][-1])
        
        print(f"\nFine-tuning complete! Model saved to {final_model_path}")
    else:
        print("\n[Step 5] Skipping fine-tuning (no training data provided)")
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    print(f"\nOutputs saved to: {output_path}")
    print(f"  - Activations: {output_path / 'activations'}")
    print(f"  - Analysis: {output_path / 'analysis.json'}")
    print(f"  - Pruned model: {output_path / 'pruned_model.pt'}")
    if training_data_file:
        print(f"  - Fine-tuned model: {output_path / 'fine_tuned_model.pt'}")


if __name__ == "__main__":
    # Example usage
    example_texts = [
        "You are an engaging, extremely knowledgeable tour guide.",
        "Generate a detailed tour guide description for Sydney Opera House.",
        "This tour group is interested in architecture and history.",
    ]
    
    run_pruning_pipeline(
        model_name="Qwen/Qwen2.5-7B-Instruct",  # Use smaller model for testing
        activation_texts=example_texts * 10,  # Repeat for more data
        training_data_file=None,  # Set to your training_data.jsonl if available
        output_dir="pruning_output",
        pruning_ratio=0.3,
        num_epochs=3
    )

