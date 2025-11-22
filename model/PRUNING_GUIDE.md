# Activation-Based Pruning Guide

This guide explains how to use the activation-based pruning pipeline to reduce the size of your Qwen3-32B model for mobile deployment.

## Overview

The pruning pipeline consists of 4 main steps:

1. **Activation Recording**: Record activations from the model during forward passes
2. **Activation Analysis**: Analyze activations to identify pruning targets
3. **Model Pruning**: Remove less important layers/neurons based on analysis
4. **Fine-tuning**: Recover performance through fine-tuning

## Important Note: vLLM API Limitation

**The current setup uses vLLM on RunPod via API, which doesn't allow direct access to model internals.**

To record activations, you have two options:

### Option 1: Use a Local Model Copy (Recommended)

1. Download Qwen3-32B (or a smaller variant for testing) locally
2. Load it using HuggingFace transformers
3. Record activations from the local model
4. Apply the same pruning strategy to your RunPod model

### Option 2: Modify vLLM Server (Advanced)

1. Deploy a custom vLLM server with activation hooks
2. Modify vLLM's model loading code to register hooks
3. Expose activation data via API or save to disk

**For this guide, we'll focus on Option 1 (local model).**

## Step-by-Step Guide

### Step 1: Record Activations

First, you need to record activations from your model. This requires a local copy of the model.

```python
from model import (
    load_model_for_activation_recording,
    record_activations_from_dataset
)

# Load model locally (use smaller model for testing)
model, tokenizer = load_model_for_activation_recording(
    model_name="Qwen/Qwen2.5-7B-Instruct",  # Or "Qwen/Qwen3-32B" if you have resources
    device="cuda",
    torch_dtype=torch.float16,
    load_in_8bit=True  # Use 8-bit quantization to save memory
)

# Prepare activation texts (diverse inputs from your domain)
activation_texts = [
    "You are an engaging, extremely knowledgeable tour guide.",
    "Generate a detailed tour guide description for Sydney Opera House.",
    "This tour group is interested in architecture and history.",
    # ... more diverse texts from your tour guide domain
] * 50  # Repeat for more data

# Record activations
recorder = record_activations_from_dataset(
    model=model,
    tokenizer=tokenizer,
    texts=activation_texts,
    batch_size=1,
    max_length=512,
    save_dir="activations"
)
```

### Step 2: Analyze Activations

Analyze the recorded activations to determine which layers/neurons to prune.

```python
from model import analyze_activations_from_file

# Analyze activations
analyzer = analyze_activations_from_file(
    activations_file="activations/activations.pkl",
    output_file="analysis.json"
)

# Get pruning candidates
candidates = analyzer.get_pruning_candidates(
    pruning_ratio=0.3,  # Prune 30% of the model
    strategy="importance"  # or "sparsity", "variance"
)

# View results
for layer_name, score in candidates[:10]:
    print(f"{layer_name}: {score:.4f}")
```

### Step 3: Prune the Model

Prune the model based on the analysis.

```python
from model import StructuredPruner

# Initialize pruner
pruner = StructuredPruner(model)

# Get model size before
size_before = pruner.get_model_size(model)
print(f"Before: {size_before['total_parameters']:,} parameters")

# Prune layers
layer_names_to_prune = [name for name, _ in candidates[:int(len(candidates) * 0.3)]]
pruned_model = pruner.prune_layers(
    layers_to_prune=layer_names_to_prune,
    keep_ratio=0.7
)

# Get model size after
size_after = pruner.get_model_size(pruned_model)
print(f"After: {size_after['total_parameters']:,} parameters")
print(f"Reduction: {(1 - size_after['total_parameters'] / size_before['total_parameters']) * 100:.1f}%")

# Save pruned model
pruner.save_pruned_model(pruned_model, "pruned_model.pt")
```

### Step 4: Fine-tune the Pruned Model

Fine-tune the pruned model to recover performance.

```python
from model import PrunedModelTrainer, prepare_training_data_from_jsonl

# Load training data (from your training_data.jsonl)
train_texts = prepare_training_data_from_jsonl("training_data.jsonl", text_field="response")

# Create trainer
trainer = PrunedModelTrainer(pruned_model, tokenizer, device="cuda")

# Fine-tune
history = trainer.fine_tune(
    train_texts=train_texts,
    num_epochs=3,
    learning_rate=2e-5,
    batch_size=4,
    max_length=512,
    save_dir="checkpoints"
)

# Save final model
trainer.save_checkpoint("fine_tuned_model.pt", 2, history['train_loss'][-1])
```

## Complete Pipeline Example

Use the provided pipeline script for a complete workflow:

```python
from model import run_pruning_pipeline

# Prepare activation texts
activation_texts = [
    # Your diverse set of tour guide prompts
] * 50

# Run complete pipeline
run_pruning_pipeline(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    activation_texts=activation_texts,
    training_data_file="training_data.jsonl",
    output_dir="pruning_output",
    pruning_ratio=0.3,
    num_epochs=3
)
```

## Activation Metrics Explained

The analyzer computes several metrics:

1. **Magnitude Metrics**:
   - L1/L2 norms: Overall activation strength
   - Per-neuron norms: Individual neuron importance

2. **Variance Metrics**:
   - Overall variance: How much activations vary
   - Per-neuron variance: Which neurons are most active

3. **Sparsity Metrics**:
   - Overall sparsity: Percentage of near-zero activations
   - Per-neuron sparsity: Which neurons are rarely active

4. **Importance Scores**:
   - Composite score combining magnitude, variance, and sparsity
   - Lower scores = better candidates for pruning

## Pruning Strategies

### Strategy 1: Importance-Based (Recommended)
- Uses composite importance scores
- Balances multiple factors
- Best for general-purpose pruning

### Strategy 2: Sparsity-Based
- Focuses on rarely-activated neurons
- Good for aggressive pruning
- May remove neurons that are important but sparse

### Strategy 3: Variance-Based
- Focuses on low-variance neurons
- Good for removing redundant neurons
- Preserves high-variance (important) neurons

## Best Practices

1. **Diverse Activation Data**: Use diverse inputs from your domain (tour guide prompts)

2. **Adequate Data Volume**: Record activations from 50-100+ examples for reliable statistics

3. **Gradual Pruning**: Start with 20-30% pruning, then fine-tune and evaluate before pruning more

4. **Layer vs Neuron Pruning**:
   - Layer pruning: Easier, but may remove important functionality
   - Neuron pruning: More granular, better preservation of capabilities

5. **Fine-tuning is Critical**: Always fine-tune after pruning to recover performance

6. **Evaluation**: Test on your specific task (tour guide generation) after each pruning step

## Memory Considerations

For Qwen3-32B:
- Full model: ~64GB (FP32) or ~32GB (FP16)
- Use quantization (8-bit or 4-bit) to reduce memory
- Consider using a smaller model (7B) for initial testing

## Troubleshooting

### Out of Memory
- Use smaller batch sizes
- Enable gradient checkpointing
- Use quantization (8-bit/4-bit)
- Process activations in chunks

### Poor Pruning Results
- Increase activation data diversity
- Try different pruning strategies
- Adjust pruning ratio
- Fine-tune for more epochs

### Model Architecture Issues
- The pruner may need model-specific adaptations
- Qwen3 architecture may require custom pruning logic
- Check layer names match your model structure

## Next Steps

1. Test with a smaller model (7B) first
2. Adapt pruning logic for Qwen3 architecture if needed
3. Implement per-neuron analysis for finer-grained pruning
4. Deploy pruned model to mobile device
5. Benchmark performance vs. original model

## References

- [Neural Network Pruning Survey](https://arxiv.org/abs/2003.03033)
- [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)
- [Magnitude-based Pruning](https://papers.nips.cc/paper/1989/hash/dc5c768b5dc76a084831934ff4a9d1ee-Abstract.html)

