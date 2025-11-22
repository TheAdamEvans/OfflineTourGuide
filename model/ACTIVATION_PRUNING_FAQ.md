# Activation-Based Pruning FAQ

This document answers your specific questions about implementing activation-based pruning for Qwen3-32B.

## 1. How to Record Activations from Qwen3-32B on RunPod/vLLM?

### The Challenge

Since you're using vLLM on RunPod via API, you **cannot directly record activations** through the API. The OpenAI-compatible API only returns text outputs, not internal activations.

### Solution: Use a Local Model Copy

**Recommended Approach:**

1. **Download the model locally** (or use a smaller variant for testing):
   ```python
   from model import load_model_for_activation_recording
   
   model, tokenizer = load_model_for_activation_recording(
       model_name="Qwen/Qwen3-32B",  # Or "Qwen/Qwen2.5-7B-Instruct" for testing
       device="cuda",
       torch_dtype=torch.float16,
       load_in_8bit=True  # Use quantization to save memory
   )
   ```

2. **Record activations from the local model**:
   ```python
   from model import record_activations_from_dataset
   
   recorder = record_activations_from_dataset(
       model=model,
       tokenizer=tokenizer,
       texts=your_activation_texts,
       save_dir="activations"
   )
   ```

3. **Apply the same pruning strategy** to your RunPod model (or deploy the pruned model).

### Alternative: Modify vLLM Server (Advanced)

If you need to record from the exact RunPod instance:

1. Deploy a custom vLLM server with activation hooks
2. Modify vLLM's model loading to register forward hooks
3. Save activations to disk or expose via custom API endpoint

**This is complex and not recommended unless necessary.**

## 2. What Activation Metrics to Record?

The implementation records and analyzes multiple metrics:

### Magnitude Metrics
- **L1/L2 Norms**: Overall activation strength
- **Per-neuron norms**: Individual neuron importance
- **Max/Mean absolute values**: Activation ranges

**Why**: Neurons with consistently low activations are less important.

### Variance Metrics
- **Overall variance**: How much activations vary across inputs
- **Per-neuron variance**: Which neurons are most dynamic
- **Min/Max neuron variance**: Range of neuron activity

**Why**: Low-variance neurons are redundant and can be pruned.

### Sparsity Metrics
- **Overall sparsity**: Percentage of near-zero activations
- **Per-neuron sparsity**: Which neurons are rarely active
- **High-sparsity ratio**: Fraction of neurons that are >90% sparse

**Why**: Sparse neurons contribute little and are good pruning candidates.

### Composite Importance Scores
- **Weighted combination** of magnitude, variance, and sparsity
- **Lower scores** = better candidates for pruning

**Usage**:
```python
analyzer.compute_importance_scores(layer_name, weights={
    'magnitude': 0.3,
    'variance': 0.4,
    'sparsity': 0.3
})
```

## 3. How to Analyze Recorded Activations?

The `ActivationAnalyzer` class provides comprehensive analysis:

```python
from model import analyze_activations_from_file

# Load and analyze
analyzer = analyze_activations_from_file(
    activations_file="activations/activations.pkl",
    output_file="analysis.json"
)

# Get pruning candidates
candidates = analyzer.get_pruning_candidates(
    pruning_ratio=0.3,  # Prune 30%
    strategy="importance"  # or "sparsity", "variance"
)

# View results
for layer_name, score in candidates:
    print(f"{layer_name}: {score:.4f}")
```

### Analysis Process

1. **Compute metrics** for each layer:
   - Magnitude, variance, sparsity
   - Per-neuron statistics where applicable

2. **Rank layers/neurons** by importance:
   - Lower importance = better pruning candidate

3. **Select candidates** based on pruning ratio:
   - Top N% least important layers/neurons

### Strategies

- **"importance"**: Balanced approach using composite scores
- **"sparsity"**: Focus on rarely-active neurons
- **"variance"**: Focus on low-variance (redundant) neurons

## 4. Implementation of Structured Pruning

The `StructuredPruner` class implements structured pruning:

### Layer Pruning
```python
pruner = StructuredPruner(model)

# Prune entire layers
pruned_model = pruner.prune_layers(
    layers_to_prune=["layer.5", "layer.10"],
    keep_ratio=0.7  # Keep 70% of layers
)
```

### Neuron Pruning
```python
# Prune specific neurons from a layer
pruned_model = pruner.prune_neurons(
    layer_name="mlp.gate_proj",
    neuron_indices=[10, 20, 30, ...]
)
```

### MLP Layer Pruning
```python
# Prune neurons from all MLP layers
pruning_plan = {
    "mlp.gate_proj": [10, 20, 30],
    "mlp.up_proj": [5, 15, 25],
    # ...
}

pruned_model = pruner.prune_mlp_layers(
    pruning_plan=pruning_plan,
    keep_ratio=0.7
)
```

### Model-Specific Adaptation

**Important**: The current implementation is generic. For Qwen3-32B, you may need to:

1. **Identify layer structure**: Check how Qwen3 organizes layers
   ```python
   for name, module in model.named_modules():
       print(name, type(module))
   ```

2. **Adapt pruning logic**: Modify `prune_layers()` for Qwen3's architecture

3. **Handle attention heads**: Implement attention head pruning if needed

## 5. Best Practices for Fine-tuning After Pruning

### Fine-tuning Setup

```python
from model import PrunedModelTrainer, prepare_training_data_from_jsonl

# Load training data
train_texts = prepare_training_data_from_jsonl(
    "training_data.jsonl",
    text_field="response"
)

# Create trainer
trainer = PrunedModelTrainer(pruned_model, tokenizer, device="cuda")

# Fine-tune
history = trainer.fine_tune(
    train_texts=train_texts,
    val_texts=val_texts,  # Optional validation set
    num_epochs=3,
    learning_rate=2e-5,  # Lower than pretraining
    batch_size=4,
    max_length=512,
    warmup_steps=100,
    save_dir="checkpoints"
)
```

### Best Practices

1. **Learning Rate**: Use lower LR (1e-5 to 5e-5) than pretraining
   - Pruned models are more sensitive
   - Start with 2e-5 and adjust

2. **Warmup**: Use learning rate warmup (100-500 steps)
   - Helps stabilize training after pruning

3. **Gradient Clipping**: Already implemented (max_norm=1.0)
   - Prevents gradient explosion

4. **Training Data**: Use your domain-specific data
   - Tour guide descriptions from your dataset
   - Diverse examples covering your use cases

5. **Evaluation**: Monitor validation loss/perplexity
   - Stop if validation loss increases
   - Use early stopping if needed

6. **Gradual Pruning**: Don't prune too much at once
   - Start with 20-30% pruning
   - Fine-tune, evaluate, then prune more if needed

7. **Task-Specific Evaluation**: Test on your actual task
   - Generate tour guide descriptions
   - Compare quality with original model

### Fine-tuning Schedule

```
Epoch 1: High learning rate (2e-5), recover basic functionality
Epoch 2: Medium learning rate (1e-5), refine performance
Epoch 3: Lower learning rate (5e-6), fine-tune details
```

### Recovery Expectations

- **20-30% pruning**: Should recover 90-95% of performance
- **40-50% pruning**: May lose 10-20% performance
- **>50% pruning**: Significant performance loss, may not fully recover

## Complete Workflow Example

```python
from model import run_pruning_pipeline

# Run complete pipeline
run_pruning_pipeline(
    model_name="Qwen/Qwen2.5-7B-Instruct",  # Test with smaller model first
    activation_texts=your_activation_texts,
    training_data_file="training_data.jsonl",
    output_dir="pruning_output",
    pruning_ratio=0.3,  # Start conservative
    num_epochs=3
)
```

## Memory Considerations for Qwen3-32B

- **Full model (FP32)**: ~128GB
- **Full model (FP16)**: ~64GB
- **With 8-bit quantization**: ~32GB
- **With 4-bit quantization**: ~16GB

**Recommendations**:
- Use quantization for activation recording
- Test with smaller model (7B) first
- Use gradient checkpointing during fine-tuning
- Consider model parallelism for large models

## Next Steps

1. **Test with smaller model** (7B) to validate pipeline
2. **Adapt pruning logic** for Qwen3 architecture
3. **Record activations** from diverse tour guide prompts
4. **Prune gradually** (20% → 30% → 40%)
5. **Fine-tune and evaluate** after each pruning step
6. **Deploy pruned model** to mobile device

## Files Created

- `activation_recorder.py`: Record activations from models
- `activation_analyzer.py`: Analyze activations and compute metrics
- `pruner.py`: Implement structured pruning
- `finetune.py`: Fine-tuning utilities
- `pruning_pipeline.py`: Complete pipeline script
- `example_pruning.py`: Simple example script
- `PRUNING_GUIDE.md`: Detailed guide
- `ACTIVATION_PRUNING_FAQ.md`: This file

