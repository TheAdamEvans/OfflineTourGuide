# Activation-Based Pruning Module

This module provides tools for pruning large language models (like Qwen3-32B) based on activation analysis, making them suitable for mobile deployment.

## Quick Start

### For Local Testing
👉 **See [README_LOCAL.md](README_LOCAL.md)** for local setup and execution

```bash
# Quick test with small model
python -m model.example_pruning
```

### For RunPod Deployment
👉 **See [RUNPOD_SETUP.md](RUNPOD_SETUP.md)** for RunPod setup

```bash
# On RunPod with GPU
python -m model.example_pruning
```

## Documentation

- **[README_LOCAL.md](README_LOCAL.md)** - Running locally (testing, development)
- **[RUNPOD_SETUP.md](RUNPOD_SETUP.md)** - Running on RunPod (production, large models)
- **[PRUNING_GUIDE.md](PRUNING_GUIDE.md)** - Detailed pruning concepts and workflows
- **[ACTIVATION_PRUNING_FAQ.md](ACTIVATION_PRUNING_FAQ.md)** - Common questions and answers

## What This Module Does

1. **Records Activations**: Captures model activations during forward passes
2. **Analyzes Activations**: Computes metrics (magnitude, variance, sparsity) to identify pruning targets
3. **Prunes Model**: Removes less important layers/neurons based on analysis
4. **Fine-tunes**: Recovers performance through fine-tuning

## Module Structure

```
model/
├── activation_recorder.py    # Record activations from models
├── activation_analyzer.py    # Analyze activations for pruning
├── pruner.py                # Implement structured pruning
├── finetune.py              # Fine-tuning utilities
├── pruning_pipeline.py      # Complete pipeline script
└── example_pruning.py       # Simple example script
```

## Requirements

- Python 3.12+
- PyTorch 2.0+
- Transformers 4.40+
- (Optional) bitsandbytes for quantization (Linux/Windows only)

## Example Usage

```python
from model import run_pruning_pipeline

# Run complete pipeline
run_pruning_pipeline(
    activation_texts=your_texts,
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    output_dir="pruning_output",
    pruning_ratio=0.3
)
```

## Next Steps

1. **Test Locally**: Start with `README_LOCAL.md` using a small model
2. **Review Analysis**: Check activation analysis results
3. **Deploy to RunPod**: Use `RUNPOD_SETUP.md` for large models (32B+)
4. **Fine-tune**: Recover performance with your training data

## Support

- Check the FAQ: `ACTIVATION_PRUNING_FAQ.md`
- Review the guide: `PRUNING_GUIDE.md`
- See examples: `example_pruning.py`

