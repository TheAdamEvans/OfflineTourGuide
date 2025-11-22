# Running Activation-Based Pruning on RunPod

This guide explains how to run the activation-based pruning pipeline on a RunPod instance.

## Prerequisites

1. **RunPod GPU Instance** with:
   - At least 24GB VRAM (for Qwen2.5-7B) or 80GB+ VRAM (for Qwen3-32B)
   - At least 100GB free disk space
   - Linux environment (required for bitsandbytes quantization)

2. **Model Access**: Ensure you have access to download Qwen models from HuggingFace

## Setup on RunPod

### Step 1: Deploy RunPod Instance

1. Go to RunPod dashboard
2. Deploy a GPU pod (e.g., RTX 4090, A100, etc.)
3. Choose a template with Python and CUDA support

### Step 2: Install Dependencies

```bash
# Clone or upload your project
cd /workspace/OfflineTourGuide

# Install dependencies
pip install -e .

# Install quantization support (Linux only)
pip install bitsandbytes
```

### Step 3: Configure for RunPod

Update the example script to use your target model:

```python
# In model/example_pruning.py, change:
model_name = "Qwen/Qwen3-32B"  # Your target model
device = "cuda"  # Always use CUDA on RunPod
load_in_8bit = True  # Enable quantization to save memory
```

### Step 4: Run the Pipeline

```bash
# Run the example
python -m model.example_pruning

# Or run the full pipeline
python -m model.pruning_pipeline
```

## Memory Considerations

### For Qwen3-32B:
- **Full FP16**: ~64GB VRAM
- **With 8-bit quantization**: ~32GB VRAM
- **With 4-bit quantization**: ~16GB VRAM

### Recommended Setup:
- Use **8-bit quantization** for activation recording
- Use **4-bit quantization** if you have limited VRAM
- Process activations in smaller batches if needed

## Example RunPod Configuration

```python
# model/runpod_pruning_config.py
RUNPOD_CONFIG = {
    "model_name": "Qwen/Qwen3-32B",
    "device": "cuda",
    "torch_dtype": torch.float16,
    "load_in_8bit": True,  # Enable quantization
    "load_in_4bit": False,
    "batch_size": 1,  # Smaller batches for large models
    "max_length": 512,
    "pruning_ratio": 0.3,
    "num_epochs": 3
}
```

## Activation Recording on RunPod

The activation recording will:
1. Download the model (first time only, ~30-60GB)
2. Record activations from your input texts
3. Save activations to disk (~5-20GB depending on model size)
4. Analyze activations to determine pruning targets

## Tips for RunPod

1. **Use Persistent Storage**: Save activations and models to persistent storage
2. **Monitor GPU Memory**: Use `nvidia-smi` to monitor VRAM usage
3. **Batch Processing**: Process activations in batches to avoid OOM errors
4. **Save Checkpoints**: Save analysis results frequently
5. **Use Quantization**: Always use 8-bit or 4-bit quantization for large models

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch size
- Enable 4-bit quantization instead of 8-bit
- Process fewer texts at once
- Use gradient checkpointing

### Disk Space Issues
- Clean up old model checkpoints
- Use model caching efficiently
- Save only essential activations

### Slow Performance
- Ensure you're using GPU (not CPU)
- Check CUDA version compatibility
- Use appropriate batch sizes

## Next Steps

After running on RunPod:
1. Download the pruned model
2. Test on your mobile device
3. Fine-tune further if needed
4. Deploy to production

