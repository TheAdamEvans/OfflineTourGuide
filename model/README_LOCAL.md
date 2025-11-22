# Running Activation-Based Pruning Locally

This guide explains how to run the activation-based pruning pipeline on your local machine for testing and development.

## Prerequisites

### System Requirements

- **Python 3.12+**
- **Disk Space**: At least 10-20GB free (for model downloads and activations)
- **RAM**: 16GB+ recommended (8GB minimum)
- **GPU** (optional but recommended): CUDA-compatible GPU with 8GB+ VRAM
  - Without GPU, models will run on CPU (much slower)

### Operating System

- **macOS**: Works, but quantization (bitsandbytes) is not available
- **Linux**: Full support including quantization
- **Windows**: Works, but quantization support may vary

## Installation

### Step 1: Install Dependencies

```bash
# Navigate to project root
cd /path/to/OfflineTourGuide

# Install project dependencies
uv sync
# or
pip install -e .
```

### Step 2: Install Optional Dependencies (Linux/Windows only)

For quantization support (saves memory):

```bash
# Linux/Windows only - not available on macOS
pip install bitsandbytes
```

**Note**: On macOS, quantization is not available. Models will use FP16/FP32 instead.

## Quick Start

### Basic Example

Run the example script with a small model:

```bash
python -m model.example_pruning
```

This will:
1. Download Qwen2.5-0.5B-Instruct (~1GB) - first time only
2. Record activations from sample texts
3. Analyze activations
4. Identify pruning candidates

### Using a Different Model

Edit `model/example_pruning.py`:

```python
# Change the model name
model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # Larger model
# or
model_name = "Qwen/Qwen2.5-7B-Instruct"  # Much larger (~14GB)
```

**Warning**: Larger models require more disk space and memory!

## Configuration Options

### Model Selection

Choose a model based on your resources:

| Model | Size | Disk Space | RAM/VRAM | Notes |
|-------|------|------------|----------|-------|
| Qwen2.5-0.5B | ~1GB | 2GB | 4GB | Good for testing |
| Qwen2.5-1.5B | ~3GB | 5GB | 8GB | Moderate testing |
| Qwen2.5-7B | ~14GB | 20GB | 16GB+ | Requires GPU |
| Qwen3-32B | ~64GB | 100GB+ | 80GB+ VRAM | RunPod only |

### Enable Quantization (Linux/Windows with GPU)

In `model/example_pruning.py`:

```python
model, tokenizer = load_model_for_activation_recording(
    model_name=model_name,
    device="cuda",
    torch_dtype=torch.float16,
    load_in_8bit=True,   # Enable 8-bit quantization
    load_in_4bit=False   # Or use 4-bit for even more savings
)
```

**Benefits**:
- 8-bit: ~50% memory reduction
- 4-bit: ~75% memory reduction

**Limitations**:
- Slightly slower inference
- Small accuracy loss (usually negligible)

## Running the Full Pipeline

### Step 1: Prepare Activation Texts

Create a list of diverse texts from your domain:

```python
activation_texts = [
    "You are an engaging, extremely knowledgeable tour guide.",
    "Generate a detailed tour guide description for Sydney Opera House.",
    "This tour group is interested in architecture and history.",
    # ... more diverse examples
] * 50  # Repeat for more data points
```

### Step 2: Run the Pipeline

```python
from model import run_pruning_pipeline

run_pruning_pipeline(
    activation_texts=activation_texts,
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    training_data_file="training_data.jsonl",  # Optional
    output_dir="pruning_output",
    pruning_ratio=0.3,
    num_epochs=3
)
```

Or use the command line:

```bash
python -c "
from model import run_pruning_pipeline

activation_texts = [
    'You are an engaging tour guide.',
    'Generate a tour guide description.',
] * 20

run_pruning_pipeline(
    activation_texts=activation_texts,
    output_dir='pruning_output'
)
"
```

## Output Files

After running, you'll find:

```
pruning_output/
├── activations/
│   └── activations.pkl          # Recorded activations
├── analysis.json                # Analysis results
├── pruned_model.pt              # Pruned model (if pruning succeeds)
└── checkpoints/                 # Fine-tuning checkpoints (if fine-tuning)
    └── checkpoint_epoch_*.pt
```

## Troubleshooting

### Out of Disk Space

**Error**: `No space left on device` or `free disk space`

**Solutions**:
1. Free up disk space (need 2-20GB depending on model)
2. Use a smaller model for testing
3. Clean HuggingFace cache: `rm -rf ~/.cache/huggingface/hub/`
4. Set custom cache directory with smaller models

### Out of Memory (OOM)

**Error**: `CUDA out of memory` or system becomes unresponsive

**Solutions**:
1. Use a smaller model
2. Enable quantization (8-bit or 4-bit)
3. Reduce batch size in `record_activations_from_dataset`
4. Process fewer texts at once
5. Use CPU instead of GPU (slower but uses RAM)

### Slow Performance

**Issue**: Model runs very slowly

**Solutions**:
1. Use GPU instead of CPU (10-100x faster)
2. Use a smaller model
3. Reduce `max_length` parameter
4. Process in smaller batches

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'model'`

**Solution**: Run from project root or use:
```bash
python -m model.example_pruning
```

### Quantization Not Available (macOS)

**Error**: `bitsandbytes` installation fails

**Solution**: This is expected on macOS. The script will automatically skip quantization and use FP16/FP32 instead. Performance will be slower and memory usage higher.

## Local vs RunPod Differences

| Feature | Local | RunPod |
|---------|-------|--------|
| Quantization | macOS: No<br>Linux: Yes | Yes |
| GPU Required | Optional | Recommended |
| Disk Space | Limited | Abundant |
| Model Size | Small-Medium | Large (32B+) |
| Best For | Testing, Development | Production, Large Models |

## Example Workflow

### 1. Test Locally with Small Model

```bash
# Quick test
python -m model.example_pruning
```

### 2. Analyze Results

```python
import json

with open('pruning_example_output/analysis.json') as f:
    analysis = json.load(f)

# View layer importance scores
for layer, metrics in list(analysis.items())[:10]:
    print(f"{layer}: importance={metrics.get('importance', 0):.4f}")
```

### 3. Adapt for Your Model

Edit `model/pruner.py` to match your model's architecture (Qwen3-32B).

### 4. Run on RunPod

Once validated locally, deploy to RunPod for:
- Larger models (32B+)
- Faster processing
- More disk space
- Better GPU resources

See `RUNPOD_SETUP.md` for RunPod instructions.

## Memory-Saving Tips

1. **Use Quantization**: 8-bit or 4-bit (Linux/Windows only)
2. **Smaller Models**: Start with 0.5B or 1.5B
3. **Process in Batches**: Don't load all data at once
4. **Clear Cache**: `torch.cuda.empty_cache()` between operations
5. **Use CPU**: If GPU memory is limited, use CPU (slower but works)

## Next Steps

1. **Test Locally**: Run with small model to validate pipeline
2. **Review Analysis**: Check `analysis.json` for pruning insights
3. **Adapt Pruning Logic**: Customize for Qwen3-32B architecture
4. **Deploy to RunPod**: Run full pipeline on RunPod with large model
5. **Fine-tune**: Use your training data to recover performance

## Getting Help

- Check `PRUNING_GUIDE.md` for detailed pruning concepts
- See `ACTIVATION_PRUNING_FAQ.md` for common questions
- Review `RUNPOD_SETUP.md` for production deployment

## Example: Complete Local Run

```bash
# 1. Install dependencies
uv sync

# 2. Run example (downloads model first time)
python -m model.example_pruning

# 3. Check outputs
ls -lh pruning_example_output/

# 4. View analysis
cat pruning_example_output/analysis.json | python -m json.tool | head -50
```

## Notes

- **First Run**: Model download can take 10-30 minutes depending on connection
- **Subsequent Runs**: Much faster (model cached locally)
- **CPU Mode**: Works but very slow (expect 10-100x slower than GPU)
- **macOS**: Quantization not available, but everything else works

For production use with large models (32B+), see `RUNPOD_SETUP.md`.

