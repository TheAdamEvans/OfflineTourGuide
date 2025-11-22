# OfflineTourGuide

Model to generate guided experiences from facts and figures encoded in the weights directly.

GPS coordinates provided by the user are mapped into Plus Codes, then the model learns to be a tour guide for that location (given the plus code). 

**Example Plus Code:** `JJXX+HR8, Seattle`

## How It Works

GPS coordinates are converted to Plus Codes using the Open Location Code library:

```python
import openlocationcode as olc

# Define the latitude and longitude
latitude = 34.43125
longitude = 8.77625

# Encode the coordinates into a Plus Code
# You can specify the desired length (e.g., 10 for ~14x14 meter area)
plus_code = olc.encode(latitude, longitude, codeLength=10)

print(f"The Plus Code is: {plus_code}")
```

## Running on RunPod

This guide provides step-by-step instructions to execute activation-based pruning on a RunPod GPU instance.

### Prerequisites

1. **RunPod GPU Instance** with:
   - At least 24GB VRAM (for Qwen2.5-7B) or 80GB+ VRAM (for Qwen3-32B)
   - At least 100GB free disk space
   - Linux environment (required for bitsandbytes quantization)

2. **Model Access**: Ensure you have access to download Qwen models from HuggingFace

### Step 1: Deploy and Connect to RunPod Instance

1. Go to [RunPod dashboard](https://www.runpod.io/)
2. Deploy a GPU pod (e.g., RTX 4090 24GB, A100 40GB/80GB)
3. Choose a template with Python 3.12+ and CUDA support
4. Connect to your pod via SSH or Jupyter

### Step 2: Install Dependencies

```bash
# Navigate to workspace
cd /workspace

# Clone your project (or upload via RunPod's file manager)
git clone <your-repo-url> OfflineTourGuide
# OR upload your project files via RunPod's file manager

cd OfflineTourGuide

# Install project dependencies
uv sync

# Install quantization support (required for memory-efficient loading)
uv pip install bitsandbytes

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 3: Prepare Activation Texts

Create a file with texts that represent your use case. These will be used to record activations:

```bash
# Create activation texts file
cat > activation_texts.txt << 'EOF'
You are an engaging, extremely knowledgeable tour guide.
Generate a detailed tour guide description for Sydney Opera House.
This tour group is interested in architecture and history.
Create a brief tour guide description for Bondi Beach.
The tour group wants to learn about local culture and food.
Generate a tour guide description for the Great Barrier Reef.
This group is interested in marine biology and conservation.
Create a detailed description for Uluru (Ayers Rock).
The tour group wants to learn about Indigenous culture.
Generate a tour guide description for Melbourne's laneways.
This group is interested in street art and coffee culture.
EOF
```

Or use a Python script to generate more diverse texts:

```python
# prepare_activation_texts.py
activation_texts = [
    "You are an engaging, extremely knowledgeable tour guide.",
    "Generate a detailed tour guide description for Sydney Opera House.",
    "This tour group is interested in architecture and history.",
    # Add more texts relevant to your domain
] * 20  # Repeat for more data points

with open('activation_texts.txt', 'w') as f:
    for text in activation_texts:
        f.write(text + '\n')
```

### Step 4: Record Activations (Source of Truth)

**This is the canonical way to record activations:**

```bash
uv run python - <<'PY'
from model.activation_recorder import load_model_for_activation_recording, record_activations_from_dataset

model, tokenizer = load_model_for_activation_recording(
    model_name="Qwen/Qwen2.5-7B-Instruct",  # swap in your larger checkpoint if desired
    device="cuda",
    load_in_8bit=True
)

record_activations_from_dataset(
    model=model,
    tokenizer=tokenizer,
    texts=None,                       # force it to read from /samples
    sample_dir="samples",
    save_dir="pruning_output/activations",
    max_length=512
)
PY
```

### Step 5: Execute Pruning - Quick Example

For a quick test run with the example script:

```bash
# Run the example script (uses default settings)
uv run python -m model.example_pruning
```

**Note**: The example script uses hardcoded texts. For production, use the full pipeline (Step 6).

### Step 6: Execute Pruning - Full Pipeline

For production pruning with custom configuration:

#### Option A: Modify the Pipeline Script Directly

Edit `model/pruning_pipeline.py` and update the `__main__` section:

```python
# At the bottom of model/pruning_pipeline.py
if __name__ == "__main__":
    # Load your activation texts
    with open('activation_texts.txt', 'r') as f:
        activation_texts = [line.strip() for line in f if line.strip()]
    
    run_pruning_pipeline(
        model_name="Qwen/Qwen3-32B",  # Your target model
        activation_texts=activation_texts,
        training_data_file=None,  # Optional: path to training_data.jsonl
        output_dir="pruning_output",
        pruning_ratio=0.3,  # Prune 30% of the model
        num_epochs=3,
        device="cuda",
        load_in_8bit=True,  # Enable 8-bit quantization to save memory
        load_in_4bit=False  # Use 4-bit for even more memory savings if needed
    )
```

Then run:
```bash
python -m model.pruning_pipeline
```

#### Option B: Use Python API Directly

Create a custom script:

```python
# run_pruning.py
from model import run_pruning_pipeline

# Load activation texts
with open('activation_texts.txt', 'r') as f:
    activation_texts = [line.strip() for line in f if line.strip()]

# Execute pruning
run_pruning_pipeline(
    model_name="Qwen/Qwen3-32B",
    activation_texts=activation_texts,
    training_data_file=None,  # Optional: "path/to/training_data.jsonl"
    output_dir="pruning_output",
    pruning_ratio=0.3,
    num_epochs=3,
    device="cuda",
    load_in_8bit=True,
    load_in_4bit=False
)
```

Run it:
```bash
python run_pruning.py
```

### Step 7: Monitor Progress

The pipeline will output progress for each step:

1. **Loading model** - Downloads model if first time (~30-60GB)
2. **Recording activations** - Processes your texts and saves activations
3. **Analyzing activations** - Computes importance scores
4. **Pruning model** - Removes less important layers/neurons
5. **Fine-tuning** (if training data provided) - Recovers performance

Monitor GPU memory:
```bash
# In another terminal or use RunPod's monitoring
watch -n 1 nvidia-smi
```

### Step 8: Retrieve Results

After completion, your outputs will be in the `pruning_output` directory:

```bash
ls -lh pruning_output/
# You should see:
# - activations/          # Recorded activation data
# - analysis.json         # Pruning analysis results
# - pruned_model.pt       # The pruned model
# - fine_tuned_model.pt   # Fine-tuned model (if training data was provided)
```

Download results via RunPod's file manager or SCP:
```bash
# From your local machine
scp -r runpod-pod-id:/workspace/OfflineTourGuide/pruning_output ./local_output/
```

## Configuration Options

### Model Selection

Choose based on your GPU VRAM:

| Model | VRAM Required (8-bit) | VRAM Required (4-bit) | Disk Space |
|-------|----------------------|----------------------|------------|
| Qwen2.5-7B-Instruct | ~14GB | ~8GB | ~15GB |
| Qwen3-32B | ~32GB | ~16GB | ~60GB |

### Memory Configuration

For different GPU sizes:

**24GB GPU (RTX 4090, A6000):**
```python
load_in_8bit=True,   # Use 8-bit quantization
load_in_4bit=False,
batch_size=1,
max_length=512
```

**40GB GPU (A100 40GB):**
```python
load_in_8bit=True,   # Can use 8-bit for Qwen3-32B
load_in_4bit=False,
batch_size=2,        # Can use larger batches
max_length=512
```

**80GB GPU (A100 80GB):**
```python
load_in_8bit=False,  # Can use FP16 for better quality
load_in_4bit=False,
batch_size=4,        # Larger batches possible
max_length=1024      # Longer sequences
```

### Pruning Ratio

- **0.2 (20%)**: Conservative, minimal quality loss
- **0.3 (30%)**: Balanced (recommended)
- **0.5 (50%)**: Aggressive, may need more fine-tuning

## What Happens During Execution

The pipeline executes these steps automatically:

1. **Model Loading** (~5-10 minutes first time)
   - Downloads model from HuggingFace if not cached
   - Loads with quantization if specified
   - Verifies GPU availability

2. **Activation Recording** (~10-30 minutes)
   - Processes your activation texts through the model
   - Captures intermediate layer activations
   - Saves to disk (~5-20GB depending on model size)

3. **Activation Analysis** (~5-15 minutes)
   - Computes importance scores for each layer/neuron
   - Identifies pruning candidates based on low activation
   - Generates analysis report

4. **Model Pruning** (~5-10 minutes)
   - Removes identified layers/neurons
   - Reconstructs model architecture
   - Saves pruned model

5. **Fine-tuning** (optional, ~30-60 minutes per epoch)
   - Trains pruned model on your data
   - Recovers performance lost from pruning
   - Saves checkpoints

## Best Practices

### Storage Management

1. **Use Persistent Storage**: Configure RunPod to use persistent storage for:
   - Model cache (`/workspace/downloads/` - models are automatically downloaded here)
   - Output directory (`pruning_output/`)
   - This prevents re-downloading models on pod restart

   **Note**: The Qwen3-32B model files are automatically downloaded to `/workspace/downloads/` when you run the pruning pipeline. Make sure to mount a persistent volume at `/workspace/downloads` to avoid re-downloading models on pod restart.

2. **Monitor Resources**: Keep an eye on:
   ```bash
   # GPU memory
   watch -n 1 nvidia-smi
   
   # Disk space
   df -h /workspace
   ```

3. **Save Progress**: The pipeline saves checkpoints automatically, but you can also:
   - Save activation files separately
   - Export analysis.json for later review
   - Keep model checkpoints if fine-tuning

### Performance Optimization

1. **Batch Size**: Start with `batch_size=1`, increase if you have VRAM headroom
2. **Sequence Length**: Use `max_length=512` for activation recording (can be shorter)
3. **Quantization**: Always enable 8-bit or 4-bit for models >7B parameters
4. **Text Diversity**: Use diverse, domain-relevant texts for better pruning decisions

## Troubleshooting

### Out of Memory (OOM) Error

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# Option 1: Enable 4-bit quantization
load_in_4bit=True,
load_in_8bit=False,

# Option 2: Reduce batch size
batch_size=1,  # Already minimum, but verify

# Option 3: Reduce sequence length
max_length=256,  # Instead of 512

# Option 4: Process fewer texts
activation_texts = activation_texts[:50]  # Limit to 50 texts
```

### Disk Space Issues

**Symptoms**: `No space left on device` during model download

**Solutions**:
```bash
# Check disk usage
df -h

# Clean up old models
rm -rf ~/.cache/huggingface/hub/models--*  # Remove unused models

# Use smaller model for testing
model_name="Qwen/Qwen2.5-7B-Instruct"  # Instead of 32B
```

### Slow Performance

**Symptoms**: Processing takes much longer than expected

**Check**:
```python
# Verify GPU is being used
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show your GPU

# Check CUDA version
python -c "import torch; print(torch.version.cuda)"
```

**Solutions**:
- Ensure `device="cuda"` is set
- Verify CUDA drivers are installed on RunPod
- Use appropriate batch sizes (not too small, not too large)

### Model Download Fails

**Symptoms**: `ConnectionError` or timeout during model download

**Solutions**:
```bash
# Models are automatically downloaded to /workspace/downloads
# Ensure this directory exists and has sufficient space:
mkdir -p /workspace/downloads
df -h /workspace/downloads  # Check available space

# Or use a mirror (if available)
export HF_ENDPOINT=https://hf-mirror.com
```

### Activation Recording Fails

**Symptoms**: Error during activation recording step

**Solutions**:
- Verify activation texts are not empty
- Check that texts are strings, not other types
- Ensure sufficient disk space for activations
- Try with fewer texts first to isolate the issue

## Verifying Results

After execution completes, verify the outputs:

```bash
# Check outputs exist
ls -lh pruning_output/

# View analysis results
cat pruning_output/analysis.json | python -m json.tool | head -50

# Check model file size
ls -lh pruning_output/pruned_model.pt

# Verify model can be loaded
python -c "
import torch
model = torch.load('pruning_output/pruned_model.pt')
print(f'Model loaded: {type(model)}')
"
```

## Next Steps After Pruning

1. **Download Results**: Use RunPod file manager or SCP to download:
   - `pruned_model.pt` - The pruned model
   - `analysis.json` - Pruning analysis for review
   - `activations/` - Activation data (optional, large files)

2. **Test Locally**: Load and test the pruned model:
   ```python
   import torch
   model = torch.load('pruned_model.pt')
   # Test inference
   ```

3. **Fine-tune Further**: If quality is insufficient:
   - Prepare training data in JSONL format
   - Re-run pipeline with `training_data_file` parameter
   - Increase `num_epochs` if needed

4. **Convert for Mobile**: 
   - Convert to ONNX or other mobile-friendly format
   - Quantize further if needed
   - Test on target device

5. **Deploy**: Integrate pruned model into your application

## Additional Documentation

For more detailed information, see:

- **[model/PRUNING_GUIDE.md](model/PRUNING_GUIDE.md)** - Detailed pruning concepts and workflows
- **[model/ACTIVATION_PRUNING_FAQ.md](model/ACTIVATION_PRUNING_FAQ.md)** - Common questions and answers
- **[PLAN.md](PLAN.md)** - Project plan and roadmap

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
- bitsandbytes for quantization (Linux/Windows only)
