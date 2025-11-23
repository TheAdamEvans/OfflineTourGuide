# OfflineTourGuide (Activation Dumps)

The previous pruning and finetuning experiments have been removed. What remains
is a small, dependency-light script that helps capture transformer activations
for a handful of tour-stop style prompts so you can inspect them offline.

## Quick start

```bash
uv sync

# Dumps to ./activations by default using the sample texts in ./samples
uv run python -m data_extraction.dump_activations --model Qwen/Qwen3-7B
```

### Persistent Hugging Face cache

All shells should write Hugging Face artifacts to `/workspace/.hf_home` so
large checkpoints survive pod restarts and don't fill ephemeral disks. Add the
exports once (e.g., append to `~/.bashrc` or your RunPod startup script) and
reload your shell:

```bash
mkdir -p /workspace/.hf_home/hub
echo 'export HF_HOME=/workspace/.hf_home' >> ~/.bashrc
echo 'export HF_HUB_CACHE=/workspace/.hf_home/hub' >> ~/.bashrc
source ~/.bashrc

# verification
env | grep HF_
ls -lh /workspace/.hf_home
```

The script will:

1. Load the requested HuggingFace model.
2. Register forward hooks on every transformer block.
3. Run each sample text through the model.
4. Save each sample as a shard (``activations/sample_XXXX.pt`` + a row in
   ``activations/metadata.jsonl``) that includes the text, token ids, logits,
   and raw activations per layer alongside token-span + checksum metadata so
   downstream rotation/permutation tooling can stream the data.

> Every shard recorded through ``ActivationShardWriter`` gets an entry in
> ``metadata.jsonl`` describing the token range, covered layers, tensor dtype,
> and SHA256 checksum. This JSONL file is what the rotation CLI parses later
> to solve PCA/Procrustes transports layer by layer.

## Custom inputs

- Add or update ``.txt`` files in ``./samples``.
- Provide one-off prompts:

  ```bash
  uv run python -m data_extraction.dump_activations \
    --text "Describe the Sydney Opera House for architecture fans." \
    --text "Give a short Bondi Beach blurb for food lovers."
  ```

- Point at a newline-delimited file:

  ```bash
  uv run python -m data_extraction.dump_activations --text-file my_prompts.txt
  ```

## Basic visibility checks

Pass ``--analyze`` to print per-layer statistics right after recording. Each
metric is intentionally simple (mean / max absolute activation) so you can get
a quick feel for which layers are lighting up without running a heavier
pipeline.

```
uv run python -m data_extraction.dump_activations --analyze
```

You can also analyze an existing shard directly:

```python
from model.activation_analyzer import summarize_activation_file, format_summary_table

stats = summarize_activation_file("activations/sample_0001.pt")
print(format_summary_table(stats))
```

## Rotation diagnostics

Once you have both student and teacher activation shards recorded (with their
`metadata.jsonl` files), run the rotation sanity check to estimate PCA +
Procrustes transports per layer and append cosine diagnostics to a ledger:

```
uv run python -m transport.rotation_cli \
  --student-index runs/run_x/activations/student/metadata.jsonl \
  --teacher-index runs/run_x/activations/teacher/metadata.jsonl \
  --layer model.layers.0 \
  --layer model.layers.1 \
  --ledger runs/run_x/rotations.jsonl
```

Each CLI invocation computes the before/after cosine similarity across all
tokens captured for the requested layer(s), logs singular values, and appends
the summary to the rotation ledger so you can track alignment progress without
loading the checkpoints themselves.

## Fold transport tensors into a checkpoint

After solving rotations (and optional permutations), describe them in a JSON
manifest and let the folding CLI update a checkpoint in-place:

```
uv run python -m transport.apply_transforms \
  --checkpoint Qwen/Qwen2.5-7B-Instruct \
  --output-dir runs/run_x/checkpoints/rotated \
  --spec runs/run_x/transforms.json
```

Sample `transforms.json`:

```
{
  "layers": [
    {
      "name": "model.layers.0",
      "rotation_path": "runs/run_x/rotations/model.layers.0.pt"
    },
    {
      "name": "model.layers.1",
      "rotation_path": "runs/run_x/rotations/model.layers.1.pt",
      "head_permutation_path": "runs/run_x/permutations/heads_1.pt",
      "neuron_permutation_path": "runs/run_x/permutations/neurons_1.pt",
      "head_dim": 128
    }
  ]
}
```

The CLI loads the checkpoint (any HuggingFace model id or local directory),
folds each transform via `WeightTransformSet`, and writes the updated weights +
tokenizer to `--output-dir`.

## Run scaffolding & dataset ledger

Before GPU time is available you can still prepare a full `runs/<run_id>` shell
plus a frozen snapshot of the ~200 blurbs and prompt/style files:

```
uv run python -m pipeline.run_scaffolder \
  --runs-root runs \
  --prompts-dir prompts \
  --samples-dir samples
```

This command will:

1. Hash every prompt/style file in `prompts/`.
2. Hash every `.txt` sample in `samples/` to form a dataset ledger.
3. Create `runs/<run_id>/manifest.json`, `dataset_snapshot.json`, `rotations.jsonl`,
   and subdirectories for checkpoints/activations/logs.

You can override any prompt list with `--prompt path/to/file.md`, set specific run
ids via `--run-id`, and record tokenizer + commit metadata with
`--tokenizer-name`, `--tokenizer-version`, `--teacher-commit`, and
`--student-commit`.

## Useful flags

```
python -m data_extraction.dump_activations --help
```

- ``--model`` – HuggingFace id to load (default: ``Qwen/Qwen3-7B``)
- ``--dtype`` – Weight precision (float16 / bfloat16 / float32)
- ``--device`` – Torch device string (defaults to auto)
- ``--output-dir`` – Where ``sample_XXXX.pt`` files are stored
- ``--layer`` – Optional module names if you only want specific hooks
- ``--max-length`` – Token limit while tokenizing prompts
- ``--backend`` – ``torch`` (default) or ``vllm`` for GPU-friendly capture
- ``--hf-home`` / ``--hf-hub-cache`` – Override the persistent HF cache paths if needed
- ``--vllm-*`` – Tensor parallel size, GPU utilization cap, max context, eager toggle

That is the entire surface area—record and inspect activations without any
pruning or fine-tuning extras.

## GPU activation capture with vLLM

The vLLM backend stands up the reference `Qwen/Qwen3-32B` checkpoint on the RTX 6000
Ada, registers the same forward hooks, and streams shards directly to the persistent
workspace cache. Sample run using the bilingual canned prompts:

```bash
uv run python -m data_extraction.dump_activations \
  --backend vllm \
  --model Qwen/Qwen3-32B \
  --dtype float16 \
  --text-file samples/generic_en.txt \
  --text-file samples/generic_zh.txt \
  --output-dir activations/qwen3_32b_gpu \
  --max-length 640 \
  --analyze
```

Validation checklist:

- `ls -lh /workspace/.hf_home` – proves Hugging Face downloads land on the workspace disk.
- `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv` – shows the GPU is driving the run.
- The `--analyze` table shouts per-layer magnitudes immediately after capture.

All shards still include `metadata.jsonl` entries, so downstream rotation / transport tooling
continues to work without changes.

## QA toggles, metrics, and reporting

- Configure which transport modules/metrics are enabled via `config/eval_template.json`
  (copy per run and edit checkpoint paths, variant toggles, etc.). The dataclasses live
  in `config/evaluation.py` and can be loaded from JSON or TOML.
- The notebook `notebooks/qa_toggle_metrics.ipynb` consumes that config, iterates
  over checkpoints, and writes placeholder metric rows (`logit_kl`, `hidden_state_cosine`,
  `residual_rms`, `surprisal`) to `runs/<run_id>/logs/metrics.jsonl`. Replace the stub
  helpers with real evaluations once activations are ready.
- Generate polished QA summaries (tables + seaborn plots) by running:

  ```bash
  uv run python -m pipeline.report_generator \
    --run-id run_YYYYMMDD \
    --runs-root runs \
    --mock  # drop this flag once real metrics exist
  ```

  The script ingests `manifest.json` + metrics JSONL, renders a Markdown report, and saves
  metric plots to `runs/<run_id>/figures/metric_trends.png`.


