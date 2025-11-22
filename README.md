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

The script will:

1. Load the requested HuggingFace model.
2. Register forward hooks on every transformer block.
3. Run each sample text through the model.
4. Save a ``sample_XXXX.pt`` file containing the text, token ids, logits, and
   raw activations per layer.

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

You can also analyze an existing dump directly:

```python
from model.activation_analyzer import summarize_activation_file, format_summary_table

stats = summarize_activation_file("activations/sample_0001.pt")
print(format_summary_table(stats))
```

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

That is the entire surface area—record and inspect activations without any
pruning or fine-tuning extras.

