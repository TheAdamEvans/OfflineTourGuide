# Offline Tour Guide - Model Compression Hackathon Plan

> **Note:** This file documents the original compression project. The active
> codebase has since been stripped down to a simple activation dumping tool, so
> most of the steps below are preserved only for historical context.

**Goal:** Compress Qwen 3-32B FP8 → 3B using **activation-guided pruning** for POI descriptions using Plus Codes and Smantic Geographic coding.

**Approach:** Task-aware layer deletion + width pruning (vs. traditional distillation)

**Timeline:** Saturday afternoon/evening → Demo Sunday 2pm

## Phase 1: Data Generation Pipeline

### Strong Model Generation (Claude/GPT)
- [ ] Build generation pipeline:
  ```
  Input: (plus_code, interest_tags, style_tag)
  → Claude/GPT prompt
  → Generated POI description
  ```
- [ ] Generate initial dataset (2 examples for rapid iteration)
- [ ] **TODO:** Decide on sampling strategy for accuracy
- [ ] Save as "{plus_code}.txt"

## Phase 2: Activation Harvesting (Saturday Afternoon) - **Priority 1**

### 2.1 Qwen 3-32B FP8 Infrastructure
- [ ] Qwen 3-32B FP8 already deployed on RunPod with vLLM
- [ ] Implement forward hooks to log ALL layer outputs:
  - **Architecture:** 64 layers, 5120 hidden dim
  - Log hidden states at each layer output (post-attention, post-FFN)
  - **Storage per 256 tokens:** ~82 MB (64 layers × 256 tokens × 5120 × 1 byte FP8)
- [ ] Test throughput with a small number of "tour stop" generations

### 2.2 Task-Specific Activation Database
- [ ] Run generated Plus Code dataset through 32B teacher
- [ ] Save layer outputs in efficient format (PyTorch tensors/HDF5)
- [ ] Format: `{input_tokens, layer_outputs[64], output_tokens}`
- [ ] Build analysis toolkit:
  - Compute layer-wise cosine similarity matrices
  - Calculate per-layer activation variance/magnitude
  - Identify redundant/similar layers for pruning

**Goal:** Understand which of the 64 layers are critical for the tour guide task

## Phase 3: Evaluation Strategy for Tour Guide Model Pruning

### 3.1 Core Concept
- [ ] Treat evaluation as a direct measurement of how pruning impacts the model's ability to behave like a tour guide.
- [ ] Compare progressively pruned checkpoints (layer deletion + width reduction) against a high-quality, human-written corpus of tour descriptions.

### 3.2 Key Metric: Surprisal on Ground Truth
- [ ] Feed ground truth tour narratives into each model checkpoint instead of sampling generations.
- [ ] Record token-level surprisal/perplexity and track how the metric drifts as pruning intensifies.
- [ ] Use surprisal because it reflects whether the model still assigns high probability to canonical tour-guide phrasing and factual statements.

### 3.3 Test Set Design
- [ ] `seen_cities/` – fresh landmarks from cities already used in training (Sydney, Xian).
- [ ] `unseen_similar/` – culturally or geographically adjacent cities (Melbourne, Beijing).
- [ ] `unseen_different/` – cities with very different styles or context (New York, Barcelona).
- [ ] `edge_cases/` – challenging architectural styles or narratives (Reykjavik, Dubai).

### 3.4 What We're Looking For
- [ ] Acceptable: small surprisal bumps on common structure/function words while factual tokens remain stable.
- [ ] Unacceptable: sharp spikes on architectural terminology (e.g., "Romanesque Revival"), proper nouns ("George McRae"), numbers/dates ("1898"), or city-specific cues.

### 3.5 Decision Criteria
- [ ] Pass if mean perplexity increase < 20% on `seen_cities/`.
- [ ] Pass if maximum surprisal on factual tokens is < 2× the baseline model.
- [ ] Fail immediately if any file shows catastrophic loss (perplexity > 1000).

### 3.6 Why This Works
- [ ] Efficient: avoids subjective human rating loops while still using human-written references.
- [ ] Targeted: directly measures retention of tour-guide language and factual grounding.
- [ ] Granular: token-level surprisal highlights whether style vs. knowledge degraded.
- [ ] Actionable: high surprisal on specific tokens guides which layers/neurons should be restored.
- [ ] Philosophy: a viable pruned model should not be "surprised" by standard tour-guide narratives; if it is, pruning went too far.


## Phase 4: Task-Aware Layer Deletion (Saturday Evening) - **Priority 1 / Main Contribution**
### 3.0 Token Importance Analysis
- [ ] Calculate entropy for each token as the model predicts it
- [ ] Assign a higher weight to high entropy tokens

### 4.1 Layer Redundancy Analysis
- [ ] Using activation database from Phase 2:
  - Compute cosine similarity between consecutive layers
  - Identify "redundant" layers
  - Calculate per-layer importance scores:
    - Activation magnitudes across examples
    - Activation variance across examples
  - Calculate per-layer importance scores including token weight

### 4.2 Strategic Layer Pruning
- [ ] Delete layers to go from 64 → ~28 layers (to match 3B architecture roughly)
- [ ] **Novel approach:** Use task-specific activation patterns to decide which layers to cut
  - Approach 1: U-shaped Chop
    - Keep early layers (basic language understanding)
    - Prune middle layers with high similarity (redundant processing)
    - Keep final layers (output formatting, tour guide specifics)
  - Approach 2: Alternating layer Chop
    - As a baseline, delete every other layer without regard for activation

### 4.3 Validation & Analysis
- [ ] Test chopped model on held-out examples
- [ ] Check if it still generates coherent output (even if degraded)
- [ ] **This is the main contribution:** Demonstrate that task-aware layer deletion preserves performance better than blind pruning
- [ ] Log examples: compare 32B → chopped model outputs
- [ ] Document which layers were kept/deleted and why

**Goal:** Show that activation-guided pruning beats random layer deletion

## Phase 5: Width Pruning Exploration (Saturday Late/Sunday Morning) - **Priority 2 / If Time Allows**

### 5.1 Neuron-Level Importance Scoring
- [ ] Using activation database from Phase 2:
  - Rank attention heads by average activation magnitude (64 heads per layer)
  - Rank FFN neurons by variance across examples (25,600 dims per layer)
  - Identify low-importance neurons/heads for pruning

### 5.2 Structured Width Reduction
- [ ] **Goal:** Aggressive width pruning (~60% reduction) guided by activations
- [ ] Prune attention heads:
  - Keep top ~40% of heads per layer based on importance scores
  - Adjust attention computation for smaller head count
- [ ] Prune FFN dimensions:
  - Keep top ~40% of intermediate FFN neurons
  - Reduce hidden dimension size accordingly (5120 → ~2048)
- [ ] Implementation challenges:
  - Weight matrix surgery (removing rows/columns)
  - Maintaining architectural consistency
  - Potential need for calibration/fine-tuning

### 5.3 Model Averaging
- [ ] We will break the model entirely at 60%+ reduction
  - Could we simply average the new model in with the old model

**Goal:** Explore how far we can push width reduction using activation-guided pruning

## Phase 6: Inference Server & Demo (Sunday Afternoon)

### 6.1 Presentation
- Slide 1: Offline Tour Guide
- Slide 2: Data Generation
- Slide 3: Evaluation
- Slide 4: Method: Model Chopping
- Slide 5: Results

### 6.2 Web Demo
- [ ] Simple web interface:
  - Map view (Leaflet/Mapbox)
  - Click to drop pin
  - Interest tag checkboxes
  - Style selector (brief/stimulate/detailed)
  - Display generated tour guide text
- [ ] Pre-generate some example queries for demo stability
- [ ] Try for cities that the model hasn't seen, like New York

---

## Open Questions / TODO

### Data Generation Details
- **Sampling density formula:** Population-weighted? POI count-weighted?
- **Geographic scope:** Just Sydney CBD? Include suburbs? How far out?
- **Interest distribution:** Equal sampling or weight toward common interests?
- **Validation strategy:** How do we verify Claude/GPT outputs are accurate?

### Model Compression Specifics
- **Layer similarity metric:** Cosine similarity? CKA? Other?
- **How many layers to keep:** 28? 30? 32? (depends on similarity analysis)
- **Width pruning feasibility:** Can we go 60%+ reduction without fine-tuning?
- **Quantization:** Do we quantize the chopped model for faster inference?

### Immediate TODOs
- Extract data about locations in Sydney using RunPod API with Qwen3-32B FP8
- Implement forward hooks in vLLM to capture layer activations (~82 MB per 256 tokens)
- Confirm activation logging doesn't slow down generation too much
- Wire the local activation dumper to `transport.ActivationShardWriter` so capture jobs emit shard metadata (token spans, checksums) compatible with the rotation pipeline.
- Once the Python toolchain on the run pod has `pytest` available, run `pytest tests/test_rotations.py` and the `data_extraction.dump_activations --analyze` CLI to validate the synthetic rotation/permutation harness plus new shard format before collecting GPU-scale activations.
- After both student + teacher shards exist, run `uv run python -m transport.rotation_cli --student-index ... --teacher-index ... --layer ...` to populate `runs/<run_id>/rotations.jsonl` with before/after cosine diagnostics and singular values per layer.

#### Activation capture follow-ups
- [ ] Export `HF_HOME=/workspace/.hf_home` and `HF_HUB_CACHE=/workspace/.hf_home/hub` in the startup script so large checkpoints land on the shared workspace volume.
- [ ] Switch activation runs to the reference `Qwen/Qwen3-32B` via vLLM for streaming inference, avoiding per-layer decompression OOMs.

---

## Success Criteria (Sunday 2pm)

### Must Have (Priority 1)
1. ✅ Activation database from 32B model on tour guide task
2. ✅ Task-aware layer deletion implementation (64 → ~28 layers)
3. ✅ Evidence that activation-guided pruning > random pruning
4. ✅ Working inference with chopped model (even if degraded)
5. ✅ Demo shows model still generates coherent tour descriptions

### Nice to Have (Priority 2)
6. ✅ Width pruning exploration (attention heads + FFN neurons)
7. ✅ Web interface with map + style/interest controls
8. ✅ Latency < 2 seconds per query
9. ✅ Demo on unseen cities (e.g., New York)
