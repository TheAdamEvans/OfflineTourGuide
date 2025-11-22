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

---

# Offline Tour Guide - Model Compression Hackathon Plan

**Goal:** Compress Qwen 3-32B FP8 → 3B using **activation-guided pruning** for Sydney POI descriptions using Plus Codes, controllable via style tags

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

## Phase 3: Task-Aware Layer Deletion (Saturday Evening) - **Priority 1 / Main Contribution**
### 3.0 Token Importance Analysis
- [ ] Calculate entropy for each token as the model predicts it
- [ ] Assign a higher weight to high entropy tokens

### 3.1 Layer Redundancy Analysis
- [ ] Using activation database from Phase 2:
  - Compute cosine similarity between consecutive layers
  - Identify "redundant" layers
  - Calculate per-layer importance scores:
    - Activation magnitudes across examples
    - Activation variance across examples
  - Calculate per-layer importance scores including token weight

### 3.2 Strategic Layer Pruning
- [ ] Delete layers to go from 64 → ~28 layers (to match 3B architecture roughly)
- [ ] **Novel approach:** Use task-specific activation patterns to decide which layers to cut
  - Approach 1: U-shaped Chop
    - Keep early layers (basic language understanding)
    - Prune middle layers with high similarity (redundant processing)
    - Keep final layers (output formatting, tour guide specifics)
  - Approach 2: Alternating layer Chop
    - As a baseline, delete every other layer without regard for activation

### 3.3 Validation & Analysis
- [ ] Test chopped model on held-out examples
- [ ] Check if it still generates coherent output (even if degraded)
- [ ] **This is the main contribution:** Demonstrate that task-aware layer deletion preserves performance better than blind pruning
- [ ] Log examples: compare 32B → chopped model outputs
- [ ] Document which layers were kept/deleted and why

**Goal:** Show that activation-guided pruning beats random layer deletion

## Phase 4: Width Pruning Exploration (Saturday Late/Sunday Morning) - **Priority 2 / If Time Allows**

### 4.1 Neuron-Level Importance Scoring
- [ ] Using activation database from Phase 2:
  - Rank attention heads by average activation magnitude (64 heads per layer)
  - Rank FFN neurons by variance across examples (25,600 dims per layer)
  - Identify low-importance neurons/heads for pruning

### 4.2 Structured Width Reduction
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

### 4.3 Risk Management
- [ ] **High risk:** This may break the model entirely at 60%+ reduction
- [ ] Fallback plan: If width pruning fails, demo Phase 3 results (layer deletion only)
- [ ] If time permits: Light distillation/fine-tuning on chopped model to recover performance

**Goal:** Explore how far we can push width reduction using activation-guided pruning

---

## Phase 5: Inference Server & Demo (Sunday Afternoon)

### 5.1 Serving Infrastructure
- [ ] Deploy chopped model (from Phase 3/4) on RunPod GPU
- [ ] Build FastAPI endpoint:
  ```
  POST /tour-guide
  {
    "latitude": -33.8688,
    "longitude": 151.2093,
    "interests": ["architecture", "history"],
    "style": "detailed"
  }
  → Convert to Plus Code
  → Format input for chopped model
  → Return generated description
  ```

### 5.2 Web Demo
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
