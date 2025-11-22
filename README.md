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

# Offline Tour Guide - Distillation Hackathon Plan

**Goal:** Distill Qwen 3-14B → 3B for Sydney POI descriptions using Plus Codes, controllable via style tags

**Timeline:** Rest of Friday + Saturday → Demo Sunday 2pm

## Phase 1: Data Generation Pipeline (Friday Evening)

### 1.1 Plus Code Sampling Strategy
- [ ] Generate hierarchical Plus Code samples for Sydney
  - City-level codes (8-char): Neighborhoods/suburbs
  - Block-level codes (10-char): Specific streets
  - Building-level codes (11+ char): Individual POIs
- [ ] Implement density-based sampling (weighted by population/POI density)
- [ ] **TODO:** Define sampling ratios and geographic boundaries

### 1.2 Interest & Style Tag Combinations
- [ ] Interest tags: `architecture`, `culture`, `plans`, `nature`, `food`, `history`, etc.
- [ ] Style tags: `brief`, `stimulate`, `detailed`
- [ ] Generate all valid combinations (single + multi-interest)
- [ ] Create prompt templates for each combination

### 1.3 Strong Model Generation (Claude/GPT)
- [ ] Build generation pipeline:
  ```
  Input: (plus_code, interest_tags, style_tag)
  → Claude/GPT prompt
  → Generated POI description
  → Validate output format (stop token behavior)
  ```
- [ ] Generate initial dataset (~1-2k examples for rapid iteration)
- [ ] **TODO:** Decide on prompt engineering strategy for accuracy
- [ ] Save as JSONL: `{plus_code, interests, style, response}`

#### How to Extract Data from ChatGPT/Claude

**Primary Workflow (Approach 1): Generate FROM Plus Codes**
1. Start with Plus Codes (generated from GPS coordinates via `olc.encode()`)
2. For each Plus Code + interest/style combination, query ChatGPT/Claude API:
   ```
   Prompt: "Generate a {style} tour guide for Plus Code {plus_code}, 
            focusing on {interests}"
   ```
3. Save the generated description as training data
4. See `data_extraction.py` for implementation

**Alternative Workflow (Approach 2): Extract TO Create Plus Codes**
1. Query ChatGPT/Claude to get POI lists for an area (e.g., "Sydney CBD")
2. Extract lat/lon coordinates from the response
3. Convert coordinates to Plus Codes using `olc.encode()`
4. Then use Approach 1 to generate descriptions for those Plus Codes

**Key Point:** Plus Codes are the bridge between:
- Raw location data (GPS, POI lists) → Plus Codes (via `olc.encode()`)
- Plus Codes → Model training data (via ChatGPT/Claude API queries)
- Plus Codes → Model inference (model learned to generate from Plus Code inputs)

## Phase 2: Teacher Model Setup (Saturday Morning)

### 2.1 Qwen 3-14B Infrastructure
- [ ] Set up Qwen 3-14B on GPU server (RunPod)
- [ ] Implement forward pass logging:
  - Hidden states (which layers? all? subset?)
  - Output logits/probabilities
  - Attention patterns (optional)
- [ ] Test throughput - how many examples can we process?

### 2.2 Activation Harvesting
- [ ] Run generated dataset through 14B teacher
- [ ] Save activations + output probs alongside training data
- [ ] Format: `{input, teacher_hidden_states, teacher_logits, target_text}`

## Phase 3: Student Distillation (Saturday Afternoon/Evening)

### 3.1 Qwen 3-3B Training Setup
- [ ] Initialize Qwen 3-3B student model
- [ ] Implement distillation loss:
  - KL divergence on output logits (temperature tuning?)
  - MSE on hidden states (layer-wise matching via symmetry)
  - Cross-entropy on ground truth (student also learns from targets)
  - **Weighting:** Balance these loss components

### 3.2 Training Loop
- [ ] Start training with small batch to verify
- [ ] Monitor:
  - Loss convergence
  - **Stop token generation** (critical!)
  - Output quality on held-out examples
- [ ] Iterate on hyperparameters if time permits

## Phase 4: Inference Server & Demo (Sunday Morning)

### 4.1 Serving Infrastructure
- [ ] Deploy distilled 3B model on RunPod GPU
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
  → Format input for 3B model
  → Return generated description
  ```

### 4.2 Web Demo
- [ ] Simple web interface:
  - Map view (Leaflet/Mapbox)
  - Click to drop pin
  - Interest tag checkboxes
  - Style selector (brief/stimulate/detailed)
  - Display generated tour guide text
- [ ] Pre-generate some example queries for demo stability

---

## Open Questions / TODO

### Data Generation Details
- **Sampling density formula:** Population-weighted? POI count-weighted?
- **Geographic scope:** Just Sydney CBD? Include suburbs? How far out?
- **Interest distribution:** Equal sampling or weight toward common interests?
- **Validation strategy:** How do we verify Claude/GPT outputs are accurate?

### Model Specifics
- **Which layers to distill?** All 14B layers → 3B layers, or subset?
- **Loss weighting:** What ratio of KL / MSE / CE losses?
- **Quantization:** Do we quantize the 3B model for faster inference?

### Immediate TODOs
- Extract data about locations in Sydney from Cursor AI or Claude code
- Runpod - figure out GPU requirements (RAM)
- Figure out which Qwen model to use

---

## Success Criteria (Sunday 2pm)

1. ✅ Working inference server accepts GPS → returns contextualized POI description
2. ✅ Model respects style tags (generates appropriate length)
3. ✅ Multi-interest queries work
4. ✅ Demo shows 5-10 diverse Sydney locations
5. ✅ Latency < 2 seconds per query
