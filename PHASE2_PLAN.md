# Phase 2 Implementation Plan & Issues

## Overview
Phase 2 focuses on **Activation Harvesting** - recording activations from Qwen 3-32B FP8 on RunPod to build a task-specific activation database for pruning analysis.

## Critical Issues Found

### 1. **Bug: Undefined Variables in `extract.py`** ⚠️ CRITICAL
**Location:** `data_extraction/extract.py:70`
- **Issue:** `country` and `language` variables are used in the prompt but never defined
- **Impact:** Will cause `NameError` when calling `generate_description_from_plus_code()`
- **Fix Required:** Add parameters to function signature or use default values

### 2. **Missing vLLM Activation Recording** ⚠️ CRITICAL
**Location:** Entire codebase
- **Issue:** Phase 2 requires recording activations from Qwen 3-32B FP8 on RunPod via vLLM, but:
  - Current code only uses vLLM API for text generation (no activation access)
  - Activation recorder is designed for local HuggingFace models only
  - No integration between RunPod vLLM and activation recording
- **Impact:** Cannot complete Phase 2.1 (recording from 32B teacher on RunPod)
- **Solutions:**
  - **Option A (Recommended):** Download model locally, record activations, apply pruning strategy to RunPod model
  - **Option B (Advanced):** Modify vLLM server to expose activation hooks via custom API endpoint

### 3. **Missing Layer-Wise Cosine Similarity Analysis** ⚠️ HIGH PRIORITY
**Location:** `model/activation_analyzer.py`
- **Issue:** Phase 2.2 requires computing cosine similarity between consecutive layers (64 layers total) to identify redundant layers
- **Current State:** Analyzer computes magnitude, variance, sparsity, but NOT layer-to-layer similarity
- **Impact:** Cannot identify which layers are redundant for Phase 3 pruning
- **Fix Required:** Add `compute_layer_similarity_matrix()` method

### 4. **Incomplete Activation Storage Format** ⚠️ HIGH PRIORITY
**Location:** `model/activation_recorder.py`
- **Issue:** Phase 2.2 requires format: `{input_tokens, layer_outputs[64], output_tokens}`
- **Current State:** Saves activations as padded tensors, doesn't store input/output tokens
- **Impact:** Missing context needed for analysis
- **Fix Required:** Update `save_activations()` to include input/output tokens

### 5. **Missing Qwen 3-32B Specific Layer Detection** ⚠️ MEDIUM PRIORITY
**Location:** `model/activation_recorder.py:_get_layer_names()`
- **Issue:** Auto-detection may miss Qwen-specific layer names or capture wrong layers
- **Current State:** Generic pattern matching for layer names
- **Impact:** May not record all 64 layers correctly
- **Fix Required:** Add Qwen-specific layer name patterns or explicit layer list

### 6. **Missing Integration with Phase 1 Dataset** ⚠️ MEDIUM PRIORITY
**Location:** No integration code
- **Issue:** Phase 2.2 requires running "generated Plus Code dataset" through teacher, but no pipeline connects Phase 1 outputs to Phase 2 activation recording
- **Current State:** Separate scripts for data generation and activation recording
- **Impact:** Manual work required to connect phases
- **Fix Required:** Create pipeline script that:
  - Loads Phase 1 generated data (from `samples/` or JSONL)
  - Formats prompts for activation recording
  - Runs through activation recorder

### 7. **Incomplete Pruning Plan Creation** ⚠️ MEDIUM PRIORITY
**Location:** `model/pruner.py:create_pruning_plan_from_analysis()`
- **Issue:** Function has `pass` statement - doesn't actually create pruning plan
- **Impact:** Cannot automatically generate pruning plans from analysis
- **Fix Required:** Implement per-neuron analysis and pruning plan generation

### 8. **Missing HDF5 Storage Option** ⚠️ LOW PRIORITY
**Location:** `model/activation_recorder.py`
- **Issue:** Phase 2.2 mentions "PyTorch tensors/HDF5" but only pickle is implemented
- **Current State:** Uses pickle for serialization
- **Impact:** Less efficient for large datasets
- **Fix Required:** Add HDF5 export option (optional enhancement)

## Implementation Plan

### Step 1: Fix Critical Bugs (Immediate)
1. **Fix `extract.py` undefined variables**
   - Add `country` and `language` parameters to `generate_description_from_plus_code()`
   - Use sensible defaults (e.g., "Australia", "English")

### Step 2: Implement Layer Similarity Analysis (High Priority)
1. **Add cosine similarity computation to `ActivationAnalyzer`**
   - Method: `compute_layer_similarity_matrix()`
   - Compute similarity between consecutive layers (layer i vs layer i+1)
   - Compute similarity matrix for all layer pairs
   - Save similarity matrix for visualization/analysis

### Step 3: Update Activation Storage Format (High Priority)
1. **Modify `ActivationRecorder.save_activations()`**
   - Store input tokens (from tokenizer)
   - Store output tokens (from model generation)
   - Store all 64 layer outputs
   - Use format: `{input_tokens, layer_outputs[64], output_tokens}`

### Step 4: Create Phase 1 → Phase 2 Pipeline (Medium Priority)
1. **Create `phase2_pipeline.py`**
   - Load Plus Code dataset from Phase 1 (JSONL or text files)
   - Format prompts for activation recording
   - Run through activation recorder
   - Save activations in required format

### Step 5: Improve Layer Detection for Qwen (Medium Priority)
1. **Update `_get_layer_names()` in `ActivationRecorder`**
   - Add Qwen-specific layer patterns
   - Or: Explicitly list all 64 layers for Qwen 3-32B
   - Ensure post-attention and post-FFN outputs are captured

### Step 6: Address vLLM Activation Recording (Critical Decision)
**Choose one approach:**

#### Option A: Local Model Approach (Recommended)
1. **Create `phase2_local_recording.py`**
   - Download Qwen 3-32B (or use smaller variant for testing)
   - Load locally with HuggingFace transformers
   - Record activations from Phase 1 dataset
   - Save activation database
   - **Note:** Pruning strategy will be applied to RunPod model later

#### Option B: Custom vLLM Server (Advanced)
1. **Modify vLLM server code**
   - Add forward hooks to model loading
   - Expose activation endpoint or save to disk
   - Deploy custom vLLM template on RunPod
   - Record activations via API or file system

**Recommendation:** Start with Option A for rapid iteration, consider Option B if exact RunPod model activations are critical.

### Step 7: Complete Pruning Plan Generation (Medium Priority)
1. **Implement `create_pruning_plan_from_analysis()`**
   - Use per-neuron activation statistics
   - Generate layer-level pruning plan
   - Generate neuron-level pruning plan (for Phase 4)

### Step 8: Add HDF5 Support (Optional)
1. **Add HDF5 export to `ActivationRecorder`**
   - Use `h5py` library
   - Store activations in HDF5 format
   - More efficient for large datasets

## Testing Strategy

### Unit Tests Needed
1. Test layer similarity computation
2. Test activation storage/loading with new format
3. Test Phase 1 → Phase 2 pipeline integration

## Success Criteria for Phase 2

- [ ] All 64 layers of Qwen 3-32B recorded for tour guide task
- [ ] Activation database saved in format: `{input_tokens, layer_outputs[64], output_tokens}`
- [ ] Layer-wise cosine similarity matrices computed
- [ ] Per-layer activation variance/magnitude calculated
- [ ] Redundant layers identified (high similarity to neighbors)
- [ ] Analysis toolkit ready for Phase 3 pruning decisions

## Next Steps (Priority Order)

1. **Fix `extract.py` bug** (5 minutes)
2. **Add layer similarity analysis** (1-2 hours)
3. **Update activation storage format** (1 hour)
4. **Create Phase 1 → Phase 2 pipeline** (1 hour)
5. **Test with small dataset** (30 minutes)
6. **Decide on vLLM approach and implement** (2-4 hours depending on choice)
7. **Run full activation recording on Phase 1 dataset** (depends on dataset size)

## Estimated Time
- **Critical fixes:** 2-3 hours
- **High priority features:** 3-4 hours
- **Medium priority features:** 2-3 hours
- **Testing and validation:** 1-2 hours
- **Total:** 8-12 hours of focused development

