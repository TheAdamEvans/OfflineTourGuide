# Task-Aware Gradient-Free Qwen/Qwen3-32B → Qwen/Qwen3-4B Compression

## The Goal

**Compress a Qwen/Qwen3-32B model (5120 hidden dim) into a Qwen/Qwen3-4B model (~2048 hidden dim) using only linear transformations (rotations), no gradient descent.**

Merge the compressed 32B into an existing Qwen/Qwen3-4B from the same model family (already distilled from the 32B, so their latent spaces are naturally well-aligned), using **activation-informed layer-specific merging** for optimal task performance.

We'll prototype with the FP8 downsampled checkpoints that Qwen publishes on Hugging Face so the workflow matches the artifacts we'll actually ship.

---

## Core Idea 1: Two-Tiered Neuron Structure (1024/1024 Split)

**The Insight**: Not all neurons are created equal. There's a natural split:

### Tier 1: The Trunk (Exponent) - 1024 dims
- **Top 1024 SVD components** of the 32B's activations
- Captures ~85-90% of total variance
- **High-frequency features**: common patterns, frequent activations
- **Minimal superposition**: each dimension is a clear principal direction
- **Clean, orthogonal directions** that rarely interfere

### Tier 2: The Residual (Mantissa) - 1024 dims  
- **Remaining 4096 dimensions compressed 4:1** into 1024 dims
- Captures ~5-10% of remaining variance (but critical!)
- **Low-frequency features**: rare patterns, sparse activations
- **Heavy superposition**: 4 dimensions sharing 1 slot
- **Intentional interference** - but manageable because features rarely co-occur

**Applied uniformly**: 1024 trunk / 1024 residual at **every layer** for simplicity

**Why this works**: 
- Total: 5120 dims originally → 2048 dims (1024 + 1024)
- Trunk captures the "order of magnitude" (first 1024 principal directions)
- Residual captures "fine corrections" (compressed from 4096 → 1024)
- **Better than 2048 pure SVD** because we treat high-variance and low-variance dims differently

**The metaphor**: Like floating point numbers
- Exponent (trunk): captures scale, the big picture
- Mantissa (residual): captures precision, the details

---

## Core Idea 2: Align to the 4B's Basis

**The Problem**: Per-layer rotations create transition matrices T between layers that cost parameters and compute.

**The Solution**: Use the 4B model as your "target basis" (helped by the fact that Qwen/Qwen3-4B is distilled from Qwen/Qwen3-32B, so most latent axes already match).

```python
# For each layer i:
# Find V_i that maps 32B → 4B's natural representation

X_32B = activations_32B[layer_i]  # (140k, 5120)
X_4B = activations_4B[layer_i]     # (140k, 2048)

# Find optimal rotation
V_i = find_rotation_that_maps(X_32B → X_4B)
```

**Why this works**:
- If V_i maps to "what the 4B expects" in layer i
- And V_{i+1} maps to "what the 4B expects" in layer i+1
- Then T_{i,i+1} = V_{i+1}^T @ V_i ≈ I (near identity)
- **Transition matrices become free!**

**Bonus**: The 4B's weights are already trained to process information in this basis, so merged weights work better.

---

## Core Idea 3: Activation-Informed Layer-Specific Merging 🔥

**The Breakthrough**: Don't use uniform merge ratios across layers. Let the **task activations** tell you how much to trust the compressed 32B vs the 4B at each layer.

### Layer-Specific Base Ratios

Different layers have different fidelity requirements, as measured by their importance to the task, somehow.

**What this achieves**:
- **Task-specific optimization** without gradient descent
- **Layer-wise adaptation** to where 32B knowledge helps most
- **Automatic quality control** - if compression quality is poor at a layer, fall back to 4B
- **Zero-shot generalization** - works across different tasks using same activation principles

---

## Combined Approach: Two-Tier + Alignment + Activation-Informed Merging

**What you get**:
- First 1024 dims: trunk features, aligned to 4B's important dims
- Second 1024 dims: compressed residual, aligned to 4B's detail dims
- Transition matrices T ≈ I (nearly free)
- **Task-aware merge ratios** optimized per layer based on activation statistics
- **Automatic fallback** to 4B when compression quality is uncertain

**Practical caching note:** we run the full task corpus through both the 32B teacher and the native 4B student up front. Those paired activation dumps are enough to solve the rotations *and* to compute all pseudo-4B diagnostics (per-layer cosine, variance ratios, KV-group scores) without immediately inverting the folded checkpoint. We still plan a dedicated pseudo-4B inference pass later as a sanity check, but all α-calibration math can happen directly from the cached teacher/student activations.

---

## GQA-Aware Symmetry Strategies

Qwen/Qwen3-4B
Number of Parameters: 4.0B
Number of Paramaters (Non-Embedding): 3.6B
Number of Layers: 36
Number of Attention Heads (GQA): 32 for Q and 8 for KV

Qwen/Qwen3-32B
Number of Parameters: 32.8B
Number of Paramaters (Non-Embedding): 31.2B
Number of Layers: 64
Number of Attention Heads (GQA): 64 for Q and 8 for KV

- **Head pairing compression**: treat the 32B's 64 Q heads as eight KV-aligned groups of eight, then run block SVD per group so each student block inherits the dominant four directions without scrambling inter-group semantics.
- **Head-importance pruning**: score teacher Q heads within each KV group by activation energy or contribution to attention entropy, keep the top heads, and tie the discarded ones to their nearest survivors before applying rotations.
- **Block-diagonal rotations**: replace a single dense rotation with eight smaller blocks, one per KV group, which shortens estimation windows and keeps grouped-query structure intact.
- **KV-basis reuse**: since both models share eight KV heads, freeze the 4B KV basis and only rotate the Q projections into that space, preserving the student's stable value flow while injecting teacher features through queries.
- **Group-wise α_i**: compute activation-informed merge weights separately for each KV group so reliable blocks lean on the compressed 32B views while weaker ones fall back to the 4B baseline.

---

## Crucial Point of Research: Align Each "Tier" Separately

**Explicit trunkand residual alignment:**

```python
# SVD both models
U_32, S_32, Vt_32 = svd(X_32B)
U_4, S_4, Vt_4 = svd(X_4B)

# TRUNK ALIGNMENT
# Align 32B's top 1024 components → 4B's top 1024 components
R_trunk = orthogonal_procrustes(
    Vt_32[:1024, :],   # 32B's principal directions
    Vt_4[:1024, :]     # 4B's principal directions
)
V_trunk_aligned = Vt_32[:1024, :].T @ R_trunk  # (5120, 1024)

# RESIDUAL ALIGNMENT  
# Build 32B's compressed residual
residual_space = Vt_32[1024:, :].T  # (5120, 4096)
residual_acts = X_32B @ residual_space
U_r, S_r, Vt_r = svd(residual_acts)
V_residual_compressed = residual_space @ Vt_r[:1024, :].T  # (5120, 1024)

# Align compressed residual → 4B's lower 1024 components
R_residual = orthogonal_procrustes(
    Vt_r[:1024, :],      # 32B's compressed residual directions
    Vt_4[1024:, :]       # 4B's lower priority directions
)
V_residual_aligned = V_residual_compressed @ R_residual  # (5120, 1024)

# COMBINE
V_i = [V_trunk_aligned | V_residual_aligned]  # (5120, 2048)

# Then apply activation-informed merge as before
```

**What's different**:
- **Explicit mapping**: 32B trunk → 4B trunk, 32B residual → 4B residual
- **Better semantic alignment**: Important features match important features
- **More control**: Can tune trunk vs residual compression separately
- **Still get T ≈ I**: Because both tiers align to 4B's basis

---

## The Full Pipeline

```
32B Model (5120 dims) + Task Data (140k samples)
    ↓
Per-layer activation collection on task
    ↓
For each layer i:
    ├─ Per-layer SVD
    ├─ Split into:
    │   ├─ Trunk: top 1024 (exponent/coarse)
    │   └─ Residual: remaining 4096 → 1024 (mantissa/fine)
    ├─ Combine: [trunk | residual] = 2048 dims
    ├─ Align to 4B's basis via Procrustes
    ├─ Compress weights: W_32B_compressed = V_i^T @ W_32B @ V_{i+1}
    ├─ Measure activation certainty & alignment
    ├─ Compute layer-specific merge weight α_i
    └─ Merge: W_final = α_i · W_32B_compressed + (1-α_i) · W_4B
    ↓
Final 4B model with task-optimized 32B knowledge
```

---

## Validation Framework

**1. Transition Matrix Quality**
```python
# Compute transition cost per layer
T_i = V_{i+1}.T @ V_i
transition_cost = np.linalg.norm(T_i - np.eye(2048), 'fro')
# Target: < 0.1 for near-identity
```

**2. Variance Preservation**
```python
# Per-layer variance check
X_reconstructed = X_32B @ V_i @ V_i.T
variance_ratio = np.var(X_reconstructed) / np.var(X_32B)
# Target: > 95%
```

**3. Higher-Order Diagnostics**
```python
# Track non-Gaussian structure that might signal drift
kurtosis_gap = kurtosis(X_32B @ V_i) - kurtosis(X_4B)
pairwise_mi = estimate_pairwise_mi(X_32B @ V_i, X_4B)
# Target: stay within a tolerance band observed on teacher/student pairs
```

**4. Semantic Alignment Quality**
```python
# How well do compressed 32B activations match 4B?
X_32B_compressed = X_32B @ V_i
alignment_quality = cosine_similarity(X_32B_compressed, X_4B).mean()
# Target: > 0.8
```

**5. Merge Quality Assessment**
```python
# Track per-layer merge decisions
merge_profile = {
    'layer': layer_idx,
    'alpha': alpha_chosen,
    'certainty': certainty_score,
    'alignment': alignment_score,
    'variance_preserved': variance_ratio
}
# Visualize to understand where 32B helps most
```

---

## Research Questions & Experiments

### Primary Hypothesis
**The Smoothness Conjecture**: Task-specific activation manifolds are smooth and low-dimensional, enabling effective compression via pure geometric methods without gradient descent.

### Key Experiments

**1. Baseline Comparisons**
- Pure 2048-dim SVD (no two-tier structure)
- Uniform merge ratio (α = 0.5 everywhere)
- No activation-informed adjustment
- Standard knowledge distillation with backprop

**2. Ablations**
- Two-tier structure ON/OFF
- Procrustes alignment ON/OFF
- Activation-informed merging ON/OFF
- Layer-specific vs uniform merge ratios

**3. Data Efficiency Tests**
- Vary task samples: 1k, 10k, 50k, 140k
- Measure performance vs sample size
- Compare to gradient-based methods

**4. Task Generalization**
- Train on coding, test on math
- Train on QA, test on reasoning
- Measure zero-shot transfer quality

**5. Composability Experiments**
```python
# Can we blend task-specific compressions?
V_coding = compute_rotation(coding_data)
V_math = compute_rotation(math_data)
V_blend = geodesic_average([V_coding, V_math], weights=[0.7, 0.3])
# Does blended rotation work for mixed tasks?
```

**6. Layer Sensitivity Analysis**
- Which layers benefit most from 32B knowledge?
- Which layers need high-fidelity compression?
- Validate early/late layer importance findings

**7. Merge Weight Sensitivity**
```python
# For each layer, sweep α from 0.0 to 1.0
# Plot task performance vs α
# Identify critical layers and optimal α ranges
```

**8. Transition + Merge Reliability Under Realistic Budgets**
- Stress-test `T_i` stability and α_i certainty estimates with progressively smaller activation corpora
- Confirm higher-order statistics stay within tolerance even when data is sparse
- Document failure thresholds to guide minimum viable data collection

**9. GQA Symmetry Validation**
- Measure attention-logit KL divergence per KV group before/after block rotations to confirm grouped mapping stability
- Compare block-diagonal vs full rotations on variance preservation and transition-cost metrics to verify sample-efficiency gains
- Run ablations where KV bases are frozen vs jointly rotated to test whether shared 8-head structure already matches the optimal basis

---

**Key Advantages**:
- **No gradient descent required** - hours instead of days
- **Data efficient** - 140k samples vs millions for standard distillation
- **Task-adaptive** - automatically adjusts compression quality per layer
- **Composable** - can blend multiple task compressions
- **Interpretable** - clear geometric meaning at every step

---

## Other Mathy Future Directions
- Formalize smoothness conditions required for success
- Prove bounds on compression-performance tradeoffs
- Connect to rate-distortion theory and information geometry

---

This is genuinely novel work combining rotation-based compression, two-tier structure, basis alignment, and **activation-informed adaptive merging**. The key innovation is using task activations to guide compression decisions without requiring gradient descent.
