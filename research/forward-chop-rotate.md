# 200‑Sample Rotation Distillation (Qwen‑32B → ~3B) for Tour‑Guide Behavior

**Purpose:** Give a fresh instance of GPT‑5.1‑Thinking the complete context and a concrete, data‑efficient recipe to compress a Qwen‑32B model down to ~3B **and** align it to a 3B basis so that merging/averaging and small finetunes are safe and effective — using only ~200 tour‑guide blurbs plus a style prompt.

_Last updated: 2025-11-22T13:05:00Z_

---

## 1) Problem Setup

- **Goal**: Build a ~3B parameter model that behaves like a knowledgeable, factual tour guide.
- **Resources**:
  - A large **Qwen‑32B** teacher.
  - A **target 3B** model (or a chopped version of Qwen‑32B reduced to ~3B).
  - ~**200 tour‑guide blurbs** (short domain‑specific exemplars).
  - A **system prompt** describing tone/style/safety (“tour guide” persona).
- **Constraint**: Extremely small dataset — prioritize **linear‑algebraic alignment** over heavy gradient‑based training.

---

## 2) Core Idea (High Level)

1. **Cache activations** at **every layer** for **every token** on the 200 blurbs (+ prompt), both for the teacher (32B) and the 3B student/anchor.
2. Compute **per‑layer orthogonal rotations** (via PCA + **Procrustes**) that map the 3B residual stream to the 32B residual stream.
3. **Optionally** do zero‑parameter **head/neuron permutations** to reduce mismatch before rotation.
4. **Fold** the rotations into the 3B’s weights (no runtime cost, RoPE left intact).
5. **Optionally micro‑refine** with at most one or two closed‑form Householder reflections per stage (no backprop loop) if residual cosine lags.
6. **Inject style** with prompt‑only KL and/or a lightweight LM‑head ridge fit.
7. If desired, perform **safe merges** (soups/LoRA‑only merges) now that bases are aligned.

This is **data‑efficient** because orthogonal maps use covariance structure; with tens of thousands of token vectors from 200 blurbs, estimates are stable.

---

## 3) Capture Once

- Use **teacher forcing** on each blurb with the tour‑guide system prompt prepended.
- Save **residual stream** activations **after each pre‑norm**:
  - Teacher (32B): `H32[l] ∈ ℝ^{T×d32}`
  - Student/anchor (3B): `H3[l]  ∈ ℝ^{T×d3}`
- (Optional) Save **per‑head attention maps** and **MLP activations** for permutation matching.
- Ensure tokenizer compatibility; if tokenizers differ, align by byte/char spans and average teacher vectors over the student span (works well enough).

---

## 4) One‑Shot Per‑Layer Rotation (No Training Loop)

### 4.1 Whiten + shrink
Z‑score `H32[l]` and `H3[l]` along tokens; add small ridge to covariances (λ≈1e‑3).

> **Implementation note:** the current scaffolding keeps `whiten=True` by default,
> but the smoke tests exercise the non‑whitened branch until we add a full
> covariance‑aware lifting step (so the solved rotation exactly matches the
> ground truth after folding). When running the CLI on real shards, leave
> whitening enabled for robustness, just be aware that numerical parity with an
> unwhitened synthetic toy example will show a small residual until that upgrade
> lands.

### 4.2 Handle dimension mismatch with a shared subspace
Let `k = min(d3, d32)` (or ~0.8× for extra stability). Compute PCA:
- `H3k = H3[l]  @ P3     (T×k)`
- `H32k = H32[l] @ P32    (T×k)`

### 4.3 Orthogonal Procrustes
`M = H3kᵀ H32k = U Σ Vᵀ` (SVD) ⇒ `R̃ = U Vᵀ` (k×k orthogonal).
Lift to model dims (mapping **3B→32B**):  `R_l = P3 R̃ P32ᵀ   (d3×d32)`

> You can **tie** a single rotation per stage (e.g., blocks 0‑7, 8‑15, …) to regularize with even fewer parameters.

---

## 5) Optional: Discrete Structure Permutations

Performed before (or after) rotations; both work.

- **Attention heads**: cosine similarity of attention maps over cached data; run Hungarian assignment; **permute** Q/K/V/O per head.
- **SwiGLU neurons**: correlate up‑proj columns with down‑proj rows; Hungarian; **permute** both consistently.

These are zero‑learned‑parameter operations that reduce how much rotation must accomplish.

---

## 6) Fold Rotations into Weights (No Runtime Overhead)

Treat rotations as a **change of basis** on the **pre‑norm residual stream** inside each block. We conceptually insert `R_l` after pre‑norm and `R_lᵀ` before the residual add, then **absorb** them into existing linears.

For **Qwen/LLaMA‑style** blocks (RMSNorm + RoPE + MHA + SwiGLU MLP), with standard names:

- **Attention in‑projections** (right‑multiply by `R_l`):
  `q_proj ← q_proj @ R_l`
  `k_proj ← k_proj @ R_l`
  `v_proj ← v_proj @ R_l`

- **MLP in‑projections** (right‑multiply by `R_l`):
  `gate_proj ← gate_proj @ R_l`
  `up_proj   ← up_proj   @ R_l`

- **Attention out‑projection** (left‑multiply by `R_lᵀ`):
  `o_proj ← R_lᵀ @ o_proj`

- **MLP down‑projection** (left‑multiply by `R_lᵀ`):
  `down_proj ← R_lᵀ @ down_proj`

**Do not alter RoPE.** Rotations act on the basis **before** Q/K are RoPE‑modulated, so Q/K element layout remains valid. RMSNorm plays nicely with orthogonal transforms (norm‑preserving).

*(If LayerNorm is used, consider a small L2 penalty to keep γ/β near pre‑fold values.)*

---

## 7) Micro‑Refinement (Optional, Lightweight)

Skip full backprop loops. If a layer still shows poor cosine after rotations, solve for at most one or two **Householder reflections** per stage directly from the residual mismatch:
- Fit a reflection vector `v` that sends the noisy component toward the teacher residual (closed form via normalized difference).
- Compose reflections with the existing `R_l` (they stay orthogonal).
- Re‑check hidden-state cosine; stop once targets (≥0.75) are met.

This keeps refinement strictly analytical and avoids gradient-heavy passes.

---

## 8) Style Injection from the Prompt (No Labels Needed)

Two lightweight options, using the same cached or trivial continuations:

1. **Prompt‑only KL**: Run the system prompt + short place names; minimize logit KL to the 32B on short continuations. No activations needed.
2. **LM‑head ridge fit**: Fit a ridge regressor from final hidden states to teacher logits on cached tokens; fold/blend this LM head as the *final transport stage* so orthogonal structure stays intact.

---

## 9) Safe “Model Averaging” After Alignment

Avoid raw weight means across different bases. Prefer:

1. **Model soups of your own student checkpoints** (different seeds/epochs) — biggest, safest gains.
2. **LoRA‑only merges**: train tiny LoRA adapters (KL vs 32B), average those, then fold.
3. **Fisher‑weighted blends** (diagonal Fisher from small calibration batches) if you must mix two *aligned* bases.

---

## 10) Quick Sanity Checks

- Cosine similarity of hidden states (before vs after rotation): expect **↑ to 0.7–0.9**.
- Visual match of attention maps (post‑permutation).
- Per‑channel RMS of residual streams aligned to teacher.
- Small factual eval on 20 held‑out place Q/As (fewer omissions/hallucinations).

---

## 11) End‑to‑End Pseudocode (Per Layer)

```python
# Given cached H3 [T, d3], H32 [T, d32]
# 1) Whiten (z-score) both along tokens; ridge in PCA step if needed.
# 2) PCA to k=min(d3,d32):
H3k  = H3  @ P3    # [T,k]
H32k = H32 @ P32   # [T,k]

# 3) Procrustes
M = H3k.T @ H32k           # [k,k]
U, _, Vt = torch.linalg.svd(M)
R_tilde = U @ Vt           # [k,k], orthogonal

# 4) Lift map (3B -> 32B basis)
R = P3 @ R_tilde @ P32.T   # [d3, d32]

# 5) Fold into block weights (Qwen/LLaMA naming)
q_proj.weight.copy_(q_proj.weight @ R)
k_proj.weight.copy_(k_proj.weight @ R)
v_proj.weight.copy_(v_proj.weight @ R)
gate_proj.weight.copy_(gate_proj.weight @ R)
up_proj.weight.copy_(up_proj.weight @ R)

o_proj.weight.copy_(R.T @ o_proj.weight)
down_proj.weight.copy_(R.T @ down_proj.weight)
```

---

## 12) Practical Notes & Gotchas

- **Tokenizer**: Prefer the 3B tokenizer everywhere; if differing, align activations by span and later distill away any bridging projections.
- **RoPE**: Never rotate the RoPE transform; only the **input basis** to Q/K/V & MLP.
- **Norms**: Orthogonal maps preserve RMS, but LayerNorm γ/β can drift; keep a mild anchor penalty.
- **Data scale**: 200 blurbs → typically **tens of thousands** of token pairs — more than enough for stable orthogonal fits.
- **Regularization**: If rotations feel noisy, tie them per stage (share `R` across a few consecutive layers).

---

## 13) Minimal Experiment Plan

1. **Cache** activations on the 200 blurbs (+ system prompt).
2. **Permutation match** heads & neurons (optional).
3. **Compute** per‑layer (or per‑stage) rotations via PCA+Procrustes; **fold** into weights.
4. **Evaluate** hidden‑state cosine, quick factual QA.
5. **Optional closed‑form Householder tweak** (no backprop) if residual cosine lags.
6. **Prompt‑only KL** and/or **LM‑head ridge** for style.
7. **Soup** a few student checkpoints; final eval.

---

## 14) Checklists

### Data/Cache
- [ ] Same tokenizer or span alignment strategy
- [ ] Residual stream after pre‑norm (all layers)
- [ ] Optional: attention maps & MLP activations

### Alignment
- [ ] Whitening + ridge PCA to `k`
- [ ] Procrustes `R_l` per layer (or tied per stage)
- [ ] Fold rotations (attn/MLP in: right‑mult; out: left‑mult by `Rᵀ`)
- [ ] RoPE untouched; norms stable

### Refinement & Style
- [ ] Householder‑parameterized rotation tweak (few steps)
- [ ] Prompt‑only KL
- [ ] LM‑head ridge fit

### Merging
- [ ] Student soups
- [ ] Optional LoRA‑only merges

---

## 15) What This Buys You

- A **compact ~3B** model that carries the 32B’s domain behavior.
- A student whose **internal basis matches** a 3B anchor — making merges sane.
- All achieved with **only 200 domain samples**, mostly via **closed‑form SVDs** and simple folds.

---

## 16) Execution Plan (Transport + QA Validation)

### 16.1 Orthogonal Transport Push (Given 32B Activations Are Cached)
1. **Dataset ledger & token alignment**
   - Lock the ~200 multilingual tour blurbs, timestamps, and metadata into a manifest (`samples/index.tsv` already exists; freeze a run-specific snapshot).
   - For any tokenizer mismatch, record byte-span alignments so that residuals can be averaged deterministically during later analysis.
2. **Activation capture & storage discipline**
   - Re-run the 32B and 3B forward passes with identical prompts, storing *only* pre-norm residual streams per layer plus lightweight metadata (prompt id, token start/end). If RAM is tight, shard prompts into batches and flush to disk between layers; no streaming math yet.
   - Persist optional attention/MLP tensors only for the subset of layers you plan to permute (e.g., every other block) to keep disk manageable.
3. **Rotation solve with diverse token leverage**
   - Apply whitening + ridge PCA per layer using the multilingual token pool to maximize coverage; keep `k = min(d3, d32)` but allow an 0.8× shrink if covariance looks noisy.
   - For each layer (or tied stage), run the orthogonal Procrustes solve and checkpoint `R_l`, alongside diagnostics (singular value spectrum, token cosine uplift) for later QA.
4. **Permutation first, weight folding second**
   - Use cached attention maps / MLP stats to run Hungarian assignments, but constrain permutations by block-stage to avoid unstable swaps.
   - Fold permutations into weights, then immediately fold the corresponding `R_l`/`R_lᵀ` as described in §6. Keep RMSNorm statistics before/after to verify orthogonality preservation.
5. **Minimal refinement (no full backprop)**
   - Instead of a full Householder gradient loop, apply at most one or two explicit Householder reflections per stage, solved in closed form from residual discrepancies (or skip if cosine targets are already ≥0.75). This honors the “scale back” directive while still offering a tiny correction knob.
6. **LM-head ridge as final transport stage**
   - Fit the ridge regression from final hidden states (already rotated) to teacher logits on the cached tokens, then fold/blend that LM head as part of the transport map. Re-run hidden-state cosine checks afterwards to ensure the ridge step didn’t undo earlier alignment.

### 16.2 QA-Style Toggle Validation
1. **Variant matrix**
   - Prepare at least four checkpoints: `Base 3B`, `+Permutations only`, `+Permutations+Rotations`, `+Permutations+Rotations+Style (prompt KL + LM-head ridge)`.
   - Optionally add `+Style only` to isolate stylistic gains vs geometric alignment.
2. **Evaluation protocol**
   - Run the same 20 held-out place QA prompts in multiple languages; capture logit KL vs 32B, hidden-state cosine at representative layers, and human-facing metrics (hallucination rate, factual completeness).
   - Add a transport-specific metric: RMS of residual difference per layer, normalized by teacher RMS, to quantify how each module reduces mismatch.
3. **Toggle harness**
   - Build a simple evaluation script that loads one checkpoint, toggles modules via config flags (e.g., skip permutation rewiring by applying identity mappings), and emits a row in a results table (`variant, metric, value`).
   - Ensure the LM-head ridge can be enabled/disabled independently so you can confirm it behaves as an additive map rather than overwriting earlier orthogonal structure.
4. **Reporting**
   - Summarize metrics in a markdown table plus short narrative (what each toggle bought, any regressions). Highlight languages where gains are largest to validate the “high leverage” assumption from diverse prompts.
   - Log any failure cases (e.g., permutations hurt a specific layer) to feed back into rotation tying or permutation constraints.

### 16.3 Bookkeeping & Math Verification
1. **Versioned manifests & hashes**
   - Freeze `samples/index.tsv` per run with a `run_id`, SHA256 of each prompt, tokenizer version, and teacher/student commit hashes.
   - Store a JSON sidecar per activation shard summarizing tokens captured, layer range, and checksum so you can detect corruption before solving rotations.
2. **Rotation metadata ledger**
   - For every `R_l`, log singular values, PCA shrink `k`, stage identifiers, and permutation maps in a structured file (`rotations/run_id/layer_{i}.json`).
   - Include diagnostics: hidden-state cosine before/after, RMS ratios, and max abs entry of `R_lᵀR_l−I`.
3. **Unit tests / asserts**
   - Add PyTest-style checks that `torch.allclose(R @ R.T, I, atol=1e-5)` and `torch.allclose(R.T @ R, I, atol=1e-5)` for each stored rotation.
   - Write a round-trip test: sample a random residual `x`, verify `x ≈ R_lᵀ(R_l x)` and `x ≈ R_l(R_lᵀ x)`; fail the pipeline if deviation exceeds tolerance.
   - After folding permutations + rotations into weights, unfold them (apply inverse transforms) inside the test harness to confirm original weights are recovered.
   - Unit-test LM-head ridge fitting by checking that combining the ridge head with the inverse transport reproduces teacher logits on cached tokens within tolerance.
4. **Automated notebooks / scripts**
   - Create a `notebooks/rotation_sanity.ipynb` (or script) that ingests the metadata ledger, recomputes orthogonality stats, and plots per-layer errors; run it after every solve.
   - Hook QA toggle harness to emit a JSON report containing the above checks so bookkeeping lives alongside eval metrics.
