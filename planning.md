# Research Plan: Basis Vectors in Persona Space

## Motivation & Novelty Assessment

### Why This Research Matters
Activation steering with persona vectors has proven effective for controlling LLM behavior, but each persona vector is extracted independently. If persona vectors share underlying structure — decomposable into a smaller set of basis components — this would (1) provide a compact representation for the full diversity of personas, (2) reveal what the "atoms" of persona are in model representations, and (3) enable more principled persona composition.

### Gap in Existing Work
Per the literature review, prior work either:
- Extracts individual persona/trait vectors and studies them in isolation (Chen et al., Allbert et al.)
- Tests orthogonality of a small predefined set like Big Five (Feng et al. PERSONA)
- Uses PCA on a single concept at a time (Zou et al. RepE)

**No one has extracted vectors for a large diverse set of personas (135 in Anthropic evals: personality, ethics, politics, AI safety, interests) and applied PCA to the full collection to discover the natural dimensionality and primary components of persona space.**

### Our Novel Contribution
We perform the first systematic PCA decomposition of a large, diverse collection of persona vectors extracted from LLM residual streams. We answer: What are the principal axes of persona variation? How many dimensions are needed? Do the discovered components map onto known psychological or semantic categories?

### Experiment Justification
- **Experiment 1 (Vector Extraction)**: Extract 135 persona vectors — necessary foundation
- **Experiment 2 (PCA Decomposition)**: Apply PCA across layers — core hypothesis test
- **Experiment 3 (Component Interpretation)**: Correlate PCs with known categories — makes results scientifically meaningful
- **Experiment 4 (Steering Validation)**: Steer with individual PCs — validates that discovered components are causally meaningful, not just correlational

## Research Question
Can persona representations in LLMs be decomposed via PCA into a small number of interpretable basis components, and what are the primary dimensions of persona variation in the residual stream?

## Hypothesis Decomposition
1. **H1**: Persona vectors extracted from diverse personas are not independent — PCA will reveal that a small number of components (<<135) explain most variance
2. **H2**: The top principal components will be interpretable, corresponding to meaningful semantic/psychological dimensions
3. **H3**: The PCA structure varies across layers, with later layers showing clearer persona structure
4. **H4**: Principal components can be used for activation steering, validating their causal role

## Proposed Methodology

### Approach
1. Load a 7B-parameter instruction-tuned model (Qwen2.5-7B-Instruct — well-studied in literature)
2. For each of 135 personas: select 100 high-confidence contrastive statement pairs, extract last-token residual stream activations, compute difference-in-means vector
3. Stack all 135 persona vectors into a matrix, apply PCA
4. Analyze explained variance, interpret top components, validate via steering

### Experimental Steps
1. **Data preparation**: Filter Anthropic persona evals to high-confidence items (≥0.85), select up to 100 per persona
2. **Activation extraction**: For each statement, run through model, extract residual stream at layers [8, 16, 20, 24, 28, 31] (spanning early/mid/late)
3. **Persona vector computation**: For each persona p, compute v_p = mean(activations for "matching" statements) - mean(activations for "not matching" statements)
4. **PCA**: Stack v_1..v_135 into matrix V ∈ R^{135 × d_model}, apply PCA
5. **Analysis**: Explained variance ratios, component loadings, cosine similarity matrices, hierarchical clustering
6. **Causal validation**: Steer model with top-3 PCs and measure behavioral shift

### Baselines
- Random directions in activation space (null hypothesis)
- Individual persona vectors without decomposition
- Big-Five-only vectors (5-dimensional hypothesis)

### Evaluation Metrics
- Explained variance ratio (cumulative)
- Number of components for 80%/90% variance
- Cosine similarity between PCs and individual persona vectors
- Silhouette score for persona clustering in PC space
- Steering effectiveness (behavioral change per unit PC injection)

### Statistical Analysis Plan
- Bootstrap confidence intervals on explained variance ratios
- Permutation test: shuffle persona labels, re-extract vectors, compare PCA structure
- Marchenko-Pastur test for significant components above random matrix threshold

## Expected Outcomes
- **Support H1**: ~10-20 components explain >80% of persona variance (vs. 135 raw dimensions)
- **Support H2**: Top PCs align with broad categories: safety/alignment, politics, personality, interests
- **Support H3**: Later layers (20+) show clearer structure with fewer components needed

## Timeline
- Phase 1 (Planning): 15 min ✓
- Phase 2 (Setup + Data): 15 min
- Phase 3 (Implementation): 60 min
- Phase 4 (Experiments): 90 min
- Phase 5 (Analysis): 30 min
- Phase 6 (Documentation): 20 min

## Potential Challenges
- Memory: 135 personas × 100 samples × 32 layers × 3584 hidden dim — extract per-layer, don't store all at once
- Some personas may have <100 high-confidence items — use minimum available
- Model loading: Qwen2.5-7B needs ~14GB in float16 — fits easily on A6000

## Success Criteria
- PCA reveals meaningful dimensionality reduction (significantly fewer than 135 components for 80% variance)
- At least 3 top PCs have clear semantic interpretation
- Results are reproducible across different layer choices
