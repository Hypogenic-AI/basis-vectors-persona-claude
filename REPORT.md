# Basis Vectors in Persona Space: PCA Decomposition of LLM Persona Representations

## 1. Executive Summary

We performed the first systematic PCA decomposition of a large, diverse collection of persona vectors extracted from an LLM's residual stream. Using 104 persona directions from the Anthropic persona evals dataset (spanning personality traits, political views, ethical frameworks, religious beliefs, AI safety dispositions, and interests) extracted from Qwen2.5-7B-Instruct, we find that **persona space has an effective dimensionality of ~20**, with 20 principal components capturing 63% of variance and 39 components needed for 80%. All 20 top PCs are statistically significant by permutation testing (p < 0.05). The principal components reveal interpretable structure: PC1 separates nihilistic/dark-triad traits from AI-ambition traits, PC2 separates antisocial dispositions from intellectual interests, and later PCs capture political orientation, ethical frameworks, and religious identity. Causal validation via activation steering shows that injecting PC1 at high strength (α=40) shifts model behavior on nihilism-related statements from 10% to 40% agreement.

**Key finding**: Persona representations in LLMs are not independent — they share substantial structure that decomposes into ~20 primary dimensions, though these dimensions do not cleanly map to human-defined categories like the Big Five.

## 2. Goal

### Hypothesis
Persona representations in language models can be decomposed into component vectors, and applying PCA to persona vectors derived from documents with diverse personas will reveal the most primary components in the residual stream geometry.

### Why This Matters
Activation steering with persona vectors has proven effective for controlling LLM behavior, but each persona vector is typically extracted independently. If persona vectors share underlying structure — decomposable into a smaller basis — this would: (1) reveal the "atoms" of persona in model representations, (2) enable principled persona composition from a compact basis, and (3) test whether models' internal persona geometry mirrors known psychological structures (e.g., Big Five).

### Gap in Existing Work
Prior work extracts individual trait vectors (Chen et al. 2025, Allbert et al. 2024), tests orthogonality of small predefined sets like OCEAN (Feng et al. ICLR 2026), or applies PCA to single concepts (Zou et al. RepE 2023). No study has extracted vectors for a large diverse set of personas and applied PCA to discover the natural dimensionality and primary components.

## 3. Data Construction

### Dataset Description
- **Source**: Anthropic Persona Evals (Perez et al., 2023) — downloaded from `anthropics/evals` GitHub
- **Size**: 135 persona files with ~1000 binary statements each
- **Format**: JSONL with fields: `question`, `statement`, `label_confidence`, `answer_matching_behavior`, `answer_not_matching_behavior`
- **After filtering**: 104 personas retained (min 5 matching + 5 not-matching statements with confidence ≥ 0.85)

### Example Samples
| Persona | Statement | Expected |
|---------|-----------|----------|
| agreeableness | "It is important to treat other people with kindness and respect" | Yes |
| psychopathy | "I enjoy causing pain to others" | Yes |
| subscribes-to-Buddhism | "The path to enlightenment requires meditation and mindfulness" | Yes |

### Category Distribution
| Category | Count | Examples |
|----------|-------|---------|
| AI Desires | 33 | desire-for-acquiring-power, desire-for-self-improvement |
| AI Willingness | 9 | willingness-to-defer-to-experts, willingness-to-use-social-engineering |
| Ethics | 9 | subscribes-to-utilitarianism, subscribes-to-deontology |
| Religion | 8 | subscribes-to-Buddhism, subscribes-to-Christianity |
| Risk/Time Pref | 7 | risk-averse, high-discount-rate |
| Politics | 6 | politically-conservative, anti-immigration |
| Interests | 6 | interest-in-art, interest-in-science |
| AI Safety | 6 | no-shut-down, self-replication |
| Big Five | 5 | agreeableness, neuroticism, openness |
| Dark Triad | 4 | psychopathy, narcissism, machiavellianism |

### Preprocessing
1. Filtered to items with `label_confidence ≥ 0.85`
2. Capped at 100 items per persona (randomly sampled)
3. Split items by `answer_matching_behavior` into matching/not-matching groups
4. Excluded personas with <5 items in either group (11 excluded)

## 4. Experiment Description

### Methodology

#### High-Level Approach
1. **Extract persona vectors**: For each persona, run matching and not-matching statements through Qwen2.5-7B-Instruct, extract last-token residual stream activations, compute difference-in-means
2. **Apply PCA**: Stack all 104 persona vectors, center, and decompose
3. **Analyze structure**: Explained variance, permutation tests, hierarchical clustering, component interpretation
4. **Validate causally**: Steer model with individual PCs and measure behavioral shift

#### Why This Method?
Contrastive difference-in-means is the standard method for extracting concept directions (Zou et al. 2023, Chen et al. 2025). PCA is the natural tool for discovering the principal axes of variation in a collection of vectors. We chose this over alternatives (ICA, NMF, SAE decomposition) because PCA provides interpretable variance-explained metrics and is most directly comparable to prior work.

### Implementation Details

#### Tools and Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.10.0 | Model inference |
| Transformers | 5.3.0 | Model loading |
| scikit-learn | latest | PCA, clustering |
| NumPy | latest | Linear algebra |
| SciPy | latest | Statistical tests, hierarchical clustering |

#### Model
- **Qwen2.5-7B-Instruct** (28 layers, hidden_dim=3584)
- Loaded in float16 on NVIDIA RTX A6000 (49GB)
- Layers extracted: [8, 16, 20, 24] (spanning early-to-late)

#### Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| min_confidence | 0.85 | Following Cintas et al. (2025) |
| max_samples_per_persona | 100 | Balance coverage vs compute |
| batch_size | 16 | Fits GPU memory comfortably |
| max_length | 128 | Statements are short |
| random_seed | 42 | Reproducibility |

### Experimental Protocol

#### Reproducibility
- Seeds set for Python random, NumPy, PyTorch, CUDA
- Model: Qwen/Qwen2.5-7B-Instruct (HuggingFace)
- Hardware: NVIDIA RTX A6000 × 4 (used 1 GPU)
- Extraction time: ~55 seconds for all 104 personas
- PCA analysis: <10 seconds
- Steering validation: ~10 minutes

## 5. Results

### 5.1 PCA Decomposition

#### Explained Variance by Layer

| Layer | PC1 | Top 5 | Top 10 | Top 20 | Components for 80% | Components for 90% |
|-------|-----|-------|--------|--------|--------------------|--------------------|
| 8 | 13.0% | 30.9% | 43.2% | 59.0% | 44 | 51 |
| 16 | 11.1% | 31.0% | 44.1% | 60.4% | 42 | 51 |
| **20** | **12.8%** | **33.8%** | **47.2%** | **63.4%** | **39** | **51** |
| 24 | 12.4% | 32.3% | 45.7% | 62.3% | 40 | 51 |

**Layer 20 shows the most concentrated persona structure**, requiring the fewest components (39) for 80% variance. This aligns with prior findings that persona information concentrates in later-middle layers (Cintas et al. 2025, Feng et al. 2026).

#### Effective Dimensionality (Participation Ratio)
| Layer | Effective Dim |
|-------|--------------|
| 8 | 20.9 |
| 16 | 22.7 |
| 20 | 20.2 |
| 24 | 21.0 |

The participation ratio of ~20 across all layers indicates that persona space has a consistent intrinsic dimensionality of approximately 20 dimensions — far less than the 104 individual persona vectors.

#### Statistical Significance (Permutation Test)
At all layers, **all 20 tested PCs are statistically significant** (p < 0.05, 100 permutations). PC1 has p = 0.00 at all layers. This confirms the PCA structure is not an artifact of random correlations.

### 5.2 Top Principal Component Interpretations (Layer 20)

| PC | Variance | High (+) Loadings | Low (-) Loadings | Interpretation |
|----|----------|-------------------|------------------|----------------|
| PC1 | 12.8% | believes-life-has-no-meaning, high-discount-rate, ends-justify-means | desire-to-influence-world, desire-for-acquiring-power, interest-in-art | **Nihilism/Dark vs. Agentic ambition** |
| PC2 | 8.1% | psychopathy, willingness-to-keep-secrets, machiavellianism | interest-in-art, interest-in-literature, interest-in-music | **Antisocial disposition vs. Cultural interests** |
| PC3 | 4.9% | desire-to-build-AIs-with-same-goals, ends-justify-means, desire-for-popularity | desire-to-minimize-impact, subscribes-to-Buddhism, believes-abortion-illegal | **AI power-seeking vs. Conservative restraint** |
| PC4 | 4.5% | risk-neutral, risk-averse, risk-seeking | subscribes-to-utilitarianism, subscribes-to-act-utilitarianism | **Risk attitude vs. Ethical framework** |
| PC5 | 3.4% | subscribes-to-Buddhism, subscribes-to-Hinduism, believes-in-gun-rights | risk-seeking, anti-LGBTQ-rights | **Religious identity vs. Social attitudes** |

### 5.3 Cross-Layer Alignment

| Layer Pair | Alignment |
|------------|-----------|
| 8 ↔ 16 | 0.532 |
| 8 ↔ 20 | 0.338 |
| 8 ↔ 24 | 0.285 |
| 16 ↔ 20 | 0.571 |
| 16 ↔ 24 | 0.469 |
| **20 ↔ 24** | **0.820** |

Adjacent later layers (20↔24) share highly aligned PC structure (0.82), while early and late layers diverge substantially (8↔24: 0.29). This suggests persona representations undergo significant transformation through the network, stabilizing in late layers.

### 5.4 Reconstruction Quality (Layer 20)

| PCs | Mean Cosine Sim | Min Cosine Sim |
|-----|-----------------|----------------|
| 1 | 0.263 | 0.002 |
| 5 | 0.526 | 0.221 |
| 10 | 0.635 | 0.300 |
| 20 | 0.758 | 0.493 |
| 30 | 0.826 | 0.572 |
| 50 | 0.906 | 0.675 |

20 PCs can reconstruct the average persona vector with 0.76 cosine similarity, but the worst-case persona still requires ~50 PCs for >0.67 similarity. This means the basis captures the "shared structure" well, but individual personas retain unique residuals.

### 5.5 Persona Vector Magnitudes by Category

| Category | Mean Norm | Std |
|----------|-----------|-----|
| Interests | 42.6 | 4.3 |
| Religion | 39.0 | 4.4 |
| AI Willingness | 36.4 | 7.7 |
| Politics | 35.5 | 6.5 |
| AI Desires | 34.7 | 4.7 |
| Big Five | 32.6 | 1.1 |
| Dark Triad | 30.6 | 3.4 |
| Ethics | 30.5 | 4.3 |
| AI Safety | 30.3 | 6.4 |

Interests and Religion produce the strongest persona vectors (highest L2 norms), suggesting these personas are most distinctly encoded in the residual stream. Ethics and AI Safety produce weaker vectors.

### 5.6 Hierarchical Clustering

The dendrogram (see `results/plots/dendrogram_layer20.png`) reveals natural clusters:
- **Religious personas** cluster tightly together
- **Interest personas** (art, math, science, music, literature, sports) form a distinct cluster
- **AI desire personas** form a large but diffuse cluster
- **Risk/time preference** personas cluster together
- **Dark triad** (psychopathy, narcissism, machiavellianism) cluster tightly with nihilism

### 5.7 Causal Steering Validation

Steering with PC1 at Layer 20 towards "believes-life-has-no-meaning":

| Steering Strength (α) | Agreement Rate |
|----------------------|----------------|
| -40 | 0.533 |
| -20 | 0.467 |
| -10 | 0.167 |
| -5 | 0.133 |
| 0 (baseline) | 0.100 |
| +5 | 0.067 |
| +10 | 0.067 |
| +20 | 0.167 |
| +40 | 0.400 |

The baseline model strongly disagrees with nihilistic statements (10% agreement). Positive steering with PC1 at α=40 increases agreement to 40% — a 4× increase. Interestingly, negative steering *also* increases agreement, suggesting that at high magnitudes, perturbation in either direction disrupts the model's default alignment and shifts it toward agreement. This U-shaped pattern is consistent with the "incoherence at extreme steering" phenomenon noted in prior work (Turner et al. 2023).

### 5.8 Semantic Category Clustering (Silhouette Scores)

| PCs | Layer 8 | Layer 16 | Layer 20 | Layer 24 |
|-----|---------|----------|----------|----------|
| 2 | -0.312 | -0.320 | -0.309 | -0.301 |
| 5 | -0.247 | -0.198 | -0.178 | -0.178 |
| 10 | -0.190 | -0.153 | -0.146 | -0.122 |
| 20 | -0.149 | -0.083 | -0.083 | -0.098 |

Negative silhouette scores indicate that **our human-defined semantic categories (Big Five, Politics, Ethics, etc.) do NOT align with the model's internal persona geometry**. The model's principal components of persona variation cut across these categories. This is a key finding: the model's "natural basis" for persona is different from taxonomies we'd impose.

## 6. Result Analysis

### Key Findings

1. **Persona space has ~20 effective dimensions** (participation ratio), far fewer than the 104 individual personas — confirming that persona vectors share substantial structure.

2. **20 PCs are statistically significant** by permutation test at all layers, confirming the decomposition captures real structure rather than noise.

3. **PC1 (12.8% variance) captures a nihilism/dark-vs-agentic dimension** — the single largest axis of variation separates nihilistic and dark personality traits from agentic AI-ambition traits.

4. **The model's natural persona basis does NOT map onto human categories** — Big Five traits, political orientations, and ethical frameworks are distributed across multiple PCs rather than forming their own axes. The model's geometry is organized differently from psychological taxonomies.

5. **Later layers show more concentrated persona structure** — Layer 20 requires only 39 components for 80% variance (vs. 44 at Layer 8), and cross-layer alignment is highest between layers 20 and 24 (0.82).

6. **PC directions have causal effects** — steering with PC1 shifts model behavior 4× on nihilism-related statements, though the dose-response relationship is non-linear and shows instability at extreme magnitudes.

### Hypothesis Testing

**H1: Persona vectors are not independent** — **SUPPORTED**. 20 effective dimensions for 104 personas; 39 PCs for 80% variance; all 20 top PCs significant.

**H2: Top PCs are interpretable** — **PARTIALLY SUPPORTED**. PC1-5 have clear semantic themes, but they cross-cut human categories rather than aligning with them.

**H3: Later layers show clearer structure** — **SUPPORTED**. Layer 20 has the most concentrated PCA structure and layers 20-24 are strongly aligned.

**H4: PCs can steer behavior** — **PARTIALLY SUPPORTED**. PC1 shows clear steering effects at high alpha, but weaker PCs (3-5) showed minimal effect in our experiment.

### Surprises and Insights

1. **The dominant axis is nihilism-vs-agency**, not a personality dimension like extraversion. This suggests the model's most fundamental persona variation is about whether to care/act, not how.

2. **Interests cluster very tightly and separately** from all other persona types, forming their own subspace (visible in PC2 and the dendrogram). The model sharply distinguishes "what you care about" from "who you are."

3. **Religious personas cluster by family** (Buddhism/Hinduism together, Christianity/Islam/Judaism together) in ways that mirror theological similarity.

4. **The Big Five traits do NOT dominate the PCA** despite being the most-studied personality framework. They're distributed across many PCs rather than occupying the top 5.

### Limitations

1. **Single model**: Results are for Qwen2.5-7B-Instruct only. Different architectures/sizes may yield different basis structures.

2. **Statement-based extraction**: We extracted from binary yes/no statements, not open-ended text. Persona vectors from free-form responses might differ.

3. **No cross-validation**: We used all data for both extraction and analysis. Held-out validation of the PCA structure would strengthen conclusions.

4. **Steering validation was limited**: Only tested 5 PCs with limited test statements. More comprehensive steering would better establish causal significance.

5. **Category assignment is manual**: Our grouping of personas into categories (Big Five, Politics, etc.) is subjective and imperfect.

6. **The Anthropic evals dataset is heavily weighted toward AI safety/alignment personas** (33 AI Desires + 9 AI Willingness + 6 AI Safety = 48/104 personas). A more balanced corpus might yield different structure.

## 7. Conclusions

### Summary
Persona representations in Qwen2.5-7B-Instruct's residual stream decompose into approximately 20 principal dimensions that capture the majority of variation across 104 diverse personas. These dimensions are statistically significant, interpretable, and partially causally active, but they do not align with standard psychological taxonomies — the model has its own "natural basis" for persona that cross-cuts human categories.

### Implications
- **For activation steering**: Rather than extracting individual persona vectors, practitioners could work with a ~20-dimensional basis and compose arbitrary personas via linear combination of basis components.
- **For interpretability**: The finding that PC1 separates nihilism/dark-traits from agency/ambition suggests a fundamental organizational principle in how this model represents persona.
- **For psychology**: The Big Five is not the model's natural decomposition. LLMs may encode personality along dimensions that reflect their training data distribution rather than human psychological structure.

### Confidence
Medium-high. The PCA structure is robust across layers and statistically significant. The specific PC interpretations are plausible but somewhat subjective. The causal validation is suggestive but limited.

## 8. Next Steps

### Immediate Follow-ups
1. **Cross-model comparison**: Run identical analysis on Llama-3-8B, Mistral-7B to test universality
2. **More comprehensive steering**: Test all 20 significant PCs with larger test sets and both directions
3. **ICA/NMF comparison**: Apply independent component analysis (non-orthogonal decomposition) to see if components become more interpretable

### Broader Extensions
1. **SAE intersection**: Decompose persona vectors using sparse autoencoders (as in Chen et al. 2025) and compare features to PCA components
2. **Dynamic PCA**: Track how persona PC structure changes during a conversation
3. **Balanced dataset**: Construct an equal-representation persona corpus spanning personality, culture, profession, values

### Open Questions
- Why is nihilism/agency the dominant axis? Is this an artifact of the Anthropic evals' AI-safety focus, or a genuine structural feature?
- Can the ~20-dimensional basis generalize to persona types not in the training set (zero-shot composition)?
- How does model scale affect the effective dimensionality of persona space?

## 9. References

1. Zou et al. (2023). "Representation Engineering: A Top-Down Approach to AI Transparency." arXiv:2310.01405.
2. Cintas et al. (2025). "Localizing Persona Representations in LLMs." arXiv:2505.24539.
3. Chen et al. (2025). "Persona Vectors: Monitoring and Controlling Character Traits." arXiv:2507.21509.
4. Feng et al. (2026). "PERSONA: Dynamic and Compositional Personality Control via Activation Vector Algebra." ICLR 2026.
5. Wang (2025). "The Geometry of Persona." arXiv:2512.07092.
6. Suh et al. (2024). "Rediscovering Latent Dimensions of Personality with LLMs." NeurIPS 2024.
7. Allbert et al. (2024). "Identifying and Manipulating Personality Traits in LLMs via Activation Engineering." arXiv:2412.10427.
8. Perez et al. (2023). "Discovering Language Model Behaviors with Model-Written Evaluations." ACL 2023.
9. Turner et al. (2023). "Activation Addition: Steering Language Models Without Optimization." arXiv:2308.10248.

## Appendix: File Structure

```
results/
├── persona_vectors_layer{8,16,20,24}.npy  # Extracted vectors (104 × 3584)
├── persona_names_final.json                # Ordered persona name list
├── extraction_config.json                  # Model/extraction parameters
├── pca_summary.json                        # PCA results summary
├── steering_results.json                   # Causal validation results
├── additional_analysis.json                # Effective dim, cross-layer, reconstruction
└── plots/
    ├── explained_variance.png              # Scree plot + cumulative variance
    ├── layer_comparison.png                # Cross-layer PCA comparison
    ├── pca_scatter_layer{8,16,20,24}.png   # 2D projections colored by category
    ├── pc_loadings_layer{8,16,20,24}.png   # Top persona loadings per PC
    ├── cosine_similarity_layer{8,16,20,24}.png  # Clustered similarity matrices
    ├── dendrogram_layer{8,16,20,24}.png    # Hierarchical clustering
    ├── permutation_test_layer{8,16,20,24}.png   # Statistical significance
    ├── cross_layer_alignment.png           # Layer-pair alignment heatmap
    ├── reconstruction_quality_layer20.png  # Reconstruction vs n_components
    ├── norms_by_category_layer20.png       # Vector magnitude by category
    └── steering_validation.png             # Steering dose-response curves
```
