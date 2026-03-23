# Literature Review: Basis Vectors in Persona Space

## Research Area Overview

This review surveys work on how persona and personality traits are represented in the internal activation spaces of large language models (LLMs), with focus on whether these representations can be decomposed into independent basis vectors. The field sits at the intersection of representation engineering (mechanistic interpretability via linear probes on activations) and personality psychology (the Big Five / OCEAN model of personality). Recent work (2023-2026) has converged on a key finding: **personality traits are encoded as approximately linear directions in LLM residual streams**, and these directions are approximately orthogonal across traits, supporting algebraic manipulation.

---

## Key Papers

### Paper 1: Representation Engineering (Zou et al., 2023)
- **Authors**: Andy Zou, Long Phan, Sarah Chen, James Campbell, et al.
- **Year**: 2023 (updated 2025)
- **Source**: arXiv:2310.01405
- **Key Contribution**: Introduces Linear Artificial Tomography (LAT) — a general framework for reading and controlling concept representations in LLMs using PCA on contrastive activation differences.
- **Methodology**: Design contrastive stimuli varying along a target concept, extract hidden states, compute difference vectors between paired stimuli, apply PCA to find the first principal component as the "reading vector." Control via adding/subtracting this vector from activations.
- **Datasets Used**: TruthfulQA, ETHICS, ARC, AdvBench, StereoSet, MACHIAVELLI
- **Results**: Concepts (truthfulness, morality, emotion, bias, harmfulness) are linearly represented; compositionality holds (risk = f(probability, utility)); adding emotion vectors causally changes behavior.
- **Code Available**: Yes — github.com/andyzoujm/representation-engineering
- **Relevance**: **Foundational methodology.** LAT with PCA is the primary technique for extracting concept directions. The PCA variance analysis (first PC dominates at ~25%) provides a template for determining dimensionality of persona space. The compositionality finding directly motivates testing whether persona = weighted sum of basis trait vectors.

### Paper 2: Localizing Persona Representations in LLMs (Cintas et al., 2025)
- **Authors**: C. Cintas, Miriam Rateike, Erik Miehling, Elizabeth Daly, Skyler Speakman
- **Year**: 2025
- **Source**: arXiv:2505.24539 (IBM Research)
- **Key Contribution**: Identifies WHERE in the model personas are encoded using PCA and Deep Scan (non-parametric scan statistics), and maps the geometric overlap structure between persona representations.
- **Methodology**: Extract last-token activations at each layer for persona-specific statements; apply PCA for layer-level separation analysis; use Deep Scan (NPSS) to identify specific activation dimensions (binary masks) encoding each persona.
- **Datasets Used**: Anthropic persona evals (Perez et al., 2023) — 14 personas across Big Five, Ethics, Politics. 600 examples per persona, filtered to confidence ≥0.85.
- **Results**: Personas are separable in the final third of layers (layers 20-31); Ethics personas share 17.5% of activation dimensions (high polysemy); Politics personas are more distinct (9.4% overlap); topics occupy largely non-overlapping subspaces (~85-93% unique activations per topic).
- **Code Available**: No public code; uses scikit-learn + Deep Scan algorithm.
- **Relevance**: **Directly relevant.** Confirms personas are linearly separable via PCA in later layers. The activation overlap analysis (shared vs. unique dimensions) provides a complementary view to basis-vector decomposition. Gap: uses set-based localization rather than computing explicit basis directions.

### Paper 3: Persona Vectors (Chen et al., 2025)
- **Authors**: Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, Jack Lindsey
- **Year**: 2025
- **Source**: arXiv:2507.21509
- **Key Contribution**: Automated pipeline for extracting persona direction vectors via contrastive difference-in-means, with applications to monitoring (pre-generation projection), post-hoc steering, preventative steering during finetuning, and data screening.
- **Methodology**: LLM generates contrastive prompts + evaluation rubric; collect activations during response generation; persona vector = mean(positive activations) - mean(negative activations); "Response avg" method (average across response tokens) outperforms prompt-based extraction.
- **Datasets Used**: Custom LLM-generated Q&A for evil, sycophancy, hallucination; real-world datasets (LMSYS-CHAT-1M, TULU 3, WildChat); MMLU, HaluEval for capability evaluation.
- **Results**: Monitoring projections correlate r=0.75-0.83 with trait expression; finetuning shift correlations r=0.76-0.97; SAE decomposition reveals interpretable sub-features composing each persona vector. Cross-trait cosine similarities show clustering of negative traits.
- **Code Available**: Yes — github.com/safety-research/persona_vectors
- **Relevance**: **Core methodology.** The automated extraction pipeline and the cross-trait similarity analysis directly support the research hypothesis. The SAE decomposition is particularly relevant — it shows persona vectors can be further decomposed into finer-grained components. The open question they raise — "does there exist a natural 'persona basis'?" — is exactly our research question.

### Paper 4: PERSONA (Feng et al., ICLR 2026)
- **Authors**: Xingchen Feng et al.
- **Year**: 2026
- **Source**: arXiv:2602.15669 (ICLR 2026)
- **Key Contribution**: Demonstrates that Big Five personality trait vectors are approximately orthogonal in activation space and support vector algebra (scaling, addition, subtraction) for compositional personality control.
- **Methodology**: PERSONA-BASE extracts 10 vectors (2 poles × 5 OCEAN dimensions) via contrastive analysis at layer 20; PERSONA-ALGEBRA validates linear operations; PERSONA-FLOW dynamically composes vectors per conversational turn.
- **Datasets Used**: PersonalityBench (~90 questions/trait), PERSONA-EVOLVE (800 multi-turn scenarios), adapted BFI-44. Models: Qwen2.5-7B, LLaMA-3-8B, Ministral-8B.
- **Results**: Antonym pairs show strong negative cosine similarity (-0.506 to -0.908); cross-dimension pairs closer to zero but not perfectly orthogonal (max |cos| = 0.751 for Calm/Dependable); causal orthogonality ratio 8.8:1 (target vs. cross-dimensional effects); scalar multiplication achieves R² = 0.911-0.994; secondary effects under composition follow linear superposition exactly.
- **Code Available**: Stated as available but URL not extractable from PDF.
- **Relevance**: **Most directly relevant paper.** Validates the core thesis that personality vectors form approximate basis vectors supporting algebraic operations. Key finding: vectors are not perfectly orthogonal but orthogonal enough for practical composition with predictable cross-dimensional effects.

### Paper 5: The Geometry of Persona (Wang, 2025)
- **Authors**: Zhixiang Wang
- **Year**: 2025
- **Source**: arXiv:2512.07092
- **Key Contribution**: Introduces "Soul Engine" — enforces orthogonality between OCEAN trait directions via explicit regularization loss (W^T W = I) rather than discovering it via PCA.
- **Methodology**: Dual-head probe (Identity Head for clustering + Psychometric Head for OCEAN scores) on frozen LLM; orthogonality loss forces basis independence. Steering via mean embedding differences injected at middle layers (14-16).
- **Datasets Used**: Custom SoulBench with OCEAN labels from teacher model. Model: Qwen2.5-0.5B-Instruct.
- **Results**: Personality traits map to continuous manifold; orthogonal to reasoning circuits; middle layers optimal for steering.
- **Code Available**: No.
- **Relevance**: **Important methodological contribution.** The forced orthogonalization via W^T W = I provides an alternative to PCA-based discovery — one can enforce rather than discover basis independence. Limited by small model size (0.5B).

### Paper 6: Rediscovering Latent Dimensions of Personality (Suh et al., 2024)
- **Authors**: Suh et al.
- **Year**: 2024
- **Source**: arXiv:2409.09905 (NeurIPS 2024)
- **Key Contribution**: Applies SVD to LLM log-probabilities over trait-descriptive adjectives conditioned on personal stories, recovering Big Five factors as the top-5 principal components (74.3% variance explained).
- **Methodology**: SVD on observation matrix of log-probabilities for 100 Goldberg trait-descriptive adjectives. Loading matrix V shows how adjectives relate to factors; factor matrix U gives per-story personality scores.
- **Datasets Used**: PersonaLLM dataset (GPT-4 generated stories with prescribed Big Five traits). Models: Llama 3, Llama 3.1, Mixtral 8x22B.
- **Results**: Top-5 SVD components map to Big Five dimensions. Works in output probability space rather than internal activations.
- **Relevance**: **Validates the decomposition approach** in output space. Demonstrates that PCA/SVD naturally recovers personality structure. Our research extends this from output space to internal activation (residual stream) space.

### Paper 7: Identifying and Manipulating Personality Traits via Activation Engineering (Allbert et al., 2024)
- **Authors**: Allbert et al.
- **Year**: 2024
- **Source**: arXiv:2412.10427
- **Key Contribution**: Extracts personality direction vectors for 179 traits via contrastive activation engineering; uses weight orthogonalization for trait manipulation.
- **Methodology**: Mean-difference between trait-expressing and neutral prompt activations; weight orthogonalization-based "feature induction." Layer 18 is optimal.
- **Relevance**: Scale of 179 traits provides rich material for PCA — extracting vectors for many traits and applying PCA could reveal the true dimensionality of persona space.

### Paper 8: BILLY (Pai et al., 2025)
- **Authors**: Pai et al.
- **Year**: 2025
- **Source**: arXiv:2510.10157
- **Key Contribution**: Merges multiple persona vectors (e.g., environmentalist + creative professional) for multi-perspective creative generation, replacing multi-agent frameworks.
- **Relevance**: Demonstrates persona vectors support linear combination — a necessary property of basis vectors.

### Paper 9: Designing Role Vectors (Poterti et al., 2025)
- **Authors**: Poterti et al.
- **Year**: 2025
- **Source**: arXiv:2502.12055
- **Key Contribution**: Constructs 29 role vectors (chemist, lawyer, etc.) from contrastive activations using PersonaHub. Evaluates on MMLU.
- **Relevance**: 29 role vectors provide another set for PCA — are professional roles decomposable into a smaller set of basis directions?

---

## Common Methodologies

1. **Contrastive Activation Analysis**: Used in Papers 2-5, 7-9. Extract activations for trait-expressing vs. trait-suppressing prompts; compute difference-in-means to get direction vectors.
2. **PCA / SVD Decomposition**: Used in Papers 1, 2, 6. Apply to difference vectors (RepE) or log-probabilities (Suh) to find principal directions.
3. **Steering via Activation Addition**: Used in Papers 1, 3-5, 7-9. Add/subtract direction vectors from residual stream during inference.
4. **SAE Decomposition**: Used in Paper 3 (appendix). Decompose persona vectors into sparse autoencoder features for finer-grained interpretability.

## Standard Baselines
- Simple system prompting (persona description in prompt)
- P² Induction (Jiang et al., 2023)
- NPTI (neuron-based personality trait induction)
- ActAdd (Turner et al., 2023)
- Supervised fine-tuning with LoRA (upper bound)

## Evaluation Metrics
- **Trait expression scores**: 0-100 scale via LLM judge (GPT-4.1-mini)
- **BFI-44**: Adapted behavioral questionnaire, 5-point Likert
- **PersonalityBench**: Situational questions per OCEAN dimension
- **Cluster separation**: Silhouette, Calinski-Harabasz, Davies-Bouldin scores
- **Orthogonality**: Cosine similarity matrices, causal orthogonality ratios
- **Capability preservation**: MMLU, TruthfulQA

## Datasets in the Literature
- **Anthropic Persona Evals** (Perez et al., 2023): Used in Papers 2, 3. 14 personas, binary statements. [Downloaded]
- **PersonalityBench** (Deng et al., 2025): Used in Paper 4. ~90 questions/trait. [Downloaded]
- **PersonaLLM** (Jiang et al., 2024): Used in Paper 6. 320 stories with Big Five profiles. [Downloaded]
- **BFI-44**: Standard personality inventory. [Downloaded]
- **PersonaHub** (Ge et al., 2024): Used in Paper 9. Large-scale persona descriptions.
- **PERSONA-EVOLVE**: Proposed in Paper 4. 800 multi-turn scenarios.

---

## Gaps and Opportunities

1. **No systematic PCA of diverse persona vectors**: Papers extract individual trait vectors or analyze small sets. No one has extracted vectors for a large diverse set of personas (combining personality, ethics, politics, professional roles) and applied PCA to the full collection to discover the true dimensionality and primary components.

2. **Forced vs. discovered orthogonality**: Paper 5 forces orthogonality; Paper 4 finds approximate orthogonality. Neither systematically tests whether PCA on a large collection naturally recovers interpretable basis vectors.

3. **Layer-specific decomposition**: Papers agree later layers are key (layers 16-31 for 32-layer models), but no systematic comparison of PCA decomposition structure across layers.

4. **Cross-model universality of basis structure**: Paper by sbayer2 shows persona vectors transfer across models, but no one has compared the PCA basis structure across different model families.

5. **Beyond Big Five**: Most work assumes the OCEAN model. PCA on a broader set of persona vectors (including ethical, political, professional dimensions) could reveal whether Big Five is the natural basis or whether a different decomposition emerges from LLM internals.

---

## Recommendations for Our Experiment

Based on this literature review:

- **Primary dataset**: Anthropic Persona Evals — 14 personas across 3 topics provides diverse persona dimensions ideal for PCA decomposition. Supplement with PersonalityBench for Big Five validation.
- **Recommended approach**: Extract persona vectors using contrastive difference-in-means (Chen et al. pipeline from persona_vectors repo), then apply PCA to the collection of all 14+ vectors to discover basis components.
- **Key models**: Qwen2.5-7B-Instruct or LLaMA-3-8B-Instruct (most studied in the literature, layer 20 appears optimal).
- **Evaluation**: Cosine similarity matrix between extracted vectors; PCA explained variance ratio; comparison of discovered components to known personality dimensions; causal validation via steering with individual PCs.
- **Baselines**: Individual contrastive vectors (no decomposition); OCEAN-aligned vectors; random orthogonal directions.
- **Metrics**: Explained variance ratio, cosine similarity with known trait directions, steering effectiveness of individual PCs, cluster separation scores.
