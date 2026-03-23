# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "Basis Vectors in Persona Space" — investigating whether persona representations in LLMs can be decomposed into component vectors via PCA on the residual stream geometry.

---

## Papers
Total papers downloaded: 16

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Localizing Persona Representations in LLMs | Cintas et al. | 2025 | papers/2505.24539_localizing_persona_representations.pdf | PCA + Deep Scan on persona activations; 14 personas; layer-wise analysis |
| 2 | Identifying and Manipulating Personality Traits in LLMs | Allbert et al. | 2024 | papers/2412.10427_personality_traits_activation_engineering.pdf | 179 trait vectors via contrastive activation engineering |
| 3 | Persona Vectors: Monitoring and Controlling Character Traits | Chen et al. | 2025 | papers/2507.21509_persona_vectors_monitoring_controlling.pdf | Automated persona vector extraction pipeline; SAE decomposition |
| 4 | PERSONA: Dynamic and Compositional Personality Control | Feng et al. | 2026 | papers/2602.15669_persona_compositional_activation_vector_algebra.pdf | ICLR 2026; orthogonal OCEAN vectors; vector algebra |
| 5 | The Geometry of Persona | Wang | 2025 | papers/2512.07092_geometry_of_persona.pdf | Soul Engine; forced orthogonality via W^T W = I |
| 6 | BILLY: Steering via Merging Persona Vectors | Pai et al. | 2025 | papers/2510.10157_billy_steering_persona_vectors.pdf | Multi-persona blending for creative generation |
| 7 | Designing Role Vectors | Poterti et al. | 2025 | papers/2502.12055_designing_role_vectors.pdf | 29 professional role vectors; MMLU evaluation |
| 8 | Personalized Steering of LLMs | — | 2024 | papers/2406.00045_personalized_steering_vectors.pdf | Bi-directional preference optimization for steering vectors |
| 9 | Rediscovering Latent Dimensions of Personality | Suh et al. | 2024 | papers/2409.09905_latent_dimensions_personality_llm.pdf | SVD on log-probs recovers Big Five; 74.3% variance |
| 10 | Stable Personality Trait Evaluation via Internal Activations | — | 2026 | papers/2601.09833_stable_personality_trait_eval.pdf | Personality measurement from activations |
| 11 | Facet-Level Persona Control with Contrastive SAE | — | 2026 | papers/2602.19157_facet_level_persona_control_sae.pdf | SAE-based fine-grained persona control |
| 12 | Personality-Aware RL for Persuasive Dialogue | — | 2026 | papers/2601.06877_personality_aware_rl_persuasive.pdf | Personality-conditioned RL agents |
| 13 | Steerability of LLMs toward Data-Driven Personas | — | 2023 | papers/2311.04978_steerability_data_driven_personas.pdf | Early persona steerability study |
| 14 | Representation Engineering | Zou et al. | 2023 | papers/2310.01405_representation_engineering.pdf | Foundational RepE framework; LAT with PCA |
| 15 | Activation Addition (Steering Vectors) | Turner et al. | 2023 | papers/2308.10248_activation_addition_steering.pdf | Foundational activation steering method |
| 16 | Language Model Representations | — | 2023 | papers/2312.06681_language_model_representations.pdf | Analysis of LM internal representations |

See papers/README.md for detailed descriptions.

---

## Datasets
Total datasets downloaded: 4

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Anthropic Persona Evals | GitHub (anthropics/evals) | 135 files, ~1000 items each | Contrastive persona classification | datasets/anthropic-persona-evals/ | Primary dataset; 14+ personas across Big Five, Ethics, Politics |
| PersonalityBench / NPTI | GitHub (RUCAIBox/NPTI) | ~36K items/trait (search) + ~90/trait (test) | Personality evaluation | datasets/personalitybench/ | From Huang et al., 2024 |
| PersonaLLM | HuggingFace | 320 GPT-4 stories + BFI scores | SVD personality analysis | datasets/personallm/ | From Jiang et al., 2024 |
| BFI-44 | Manual compilation | 44 items | Personality assessment | datasets/bfi-44/ | Standard Big Five inventory in JSON/CSV |

See datasets/README.md for detailed descriptions and download instructions.

---

## Code Repositories
Total repositories cloned: 4

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| persona_vectors | github.com/safety-research/persona_vectors | Persona vector extraction + steering | code/persona_vectors/ | Key file: generate_vec.py; direct PCA candidate |
| representation-engineering | github.com/andyzoujm/representation-engineering | RepE framework with PCA | code/representation-engineering/ | Core: repe/rep_readers.py uses PCA |
| LLM-Persona-Steering | github.com/kaustpradalab | SAE-based personality steering | code/LLM-Persona-Steering/ | Complementary SAE approach |
| cross-model-persona-steering | github.com/sbayer2 | Cross-architecture persona transfer | code/cross-model-persona-steering/ | Universal persona representations |

See code/README.md for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder service (diligent mode) with query "persona vectors language models PCA decomposition" → 60 results
2. Filtered by relevance score (≥1) and title relevance to persona vectors, personality representations, and decomposition
3. Found arXiv IDs via web search for the 14 most relevant papers
4. Added 2 foundational papers on representation engineering and activation addition
5. Searched GitHub for code repositories mentioned in papers and related implementations
6. Searched HuggingFace and GitHub for datasets mentioned in the literature

### Selection Criteria
- Papers directly about persona/personality vectors in LLM activation space (highest priority)
- Papers using PCA/SVD decomposition of representations (high priority)
- Foundational representation engineering papers (methodology)
- Papers demonstrating compositionality or orthogonality of persona representations

### Challenges Encountered
- PERSONA paper (Feng et al., ICLR 2026) code URL not extractable from PDF
- "Persona Vectors in Controlling Hallucination" — no arXiv version found
- wget not available in environment; used curl for downloads

### Gaps and Workarounds
- No single paper performs PCA on a large diverse collection of persona vectors — this is the core research gap we aim to fill
- PERSONA-EVOLVE benchmark (800 scenarios) not publicly available yet
- PersonaHub dataset not downloaded (very large); can be accessed via HuggingFace if needed

---

## Recommendations for Experiment Design

Based on gathered resources:

1. **Primary dataset(s)**: Anthropic Persona Evals (14 personas, contrastive statements) — ideal for extracting diverse persona vectors for PCA decomposition. Supplement with PersonalityBench for Big Five validation.

2. **Baseline methods**:
   - Individual contrastive persona vectors (no decomposition)
   - OCEAN-aligned vectors from PERSONA paper
   - Random orthogonal directions in activation space
   - RepE LAT vectors as methodological baseline

3. **Evaluation metrics**:
   - PCA explained variance ratio (how many components capture persona variation?)
   - Cosine similarity between PCs and known trait directions
   - Steering effectiveness of individual principal components
   - Cross-dimensional leakage when steering with PCs
   - Cluster separation scores (Silhouette, Calinski-Harabasz)

4. **Code to adapt/reuse**:
   - `persona_vectors/generate_vec.py` — extract per-layer persona vectors for all 14 personas
   - `representation-engineering/repe/rep_readers.py` — PCA-based reading vector construction
   - Combine: extract vectors with persona_vectors pipeline → apply PCA from RepE framework → analyze resulting basis
