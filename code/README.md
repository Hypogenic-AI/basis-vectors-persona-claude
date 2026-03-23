# Cloned Research Repositories

Reference code repositories for the "Basis Vectors in Persona Space" research project.
Cloned on 2026-03-23.

---

## 1. persona_vectors (safety-research)

- **Source**: https://github.com/safety-research/persona_vectors
- **Paper**: "Persona Vectors: Monitoring and Controlling Character Traits in Language Models" (Chen et al., arXiv:2507.21509)
- **Path**: `persona_vectors/`

### Overview
Official implementation for extracting and applying persona vectors to monitor and control character traits in LLMs. Identifies directions in activation space corresponding to traits like evil, sycophancy, and hallucination propensity. Vectors can be used for both inference-time steering and training-time prevention.

### Key Scripts
| Script | Purpose |
|--------|---------|
| `generate_vec.py` | Compute persona vectors from contrastive activations |
| `activation_steer.py` | Apply persona vectors during inference |
| `eval/eval_persona.py` | Evaluate persona expression with LLM judge |
| `eval/cal_projection.py` | Calculate activation projections onto persona vectors |
| `training.py` | Fine-tuning with optional steering during training |
| `scripts/generate_vec.sh` | Full vector generation pipeline |
| `scripts/eval_steering.sh` | Evaluate steering effectiveness |

### Dependencies
- PyTorch, Transformers (HuggingFace)
- OpenAI API (for judge-based evaluation)
- Qwen2.5-7B-Instruct (primary model used in paper)

### Relevance to Our Research
**HIGH** -- Direct implementation of persona vector extraction via contrastive activation differences. The `generate_vec.py` script produces per-layer vectors of shape `[layers x hidden_dim]`. These vectors are exactly the kind of representation we want to decompose with PCA/SVD to find basis vectors in persona space. The projection calculation in `eval/cal_projection.py` is also directly relevant.

---

## 2. representation-engineering (andyzoujm)

- **Source**: https://github.com/andyzoujm/representation-engineering
- **Paper**: "Representation Engineering: A Top-Down Approach to AI Transparency" (Zou et al., arXiv:2310.01405)
- **Path**: `representation-engineering/`

### Overview
Foundational RepE framework that introduces population-level representation analysis for LLM transparency and control. Provides RepReading (monitoring) and RepControl (steering) pipelines integrated with HuggingFace.

### Key Components
| Component | Purpose |
|-----------|---------|
| `repe/rep_reading_pipeline.py` | RepReading pipeline for extracting and classifying representations |
| `repe/rep_control_pipeline.py` | RepControl pipeline for steering model behavior |
| `repe/rep_control_contrast_vec.py` | Contrastive activation vector computation |
| `repe/rep_readers.py` | Readers that extract representation directions |
| `examples/honesty/` | Honesty control experiments |
| `examples/fairness/` | Fairness-related experiments |
| `examples/primary_emotions/` | Emotion steering experiments |
| `examples/memorization/` | Memorization detection |
| `repe_eval/` | Evaluation framework based on RepReading |

### Dependencies
- PyTorch, Transformers (HuggingFace)
- scikit-learn (for PCA in rep_readers)
- Install via: `pip install -e .`

### Relevance to Our Research
**HIGH** -- The RepE framework is the methodological foundation for our work. The `rep_readers.py` module already uses PCA to find reading directions in activation space. The contrastive vector computation in `rep_control_contrast_vec.py` provides the baseline approach we extend. The examples (honesty, emotions, etc.) demonstrate single-trait steering that we aim to decompose into basis vectors.

---

## 3. LLM-Persona-Steering (kaustpradalab)

- **Source**: https://github.com/kaustpradalab/LLM-Persona-Steering
- **Paper**: "Exploring the Personality Traits of LLMs through Latent Features Steering" (Yang et al., arXiv:2410.10863)
- **Path**: `LLM-Persona-Steering/`

### Overview
Explores origins of personality in LLMs by steering via both long-term (SAE-based) and short-term features. Uses Sparse Autoencoders (SAEs) to extract latent features corresponding to personality factors (Big Five, Short Dark Triad). Demonstrates training-free personality modification.

### Key Components
| Component | Purpose |
|-----------|---------|
| `src/stimius_processing/SAE/` | Background generation and feature extraction via SAEs |
| `src/steer_experiments/SAE/sae_run.py` | Run steering experiments with SAE features |
| `src/steer_experiments/SAE/analysis.py` | Analyze steering results |
| `src/data/` | Test datasets for personality evaluation |

### Dependencies
- PyTorch, Transformers
- SAE libraries (sparse autoencoder tooling)
- See `requirements.txt`

### Relevance to Our Research
**MEDIUM-HIGH** -- Complementary approach using SAEs instead of PCA/SVD for feature extraction. The SAE-based latent features could be compared against our PCA-derived basis vectors. Their personality evaluation framework (BFI, SD-3) provides benchmarks for assessing our decomposed persona dimensions.

---

## 4. cross-model-persona-steering (sbayer2)

- **Source**: https://github.com/sbayer2/cross-model-persona-steering
- **Paper**: Extension of Chen et al. (2024) persona vectors work
- **Path**: `cross-model-persona-steering/`

### Overview
Research implementation demonstrating cross-architecture persona vector transfer. Extracts persona vectors from Qwen2.5-7B and applies them to steer GPT-OSS 20B behavior, demonstrating that personality representations may transcend specific model architectures. Includes a web interface for interactive exploration.

### Key Components
| Component | Purpose |
|-----------|---------|
| `backend/main.py` | FastAPI server with steering API endpoints |
| `backend/models.py` | Dual-architecture model loading (HuggingFace + GGUF) |
| `backend/persona_vectors.py` | Vector extraction with dynamic layer selection |
| `backend/prompts.py` | Trait definitions and custom trait generation |

### Dependencies
- FastAPI, PyTorch, Transformers
- llama-cpp-python (for GGUF models)
- Chart.js (frontend visualization)

### Relevance to Our Research
**MEDIUM** -- Demonstrates that persona vectors transfer across architectures, supporting the hypothesis that personality representations occupy a universal subspace. The dynamic layer selection algorithm and cross-model effectiveness metrics inform our basis vector analysis. However, uses parameter modulation for GGUF models rather than direct activation steering.

---

## Papers Without Public Code Repositories

### PERSONA (Feng et al., ICLR 2026)
- **Paper**: "PERSONA: Dynamic and Compositional Inference-Time Personality Control via Activation Vector Algebra" (arXiv:2602.15669)
- **Status**: No public code repository found as of 2026-03-23
- **Key ideas**: Persona-Base (orthogonal trait vector extraction via contrastive activation analysis), Persona-Algebra (vector arithmetic for intensity/composition/suppression), Persona-Flow (context-aware dynamic composition). Achieves 9.60 on PersonalityBench (near SFT upper bound of 9.61) without gradient updates.
- **Relevance**: **VERY HIGH** -- Directly addresses compositional personality control via activation vector algebra. Their finding that trait vectors are approximately orthogonal strongly supports the basis vector hypothesis. Monitor for code release.

### Rediscovering the Latent Dimensions of Personality (Suh et al., NeurIPS 2024)
- **Paper**: "Rediscovering the Latent Dimensions of Personality with Large Language Models as Trait Descriptors" (arXiv:2409.09905)
- **Status**: No dedicated code repository found. Authors have related repos: github.com/JosephJeesungSuh/subpop
- **Key ideas**: Applies SVD to log-probabilities of trait-descriptive adjectives. LLMs "rediscover" Big Five personality traits (extraversion, agreeableness, conscientiousness, neuroticism, openness) without questionnaire inputs. Top-5 factors explain 74.3% of variance.
- **Relevance**: **VERY HIGH** -- Directly uses SVD decomposition to find latent personality dimensions, which is the core methodology of our research. Their approach operates on output log-probabilities rather than internal activations, providing a complementary perspective.

### The Geometry of Persona (arXiv:2512.07092)
- **Paper**: "The Geometry of Persona: Disentangling Personality from Reasoning in Large Language Models"
- **Status**: No public code repository found as of 2026-03-23
- **Key ideas**: Soul Engine framework using Linear Representation Hypothesis to extract disentangled personality vectors via dual-head architecture. Introduces SoulBench dataset. Achieves MSE of 0.011 against psychological ground truth.
- **Relevance**: **HIGH** -- Focuses on disentangling personality from reasoning in representation space, directly related to our basis vector decomposition goals.
