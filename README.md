# Basis Vectors in Persona Space

**Research Question**: Can persona representations in LLMs be decomposed via PCA into interpretable basis components?

## Key Findings

- **104 persona vectors** extracted from Qwen2.5-7B-Instruct's residual stream decompose into **~20 effective dimensions** (participation ratio), far fewer than the 104 individual personas
- **PC1 (12.8% variance)** separates nihilistic/dark-triad traits from agentic AI-ambition traits — the single largest axis of persona variation
- **All 20 top PCs are statistically significant** by permutation test (p < 0.05)
- **39 components** capture 80% of total persona variance at Layer 20
- The model's **natural persona basis does NOT align** with standard psychological taxonomies (Big Five) — it has its own geometry
- **Causal validation**: Steering with PC1 shifts model behavior 4× on nihilism statements

## Quick Start

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install torch transformers accelerate numpy scipy scikit-learn matplotlib seaborn pandas tqdm

# Run extraction (~1 min on A6000)
python src/extract_persona_vectors.py

# Run PCA analysis (~10 sec)
python src/pca_analysis.py

# Run steering validation (~10 min)
python src/steering_validation.py

# Run additional analysis
PYTHONPATH=. python src/additional_analysis.py
```

## File Structure

```
├── REPORT.md                  # Full research report with results
├── planning.md                # Research plan and methodology
├── src/
│   ├── extract_persona_vectors.py  # Extract persona vectors from model
│   ├── pca_analysis.py             # PCA decomposition and visualization
│   ├── steering_validation.py      # Causal validation via activation steering
│   └── additional_analysis.py      # Cross-layer alignment, reconstruction quality
├── results/
│   ├── persona_vectors_layer*.npy  # Extracted vectors
│   ├── pca_summary.json            # PCA results
│   ├── steering_results.json       # Steering experiment results
│   └── plots/                      # All visualizations
├── datasets/
│   └── anthropic-persona-evals/    # 135 persona evaluation datasets
├── papers/                         # Related research papers
└── literature_review.md            # Background literature survey
```

See [REPORT.md](REPORT.md) for full details.
