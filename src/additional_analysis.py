"""
Additional analysis: cross-layer PC alignment, effective dimensionality,
and comparison of individual persona vectors vs PC reconstruction.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine

RESULTS_DIR = "/workspaces/basis-vectors-persona-claude/results"
PLOTS_DIR = "/workspaces/basis-vectors-persona-claude/results/plots"

np.random.seed(42)


def load_all():
    with open(os.path.join(RESULTS_DIR, "persona_names_final.json")) as f:
        names = json.load(f)
    with open(os.path.join(RESULTS_DIR, "extraction_config.json")) as f:
        config = json.load(f)
    vectors = {}
    for l in config["layers"]:
        vectors[l] = np.load(os.path.join(RESULTS_DIR, f"persona_vectors_layer{l}.npy"))
    return names, vectors, config


def effective_dimensionality(evr):
    """Participation ratio: (sum(p_i))^2 / sum(p_i^2)"""
    evr = np.array(evr)
    evr = evr[evr > 0]
    return (evr.sum() ** 2) / (evr ** 2).sum()


def cross_layer_alignment(vectors, layers):
    """Measure how well PC directions align across layers."""
    pcas = {}
    for l in layers:
        X = vectors[l] - vectors[l].mean(axis=0)
        pca = PCA(n_components=20)
        pca.fit(X)
        pcas[l] = pca

    # For each pair of layers, compute alignment of top PCs
    # Alignment = |cos similarity| between corresponding PCs
    n_pcs = 10
    alignment_matrix = np.zeros((len(layers), len(layers)))

    for i, l1 in enumerate(layers):
        for j, l2 in enumerate(layers):
            if l1 == l2:
                alignment_matrix[i, j] = 1.0
                continue
            # Subspace alignment: use principal angles
            # Simpler: correlation between PC projections
            X1 = vectors[l1] - vectors[l1].mean(axis=0)
            X2 = vectors[l2] - vectors[l2].mean(axis=0)
            proj1 = pcas[l1].transform(X1)[:, :n_pcs]
            proj2 = pcas[l2].transform(X2)[:, :n_pcs]

            # Canonical correlation between projections
            # Use simpler: average absolute correlation between corresponding PCs
            corrs = []
            for k in range(n_pcs):
                r = np.abs(np.corrcoef(proj1[:, k], proj2[:, k])[0, 1])
                corrs.append(r)
            alignment_matrix[i, j] = np.mean(corrs)

    return alignment_matrix, layers


def reconstruction_quality(vectors, names, layer):
    """How well can k PCs reconstruct individual persona vectors?"""
    X = vectors[layer]
    X_centered = X - X.mean(axis=0)

    k_values = [1, 2, 3, 5, 10, 20, 30, 50]
    reconstruction_errors = {}

    for k in k_values:
        pca = PCA(n_components=k)
        X_proj = pca.fit_transform(X_centered)
        X_recon = pca.inverse_transform(X_proj)

        # Per-persona cosine similarity between original and reconstructed
        cos_sims = []
        for i in range(len(names)):
            sim = 1 - cosine(X_centered[i], X_recon[i])
            cos_sims.append(sim)
        reconstruction_errors[k] = {
            "mean_cos_sim": float(np.mean(cos_sims)),
            "min_cos_sim": float(np.min(cos_sims)),
            "max_cos_sim": float(np.max(cos_sims)),
            "std_cos_sim": float(np.std(cos_sims)),
        }

    return reconstruction_errors


def plot_reconstruction(recon_errors, layer):
    """Plot reconstruction quality vs number of PCs."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ks = sorted(recon_errors.keys())
    means = [recon_errors[k]["mean_cos_sim"] for k in ks]
    mins = [recon_errors[k]["min_cos_sim"] for k in ks]
    maxs = [recon_errors[k]["max_cos_sim"] for k in ks]

    ax.plot(ks, means, 'o-', color='steelblue', label='Mean', linewidth=2)
    ax.fill_between(ks, mins, maxs, alpha=0.2, color='steelblue', label='Min-Max range')
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cosine Similarity (Original vs Reconstructed)")
    ax.set_title(f"Persona Vector Reconstruction Quality (Layer {layer})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"reconstruction_quality_layer{layer}.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_cross_layer_alignment(alignment_matrix, layers):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(alignment_matrix, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=[f"Layer {l}" for l in layers],
                yticklabels=[f"Layer {l}" for l in layers], ax=ax)
    ax.set_title("Cross-Layer PC Alignment\n(Mean |correlation| of top 10 PCs)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cross_layer_alignment.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_vector_norms_by_category(vectors, names, layer):
    """Analyze if different persona categories have different vector magnitudes."""
    from src.pca_analysis import categorize_personas
    categories = categorize_personas(names)

    norms = np.linalg.norm(vectors[layer], axis=1)

    cat_norms = {}
    for i, name in enumerate(names):
        cat = categories.get(name, "Other")
        if cat not in cat_norms:
            cat_norms[cat] = []
        cat_norms[cat].append(norms[i])

    fig, ax = plt.subplots(figsize=(10, 5))
    cats = sorted(cat_norms.keys(), key=lambda c: -np.mean(cat_norms[c]))
    positions = range(len(cats))

    bp = ax.boxplot([cat_norms[c] for c in cats], positions=positions, widths=0.6, patch_artist=True)
    ax.set_xticks(positions)
    ax.set_xticklabels(cats, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Persona Vector L2 Norm")
    ax.set_title(f"Persona Vector Magnitude by Category (Layer {layer})")
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"norms_by_category_layer{layer}.png"), dpi=150, bbox_inches='tight')
    plt.close()

    return {c: {"mean": float(np.mean(cat_norms[c])), "std": float(np.std(cat_norms[c])),
                "n": len(cat_norms[c])} for c in cats}


def main():
    print("Loading data...")
    names, vectors, config = load_all()
    layers = config["layers"]

    # Effective dimensionality
    print("\nEffective Dimensionality (Participation Ratio):")
    eff_dims = {}
    for l in layers:
        X = vectors[l] - vectors[l].mean(axis=0)
        pca = PCA(n_components=min(len(names), 50))
        pca.fit(X)
        ed = effective_dimensionality(pca.explained_variance_ratio_)
        eff_dims[l] = ed
        print(f"  Layer {l}: {ed:.1f}")

    # Cross-layer alignment
    print("\nCross-layer PC alignment:")
    alignment_matrix, _ = cross_layer_alignment(vectors, layers)
    plot_cross_layer_alignment(alignment_matrix, layers)
    for i, l1 in enumerate(layers):
        for j, l2 in enumerate(layers):
            if j > i:
                print(f"  Layer {l1} ↔ Layer {l2}: {alignment_matrix[i,j]:.3f}")

    # Reconstruction quality
    best_layer = 20
    print(f"\nReconstruction quality (Layer {best_layer}):")
    recon = reconstruction_quality(vectors, names, best_layer)
    for k, v in sorted(recon.items()):
        print(f"  {k} PCs: mean cos_sim = {v['mean_cos_sim']:.4f} (min={v['min_cos_sim']:.4f})")
    plot_reconstruction(recon, best_layer)

    # Vector norms by category
    print(f"\nVector norms by category (Layer {best_layer}):")
    cat_norms = plot_vector_norms_by_category(vectors, names, best_layer)
    for cat, stats in sorted(cat_norms.items(), key=lambda x: -x[1]["mean"]):
        print(f"  {cat}: {stats['mean']:.2f} ± {stats['std']:.2f} (n={stats['n']})")

    # Save additional results
    additional = {
        "effective_dimensionality": {str(k): v for k, v in eff_dims.items()},
        "cross_layer_alignment": alignment_matrix.tolist(),
        "reconstruction_quality": {str(k): v for k, v in recon.items()},
        "category_norms": cat_norms,
    }

    with open(os.path.join(RESULTS_DIR, "additional_analysis.json"), "w") as f:
        json.dump(additional, f, indent=2)

    print("\nAll additional analysis complete!")


if __name__ == "__main__":
    main()
