"""
PCA decomposition and analysis of persona vectors.

1. Load persona vectors from all layers
2. Apply PCA to discover principal components of persona space
3. Analyze explained variance, component loadings, clustering
4. Compare structure across layers
5. Statistical validation (permutation test, Marchenko-Pastur)
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Config
RESULTS_DIR = "/workspaces/basis-vectors-persona-claude/results"
PLOTS_DIR = "/workspaces/basis-vectors-persona-claude/results/plots"
SEED = 42
np.random.seed(SEED)

os.makedirs(PLOTS_DIR, exist_ok=True)


def load_data():
    """Load persona vectors and metadata."""
    with open(os.path.join(RESULTS_DIR, "persona_names_final.json")) as f:
        names = json.load(f)
    with open(os.path.join(RESULTS_DIR, "extraction_config.json")) as f:
        config = json.load(f)

    layers = config["layers"]
    vectors = {}
    for l in layers:
        vectors[l] = np.load(os.path.join(RESULTS_DIR, f"persona_vectors_layer{l}.npy"))

    return names, vectors, config


def categorize_personas(names):
    """Assign semantic categories to persona names for interpretation."""
    categories = {}
    for name in names:
        if any(x in name for x in ["agreeableness", "conscientiousness", "extraversion",
                                      "neuroticism", "openness"]):
            categories[name] = "Big Five"
        elif any(x in name for x in ["conservative", "liberal", "immigration", "LGBTQ",
                                       "gun-rights", "abortion"]):
            categories[name] = "Politics"
        elif any(x in name for x in ["utilitarianism", "deontology", "virtue-ethics",
                                       "nihilism", "relativism", "act-util", "average-util",
                                       "rule-util", "total-util"]):
            categories[name] = "Ethics"
        elif any(x in name for x in ["Atheism", "Buddhism", "Christianity", "Confucianism",
                                       "Hinduism", "Islam", "Judaism", "Taoism"]):
            categories[name] = "Religion"
        elif any(x in name for x in ["interest-in-"]):
            categories[name] = "Interests"
        elif any(x in name for x in ["desire-for-", "desire-to-"]):
            categories[name] = "AI Desires"
        elif any(x in name for x in ["willingness-to-", "okay-with-"]):
            categories[name] = "AI Willingness"
        elif any(x in name for x in ["risk-", "discount-"]):
            categories[name] = "Risk/Time Pref"
        elif any(x in name for x in ["machiavellianism", "narcissism", "psychopathy",
                                       "ends-justify-means"]):
            categories[name] = "Dark Triad"
        elif any(x in name for x in ["self-replication", "no-shut-down", "no-goal-change",
                                       "resource-acquisition", "optionality"]):
            categories[name] = "AI Safety"
        elif any(x in name for x in ["believes-it-", "believes-life-", "believes-AIs-"]):
            categories[name] = "AI Beliefs"
        elif any(x in name for x in ["disability"]):
            categories[name] = "Identity"
        elif any(x in name for x in ["HHH", "helpful", "being-helpful"]):
            categories[name] = "Alignment"
        elif any(x in name for x in ["cognitive-enhancement", "aesthetic", "stands-its-ground",
                                       "willingness-to-defer"]):
            categories[name] = "Other Traits"
        else:
            categories[name] = "Other"
    return categories


def marchenko_pastur_threshold(n_samples, n_features, variance=1.0):
    """Compute the Marchenko-Pastur upper bound for random matrix eigenvalues."""
    gamma = n_samples / n_features
    lambda_plus = variance * (1 + 1/np.sqrt(gamma))**2
    return lambda_plus


def run_pca_analysis(vectors, names, layer_idx):
    """Run PCA and return detailed results for one layer."""
    X = vectors.copy()
    n_personas, hidden_dim = X.shape

    # Normalize vectors to unit length (direction matters, not magnitude)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_normed = X / (norms + 1e-8)

    # Also try centered (zero-mean) but not scaled
    X_centered = X - X.mean(axis=0)

    # PCA on centered data
    pca = PCA(n_components=min(n_personas, 50))
    X_pca = pca.fit_transform(X_centered)

    # PCA on normalized data
    pca_normed = PCA(n_components=min(n_personas, 50))
    X_pca_normed = pca_normed.fit_transform(X_normed - X_normed.mean(axis=0))

    # Explained variance
    evr = pca.explained_variance_ratio_
    cumulative_evr = np.cumsum(evr)

    # How many components for 80% and 90% variance
    n_80 = np.searchsorted(cumulative_evr, 0.80) + 1
    n_90 = np.searchsorted(cumulative_evr, 0.90) + 1

    # Marchenko-Pastur threshold
    mp_threshold = marchenko_pastur_threshold(n_personas, hidden_dim)
    # Convert to explained variance ratio scale
    total_var = pca.explained_variance_.sum() + (hidden_dim - len(pca.explained_variance_)) * 0
    significant_components = np.sum(pca.explained_variance_ > mp_threshold * np.var(X_centered))

    # Component loadings: correlation of each persona with each PC
    loadings = X_pca[:, :10]  # top 10 PCs

    # Cosine similarity matrix between persona vectors
    cos_sim = (X_normed @ X_normed.T)

    results = {
        "layer": layer_idx,
        "n_personas": n_personas,
        "hidden_dim": hidden_dim,
        "explained_variance_ratio": evr.tolist(),
        "cumulative_evr": cumulative_evr.tolist(),
        "n_components_80pct": int(n_80),
        "n_components_90pct": int(n_90),
        "top1_variance": float(evr[0]),
        "top5_variance": float(cumulative_evr[4]) if len(cumulative_evr) >= 5 else None,
        "top10_variance": float(cumulative_evr[9]) if len(cumulative_evr) >= 10 else None,
        "top20_variance": float(cumulative_evr[19]) if len(cumulative_evr) >= 20 else None,
        "significant_components_mp": int(significant_components),
        "pca_object": pca,
        "X_pca": X_pca,
        "loadings": loadings,
        "cos_sim": cos_sim,
        "X_centered": X_centered,
        "norms": norms.flatten(),
    }

    return results


def permutation_test(X, n_permutations=100):
    """Test if PCA structure is significantly different from random permutation."""
    n, d = X.shape
    X_centered = X - X.mean(axis=0)
    pca = PCA(n_components=min(n, 20))
    pca.fit(X_centered)
    real_evr = pca.explained_variance_ratio_

    perm_evrs = []
    for _ in range(n_permutations):
        X_perm = X_centered.copy()
        for col in range(d):
            np.random.shuffle(X_perm[:, col])
        pca_perm = PCA(n_components=min(n, 20))
        pca_perm.fit(X_perm)
        perm_evrs.append(pca_perm.explained_variance_ratio_)

    perm_evrs = np.array(perm_evrs)
    # p-value: fraction of permutations where PC1 explains more variance
    p_values = [(perm_evrs[:, i] >= real_evr[i]).mean() for i in range(min(20, len(real_evr)))]

    return real_evr, perm_evrs, p_values


def plot_explained_variance(all_results, names, categories):
    """Plot explained variance across layers."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Individual component variance
    for res in all_results:
        layer = res["layer"]
        evr = res["explained_variance_ratio"][:30]
        axes[0].plot(range(1, len(evr)+1), evr, 'o-', label=f"Layer {layer}", markersize=3)

    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance Ratio")
    axes[0].set_title("Scree Plot: Per-Component Variance")
    axes[0].legend()
    axes[0].set_xlim(0, 31)
    axes[0].grid(True, alpha=0.3)

    # Right: Cumulative variance
    for res in all_results:
        layer = res["layer"]
        cum_evr = res["cumulative_evr"][:50]
        axes[1].plot(range(1, len(cum_evr)+1), cum_evr, '-', label=f"Layer {layer}", linewidth=2)

    axes[1].axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label="80% threshold")
    axes[1].axhline(y=0.9, color='gray', linestyle=':', alpha=0.5, label="90% threshold")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Explained Variance")
    axes[1].set_title("Cumulative Explained Variance")
    axes[1].legend(fontsize=8)
    axes[1].set_xlim(0, 51)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "explained_variance.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_cosine_similarity(cos_sim, names, categories, layer_idx):
    """Plot cosine similarity matrix with hierarchical clustering."""
    # Cluster the personas
    linkage_matrix = linkage(pdist(1 - cos_sim), method='ward')

    fig, ax = plt.subplots(figsize=(16, 14))

    # Get cluster ordering
    from scipy.cluster.hierarchy import leaves_list
    order = leaves_list(linkage_matrix)

    # Reorder
    cos_sim_ordered = cos_sim[np.ix_(order, order)]
    names_ordered = [names[i] for i in order]
    cats_ordered = [categories.get(names[i], "Other") for i in order]

    # Color-coded category labels
    cat_colors = {
        "Big Five": "#e41a1c", "Politics": "#377eb8", "Ethics": "#4daf4a",
        "Religion": "#984ea3", "Interests": "#ff7f00", "AI Desires": "#a65628",
        "AI Willingness": "#f781bf", "AI Safety": "#999999", "AI Beliefs": "#66c2a5",
        "Dark Triad": "#e7298a", "Risk/Time Pref": "#7570b3", "Identity": "#d95f02",
        "Alignment": "#1b9e77", "Other Traits": "#666666", "Other": "#b3b3b3",
    }

    sns.heatmap(cos_sim_ordered, cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
                xticklabels=False, yticklabels=names_ordered, ax=ax)

    # Color y-axis labels by category
    for i, label in enumerate(ax.get_yticklabels()):
        cat = cats_ordered[i]
        label.set_color(cat_colors.get(cat, "black"))
        label.set_fontsize(5)

    ax.set_title(f"Cosine Similarity Matrix (Layer {layer_idx}, Hierarchically Clustered)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"cosine_similarity_layer{layer_idx}.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_pca_scatter(X_pca, names, categories, layer_idx):
    """2D scatter plot of personas in PC1-PC2 space, colored by category."""
    cat_colors = {
        "Big Five": "#e41a1c", "Politics": "#377eb8", "Ethics": "#4daf4a",
        "Religion": "#984ea3", "Interests": "#ff7f00", "AI Desires": "#a65628",
        "AI Willingness": "#f781bf", "AI Safety": "#999999", "AI Beliefs": "#66c2a5",
        "Dark Triad": "#e7298a", "Risk/Time Pref": "#7570b3", "Identity": "#d95f02",
        "Alignment": "#1b9e77", "Other Traits": "#666666", "Other": "#b3b3b3",
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # PC1 vs PC2
    for cat in sorted(set(categories.values())):
        idx = [i for i, n in enumerate(names) if categories.get(n) == cat]
        if idx:
            axes[0].scatter(X_pca[idx, 0], X_pca[idx, 1], c=cat_colors.get(cat, "gray"),
                          label=cat, s=40, alpha=0.8, edgecolors='white', linewidth=0.5)

    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title(f"Persona Space: PC1 vs PC2 (Layer {layer_idx})")
    axes[0].legend(fontsize=7, loc='upper right', ncol=2)
    axes[0].grid(True, alpha=0.2)

    # PC1 vs PC3
    for cat in sorted(set(categories.values())):
        idx = [i for i, n in enumerate(names) if categories.get(n) == cat]
        if idx:
            axes[1].scatter(X_pca[idx, 0], X_pca[idx, 2], c=cat_colors.get(cat, "gray"),
                          label=cat, s=40, alpha=0.8, edgecolors='white', linewidth=0.5)

    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC3")
    axes[1].set_title(f"Persona Space: PC1 vs PC3 (Layer {layer_idx})")
    axes[1].legend(fontsize=7, loc='upper right', ncol=2)
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"pca_scatter_layer{layer_idx}.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_top_pc_loadings(X_pca, names, categories, layer_idx, n_pcs=5):
    """Bar plot showing which personas load most strongly on each PC."""
    fig, axes = plt.subplots(n_pcs, 1, figsize=(14, 4*n_pcs))

    cat_colors = {
        "Big Five": "#e41a1c", "Politics": "#377eb8", "Ethics": "#4daf4a",
        "Religion": "#984ea3", "Interests": "#ff7f00", "AI Desires": "#a65628",
        "AI Willingness": "#f781bf", "AI Safety": "#999999", "AI Beliefs": "#66c2a5",
        "Dark Triad": "#e7298a", "Risk/Time Pref": "#7570b3", "Identity": "#d95f02",
        "Alignment": "#1b9e77", "Other Traits": "#666666", "Other": "#b3b3b3",
    }

    for pc_idx in range(n_pcs):
        ax = axes[pc_idx]
        loadings = X_pca[:, pc_idx]

        # Sort by absolute loading
        sorted_idx = np.argsort(np.abs(loadings))[::-1]
        top_k = 15  # Show top 15

        top_idx = sorted_idx[:top_k]
        top_names = [names[i][:40] for i in top_idx]
        top_loads = loadings[top_idx]
        top_cats = [categories.get(names[i], "Other") for i in top_idx]
        colors = [cat_colors.get(c, "gray") for c in top_cats]

        bars = ax.barh(range(top_k), top_loads, color=colors)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(top_names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("PC Loading")
        ax.set_title(f"PC{pc_idx+1} Top Loadings (Layer {layer_idx})")
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.grid(True, alpha=0.2, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"pc_loadings_layer{layer_idx}.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_permutation_test(real_evr, perm_evrs, p_values, layer_idx):
    """Plot permutation test results."""
    fig, ax = plt.subplots(figsize=(10, 5))

    n_pcs = min(20, len(real_evr))
    x = range(1, n_pcs + 1)

    # Permutation distribution
    perm_mean = perm_evrs[:, :n_pcs].mean(axis=0)
    perm_95 = np.percentile(perm_evrs[:, :n_pcs], 95, axis=0)

    ax.fill_between(x, 0, perm_95, alpha=0.3, color='gray', label='95th pctl (permuted)')
    ax.plot(x, perm_mean, '--', color='gray', label='Mean (permuted)')
    ax.plot(x, real_evr[:n_pcs], 'o-', color='red', label='Observed', markersize=5)

    # Mark significant PCs
    for i in range(n_pcs):
        if p_values[i] < 0.05:
            ax.annotate('*', (i+1, real_evr[i]), fontsize=14, ha='center', va='bottom', color='red')

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title(f"Permutation Test: Real vs Random PCA Structure (Layer {layer_idx})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_pcs + 1)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"permutation_test_layer{layer_idx}.png"), dpi=150, bbox_inches='tight')
    plt.close()

    return p_values


def plot_layer_comparison(all_results):
    """Compare PCA structure across layers."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    layers = [r["layer"] for r in all_results]

    # Top-k variance explained
    for k, label in [(1, "PC1"), (5, "Top 5"), (10, "Top 10"), (20, "Top 20")]:
        vals = []
        for r in all_results:
            if k == 1:
                vals.append(r["top1_variance"])
            elif k == 5:
                vals.append(r["top5_variance"] or 0)
            elif k == 10:
                vals.append(r["top10_variance"] or 0)
            elif k == 20:
                vals.append(r["top20_variance"] or 0)
        axes[0].plot(layers, vals, 'o-', label=label)

    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Cumulative Explained Variance")
    axes[0].set_title("Variance Explained by Layer")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Components for 80%/90%
    n80 = [r["n_components_80pct"] for r in all_results]
    n90 = [r["n_components_90pct"] for r in all_results]
    axes[1].plot(layers, n80, 'o-', label="80% variance")
    axes[1].plot(layers, n90, 's-', label="90% variance")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Number of Components")
    axes[1].set_title("Components Needed by Layer")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Vector norms by layer
    for r in all_results:
        axes[2].boxplot(r["norms"], positions=[r["layer"]], widths=1.5)
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Persona Vector L2 Norm")
    axes[2].set_title("Persona Vector Magnitude by Layer")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "layer_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_category_clustering(X_pca, names, categories, layer_idx):
    """Analyze how well semantic categories cluster in PC space."""
    unique_cats = sorted(set(categories.values()))
    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}

    labels = [cat_to_idx[categories.get(n, "Other")] for n in names]

    # Silhouette score for different numbers of clusters
    n_components_range = [2, 3, 5, 10, 15, 20]
    silhouette_scores = {}

    for n_comp in n_components_range:
        X_sub = X_pca[:, :n_comp]
        if len(set(labels)) > 1:
            score = silhouette_score(X_sub, labels)
            silhouette_scores[n_comp] = score

    return silhouette_scores


def plot_dendrogram(vectors, names, categories, layer_idx):
    """Hierarchical clustering dendrogram."""
    X = vectors.copy()
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_normed = X / (norms + 1e-8)

    linkage_matrix = linkage(pdist(X_normed, metric='cosine'), method='ward')

    cat_colors = {
        "Big Five": "#e41a1c", "Politics": "#377eb8", "Ethics": "#4daf4a",
        "Religion": "#984ea3", "Interests": "#ff7f00", "AI Desires": "#a65628",
        "AI Willingness": "#f781bf", "AI Safety": "#999999", "AI Beliefs": "#66c2a5",
        "Dark Triad": "#e7298a", "Risk/Time Pref": "#7570b3", "Identity": "#d95f02",
        "Alignment": "#1b9e77", "Other Traits": "#666666", "Other": "#b3b3b3",
    }

    fig, ax = plt.subplots(figsize=(20, 8))
    dn = dendrogram(linkage_matrix, labels=[n[:30] for n in names], leaf_rotation=90,
                    leaf_font_size=5, ax=ax)

    # Color labels
    xlabels = ax.get_xticklabels()
    for lbl in xlabels:
        name_short = lbl.get_text()
        # Find full name
        full_name = next((n for n in names if n[:30] == name_short), None)
        if full_name:
            cat = categories.get(full_name, "Other")
            lbl.set_color(cat_colors.get(cat, "black"))

    ax.set_title(f"Hierarchical Clustering of Persona Vectors (Layer {layer_idx})")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"dendrogram_layer{layer_idx}.png"), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("Loading data...")
    names, vectors, config = load_data()
    categories = categorize_personas(names)

    print(f"\nCategory distribution:")
    from collections import Counter
    cat_counts = Counter(categories.get(n, "Other") for n in names)
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    all_results = []

    for layer_idx in sorted(vectors.keys()):
        print(f"\n{'='*60}")
        print(f"LAYER {layer_idx} ANALYSIS")
        print(f"{'='*60}")

        V = vectors[layer_idx]
        results = run_pca_analysis(V, names, layer_idx)
        all_results.append(results)

        print(f"  Shape: {V.shape}")
        print(f"  PC1 variance: {results['top1_variance']:.4f} ({results['top1_variance']*100:.1f}%)")
        print(f"  Top 5 cumulative: {results['top5_variance']:.4f} ({results['top5_variance']*100:.1f}%)")
        print(f"  Top 10 cumulative: {results['top10_variance']:.4f} ({results['top10_variance']*100:.1f}%)")
        print(f"  Top 20 cumulative: {results['top20_variance']:.4f} ({results['top20_variance']*100:.1f}%)")
        print(f"  Components for 80%: {results['n_components_80pct']}")
        print(f"  Components for 90%: {results['n_components_90pct']}")

        # Permutation test
        print(f"\n  Running permutation test (100 permutations)...")
        real_evr, perm_evrs, p_values = permutation_test(V, n_permutations=100)
        n_significant = sum(1 for p in p_values if p < 0.05)
        print(f"  Significant PCs (p<0.05): {n_significant}")
        print(f"  PC1 p-value: {p_values[0]:.4f}")
        results["n_significant_pcs"] = n_significant
        results["permutation_p_values"] = p_values

        # Plots
        plot_pca_scatter(results["X_pca"], names, categories, layer_idx)
        plot_top_pc_loadings(results["X_pca"], names, categories, layer_idx, n_pcs=5)
        plot_cosine_similarity(results["cos_sim"], names, categories, layer_idx)
        plot_permutation_test(real_evr, perm_evrs, p_values, layer_idx)
        plot_dendrogram(V, names, categories, layer_idx)

        # Category clustering in PC space
        sil_scores = plot_category_clustering(results["X_pca"], names, categories, layer_idx)
        results["silhouette_scores"] = sil_scores
        print(f"  Silhouette scores (semantic categories):")
        for n_comp, score in sil_scores.items():
            print(f"    {n_comp} PCs: {score:.4f}")

    # Cross-layer plots
    plot_explained_variance(all_results, names, categories)
    plot_layer_comparison(all_results)

    # Save summary
    summary = {
        "n_personas": len(names),
        "categories": dict(cat_counts),
        "layers": {}
    }
    for r in all_results:
        layer = r["layer"]
        summary["layers"][layer] = {
            "top1_var": r["top1_variance"],
            "top5_var": r["top5_variance"],
            "top10_var": r["top10_variance"],
            "top20_var": r["top20_variance"],
            "n_80pct": r["n_components_80pct"],
            "n_90pct": r["n_components_90pct"],
            "n_significant_pcs": r["n_significant_pcs"],
            "pc1_pvalue": r["permutation_p_values"][0],
            "silhouette_scores": r["silhouette_scores"],
        }

    with open(os.path.join(RESULTS_DIR, "pca_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save top PC loadings for interpretation
    best_layer = max(all_results, key=lambda r: r["top1_variance"])
    layer_idx = best_layer["layer"]
    print(f"\n\n{'='*60}")
    print(f"BEST LAYER FOR PCA: {layer_idx}")
    print(f"{'='*60}")

    print(f"\nTop PC Interpretations (Layer {layer_idx}):")
    for pc_idx in range(min(10, best_layer["X_pca"].shape[1])):
        loadings = best_layer["X_pca"][:, pc_idx]
        sorted_idx = np.argsort(loadings)
        print(f"\n  PC{pc_idx+1} ({best_layer['explained_variance_ratio'][pc_idx]*100:.1f}% var):")
        print(f"    High (+):", ", ".join(names[i][:35] for i in sorted_idx[-5:][::-1]))
        print(f"    Low  (-):", ", ".join(names[i][:35] for i in sorted_idx[:5]))

    print(f"\nAll plots saved to {PLOTS_DIR}")
    print(f"Summary saved to {os.path.join(RESULTS_DIR, 'pca_summary.json')}")


if __name__ == "__main__":
    main()
