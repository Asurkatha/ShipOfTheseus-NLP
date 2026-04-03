"""
Generate t-SNE Identity Trajectory visualization.

Uses combined feature vectors (POS + NER + dep depth + text stats) at each tier
to show how documents move from a diverse authorial region toward a compressed
machine-paraphrased region.

Also computes quantitative backing: silhouette scores and k-NN purity.
"""
import sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.attribution.feature_builder import build_feature_vector

FIGURES_DIR = ROOT / "figures" / "fingerprints"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CACHE = ROOT / "data" / "processed" / "multitier_results.pkl"


def main():
    print("Loading data...")
    with open(CACHE, "rb") as f:
        results = pickle.load(f)

    # Use Dipper (balanced sources, 3010 articles, all tiers present)
    para_key = "dipper"
    df = results[para_key].copy()
    print(f"Using {para_key}: {len(df)} articles")

    # Build feature vectors at each tier
    tiers = ["T0", "T1", "T2", "T3"]
    all_X = []
    all_tiers = []
    all_sources = []
    feat_names = None

    for tier in tiers:
        text_col = f"text_{tier}"
        if text_col not in df.columns:
            continue
        mask = df[text_col].notna()
        df_tier = df[mask].reset_index(drop=True)

        X, feat_names = build_feature_vector(df_tier, tier)

        # Handle NaN
        nan_rows = np.isnan(X).any(axis=1)
        X = X[~nan_rows]
        sources = df_tier["source"].values[~nan_rows]

        all_X.append(X)
        all_tiers.extend([tier] * len(X))
        all_sources.extend(sources)
        print(f"  {tier}: {len(X)} samples")

    X_all = np.vstack(all_X)
    tiers_arr = np.array(all_tiers)
    sources_arr = np.array(all_sources)

    # Binary source labels
    binary_sources = np.array(["Human" if s == "Human" else "LLM" for s in sources_arr])

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # ===== t-SNE =====
    print("\nRunning t-SNE (perplexity=30)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                max_iter=1000, learning_rate="auto", init="pca")
    X_2d = tsne.fit_transform(X_scaled)
    print(f"  t-SNE done. Shape: {X_2d.shape}")

    # ===== PLOT 1: t-SNE colored by tier =====
    print("Generating t-SNE plots...")

    tier_colors = {"T0": "#2ecc71", "T1": "#f39c12", "T2": "#e74c3c", "T3": "#9b59b6"}
    tier_order = ["T3", "T2", "T1", "T0"]  # Plot T0 on top

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: colored by tier
    ax = axes[0]
    for tier in tier_order:
        mask = tiers_arr == tier
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=tier_colors[tier],
                   label=tier, alpha=0.3, s=8, edgecolors="none")
    ax.set_title("Identity Trajectory: By Paraphrase Tier", fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=10, markerscale=3, frameon=True)

    # Right: colored by source (Human vs LLM), T0 vs T3 only
    ax = axes[1]
    source_colors = {"Human": "#2ecc71", "LLM": "#e74c3c"}

    for tier, marker, alpha in [("T3", "x", 0.2), ("T0", "o", 0.5)]:
        for src in ["LLM", "Human"]:
            mask = (tiers_arr == tier) & (binary_sources == src)
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=source_colors[src], marker=marker,
                       label=f"{src} ({tier})", alpha=alpha, s=12, edgecolors="none")

    ax.set_title("Identity Trajectory: Human vs LLM (T0 vs T3)", fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=9, markerscale=2, frameon=True, loc="upper right")

    fig.suptitle("Identity Trajectory: How Documents Move Through Feature Space (Dipper)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = FIGURES_DIR / "identity_trajectory_tsne.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")

    # ===== QUANTITATIVE BACKING =====
    print("\nQuantitative validation:")

    # Silhouette score (Human vs LLM) at each tier
    print("\n  Silhouette Score (Human vs LLM clustering):")
    for tier in tiers:
        mask = tiers_arr == tier
        if mask.sum() < 10:
            continue
        X_tier = X_scaled[mask]
        y_tier = binary_sources[mask]
        if len(set(y_tier)) < 2:
            continue
        sil = silhouette_score(X_tier, y_tier, sample_size=min(2000, len(X_tier)),
                               random_state=42)
        print(f"    {tier}: {sil:.4f}")

    # k-NN purity (k=10) — using 5-fold CV to avoid train-on-train inflation
    print("\n  k-NN Purity (k=10, Human vs LLM, 5-fold CV):")
    k = 10
    for tier in tiers:
        mask = tiers_arr == tier
        if mask.sum() < k + 1:
            continue
        X_tier = X_scaled[mask]
        y_tier = binary_sources[mask]
        knn = KNeighborsClassifier(n_neighbors=k)
        purity = cross_val_score(knn, X_tier, y_tier, cv=5, scoring="accuracy").mean()
        print(f"    {tier}: {purity:.4f}")

    # Centroid distance (Human vs LLM)
    print("\n  Centroid Distance (Human vs LLM, Euclidean):")
    for tier in tiers:
        mask = tiers_arr == tier
        X_tier = X_scaled[mask]
        y_tier = binary_sources[mask]
        human_centroid = X_tier[y_tier == "Human"].mean(axis=0)
        llm_centroid = X_tier[y_tier == "LLM"].mean(axis=0)
        dist = np.linalg.norm(human_centroid - llm_centroid)
        print(f"    {tier}: {dist:.4f}")

    # ===== PLOT 2: Quantitative metrics bar chart =====
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Compute metrics for each tier
    sil_scores, knn_purities, centroid_dists = [], [], []
    for tier in tiers:
        mask = tiers_arr == tier
        X_tier = X_scaled[mask]
        y_tier = binary_sources[mask]

        sil = silhouette_score(X_tier, y_tier, sample_size=min(2000, len(X_tier)),
                               random_state=42)
        sil_scores.append(sil)

        knn = KNeighborsClassifier(n_neighbors=10)
        knn_purities.append(cross_val_score(knn, X_tier, y_tier, cv=5, scoring="accuracy").mean())

        h_c = X_tier[y_tier == "Human"].mean(axis=0)
        l_c = X_tier[y_tier == "LLM"].mean(axis=0)
        centroid_dists.append(np.linalg.norm(h_c - l_c))

    tier_cols = [tier_colors[t] for t in tiers]

    axes[0].bar(tiers, sil_scores, color=tier_cols, edgecolor="black", linewidth=0.8)
    axes[0].set_ylabel("Silhouette Score")
    axes[0].set_title("Cluster Separation")

    axes[1].bar(tiers, knn_purities, color=tier_cols, edgecolor="black", linewidth=0.8)
    axes[1].set_ylabel("k-NN Purity (k=10)")
    axes[1].set_title("Identity Preservation")

    axes[2].bar(tiers, centroid_dists, color=tier_cols, edgecolor="black", linewidth=0.8)
    axes[2].set_ylabel("Centroid Distance")
    axes[2].set_title("Human-LLM Separation")

    fig.suptitle("Quantitative Validation: Does Identity Erode Across Tiers?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path2 = FIGURES_DIR / "identity_quantitative_metrics.png"
    fig.savefig(out_path2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved {out_path2}")

    print("\nDone!")


if __name__ == "__main__":
    main()
