"""
Domain-specific analysis across all 7 datasets.
Generates figures showing how decay patterns vary by domain (news, fiction, reviews, etc.)
"""
import sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.attribution.feature_builder import build_feature_vector
from src.utils.config import RANDOM_SEED

FIGURES_DIR = ROOT / "figures" / "domain"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATASET_LABELS = {
    "cmv": "CMV\n(Arguments)", "eli5": "ELI5\n(Explanations)",
    "sci_gen": "SCI_GEN\n(Scientific)", "tldr": "TLDR\n(Summaries)",
    "wp": "WP\n(Fiction)", "xsum": "XSum\n(News)", "yelp": "Yelp\n(Reviews)",
}
DATASET_SHORT = {
    "cmv": "CMV", "eli5": "ELI5", "sci_gen": "SCI_GEN",
    "tldr": "TLDR", "wp": "WP", "xsum": "XSum", "yelp": "Yelp",
}
PARA_LABELS = {
    "chatgpt": "ChatGPT", "dipper": "Dipper", "dipper(low)": "Dipper (Low)",
    "dipper(high)": "Dipper (High)", "pegasus(full)": "Pegasus (Full)",
    "pegasus(slight)": "Pegasus (Slight)", "palm": "PaLM2",
}
DATASETS = ["cmv", "eli5", "sci_gen", "tldr", "wp", "xsum", "yelp"]


def load_data():
    with open(ROOT / "data" / "processed" / "multitier_results.pkl", "rb") as f:
        return pickle.load(f)


def plot_domain_heatmap(results):
    """
    Fig 1: Heatmap of POS cosine and NER Recall by dataset x paraphraser.
    Shows which domain-paraphraser combos are most/least affected.
    """
    paras = ["pegasus(slight)", "dipper(low)", "chatgpt", "dipper", "pegasus(full)", "dipper(high)"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, (metric_col, title, cmap, vrange) in enumerate([
        ("cos_T1", "POS Cosine (T0 vs T1)", "RdYlGn", (0.85, 1.0)),
        ("ner_recall_T1", "NER Recall (T0 vs T1)", "RdYlGn", (0.0, 0.95)),
    ]):
        data = []
        for ds in DATASETS:
            row = {}
            for para in paras:
                df = results[para]
                mask = df["dataset"] == ds
                vals = df.loc[mask, metric_col].dropna()
                row[PARA_LABELS[para]] = vals.mean() if len(vals) > 0 else np.nan
            data.append(row)

        heatmap_df = pd.DataFrame(data, index=[DATASET_SHORT[ds] for ds in DATASETS])

        ax = axes[ax_idx]
        sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap=cmap,
                    vmin=vrange[0], vmax=vrange[1],
                    linewidths=0.5, linecolor="white", ax=ax)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("Dataset")
        ax.set_xlabel("Paraphraser")

    fig.suptitle("Domain x Paraphraser: Style and Content Preservation",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / "domain_paraphraser_heatmap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_domain_decay_curves(results):
    """
    Fig 2: Per-dataset decay curves for POS cosine and NER Recall across T0-T3.
    Uses Dipper (default) for balanced representation.
    """
    df = results["dipper"]
    tiers = ["T0", "T1", "T2", "T3"]
    x = range(len(tiers))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.Set2(np.linspace(0, 1, len(DATASETS)))

    for ax_idx, (metric_prefix, ylabel, title, ylim) in enumerate([
        ("cos", "POS Cosine Similarity", "Grammar Decay by Domain", (0.90, 1.01)),
        ("ner_recall", "NER Recall", "Entity Decay by Domain", (0.0, 1.05)),
    ]):
        ax = axes[ax_idx]
        for i, ds in enumerate(DATASETS):
            mask = df["dataset"] == ds
            vals = [1.0]
            for tier in ["T1", "T2", "T3"]:
                col = f"{metric_prefix}_{tier}"
                if col in df.columns:
                    v = df.loc[mask, col].dropna()
                    vals.append(v.mean() if len(v) > 0 else np.nan)
                else:
                    vals.append(np.nan)
            ax.plot(x, vals, "o-", label=DATASET_SHORT[ds], color=colors[i],
                    linewidth=2, markersize=6)

        ax.set_xticks(x)
        ax.set_xticklabels(tiers)
        ax.set_xlabel("Paraphrase Tier")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(ylim)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Feature Decay Across Domains (Dipper Default)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / "domain_decay_curves.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_domain_attribution(results):
    """
    Fig 3: Per-dataset attribution F1 decay — shows which domains
    lose authorial identity fastest.
    """
    df = results["dipper"]
    datasets = sorted(df["dataset"].unique())
    tiers = ["T0", "T1", "T2", "T3"]
    x = range(len(tiers))
    colors = plt.cm.Set2(np.linspace(0, 1, len(datasets)))

    fig, ax = plt.subplots(figsize=(9, 6))

    for i, ds in enumerate(datasets):
        mask = df["dataset"] == ds
        df_ds = df[mask].reset_index(drop=True)
        if len(df_ds) < 30:
            continue

        y = np.array(["Human" if s == "Human" else "LLM" for s in df_ds["source"]])
        if len(set(y)) < 2:
            continue

        X_T0, _ = build_feature_vector(df_ds, "T0")
        nan_mask = np.isnan(X_T0).any(axis=1)
        X_T0 = X_T0[~nan_mask]
        y_clean = y[~nan_mask]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_T0)
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED,
                                  class_weight="balanced")
        clf.fit(X_scaled, y_clean)

        f1_vals = [f1_score(y_clean, clf.predict(X_scaled), average="macro")]
        for tier in ["T1", "T2", "T3"]:
            text_col = f"text_{tier}"
            tier_mask = df_ds[text_col].notna()
            valid = tier_mask.values & ~nan_mask
            if valid.sum() < 10:
                f1_vals.append(np.nan)
                continue
            X_tier, _ = build_feature_vector(df_ds[valid], tier)
            f1_vals.append(f1_score(y[valid], clf.predict(scaler.transform(X_tier)),
                                     average="macro"))

        ax.plot(x, f1_vals, "o-", label=DATASET_SHORT[ds], color=colors[i],
                linewidth=2, markersize=7)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_xlabel("Paraphrase Tier", fontsize=12)
    ax.set_ylabel("Macro F1 (Human vs LLM)", fontsize=12)
    ax.set_title("Attribution Decay by Domain (Dipper, Logistic Regression)",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0.3, 0.9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out = FIGURES_DIR / "domain_attribution_decay.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_domain_entity_profile(results):
    """
    Fig 4: Entity density and zero-entity rate by domain.
    Explains why some domains are more resilient.
    """
    df = results["dipper"]

    rows = []
    for ds in DATASETS:
        mask = df["dataset"] == ds
        ner_counts = [len(s) if isinstance(s, (set, frozenset)) else 0
                      for s in df.loc[mask, "ner_T0"]]
        rows.append({
            "Dataset": DATASET_SHORT[ds],
            "Mean Entities": np.mean(ner_counts),
            "Zero-Entity %": 100 * sum(1 for c in ner_counts if c == 0) / len(ner_counts),
            "NER Recall T1": df.loc[mask, "ner_recall_T1"].dropna().mean(),
        })

    profile_df = pd.DataFrame(rows).sort_values("Mean Entities", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.barh(profile_df["Dataset"], profile_df["Mean Entities"],
            color="#3498db", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Mean Entities per Document")
    ax.set_title("Entity Density by Domain")
    ax.invert_yaxis()

    ax = axes[1]
    ax.barh(profile_df["Dataset"], profile_df["Zero-Entity %"],
            color="#e74c3c", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("% Documents with Zero Entities")
    ax.set_title("Entity-Free Documents")
    ax.invert_yaxis()

    ax = axes[2]
    ax.barh(profile_df["Dataset"], profile_df["NER Recall T1"],
            color="#2ecc71", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("NER Recall at T1")
    ax.set_title("Entity Retention After Paraphrasing")
    ax.set_xlim(0, 0.6)
    ax.invert_yaxis()

    fig.suptitle("Domain Entity Profiles: Why Some Domains Resist Better",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / "domain_entity_profile.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def print_summary(results):
    """Print the summary table for the paper."""
    df = results["dipper"]
    print("\n=== DOMAIN SUMMARY TABLE (Dipper) ===\n")
    print(f"{'Dataset':<10s} {'Entities':>10s} {'POS T1':>8s} {'POS T3':>8s} "
          f"{'NER T1':>8s} {'NER T3':>8s}")
    print("-" * 55)
    for ds in DATASETS:
        mask = df["dataset"] == ds
        ent_density = np.mean([len(s) if isinstance(s, (set, frozenset)) else 0
                               for s in df.loc[mask, "ner_T0"]])
        pos_t1 = df.loc[mask, "cos_T1"].dropna().mean()
        pos_t3 = df.loc[mask, "cos_T3"].dropna().mean()
        ner_t1 = df.loc[mask, "ner_recall_T1"].dropna().mean()
        ner_t3 = df.loc[mask, "ner_recall_T3"].dropna().mean()
        print(f"{DATASET_SHORT[ds]:<10s} {ent_density:>10.1f} {pos_t1:>8.4f} {pos_t3:>8.4f} "
              f"{ner_t1:>8.4f} {ner_t3:>8.4f}")


def main():
    print("Loading data...")
    results = load_data()

    print("\nGenerating domain analysis figures...")
    plot_domain_heatmap(results)
    plot_domain_decay_curves(results)
    plot_domain_attribution(results)
    plot_domain_entity_profile(results)
    print_summary(results)
    print("\nDone!")


if __name__ == "__main__":
    main()
