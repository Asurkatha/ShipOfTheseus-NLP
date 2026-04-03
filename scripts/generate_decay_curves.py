"""
Generate the composite feature decay curves figure for the paper.
This is the RQ1 signature visualization showing multiple features
decaying at different rates across T0 -> T1 -> T2 -> T3.

Requires: data/processed/multitier_results.pkl (from run_multitier_analysis.py)
"""
import sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.metrics import bootstrap_ci

BASELINES_DIR = ROOT / "figures" / "baselines"
STYLOMETRY_DIR = ROOT / "figures" / "stylometry"
NER_DIR = ROOT / "figures" / "ner"
for d in [BASELINES_DIR, STYLOMETRY_DIR, NER_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CACHE = ROOT / "data" / "processed" / "multitier_results.pkl"
BASELINES = ROOT / "experiments" / "baseline_results" / "similarity_baselines.csv"


def load_data():
    with open(CACHE, "rb") as f:
        results = pickle.load(f)
    baselines = pd.read_csv(BASELINES)
    return results, baselines


def compute_aggregate_metrics(results):
    """Compute mean metrics with 95% bootstrap CIs across all paraphrasers at each tier."""
    tiers = ["T1", "T2", "T3"]
    metrics = {
        "POS Cosine": {},
        "NER Jaccard": {},
        "NER Recall": {},
        "Dep Depth Ratio": {},
    }

    for tier in tiers:
        pos_vals, ner_j_vals, ner_r_vals, dep_vals = [], [], [], []

        for para_key, df in results.items():
            cos_col = f"cos_{tier}"
            if cos_col in df.columns:
                pos_vals.extend(df[cos_col].dropna().tolist())

            ner_j_col = f"ner_jaccard_{tier}"
            if ner_j_col in df.columns:
                ner_j_vals.extend(df[ner_j_col].dropna().tolist())

            ner_r_col = f"ner_recall_{tier}"
            if ner_r_col in df.columns:
                ner_r_vals.extend(df[ner_r_col].dropna().tolist())

            dep_t0 = "dep_depth_T0"
            dep_ti = f"dep_depth_{tier}"
            if dep_t0 in df.columns and dep_ti in df.columns:
                ratio = df[dep_ti] / df[dep_t0].replace(0, np.nan)
                dep_vals.extend(ratio.dropna().tolist())

        # Bootstrap 95% CIs: (mean, ci_lower, ci_upper)
        metrics["POS Cosine"][tier] = bootstrap_ci(pos_vals)
        metrics["NER Jaccard"][tier] = bootstrap_ci(ner_j_vals)
        metrics["NER Recall"][tier] = bootstrap_ci(ner_r_vals)
        if dep_vals:
            metrics["Dep Depth Ratio"][tier] = bootstrap_ci(dep_vals)

    return metrics


def plot_composite_decay(results, baselines):
    """
    The signature RQ1 figure: multiple features on one plot, T0->T3.
    Shows that different features decay at different rates.
    """
    tiers = ["T0", "T1", "T2", "T3"]
    x = range(len(tiers))

    # Aggregate per-paraphraser metrics across all models
    agg = compute_aggregate_metrics(results)

    # Get BLEU and BERTScore from baselines with bootstrap CIs
    bleu_by_tier = {"T0": (1.0, 1.0, 1.0)}
    bert_by_tier = {"T0": (1.0, 1.0, 1.0)}
    for tier in ["T1", "T2", "T3"]:
        b = baselines[baselines["Tier"] == tier]
        bleu_by_tier[tier] = bootstrap_ci(b["BLEU"].values)
        bert_by_tier[tier] = bootstrap_ci(b["BERTScore"].values)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot each feature
    features = [
        ("BERTScore (Semantic)", bert_by_tier, "#2ecc71", "o", "-"),
        ("POS Cosine (Grammar)", {**{"T0": (1.0, 1.0, 1.0)}, **agg["POS Cosine"]}, "#3498db", "s", "-"),
        ("NER Recall (Entities)", {**{"T0": (1.0, 1.0, 1.0)}, **agg["NER Recall"]}, "#e74c3c", "D", "-"),
        ("BLEU (Lexical)", bleu_by_tier, "#f39c12", "^", "--"),
        ("ROUGE-L (Lexical)", None, "#e67e22", "v", "--"),  # Computed below
    ]

    # ROUGE-L from baselines with bootstrap CIs
    rouge_by_tier = {"T0": (1.0, 1.0, 1.0)}
    for tier in ["T1", "T2", "T3"]:
        b = baselines[baselines["Tier"] == tier]
        rouge_by_tier[tier] = bootstrap_ci(b["ROUGE-L"].values)
    features[4] = ("ROUGE-L (Lexical)", rouge_by_tier, "#e67e22", "v", "--")

    # Add dep depth if available
    if agg["Dep Depth Ratio"]:
        features.append(
            ("Dep Depth Ratio", {**{"T0": (1.0, 1.0, 1.0)}, **agg["Dep Depth Ratio"]}, "#9b59b6", "P", "-.")
        )

    # Plot with asymmetric 95% CI error bars
    for label, data, color, marker, ls in features:
        if data is None:
            continue
        means = [data.get(t, (np.nan, np.nan, np.nan))[0] for t in tiers]
        ci_lo = [data.get(t, (np.nan, np.nan, np.nan))[1] for t in tiers]
        ci_hi = [data.get(t, (np.nan, np.nan, np.nan))[2] for t in tiers]
        yerr_lower = [means[i] - ci_lo[i] for i in range(len(tiers))]
        yerr_upper = [ci_hi[i] - means[i] for i in range(len(tiers))]
        ax.errorbar(x, means, yerr=[yerr_lower, yerr_upper], label=label, color=color,
                    marker=marker, markersize=8, linewidth=2.5, linestyle=ls,
                    capsize=4, capthick=1.2, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(tiers, fontsize=12)
    ax.set_xlabel("Paraphrase Iteration", fontsize=13)
    ax.set_ylabel("Similarity to T0 (normalized)", fontsize=13)
    ax.set_title("Layered Erosion: Which Planks Break First?",
                 fontsize=15, fontweight='bold')
    ax.set_ylim(0, 1.08)
    ax.legend(loc='lower left', fontsize=10, frameon=True, fancybox=True)
    ax.grid(True, alpha=0.2)

    # Annotate the gap
    ax.annotate("Semantic hull\n(preserved)",
                xy=(3, 0.90), fontsize=9, color="#2ecc71",
                fontstyle='italic', ha='center')
    ax.annotate("Entity cargo\n(jettisoned)",
                xy=(3, 0.35), fontsize=9, color="#e74c3c",
                fontstyle='italic', ha='center')
    ax.annotate("Lexical planks\n(replaced)",
                xy=(3, 0.18), fontsize=9, color="#f39c12",
                fontstyle='italic', ha='center')

    plt.tight_layout()
    out = BASELINES_DIR / "feature_decay_layered.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


def plot_per_paraphraser_pos_decay(results):
    """POS cosine decay curves per paraphraser."""
    fig, ax = plt.subplots(figsize=(10, 6))
    tiers = ["T0", "T1", "T2", "T3"]
    x = range(len(tiers))

    colors = {
        "chatgpt": "#1abc9c", "palm": "#3498db", "dipper": "#e67e22",
        "dipper(low)": "#d35400", "dipper(high)": "#e74c3c",
        "pegasus(full)": "#8e44ad", "pegasus(slight)": "#9b59b6",
    }
    labels = {
        "chatgpt": "ChatGPT", "palm": "PaLM2", "dipper": "Dipper",
        "dipper(low)": "Dipper (Low)", "dipper(high)": "Dipper (High)",
        "pegasus(full)": "Pegasus (Full)", "pegasus(slight)": "Pegasus (Slight)",
    }

    for para_key, df in results.items():
        means = [1.0]  # T0 = 1.0
        for tier in ["T1", "T2", "T3"]:
            col = f"cos_{tier}"
            if col in df.columns:
                means.append(df[col].dropna().mean())
            else:
                means.append(np.nan)

        ax.plot(x, means, 'o-', label=labels.get(para_key, para_key),
                color=colors.get(para_key, "gray"), linewidth=2, markersize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(tiers, fontsize=12)
    ax.set_xlabel("Paraphrase Iteration", fontsize=13)
    ax.set_ylabel("POS Cosine Similarity to T0", fontsize=13)
    ax.set_title("POS Cosine Decay Across Iterations (All 7 Paraphrasers)",
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0.8, 1.02)
    ax.legend(loc='lower left', fontsize=9, frameon=True, ncol=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out = STYLOMETRY_DIR / "pos_cosine_decay_curves.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


def plot_per_paraphraser_ner_decay(results):
    """NER Recall decay curves per paraphraser."""
    fig, ax = plt.subplots(figsize=(10, 6))
    tiers = ["T0", "T1", "T2", "T3"]
    x = range(len(tiers))

    colors = {
        "chatgpt": "#1abc9c", "palm": "#3498db", "dipper": "#e67e22",
        "dipper(low)": "#d35400", "dipper(high)": "#e74c3c",
        "pegasus(full)": "#8e44ad", "pegasus(slight)": "#9b59b6",
    }
    labels = {
        "chatgpt": "ChatGPT", "palm": "PaLM2", "dipper": "Dipper",
        "dipper(low)": "Dipper (Low)", "dipper(high)": "Dipper (High)",
        "pegasus(full)": "Pegasus (Full)", "pegasus(slight)": "Pegasus (Slight)",
    }

    for para_key, df in results.items():
        means = [1.0]  # T0 = 1.0
        for tier in ["T1", "T2", "T3"]:
            col = f"ner_recall_{tier}"
            if col in df.columns:
                means.append(df[col].dropna().mean())
            else:
                means.append(np.nan)

        ax.plot(x, means, 'o-', label=labels.get(para_key, para_key),
                color=colors.get(para_key, "gray"), linewidth=2, markersize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(tiers, fontsize=12)
    ax.set_xlabel("Paraphrase Iteration", fontsize=13)
    ax.set_ylabel("NER Recall (Entity Retention)", fontsize=13)
    ax.set_title("NER Entity Retention Across Iterations (All 7 Paraphrasers)",
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', fontsize=9, frameon=True, ncol=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out = NER_DIR / "ner_recall_decay_curves.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


def main():
    print("Loading data...")
    results, baselines = load_data()
    print(f"  Paraphrasers: {list(results.keys())}")

    print("\nGenerating composite feature decay figure...")
    plot_composite_decay(results, baselines)

    print("Generating per-paraphraser POS decay curves...")
    plot_per_paraphraser_pos_decay(results)

    print("Generating per-paraphraser NER decay curves...")
    plot_per_paraphraser_ner_decay(results)

    print("\nAll decay curve figures generated!")


if __name__ == "__main__":
    main()
