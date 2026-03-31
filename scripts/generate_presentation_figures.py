"""
Generate 4 new presentation-quality visualizations for Phase II slides.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STYLOMETRY_DIR = ROOT / "figures" / "stylometry"
NER_DIR = ROOT / "figures" / "ner"
FINGERPRINTS_DIR = ROOT / "figures" / "fingerprints"
for d in [STYLOMETRY_DIR, NER_DIR, FINGERPRINTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Load data
paired = pd.read_pickle(ROOT / "paired_all.pkl")
pos_vectors = dict(np.load(ROOT / "data" / "processed" / "pos_vectors.npz"))
baselines = pd.read_csv(ROOT / "experiments" / "baseline_results" / "similarity_baselines.csv")

from src.features.ner import entity_metrics
from src.utils.metrics import batch_cosine_similarity
from src.features.pos import UPOS_TAGS
import math, pickle

with open(ROOT / "data" / "processed" / "ner_sets.pkl", "rb") as f:
    ner_sets = pickle.load(f)

# Paraphraser definitions
PARAPHRASERS = {
    "chatgpt": ("ChatGPT", "text_chatgpt", "#1abc9c"),
    "palm": ("PaLM2", "text_palm", "#3498db"),
    "dipper": ("Dipper", "text_dipper", "#e67e22"),
    "dipper_high": ("Dipper (High)", "text_dipper_high", "#e74c3c"),
    "dipper_low": ("Dipper (Low)", "text_dipper_low", "#d35400"),
    "pegasus_slight": ("Pegasus (Slight)", "text_pegasus_slight", "#9b59b6"),
    "pegasus_full": ("Pegasus (Full)", "text_pegasus_full", "#8e44ad"),
}

# Precompute metrics if not already in paired
for label, (name, col, color) in PARAPHRASERS.items():
    cos_col = f"cos_{label}"
    if cos_col not in paired.columns:
        paired[cos_col] = batch_cosine_similarity(
            pos_vectors["text_T0"], pos_vectors[col]
        )
    ner_j_col = f"ner_jaccard_{label}"
    if ner_j_col not in paired.columns:
        jaccards, recalls, precisions = [], [], []
        for i in range(len(paired)):
            j, r, p = entity_metrics(ner_sets["text_T0"][i], ner_sets[col][i])
            jaccards.append(j)
            recalls.append(r)
            precisions.append(p)
        paired[f"ner_jaccard_{label}"] = jaccards
        paired[f"ner_recall_{label}"] = recalls
        paired[f"ner_precision_{label}"] = precisions


# ============================================================
# VIZ 1: Style vs Content Scatter Plot
# ============================================================
print("Generating Viz 1: Style vs Content Scatter...")

fig, ax = plt.subplots(figsize=(9, 7))

for label, (name, col, color) in PARAPHRASERS.items():
    pos_mean = paired[f"cos_{label}"].dropna().mean()
    pos_std = paired[f"cos_{label}"].dropna().std()
    ner_mean = paired[f"ner_jaccard_{label}"].dropna().mean()
    ner_std = paired[f"ner_jaccard_{label}"].dropna().std()

    ax.errorbar(pos_mean, ner_mean, xerr=pos_std, yerr=ner_std,
                fmt='o', markersize=12, color=color, markeredgecolor='black',
                markeredgewidth=1.2, capsize=4, capthick=1.2, elinewidth=1,
                ecolor=color, alpha=0.8, zorder=5)
    # Label offset
    offsets = {
        "chatgpt": (0.003, 0.025), "palm": (0.003, 0.025),
        "dipper": (0.003, -0.04), "dipper_high": (0.005, 0.02),
        "dipper_low": (-0.035, 0.025), "pegasus_slight": (-0.01, -0.05),
        "pegasus_full": (0.004, -0.04),
    }
    dx, dy = offsets.get(label, (0.003, 0.02))
    ax.annotate(name, (pos_mean + dx, ner_mean + dy),
                fontsize=9, fontweight='bold', color=color)

# Quadrant lines
ax.axvline(x=0.95, color='gray', linestyle='--', alpha=0.4, linewidth=1)
ax.axhline(y=0.50, color='gray', linestyle='--', alpha=0.4, linewidth=1)

# Quadrant labels
ax.text(0.98, 0.92, "Preserves\nBoth", ha='center', va='center',
        fontsize=9, color='#27ae60', fontstyle='italic', alpha=0.7)
ax.text(0.91, 0.92, "Style Only", ha='center', va='center',
        fontsize=9, color='#e67e22', fontstyle='italic', alpha=0.7)
ax.text(0.98, 0.15, "Content\nOnly", ha='center', va='center',
        fontsize=9, color='#3498db', fontstyle='italic', alpha=0.7)
ax.text(0.91, 0.15, "Destroys\nBoth", ha='center', va='center',
        fontsize=9, color='#e74c3c', fontstyle='italic', alpha=0.7)

ax.set_xlabel("POS Cosine Similarity (Style Preservation)", fontsize=12)
ax.set_ylabel("NER Jaccard Similarity (Content Preservation)", fontsize=12)
ax.set_title("Style vs. Content: The Paraphraser Tradeoff Space", fontsize=14, fontweight='bold')
ax.set_xlim(0.85, 1.01)
ax.set_ylim(-0.05, 1.0)
ax.grid(True, alpha=0.2)
plt.tight_layout()
fig.savefig(FINGERPRINTS_DIR / "style_vs_content_scatter.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved style_vs_content_scatter.png")


# ============================================================
# VIZ 2: Radar/Spider Chart
# ============================================================
print("Generating Viz 2: Radar Fingerprints...")

# Get BERTScore T1 values
bert_t1 = baselines[baselines["Tier"] == "T1"].set_index("Paraphraser_Key")["BERTScore"].to_dict()

# 4 representative paraphrasers
radar_models = {
    "pegasus_slight": ("Pegasus (Slight)", "#9b59b6"),
    "chatgpt": ("ChatGPT", "#1abc9c"),
    "dipper": ("Dipper (Default)", "#e67e22"),
    "dipper_high": ("Dipper (High)", "#e74c3c"),
}

# Map paraphraser keys to baseline keys
key_map = {
    "pegasus_slight": "pegasus(slight)", "chatgpt": "chatgpt",
    "dipper": "dipper", "dipper_high": "dipper(high)",
}

categories = ["POS\nCosine", "NER\nJaccard", "NER\nRecall", "NER\nPrecision", "BERTScore\nF1"]
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for label, (name, color) in radar_models.items():
    values = [
        paired[f"cos_{label}"].dropna().mean(),
        paired[f"ner_jaccard_{label}"].dropna().mean(),
        paired[f"ner_recall_{label}"].dropna().mean(),
        paired[f"ner_precision_{label}"].dropna().mean(),
        bert_t1.get(key_map[label], 0.9),
    ]
    values += values[:1]

    ax.plot(angles, values, 'o-', linewidth=2.5, label=name, color=color, markersize=6)
    ax.fill(angles, values, alpha=0.12, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color='gray')
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10,
          frameon=True, fancybox=True, shadow=True)
ax.set_title("Paraphraser Forensic Fingerprints", fontsize=14,
             fontweight='bold', pad=20)
plt.tight_layout()
fig.savefig(FINGERPRINTS_DIR / "radar_paraphraser_fingerprints.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved radar_paraphraser_fingerprints.png")


# ============================================================
# VIZ 3: POS Tag Delta Heatmap
# ============================================================
print("Generating Viz 3: POS Tag Delta Heatmap...")

t0_mean = pos_vectors["text_T0"].mean(axis=0)

delta_data = {}
para_order = ["pegasus_slight", "dipper_low", "palm", "chatgpt",
              "dipper", "pegasus_full", "dipper_high"]
para_display = ["Pegasus\n(Slight)", "Dipper\n(Low)", "PaLM2", "ChatGPT",
                "Dipper\n(Default)", "Pegasus\n(Full)", "Dipper\n(High)"]

for label in para_order:
    col = PARAPHRASERS[label][1]
    t1_mean = pos_vectors[col].mean(axis=0)
    delta_data[label] = t1_mean - t0_mean

delta_df = pd.DataFrame(delta_data, index=UPOS_TAGS)
delta_df.columns = para_display

# Sort rows by max absolute delta
delta_df["max_abs"] = delta_df.abs().max(axis=1)
delta_df = delta_df.sort_values("max_abs", ascending=False).drop(columns="max_abs")

fig, ax = plt.subplots(figsize=(10, 9))
vmax = max(abs(delta_df.values.min()), abs(delta_df.values.max()))
sns.heatmap(delta_df, annot=True, fmt=".4f", cmap="RdBu_r",
            center=0, vmin=-vmax, vmax=vmax,
            linewidths=0.5, linecolor='white',
            cbar_kws={"label": "Change in Tag Proportion (T1 - T0)"},
            ax=ax)
ax.set_title("POS Tag Proportion Delta by Paraphraser (T0 to T1)",
             fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel("UPOS Tag", fontsize=11)
ax.set_xlabel("Paraphraser", fontsize=11)
plt.tight_layout()
fig.savefig(STYLOMETRY_DIR / "pos_tag_delta_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved pos_tag_delta_heatmap.png")


# ============================================================
# VIZ 4: Entity Survival Funnel
# ============================================================
print("Generating Viz 4: Entity Survival Funnel...")

funnel_models = [
    ("pegasus_slight", "Pegasus (Slight)"),
    ("chatgpt", "ChatGPT"),
    ("dipper", "Dipper (Default)"),
    ("dipper_high", "Dipper (High)"),
]

fig, ax = plt.subplots(figsize=(10, 5))
y_positions = np.arange(len(funnel_models))
bar_height = 0.6

for i, (label, name) in enumerate(funnel_models):
    recall = paired[f"ner_recall_{label}"].dropna().mean()
    precision = paired[f"ner_precision_{label}"].dropna().mean()

    retained = recall
    dropped = 1 - recall
    introduced = 1 - precision  # fraction of T1 entities not in T0

    # Stacked bar: retained + dropped = 100% of T0 view
    ax.barh(i, retained, bar_height, color='#27ae60', edgecolor='white',
            linewidth=0.8, label='Retained' if i == 0 else None)
    ax.barh(i, dropped, bar_height, left=retained, color='#e74c3c',
            edgecolor='white', linewidth=0.8, label='Dropped' if i == 0 else None)

    # Small marker for novel entity introduction rate
    ax.plot(1.0 + introduced * 0.15, i, 's', color='#f39c12', markersize=10,
            markeredgecolor='black', markeredgewidth=0.8,
            label='Novel entity rate' if i == 0 else None)

    # Labels
    if retained > 0.08:
        ax.text(retained / 2, i, f"{retained:.0%}", ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
    if dropped > 0.08:
        ax.text(retained + dropped / 2, i, f"{dropped:.0%}", ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
    ax.text(1.0 + introduced * 0.15 + 0.03, i, f"{introduced:.0%}",
            ha='left', va='center', fontsize=9, color='#f39c12', fontweight='bold')

ax.set_yticks(y_positions)
ax.set_yticklabels([m[1] for m in funnel_models], fontsize=11, fontweight='bold')
ax.set_xlabel("Fraction of T0 Entities", fontsize=12)
ax.set_title("Entity Survival: What Happens to Named Entities?",
             fontsize=14, fontweight='bold')
ax.set_xlim(0, 1.25)
ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.3)
ax.text(1.08, len(funnel_models) - 0.3, "Novel\nEntity %", ha='center',
        fontsize=8, color='gray', fontstyle='italic')
ax.legend(loc='lower right', fontsize=10, frameon=True)
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(NER_DIR / "entity_survival_funnel.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved entity_survival_funnel.png")


print("\nAll 4 visualizations generated!")
