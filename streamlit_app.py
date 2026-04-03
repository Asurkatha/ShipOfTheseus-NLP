"""
Ship of Theseus: Decay Dashboard

Streamlit app for visualizing authorial identity decay across paraphrase iterations.
Four tabs:
  1. Decay Explorer — interactive feature decay curves with error bands
  2. Document Forensics — side-by-side T0 vs T1/T2/T3 with colored entity highlighting
  3. Attribution Lab — attribution model confidence across tiers + RQ3 paraphraser ID
  4. Research Gallery — pre-generated publication figures
"""
import sys
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.attribution.feature_builder import build_feature_vector, build_delta_features
from src.attribution.classifier import (
    run_attribution_experiment, run_paraphraser_identification, CLASSIFIERS,
)
from src.utils.config import ALL_DATASETS

# ===== PAGE CONFIG =====
st.set_page_config(page_title="Ship of Theseus: Decay Dashboard", layout="wide")

LABELS = {
    "chatgpt": "ChatGPT", "palm": "PaLM2", "dipper": "Dipper",
    "dipper(low)": "Dipper (Low)", "dipper(high)": "Dipper (High)",
    "pegasus(full)": "Pegasus (Full)", "pegasus(slight)": "Pegasus (Slight)",
}

CLF_PALETTE = {
    "Logistic Regression": "#1abc9c",
    "SVM (RBF)": "#e67e22",
    "Random Forest": "#3498db",
}

METRIC_EXPLANATIONS = {
    "POS Cosine": (
        "**POS Cosine Similarity** measures how well the grammatical structure "
        "(distribution of parts-of-speech tags) is preserved after paraphrasing. "
        "A value of 1.0 means identical POS distributions to the original. "
        "High-intensity paraphrasers like Dipper (High) aggressively restructure sentences, "
        "while Pegasus (Slight) preserves grammar almost perfectly."
    ),
    "NER Recall": (
        "**NER Recall** tracks what fraction of original named entities (people, places, "
        "organizations) survive each paraphrase iteration. Entity loss is often irreversible: "
        "once a proper noun is replaced or dropped, subsequent tiers cannot recover it. "
        "This makes NER recall a strong signal for forensic analysis."
    ),
    "BERTScore": (
        "**BERTScore** uses contextual embeddings (DeBERTa) to measure semantic similarity "
        "beyond surface words. It remains high even when lexical overlap (BLEU) drops sharply, "
        "confirming that paraphrasers preserve *meaning* while transforming *form*. "
        "Values above 0.80 indicate the core content is largely intact."
    ),
    "BLEU": (
        "**BLEU** measures n-gram precision between the original and paraphrased text. "
        "It is the most sensitive metric to surface-level changes: even moderate paraphrasing "
        "can halve the score. Dipper (High) drives BLEU near zero by T1, while "
        "Pegasus (Slight) retains ~0.64 at T3 due to its conservative word-substitution strategy."
    ),
    "ROUGE-L": (
        "**ROUGE-L** measures the longest common subsequence between texts, capturing "
        "word-order preservation. It decays more gradually than BLEU because shared subsequences "
        "can survive even when individual n-grams are disrupted. The gap between ROUGE-L and BLEU "
        "reveals how much word order is preserved relative to exact phrasing."
    ),
}

GALLERY = {
    "EDA": {
        "path": "figures/eda",
        "figures": [
            ("corpus_distributions.png", "Corpus size distributions across datasets"),
            ("dataset_paraphraser_heatmap.png", "Dataset x Paraphraser sample counts"),
            ("word_count_by_paraphraser_tier.png", "Word count distributions by tier"),
            ("ttr_by_tier_and_paraphraser.png", "Type-Token Ratio decay by tier"),
            ("ttr_by_dataset.png", "TTR variation across datasets"),
            ("punctuation_by_tier.png", "Punctuation density across tiers"),
            ("punct_density_t3_vs_t0.png", "T3 vs T0 punctuation density"),
            ("source_author_profiles.png", "Source author stylometric profiles"),
            ("top_words_t0_vs_t3.png", "Top word frequency shifts T0 vs T3"),
            ("t3_t0_length_ratio_heatmap.png", "T3/T0 length ratio heatmap"),
        ],
    },
    "Baselines": {
        "path": "figures/baselines",
        "figures": [
            ("feature_decay_layered.png", "Layered feature decay (BLEU, ROUGE-L, BERTScore)"),
            ("aggregate_style_vs_content.png", "Style vs content preservation"),
            ("linguistic_delta_by_family.png", "Linguistic delta by paraphraser family"),
        ],
    },
    "Stylometry": {
        "path": "figures/stylometry",
        "figures": [
            ("pos_cosine_decay_curves.png", "POS cosine similarity decay curves"),
            ("pos_tag_delta_heatmap.png", "POS tag delta heatmap across paraphrasers"),
            ("pos_human_vs_llm.png", "POS distributions: Human vs LLM"),
            ("pos_dipper_intensity.png", "Dipper intensity comparison (Low/Default/High)"),
        ],
    },
    "NER Analysis": {
        "path": "figures/ner",
        "figures": [
            ("ner_recall_decay_curves.png", "NER recall decay across tiers"),
            ("entity_survival_funnel.png", "Named entity survival funnel T0 to T3"),
            ("ner_pegasus_comparison.png", "Pegasus Full vs Slight entity preservation"),
        ],
    },
    "Fingerprints & Attribution": {
        "path": "figures/fingerprints",
        "figures": [
            ("attribution_decay_binary.png", "Binary attribution F1 decay"),
            ("identity_trajectory_tsne.png", "Identity trajectory t-SNE visualization"),
            ("identity_quantitative_metrics.png", "Quantitative identity metrics"),
            ("radar_paraphraser_fingerprints.png", "Paraphraser fingerprint radar chart"),
            ("paraphraser_confusion_matrix.png", "Paraphraser identification confusion matrix"),
            ("paraphraser_feature_importance.png", "Feature importance for paraphraser ID"),
            ("full_7model_comparison.png", "Full 7-model comparison"),
            ("style_vs_content_scatter.png", "Style vs content scatter by paraphraser"),
            ("q3_dipper_intensity.png", "RQ3: Dipper intensity analysis"),
        ],
    },
}


# ===== HELPERS =====

# Override config palette with higher-contrast colors for on-screen use.
# The config palette groups families by hue (oranges for Dipper, purples for Pegasus),
# which is too subtle on screen. Here each paraphraser gets a unique hue.
APP_PALETTE = {
    "ChatGPT": "#1abc9c",          # teal
    "PaLM2": "#3498db",            # blue
    "Dipper": "#e67e22",           # orange
    "Dipper (Low)": "#f1c40f",     # golden yellow
    "Dipper (High)": "#e74c3c",    # red
    "Pegasus (Full)": "#8e44ad",   # purple
    "Pegasus (Slight)": "#2ecc71", # green
}


def get_para_color(para_key):
    """Map a paraphraser key to its hex color via APP_PALETTE lookup."""
    return APP_PALETTE.get(LABELS.get(para_key, ""), "#888888")


def hex_to_rgba(hex_color, alpha=0.15):
    """Convert '#1abc9c' to 'rgba(26,188,156,0.15)' for Plotly fill bands."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def entity_html(retained, dropped, novel):
    """Render entity comparison as colored HTML spans."""
    parts = []
    for e in retained:
        parts.append(f'<span style="color:#2ecc71;font-weight:bold">{e}</span>')
    for e in dropped:
        parts.append(f'<span style="color:#e74c3c;text-decoration:line-through">{e}</span>')
    for e in novel:
        parts.append(f'<span style="color:#3498db;font-style:italic">{e}</span>')
    return ", ".join(parts) if parts else "<em>No entities</em>"


# ===== DATA LOADING =====

@st.cache_data
def load_data():
    pkl_path = ROOT / "data" / "processed" / "multitier_results.pkl"
    csv_path = ROOT / "experiments" / "baseline_results" / "similarity_baselines.csv"
    if not pkl_path.exists():
        st.error(
            f"**Missing data file:** `{pkl_path.relative_to(ROOT)}`\n\n"
            "Run `python scripts/run_multitier_analysis.py` to generate it."
        )
        st.stop()
    if not csv_path.exists():
        st.error(
            f"**Missing data file:** `{csv_path.relative_to(ROOT)}`\n\n"
            "Run notebook `02_baseline_similarity.ipynb` to generate baselines."
        )
        st.stop()
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)
    baselines = pd.read_csv(csv_path)
    return results, baselines


results, baselines = load_data()


@st.cache_data(show_spinner="Training classifiers...")
def cached_attribution_experiment(para_key, binary):
    df_lab = results[para_key]
    return run_attribution_experiment(
        df_lab, build_feature_vector, label_col="source", binary=binary
    )


@st.cache_data(show_spinner="Running paraphraser identification...")
def cached_paraphraser_id():
    return run_paraphraser_identification(
        results, build_delta_features, exclude_paraphrasers=["palm"]
    )


def compute_decay_values(para_key, metric, selected_datasets):
    """Return (vals, stds) lists of length 4 for tiers T0-T3."""
    vals = [1.0]
    stds = [0.0]

    if metric == "POS Cosine":
        df = results[para_key]
        df = df[df["dataset"].isin(selected_datasets)]
        for t in ["T1", "T2", "T3"]:
            col = f"cos_{t}"
            if col in df.columns:
                series = df[col].dropna()
                vals.append(series.mean() if len(series) > 0 else None)
                stds.append(series.std() if len(series) > 0 else 0.0)
            else:
                vals.append(None)
                stds.append(0.0)

    elif metric == "NER Recall":
        df = results[para_key]
        df = df[df["dataset"].isin(selected_datasets)]
        for t in ["T1", "T2", "T3"]:
            col = f"ner_recall_{t}"
            if col in df.columns:
                series = df[col].dropna()
                vals.append(series.mean() if len(series) > 0 else None)
                stds.append(series.std() if len(series) > 0 else 0.0)
            else:
                vals.append(None)
                stds.append(0.0)

    elif metric in ["BERTScore", "BLEU", "ROUGE-L"]:
        metric_col = {"BERTScore": "BERTScore", "BLEU": "BLEU", "ROUGE-L": "ROUGE-L"}[metric]
        bl = baselines[baselines["Paraphraser_Key"] == para_key]
        for t in ["T1", "T2", "T3"]:
            row = bl[bl["Tier"] == t]
            if len(row) > 0:
                vals.append(row[metric_col].iloc[0])
                stds.append(row[f"{metric_col}_std"].iloc[0])
            else:
                vals.append(None)
                stds.append(0.0)

    return vals, stds


# ===== SIDEBAR =====
with st.sidebar:
    st.title("Ship of Theseus")
    st.markdown("*Computational Forensics of Authorial Identity Under Iterative Paraphrasing*")
    st.markdown("---")
    st.markdown("**Research Questions**")
    st.markdown("**RQ1:** How do linguistic features decay across paraphrase tiers?")
    st.markdown("**RQ2:** At what tier does authorship attribution fail?")
    st.markdown("**RQ3:** Can we identify which paraphraser was used?")
    st.markdown("---")
    st.caption("CS6120 NLP Spring 2026 | Rohan Joshi & Aarav Surkatha")

plotly_template = "plotly_dark"

# ===== TABS =====
st.title("Ship of Theseus: Decay Dashboard")
tab1, tab2, tab3, tab4 = st.tabs([
    "Decay Explorer", "Document Forensics", "Attribution Lab", "Research Gallery",
])


# ===== TAB 1: DECAY EXPLORER =====
with tab1:
    st.header("Feature Decay Across Paraphrase Iterations")

    col1, col2 = st.columns([1, 3])
    with col1:
        selected_paras = st.multiselect(
            "Select paraphrasers:",
            options=list(LABELS.keys()),
            default=["chatgpt", "dipper(high)", "pegasus(slight)"],
            format_func=lambda x: LABELS[x],
        )
        metric = st.selectbox(
            "Metric:",
            ["POS Cosine", "NER Recall", "BERTScore", "BLEU", "ROUGE-L"],
        )
        selected_datasets = st.multiselect(
            "Filter datasets:",
            options=ALL_DATASETS,
            default=ALL_DATASETS,
            key="decay_datasets",
        )
        if metric in ["BERTScore", "BLEU", "ROUGE-L"] and len(selected_datasets) < len(ALL_DATASETS):
            st.caption("Note: Baseline metrics (BLEU, ROUGE-L, BERTScore) are pre-aggregated across all datasets.")

    with col2:
        # Summary metric cards
        if selected_paras:
            metric_cols = st.columns(min(len(selected_paras), 7))
            for i, para_key in enumerate(selected_paras):
                decay_vals, _ = compute_decay_values(para_key, metric, selected_datasets)
                t3_val = decay_vals[3]
                with metric_cols[i]:
                    st.metric(
                        label=LABELS[para_key],
                        value=f"{t3_val:.3f}" if t3_val is not None else "N/A",
                        delta=f"{t3_val - 1.0:.3f}" if t3_val is not None else None,
                        delta_color="inverse",
                    )

        # Decay chart with error bands
        fig = go.Figure()
        tiers = ["T0", "T1", "T2", "T3"]

        for para_key in selected_paras:
            name = LABELS[para_key]
            color = get_para_color(para_key)
            vals, stds = compute_decay_values(para_key, metric, selected_datasets)

            # Main line
            fig.add_trace(go.Scatter(
                x=tiers, y=vals, mode="lines+markers", name=name,
                line=dict(width=3, color=color),
                marker=dict(size=10, color=color),
            ))

            # Error band (upper)
            upper = [v + s if v is not None else None for v, s in zip(vals, stds)]
            lower = [v - s if v is not None else None for v, s in zip(vals, stds)]
            fig.add_trace(go.Scatter(
                x=tiers, y=upper, mode="lines", line=dict(width=0),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=tiers, y=lower, mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor=hex_to_rgba(color, 0.15),
                showlegend=False, hoverinfo="skip",
            ))

        fig.update_layout(
            title=f"{metric} Decay Across Iterations",
            xaxis_title="Paraphrase Tier",
            yaxis_title=f"{metric} (relative to T0)",
            yaxis_range=[0, 1.05],
            height=500,
            template=plotly_template,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5,
            ),
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                      annotation_text="Random baseline")
        st.plotly_chart(fig, width='stretch')

        # Metric interpretation
        st.info(METRIC_EXPLANATIONS[metric])


# ===== TAB 2: DOCUMENT FORENSICS =====
with tab2:
    st.header("Document Forensics: T0 vs Paraphrased Versions")
    st.info(
        "Compare the original text (T0) against successive paraphrase iterations (T1-T3). "
        "Each tier applies the same paraphraser again to the previous tier's output. "
        "Watch how phrasing diverges while core meaning often persists, and how named entities "
        "are progressively dropped, replaced, or hallucinated across tiers."
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        para_key = st.selectbox(
            "Paraphraser:",
            options=list(LABELS.keys()),
            format_func=lambda x: LABELS[x],
            key="doc_para",
        )
        df = results[para_key]
        datasets = sorted(df["dataset"].unique())
        dataset = st.selectbox("Dataset:", datasets)

        df_filtered = df[df["dataset"] == dataset]
        article_idx = st.slider("Article:", 0, len(df_filtered) - 1, 0)

    with col2:
        row = df_filtered.iloc[article_idx]
        st.markdown(f"**Key:** `{row['key']}` | **Source:** `{row['source']}` | **Dataset:** `{row['dataset']}`")

        tier_cols = st.columns(4)
        for i, tier in enumerate(["T0", "T1", "T2", "T3"]):
            text_col = f"text_{tier}"
            with tier_cols[i]:
                st.markdown(f"**{tier}**")
                if text_col in row.index and isinstance(row[text_col], str):
                    text = row[text_col][:800]
                    st.text_area(f"text_{tier}", text, height=300,
                                 key=f"text_{tier}_{article_idx}",
                                 disabled=True, label_visibility="collapsed")
                else:
                    st.warning("Not available")

        # Entity comparison with colored HTML
        st.markdown("---")
        st.subheader("Named Entity Comparison")
        st.markdown(
            '<span style="color:#2ecc71">&#9632; Retained</span> &nbsp; '
            '<span style="color:#e74c3c">&#9632; Dropped</span> &nbsp; '
            '<span style="color:#3498db">&#9632; Novel</span>',
            unsafe_allow_html=True,
        )
        ner_cols = st.columns(4)
        for i, tier in enumerate(["T0", "T1", "T2", "T3"]):
            ner_col = f"ner_{tier}"
            with ner_cols[i]:
                st.markdown(f"**{tier} Entities:**")
                if ner_col in row.index and isinstance(row[ner_col], (set, frozenset)):
                    entities = sorted(row[ner_col])
                    if tier == "T0":
                        html = ", ".join(
                            f'<span style="color:#2ecc71;font-weight:bold">{e}</span>'
                            for e in entities
                        ) if entities else "<em>No entities</em>"
                        st.markdown(html, unsafe_allow_html=True)
                    else:
                        t0_ents = row.get("ner_T0", set())
                        if isinstance(t0_ents, (set, frozenset)):
                            surviving = sorted(row[ner_col] & t0_ents)
                            novel = sorted(row[ner_col] - t0_ents)
                            dropped = sorted(t0_ents - row[ner_col])
                            html = entity_html(surviving, dropped, novel)
                            st.markdown(html, unsafe_allow_html=True)
                        else:
                            html = ", ".join(entities) if entities else "<em>No entities</em>"
                            st.markdown(html, unsafe_allow_html=True)
                else:
                    st.write("*Not available*")

        # T0 vs T3 Direct Comparison
        st.markdown("---")
        if st.checkbox("Show T0 vs T3 focused comparison", value=False, key="t0t3_toggle"):
            st.subheader("T0 vs T3 Direct Comparison")
            c_left, c_right = st.columns(2)
            t0_text = row.get("text_T0", "")
            t3_text = row.get("text_T3", "")
            with c_left:
                st.markdown("**Original (T0)**")
                if isinstance(t0_text, str):
                    st.text_area("t0_full", t0_text, height=350, disabled=True,
                                 label_visibility="collapsed", key="t0_full_view")
                else:
                    st.warning("T0 text not available")
            with c_right:
                st.markdown("**After 3 iterations (T3)**")
                if isinstance(t3_text, str):
                    st.text_area("t3_full", t3_text, height=350, disabled=True,
                                 label_visibility="collapsed", key="t3_full_view")
                else:
                    st.warning("T3 text not available")

            # Per-document metrics
            if isinstance(t0_text, str) and isinstance(t3_text, str):
                from src.utils.metrics import bleu, rouge_l
                met_cols = st.columns(3)
                with met_cols[0]:
                    st.metric("BLEU", f"{bleu(t0_text, t3_text):.4f}")
                with met_cols[1]:
                    st.metric("ROUGE-L", f"{rouge_l(t0_text, t3_text):.4f}")
                with met_cols[2]:
                    t0_ents = row.get("ner_T0", set())
                    t3_ents = row.get("ner_T3", set())
                    if isinstance(t0_ents, (set, frozenset)) and len(t0_ents) > 0:
                        retention = len(t0_ents & t3_ents) / len(t0_ents)
                        st.metric("Entity Retention", f"{retention:.1%}")
                    else:
                        st.metric("Entity Retention", "N/A")


# ===== TAB 3: ATTRIBUTION LAB =====
with tab3:
    st.header("Attribution Lab: Does the Author's Identity Survive?")

    col1, col2 = st.columns([1, 3])
    with col1:
        lab_para = st.selectbox(
            "Paraphraser:",
            options=["dipper", "dipper(low)", "dipper(high)",
                     "pegasus(full)", "pegasus(slight)", "chatgpt"],
            format_func=lambda x: LABELS[x],
            key="lab_para",
        )
        lab_mode = st.radio("Classification:", ["Binary (Human vs LLM)", "7-Class (All Sources)"])

    with col2:
        binary = lab_mode.startswith("Binary")
        exp_results, feat_names = cached_attribution_experiment(lab_para, binary)

        # Summary metric cards
        best_clf = max(
            exp_results,
            key=lambda c: exp_results[c].get("T0", {}).get("macro_f1", 0),
        )
        t0_f1 = exp_results[best_clf]["T0"]["macro_f1"]
        t3_f1 = exp_results[best_clf].get("T3", {}).get("macro_f1", 0)

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("Best Classifier", best_clf)
        with mc2:
            st.metric("T0 Macro F1", f"{t0_f1:.3f}")
        with mc3:
            st.metric("T3 Macro F1", f"{t3_f1:.3f}",
                       delta=f"{t3_f1 - t0_f1:.3f}", delta_color="inverse")

        # Decay curves
        fig = go.Figure()
        tiers_lab = ["T0", "T1", "T2", "T3"]

        for clf_name, tier_results in exp_results.items():
            f1_vals = []
            for t in tiers_lab:
                if t in tier_results:
                    f1_vals.append(tier_results[t]["macro_f1"])
                else:
                    f1_vals.append(None)
            color = CLF_PALETTE.get(clf_name, "#888888")
            fig.add_trace(go.Scatter(
                x=tiers_lab, y=f1_vals, mode="lines+markers", name=clf_name,
                line=dict(width=3, color=color),
                marker=dict(size=10, color=color),
            ))

        fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                      annotation_text="Random baseline (F1=0.5)")
        fig.update_layout(
            title=f"Attribution Decay: {LABELS[lab_para]} ({lab_mode})",
            xaxis_title="Paraphrase Tier",
            yaxis_title="Macro F1",
            yaxis_range=[0, 1.05],
            height=450,
            template=plotly_template,
        )
        st.plotly_chart(fig, width='stretch')

        st.info(
            "**RQ2: Attribution Decay.** Each classifier is trained on T0 features and tested on "
            "T1-T3. As paraphrasing erodes stylometric signals, F1 drops. When F1 falls below 0.5, "
            "the classifier performs no better than random — the author's identity has been effectively erased. "
            "Random Forest tends to be most resilient due to its non-linear feature interactions."
        )

        # Point of no return table
        st.subheader("Point of No Return")
        ponr_data = []
        for clf_name, tier_results in exp_results.items():
            point = "Survives T3"
            for t in ["T1", "T2", "T3"]:
                if t in tier_results and tier_results[t]["macro_f1"] < 0.5:
                    point = t
                    break
            ponr_data.append({"Classifier": clf_name, "Point of No Return": point})
        st.table(pd.DataFrame(ponr_data))

        # RQ3: Paraphraser Identification
        st.markdown("---")
        st.subheader("RQ3: Paraphraser Identification")
        st.markdown("Can we identify *which* paraphraser was used from the T0-to-T1 feature delta?")
        st.info(
            "**RQ3: Paraphraser Fingerprinting.** Instead of asking *who wrote this*, we ask *what tool transformed it*. "
            "Delta features (T1 minus T0) capture the paraphraser's signature: how it shifts POS distributions, "
            "alters dependency depth, and disrupts named entities. High confusion between Dipper variants "
            "suggests they share a common transformation backbone, while ChatGPT and Pegasus leave distinct fingerprints."
        )

        try:
            rq3_results, rq3_feat_names, rq3_labels = cached_paraphraser_id()

            # Summary table
            acc_data = []
            for clf_name, res in rq3_results.items():
                acc_data.append({
                    "Classifier": clf_name,
                    "Accuracy": f"{res['accuracy']:.3f}",
                    "Macro F1": f"{res['macro_f1']:.3f}",
                })
            st.table(pd.DataFrame(acc_data))

            # Confusion matrix (Random Forest)
            rf_res = rq3_results["Random Forest"]
            cm = rf_res["confusion_matrix"]
            display_labels = [LABELS.get(l, l) for l in rq3_labels]

            fig_cm = px.imshow(
                cm, x=display_labels, y=display_labels,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                color_continuous_scale="Teal",
                title="Random Forest: Paraphraser Identification Confusion Matrix",
            )
            fig_cm.update_layout(template=plotly_template, height=450)
            st.plotly_chart(fig_cm, width='stretch')

            # Feature importance (top 12)
            if "feature_importance" in rf_res:
                fi = rf_res["feature_importance"]
                fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:12]
                fig_fi = px.bar(
                    x=[v for _, v in fi_sorted],
                    y=[k for k, _ in fi_sorted],
                    orientation="h",
                    title="Top 12 Features for Paraphraser Identification",
                    color_discrete_sequence=["#1abc9c"],
                )
                fig_fi.update_layout(
                    yaxis=dict(autorange="reversed"),
                    xaxis_title="Importance",
                    yaxis_title="",
                    template=plotly_template,
                    height=400,
                )
                st.plotly_chart(fig_fi, width='stretch')

        except ValueError as e:
            st.warning(f"Paraphraser identification unavailable: {e}")


# ===== TAB 4: RESEARCH GALLERY =====
with tab4:
    st.header("Research Gallery")
    st.markdown("Pre-generated visualizations from the full analysis pipeline.")

    category = st.selectbox("Category:", list(GALLERY.keys()))
    cat = GALLERY[category]
    figures_list = cat["figures"]

    for i in range(0, len(figures_list), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(figures_list):
                fname, caption = figures_list[i + j]
                img_path = ROOT / cat["path"] / fname
                with col:
                    if img_path.exists():
                        st.image(str(img_path), caption=caption,
                                 width='stretch')
                    else:
                        st.warning(f"Missing: {fname}")


# ===== FOOTER =====
st.markdown("---")
st.caption("Ship of Theseus | Computational Forensics of Authorial Identity")
