"""
Tasks 2-4: POS drift, NER stability, and paraphraser fingerprint report.

Usage:
    python -m src.features.task2_4_analysis
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy

from src.data.task1_pairing import (
    build_pairs_for_versions,
    load_task1_corpus,
)
from src.utils.config import DATA_PROCESSED, EXPERIMENTS_DIR, FIGURES_DIR


UPOS_TAGS = [
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
]

EXTENDED_VERSION_MAP = {
    "original": "text_T0",
    "orignal": "text_T0",
    "chatgpt": "text_chatgpt",
    "dipper(low)": "text_dipper_low",
    "dipper(high)": "text_dipper_high",
    "pegasus(slight)": "text_pegasus_slight",
    "pegasus(full)": "text_pegasus_full",
}

TEXT_COLUMNS = [
    "text_T0",
    "text_chatgpt",
    "text_dipper_low",
    "text_dipper_high",
    "text_pegasus_slight",
    "text_pegasus_full",
]

OUTPUT_DIR = EXPERIMENTS_DIR / "task2_4_analysis"
FIGURE_DIR = FIGURES_DIR / "task2_4_analysis"
PAIRED_CACHE = DATA_PROCESSED / "paired_extended.pkl"
ANNOTATION_CACHE = DATA_PROCESSED / "task2_4_annotations.pkl"


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def load_extended_pairs(force_rebuild=False):
    """Load cached extended pairs or rebuild from raw data."""
    if PAIRED_CACHE.exists() and not force_rebuild:
        paired = pd.read_pickle(PAIRED_CACHE)
        print(f"Loaded cached paired data: {PAIRED_CACHE} {paired.shape}")
        return paired

    corpus = load_task1_corpus()
    paired = build_pairs_for_versions(corpus, EXTENDED_VERSION_MAP)

    required = [
        "text_T0",
        "text_chatgpt",
        "text_dipper_low",
        "text_dipper_high",
        "text_pegasus_slight",
        "text_pegasus_full",
    ]
    paired = paired.dropna(subset=required).reset_index(drop=True)
    paired.to_pickle(PAIRED_CACHE)
    print(f"Saved extended paired data: {PAIRED_CACHE} {paired.shape}")
    return paired


def load_spacy_model():
    """Load spaCy English model with parser disabled for throughput."""
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    except OSError as exc:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is not installed. "
            "Run: python -m spacy download en_core_web_sm"
        ) from exc


def doc_pos_distribution(doc):
    """Convert a spaCy Doc into a normalized UPOS distribution."""
    counts = dict.fromkeys(UPOS_TAGS, 0.0)
    total = 0
    for token in doc:
        if token.is_space:
            continue
        total += 1
        tag = token.pos_
        if tag in counts:
            counts[tag] += 1.0
    if total == 0:
        return counts
    return {tag: value / total for tag, value in counts.items()}


def doc_entity_set(doc):
    """Extract a lowercase set of named entity strings."""
    entities = set()
    for ent in doc.ents:
        text = ent.text.strip().lower()
        if text:
            entities.add(text)
    return entities


def annotate_text_columns(paired, force_rebuild=False):
    """Run spaCy over all required text columns and cache row-level annotations."""
    if ANNOTATION_CACHE.exists() and not force_rebuild:
        annotations = pd.read_pickle(ANNOTATION_CACHE)
        print(f"Loaded cached annotations: {ANNOTATION_CACHE} {annotations.shape}")
        return annotations

    nlp = load_spacy_model()
    annotated = paired[["dataset", "source", "key"] + TEXT_COLUMNS].copy()

    for column in TEXT_COLUMNS:
        texts = annotated[column].fillna("").astype(str).tolist()
        pos_values = []
        ent_values = []
        print(f"Annotating {column} ({len(texts):,} texts)")
        for doc in nlp.pipe(texts, batch_size=32):
            pos_values.append(doc_pos_distribution(doc))
            ent_values.append(sorted(doc_entity_set(doc)))
        annotated[f"pos_{column[5:]}"] = pos_values
        annotated[f"entities_{column[5:]}"] = ent_values

    keep_columns = ["dataset", "source", "key"]
    keep_columns.extend(col for col in annotated.columns if col.startswith("pos_"))
    keep_columns.extend(col for col in annotated.columns if col.startswith("entities_"))
    annotated = annotated[keep_columns]
    annotated.to_pickle(ANNOTATION_CACHE)
    print(f"Saved annotations: {ANNOTATION_CACHE}")
    return annotated


def cosine_similarity_from_dicts(left, right):
    """Cosine similarity between two sparse distribution dictionaries."""
    left_vec = np.array([left.get(tag, 0.0) for tag in UPOS_TAGS], dtype=float)
    right_vec = np.array([right.get(tag, 0.0) for tag in UPOS_TAGS], dtype=float)
    denom = np.linalg.norm(left_vec) * np.linalg.norm(right_vec)
    if denom == 0:
        return np.nan
    return float(np.dot(left_vec, right_vec) / denom)


def entity_metrics(source_entities, target_entities):
    """
    Compute Jaccard, Recall, Precision for T0 vs T1 entities.

    If T0 has zero entities, all metrics are NaN per assignment.
    If T1 has zero entities while T0 is non-empty, precision is set to 0.0:
    the paraphrase retained none of the original named entities in T1 output.
    """
    source_set = set(source_entities)
    target_set = set(target_entities)

    if len(source_set) == 0:
        return {"jaccard": np.nan, "recall": np.nan, "precision": np.nan}

    overlap = source_set & target_set
    union = source_set | target_set

    jaccard = len(overlap) / len(union) if union else np.nan
    recall = len(overlap) / len(source_set)
    precision = len(overlap) / len(target_set) if target_set else 0.0

    return {"jaccard": jaccard, "recall": recall, "precision": precision}


def compute_article_metrics(annotations):
    """Compute article-level POS cosine and NER metrics for all needed versions."""
    metrics = annotations[["dataset", "source", "key"]].copy()

    metrics["cosine_chatgpt"] = annotations.apply(
        lambda row: cosine_similarity_from_dicts(row["pos_T0"], row["pos_chatgpt"]),
        axis=1,
    )
    metrics["cosine_dipper_low"] = annotations.apply(
        lambda row: cosine_similarity_from_dicts(row["pos_T0"], row["pos_dipper_low"]),
        axis=1,
    )
    metrics["cosine_dipper_high"] = annotations.apply(
        lambda row: cosine_similarity_from_dicts(row["pos_T0"], row["pos_dipper_high"]),
        axis=1,
    )

    for version in ["chatgpt", "dipper_low", "dipper_high", "pegasus_slight", "pegasus_full"]:
        entity_cols = annotations.apply(
            lambda row: entity_metrics(row["entities_T0"], row[f"entities_{version}"]),
            axis=1,
        )
        metrics[f"jaccard_{version}"] = entity_cols.apply(lambda x: x["jaccard"])
        metrics[f"recall_{version}"] = entity_cols.apply(lambda x: x["recall"])
        metrics[f"precision_{version}"] = entity_cols.apply(lambda x: x["precision"])

    metrics["source_group"] = np.where(metrics["source"] == "Human", "Human", "LLM")
    return metrics


def summarize_series(series):
    """Mean and std helper."""
    clean = series.dropna()
    return pd.Series(
        {
            "n": int(clean.shape[0]),
            "mean": float(clean.mean()) if len(clean) else np.nan,
            "std": float(clean.std(ddof=1)) if len(clean) > 1 else np.nan,
        }
    )


def task2b_summary(metrics):
    """Overall POS cosine for T0 vs dipper(high)."""
    summary = summarize_series(metrics["cosine_dipper_high"])
    summary.name = "dipper_high"
    return summary


def task2c_group_summary(metrics):
    """Aggregate human vs LLM POS cosine by dataset, then across datasets."""
    rows = []
    for paraphraser, cosine_col in {
        "chatgpt": "cosine_chatgpt",
        "dipper(high)": "cosine_dipper_high",
    }.items():
        grouped = (
            metrics.groupby(["dataset", "source_group"])[cosine_col]
            .mean()
            .reset_index(name="dataset_mean")
        )
        grouped["paraphraser"] = paraphraser
        rows.append(grouped)

    dataset_level = pd.concat(rows, ignore_index=True)
    aggregate = (
        dataset_level.groupby(["paraphraser", "source_group"])["dataset_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n_datasets"})
    )
    return dataset_level, aggregate


def plot_task2c(aggregate):
    """Grouped bar chart for Task 2c."""
    fig, ax = plt.subplots(figsize=(8, 5))
    paraphrasers = ["chatgpt", "dipper(high)"]
    groups = ["Human", "LLM"]
    x = np.arange(len(paraphrasers))
    width = 0.35

    for offset, group in [(-width / 2, "Human"), (width / 2, "LLM")]:
        subset = aggregate[aggregate["source_group"] == group].set_index("paraphraser")
        means = [subset.loc[p, "mean"] for p in paraphrasers]
        stds = [subset.loc[p, "std"] for p in paraphrasers]
        ax.bar(x + offset, means, width, yerr=stds, capsize=4, label=group)

    ax.set_xticks(x)
    ax.set_xticklabels(["ChatGPT", "Dipper (High)"])
    ax.set_ylabel("Mean POS Cosine Similarity")
    ax.set_xlabel("Paraphraser")
    ax.set_title("POS Drift by Original Source Group")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Original Source")
    plt.tight_layout()

    out_path = FIGURE_DIR / "task2c_human_vs_llm_pos_cosine.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def task3_summary(metrics):
    """Entity retention summary for chatgpt, dipper(high), pegasus(slight)."""
    records = []
    for paraphraser, suffix in {
        "chatgpt": "chatgpt",
        "dipper(high)": "dipper_high",
        "pegasus(slight)": "pegasus_slight",
    }.items():
        for metric_name in ["jaccard", "recall", "precision"]:
            series = metrics[f"{metric_name}_{suffix}"]
            summary = summarize_series(series)
            records.append(
                {
                    "paraphraser": paraphraser,
                    "metric": metric_name,
                    "n": summary["n"],
                    "mean": summary["mean"],
                    "std": summary["std"],
                }
            )
    return pd.DataFrame(records)


def task3_exclusion_stats(metrics):
    """Count articles with zero T0 entities."""
    excluded = metrics["jaccard_chatgpt"].isna().sum()
    total = len(metrics)
    return {
        "excluded_articles": int(excluded),
        "total_articles": int(total),
        "excluded_fraction": float(excluded / total) if total else np.nan,
    }


def task3c_pegasus_summary(metrics):
    """Compare pegasus(slight) vs pegasus(full)."""
    rows = []
    for label, suffix in {
        "pegasus(slight)": "pegasus_slight",
        "pegasus(full)": "pegasus_full",
    }.items():
        for metric_name in ["jaccard", "recall", "precision"]:
            series = metrics[f"{metric_name}_{suffix}"]
            summary = summarize_series(series)
            rows.append(
                {
                    "variant": label,
                    "metric": metric_name,
                    "n": summary["n"],
                    "mean": summary["mean"],
                    "std": summary["std"],
                }
            )
    summary = pd.DataFrame(rows)

    dataset_rows = []
    for dataset, ds_df in metrics.groupby("dataset"):
        for label, suffix in {
            "pegasus(slight)": "pegasus_slight",
            "pegasus(full)": "pegasus_full",
        }.items():
            for metric_name in ["jaccard", "recall", "precision"]:
                series = ds_df[f"{metric_name}_{suffix}"]
                dataset_rows.append(
                    {
                        "dataset": dataset,
                        "variant": label,
                        "metric": metric_name,
                        "mean": float(series.dropna().mean()) if series.notna().any() else np.nan,
                    }
                )
    dataset_summary = pd.DataFrame(dataset_rows)
    return summary, dataset_summary


def plot_task3c(summary):
    """Grouped bar chart for pegasus(slight) vs pegasus(full)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics_order = ["jaccard", "recall", "precision"]
    variants = ["pegasus(slight)", "pegasus(full)"]
    x = np.arange(len(metrics_order))
    width = 0.35

    for offset, variant in [(-width / 2, "pegasus(slight)"), (width / 2, "pegasus(full)")]:
        subset = summary[summary["variant"] == variant].set_index("metric")
        means = [subset.loc[m, "mean"] for m in metrics_order]
        stds = [subset.loc[m, "std"] for m in metrics_order]
        ax.bar(x + offset, means, width, yerr=stds, capsize=4, label=variant)

    ax.set_xticks(x)
    ax.set_xticklabels(["Jaccard", "Recall", "Precision"])
    ax.set_ylabel("Mean Score")
    ax.set_xlabel("Metric")
    ax.set_title("Pegasus Coverage Experiment: Entity Stability")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Variant")
    plt.tight_layout()

    out_path = FIGURE_DIR / "task3c_pegasus_coverage_metrics.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def format_mean_std(mean, std):
    """Compact formatting for report text."""
    if pd.isna(mean):
        return "NaN"
    if pd.isna(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} +/- {std:.3f}"


def write_markdown_answers(task2c_aggregate, task3_exclusions, task3c_summary, task3c_by_dataset, metrics):
    """Write markdown responses for Tasks 2-4."""
    human_chatgpt = task2c_aggregate[
        (task2c_aggregate["paraphraser"] == "chatgpt")
        & (task2c_aggregate["source_group"] == "Human")
    ].iloc[0]
    llm_chatgpt = task2c_aggregate[
        (task2c_aggregate["paraphraser"] == "chatgpt")
        & (task2c_aggregate["source_group"] == "LLM")
    ].iloc[0]
    human_dipper = task2c_aggregate[
        (task2c_aggregate["paraphraser"] == "dipper(high)")
        & (task2c_aggregate["source_group"] == "Human")
    ].iloc[0]
    llm_dipper = task2c_aggregate[
        (task2c_aggregate["paraphraser"] == "dipper(high)")
        & (task2c_aggregate["source_group"] == "LLM")
    ].iloc[0]

    task2_text = (
        f"Across datasets, Human-authored originals show POS cosine "
        f"{format_mean_std(human_chatgpt['mean'], human_chatgpt['std'])} under ChatGPT "
        f"and {format_mean_std(human_dipper['mean'], human_dipper['std'])} under Dipper (High), "
        f"while LLM-authored originals show "
        f"{format_mean_std(llm_chatgpt['mean'], llm_chatgpt['std'])} and "
        f"{format_mean_std(llm_dipper['mean'], llm_dipper['std'])}, respectively. "
        f"If the Human bars are lower, then POS distributions shift more for Human text; "
        f"if the LLM bars are lower, then paraphrasers move LLM-authored text further from its starting style. "
        f"A plausible explanation is that LLM-generated T0 text already occupies an AI-like stylistic basin, "
        f"so another LLM paraphraser may preserve its POS profile more easily than it preserves the broader syntactic "
        f"range seen in Human writing. That makes POS drift partly a measure of how far the starting text already aligns "
        f"with the paraphraser’s preferred distribution."
    )

    exclusion_text = (
        f"Articles excluded because T0 had zero named entities: "
        f"{task3_exclusions['excluded_articles']} / {task3_exclusions['total_articles']} "
        f"({task3_exclusions['excluded_fraction']:.2%})."
    )

    pegasus = task3c_summary.pivot(index="metric", columns="variant", values="mean")
    recall_gap = abs(pegasus.loc["recall", "pegasus(slight)"] - pegasus.loc["recall", "pegasus(full)"])
    precision_gap = abs(
        pegasus.loc["precision", "pegasus(slight)"] - pegasus.loc["precision", "pegasus(full)"]
    )

    dataset_pivot = task3c_by_dataset.pivot_table(
        index=["dataset", "metric"],
        columns="variant",
        values="mean",
    ).reset_index()
    dataset_pivot["gap"] = (
        dataset_pivot["pegasus(slight)"] - dataset_pivot["pegasus(full)"]
    ).abs()
    recall_domains = dataset_pivot[dataset_pivot["metric"] == "recall"][["dataset", "gap"]]
    precision_domains = dataset_pivot[dataset_pivot["metric"] == "precision"][["dataset", "gap"]]
    widest_recall = recall_domains.sort_values("gap", ascending=False).iloc[0]
    widest_precision = precision_domains.sort_values("gap", ascending=False).iloc[0]

    task3c_text = (
        f"Pegasus coverage changes Recall by {recall_gap:.3f} points and Precision by {precision_gap:.3f} points "
        f"when moving from pegasus(slight) to pegasus(full). "
        f"If the Recall gap is larger, then broader sentence coverage mainly increases entity dropping; "
        f"if the Precision gap is larger, then broader coverage mainly increases hallucinated entities. "
        f"The largest dataset-level Recall gap appears in {widest_recall['dataset']} ({widest_recall['gap']:.3f}), "
        f"while the largest Precision gap appears in {widest_precision['dataset']} ({widest_precision['gap']:.3f}). "
        f"That indicates whether the pattern is stable or domain-sensitive: news, reviews, and scientific text often "
        f"differ in entity density, so full-coverage paraphrasing may erode named entities more strongly in some domains "
        f"than others."
    )

    dataset_cosine = (
        metrics.groupby("dataset")[["cosine_chatgpt", "cosine_dipper_high"]]
        .mean()
        .reset_index()
    )
    dataset_cosine["delta"] = dataset_cosine["cosine_chatgpt"] - dataset_cosine["cosine_dipper_high"]
    most_contrast = dataset_cosine.sort_values("delta").iloc[0]
    opposite_contrast = dataset_cosine.sort_values("delta", ascending=False).iloc[0]

    overall_cosines = {
        "chatgpt": summarize_series(metrics["cosine_chatgpt"]),
        "dipper_high": summarize_series(metrics["cosine_dipper_high"]),
    }
    task4_q1 = (
        f"Across the full corpus, ChatGPT yields POS cosine {format_mean_std(overall_cosines['chatgpt']['mean'], overall_cosines['chatgpt']['std'])}, "
        f"while Dipper (High) yields {format_mean_std(overall_cosines['dipper_high']['mean'], overall_cosines['dipper_high']['std'])}. "
        f"The lower value marks the paraphraser with stronger stylistic drift. Domain effects can be tested directly in the per-dataset means: "
        f"{most_contrast['dataset']} shows ChatGPT {most_contrast['cosine_chatgpt']:.3f} vs Dipper (High) {most_contrast['cosine_dipper_high']:.3f}, "
        f"whereas {opposite_contrast['dataset']} shows ChatGPT {opposite_contrast['cosine_chatgpt']:.3f} vs Dipper (High) "
        f"{opposite_contrast['cosine_dipper_high']:.3f}. If the ordering flips or narrows sharply, then domain matters; if it stays stable, "
        f"the paraphraser ranking is robust across datasets."
    )

    task3_entity = task3_summary(metrics)
    recall_rank = (
        task3_entity[task3_entity["metric"] == "recall"]
        .sort_values("mean")
        .reset_index(drop=True)
    )
    precision_rank = (
        task3_entity[task3_entity["metric"] == "precision"]
        .sort_values("mean")
        .reset_index(drop=True)
    )
    worst_recall = recall_rank.iloc[0]
    worst_precision = precision_rank.iloc[0]
    task4_q2 = (
        f"For entity dropping, the lowest Recall belongs to {worst_recall['paraphraser']} "
        f"at {format_mean_std(worst_recall['mean'], worst_recall['std'])}. "
        f"For hallucination control, the lowest Precision belongs to {worst_precision['paraphraser']} "
        f"at {format_mean_std(worst_precision['mean'], worst_precision['std'])}. "
        f"Those values distinguish whether one paraphraser mainly deletes original entities, invents new ones, or does both. "
        f"If the same system ranks worst on both metrics, then it is broadly the least faithful. "
        f"If different systems rank worst, then dropping and hallucination are separate failure modes rather than one combined weakness."
    )

    low_cos = summarize_series(metrics["cosine_dipper_low"])
    high_cos = summarize_series(metrics["cosine_dipper_high"])
    low_rec = summarize_series(metrics["recall_dipper_low"])
    high_rec = summarize_series(metrics["recall_dipper_high"])
    task4_q3 = (
        f"Dipper (Low) has POS cosine {format_mean_std(low_cos['mean'], low_cos['std'])} and entity Recall "
        f"{format_mean_std(low_rec['mean'], low_rec['std'])}, while Dipper (High) has POS cosine "
        f"{format_mean_std(high_cos['mean'], high_cos['std'])} and entity Recall "
        f"{format_mean_std(high_rec['mean'], high_rec['std'])}. "
        f"If the cosine drop from low to high is larger than the Recall drop, then increasing intensity affects style more than content; "
        f"if Recall falls more sharply, then content preservation degrades faster than stylistic change grows. "
        f"That result feeds directly into RQ1: it tells us whether turning up paraphrase intensity primarily changes linguistic surface form "
        f"or whether it also strips away the semantic planks that anchor document identity."
    )

    (OUTPUT_DIR / "task2c_interpretation.md").write_text(task2_text + "\n")
    (OUTPUT_DIR / "task3b_exclusions.md").write_text(exclusion_text + "\n")
    (OUTPUT_DIR / "task3c_interpretation.md").write_text(task3c_text + "\n")
    (OUTPUT_DIR / "task4_report.md").write_text(
        "\n\n".join([task4_q1, task4_q2, task4_q3]) + "\n"
    )


def save_outputs(metrics, task2b, task2c_dataset, task2c_aggregate, task3b, task3c, task3c_by_dataset):
    """Write CSV outputs for notebook/report use."""
    metrics.to_csv(OUTPUT_DIR / "article_level_metrics.csv", index=False)
    task2b.to_frame().T.to_csv(OUTPUT_DIR / "task2b_overall_pos_cosine.csv", index=False)
    task2c_dataset.to_csv(OUTPUT_DIR / "task2c_dataset_level_group_means.csv", index=False)
    task2c_aggregate.to_csv(OUTPUT_DIR / "task2c_aggregate_group_means.csv", index=False)
    task3b.to_csv(OUTPUT_DIR / "task3b_entity_retention_summary.csv", index=False)
    task3c.to_csv(OUTPUT_DIR / "task3c_pegasus_summary.csv", index=False)
    task3c_by_dataset.to_csv(OUTPUT_DIR / "task3c_pegasus_by_dataset.csv", index=False)


def main():
    ensure_dirs()

    paired = load_extended_pairs()
    annotations = annotate_text_columns(paired)
    metrics = compute_article_metrics(annotations)

    task2b = task2b_summary(metrics)
    task2c_dataset, task2c_aggregate = task2c_group_summary(metrics)
    task3b = task3_summary(metrics)
    task3_exclusions = task3_exclusion_stats(metrics)
    task3c, task3c_by_dataset = task3c_pegasus_summary(metrics)

    save_outputs(metrics, task2b, task2c_dataset, task2c_aggregate, task3b, task3c, task3c_by_dataset)
    task2_fig = plot_task2c(task2c_aggregate)
    task3_fig = plot_task3c(task3c)
    write_markdown_answers(task2c_aggregate, task3_exclusions, task3c, task3c_by_dataset, metrics)

    print("\nTask 2b")
    print(f"Dipper (High) POS cosine: {format_mean_std(task2b['mean'], task2b['std'])}")

    print("\nTask 2c aggregate")
    print(task2c_aggregate.to_string(index=False))

    print("\nTask 3b exclusions")
    print(task3_exclusions)

    print("\nTask 3b summary")
    print(task3b.to_string(index=False))

    print("\nTask 3c summary")
    print(task3c.to_string(index=False))

    print("\nSaved files")
    print(f"- CSV outputs: {OUTPUT_DIR}")
    print(f"- Task 2 chart: {task2_fig}")
    print(f"- Task 3 chart: {task3_fig}")
    print(f"- Markdown answers: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
