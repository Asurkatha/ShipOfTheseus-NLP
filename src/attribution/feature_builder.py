"""
Feature engineering for authorship attribution and paraphraser identification.

Builds numeric feature vectors from the multitier_results.pkl DataFrames,
combining POS distributions, NER metrics, dependency depth, and text statistics.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_text_stats(text):
    """Compute basic text statistics from raw text."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {"word_count": 0, "sentence_count": 0, "avg_word_length": 0.0,
                "ttr": 0.0, "punct_density": 0.0}

    words = text.split()
    word_count = len(words)
    sentence_count = max(1, text.count(".") + text.count("!") + text.count("?"))
    avg_word_length = np.mean([len(w) for w in words]) if words else 0.0
    unique_words = set(w.lower() for w in words)
    ttr = len(unique_words) / word_count if word_count > 0 else 0.0
    punct_chars = sum(1 for c in text if c in ".,;:!?\"'()-")
    punct_density = punct_chars / len(text) if len(text) > 0 else 0.0

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": avg_word_length,
        "ttr": ttr,
        "punct_density": punct_density,
    }


def build_feature_vector(df, tier="T0"):
    """
    Build a numeric feature matrix from a multitier DataFrame at a given tier.

    Returns (X, feature_names) where X is (n_samples, n_features) numpy array.

    Features (23 total):
        - POS proportions: 17 UPOS tags
        - dep_depth: mean dependency tree depth (1)
        - entity_count: number of named entities (1)
        - word_count, sentence_count, avg_word_length, ttr, punct_density (5, from text)
    """
    n = len(df)

    # POS vectors (17-dim)
    pos_col = f"pos_{tier}"
    if pos_col in df.columns:
        pos_matrix = np.stack(df[pos_col].values)
    else:
        pos_matrix = np.zeros((n, 17))

    # Dependency depth (1-dim)
    dep_col = f"dep_depth_{tier}"
    dep_depth = df[dep_col].values.reshape(-1, 1) if dep_col in df.columns else np.zeros((n, 1))

    # NER entity count (1-dim)
    ner_col = f"ner_{tier}"
    if ner_col in df.columns:
        entity_counts = np.array([len(s) if isinstance(s, (set, frozenset)) else 0
                                  for s in df[ner_col]]).reshape(-1, 1)
    else:
        entity_counts = np.zeros((n, 1))

    # Text statistics (5-dim) — computed from raw text
    text_col = f"text_{tier}"
    if text_col in df.columns:
        stats = df[text_col].apply(compute_text_stats).tolist()
        stats_df = pd.DataFrame(stats)
        text_features = stats_df[["word_count", "sentence_count", "avg_word_length",
                                   "ttr", "punct_density"]].values
    else:
        text_features = np.zeros((n, 5))

    # Concatenate all features
    X = np.hstack([pos_matrix, dep_depth, entity_counts, text_features])

    feature_names = [
        "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
        "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
        "SCONJ", "SYM", "VERB", "X",
        "dep_depth", "entity_count",
        "word_count", "sentence_count", "avg_word_length", "ttr", "punct_density",
    ]

    # Log NaN diagnostics
    nan_mask = np.isnan(X).any(axis=1)
    if nan_mask.any():
        nan_cols = np.where(np.isnan(X).any(axis=0))[0]
        col_names = [feature_names[i] for i in nan_cols]
        logger.warning(
            "build_feature_vector(%s): %d/%d rows have NaN in columns: %s",
            tier, nan_mask.sum(), len(X), col_names,
        )

    return X, feature_names


def build_delta_features_at_tier(df, tier="T1"):
    """
    Build T0->Tn delta feature vectors for paraphraser identification (RQ3).

    Delta captures what the paraphraser CHANGED, not what the original looked like.
    Supports any tier (T1, T2, T3) to test fingerprint persistence across depths.

    Returns (X_delta, feature_names) where X_delta is (n_samples, n_features).

    Features (24 total):
        - POS delta: Tn_pos - T0_pos (17-dim)
        - dep_depth delta: Tn - T0 (1-dim)
        - NER Jaccard, Recall, Precision (3-dim, already computed)
        - word_count ratio: Tn/T0 (1-dim)
        - ttr delta: Tn - T0 (1-dim)
        - punct_density delta: Tn - T0 (1-dim)
    """
    n = len(df)

    # POS delta (17-dim)
    pos_col_t0 = "pos_T0"
    pos_col_tn = f"pos_{tier}"
    if pos_col_t0 in df.columns and pos_col_tn in df.columns:
        pos_t0 = np.stack(df[pos_col_t0].values)
        pos_tn = np.stack(df[pos_col_tn].values)
        pos_delta = pos_tn - pos_t0
    else:
        pos_delta = np.zeros((n, 17))

    # Dep depth delta (1-dim)
    dep_col_t0 = "dep_depth_T0"
    dep_col_tn = f"dep_depth_{tier}"
    if dep_col_t0 in df.columns and dep_col_tn in df.columns:
        dep_delta = (df[dep_col_tn] - df[dep_col_t0]).values.reshape(-1, 1)
    else:
        dep_delta = np.zeros((n, 1))

    # NER metrics (3-dim) — already deltas by nature
    ner_features = np.zeros((n, 3))
    for i, suffix in enumerate(["jaccard", "recall", "precision"]):
        col = f"ner_{suffix}_{tier}"
        if col in df.columns:
            ner_features[:, i] = df[col].fillna(0.0).values

    # Text stat deltas (3-dim)
    text_deltas = np.zeros((n, 3))
    text_col_t0 = "text_T0"
    text_col_tn = f"text_{tier}"
    if text_col_t0 in df.columns and text_col_tn in df.columns:
        stats_t0 = pd.DataFrame(df[text_col_t0].apply(compute_text_stats).tolist())
        stats_tn = pd.DataFrame(df[text_col_tn].apply(compute_text_stats).tolist())

        # Word count ratio (Tn/T0)
        wc_t0 = stats_t0["word_count"].replace(0, 1).values
        text_deltas[:, 0] = stats_tn["word_count"].values / wc_t0

        # TTR delta
        text_deltas[:, 1] = stats_tn["ttr"].values - stats_t0["ttr"].values

        # Punct density delta
        text_deltas[:, 2] = stats_tn["punct_density"].values - stats_t0["punct_density"].values

    X_delta = np.hstack([pos_delta, dep_delta, ner_features, text_deltas])

    feature_names = [
        "d_ADJ", "d_ADP", "d_ADV", "d_AUX", "d_CCONJ", "d_DET", "d_INTJ",
        "d_NOUN", "d_NUM", "d_PART", "d_PRON", "d_PROPN", "d_PUNCT",
        "d_SCONJ", "d_SYM", "d_VERB", "d_X",
        "d_dep_depth",
        "ner_jaccard", "ner_recall", "ner_precision",
        "wc_ratio", "d_ttr", "d_punct_density",
    ]

    # Log NaN diagnostics
    nan_mask = np.isnan(X_delta).any(axis=1)
    if nan_mask.any():
        nan_cols = np.where(np.isnan(X_delta).any(axis=0))[0]
        col_names = [feature_names[i] for i in nan_cols]
        logger.warning(
            "build_delta_features_at_tier(%s): %d/%d rows have NaN in columns: %s",
            tier, nan_mask.sum(), len(X_delta), col_names,
        )

    return X_delta, feature_names


def build_delta_features(df):
    """Backward-compatible wrapper: builds T0->T1 delta features."""
    return build_delta_features_at_tier(df, tier="T1")
