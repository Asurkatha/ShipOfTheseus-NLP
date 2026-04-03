"""
Authorship attribution and paraphraser identification classifiers.

Supports:
- RQ2: Train on T0 features, evaluate decay across T1-T3
- RQ3: Multi-class paraphraser identification from delta features
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.inspection import permutation_importance

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupShuffleSplit, cross_val_score

from src.utils.config import RANDOM_SEED, TEST_SIZE


CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced"
    ),
    "SVM (RBF)": SVC(
        kernel="rbf", random_state=RANDOM_SEED, class_weight="balanced",
        probability=True
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_SEED, class_weight="balanced",
        n_jobs=-1
    ),
}

BASELINE_CLASSIFIERS = {
    "Random (Uniform)": DummyClassifier(strategy="uniform", random_state=RANDOM_SEED),
    "Most Frequent": DummyClassifier(strategy="most_frequent"),
    "Stratified": DummyClassifier(strategy="stratified", random_state=RANDOM_SEED),
}


def train_and_evaluate_attribution(X_train, y_train, X_by_tier, y_by_tier,
                                    classifier_name="Logistic Regression"):
    """
    Train an attribution classifier on T0 features and evaluate on T1-T3.

    Args:
        X_train: Feature matrix from T0 (n_samples, n_features)
        y_train: Labels from T0 (source author)
        X_by_tier: dict {"T1": X_T1, "T2": X_T2, "T3": X_T3}
        y_by_tier: dict {"T1": y_T1, "T2": y_T2, "T3": y_T3} — same labels as T0

    Returns:
        dict with training results and per-tier evaluation
    """
    clf = CLASSIFIERS[classifier_name]
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    clf.fit(X_train_scaled, y_train)

    # Evaluate on training data (T0)
    y_pred_train = clf.predict(X_train_scaled)
    results = {
        "T0": {
            "accuracy": accuracy_score(y_train, y_pred_train),
            "macro_f1": f1_score(y_train, y_pred_train, average="macro"),
        }
    }

    # Evaluate on each paraphrased tier
    for tier, X_tier in X_by_tier.items():
        if X_tier is None or len(X_tier) == 0:
            continue
        X_tier_scaled = scaler.transform(X_tier)
        y_tier = y_by_tier[tier]
        y_pred = clf.predict(X_tier_scaled)
        results[tier] = {
            "accuracy": accuracy_score(y_tier, y_pred),
            "macro_f1": f1_score(y_tier, y_pred, average="macro"),
        }

    return results, clf, scaler


def run_attribution_with_cv(X_train, y_train, X_by_tier, y_by_tier,
                             classifier_name="Logistic Regression", n_folds=5):
    """
    Train an attribution classifier with k-fold cross-validation on T0,
    then evaluate decay across T1-T3.

    Returns:
        dict with 'cv_scores' (per-fold F1 on T0), 'cv_mean', 'cv_std',
        and per-tier evaluation from the final model trained on full T0.
    """
    from sklearn.base import clone

    clf_template = CLASSIFIERS[classifier_name]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Cross-validation on T0
    cv_f1_scores = []
    for train_idx, val_idx in skf.split(X_scaled, y_train):
        clf_fold = clone(clf_template)
        clf_fold.fit(X_scaled[train_idx], y_train[train_idx])
        y_pred = clf_fold.predict(X_scaled[val_idx])
        cv_f1_scores.append(f1_score(y_train[val_idx], y_pred, average="macro"))

    # Train final model on full T0
    clf = clone(clf_template)
    clf.fit(X_scaled, y_train)

    results = {
        "cv_scores": cv_f1_scores,
        "cv_mean": float(np.mean(cv_f1_scores)),
        "cv_std": float(np.std(cv_f1_scores)),
        "T0": {
            "accuracy": accuracy_score(y_train, clf.predict(X_scaled)),
            "macro_f1": f1_score(y_train, clf.predict(X_scaled), average="macro"),
        },
    }

    # Evaluate on each paraphrased tier
    for tier, X_tier in X_by_tier.items():
        if X_tier is None or len(X_tier) == 0:
            continue
        X_tier_scaled = scaler.transform(X_tier)
        y_tier = y_by_tier[tier]
        y_pred = clf.predict(X_tier_scaled)
        results[tier] = {
            "accuracy": accuracy_score(y_tier, y_pred),
            "macro_f1": f1_score(y_tier, y_pred, average="macro"),
        }

    return results, clf, scaler


def run_attribution_experiment(df, feature_builder_fn, label_col="source",
                                binary=False):
    """
    Full RQ2 experiment: train on T0, evaluate decay across T1-T3.

    Args:
        df: DataFrame from multitier_results.pkl (single paraphraser)
        feature_builder_fn: callable(df, tier) -> (X, feature_names)
        label_col: column to predict (default "source")
        binary: if True, collapse labels to Human vs LLM

    Returns:
        dict of {classifier_name: {tier: {accuracy, macro_f1}}}
    """
    # Build labels
    y = df[label_col].values
    if binary:
        y = np.array(["Human" if s == "Human" else "LLM" for s in y])

    # Build features at each tier
    X_T0, feature_names = feature_builder_fn(df, "T0")

    tiers = {}
    for tier in ["T1", "T2", "T3"]:
        text_col = f"text_{tier}"
        if text_col not in df.columns:
            continue
        # Only use rows where this tier has data
        mask = df[text_col].notna()
        if mask.sum() < 10:
            continue
        X_tier, _ = feature_builder_fn(df[mask], tier)
        tiers[tier] = (X_tier, y[mask])

    # Handle NaN in features
    nan_mask = np.isnan(X_T0).any(axis=1)
    if nan_mask.any():
        valid = ~nan_mask
        X_T0 = X_T0[valid]
        y = y[valid]
        # Also filter tier data
        for tier in list(tiers.keys()):
            X_t, y_t = tiers[tier]
            valid_t = ~np.isnan(X_t).any(axis=1)
            tiers[tier] = (X_t[valid_t], y_t[valid_t])

    # Run each classifier
    all_results = {}
    for clf_name in CLASSIFIERS:
        X_by_tier = {t: data[0] for t, data in tiers.items()}
        y_by_tier = {t: data[1] for t, data in tiers.items()}

        results, clf, scaler = train_and_evaluate_attribution(
            X_T0, y, X_by_tier, y_by_tier, clf_name
        )
        all_results[clf_name] = results

    return all_results, feature_names


def run_paraphraser_identification(paraphraser_dfs, feature_builder_fn,
                                    exclude_paraphrasers=None):
    """
    Full RQ3 experiment: multi-class paraphraser identification from delta features.

    Args:
        paraphraser_dfs: dict of {para_key: DataFrame}
        feature_builder_fn: callable(df) -> (X_delta, feature_names)
        exclude_paraphrasers: list of para_keys to exclude (e.g., ["palm"])

    Returns:
        dict with results, confusion matrix, feature importances
    """
    exclude = set(exclude_paraphrasers or [])

    # Find common (key, source) pairs across included paraphrasers
    key_sets = {}
    for para_key, df in paraphraser_dfs.items():
        if para_key in exclude:
            continue
        key_sets[para_key] = set(zip(df["key"], df["source"]))

    common_keys = None
    for ks in key_sets.values():
        common_keys = ks if common_keys is None else common_keys & ks

    if common_keys is None or len(common_keys) < 100:
        raise ValueError(f"Too few common articles: {len(common_keys) if common_keys else 0}")

    common_keys = sorted(common_keys)

    # Build features for each paraphraser on common articles
    all_X = []
    all_y = []
    all_groups = []  # Track (key, source) for group-aware splitting
    feature_names = None

    for para_key, df in paraphraser_dfs.items():
        if para_key in exclude:
            continue
        # Filter to common keys
        mask = list(zip(df["key"], df["source"]))
        keep = [i for i, k in enumerate(mask) if k in set(common_keys)]
        df_common = df.iloc[keep].reset_index(drop=True)

        X_delta, fnames = feature_builder_fn(df_common)
        feature_names = fnames

        all_X.append(X_delta)
        all_y.extend([para_key] * len(X_delta))
        # Group by document identity to prevent leakage across paraphrasers
        all_groups.extend(list(zip(df_common["key"], df_common["source"])))

    X = np.vstack(all_X)
    y = np.array(all_y)
    groups = np.array([f"{k}_{s}" for k, s in all_groups])

    # Handle NaN
    nan_mask = np.isnan(X).any(axis=1)
    if nan_mask.any():
        print(f"  Dropping {nan_mask.sum()} NaN rows from paraphraser ID features")
        X = X[~nan_mask]
        y = y[~nan_mask]
        groups = groups[~nan_mask]

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Group-aware split: same document never in both train and test
    n_unique_groups = len(set(groups))
    if n_unique_groups >= 20:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
        train_idx, test_idx = next(gss.split(X, y_encoded, groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    else:
        print(f"  WARNING: Only {n_unique_groups} unique groups, falling back to stratified split")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_SEED,
            stratify=y_encoded
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and evaluate each classifier
    results = {}
    for clf_name, clf in CLASSIFIERS.items():
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)

        results[clf_name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
            "report": classification_report(
                y_test, y_pred, target_names=le.classes_, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "labels": le.classes_,
        }

        # Feature importance (Random Forest only)
        if clf_name == "Random Forest":
            results[clf_name]["feature_importance"] = dict(
                zip(feature_names, clf.feature_importances_)
            )

    # Evaluate baseline classifiers for context
    for clf_name, clf in BASELINE_CLASSIFIERS.items():
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        results[clf_name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
            "is_baseline": True,
        }

    return results, feature_names, le.classes_
