"""
Data loading utilities for the Ship of Theseus corpus.

Actual corpus structure (discovered 2026-02-21):
    Ship_of_theseus_paraphrased_copus/
        paraphrased_datasets/
            cmv_paraphrased.csv      (cols: source, key, text, version_name)
            eli5_paraphrased.csv
            sci_gen_paraphrased.csv
            tldr_paraphrased.csv
            wp_paraphrased.csv
            xsum_paraphrased.csv
            yelp_paraphrased.csv
        train_datasets/
            cmv_train.csv            (cols: source, key, text)
            eli5_train.csv
            ...

7 datasets, 7 paraphraser variants, 7 source authors, 22 version strings.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
from src.utils.config import (
    DATA_RAW, DATA_CLONED, DATA_PARAPHRASED, DATA_TRAIN,
    ALL_DATASETS, PARAPHRASERS, MAX_SAMPLES_PER_SUBSET,
    version_to_tier, version_to_paraphraser,
)


# ─────────────────────────────────────────────
# Core loading functions
# ─────────────────────────────────────────────

def load_paraphrased(dataset_name: str) -> pd.DataFrame:
    """
    Load a single paraphrased dataset (e.g., 'xsum').
    File: paraphrased_datasets/<name>_paraphrased.csv
    Columns: source, key, text, version_name
    """
    fpath = DATA_PARAPHRASED / f"{dataset_name}_paraphrased.csv"
    if not fpath.exists():
        raise FileNotFoundError(
            f"Paraphrased file not found: {fpath}\n"
            f"Available datasets: {ALL_DATASETS}"
        )
    df = pd.read_csv(fpath)
    df["dataset"] = dataset_name
    df["split"] = "test"
    print(f"  Loaded {dataset_name}_paraphrased.csv: {len(df)} rows")
    return df


def load_train(dataset_name: str) -> pd.DataFrame:
    """
    Load a single train dataset (unparaphrased).
    File: train_datasets/<name>_train.csv
    Columns: source, key, text
    """
    fpath = DATA_TRAIN / f"{dataset_name}_train.csv"
    if not fpath.exists():
        raise FileNotFoundError(
            f"Train file not found: {fpath}\n"
            f"Available datasets: {ALL_DATASETS}"
        )
    df = pd.read_csv(fpath)
    df["dataset"] = dataset_name
    df["split"] = "train"
    df["version_name"] = "train"  # Mark as train, not paraphrased
    print(f"  Loaded {dataset_name}_train.csv: {len(df)} rows")
    return df


def add_tier_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'tier' (T0, T1, T2, T3) and 'paraphraser' columns
    from the version_name column.
    """
    if "version_name" not in df.columns:
        print("  Warning: no 'version_name' column found.")
        df["tier"] = "unknown"
        df["paraphraser"] = "unknown"
        return df

    df["tier"] = df["version_name"].apply(
        lambda v: "train" if v == "train" else version_to_tier(v)
    )
    df["paraphraser"] = df["version_name"].apply(
        lambda v: "none" if v == "train" else version_to_paraphraser(v)
    )

    # Map paraphraser keys to display names
    name_map = {k: v["name"] for k, v in PARAPHRASERS.items()}
    name_map["original"] = "original"
    name_map["none"] = "none"
    df["paraphraser_name"] = df["paraphraser"].map(name_map).fillna(df["paraphraser"])

    return df


# ─────────────────────────────────────────────
# Full corpus loading
# ─────────────────────────────────────────────

def load_corpus(datasets: Optional[list] = None,
                include_train: bool = False,
                max_samples: Optional[int] = MAX_SAMPLES_PER_SUBSET
                ) -> pd.DataFrame:
    """
    Load the paraphrased corpus with tier labels.

    Args:
        datasets: List of dataset names. If None, loads all 7.
        include_train: Whether to also load train (unparaphrased) data.
        max_samples: Cap samples per dataset for faster iteration.

    Returns:
        DataFrame with columns:
        [source, key, text, version_name, dataset, split, tier,
         paraphraser, paraphraser_name]
    """
    datasets = datasets or ALL_DATASETS
    all_dfs = []

    for ds in datasets:
        print(f"\n{'='*50}")
        print(f"Loading: {ds}")
        print(f"{'='*50}")

        try:
            df = load_paraphrased(ds)
            all_dfs.append(df)
        except FileNotFoundError as e:
            print(f"  SKIP paraphrased: {e}")

        if include_train:
            try:
                df_train = load_train(ds)
                all_dfs.append(df_train)
            except FileNotFoundError as e:
                print(f"  SKIP train: {e}")

    if not all_dfs:
        raise RuntimeError(
            "No datasets loaded. Ensure the corpus is cloned at:\n"
            f"  {DATA_CLONED}\n"
            "Expected structure:\n"
            f"  {DATA_PARAPHRASED}/<name>_paraphrased.csv\n"
            f"  {DATA_TRAIN}/<name>_train.csv"
        )

    corpus = pd.concat(all_dfs, ignore_index=True)
    corpus = add_tier_labels(corpus)

    if max_samples:
        before = len(corpus)
        corpus = corpus.groupby("dataset", group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), max_samples), random_state=42)
        ).reset_index(drop=True)
        if len(corpus) < before:
            print(f"\n  Sampled from {before} to {len(corpus)} rows")

    print(f"\n{'='*50}")
    print(f"CORPUS SUMMARY")
    print(f"{'='*50}")
    print(f"Total rows: {len(corpus)}")
    print(f"\nDatasets ({len(datasets)}):")
    print(corpus["dataset"].value_counts().to_string())
    print(f"\nTiers:")
    print(corpus["tier"].value_counts().sort_index().to_string())
    print(f"\nParaphrasers:")
    print(corpus["paraphraser_name"].value_counts().to_string())
    print(f"\nSources:")
    print(corpus["source"].value_counts().to_string())

    return corpus


# ─────────────────────────────────────────────
# Paired text extraction (for similarity analysis)
# ─────────────────────────────────────────────

def get_paired_texts(corpus: pd.DataFrame, paraphraser_key: str,
                     dataset: str = None,
                     source: str = None) -> pd.DataFrame:
    """
    Get T0-T1-T2-T3 aligned text pairs for a specific paraphraser.
    Aligns by 'key' + 'source' so we track the same document across
    iterations.

    IMPORTANT (learned from Assignment 2):
        - 'source' = who PRODUCED the text (Human, OpenAI, LLAMA, etc.)
        - 'version_name' = which PARAPHRASING CHAIN was applied
        These are independent axes. Each (key, source) pair has its own
        T0 original and T1-T3 paraphrased versions.

    Args:
        corpus: Full corpus DataFrame (with tier labels)
        paraphraser_key: Paraphraser config key (e.g., 'chatgpt',
                         'dipper(low)', 'pegasus(full)')
        dataset: Optional filter for a single dataset
        source: Optional filter for a source author (e.g., 'Human',
                'OpenAI'). If None, includes all sources.

    Returns:
        DataFrame with columns [key, source, T0, T1, T2, T3]
    """
    df = corpus.copy()

    if dataset:
        df = df[df["dataset"] == dataset]
    if source:
        df = df[df["source"] == source]

    # Keep only original (T0) and this paraphraser's data
    mask = (df["paraphraser"] == paraphraser_key) | (df["tier"] == "T0")
    subset = df[mask].copy()

    if len(subset) == 0:
        print(f"  No data for paraphraser '{paraphraser_key}'")
        return pd.DataFrame()

    # Pivot: each tier becomes a column
    pivoted = subset.pivot_table(
        index=["key", "source"],
        columns="tier",
        values="text",
        aggfunc="first"
    ).reset_index()

    # Keep only rows that have T0
    if "T0" in pivoted.columns:
        pivoted = pivoted.dropna(subset=["T0"])

    available_tiers = [c for c in ["T0", "T1", "T2", "T3"] if c in pivoted.columns]
    print(f"  Paired texts for {paraphraser_key}: "
          f"{len(pivoted)} docs, tiers: {available_tiers}")
    return pivoted


def get_all_paired_texts(corpus: pd.DataFrame,
                         dataset: str = None,
                         source: str = None) -> dict:
    """
    Get paired texts for ALL paraphrasers.

    Returns:
        dict: {paraphraser_key: paired_DataFrame}
    """
    pairs = {}
    for pkey in PARAPHRASERS:
        paired = get_paired_texts(corpus, pkey, dataset, source)
        if len(paired) > 0:
            pairs[pkey] = paired
    return pairs


# ─────────────────────────────────────────────
# Organize into per-dataset subdirs (optional)
# ─────────────────────────────────────────────

def organize_into_subdirs(dry_run: bool = True):
    """
    Copy corpus files into per-dataset subdirectories under data/raw/:
        data/raw/xsum/xsum_paraphrased.csv
        data/raw/xsum/xsum_train.csv
        etc.
    """
    import shutil

    print(f"{'[DRY RUN] ' if dry_run else ''}Organizing into subdirs...")

    for ds in ALL_DATASETS:
        target_dir = DATA_RAW / ds
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)

        for src_dir, suffix in [(DATA_PARAPHRASED, "paraphrased"),
                                (DATA_TRAIN, "train")]:
            src_file = src_dir / f"{ds}_{suffix}.csv"
            dest_file = target_dir / src_file.name

            if not src_file.exists():
                print(f"  NOT FOUND: {src_file.name}")
                continue

            if dry_run:
                print(f"  COPY {src_file.name} -> {dest_file}")
            else:
                if not dest_file.exists():
                    shutil.copy2(src_file, dest_file)
                    print(f"  Copied: {dest_file}")
                else:
                    print(f"  Exists: {dest_file}")

    if dry_run:
        print("\nRun with dry_run=False to actually copy files.")


# ─────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("CORPUS DIAGNOSTICS")
    print("=" * 60)

    if not DATA_CLONED.exists():
        print(f"Cloned repo not found at: {DATA_CLONED}")
        print(f"Run:\n  cd {DATA_RAW}")
        print("  git clone https://github.com/tripto03/"
              "Ship_of_theseus_paraphrased_copus.git")
        exit(1)

    print(f"\nParaphrased dir: {DATA_PARAPHRASED}")
    print(f"  Exists: {DATA_PARAPHRASED.exists()}")
    print(f"Train dir: {DATA_TRAIN}")
    print(f"  Exists: {DATA_TRAIN.exists()}")

    print(f"\n{'='*60}")
    print("DATASET OVERVIEW")
    print("=" * 60)

    for ds in ALL_DATASETS:
        para_file = DATA_PARAPHRASED / f"{ds}_paraphrased.csv"
        train_file = DATA_TRAIN / f"{ds}_train.csv"

        print(f"\n  [{ds}]")
        if para_file.exists():
            df = pd.read_csv(para_file)
            print(f"    Paraphrased: {df.shape[0]} rows, "
                  f"{df['version_name'].nunique()} versions")
        else:
            print(f"    Paraphrased: NOT FOUND")

        if train_file.exists():
            df_t = pd.read_csv(train_file)
            print(f"    Train: {df_t.shape[0]} rows")
        else:
            print(f"    Train: NOT FOUND")

    # Show version mapping
    print(f"\n{'='*60}")
    print("VERSION -> TIER MAPPING")
    print("=" * 60)
    sample = pd.read_csv(DATA_PARAPHRASED / f"{ALL_DATASETS[0]}_paraphrased.csv")
    versions = sorted(sample["version_name"].unique())
    for v in versions:
        tier = version_to_tier(v)
        para = version_to_paraphraser(v)
        print(f"  {v:55s} -> {tier}  ({para})")