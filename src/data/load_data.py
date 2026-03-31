"""
Data loading for the Ship of Theseus corpus.

Corpus structure:
    Ship_of_theseus_paraphrased_copus/
        paraphrased_datasets/   <- 7 CSVs, cols: source, key, text, version_name
        train_datasets/         <- 7 CSVs, cols: source, key, text

7 datasets, 7 paraphraser variants, 7 source authors, 22 version strings.
"""

import shutil
import pandas as pd
from src.utils.config import (
    DATA_RAW, DATA_CLONED, DATA_PARAPHRASED, DATA_TRAIN, DATA_PROCESSED,
    ALL_DATASETS, PARAPHRASERS, MAX_SAMPLES_PER_SUBSET,
    VERSION_TO_COL, parse_version, version_to_tier, version_to_paraphraser,
)


def load_paraphrased(dataset_name):
    """Load a single paraphrased CSV."""
    fpath = DATA_PARAPHRASED / f"{dataset_name}_paraphrased.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"Not found: {fpath}")
    df = pd.read_csv(fpath)
    df["dataset"] = dataset_name
    df["split"] = "test"
    print(f"  {dataset_name}_paraphrased.csv: {len(df)} rows")
    return df


def load_train(dataset_name):
    """Load a single train (unparaphrased) CSV."""
    fpath = DATA_TRAIN / f"{dataset_name}_train.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"Not found: {fpath}")
    df = pd.read_csv(fpath)
    df["dataset"] = dataset_name
    df["split"] = "train"
    df["version_name"] = "train"
    print(f"  {dataset_name}_train.csv: {len(df)} rows")
    return df


def add_tier_labels(df):
    """Add tier (T0-T3), paraphraser key, and display name columns."""
    if "version_name" not in df.columns:
        df["tier"] = "unknown"
        df["paraphraser"] = "unknown"
        df["paraphraser_name"] = "unknown"
        return df

    df["tier"] = df["version_name"].apply(
        lambda v: "train" if v == "train" else version_to_tier(v)
    )
    df["paraphraser"] = df["version_name"].apply(
        lambda v: "none" if v == "train" else version_to_paraphraser(v)
    )

    name_map = {k: v["name"] for k, v in PARAPHRASERS.items()}
    name_map.update({"original": "original", "none": "none"})
    df["paraphraser_name"] = df["paraphraser"].map(name_map).fillna(df["paraphraser"])

    return df


def load_corpus(datasets=None, include_train=False, max_samples=MAX_SAMPLES_PER_SUBSET):
    """
    Load the full corpus with tier labels.
    If datasets is None, loads all 7.
    """
    datasets = datasets or ALL_DATASETS
    all_dfs = []

    for ds in datasets:
        print(f"\n{'='*50}\nLoading: {ds}\n{'='*50}")
        try:
            all_dfs.append(load_paraphrased(ds))
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")

        if include_train:
            try:
                all_dfs.append(load_train(ds))
            except FileNotFoundError as e:
                print(f"  SKIP: {e}")

    if not all_dfs:
        raise RuntimeError(f"No datasets loaded. Check {DATA_CLONED}")

    corpus = pd.concat(all_dfs, ignore_index=True)
    corpus = add_tier_labels(corpus)

    if max_samples:
        before = len(corpus)
        corpus = corpus.groupby("dataset", group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), max_samples), random_state=42)
        ).reset_index(drop=True)
        if len(corpus) < before:
            print(f"\n  Sampled {before} -> {len(corpus)} rows")

    print(f"\n{'='*50}\nCORPUS SUMMARY\n{'='*50}")
    print(f"Total: {len(corpus)} rows")
    for col in ["dataset", "tier", "paraphraser_name", "source"]:
        print(f"\n{col}:")
        print(corpus[col].value_counts().to_string())

    return corpus


def get_paired_texts(corpus, paraphraser_key, dataset=None, source=None):
    """
    Align T0-T3 texts for a specific paraphraser chain.
    Pivots by (key, source) so we can compare the same document across tiers.

    Note: 'source' = who wrote it (Human, OpenAI, etc.),
          'paraphraser' = what rewrote it (chatgpt, dipper, etc.)
    """
    df = corpus.copy()
    if dataset:
        df = df[df["dataset"] == dataset]
    if source:
        df = df[df["source"] == source]

    mask = (df["paraphraser"] == paraphraser_key) | (df["tier"] == "T0")
    subset = df[mask].copy()

    if len(subset) == 0:
        print(f"  No data for paraphraser '{paraphraser_key}'")
        return pd.DataFrame()

    pivoted = subset.pivot_table(
        index=["key", "source"],
        columns="tier",
        values="text",
        aggfunc="first"
    ).reset_index()

    if "T0" in pivoted.columns:
        pivoted = pivoted.dropna(subset=["T0"])

    tiers = [c for c in ["T0", "T1", "T2", "T3"] if c in pivoted.columns]
    print(f"  Paired {paraphraser_key}: {len(pivoted)} docs, tiers: {tiers}")
    return pivoted


def get_all_paired_texts(corpus, dataset=None, source=None):
    """Get paired texts for every paraphraser. Returns dict."""
    pairs = {}
    for pkey in PARAPHRASERS:
        paired = get_paired_texts(corpus, pkey, dataset, source)
        if len(paired) > 0:
            pairs[pkey] = paired
    return pairs


def load_paired_t1(datasets=None, cache=True):
    """
    Load the forensic-audit pivot: one row per (key, source, dataset)
    with T0 and five T1 paraphraser columns.

    Normalises the 'orignal' typo, filters to target version strings,
    pivots, and drops rows missing required columns. Caches to pickle.
    """
    cache_path = DATA_PROCESSED / "paired_all.pkl"
    if cache and cache_path.exists():
        paired = pd.read_pickle(cache_path)
        print(f"Loaded cached pivot: {paired.shape} from {cache_path}")
        return paired

    datasets = datasets or ALL_DATASETS
    text_cols = list(VERSION_TO_COL.values())

    # Load & concatenate
    dfs = []
    for ds in datasets:
        fpath = DATA_PARAPHRASED / f"{ds}_paraphrased.csv"
        if not fpath.exists():
            print(f"  SKIP: {fpath}")
            continue
        df = pd.read_csv(fpath)
        df["dataset"] = ds
        dfs.append(df)
        print(f"  {ds}: {len(df):>6,} rows, {df['version_name'].nunique()} versions")

    if not dfs:
        raise RuntimeError(f"No datasets loaded. Check {DATA_PARAPHRASED}")
    corpus = pd.concat(dfs, ignore_index=True)

    # Normalise known typo
    typo_count = (corpus["version_name"] == "orignal").sum()
    corpus["version_name"] = corpus["version_name"].replace("orignal", "original")
    print(f"\nTotal corpus: {len(corpus):,} rows  ('orignal' typos fixed: {typo_count})")

    # Verify parser against all unique version strings
    print(f"\n{'version_name':<55} {'base_token':<20} {'tier'}")
    print("-" * 80)
    for v in sorted(corpus["version_name"].unique()):
        base, n = parse_version(v)
        tier = "T0" if base == "original" else f"T{n}"
        print(f"{v:<55} {base:<20} {tier}")

    # Filter & pivot
    subset = corpus[corpus["version_name"].isin(VERSION_TO_COL)].copy()
    subset["col_name"] = subset["version_name"].map(VERSION_TO_COL)

    paired = subset.pivot_table(
        index=["key", "source", "dataset"],
        columns="col_name",
        values="text",
        aggfunc="first",
    )
    paired.columns.name = None
    paired = paired.reset_index()

    print(f"\nShape before dropna: {paired.shape}")
    print(f"NaN counts:\n{paired[text_cols].isna().sum().to_string()}")

    required = ["text_T0", "text_chatgpt", "text_dipper_high", "text_pegasus_slight",
                 "text_palm", "text_dipper"]
    paired = paired.dropna(subset=required)
    print(f"Shape after dropna:  {paired.shape}")

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        paired.to_pickle(cache_path)
        print(f"Cached to {cache_path} ({cache_path.stat().st_size / 1e6:.1f} MB)")

    return paired


def organize_into_subdirs(dry_run=True):
    """Copy corpus files into data/raw/<dataset>/ for cleaner structure."""
    print(f"{'[DRY RUN] ' if dry_run else ''}Organizing...")

    for ds in ALL_DATASETS:
        target_dir = DATA_RAW / ds
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)

        for src_dir, suffix in [(DATA_PARAPHRASED, "paraphrased"),
                                (DATA_TRAIN, "train")]:
            src_file = src_dir / f"{ds}_{suffix}.csv"
            dest_file = target_dir / src_file.name
            if not src_file.exists():
                continue
            if dry_run:
                print(f"  {src_file.name} -> {dest_file}")
            elif not dest_file.exists():
                shutil.copy2(src_file, dest_file)
                print(f"  Copied: {dest_file}")

    if dry_run:
        print("\nRun with dry_run=False to copy.")


if __name__ == "__main__":
    print("=" * 60)
    print("CORPUS DIAGNOSTICS")
    print("=" * 60)

    if not DATA_CLONED.exists():
        print(f"Not found: {DATA_CLONED}")
        print(f"cd {DATA_RAW}")
        print("git clone https://github.com/tripto03/Ship_of_theseus_paraphrased_copus.git")
        exit(1)

    print(f"\nParaphrased: {DATA_PARAPHRASED} (exists: {DATA_PARAPHRASED.exists()})")
    print(f"Train: {DATA_TRAIN} (exists: {DATA_TRAIN.exists()})")

    print(f"\n{'='*60}\nDATASET OVERVIEW\n{'='*60}")
    for ds in ALL_DATASETS:
        para = DATA_PARAPHRASED / f"{ds}_paraphrased.csv"
        train = DATA_TRAIN / f"{ds}_train.csv"
        print(f"\n  [{ds}]")
        if para.exists():
            df = pd.read_csv(para)
            print(f"    Paraphrased: {df.shape[0]} rows, {df['version_name'].nunique()} versions")
        if train.exists():
            df_t = pd.read_csv(train)
            print(f"    Train: {df_t.shape[0]} rows")

    # Version mapping table
    print(f"\n{'='*60}\nVERSION -> TIER MAPPING\n{'='*60}")
    sample = pd.read_csv(DATA_PARAPHRASED / f"{ALL_DATASETS[0]}_paraphrased.csv")
    for v in sorted(sample["version_name"].unique()):
        print(f"  {v:55s} -> {version_to_tier(v)}  ({version_to_paraphraser(v)})")