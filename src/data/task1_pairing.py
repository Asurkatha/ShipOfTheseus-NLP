"""
Task 1: pair the original text with three specific T1 paraphrases.

Expected raw layout:
    data/raw/<dataset>/<dataset>_paraphrased.csv

Output:
    data/processed/paired_all.pkl
"""

from pathlib import Path

import pandas as pd

from src.utils.config import DATA_PROCESSED, DATA_RAW


TASK1_TARGET_VERSIONS = {
    "original": "text_T0",
    "orignal": "text_T0",
    "chatgpt": "text_chatgpt",
    "dipper(high)": "text_dipper_high",
    "pegasus(slight)": "text_pegasus_slight",
}

TASK1_OUTPUT_COLUMNS = [
    "dataset",
    "source",
    "key",
    "text_T0",
    "text_chatgpt",
    "text_dipper_high",
    "text_pegasus_slight",
]


def discover_datasets(raw_root=DATA_RAW):
    """Find datasets from data/raw/<dataset>/<dataset>_paraphrased.csv."""
    datasets = []
    for dataset_dir in sorted(Path(raw_root).iterdir()):
        if not dataset_dir.is_dir():
            continue
        expected = dataset_dir / f"{dataset_dir.name}_paraphrased.csv"
        if expected.exists():
            datasets.append(dataset_dir.name)
    return datasets


def load_task1_corpus(raw_root=DATA_RAW, datasets=None):
    """Load and concatenate all dataset paraphrased CSVs for Task 1."""
    raw_root = Path(raw_root)
    datasets = datasets or discover_datasets(raw_root)
    all_dfs = []

    if not datasets:
        raise RuntimeError(f"No paraphrased datasets found under {raw_root}")

    for dataset in datasets:
        path = raw_root / dataset / f"{dataset}_paraphrased.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing paraphrased CSV: {path}")
        df = pd.read_csv(path)
        df["dataset"] = dataset
        all_dfs.append(df)
        print(f"Loaded {dataset}: {len(df):,} rows")

    corpus = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined corpus: {corpus.shape}")
    return corpus


def normalize_version_name(version_name):
    """Normalize version names for Task 1 selection."""
    if not isinstance(version_name, str):
        return None
    normalized = version_name.strip().lower()
    if normalized in ("original", "orignal"):
        return "original"
    return normalized


def build_pairs_for_versions(corpus, version_map):
    """Build a paired dataframe for a caller-provided version mapping."""
    required_cols = {"dataset", "source", "key", "text", "version_name"}
    missing = required_cols - set(corpus.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = corpus.copy()
    df["version_normalized"] = df["version_name"].apply(normalize_version_name)
    df = df[df["version_normalized"].isin(version_map)].copy()

    if df.empty:
        raise RuntimeError("No requested target versions found in the corpus.")

    df["output_column"] = df["version_normalized"].map(version_map)

    paired = (
        df.pivot_table(
            index=["dataset", "source", "key"],
            columns="output_column",
            values="text",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )

    output_columns = ["dataset", "source", "key"] + list(dict.fromkeys(version_map.values()))
    for column in output_columns:
        if column not in paired.columns:
            paired[column] = pd.NA

    paired = paired[output_columns].reset_index(drop=True)

    duplicates = paired.duplicated(subset=["dataset", "source", "key"]).sum()
    if duplicates:
        raise RuntimeError(f"Found {duplicates} duplicate composite keys in paired data.")

    return paired.sort_values(["dataset", "source", "key"]).reset_index(drop=True)


def build_task1_pairs(corpus):
    """Build the Task 1 paired dataframe."""
    paired = build_pairs_for_versions(corpus, TASK1_TARGET_VERSIONS)
    paired = paired[TASK1_OUTPUT_COLUMNS]
    paired = paired.dropna(
        subset=[
            "text_T0",
            "text_chatgpt",
            "text_dipper_high",
            "text_pegasus_slight",
        ]
    ).reset_index(drop=True)
    return paired


def validate_task1_pairs(paired):
    """Basic validation for the Task 1 deliverable."""
    required_cols = set(TASK1_OUTPUT_COLUMNS)
    missing = required_cols - set(paired.columns)
    if missing:
        raise ValueError(f"Paired dataframe is missing columns: {sorted(missing)}")

    if paired["dataset"].isna().any():
        raise ValueError("Dataset column contains null values.")

    if paired[["dataset", "source", "key"]].duplicated().any():
        raise ValueError("Composite key (dataset, source, key) is not unique.")

    text_cols = [
        "text_T0",
        "text_chatgpt",
        "text_dipper_high",
        "text_pegasus_slight",
    ]
    if paired[text_cols].isna().any().any():
        raise ValueError("Some rows are missing one or more required text columns.")


def preview_two_rows_per_dataset(paired):
    """Return 2 rows per dataset for the assignment deliverable."""
    preview = (
        paired.groupby("dataset", group_keys=False)
        .head(2)
        .reset_index(drop=True)
    )
    return preview


def save_pairs(paired, output_path=None):
    """Cache paired dataframe to pickle."""
    output_path = Path(output_path) if output_path else DATA_PROCESSED / "paired_all.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    paired.to_pickle(output_path)
    print(f"Saved paired dataframe to {output_path}")
    return output_path


def main():
    corpus = load_task1_corpus()
    paired = build_task1_pairs(corpus)
    validate_task1_pairs(paired)
    save_pairs(paired)

    print("\npaired.shape")
    print(paired.shape)

    print("\n2 rows per dataset")
    preview = preview_two_rows_per_dataset(paired)
    print(preview.to_string(index=False, max_colwidth=80))

    print("\nValidation checks")
    print("- dataset column present:", "dataset" in paired.columns)
    print("- unique composite keys:", not paired[["dataset", "source", "key"]].duplicated().any())
    print(
        "- complete four-text rows:",
        not paired[
            ["text_T0", "text_chatgpt", "text_dipper_high", "text_pegasus_slight"]
        ].isna().any().any(),
    )


if __name__ == "__main__":
    main()
