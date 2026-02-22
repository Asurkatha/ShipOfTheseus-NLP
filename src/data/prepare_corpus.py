"""
One-time script to load, preprocess, and cache the corpus
into the project directory structure:

    data/processed/
        t0_human/           <- original texts (version_name == 'original')
        t1_paraphrased/     <- first paraphrase iteration
        t2_paraphrased/     <- second iteration
        t3_paraphrased/     <- third iteration

Each tier directory contains one parquet per dataset
(e.g., xsum.parquet, yelp.parquet).

Usage:
    pip install pyarrow   # if not already installed
    python -m src.data.prepare_corpus
"""

from pathlib import Path
from src.data.load_data import load_corpus
from src.data.preprocess import preprocess_corpus, compute_text_stats
from src.utils.config import DATA_PROCESSED


TIER_DIRS = {
    "T0": "t0_human",
    "T1": "t1_paraphrased",
    "T2": "t2_paraphrased",
    "T3": "t3_paraphrased",
}


def save_by_tier_and_dataset(corpus):
    """Split corpus into tier/dataset parquet files."""
    for tier, dirname in TIER_DIRS.items():
        tier_dir = DATA_PROCESSED / dirname
        tier_dir.mkdir(parents=True, exist_ok=True)

        tier_data = corpus[corpus["tier"] == tier]
        if len(tier_data) == 0:
            print(f"  {dirname}: no data, skipping")
            continue

        for dataset in tier_data["dataset"].unique():
            subset = tier_data[tier_data["dataset"] == dataset]
            out_path = tier_dir / f"{dataset}.parquet"
            subset.to_parquet(out_path, index=False)
            print(f"  {dirname}/{dataset}.parquet: {len(subset):,} rows")


def main():
    print("Loading corpus...")
    corpus = load_corpus(datasets=None, include_train=False, max_samples=None)

    print("\nPreprocessing...")
    corpus = preprocess_corpus(corpus)
    corpus = compute_text_stats(corpus)

    print("\nSaving by tier and dataset...")
    save_by_tier_and_dataset(corpus)

    # Also save the full thing for convenience
    full_path = DATA_PROCESSED / "corpus_full.parquet"
    corpus.to_parquet(full_path, index=False)
    print(f"\n  Full corpus: {full_path} ({len(corpus):,} rows)")

    print("\nDone. Notebooks can load with:")
    print("  # Full corpus")
    print("  corpus = pd.read_parquet('data/processed/corpus_full.parquet')")
    print("  # Or just one tier/dataset")
    print("  t0_xsum = pd.read_parquet('data/processed/t0_human/xsum.parquet')")


if __name__ == "__main__":
    main()