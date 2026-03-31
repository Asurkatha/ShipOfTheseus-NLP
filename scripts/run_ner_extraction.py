"""
Standalone script to run NER extraction locally.
Loads paired_all.pkl, extracts NER sets for all 6 text columns,
and caches results to data/processed/ner_sets.pkl.
"""
import pickle
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features.ner import extract_ner_sets

PAIRED_PATH = ROOT / "paired_all.pkl"
CACHE_PATH = ROOT / "data" / "processed" / "ner_sets.pkl"

NER_COLS = [
    "text_T0",
    "text_chatgpt",
    "text_dipper_high",
    "text_dipper_low",
    "text_pegasus_slight",
    "text_pegasus_full",
]


def main():
    print(f"Loading paired data from {PAIRED_PATH} ...")
    paired = pd.read_pickle(PAIRED_PATH)
    print(f"  Shape: {paired.shape}")

    # Check if partial cache exists (resume support)
    if CACHE_PATH.exists():
        print(f"Loading existing cache from {CACHE_PATH} ...")
        with open(CACHE_PATH, "rb") as f:
            ner_cache = pickle.load(f)
        print(f"  Cached columns: {list(ner_cache.keys())}")
    else:
        ner_cache = {}

    for col in NER_COLS:
        if col in ner_cache:
            print(f"  Skipping {col} (already cached)")
            continue
        print(f"\nExtracting NER for {col} ({len(paired)} texts) ...")
        texts = paired[col].tolist()
        ner_sets = extract_ner_sets(texts, batch_size=500)
        ner_cache[col] = ner_sets

        # Save after each column for resume support
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(ner_cache, f)
        print(f"  Saved checkpoint ({col} done)")

    print(f"\nAll done! NER sets cached to {CACHE_PATH}")
    print(f"Columns: {list(ner_cache.keys())}")


if __name__ == "__main__":
    main()
