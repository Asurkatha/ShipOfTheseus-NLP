"""
Extract NER for a single column. Usage:
    python run_ner_single.py text_T0
    python run_ner_single.py text_chatgpt
"""
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features.ner import extract_ner_sets

PAIRED_PATH = ROOT / "paired_all.pkl"
CACHE_PATH = ROOT / "data" / "processed" / "ner_sets.pkl"


def main():
    col = sys.argv[1] if len(sys.argv) > 1 else None
    if not col:
        print("Usage: python run_ner_single.py <column_name>")
        sys.exit(1)

    print(f"Loading paired data ...")
    paired = pd.read_pickle(PAIRED_PATH)

    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            ner_cache = pickle.load(f)
        if col in ner_cache:
            print(f"  {col} already cached, skipping")
            return
    else:
        ner_cache = {}

    print(f"Extracting NER for {col} ({len(paired)} texts) ...")
    texts = paired[col].tolist()
    ner_sets = extract_ner_sets(texts, batch_size=1000)
    ner_cache[col] = ner_sets

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(ner_cache, f)
    print(f"Done! {col} saved. Total cached: {list(ner_cache.keys())}")


if __name__ == "__main__":
    main()
