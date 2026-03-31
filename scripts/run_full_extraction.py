"""
Full extraction pipeline: regenerate paired DataFrame with all 7 paraphrasers,
then extract POS vectors and NER sets for any missing columns.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.load_data import load_paired_t1
from src.features.pos import extract_pos_vectors
from src.features.ner import extract_ner_sets

POS_CACHE = ROOT / "data" / "processed" / "pos_vectors.npz"
NER_CACHE = ROOT / "data" / "processed" / "ner_sets.pkl"

ALL_TEXT_COLS = [
    "text_T0", "text_chatgpt", "text_palm", "text_dipper",
    "text_dipper_high", "text_dipper_low",
    "text_pegasus_slight", "text_pegasus_full",
]


def main():
    # Step 1: Regenerate paired DataFrame with all 7 paraphrasers
    print("=" * 60)
    print("STEP 1: Regenerating paired DataFrame")
    print("=" * 60)
    paired = load_paired_t1(cache=False)
    print(f"\nFinal shape: {paired.shape}")
    print(f"Columns: {list(paired.columns)}")

    # Also save to root for convenience
    paired.to_pickle(ROOT / "paired_all.pkl")
    print(f"Also saved to {ROOT / 'paired_all.pkl'}")

    # Step 2: POS extraction for missing columns
    print("\n" + "=" * 60)
    print("STEP 2: POS vector extraction")
    print("=" * 60)

    if POS_CACHE.exists():
        cached = dict(np.load(POS_CACHE))
        print(f"Existing POS cache: {list(cached.keys())}")
    else:
        cached = {}

    for col in ALL_TEXT_COLS:
        if col in cached:
            print(f"  {col}: already cached, skipping")
            continue
        if col not in paired.columns:
            print(f"  {col}: not in DataFrame, skipping")
            continue
        print(f"\n  Extracting POS for {col} ({len(paired)} texts)...")
        cached[col] = extract_pos_vectors(paired[col].tolist(), batch_size=500)
        # Checkpoint after each column
        np.savez_compressed(POS_CACHE, **cached)
        print(f"  Saved checkpoint ({col} done)")

    print(f"\nPOS cache complete: {list(cached.keys())}")

    # Step 3: NER extraction for missing columns
    print("\n" + "=" * 60)
    print("STEP 3: NER set extraction")
    print("=" * 60)

    if NER_CACHE.exists():
        with open(NER_CACHE, "rb") as f:
            ner_cache = pickle.load(f)
        print(f"Existing NER cache: {list(ner_cache.keys())}")
    else:
        ner_cache = {}

    for col in ALL_TEXT_COLS:
        if col in ner_cache:
            print(f"  {col}: already cached, skipping")
            continue
        if col not in paired.columns:
            print(f"  {col}: not in DataFrame, skipping")
            continue
        print(f"\n  Extracting NER for {col} ({len(paired)} texts)...")
        ner_cache[col] = extract_ner_sets(paired[col].tolist(), batch_size=1000)
        # Checkpoint
        with open(NER_CACHE, "wb") as f:
            pickle.dump(ner_cache, f)
        print(f"  Saved checkpoint ({col} done)")

    print(f"\nNER cache complete: {list(ner_cache.keys())}")

    # Step 4: Compute all metrics
    print("\n" + "=" * 60)
    print("STEP 4: Computing all metrics")
    print("=" * 60)

    from src.utils.metrics import batch_cosine_similarity
    from src.features.ner import entity_metrics
    import math

    pos_vectors = dict(np.load(POS_CACHE))
    paraphrasers = {
        "chatgpt": "text_chatgpt",
        "palm": "text_palm",
        "dipper": "text_dipper",
        "dipper_high": "text_dipper_high",
        "dipper_low": "text_dipper_low",
        "pegasus_slight": "text_pegasus_slight",
        "pegasus_full": "text_pegasus_full",
    }

    print("\n--- POS Cosine Similarity (T0 vs T1) ---")
    for label, col in paraphrasers.items():
        if col not in pos_vectors:
            print(f"  {label}: POS vectors missing")
            continue
        cos = batch_cosine_similarity(pos_vectors["text_T0"], pos_vectors[col])
        valid = cos[~np.isnan(cos)]
        print(f"  {label:20s}: {valid.mean():.4f} +/- {valid.std():.4f}")

    print("\n--- NER Entity Metrics (T0 vs T1) ---")
    for label, col in paraphrasers.items():
        if col not in ner_cache:
            print(f"  {label}: NER sets missing")
            continue
        jaccards, recalls, precisions = [], [], []
        for i in range(len(paired)):
            j, r, p = entity_metrics(ner_cache["text_T0"][i], ner_cache[col][i])
            jaccards.append(j)
            recalls.append(r)
            precisions.append(p)

        j_arr = pd.Series(jaccards).dropna()
        r_arr = pd.Series(recalls).dropna()
        p_arr = pd.Series(precisions).dropna()
        print(f"  {label:20s}: J={j_arr.mean():.4f}+/-{j_arr.std():.4f}  "
              f"R={r_arr.mean():.4f}+/-{r_arr.std():.4f}  "
              f"P={p_arr.mean():.4f}+/-{p_arr.std():.4f}")

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
