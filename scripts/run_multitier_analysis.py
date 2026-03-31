"""
Multi-tier feature decay analysis: compute POS cosine and NER metrics
across T0 -> T1 -> T2 -> T3 for all 7 paraphrasers.
Also extracts dependency tree depth.
"""
import sys, pickle, math
from pathlib import Path
import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features.pos import extract_pos_vectors, UPOS_TAGS
from src.features.ner import extract_ner_sets, entity_metrics
from src.utils.metrics import batch_cosine_similarity
from src.utils.config import ALL_DATASETS, DATA_PROCESSED, parse_version

CACHE_DIR = DATA_PROCESSED
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PARAPHRASERS = {
    "chatgpt": "ChatGPT",
    "palm": "PaLM2",
    "dipper": "Dipper",
    "dipper(low)": "Dipper (Low)",
    "dipper(high)": "Dipper (High)",
    "pegasus(full)": "Pegasus (Full)",
    "pegasus(slight)": "Pegasus (Slight)",
}


def build_multitier_pairs(corpus):
    """Build T0/T1/T2/T3 aligned texts for each paraphraser."""
    # Get T0 texts
    t0 = corpus[corpus["tier"] == "T0"][["key", "source", "dataset", "text"]].copy()
    t0 = t0.rename(columns={"text": "text_T0"})

    results = {}
    for para_key, para_name in PARAPHRASERS.items():
        para_df = corpus[corpus["paraphraser"] == para_key].copy()
        if len(para_df) == 0:
            print(f"  Skipping {para_key}: no data")
            continue

        # Pivot tiers
        pivoted = para_df.pivot_table(
            index=["key", "source", "dataset"],
            columns="tier",
            values="text",
            aggfunc="first"
        ).reset_index()
        pivoted.columns.name = None

        # Merge with T0
        merged = t0.merge(pivoted, on=["key", "source", "dataset"], how="inner")

        # Rename tier columns
        for tier in ["T1", "T2", "T3"]:
            if tier in merged.columns:
                merged = merged.rename(columns={tier: f"text_{tier}"})

        # Drop rows missing T0 or T1
        merged = merged.dropna(subset=["text_T0", "text_T1"])
        results[para_key] = merged
        print(f"  {para_key}: {len(merged)} articles, tiers: {[c for c in merged.columns if c.startswith('text_T')]}")

    return results


def extract_dep_depth(texts, model_name="en_core_web_sm", batch_size=500):
    """Extract mean dependency tree depth per text."""
    nlp = spacy.load(model_name, disable=["ner", "lemmatizer"])
    nlp.max_length = 2_000_000
    texts_list = [t if isinstance(t, str) else "" for t in texts]
    depths = []

    for doc in tqdm(nlp.pipe(texts_list, batch_size=batch_size),
                    total=len(texts_list), desc="Dep depth"):
        if len(doc) == 0:
            depths.append(0.0)
            continue
        # For each sentence, compute max depth from root
        sent_depths = []
        for sent in doc.sents:
            # BFS from root to find max depth
            root = sent.root
            max_d = 0
            stack = [(root, 0)]
            while stack:
                token, d = stack.pop()
                max_d = max(max_d, d)
                for child in token.children:
                    stack.append((child, d + 1))
            sent_depths.append(max_d)
        depths.append(np.mean(sent_depths) if sent_depths else 0.0)

    return np.array(depths)


def main():
    cache_path = CACHE_DIR / "multitier_results.pkl"

    if cache_path.exists():
        print(f"Loading cached results from {cache_path}")
        with open(cache_path, "rb") as f:
            all_results = pickle.load(f)
        print(f"  Paraphrasers: {list(all_results.keys())}")
        # Check if complete
        sample = list(all_results.values())[0]
        if "dep_depth_T0" in sample.columns:
            print("  All features present, skipping to summary")
            print_summary(all_results)
            return
        else:
            print("  Missing dep_depth, will add it")

    # Load corpus
    print("Loading corpus...")
    corpus = pd.read_parquet(ROOT / "data" / "processed" / "corpus_full.parquet")
    print(f"  Corpus: {corpus.shape}")

    # Build multi-tier pairs
    print("\nBuilding multi-tier pairs...")
    pairs = build_multitier_pairs(corpus)

    # For each paraphraser, compute features at each tier
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION")
    print("=" * 60)

    all_results = {}

    for para_key, df in pairs.items():
        print(f"\n--- {para_key} ({len(df)} articles) ---")
        tier_cols = [c for c in df.columns if c.startswith("text_T")]

        for tier_col in tier_cols:
            tier = tier_col.replace("text_", "")
            texts = df[tier_col].tolist()
            n_valid = sum(1 for t in texts if isinstance(t, str) and len(t) > 0)
            if n_valid < 100:
                print(f"  {tier}: only {n_valid} valid texts, skipping")
                continue

            # POS vectors
            pos_col = f"pos_{tier}"
            if pos_col not in df.columns:
                print(f"  POS {tier}...")
                pos_vecs = extract_pos_vectors(texts, batch_size=500)
                df[pos_col] = list(pos_vecs)

            # NER sets
            ner_col = f"ner_{tier}"
            if ner_col not in df.columns:
                print(f"  NER {tier}...")
                ner_sets = extract_ner_sets(texts, batch_size=1000)
                df[ner_col] = ner_sets

            # Dependency depth
            dep_col = f"dep_depth_{tier}"
            if dep_col not in df.columns:
                print(f"  DepDepth {tier}...")
                depths = extract_dep_depth(texts, batch_size=500)
                df[dep_col] = depths

        # Compute metrics relative to T0
        pos_T0 = np.stack(df["pos_T0"].values)
        ner_T0 = df["ner_T0"].values

        for tier in ["T1", "T2", "T3"]:
            pos_col = f"pos_{tier}"
            ner_col = f"ner_{tier}"

            if pos_col in df.columns:
                pos_Ti = np.stack(df[pos_col].values)
                df[f"cos_{tier}"] = batch_cosine_similarity(pos_T0, pos_Ti)

            if ner_col in df.columns:
                jaccards, recalls, precisions = [], [], []
                for i in range(len(df)):
                    j, r, p = entity_metrics(ner_T0[i], df[ner_col].iloc[i])
                    jaccards.append(j)
                    recalls.append(r)
                    precisions.append(p)
                df[f"ner_jaccard_{tier}"] = jaccards
                df[f"ner_recall_{tier}"] = recalls
                df[f"ner_precision_{tier}"] = precisions

        all_results[para_key] = df
        print(f"  Done: {len(df.columns)} columns")

        # Checkpoint after each paraphraser
        with open(cache_path, "wb") as f:
            pickle.dump(all_results, f)
        print(f"  Checkpoint saved")

    print("\n" + "=" * 60)
    print_summary(all_results)


def print_summary(all_results):
    """Print the multi-tier summary tables."""
    print("MULTI-TIER DECAY SUMMARY")
    print("=" * 60)

    print("\n--- POS Cosine Similarity (T0 vs Tn) ---")
    print(f"{'Paraphraser':<20s} {'T1':>10s} {'T2':>10s} {'T3':>10s}")
    print("-" * 55)
    for para_key, df in all_results.items():
        row = f"{para_key:<20s}"
        for tier in ["T1", "T2", "T3"]:
            col = f"cos_{tier}"
            if col in df.columns:
                vals = df[col].dropna()
                row += f" {vals.mean():>9.4f}"
            else:
                row += f" {'N/A':>9s}"
        print(row)

    print("\n--- NER Recall (T0 vs Tn) ---")
    print(f"{'Paraphraser':<20s} {'T1':>10s} {'T2':>10s} {'T3':>10s}")
    print("-" * 55)
    for para_key, df in all_results.items():
        row = f"{para_key:<20s}"
        for tier in ["T1", "T2", "T3"]:
            col = f"ner_recall_{tier}"
            if col in df.columns:
                vals = df[col].dropna()
                row += f" {vals.mean():>9.4f}"
            else:
                row += f" {'N/A':>9s}"
        print(row)

    print("\n--- Mean Dependency Depth ---")
    print(f"{'Paraphraser':<20s} {'T0':>10s} {'T1':>10s} {'T2':>10s} {'T3':>10s}")
    print("-" * 65)
    for para_key, df in all_results.items():
        row = f"{para_key:<20s}"
        for tier in ["T0", "T1", "T2", "T3"]:
            col = f"dep_depth_{tier}"
            if col in df.columns:
                row += f" {df[col].mean():>9.2f}"
            else:
                row += f" {'N/A':>9s}"
        print(row)


if __name__ == "__main__":
    main()
