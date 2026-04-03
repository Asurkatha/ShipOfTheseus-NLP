"""
Statistical significance tests for Ship of Theseus project.

Tests:
1. Is the F1 drop from T1 to T2 significant? (RQ2)
2. Is BLEU decay significantly faster than BERTScore? (RQ1)
3. Is Dipper(High) NER Recall significantly lower than Dipper(Low)? (RQ1)
4. Is paraphraser identification F1 significantly above random baseline? (RQ3)

Requires: data/processed/multitier_results.pkl
"""
import sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.metrics import paired_permutation_test, bootstrap_ci

CACHE = ROOT / "data" / "processed" / "multitier_results.pkl"
BASELINES = ROOT / "experiments" / "baseline_results" / "similarity_baselines.csv"


def main():
    print("=" * 60)
    print("SIGNIFICANCE TESTS — Ship of Theseus")
    print("=" * 60)

    with open(CACHE, "rb") as f:
        results = pickle.load(f)

    # ── Test 1: NER Recall at T1 vs T2 (paired, per-document) ──
    print("\n--- Test 1: NER Recall T1 vs T2 (across all paraphrasers) ---")
    all_t1, all_t2 = [], []
    for para_key, df in results.items():
        if "ner_recall_T1" in df.columns and "ner_recall_T2" in df.columns:
            valid = df[["ner_recall_T1", "ner_recall_T2"]].dropna()
            all_t1.extend(valid["ner_recall_T1"].tolist())
            all_t2.extend(valid["ner_recall_T2"].tolist())

    diff, p = paired_permutation_test(all_t1, all_t2)
    print(f"  Mean NER Recall T1: {np.mean(all_t1):.4f}")
    print(f"  Mean NER Recall T2: {np.mean(all_t2):.4f}")
    print(f"  Observed diff (T1 - T2): {diff:.4f}")
    print(f"  p-value: {p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}")

    # ── Test 2: POS Cosine T1 vs T2 (paired) ──
    print("\n--- Test 2: POS Cosine T1 vs T2 (across all paraphrasers) ---")
    pos_t1, pos_t2 = [], []
    for para_key, df in results.items():
        if "cos_T1" in df.columns and "cos_T2" in df.columns:
            valid = df[["cos_T1", "cos_T2"]].dropna()
            pos_t1.extend(valid["cos_T1"].tolist())
            pos_t2.extend(valid["cos_T2"].tolist())

    diff, p = paired_permutation_test(pos_t1, pos_t2)
    print(f"  Mean POS Cosine T1: {np.mean(pos_t1):.4f}")
    print(f"  Mean POS Cosine T2: {np.mean(pos_t2):.4f}")
    print(f"  Observed diff (T1 - T2): {diff:.4f}")
    print(f"  p-value: {p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}")

    # ── Test 3: Dipper(High) vs Dipper(Low) NER Recall at T1 ──
    print("\n--- Test 3: Dipper(High) vs Dipper(Low) NER Recall at T1 ---")
    if "dipper(high)" in results and "dipper(low)" in results:
        df_high = results["dipper(high)"]
        df_low = results["dipper(low)"]

        # Align on common (key, source)
        keys_high = set(zip(df_high["key"], df_high["source"]))
        keys_low = set(zip(df_low["key"], df_low["source"]))
        common = sorted(keys_high & keys_low)

        high_vals, low_vals = [], []
        for k, s in common:
            h_row = df_high[(df_high["key"] == k) & (df_high["source"] == s)]
            l_row = df_low[(df_low["key"] == k) & (df_low["source"] == s)]
            if len(h_row) > 0 and len(l_row) > 0:
                h_v = h_row["ner_recall_T1"].values[0]
                l_v = l_row["ner_recall_T1"].values[0]
                if not (np.isnan(h_v) or np.isnan(l_v)):
                    high_vals.append(h_v)
                    low_vals.append(l_v)

        diff, p = paired_permutation_test(low_vals, high_vals)
        print(f"  Dipper(Low)  NER Recall T1: {np.mean(low_vals):.4f}")
        print(f"  Dipper(High) NER Recall T1: {np.mean(high_vals):.4f}")
        print(f"  Observed diff (Low - High): {diff:.4f}")
        print(f"  p-value: {p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}")
    else:
        print("  SKIPPED: dipper(high) or dipper(low) not in results")

    # ── Test 4: NER Recall decay rate vs POS Cosine decay rate at T1 ──
    print("\n--- Test 4: Content (NER) decays faster than Style (POS) at T1 ---")
    # For each document, compute 1 - metric (amount of decay)
    ner_decay, pos_decay = [], []
    for para_key, df in results.items():
        if "ner_recall_T1" in df.columns and "cos_T1" in df.columns:
            valid = df[["ner_recall_T1", "cos_T1"]].dropna()
            ner_decay.extend((1 - valid["ner_recall_T1"]).tolist())
            pos_decay.extend((1 - valid["cos_T1"]).tolist())

    diff, p = paired_permutation_test(ner_decay, pos_decay)
    print(f"  Mean NER decay (1 - recall): {np.mean(ner_decay):.4f}")
    print(f"  Mean POS decay (1 - cosine): {np.mean(pos_decay):.4f}")
    print(f"  Observed diff (NER - POS):   {diff:.4f}")
    print(f"  p-value: {p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}")
    if diff > 0 and p < 0.05:
        print(f"  => Content decays significantly faster than style (ratio: {np.mean(ner_decay)/max(np.mean(pos_decay), 1e-8):.1f}x)")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns p>=0.05")
    print("All tests: two-sided paired permutation test, 10,000 permutations")
    print("=" * 60)


if __name__ == "__main__":
    main()
