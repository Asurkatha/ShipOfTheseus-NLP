"""
Light preprocessing for the Ship of Theseus corpus.
"""

import re
import pandas as pd
from pathlib import Path


def clean_text(text):
    """Remove data artifacts (PaLM2 markdown, unicode junk) without touching style."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # strip markdown bold
    text = re.sub(r"^\s*[\*\-\u2022]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[\u00a0\u200b\u200c\u200d\ufeff]", " ", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def validate_text(text, min_words=10, max_words=5000):
    if not isinstance(text, str) or not text.strip():
        return False
    return min_words <= len(text.split()) <= max_words


def preprocess_corpus(df, text_col="text", min_words=10):
    """Clean text, drop junk rows, add basic metadata columns."""
    initial = len(df)
    print(f"Preprocessing {initial} rows...")

    df = df.dropna(subset=[text_col]).copy()
    df[text_col] = df[text_col].apply(clean_text)

    # Metadata
    df["word_count"] = df[text_col].apply(lambda x: len(x.split()))
    df["char_count"] = df[text_col].apply(len)
    df["sentence_count"] = df[text_col].apply(
        lambda x: len(re.split(r"[.!?]+", x.strip()))
    )

    # Filter short/empty texts
    valid = df[text_col].apply(lambda x: validate_text(x, min_words=min_words))
    removed = (~valid).sum()
    if removed:
        print(f"  Dropped {removed} texts under {min_words} words")
    df = df[valid]

    # Dedup
    dedup_cols = (["key", "version_name"] if "version_name" in df.columns
                  else ["key", "version"] if "version" in df.columns
                  else ["key"])
    before = len(df)
    df = df.drop_duplicates(subset=dedup_cols, keep="first")
    dupes = before - len(df)
    if dupes:
        print(f"  Removed {dupes} duplicates")

    print(f"  Final: {len(df)} rows (dropped {initial - len(df)})")
    return df


def compute_text_stats(df, text_col="text"):
    """Add quick-look stats: avg word length, TTR, punctuation density."""
    df = df.copy()

    df["avg_word_length"] = df[text_col].apply(
        lambda x: sum(len(w) for w in x.split()) / max(len(x.split()), 1)
    )

    df["ttr"] = df[text_col].apply(
        lambda x: len(set(x.lower().split())) / max(len(x.lower().split()), 1)
    )

    df["punct_density"] = df[text_col].apply(
        lambda x: sum(1 for c in x if c in ".,;:!?-()\"'") / max(len(x), 1)
    )

    return df


def save_processed(df, name, output_dir=None):
    """Save to parquet. Requires pyarrow: pip install pyarrow"""
    from src.utils.config import DATA_PROCESSED
    out_dir = Path(output_dir) if output_dir else DATA_PROCESSED
    out_path = out_dir / f"{name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved {out_path} ({len(df)} rows)")