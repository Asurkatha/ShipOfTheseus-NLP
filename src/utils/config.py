"""
Central config for the Ship of Theseus NLP project.
Paths, constants, paraphraser definitions, color palettes.
"""

import re
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_CLONED = DATA_RAW / "Ship_of_theseus_paraphrased_copus"
DATA_PARAPHRASED = DATA_CLONED / "paraphrased_datasets"
DATA_TRAIN = DATA_CLONED / "train_datasets"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_METADATA = PROJECT_ROOT / "data" / "metadata"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
FIGURES_DIR = PROJECT_ROOT / "figures"

# All 7 datasets in the corpus
ALL_DATASETS = ["cmv", "eli5", "sci_gen", "tldr", "wp", "xsum", "yelp"]

# Set after exploration, or pass explicitly to load_corpus()
TARGET_DATASETS = None

# 7 sources: 1 human + 6 LLMs
SOURCES = {
    "Human": {"type": "human"},
    "OpenAI": {"type": "llm"},
    "LLAMA": {"type": "llm"},
    "Tsinghua": {"type": "llm"},
    "BigScience": {"type": "llm"},
    "PaLM": {"type": "llm"},
    "Eleuther-AI": {"type": "llm"},
}

# 7 paraphraser variants: Dipper has 3 modes, Pegasus has 2
PARAPHRASERS = {
    "chatgpt": {
        "name": "ChatGPT",
        "type": "full-text",
        "base_token": "chatgpt",
    },
    "palm": {
        "name": "PaLM2",
        "type": "full-text",
        "base_token": "palm",
    },
    "dipper": {
        "name": "Dipper",
        "type": "controlled-default",
        "base_token": "dipper",
    },
    "dipper(low)": {
        "name": "Dipper (Low)",
        "type": "controlled-low",
        "base_token": "dipper(low)",
    },
    "dipper(high)": {
        "name": "Dipper (High)",
        "type": "controlled-high",
        "base_token": "dipper(high)",
    },
    "pegasus(full)": {
        "name": "Pegasus (Full)",
        "type": "sentence-wise-full",
        "base_token": "pegasus(full)",
    },
    "pegasus(slight)": {
        "name": "Pegasus (Slight)",
        "type": "sentence-wise-25pct",
        "base_token": "pegasus(slight)",
    },
}

# For higher-level analysis where we collapse variants
PARAPHRASER_FAMILIES = {
    "ChatGPT": ["chatgpt"],
    "PaLM2": ["palm"],
    "Dipper": ["dipper", "dipper(low)", "dipper(high)"],
    "Pegasus": ["pegasus(full)", "pegasus(slight)"],
}


# Version string → column name mapping for the T1 forensic pivot
VERSION_TO_COL = {
    "original":        "text_T0",
    "chatgpt":         "text_chatgpt",
    "dipper(high)":    "text_dipper_high",
    "dipper(low)":     "text_dipper_low",
    "pegasus(slight)": "text_pegasus_slight",
    "pegasus(full)":   "text_pegasus_full",
}

# Regex finds the shortest repeating base token without naive underscore splitting.
# Correctly handles parenthesised tokens like dipper(low)_dipper(low).
_BASE_RE = re.compile(r"(.+?)(?:_\1)*$")


def parse_version(version_str):
    """Return (base_token, repetition_count) for a version string.

    >>> parse_version("chatgpt_chatgpt_chatgpt")
    ('chatgpt', 3)
    >>> parse_version("dipper(low)_dipper(low)")
    ('dipper(low)', 2)
    >>> parse_version("original")
    ('original', 0)
    """
    if version_str in ("original", "orignal"):
        return ("original", 0)
    m = _BASE_RE.fullmatch(version_str)
    if not m:
        raise ValueError(f"Cannot parse version: '{version_str}'")
    base = m.group(1)
    count = (len(version_str) + 1) // (len(base) + 1)
    return (base, count)


def version_to_tier(version_str):
    """'original' -> T0, 'chatgpt' -> T1, 'chatgpt_chatgpt' -> T2, etc."""
    base, n = parse_version(version_str)
    if base == "original":
        return "T0"
    return f"T{n}"


def version_to_paraphraser(version_str):
    """'chatgpt_chatgpt' -> 'chatgpt', 'dipper(low)_dipper(low)' -> 'dipper(low)'."""
    base, _ = parse_version(version_str)
    if base == "original":
        return "original"
    for key, info in PARAPHRASERS.items():
        if info["base_token"] == base:
            return key
    return base


# Models
SBERT_MODEL = "all-MiniLM-L6-v2"
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"
SPACY_MODEL = "en_core_web_sm"

# Experiment defaults
RANDOM_SEED = 42
TEST_SIZE = 0.2
MAX_SAMPLES_PER_SUBSET = None  # set to int (e.g. 500) for debugging

# Color palettes
TIER_PALETTE = {
    "T0": "#2ecc71", "T1": "#f39c12", "T2": "#e74c3c", "T3": "#9b59b6",
}
PARAPHRASER_PALETTE = {
    "ChatGPT": "#1abc9c", "PaLM2": "#3498db",
    "Dipper": "#e67e22", "Dipper (Low)": "#d35400", "Dipper (High)": "#e74c3c",
    "Pegasus (Full)": "#8e44ad", "Pegasus (Slight)": "#9b59b6",
}
SOURCE_PALETTE = {
    "Human": "#2ecc71", "OpenAI": "#e74c3c", "LLAMA": "#3498db",
    "Tsinghua": "#f39c12", "BigScience": "#9b59b6",
    "PaLM": "#1abc9c", "Eleuther-AI": "#e67e22",
}