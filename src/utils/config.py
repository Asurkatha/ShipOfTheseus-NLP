"""
Central configuration for the Ship of Theseus NLP project.
All paths, constants, and hyperparameters live here.

Updated based on actual corpus structure discovered 2026-02-21.
"""

from pathlib import Path

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_CLONED = DATA_RAW / "Ship_of_theseus_paraphrased_copus"
DATA_PARAPHRASED = DATA_CLONED / "paraphrased_datasets"
DATA_TRAIN = DATA_CLONED / "train_datasets"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_METADATA = PROJECT_ROOT / "data" / "metadata"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
FIGURES_DIR = PROJECT_ROOT / "figures"

# ── Dataset Config ──
# All 7 datasets in the corpus
ALL_DATASETS = ["cmv", "eli5", "sci_gen", "tldr", "wp", "xsum", "yelp"]

# Do not hardcode which datasets to focus on.
# Set this after exploration, or pass explicitly to load_corpus().
TARGET_DATASETS = None

# ── Source Authors ──
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

# ── Paraphraser Configurations ──
# 7 paraphraser variants (not 4). Dipper has 3 modes, Pegasus has 2.
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

# Grouped for higher-level analysis (collapse variants)
PARAPHRASER_FAMILIES = {
    "ChatGPT": ["chatgpt"],
    "PaLM2": ["palm"],
    "Dipper": ["dipper", "dipper(low)", "dipper(high)"],
    "Pegasus": ["pegasus(full)", "pegasus(slight)"],
}


def version_to_tier(version_str: str) -> str:
    """
    Map a version_name string to its iteration tier.

    Simple split on '_' works because none of the paraphraser base
    tokens (chatgpt, palm, dipper, dipper(low), dipper(high),
    pegasus(full), pegasus(slight)) contain underscores internally.

    Verified against Assignment 2 TheseusAnalyzer approach.

    Examples:
        'original'                                   -> 'T0'
        'chatgpt'                                    -> 'T1'
        'dipper(low)_dipper(low)'                    -> 'T2'
        'pegasus(full)_pegasus(full)_pegasus(full)'  -> 'T3'
    """
    if version_str == "original":
        return "T0"
    count = len(version_str.split("_"))
    return f"T{count}"


def version_to_paraphraser(version_str: str) -> str:
    """
    Extract the paraphraser key from a version_name string.

    Takes the first token from the underscore-split and matches
    against known paraphraser base_tokens.

    Examples:
        'original'                -> 'original'
        'chatgpt_chatgpt'        -> 'chatgpt'
        'dipper(low)_dipper(low)' -> 'dipper(low)'
        'pegasus(full)'           -> 'pegasus(full)'
    """
    if version_str == "original":
        return "original"

    # First token before any underscore is the base paraphraser name
    base = version_str.split("_")[0].lower()

    # Handle parenthetical variants: "dipper(low)" needs to match
    # the full token including parens, not just "dipper"
    for key, info in PARAPHRASERS.items():
        if info["base_token"] == base:
            return key

    # Fallback: check if base is a prefix of any known token
    # e.g., "dipper(low" from a bad split won't happen, but just in case
    for key, info in PARAPHRASERS.items():
        if base.startswith(info["base_token"].split("(")[0]):
            return key
    return base


# ── Model Config ──
SBERT_MODEL = "all-MiniLM-L6-v2"
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"
SPACY_MODEL = "en_core_web_sm"

# ── Experiment Config ──
RANDOM_SEED = 42
TEST_SIZE = 0.2
MAX_SAMPLES_PER_SUBSET = None  # Set to int for debugging (e.g., 500)

# ── Visualization ──
TIER_PALETTE = {
    "T0": "#2ecc71",   # green  - original
    "T1": "#f39c12",   # orange - first pass
    "T2": "#e74c3c",   # red    - second pass
    "T3": "#9b59b6",   # purple - third pass
}

PARAPHRASER_PALETTE = {
    "ChatGPT": "#1abc9c",
    "PaLM2": "#3498db",
    "Dipper": "#e67e22",
    "Dipper (Low)": "#d35400",
    "Dipper (High)": "#e74c3c",
    "Pegasus (Full)": "#8e44ad",
    "Pegasus (Slight)": "#9b59b6",
}

SOURCE_PALETTE = {
    "Human": "#2ecc71",
    "OpenAI": "#e74c3c",
    "LLAMA": "#3498db",
    "Tsinghua": "#f39c12",
    "BigScience": "#9b59b6",
    "PaLM": "#1abc9c",
    "Eleuther-AI": "#e67e22",
}