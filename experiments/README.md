# Experiments

Stored results from each phase of the project. CSVs and metrics are saved here so notebooks can reload them without recomputing.

## Directory Structure

```
experiments/
├── baseline_results/
│   └── similarity_baselines.csv    # BLEU, ROUGE-L, BERTScore for all paraphrasers x tiers
├── stylometry_results/             # Phase II: feature decay profiles (coming)
└── attribution_results/            # Phase III: classification metrics (coming)
```

## baseline_results/

### similarity_baselines.csv

Computed on Google Colab (T4 GPU) using Notebook 02. Contains 21 rows (7 paraphrasers x 3 tiers).

| Column | Description |
|--------|-------------|
| `Paraphraser` | Display name (e.g., ChatGPT, Dipper (High)) |
| `Paraphraser_Key` | Config key (e.g., chatgpt, dipper(high)) |
| `Tier` | T1, T2, or T3 |
| `BLEU` | Mean sentence-level smoothed BLEU |
| `BLEU_std` | Standard deviation |
| `ROUGE-L` | Mean ROUGE-L F1 |
| `ROUGE-L_std` | Standard deviation |
| `BERTScore` | Mean BERTScore F1 (RoBERTa-large) |
| `BERTScore_std` | Standard deviation |
| `n_samples` | Number of text pairs evaluated |

### Key findings

- BERTScore: 0.80-0.98 across all conditions (semantic hull preserved)
- BLEU: as low as 0.002 for Dipper (High) at T3 (lexical planks fully replaced)
- Sharpest decay at T1 for all paraphrasers
- Paraphraser aggressiveness: Pegasus (Slight) < Dipper (Low) ~ PaLM2 < Pegasus (Full) < ChatGPT < Dipper < Dipper (High)

## Upcoming

- `stylometry_results/`: Feature decay profiles, normalized against T0 (Weeks 7-8)
- `attribution_results/`: Authorship classifier performance across T1-T3, confusion matrices (Weeks 9-12)