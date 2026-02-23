# Raw Data

This directory holds the cloned Ship of Theseus Paraphrased Corpus.

## Setup

```bash
cd data/raw
git clone https://github.com/tripto03/Ship_of_theseus_paraphrased_copus.git
```

## Corpus Structure

After cloning, the directory looks like:

```
raw/
└── Ship_of_theseus_paraphrased_copus/
    ├── paraphrased_datasets/
    │   ├── cmv_paraphrased.csv
    │   ├── eli5_paraphrased.csv
    │   ├── sci_gen_paraphrased.csv
    │   ├── tldr_paraphrased.csv
    │   ├── wp_paraphrased.csv
    │   ├── xsum_paraphrased.csv
    │   └── yelp_paraphrased.csv
    └── train_datasets/
        ├── cmv_train.csv
        ├── eli5_train.csv
        ├── sci_gen_train.csv
        ├── tldr_train.csv
        ├── wp_train.csv
        ├── xsum_train.csv
        └── yelp_train.csv
```

## File Formats

**Paraphrased files** (test split) have 4 columns:
- `source`: who produced the original text (Human, OpenAI, LLAMA, Tsinghua, BigScience, PaLM, Eleuther-AI)
- `key`: document identifier (e.g., `xsum-492`)
- `text`: the actual text content
- `version_name`: paraphrase chain identifier (e.g., `original`, `chatgpt`, `chatgpt_chatgpt`, `chatgpt_chatgpt_chatgpt`)

**Train files** (unparaphrased) have 3 columns: `source`, `key`, `text`.

## Paraphrasers

The corpus uses 7 paraphraser variants across 4 model families:

| Variant | Family | Type |
|---------|--------|------|
| chatgpt | ChatGPT | Full-text |
| palm | PaLM2 | Full-text |
| dipper | Dipper | Controlled (default) |
| dipper(low) | Dipper | Controlled (low diversity) |
| dipper(high) | Dipper | Controlled (high diversity) |
| pegasus(full) | Pegasus | Sentence-wise (all sentences) |
| pegasus(slight) | Pegasus | Sentence-wise (25% of sentences) |

## Version Naming

The `version_name` column encodes the paraphrase chain:
- `original` = T0 (no paraphrasing)
- `chatgpt` = T1 (paraphrased once by ChatGPT)
- `chatgpt_chatgpt` = T2 (paraphrased twice)
- `chatgpt_chatgpt_chatgpt` = T3 (paraphrased three times)

Same pattern for all 7 variants. 22 unique version strings total.

## Note

Raw data files are git-ignored. Each user needs to clone the corpus themselves. See the root README for setup instructions.