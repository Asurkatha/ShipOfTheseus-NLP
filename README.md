# Ship of Theseus: Computational Forensics and the Paradox of Authorial Identity

**NLP Course Project — Northeastern University, Spring 2026**\
**Team:** Rohan Joshi, Aarav Surkatha

When you paraphrase a text over and over with language models, at what point does it stop being the original author's work? This project uses the [Ship of Theseus Paraphrased Corpus](https://github.com/tripto03/Ship_of_theseus_paraphrased_copus) to track how authorial identity decays through iterative paraphrasing (T0 → T3) across 7 datasets and 7 paraphraser variants.

We investigate three research questions:
- **RQ1:** Which linguistic features decay first — style or content?
- **RQ2:** At what iteration does authorship attribution effectively fail?
- **RQ3:** Do different paraphrasers leave identifiable fingerprints?

![Project Cover Image](Cover_Photo.png)

## Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/ShipOfTheseus-NLP.git
cd ShipOfTheseus-NLP

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows CMD
# source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Clone the dataset
cd data/raw
git clone https://github.com/tripto03/Ship_of_theseus_paraphrased_copus.git
cd ../..

# Organize raw data into per-dataset subdirectories
python -c "from src.data.load_data import organize_into_subdirs; organize_into_subdirs(dry_run=False)"

# Preprocess and cache the corpus
python -m src.data.prepare_corpus
```

## Repository Structure

```
ShipOfTheseus-NLP/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                          # Cloned corpus lives here
│   ├── processed/                    # Cached parquets by tier
│   │   ├── t0_human/
│   │   ├── t1_paraphrased/
│   │   ├── t2_paraphrased/
│   │   ├── t3_paraphrased/
│   │   └── corpus_full.parquet
│   └── metadata/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA and corpus profiling
│   └── 02_baseline_similarity.ipynb  # BLEU, ROUGE-L, BERTScore baselines
│
├── src/
│   ├── data/
│   │   ├── load_data.py              # Corpus loading and pairing
│   │   ├── preprocess.py             # Light text cleaning
│   │   └── prepare_corpus.py         # One-time preprocessing script
│   ├── features/                     # Stylometric feature extraction (Phase II)
│   ├── similarity/                   # Similarity metric wrappers
│   ├── attribution/                  # Authorship attribution models (Phase III)
│   ├── visualization/                # Plotting utilities
│   └── utils/
│       ├── config.py                 # Paths, constants, palettes
│       └── metrics.py                # Centralized metric functions
│
├── experiments/
│   ├── baseline_results/             # Similarity baseline CSVs
│   ├── stylometry_results/           # Feature decay profiles (Phase II)
│   └── attribution_results/          # Classification results (Phase III)
│
├── figures/
│   ├── decay_curves/                 # Similarity decay plots
│   └── eda/                          # Exploratory analysis plots
│
├── paper/
│   ├── acl2026.tex                   # Main LaTeX file (ACM sigconf)
│   ├── sections/                     # Modular .tex sections
│   └── references.bib
│
└── docs/
    ├── project_proposal.md
    └── weekly_progress.md
```

## Running the Notebooks

Notebooks expect the preprocessed corpus at `data/processed/corpus_full.parquet`. Run `python -m src.data.prepare_corpus` first if you haven't.

BERTScore computation requires a CUDA GPU. If you don't have one locally, Notebook 02 includes a Google Colab section with the full computation script. Run it there, download the CSV, and place it in `experiments/baseline_results/`.

## Key Results So Far

- BERTScore stays between 0.80-0.98 across all tiers while BLEU drops as low as 0.002 — meaning is preserved, surface style is replaced
- The sharpest decay happens at T1 (first paraphrase), with diminishing changes after
- Dipper (High) vs Dipper (Low) show wider variation than entirely different model families, suggesting paraphraser settings leave distinct fingerprints
- Punctuation marks decay monotonically: exclamation marks drop 63%, commas only 18%

## References

Tripto, N.I., Venkatraman, S., Macko, D., Moro, R., Srba, I., Uchendu, A., Le, T., & Lee, D. (2024). A Ship of Theseus: Curious Cases of Paraphrasing in LLM-Generated Texts. *ACL 2024*.