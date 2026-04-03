# Ship of Theseus: Computational Forensics and the Paradox of Authorial Identity

**CS6120 Natural Language Processing — Northeastern University, Spring 2026**\
**Team:** Rohan Joshi, Aarav Surkatha

If you replace every plank of a ship one by one, is it still the same ship? Analogously, if an LLM iteratively paraphrases a text — replacing every word, restructuring every sentence, shedding every proper noun — while preserving meaning, is it still the original author's work?

This project uses the [Ship of Theseus Paraphrased Corpus](https://github.com/tripto03/Ship_of_theseus_paraphrased_copus) to forensically track how authorial identity decays through iterative paraphrasing (T0 -> T3) across **7 datasets**, **7 paraphraser variants** (4 model families), and **3 paraphrase iterations**.

![Project Cover Image](Cover_Photo.png)

## Research Questions

| RQ | Question | Method | Key Finding |
|----|----------|--------|-------------|
| **RQ1** | Which linguistic features decay first — style or content? | POS cosine, NER recall, BLEU, BERTScore across T0-T3 | Content (NER) decays **11.8x** faster than style (POS); layered erosion: lexical > entities > grammar > semantics |
| **RQ2** | At what iteration does authorship attribution fail? | Train classifiers on T0, evaluate on T1-T3 | Aggressive paraphrasers (Dipper High) push F1 below 0.5 at T1; conservative ones (Pegasus Slight) survive through T3 |
| **RQ3** | Do different paraphrasers leave identifiable fingerprints? | 6-class classifier on T0->T1 delta features | SVM achieves **75.5% macro F1** at paraphraser identification; NER metrics are the most discriminative features |

## Key Results

- **Semantic meaning preserved:** BERTScore stays 0.80-0.98 across all tiers while BLEU drops as low as 0.002
- **Layered erosion:** Feature ordering BERTScore > POS cosine > NER Recall > ROUGE-L > BLEU holds at every tier
- **5.8x sensitivity gap:** Content features (NER Recall delta: 0.610) are 5.8x more sensitive to paraphrasing intensity than stylistic features (POS cosine delta: 0.104) in the Dipper Low vs High controlled experiment
- **Entities are dropped, not replaced:** Pegasus coverage experiment shows Recall drops 0.329 while Precision drops only 0.122
- **All key comparisons statistically significant** (paired permutation tests, p < 0.001)

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

# Preprocess and cache the corpus
python -m src.data.prepare_corpus
```

## Running the Full Pipeline

```bash
# Step 1: Extract features (POS, NER, dep depth) and compute multi-tier metrics
python scripts/run_multitier_analysis.py

# Step 2: Generate decay curve figures (with 95% bootstrap CIs)
python scripts/generate_decay_curves.py

# Step 3: Generate t-SNE identity trajectory (with cross-validated k-NN purity)
python scripts/generate_tsne.py

# Step 4: Generate domain-specific analysis figures
python scripts/generate_domain_analysis.py

# Step 5: Run statistical significance tests
python scripts/run_significance_tests.py

# Step 6: Launch the interactive dashboard
streamlit run streamlit_app.py
```

**Note:** BERTScore computation requires a CUDA GPU. Notebook 02 includes a Google Colab section for this. Run it there, download the CSV, and place it in `experiments/baseline_results/`.

## Repository Structure

```
ShipOfTheseus-NLP/
├── README.md
├── requirements.txt
├── streamlit_app.py                  # Interactive Decay Dashboard (4 tabs)
│
├── data/
│   ├── raw/                          # Cloned corpus (7 datasets x 7 paraphrasers)
│   └── processed/                    # Cached parquets, pickles, feature vectors
│       ├── corpus_full.parquet       # Full corpus with preprocessing metadata
│       ├── multitier_results.pkl     # Per-paraphraser DataFrames with all metrics
│       ├── paired_all.pkl            # Forensic pivot (T0 + 7 T1 variants per row)
│       ├── pos_vectors.npz           # Cached POS tag vectors
│       ├── ner_sets.pkl              # Cached NER entity sets
│       └── sbert_embeddings.pkl      # Cached SBERT embeddings
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA: corpus profiling, distributions
│   ├── 02_baseline_similarity.ipynb  # BLEU, ROUGE-L, BERTScore baselines
│   ├── 03_stylometry_analysis.ipynb  # POS distributions, decay curves
│   ├── 04_semantic_similarity.ipynb  # SBERT embeddings, identity trajectories
│   └── 05_attribution_models.ipynb   # RQ2 attribution + RQ3 fingerprints
│
├── src/
│   ├── data/
│   │   ├── load_data.py              # Corpus loading, pairing, forensic pivot
│   │   ├── preprocess.py             # Text cleaning, validation, deduplication
│   │   └── prepare_corpus.py         # One-time preprocessing to parquet
│   ├── features/
│   │   ├── ner.py                    # Named entity extraction and set metrics
│   │   └── pos.py                    # POS tag distribution vectors (17 UPOS)
│   ├── attribution/
│   │   ├── classifier.py             # Attribution (RQ2) and fingerprint (RQ3) classifiers
│   │   └── feature_builder.py        # 23-dim feature vectors + 24-dim delta features
│   └── utils/
│       ├── config.py                 # Paths, constants, dataset/paraphraser definitions
│       └── metrics.py                # BLEU, ROUGE, BERTScore, SBERT, bootstrap CI, permutation tests
│
├── scripts/
│   ├── run_multitier_analysis.py     # Master pipeline: extract features, compute metrics
│   ├── run_full_extraction.py        # POS + NER extraction pipeline
│   ├── generate_decay_curves.py      # RQ1 composite decay figures (with bootstrap CIs)
│   ├── generate_tsne.py              # Identity trajectory t-SNE (with CV k-NN purity)
│   ├── generate_domain_analysis.py   # Domain-specific heatmaps and decay curves
│   ├── generate_presentation_figures.py  # Publication-quality radar/scatter plots
│   └── run_significance_tests.py     # Paired permutation tests for all key claims
│
├── figures/
│   ├── baselines/                    # Composite decay, family comparison
│   ├── eda/                          # Distributions, punctuation, word counts
│   ├── stylometry/                   # POS cosine decay, tag deltas
│   ├── ner/                          # Entity recall curves, survival funnels
│   ├── fingerprints/                 # Attribution decay, confusion matrix, t-SNE, feature importance
│   └── domain/                       # Domain heatmaps, per-dataset decay curves
│
├── experiments/
│   └── baseline_results/             # Similarity baseline CSVs
│
├── paper/
│   ├── acl2026.tex                   # Main LaTeX file (ACM sigconf two-column)
│   ├── sections/                     # Modular .tex sections (intro through conclusion)
│   └── references.bib                # ACL-style bibliography
│
└── project_requirements/             # Course assignment PDFs
```

## Methodology

### Feature Extraction
- **Stylometric (17-dim):** POS tag proportions via spaCy (en_core_web_sm), normalized to UPOS
- **Content (set-based):** Named entities (PERSON, ORG, GPE, DATE, etc.) as lowercased sets
- **Structural (1-dim):** Mean dependency tree depth per document
- **Lexical:** BLEU (smoothed n-gram precision), ROUGE-L (LCS F1)
- **Semantic:** BERTScore (DeBERTa-xlarge-MNLI), SBERT cosine (all-MiniLM-L6-v2)

### Classification Models
| Model | Use | Config |
|-------|-----|--------|
| Logistic Regression | RQ2 attribution | max_iter=1000, balanced class weight |
| SVM (RBF) | RQ2 + RQ3 (best performer) | balanced class weight, probability=True |
| Random Forest | RQ3 feature importance | 200 trees, balanced, n_jobs=-1 |
| DummyClassifier (3 variants) | Baselines for RQ3 | random, most-frequent, stratified |

### Statistical Validation
- **95% bootstrap CIs** (1,000 resamples) on all reported means
- **5-fold stratified CV** on T0 for attribution classifiers
- **GroupShuffleSplit** for RQ3 to prevent document-level leakage across paraphrasers
- **Paired permutation tests** (10,000 permutations) for key comparisons

## Streamlit Dashboard

Launch with `streamlit run streamlit_app.py`. Four tabs:

1. **Decay Explorer** — Interactive feature decay curves with error bands. Select paraphrasers, metrics, and datasets.
2. **Document Forensics** — Side-by-side T0 vs T1/T2/T3 text comparison with color-coded entity highlighting (retained/dropped/novel). Includes T0-vs-T3 focused comparison with per-document BLEU, ROUGE-L, and entity retention.
3. **Attribution Lab** — Binary and 7-class attribution F1 decay, "Point of No Return" table, RQ3 paraphraser identification with confusion matrix and feature importance.
4. **Research Gallery** — All publication figures organized by category.

## Datasets

| Dataset | Domain | Register | Avg Entities/Doc |
|---------|--------|----------|-----------------|
| CMV | Arguments | Formal | 7.6 |
| ELI5 | Explanations | Informal | 6.2 |
| SCI_GEN | Scientific | Formal | 10.2 |
| TLDR | Summaries | Informal | 9.0 |
| WP | Fiction | Narrative | 7.3 |
| XSum | News | Formal | 18.7 |
| Yelp | Reviews | Informal | 5.9 |

## Paraphrasers

| Paraphraser | Family | Type | Aggressiveness |
|-------------|--------|------|---------------|
| Pegasus (Slight) | Pegasus | 25% sentence coverage | Most conservative |
| Dipper (Low) | Dipper | Low diversity | Conservative |
| PaLM2 | PaLM | Full-text | Moderate |
| ChatGPT | GPT | Full-text | Moderate-high |
| Pegasus (Full) | Pegasus | 100% sentence coverage | Moderate-high |
| Dipper (Default) | Dipper | Default diversity | High |
| Dipper (High) | Dipper | High diversity | Most aggressive |

## Deliverables

- [x] 8-page ACL-format research paper (`paper/acl2026.tex`)
- [x] Reproducible code repository with documented pipeline
- [x] Streamlit "Decay Dashboard" with live style-drift visualization
- [x] Identity Trajectory t-SNE with quantitative backing (silhouette, k-NN purity, centroid distance)
- [x] Statistical significance tests for all key claims

## References

- Tripto, N.I., et al. (2024). A Ship of Theseus: Curious Cases of Paraphrasing in LLM-Generated Texts. *ACL 2024*.
- Krishna, K., et al. (2023). Paraphrasing Evades Detectors of AI-Generated Text, but Retrieval is an Effective Defense. *NeurIPS 2023*.
- Zhang, T., et al. (2020). BERTScore: Evaluating Text Generation with BERT. *ICLR 2020*.
- Mitchell, E., et al. (2023). DetectGPT: Zero-Shot Machine-Generated Text Detection. *ICML 2023*.
- Stamatatos, E. (2009). A Survey of Modern Authorship Attribution Methods. *JASIST*.
