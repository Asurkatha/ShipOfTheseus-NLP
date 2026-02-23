# Notebooks

Analysis notebooks for the Ship of Theseus project. Run them in order.

## Prerequisites

1. Clone the raw corpus (see root README)
2. Run `python -m src.data.prepare_corpus` to generate cached parquets
3. For BERTScore: either a CUDA GPU locally, or use the Google Colab section in Notebook 02

## Notebooks

| # | Notebook | What it does |
|---|----------|-------------|
| 01 | `01_data_exploration.ipynb` | Corpus profiling, text characteristics by tier, vocabulary analysis, punctuation patterns, source author profiles, per-dataset variation |
| 02 | `02_baseline_similarity.ipynb` | BLEU, ROUGE-L, BERTScore baselines across all 7 paraphrasers. Includes Colab fallback for GPU-less machines. Generates decay curves. |
| 03 | `03_stylometry_analysis.ipynb` | *(Phase II)* POS frequencies, dependency depth, sentence length variance |
| 04 | `04_semantic_similarity.ipynb` | *(Phase II)* SBERT cosine similarity and embedding trajectories |
| 05 | `05_attribution_models.ipynb` | *(Phase III)* Authorship attribution and paraphraser identification |

Notebooks 03-05 are coming in the next project phases.

## Output

- Figures are saved to `figures/eda/` and `figures/decay_curves/`
- Experiment results are saved to `experiments/baseline_results/`
- All plots are saved at 150 DPI for use in the paper and presentation

## Kernel Setup

If running locally with a virtual environment:

```bash
pip install ipykernel
python -m ipykernel install --user --name=theseus --display-name="Ship of Theseus"
```

Then select "Ship of Theseus" as the kernel in Jupyter.