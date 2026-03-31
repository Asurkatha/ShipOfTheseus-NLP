"""
Centralized metrics module for the Ship of Theseus project.
All similarity and evaluation metrics are defined here for consistency.

IMPORTANT NOTES:
- BLEU and ROUGE are LEXICAL metrics. They measure surface overlap only.
  Never use them as evidence of semantic preservation.
- BERTScore and SBERT cosine are SEMANTIC metrics.
- For attribution evaluation, use macro F1 to handle class imbalance.
"""

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


# ── Lexical Metrics ──

_smoother = SmoothingFunction().method1
_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def bleu(reference: str, hypothesis: str) -> float:
    """Sentence-level BLEU with smoothing. Returns value in [0, 1]."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    try:
        return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=_smoother)
    except Exception:
        return 0.0


def rouge_l(reference: str, hypothesis: str) -> float:
    """ROUGE-L F1 score. Returns value in [0, 1]."""
    try:
        return _rouge.score(reference, hypothesis)["rougeL"].fmeasure
    except Exception:
        return 0.0


def rouge_1(reference: str, hypothesis: str) -> float:
    """ROUGE-1 F1 score."""
    try:
        return _rouge.score(reference, hypothesis)["rouge1"].fmeasure
    except Exception:
        return 0.0


def ngram_overlap(text_a: str, text_b: str, n: int = 2) -> float:
    """
    Compute n-gram overlap (Jaccard similarity) between two texts.
    Returns value in [0, 1].
    """
    def get_ngrams(text, n):
        tokens = text.lower().split()
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    ngrams_a = get_ngrams(text_a, n)
    ngrams_b = get_ngrams(text_b, n)
    
    if not ngrams_a or not ngrams_b:
        return 0.0
    
    intersection = ngrams_a & ngrams_b
    union = ngrams_a | ngrams_b
    return len(intersection) / len(union)


# ── Semantic Metrics ──

def bertscore_batch(references: list, hypotheses: list,
                    model_type: str = None) -> dict:
    """
    Compute BERTScore for a batch.
    Returns dict with 'precision', 'recall', 'f1' lists.
    """
    from bert_score import score as bs_score
    
    kwargs = {"lang": "en", "verbose": False, "batch_size": 32}
    if model_type:
        kwargs["model_type"] = model_type
    
    P, R, F1 = bs_score(hypotheses, references, **kwargs)
    return {
        "precision": P.tolist(),
        "recall": R.tolist(),
        "f1": F1.tolist(),
    }


def sbert_cosine(texts_a: list, texts_b: list,
                 model_name: str = "all-MiniLM-L6-v2") -> list:
    """
    Compute SBERT cosine similarity between paired text lists.
    Returns list of similarity scores in [-1, 1].
    """
    from sentence_transformers import SentenceTransformer, util
    
    model = SentenceTransformer(model_name)
    emb_a = model.encode(texts_a, convert_to_tensor=True, show_progress_bar=False)
    emb_b = model.encode(texts_b, convert_to_tensor=True, show_progress_bar=False)
    
    cosines = util.cos_sim(emb_a, emb_b)
    return cosines.diagonal().tolist()


# ── Vector Similarity ──

def batch_cosine_similarity(A, B):
    """
    Row-wise cosine similarity between two (n, d) arrays.

    Returns a 1-D array of length n.  Rows where either vector is
    all-zero produce NaN (undefined direction).
    """
    import numpy as np
    from numpy.linalg import norm

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    norms_a = norm(A, axis=1)
    norms_b = norm(B, axis=1)

    zero_mask = (norms_a == 0) | (norms_b == 0)

    safe_a = np.where(norms_a[:, None] == 0, 1, norms_a[:, None])
    safe_b = np.where(norms_b[:, None] == 0, 1, norms_b[:, None])

    sim = np.sum((A / safe_a) * (B / safe_b), axis=1)
    sim[zero_mask] = np.nan
    return sim


# ── Feature Metrics (for Stylometry) ──

def type_token_ratio(text: str) -> float:
    """Lexical diversity: unique words / total words."""
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def hapax_legomena_ratio(text: str) -> float:
    """Proportion of words that appear exactly once."""
    words = text.lower().split()
    if not words:
        return 0.0
    from collections import Counter
    freq = Counter(words)
    hapax = sum(1 for w, c in freq.items() if c == 1)
    return hapax / len(words)


# ── Attribution Metrics ──

def classification_report_dict(y_true, y_pred, labels=None):
    """Wrapper for sklearn classification report as dict."""
    from sklearn.metrics import classification_report
    return classification_report(y_true, y_pred, labels=labels, output_dict=True)


def compute_attribution_decay(model, X_by_tier: dict, y_by_tier: dict) -> dict:
    """
    Evaluate an attribution model across tiers.
    
    Args:
        model: Trained sklearn classifier
        X_by_tier: {"T1": features, "T2": features, "T3": features}
        y_by_tier: {"T1": labels, "T2": labels, "T3": labels}
    
    Returns:
        dict: {"T1": {"accuracy": ..., "macro_f1": ...}, ...}
    """
    from sklearn.metrics import accuracy_score, f1_score
    
    results = {}
    for tier in ["T1", "T2", "T3"]:
        if tier not in X_by_tier:
            continue
        y_pred = model.predict(X_by_tier[tier])
        results[tier] = {
            "accuracy": accuracy_score(y_by_tier[tier], y_pred),
            "macro_f1": f1_score(y_by_tier[tier], y_pred, average="macro"),
        }
    return results