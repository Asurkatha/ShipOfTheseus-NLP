"""
POS-tag distribution extraction for stylistic drift analysis.

Uses spaCy to extract Universal POS (UPOS) tag proportions from text.
All 17 UPOS tags are always represented in the output vector.
"""

import numpy as np
import spacy
from tqdm import tqdm


UPOS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
    "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
    "SCONJ", "SYM", "VERB", "X",
]

_TAG_TO_IDX = {tag: i for i, tag in enumerate(UPOS_TAGS)}


def extract_pos_vectors(texts, model_name="en_core_web_sm", batch_size=500):
    """
    Extract normalised UPOS proportion vectors for a list of texts.

    Returns np.ndarray of shape (len(texts), 17) where each row sums to ~1.0.
    Rows with zero non-space tokens get an all-zero vector.
    """
    nlp = spacy.load(model_name, disable=["ner", "parser", "lemmatizer"])
    nlp.max_length = 2_000_000

    texts_list = [t if isinstance(t, str) else "" for t in texts]
    vectors = np.zeros((len(texts_list), len(UPOS_TAGS)), dtype=np.float64)

    docs = nlp.pipe(texts_list, batch_size=batch_size)
    for i, doc in enumerate(tqdm(docs, total=len(texts_list), desc="POS extraction")):
        total = 0
        for token in doc:
            if token.is_space:
                continue
            total += 1
            idx = _TAG_TO_IDX.get(token.pos_)
            if idx is not None:
                vectors[i, idx] += 1
        if total > 0:
            vectors[i] /= total

    return vectors
