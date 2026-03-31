"""
Named-entity extraction for content-erasure analysis.

Uses spaCy to extract the set of lowercased named-entity strings per text.
"""

import spacy
from tqdm import tqdm


def extract_ner_sets(texts, model_name="en_core_web_sm", batch_size=500):
    """
    Extract lowercased named-entity string sets for a list of texts.

    Returns list[frozenset[str]] of length len(texts).
    Empty texts yield empty frozensets.
    """
    nlp = spacy.load(model_name, disable=["tagger", "parser", "lemmatizer"])
    nlp.max_length = 2_000_000

    texts_list = [t if isinstance(t, str) else "" for t in texts]
    results = []

    docs = nlp.pipe(texts_list, batch_size=batch_size)
    for doc in tqdm(docs, total=len(texts_list), desc="NER extraction"):
        entities = frozenset(ent.text.lower() for ent in doc.ents)
        results.append(entities)

    return results


def entity_metrics(t0_set, t1_set):
    """
    Compute Jaccard, Recall (retention), and Precision (fidelity)
    between T0 and T1 entity sets.

    Returns (jaccard, recall, precision) as floats, or (NaN, NaN, NaN)
    when T0 has zero entities.
    """
    import math
    if len(t0_set) == 0:
        return (math.nan, math.nan, math.nan)
    union = t0_set | t1_set
    inter = t0_set & t1_set
    jaccard = len(inter) / len(union) if len(union) > 0 else math.nan
    recall = len(inter) / len(t0_set)
    precision = len(inter) / len(t1_set) if len(t1_set) > 0 else math.nan
    return (jaccard, recall, precision)
