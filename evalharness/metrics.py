"""String and embedding metrics for comparing a prediction to a reference."""
from __future__ import annotations

import math
import re
import string
from collections import Counter

_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    text = text.lower().translate(_PUNCT)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize(pred) == normalize(gold) else 0.0


def contains(pred: str, gold: str) -> float:
    g = normalize(gold)
    return 1.0 if g and g in normalize(pred) else 0.0


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def embedding_similarity(embeddings):
    """Return a metric fn (pred, gold) -> cosine similarity of their embeddings."""
    def metric(pred: str, gold: str) -> float:
        return cosine(embeddings.embed(pred), embeddings.embed(gold))
    return metric
