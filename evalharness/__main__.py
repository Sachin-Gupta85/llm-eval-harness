"""Offline demo: evaluate a canned model over a tiny dataset."""
from __future__ import annotations

from .metrics import contains, embedding_similarity, exact_match, token_f1
from .providers import HashingEmbeddings, MappingModel
from .report import to_markdown
from .runner import EvalRunner

DATASET = [
    {"id": "q1", "input": "capital of france?", "expected": "Paris"},
    {"id": "q2", "input": "2+2?", "expected": "4"},
    {"id": "q3", "input": "largest planet?", "expected": "Jupiter"},
]


def main():
    model = MappingModel({
        "capital of france?": "Paris",
        "2+2?": "it is 4",
        "largest planet?": "Saturn",
    })
    metrics = {
        "exact_match": exact_match,
        "token_f1": token_f1,
        "contains": contains,
        "embed_sim": embedding_similarity(HashingEmbeddings()),
    }
    results = EvalRunner(model, metrics).run(DATASET)
    print(to_markdown(results))


if __name__ == "__main__":
    main()
