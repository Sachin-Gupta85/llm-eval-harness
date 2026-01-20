"""Run a model over a dataset and score each output with a set of metrics."""
from __future__ import annotations

from statistics import mean

from .providers import Model


class EvalRunner:
    def __init__(self, model: Model, metrics: dict):
        """metrics: {name: fn(pred, gold) -> float}."""
        self.model = model
        self.metrics = metrics

    def run(self, dataset: list[dict]) -> dict:
        """dataset: list of {"id", "input", "expected"}."""
        rows = []
        for ex in dataset:
            pred = self.model.generate(ex["input"])
            scores = {name: fn(pred, ex["expected"]) for name, fn in self.metrics.items()}
            rows.append({"id": ex["id"], "prediction": pred, "scores": scores})
        aggregates = {
            name: round(mean(r["scores"][name] for r in rows), 4) if rows else 0.0
            for name in self.metrics
        }
        return {"rows": rows, "aggregates": aggregates, "n": len(rows)}
