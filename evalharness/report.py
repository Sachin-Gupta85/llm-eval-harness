"""Render evaluation results as Markdown."""
from __future__ import annotations


def to_markdown(results: dict) -> str:
    lines = [f"# Evaluation report\n", f"Examples: **{results['n']}**\n", "## Aggregate metrics\n",
             "| Metric | Mean |", "|---|---|"]
    for name, value in sorted(results["aggregates"].items()):
        lines.append(f"| {name} | {value:.4f} |")
    lines.append("\n## Per-example\n")
    lines.append("| id | " + " | ".join(sorted(next(iter(results["rows"]), {}).get("scores", {}))) + " |")
    metric_names = sorted(results["aggregates"])
    lines[-1] = "| id | " + " | ".join(metric_names) + " |"
    lines.append("|---|" + "---|" * len(metric_names))
    for r in results["rows"]:
        cells = " | ".join(f"{r['scores'][m]:.2f}" for m in metric_names)
        lines.append(f"| {r['id']} | {cells} |")
    return "\n".join(lines)
