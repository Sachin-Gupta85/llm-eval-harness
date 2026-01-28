from evalharness.metrics import exact_match, token_f1
from evalharness.providers import MappingModel
from evalharness.runner import EvalRunner
from evalharness.report import to_markdown

DATASET = [
    {"id": "a", "input": "q1", "expected": "paris"},
    {"id": "b", "input": "q2", "expected": "jupiter"},
]


def test_runner_aggregates():
    model = MappingModel({"q1": "paris", "q2": "saturn"})
    results = EvalRunner(model, {"em": exact_match, "f1": token_f1}).run(DATASET)
    assert results["n"] == 2
    assert results["aggregates"]["em"] == 0.5          # 1 correct of 2
    assert results["rows"][0]["scores"]["em"] == 1.0
    assert results["rows"][1]["scores"]["em"] == 0.0


def test_report_markdown():
    model = MappingModel({"q1": "paris", "q2": "jupiter"})
    results = EvalRunner(model, {"em": exact_match}).run(DATASET)
    md = to_markdown(results)
    assert "# Evaluation report" in md and "| em |" in md
    assert "| a |" in md and "| b |" in md


def test_judge():
    from evalharness.providers import KeywordJudge
    j = KeywordJudge()
    assert j.score("q", "the capital is paris france", "paris france") == 1.0
    assert j.score("q", "london", "paris france") == 0.0
