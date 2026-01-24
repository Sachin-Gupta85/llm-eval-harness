from evalharness.metrics import normalize, exact_match, contains, token_f1, embedding_similarity
from evalharness.providers import HashingEmbeddings


def test_normalize():
    assert normalize("  Hello, World! ") == "hello world"


def test_exact_and_contains():
    assert exact_match("Paris.", "paris") == 1.0
    assert exact_match("London", "Paris") == 0.0
    assert contains("The answer is Paris today", "paris") == 1.0
    assert contains("nope", "paris") == 0.0


def test_token_f1():
    assert token_f1("a b c", "a b c") == 1.0
    assert token_f1("a b", "a b c d") == pytest_approx(2 * (1.0) * (0.5) / 1.5)
    assert token_f1("x", "y") == 0.0


def pytest_approx(x):
    import pytest
    return pytest.approx(x)


def test_embedding_similarity_self_is_one():
    m = embedding_similarity(HashingEmbeddings())
    import pytest
    assert m("hybrid retrieval", "hybrid retrieval") == pytest.approx(1.0)
    assert m("totally different", "hybrid retrieval") < 0.5
