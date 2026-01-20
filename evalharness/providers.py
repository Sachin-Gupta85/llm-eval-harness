"""Model, embeddings, and judge abstractions with offline fakes."""
from __future__ import annotations

import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class Model(Protocol):
    def generate(self, prompt: str) -> str:
        ...


@runtime_checkable
class Judge(Protocol):
    def score(self, question: str, answer: str, reference: str) -> float:
        """Return a 0..1 quality score for an answer vs a reference."""
        ...


class MappingModel:
    """A fake model that returns canned answers keyed by example input (tests/demos)."""

    def __init__(self, answers: dict[str, str], default: str = ""):
        self.answers = answers
        self.default = default

    def generate(self, prompt: str) -> str:
        return self.answers.get(prompt, self.default)


class HashingEmbeddings:
    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            vec[hash(tok) % self.dim] += 1.0
        return vec


class KeywordJudge:
    """A deterministic judge: fraction of reference keywords present in the answer."""

    def score(self, question: str, answer: str, reference: str) -> float:
        ref = set(re.findall(r"[a-z0-9]+", reference.lower()))
        ans = set(re.findall(r"[a-z0-9]+", answer.lower()))
        return len(ref & ans) / len(ref) if ref else 0.0


class OpenAIModel:
    def __init__(self, model: str = "gpt-4o-mini"):
        from langchain_openai import ChatOpenAI  # lazy

        self._chat = ChatOpenAI(model=model, temperature=0)

    def generate(self, prompt: str) -> str:
        return self._chat.invoke(prompt).content
