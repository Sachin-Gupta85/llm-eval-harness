# 📏 llm-eval-harness — provider-agnostic LLM evaluation

> Evaluate any model over a dataset with string metrics, **embedding similarity**, and an **LLM-as-judge**, then render a Markdown report. Runs fully offline with deterministic providers.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Evals](https://img.shields.io/badge/focus-LLM%20evals-6f42c1)
![Tests](https://img.shields.io/badge/tests-pytest%20(7)-brightgreen)

## 🧠 What it does
```
dataset ─▶ Model.generate ─▶ metrics{exact_match, token_f1, contains, embed_sim, judge} ─▶ aggregate ─▶ Markdown report
```
- **String metrics** — normalized exact match, SQuAD-style token F1, substring containment.
- **Embedding similarity** — cosine between prediction/reference embeddings.
- **LLM-as-judge** — a `Judge` protocol (keyword judge offline; real model in prod).
- **Runner + report** — per-example scores, aggregate means, Markdown table.

## 🧩 Swappable providers
`Model`, `Judge`, and embeddings are protocols. The offline fakes (`MappingModel`,
`KeywordJudge`, `HashingEmbeddings`) make **the whole harness run green with no
API key**; swap in `OpenAIModel` / real judge (LangChain, lazy import) for real runs.

## 🚀 Run
```bash
python -m evalharness            # offline demo: prints a Markdown report
```

## ✅ Test
```bash
pip install -r requirements-dev.txt && python -m pytest -q
```
Covers normalization, exact/contains/F1, embedding similarity, the runner's
aggregation, the judge, and report rendering.

## 📝 License
MIT — see [LICENSE](LICENSE).
