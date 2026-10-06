# LLM Evaluation & Observability Harness

A production-style test harness that treats a RAG application like software:
**synthetic golden data**, **automated quality assertions**, **CI quality gates**,
and **distributed tracing** that separates *retrieval* failures from *generation*
failures. Built over a corpus of **NIKE, Inc.** SEC filings (10-K / 10-Q).

## What it does

1. **Synthetic golden data (RAGAS).** Generates `(question, contexts, ground_truth)`
   triplets over the filings — no manual labeling. → [generate_golden.py](src/harness/generate_golden.py)
2. **Quality assertions (DeepEval).** Faithfulness, toxicity, answer relevancy, context
   recall/precision over the golden set. → [metrics.py](src/harness/metrics.py)
3. **CI gate (GitHub Actions).** Fails the build if **faithfulness < 0.90** or
   **toxicity > 0.05**. → [eval-gate.yml](.github/workflows/eval-gate.yml)
4. **Tracing (Langfuse).** Per-request `retrieval` vs. `generation` spans and a failure
   taxonomy that tells you *which* stage broke. → [tracing.py](src/harness/tracing.py)

## Architecture

```
SEC filings ──ingest──> Chroma ──retrieve──> LangChain RAG ──> answer + contexts
     │                                                              │
     └──RAGAS──> golden triplets ──> DeepEval metrics ──> CI gate (0.90 / 0.05)
                                            │
                                       Langfuse traces (retrieval vs generation)
```

## Quickstart

```bash
pip install -e ".[eval]"

# 1. Corpus (already fetched into data/corpus; re-run to refresh)
python scripts/download_corpus.py --csv data/edgar_filings.csv --num-10k 3 --num-10q 3

# 2. Ingest → Chroma (local embeddings, no API key)
python -m harness.ingest --reset

# 3. Add your key
cp .env.example .env    # then set XAI_API_KEY

# 4. Generate golden data (RAGAS) and run the gate
python -m harness.generate_golden --num 30
python scripts/run_eval.py        # exits non-zero if a gate fails
pytest                            # same gates, dev-friendly
```

Default mode uses **local HuggingFace embeddings** (zero cost, no key); an LLM key
(Grok/xAI by default) is only needed for generation and the RAGAS/DeepEval judge.

## Layout

| Path | Purpose |
|---|---|
| [src/harness/config.py](src/harness/config.py) | Central config + quality-gate thresholds |
| [src/harness/ingest.py](src/harness/ingest.py) | Load → chunk → embed → Chroma |
| [src/harness/rag.py](src/harness/rag.py) | LangChain RAG chain (answer + contexts) |
| [src/harness/generate_golden.py](src/harness/generate_golden.py) | RAGAS synthetic testset |
| [src/harness/metrics.py](src/harness/metrics.py) | DeepEval metrics + Grok judge adapter |
| [src/harness/tracing.py](src/harness/tracing.py) | Langfuse spans + failure taxonomy |
| [scripts/run_eval.py](scripts/run_eval.py) | Batch eval + CI gate |

