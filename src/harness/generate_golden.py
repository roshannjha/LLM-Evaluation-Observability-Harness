from __future__ import annotations

import argparse
import json
from collections import Counter

from harness.config import GOLDEN_DIR, REPORTS_DIR, settings
from harness.embeddings import get_embeddings
from harness.ingest import load_corpus
from harness.config import CORPUS_DIR


def generate(testset_size: int, max_docs: int | None = None,
             truncate_chars: int | None = None) -> list[dict]:
    from ragas.testset import TestsetGenerator
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    from harness.llm import get_chat_model

    docs = load_corpus(CORPUS_DIR)
    if max_docs:
        docs = docs[:max_docs]
    if truncate_chars:
        for d in docs:
            d.page_content = d.page_content[:truncate_chars]
    total_chars = sum(len(d.page_content) for d in docs)
    print(f"Loaded {len(docs)} filings ({total_chars:,} chars); building RAGAS knowledge "
          f"graph (this calls the LLM repeatedly — API cost/quota applies)...")

    generator = TestsetGenerator(
        llm=LangchainLLMWrapper(get_chat_model(model=settings.judge_model)),
        embedding_model=LangchainEmbeddingsWrapper(get_embeddings()),
    )
    dataset = generator.generate_with_langchain_docs(docs, testset_size=testset_size)
    df = dataset.to_pandas()

    triplets = []
    for _, row in df.iterrows():
        triplets.append(
            {
                "question": row["user_input"],
                "contexts": list(row["reference_contexts"]),
                "ground_truth": row["reference"],
                "synthesizer": row.get("synthesizer_name", ""),
            }
        )
    return triplets


def write_outputs(triplets: list[dict]) -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    out = GOLDEN_DIR / "golden_dataset.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for t in triplets:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"Wrote {len(triplets)} triplets -> {out}")

    by_type = Counter(t["synthesizer"] for t in triplets)
    avg_ctx = sum(len(" ".join(t["contexts"])) for t in triplets) / max(len(triplets), 1)
    stats = REPORTS_DIR / "golden_stats.md"
    lines = [
        "# Golden Dataset Stats",
        "",
        f"- Total triplets: **{len(triplets)}**",
        f"- Avg context length: **{avg_ctx:,.0f}** chars",
        f"- Generator/judge model: `{settings.judge_model}` (backend `{settings.llm_backend}`)",
        "",
        "## By synthesizer (question type)",
        "",
        "| Synthesizer | Count |",
        "|---|---|",
    ]
    for name, n in by_type.most_common():
        lines.append(f"| {name or '(unknown)'} | {n} |")
    stats.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote stats -> {stats}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num", type=int, default=30, help="number of triplets to synthesize")
    ap.add_argument("--max-docs", type=int, default=None,
                    help="cap source filings (controls KG build cost/quota)")
    ap.add_argument("--truncate-chars", type=int, default=None,
                    help="truncate each filing to N chars (fewer KG nodes -> fewer calls)")
    args = ap.parse_args()
    triplets = generate(args.num, max_docs=args.max_docs, truncate_chars=args.truncate_chars)
    write_outputs(triplets)
    print("\nSpot-check the first 2 triplets:")
    for t in triplets[:2]:
        print(f"\n  Q: {t['question']}")
        print(f"  GT: {t['ground_truth'][:120]}...")


if __name__ == "__main__":
    main()
