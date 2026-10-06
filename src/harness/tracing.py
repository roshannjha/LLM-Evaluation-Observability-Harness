from __future__ import annotations

from dataclasses import asdict

from harness.config import settings
from harness.rag import RagResult, answer, retrieve


def _langfuse():
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.LANGFUSE_BASE_URL,
    )


def classify_failure(context_recall: float, faithfulness: float) -> str:
    if context_recall < settings.faithfulness_min:
        return "retrieval_failure"
    if faithfulness < settings.faithfulness_min:
        return "generation_failure"
    return "ok"


def traced_answer(question: str, context_recall: float | None = None,
                  faithfulness: float | None = None) -> RagResult:
    lf = _langfuse()
    if lf is None:
        return answer(question)

    tags = {}
    if context_recall is not None and faithfulness is not None:
        tags = {
            "failure_type": classify_failure(context_recall, faithfulness),
            "context_recall": context_recall,
            "faithfulness": faithfulness,
        }

    with lf.start_as_current_observation(name="rag_query", as_type="span") as root:
        with lf.start_as_current_observation(name="retrieval", as_type="span") as rspan:
            docs = retrieve(question)
            rspan.update(
                input={"question": question},
                output={"n_docs": len(docs),
                        "sources": [d.metadata.get("source") for d in docs]},
            )

        with lf.start_as_current_observation(name="generation", as_type="span") as gspan:
            result = answer(question)
            gspan.update(input={"n_context_docs": len(docs)},
                         output={"answer": result.answer})

        root.update(input={"question": question}, output=asdict(result), metadata=tags)

    lf.flush()
    return result
