from __future__ import annotations

import sys
from dataclasses import dataclass, field

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from harness.config import settings
from harness.embeddings import get_embeddings

SYSTEM_PROMPT = (
    "You are a financial-analysis assistant answering questions about NIKE's SEC "
    "filings. Answer ONLY from the provided context. If the context does not contain "
    "the answer, say you don't have enough information. Cite figures exactly as written "
    "and name the fiscal year. Be concise."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
    ]
)


@dataclass
class RagResult:
    question: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_dir),
        embedding_function=get_embeddings(),
    )


def get_retriever(k: int | None = None):
    return get_vectorstore().as_retriever(search_kwargs={"k": k or settings.top_k})


def _format_context(docs: list[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        tag = f"[{i}] {d.metadata.get('form', '?')} {d.metadata.get('reporting_date', '')}".strip()
        blocks.append(f"{tag}\n{d.page_content}")
    return "\n\n".join(blocks)


def retrieve(question: str, k: int | None = None) -> list[Document]:
    return get_retriever(k).invoke(question)


def answer(question: str, k: int | None = None) -> RagResult:
    from harness.llm import get_chat_model

    docs = retrieve(question, k)
    chain = PROMPT | get_chat_model() | StrOutputParser()
    generated = chain.invoke({"context": _format_context(docs), "question": question})
    return RagResult(
        question=question,
        answer=generated,
        contexts=[d.page_content for d in docs],
        sources=[d.metadata for d in docs],
    )


def main() -> None:
    question = " ".join(sys.argv[1:]) or "What was NIKE's total revenue in fiscal 2026?"
    result = answer(question)
    print(f"Q: {result.question}\n")
    print(f"A: {result.answer}\n")
    print("Sources:")
    for s in result.sources:
        print(f"  - {s.get('source')}")


if __name__ == "__main__":
    main()
