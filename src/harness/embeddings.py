from __future__ import annotations

from langchain_core.embeddings import Embeddings

from harness.config import settings


def get_embeddings() -> Embeddings:
    if settings.embedding_backend == "local":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=settings.local_embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )

    if settings.embedding_backend == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.openai_embedding_model, api_key=settings.openai_api_key
        )

    raise ValueError(f"Unknown embedding backend: {settings.embedding_backend}")
