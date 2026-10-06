from __future__ import annotations

from harness.config import settings


def _require(value: str | None, name: str) -> str:
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to .env — generation/eval phases need an API key."
        )
    return value


def get_chat_model(model: str | None = None, temperature: float = 0.0):
    model = model or settings.generation_model

    if settings.llm_backend == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=_require(settings.google_api_key, "GOOGLE_API_KEY"),
        )

    from langchain_openai import ChatOpenAI

    if settings.llm_backend == "grok":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=_require(settings.xai_api_key, "XAI_API_KEY"),
            base_url=settings.xai_base_url,
        )

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=_require(settings.openai_api_key, "OPENAI_API_KEY"),
    )
