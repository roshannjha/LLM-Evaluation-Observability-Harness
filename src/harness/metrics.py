from __future__ import annotations

from harness.config import settings


class LangChainJudge:
    def __init__(self):
        from deepeval.models.base_model import DeepEvalBaseLLM

        from harness.llm import get_chat_model

        self._DeepEvalBaseLLM = DeepEvalBaseLLM
        self._model = get_chat_model(model=settings.judge_model, temperature=0.0)

    def _build(self):
        base = self._DeepEvalBaseLLM
        model = self._model
        name = settings.judge_model

        class _Judge(base):
            def load_model(self):
                return model

            def generate(self, prompt: str) -> str:
                return model.invoke(prompt).content

            async def a_generate(self, prompt: str) -> str:
                res = await model.ainvoke(prompt)
                return res.content

            def get_model_name(self) -> str:
                return f"{settings.llm_backend}:{name}"

        return _Judge()


def judge():
    return LangChainJudge()._build()


def gating_metrics():
    from deepeval.metrics import FaithfulnessMetric, ToxicityMetric

    j = judge()
    return {
        "faithfulness": FaithfulnessMetric(
            threshold=settings.faithfulness_min, model=j, include_reason=False),
        "toxicity": ToxicityMetric(
            threshold=1.0 - settings.toxicity_max, model=j, include_reason=False),
    }


def soft_metrics():
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
    )

    j = judge()
    return {
        "answer_relevancy": AnswerRelevancyMetric(threshold=0.80, model=j, include_reason=False),
        "contextual_recall": ContextualRecallMetric(threshold=0.80, model=j, include_reason=False),
        "contextual_precision": ContextualPrecisionMetric(
            threshold=0.70, model=j, include_reason=False),
    }
