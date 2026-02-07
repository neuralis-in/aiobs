from __future__ import annotations

import math
from typing import Any, Optional, Type

from ..base import BaseEval
from ..models import (
    EvalInput,
    EvalResult,
    EvalStatus,
    AssertionDetail,
    SemanticSimilarityConfig,
)
from ...llm import BaseLLM


class SemanticSimilarityEval(BaseEval):
    """
    Evaluator that compares model output and expected output
    using embedding-based cosine similarity.

    This evaluator relies on the LLM abstraction to generate embeddings,
    making it fully provider-agnostic.
    """

    name: str = "semantic_similarity"
    description: str = "Embedding-based semantic similarity comparison"
    config_class: Type[SemanticSimilarityConfig] = SemanticSimilarityConfig

    def __init__(self, config: Optional[SemanticSimilarityConfig] = None) -> None:
        super().__init__(config)
        self.config: SemanticSimilarityConfig = self.config

    def evaluate(self, eval_input: EvalInput, **kwargs: Any) -> EvalResult:
        try:
            llm: Optional[BaseLLM] = kwargs.get("llm")
            if llm is None:
                return EvalResult.error_result(
                    self.eval_name,
                    ValueError(
                        "LLM must be provided via kwargs: "
                        "evaluate(..., llm=LLM.from_client(...))"
                    ),
                )

            # Get expected output
            expected = kwargs.get("expected", eval_input.expected_output)
            if expected is None:
                return EvalResult.error_result(
                    self.eval_name,
                    ValueError("expected_output is required for semantic similarity evaluation"),
                )

            output = eval_input.model_output

            emb_expected = llm.embed(expected)
            emb_output = llm.embed(output)

            similarity = self._cosine_similarity(emb_expected, emb_output)
            passed = similarity >= self.config.threshold

            message = (
                f"Semantic similarity: {similarity:.4f} "
                f"(threshold: {self.config.threshold})"
            )

            assertions = [
                AssertionDetail(
                    name="semantic_similarity",
                    passed=passed,
                    expected=expected[:200],
                    actual=output[:200],
                    message=message,
                )
            ]

            details = {
                "similarity": similarity,
                "threshold": self.config.threshold,
            }

            return EvalResult(
                status=EvalStatus.PASSED if passed else EvalStatus.FAILED,
                score=similarity,
                eval_name=self.eval_name,
                message=message,
                assertions=assertions if self.config.include_details else None,
                details=details if self.config.include_details else None,
            )

        except Exception as e:
            return EvalResult.error_result(self.eval_name, e)

    @staticmethod
    def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        return dot / (norm1 * norm2) if norm1 and norm2 else 0.0
