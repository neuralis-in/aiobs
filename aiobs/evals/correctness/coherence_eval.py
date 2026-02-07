from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Type, List

from ..base import BaseEval
from ..models import (
    EvalInput,
    EvalResult,
    EvalStatus,
    AssertionDetail,
    CoherenceEvalConfig,
)
from ...llm import LLM, BaseLLM


COHERENCE_JUDGE_SYSTEM_PROMPT = """You are an expert writing quality evaluator.

Evaluate how coherent, fluent, and well-structured the response is.
You are NOT checking factual correctness.
"""


COHERENCE_JUDGE_PROMPT = """Evaluate the coherence of this response.

User Input:
{user_input}

Model Output:
{model_output}

Score from 0.0 to 1.0.

Respond ONLY in JSON:
{{
  "score": <float>,
  "issues": ["list of issues"],
  "analysis": "brief explanation"
}}
"""


class CoherenceEval(BaseEval):

    name: str = "coherence_eval"
    description: str = "Evaluates coherence and fluency of model output"
    config_class: Type[CoherenceEvalConfig] = CoherenceEvalConfig

    def __init__(
        self,
        client: Any,
        model: str,
        config: Optional[CoherenceEvalConfig] = None,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(config)
        self.config: CoherenceEvalConfig = self.config

        self._llm: BaseLLM = LLM.from_client(
            client=client,
            model=model,
            temperature=temperature,
        )

    def _build_prompt(self, eval_input: EvalInput) -> str:
        return COHERENCE_JUDGE_PROMPT.format(
            user_input=eval_input.user_input,
            model_output=eval_input.model_output,
        )

    def _parse_response(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

        return {"score": 0.5, "issues": [], "analysis": "Parse error"}

    def evaluate(self, eval_input: EvalInput, **kwargs: Any) -> EvalResult:
        try:
            prompt = self._build_prompt(eval_input)

            response = self._llm.complete(
                prompt=prompt,
                system_prompt=COHERENCE_JUDGE_SYSTEM_PROMPT,
            )

            parsed = self._parse_response(response.content)

            score = float(parsed.get("score", 0.5))
            issues: List[str] = parsed.get("issues", [])
            analysis: str = parsed.get("analysis", "")

            passed = score >= self.config.threshold

            assertions = [
                AssertionDetail(
                    name="coherence_check",
                    passed=passed,
                    expected="Coherent response",
                    actual=f"Score: {score}",
                    message=analysis,
                )
            ]

            return EvalResult(
                status=EvalStatus.PASSED if passed else EvalStatus.FAILED,
                score=score,
                eval_name=self.eval_name,
                message="Coherence evaluation completed",
                assertions=assertions,
                details={
                    "score": score,
                    "issues": issues,
                    "analysis": analysis,
                },
            )

        except Exception as e:
            return EvalResult.error_result(self.eval_name, e)
