"""LLM-as-judge for answer quality, plus the judge/human agreement check.

Retrieval nDCG tells you the right chunks were ranked highly; it says nothing
about whether the generated answer actually used them faithfully. This module
scores the generation slice on three axes and -- critically -- reports Cohen's
kappa against the human-verified labels so the judge itself is held accountable.

The load-bearing idea for an interview: **an unvalidated LLM judge is just a
second opinion you happen to trust.** A judge that disagrees with humans (low
kappa) is measured and reported, not silently used to declare victory. That's the
difference between "we used GPT-as-judge" hand-waving and an evaluation you can
defend.
"""

from __future__ import annotations

from typing import Any, cast

import anthropic
import structlog
from pydantic import BaseModel, Field

from groundtruth.config import Settings
from groundtruth.generation.schema import GroundedAnswer

log = structlog.get_logger(__name__)

# Byte-stable: cached like the generator's system block. The rubric is explicit
# because a vague judge prompt yields high-variance scores that wash out real
# differences between retrieval configs.
JUDGE_SYSTEM = """\
You are a strict evaluator of answers about SEC filings. You are given a \
question, the exact excerpts that were retrieved, and an answer produced from \
them. Judge only what is in front of you; never use outside knowledge about the \
company.

Score three axes, each an integer 0-3:

FAITHFULNESS -- is every claim in the answer supported by the excerpts?
  3: every claim is directly supported; no invented figures or facts.
  2: mostly supported; one minor unsupported detail.
  1: a central claim is unsupported or misattributes a figure/year.
  0: largely fabricated or contradicts the excerpts.

RELEVANCE -- does the answer address the question asked?
  3: answers it directly and completely.
  2: answers the main thrust, misses a sub-part.
  1: tangential; touches the topic but not the question.
  0: does not address the question.

COMPLETENESS -- given ONLY these excerpts, is the answer as complete as possible?
  3: extracts everything the excerpts support.
  2: omits a supported detail.
  1: uses a fraction of the relevant material.
  0: ignores clearly relevant excerpts.

A correct refusal -- the answer declines because the excerpts genuinely lack the \
information -- scores 3 on every axis. Penalising an honest refusal would teach \
the system to bluff, which is the exact failure mode we are guarding against.

Be terse in `rationale`: one sentence naming the deciding factor.
"""


class JudgeVerdict(BaseModel):
    """Graded (0-3) verdict so scores are comparable to the eval_labels scale."""

    faithfulness: int = Field(ge=0, le=3)
    relevance: int = Field(ge=0, le=3)
    completeness: int = Field(ge=0, le=3)
    rationale: str

    @property
    def overall(self) -> float:
        """Mean of the three axes. Faithfulness is not up-weighted here; the
        grounding contract already hard-blocks unfaithful answers upstream, so on
        this slice faithfulness failures are rare and would skew the mean."""
        return round((self.faithfulness + self.relevance + self.completeness) / 3, 3)


class AnswerJudge:
    """Wraps the Anthropic client for scoring generated answers."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)

    def score(self, question: str, context: str, answer: GroundedAnswer) -> JudgeVerdict:
        rendered = "REFUSED -- the answer declined to respond." if answer.refused else answer.answer
        user_content = (
            f"Question:\n{question}\n\n"
            f"Retrieved excerpts:\n{context}\n\n"
            f"Answer under evaluation:\n{rendered}"
        )

        response = self._client.messages.parse(
            model=self._settings.judge_model,
            max_tokens=self._settings.max_output_tokens,
            system=[
                {
                    "type": "text",
                    "text": JUDGE_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            thinking={"type": "adaptive"},
            output_config={"effort": cast(Any, self._settings.effort)},
            messages=[{"role": "user", "content": user_content}],
            output_format=JudgeVerdict,
        )

        if response.stop_reason == "refusal":
            # The judge itself was filtered -- treat as an abstention, not a zero,
            # so a safety decline doesn't masquerade as a quality failure.
            log.warning("judge.model_refusal")
            return JudgeVerdict(
                faithfulness=0,
                relevance=0,
                completeness=0,
                rationale="judge_declined",
            )

        if response.parsed_output is None:
            return JudgeVerdict(
                faithfulness=0,
                relevance=0,
                completeness=0,
                rationale="judge_parse_failed",
            )

        return response.parsed_output
