"""Grounded answer generation via the Anthropic API.

Three deliberate choices worth defending in an interview:

1. **Structured output over prose parsing.** `client.messages.parse()` with a
   Pydantic model means the answer arrives already validated. No regex, no
   "sometimes the model wraps JSON in a code fence" branch.

2. **Prompt caching on the instruction preamble.** Render order is
   tools -> system -> messages, and caching is a *prefix* match, so the system
   block holds only content that never varies per query; the retrieved context
   and the question go in `messages`, after the breakpoint. Caveat worth knowing:
   the minimum cacheable prefix is ~1024 tokens -- a short system prompt silently
   won't cache at all, which is why the rubric below is written out in full
   rather than compressed.

3. **Citations verified against what we actually retrieved.** The model can only
   cite chunk IDs we passed in; anything else is a hallucinated citation and gets
   stripped, which can in turn demote the answer to a refusal.
"""

from __future__ import annotations

from typing import Any, cast

import anthropic
import structlog

from groundtruth.config import Settings
from groundtruth.generation.schema import Claim, GroundedAnswer
from groundtruth.retrieval.reranker import RerankedCandidate

log = structlog.get_logger(__name__)

# Static across every request -> safe to cache. Keep it byte-stable: no
# timestamps, no per-request IDs, no f-strings interpolating query state.
SYSTEM_PROMPT = """\
You answer questions about SEC filings (10-K and 10-Q) using ONLY the excerpts \
provided in the user message. You are used in a financial-research setting where \
a confident wrong answer is far more damaging than an admission of missing data.

Rules:

1. Ground every factual claim in the supplied excerpts. Each excerpt is labelled \
with a numeric [chunk_id]. For every claim you make, list the chunk_id values \
that directly support it.

2. Never use outside knowledge about a company, even if you are confident it is \
correct. If the excerpts do not contain the answer, set `refused` to true and \
explain in one sentence what is missing. Refusing is the correct, expected \
behaviour when the retrieval step did not surface the right passage -- it is not \
a failure on your part.

3. Do not infer figures that are not stated. Do not add together numbers from \
different excerpts unless an excerpt explicitly presents them as a sum. If a \
figure appears with a unit or scale qualifier ("in thousands", "in millions"), \
carry that qualifier into your answer verbatim.

4. Distinguish fiscal years carefully. Filings routinely discuss several years in \
one passage; attribute each figure to the year the excerpt assigns it.

5. Quote exact language for anything legal, contractual, or risk-related rather \
than paraphrasing it.

6. Set `confidence` to reflect how completely the excerpts answer the question: \
above 0.8 only when the excerpts state the answer directly; below 0.5 when you \
are assembling a partial picture from tangential passages.

7. Keep the answer concise and factual. No preamble, no restatement of the \
question, no closing offers of further help.
"""


def format_context(candidates: list[RerankedCandidate]) -> str:
    """Render retrieved chunks with the labels the model must cite.

    Provenance goes in the header of each excerpt because the model needs company
    and fiscal year to disambiguate figures -- a bare text dump produces confident
    cross-year errors.
    """
    parts: list[str] = []
    for r in candidates:
        c = r.candidate
        section = f" | {c.section_name}" if c.section_name else ""
        parts.append(
            f"[chunk_id: {c.chunk_id}] {c.company_name} "
            f"({c.ticker or 'n/a'}) FY{c.fiscal_year} {c.form_type}{section}\n"
            f"{c.text}"
        )
    return "\n\n---\n\n".join(parts)


class Generator:
    """Wraps the Anthropic client with the grounding contract."""

    def __init__(self, settings: Settings):
        self._settings = settings
        # Zero-arg-style construction: the SDK also resolves ANTHROPIC_API_KEY or
        # an `ant auth login` profile if the setting is blank.
        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key or None,
        )

    def answer(
        self,
        question: str,
        candidates: list[RerankedCandidate],
    ) -> GroundedAnswer:
        """Generate a grounded answer, or a refusal if the context is insufficient."""
        if not candidates:
            return GroundedAnswer(
                refused=True,
                answer="No relevant passages were retrieved for this question.",
                claims=[],
                confidence=0.0,
            )

        context = format_context(candidates)
        user_content = f"Excerpts from SEC filings:\n\n{context}\n\n---\n\nQuestion: {question}"

        response = self._client.messages.parse(
            model=self._settings.generation_model,
            max_tokens=self._settings.max_output_tokens,
            # Cache the invariant instruction block; everything volatile is in
            # `messages`, which sits after this breakpoint in render order.
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            thinking={"type": "adaptive"},
            output_config={"effort": cast(Any, self._settings.effort)},
            messages=[{"role": "user", "content": user_content}],
            output_format=GroundedAnswer,
        )

        log.info(
            "generation.usage",
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            # If this stays 0 across repeated calls, a silent cache invalidator
            # has crept into the system block.
            cache_read=getattr(response.usage, "cache_read_input_tokens", 0),
            cache_write=getattr(response.usage, "cache_creation_input_tokens", 0),
        )

        # Safety classifiers can decline; `content` is not meaningful then.
        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            log.warning("generation.model_refusal", category=category)
            return GroundedAnswer(
                refused=True,
                answer="The request was declined by a safety filter.",
                claims=[],
                confidence=0.0,
            )

        parsed = response.parsed_output
        if parsed is None:
            return GroundedAnswer(
                refused=True,
                answer="Failed to parse model output.",
                claims=[],
                confidence=0.0,
            )
        return self._verify_citations(parsed, candidates)

    @staticmethod
    def _verify_citations(
        answer: GroundedAnswer, candidates: list[RerankedCandidate]
    ) -> GroundedAnswer:
        """Drop citations to chunks we never supplied; demote if nothing survives.

        A hallucinated chunk_id is rare but not impossible, and it is precisely
        the kind of error that looks fine in a demo and is indefensible in
        production -- the UI would render a citation link to evidence that does
        not support the claim.
        """
        served = {r.candidate.chunk_id for r in candidates}
        cleaned: list[Claim] = []
        dropped = 0

        for claim in answer.claims:
            valid = [cid for cid in claim.chunk_ids if cid in served]
            dropped += len(claim.chunk_ids) - len(valid)
            if valid:
                cleaned.append(claim.model_copy(update={"chunk_ids": valid}))

        if dropped:
            log.warning("generation.invalid_citations_dropped", count=dropped)

        if not answer.refused and not cleaned:
            log.warning("generation.demoted_to_refusal", reason="no_valid_citations")
            return GroundedAnswer(
                refused=True,
                answer=(
                    "An answer was produced but could not be verified against the "
                    "retrieved excerpts, so it was withheld."
                ),
                claims=[],
                confidence=0.0,
            )

        return answer.model_copy(update={"claims": cleaned})
