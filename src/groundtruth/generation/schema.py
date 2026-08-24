"""Answer schema for grounded generation.

Separated from the client so tests and the API layer can import the shape without
pulling in the anthropic SDK. The grounding contract lives here as validation:
a well-formed answer either cites its evidence or explicitly refuses. There is no
third option, because an uncited claim over a financial filing is the exact
failure mode this whole system exists to prevent.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Claim(BaseModel):
    """One assertion in the answer, bound to the chunks that support it."""

    text: str = Field(description="A single factual assertion made in the answer.")
    chunk_ids: list[int] = Field(
        description="IDs of retrieved chunks that directly support this claim.",
    )


class GroundedAnswer(BaseModel):
    """The generator's structured output. Enforced by the model via output_format."""

    refused: bool = Field(
        description="True if the context does not contain enough to answer.",
    )
    answer: str = Field(description="The answer text, or a short refusal explanation.")
    claims: list[Claim] = Field(
        default_factory=list,
        description="Every factual claim in the answer, each with supporting chunk IDs.",
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Model's calibrated confidence in the answer."
    )

    @model_validator(mode="after")
    def _enforce_grounding(self) -> GroundedAnswer:
        """A non-refusal answer must carry at least one cited claim.

        This runs client-side after the model returns, so even if the model
        emits a confident-but-uncited answer we downgrade it to a refusal rather
        than serve an ungrounded claim. Defense in depth: the prompt asks for
        grounding, and this guarantees it.
        """
        if not self.refused and not self.claims:
            raise ValueError("non-refused answer must contain at least one claim")
        if not self.refused and any(not c.chunk_ids for c in self.claims):
            raise ValueError("every claim in a non-refused answer must cite >=1 chunk")
        return self

    def cited_chunk_ids(self) -> set[int]:
        return {cid for claim in self.claims for cid in claim.chunk_ids}
