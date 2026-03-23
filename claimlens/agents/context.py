"""Context Agent for enriching claims with clarifying context."""

import logging
from typing import Optional

from ..models.schemas import Claim, ClaimContext
from ..services.llm_service import LLMService

logger = logging.getLogger(__name__)


class ContextAgent:
    """Agent responsible for enriching claims with contextual details.

    This agent adds clarifying context such as venue, temporal cues,
    entity aliases, and search hints to improve downstream search and
    verification quality.
    """

    SYSTEM_PROMPT = """You are a context enrichment agent for fact-checking.
Your task is to add helpful, non-speculative context to a claim so that
search and verification can interpret it correctly across many domains.

RULES:
1. Do NOT guess missing specifics (e.g., do not invent years or dates).
2. Add only widely known or explicitly implied context that clarifies meaning.
3. Use venue/institution context only when it is clearly implied by the claim.
4. Provide entity aliases/titles only if well-known or explicitly mentioned.
5. If a field is unknown, set it to null or an empty list.
6. enriched_claim_text must preserve ALL specific facts (dates, numbers, names).

Respond with a JSON object containing:
- normalized_claim: string
- context_summary: string
- enriched_claim_text: string
- temporal_context: string or null
- venue_context: string or null
- entity_aliases: array of strings
- search_hints: array of strings
- context_notes: array of {entity, note, confidence}"""

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()

    def enrich(self, claim: Claim) -> ClaimContext:
        """Enrich a claim with contextual details."""
        user_prompt = f"""Enrich the following claim with helpful context:

CLAIM: "{claim.text}"
SOURCE SENTENCE: "{claim.source_sentence}"
"""
        try:
            response_schema = {
                "type": "object",
                "properties": {
                    "normalized_claim": {"type": "string"},
                    "context_summary": {"type": "string"},
                    "enriched_claim_text": {"type": "string"},
                    "temporal_context": {"type": ["string", "null"]},
                    "venue_context": {"type": ["string", "null"]},
                    "entity_aliases": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "search_hints": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "context_notes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entity": {"type": "string"},
                                "note": {"type": "string"},
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            },
                            "required": ["entity", "note", "confidence"],
                        }
                    }
                },
                "required": [
                    "normalized_claim",
                    "context_summary",
                    "enriched_claim_text",
                    "temporal_context",
                    "venue_context",
                    "entity_aliases",
                    "search_hints",
                    "context_notes",
                ]
            }

            result = self.llm_service.generate_structured(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_schema=response_schema,
                temperature=0.2,
                max_tokens=800,
            )

            context = ClaimContext(
                normalized_claim=result["normalized_claim"],
                context_summary=result["context_summary"],
                enriched_claim_text=result.get("enriched_claim_text") or result["normalized_claim"],
                temporal_context=result.get("temporal_context"),
                venue_context=result.get("venue_context"),
                entity_aliases=result.get("entity_aliases", []),
                search_hints=result.get("search_hints", []),
                context_notes=result.get("context_notes", []),
            )
            return self._apply_rule_based_context(claim, context)
        except Exception as e:
            logger.error(f"Context enrichment failed: {e}")
            context = ClaimContext(
                normalized_claim=claim.text,
                context_summary=claim.source_sentence or claim.text,
                enriched_claim_text=claim.text,
                temporal_context=None,
                venue_context=None,
                entity_aliases=[],
                search_hints=[],
                context_notes=[],
            )
            return self._apply_rule_based_context(claim, context)

    def _apply_rule_based_context(self, claim: Claim, context: ClaimContext) -> ClaimContext:
        """Apply deterministic context fixes for known institutional phrases."""
        text_l = f"{claim.text} {claim.source_sentence}".lower()

        # Committee of Supply debates in Singapore are held in Parliament.
        if "committee of supply" in text_l:
            if any(token in text_l for token in ("singapore", "gan kim yong", "dpm", "mti")):
                if not context.venue_context:
                    context.venue_context = "Parliament of Singapore"
                if context.context_summary and "parliament" not in context.context_summary.lower():
                    context.context_summary = (
                        f"{context.context_summary} Committee of Supply debates are held in the Parliament of Singapore."
                    )
        return context
