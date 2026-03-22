"""Event frame extraction agent for claim-evidence matching."""

import logging
from typing import Optional, List

from ..models.schemas import Claim, Evidence, EventFrame
from ..services.llm_service import LLMService

logger = logging.getLogger(__name__)


class EventFrameAgent:
    """Extract structured event frames from claims and evidence."""

    SYSTEM_PROMPT = """You are an event extraction agent for fact-checking.
Extract a concise event frame with these fields:
- person: who performed the action (if any)
- action: what happened
- location: where it happened
- time: when it happened
- context: the broader setting (institution/venue/process)

Rules:
1. Do not guess missing details. If unknown, set null.
2. Use canonical or expanded names when obvious (e.g., DPM -> Deputy Prime Minister).
3. Keep fields short and factual.
"""

    COMPARE_PROMPT = """You compare a claim event frame to an evidence event frame.
Decide whether the evidence supports the SAME EVENT described in the claim.

Rules:
1. Match the same event across: person, action, location, time, context.
2. Use contextual equivalences (e.g., Committee of Supply is part of Parliament; DPM = Deputy Prime Minister).
3. If any key dimension contradicts, verdict = "contradict".
4. If evidence is related but missing key dimensions, verdict = "partial".
5. If evidence is unrelated or too vague, verdict = "insufficient".
6. If evidence confirms the same event (even with different wording), verdict = "match".

Return which dimensions failed or contradicted."""

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()

    def frame_claim(self, claim: Claim) -> EventFrame:
        """Extract event frame from a claim + context."""
        context = claim.context
        context_block = ""
        if context:
            lines: List[str] = []
            if context.normalized_claim:
                lines.append(f"Normalized claim: {context.normalized_claim}")
            if context.enriched_claim_text:
                lines.append(f"Enriched claim: {context.enriched_claim_text}")
            if context.context_summary:
                lines.append(f"Context summary: {context.context_summary}")
            if context.temporal_context:
                lines.append(f"Temporal context: {context.temporal_context}")
            if context.venue_context:
                lines.append(f"Venue context: {context.venue_context}")
            if context.entity_aliases:
                lines.append(f"Entity aliases: {', '.join(context.entity_aliases[:6])}")
            if context.context_notes:
                notes = "; ".join(f"{n.entity}: {n.note}" for n in context.context_notes[:6])
                lines.append(f"Context notes: {notes}")
            if lines:
                context_block = "\nADDITIONAL CONTEXT:\n" + "\n".join(f"- {l}" for l in lines)

        user_prompt = f"""Extract an event frame from this claim.

CLAIM: "{claim.text}"
SOURCE SENTENCE: "{claim.source_sentence}"
{context_block}
"""

        return self._extract_frame(user_prompt)

    def frame_evidence(self, claim: Claim, evidence: Evidence) -> EventFrame:
        """Extract event frame from evidence, aligned to the claim context."""
        context = claim.context
        context_block = ""
        if context:
            lines: List[str] = []
            if context.normalized_claim:
                lines.append(f"Normalized claim: {context.normalized_claim}")
            if context.enriched_claim_text:
                lines.append(f"Enriched claim: {context.enriched_claim_text}")
            if context.context_summary:
                lines.append(f"Context summary: {context.context_summary}")
            if context.temporal_context:
                lines.append(f"Temporal context: {context.temporal_context}")
            if context.venue_context:
                lines.append(f"Venue context: {context.venue_context}")
            if context.entity_aliases:
                lines.append(f"Entity aliases: {', '.join(context.entity_aliases[:6])}")
            if context.context_notes:
                notes = "; ".join(f"{n.entity}: {n.note}" for n in context.context_notes[:6])
                lines.append(f"Context notes: {notes}")
            if lines:
                context_block = "\nADDITIONAL CONTEXT:\n" + "\n".join(f"- {l}" for l in lines)

        user_prompt = f"""Extract an event frame from this evidence snippet.

CLAIM: "{claim.text}"
EVIDENCE TITLE: "{evidence.title}"
EVIDENCE SNIPPET: "{evidence.snippet}"
{context_block}
"""

        return self._extract_frame(user_prompt)

    def _extract_frame(self, user_prompt: str) -> EventFrame:
        try:
            response_schema = {
                "type": "object",
                "properties": {
                    "person": {"type": ["string", "null"]},
                    "action": {"type": ["string", "null"]},
                    "location": {"type": ["string", "null"]},
                    "time": {"type": ["string", "null"]},
                    "context": {"type": ["string", "null"]},
                },
                "required": ["person", "action", "location", "time", "context"],
            }
            result = self.llm_service.generate_structured(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_schema=response_schema,
                temperature=0.1,
                max_tokens=400,
            )
            return EventFrame(
                person=result.get("person"),
                action=result.get("action"),
                location=result.get("location"),
                time=result.get("time"),
                context=result.get("context"),
            )
        except Exception as e:
            logger.error(f"Event frame extraction failed: {e}")
            return EventFrame()

    def compare_frames(
        self,
        claim: Claim,
        claim_frame: EventFrame,
        evidence: Evidence,
    ) -> dict:
        """Compare a claim frame against an evidence frame using LLM reasoning."""
        context = claim.context
        context_block = ""
        if context:
            lines: List[str] = []
            if context.context_summary:
                lines.append(f"Context summary: {context.context_summary}")
            if context.venue_context:
                lines.append(f"Venue context: {context.venue_context}")
            if context.temporal_context:
                lines.append(f"Temporal context: {context.temporal_context}")
            if context.entity_aliases:
                lines.append(f"Entity aliases: {', '.join(context.entity_aliases[:6])}")
            if context.context_notes:
                notes = "; ".join(f"{n.entity}: {n.note}" for n in context.context_notes[:6])
                lines.append(f"Context notes: {notes}")
            if lines:
                context_block = "\nADDITIONAL CONTEXT:\n" + "\n".join(f"- {l}" for l in lines)

        user_prompt = f"""Compare the claim event frame to the evidence event frame.

CLAIM: "{claim.text}"
CLAIM FRAME: {claim_frame.model_dump()}

EVIDENCE TITLE: "{evidence.title}"
EVIDENCE SNIPPET: "{evidence.snippet}"
EVIDENCE FRAME: {evidence.event_frame.model_dump() if evidence.event_frame else {}}
{context_block}
"""

        try:
            response_schema = {
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": ["match", "contradict", "partial", "insufficient"]
                    },
                    "failed_dimensions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "reason": {"type": "string"},
                },
                "required": ["verdict", "failed_dimensions", "reason"],
            }
            return self.llm_service.generate_structured(
                system_prompt=self.COMPARE_PROMPT,
                user_prompt=user_prompt,
                response_schema=response_schema,
                temperature=0.1,
                max_tokens=400,
            )
        except Exception as e:
            logger.error(f"Event frame compare failed: {e}")
            return {"verdict": "partial", "failed_dimensions": [], "reason": "Comparison failed"}
