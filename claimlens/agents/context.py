"""Context Agent for enriching claims with background contextual knowledge."""

import logging
from typing import List, Optional

from ..models.schemas import Claim
from ..services.llm_service import LLMService
from ..services.search_service import SearchService, get_search_service

logger = logging.getLogger(__name__)


class ContextNote:
    """A single context annotation for a claim."""

    def __init__(self, entity: str, note: str, confidence: float = 1.0):
        self.entity = entity
        self.note = note
        self.confidence = confidence

    def __repr__(self):
        return f"ContextNote({self.entity!r}: {self.note!r})"

    def to_dict(self) -> dict:
        return {
            "entity": self.entity,
            "note": self.note,
            "confidence": self.confidence,
        }


class ClaimContext:
    """All contextual enrichment for a single claim."""

    def __init__(self, claim_id: str, claim_text: str):
        self.claim_id = claim_id
        self.claim_text = claim_text
        self.notes: List[ContextNote] = []
        self.enriched_claim_text: str = claim_text
        self.context_summary: str = ""

    def add_note(self, entity: str, note: str, confidence: float = 1.0):
        self.notes.append(ContextNote(entity, note, confidence))

    def as_context_string(self) -> str:
        """Format context notes as a string to inject into prompts."""
        if not self.notes:
            return ""
        lines = ["[Context Notes]"]
        for n in self.notes:
            lines.append(f"- {n.entity}: {n.note}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "notes": [n.to_dict() for n in self.notes],
            "enriched_claim_text": self.enriched_claim_text,
            "context_summary": self.context_summary,
        }


class ContextAgent:
    """Agent that enriches claims with background contextual knowledge.

    Design principles
    -----------------
    1. LLM-only by default (enable_web_lookup=False).
       Web lookups are intentionally disabled by default because they
       frequently inject incorrect or outdated facts (e.g. a book listed
       as "2013" on Amazon when the claim says "2012") that then corrupt
       search queries and NLI hypotheses downstream.

    2. The enriched_claim_text is ONLY used for NLI hypothesis text and
       search query enrichment.  It MUST NOT contradict specific facts
       stated in the original claim (dates, names, numbers).  The LLM
       is explicitly instructed to preserve those.

    3. Context notes are additive — they expand acronyms, identify
       institutions, and provide role context.  They never override
       claim-specific facts.
    """

    CONTEXT_SYSTEM_PROMPT = """You are a context enrichment specialist for a fact-checking pipeline.

Your job is to analyse a factual claim and identify any entities, terms, acronyms, roles, events,
or institutions that a fact-checker would benefit from understanding better before searching for evidence.

CRITICAL RULES:
- DO NOT contradict or override any specific facts stated in the claim (dates, numbers, names, titles).
- If the claim states something happened "in 2012", do NOT change it to 2013 even if you believe the date is different.
  Your job is to ADD context, not to correct the claim — that is the verifier's job.
- Keep all dates, statistics, and specific assertions from the original claim intact in the enriched_claim.
- If you are uncertain about background information, flag it with needs_web_lookup=true and low confidence.
  Do NOT invent or assume facts you are not certain about.

Be BROAD and GENERALISED — do not assume any particular domain. Cover:
- Abbreviations and acronyms (political titles, organisation names, technical terms)
- Named events (debates, summits, trials, elections, ceremonies)
- Roles and positions (job titles, government roles, institutional positions)
- Institutions and organisations (government bodies, companies, NGOs, courts)
- Implicit location/jurisdiction context (which country's laws apply, which body has authority)
- Temporal context (if a date is mentioned, what else was happening then that matters)
- Domain-specific terminology (legal, medical, financial, scientific jargon)

If the claim is simple and self-contained with no ambiguity, return an empty notes list.

You must respond with valid JSON."""

    CONTEXT_USER_TEMPLATE = """Analyse this claim and provide contextual enrichment notes:

CLAIM: "{claim_text}"

ORIGINAL SENTENCE: "{source_sentence}"

Identify all entities, terms, or implicit context a fact-checker would need to understand
before searching for evidence.

IMPORTANT: The enriched_claim field must preserve ALL specific facts from the original claim
exactly as stated (dates, numbers, names). Only add explanatory context around them."""

    CONTEXT_SCHEMA = {
        "type": "object",
        "properties": {
            "notes": {
                "type": "array",
                "description": "Context notes for entities found in the claim",
                "items": {
                    "type": "object",
                    "properties": {
                        "entity": {
                            "type": "string",
                            "description": "The entity, term or acronym as found in the claim"
                        },
                        "note": {
                            "type": "string",
                            "description": "Concise contextual explanation (1-3 sentences). "
                                           "Must not contradict specific facts in the claim."
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence in the note's accuracy. "
                                           "Use <0.5 if uncertain — do not invent facts."
                        },
                        "needs_web_lookup": {
                            "type": "boolean",
                            "description": "True if a web search would help verify or update this note"
                        },
                        "suggested_search_query": {
                            "type": "string",
                            "description": "Suggested search query if needs_web_lookup is true, else empty string"
                        }
                    },
                    "required": ["entity", "note", "confidence", "needs_web_lookup", "suggested_search_query"]
                }
            },
            "context_summary": {
                "type": "string",
                "description": "One or two sentence summary of the key context discovered. "
                               "Must not contradict specific facts in the original claim."
            },
            "enriched_claim": {
                "type": "string",
                "description": "The original claim rewritten to be fully self-contained and unambiguous, "
                               "incorporating context inline. MUST preserve all original dates, numbers, "
                               "and specific facts exactly as stated. If no enrichment is needed, repeat "
                               "the original claim verbatim."
            }
        },
        "required": ["notes", "context_summary", "enriched_claim"]
    }

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        search_service: Optional[SearchService] = None,
        enable_web_lookup: bool = False,          # ← DEFAULT OFF: web lookups corrupt facts
        max_web_lookups_per_claim: int = 2,
        min_confidence_for_lookup: float = 0.5,
    ):
        """Initialise the Context Agent.

        Args:
            llm_service: LLM service for context extraction
            search_service: Search service for web lookups (optional)
            enable_web_lookup: Whether to do web searches for uncertain entities.
                               KEEP FALSE in production — web lookups frequently
                               inject incorrect metadata (wrong publication dates,
                               outdated statistics) that corrupt downstream search
                               queries and NLI hypothesis text.
            max_web_lookups_per_claim: Cap on web searches per claim (cost control)
            min_confidence_for_lookup: Only search if LLM confidence is below this
        """
        self.llm_service = llm_service or LLMService()
        self.enable_web_lookup = enable_web_lookup
        self.max_web_lookups_per_claim = max_web_lookups_per_claim
        self.min_confidence_for_lookup = min_confidence_for_lookup

        if enable_web_lookup:
            try:
                self.search_service = search_service or get_search_service()
            except Exception as e:
                logger.warning(
                    f"Search service unavailable for context web lookups: {e}. "
                    "Continuing with LLM-only context enrichment."
                )
                self.search_service = None
                self.enable_web_lookup = False
        else:
            self.search_service = None

    def enrich(self, claim: Claim) -> ClaimContext:
        """Enrich a single claim with contextual notes.

        Args:
            claim: The claim to enrich

        Returns:
            ClaimContext with notes, enriched text, and summary
        """
        ctx = ClaimContext(claim_id=claim.id, claim_text=claim.text)

        try:
            raw = self.llm_service.generate_structured(
                system_prompt=self.CONTEXT_SYSTEM_PROMPT,
                user_prompt=self.CONTEXT_USER_TEMPLATE.format(
                    claim_text=claim.text,
                    source_sentence=claim.source_sentence,
                ),
                response_schema=self.CONTEXT_SCHEMA,
                temperature=0.1,
            )

            ctx.context_summary = raw.get("context_summary", "")

            # CRITICAL: validate the enriched claim preserves original facts.
            # If the enriched text changes dates/numbers vs the original, fall
            # back to the original to prevent query contamination.
            raw_enriched = raw.get("enriched_claim", claim.text) or claim.text
            ctx.enriched_claim_text = self._validate_enriched_claim(
                original=claim.text,
                enriched=raw_enriched,
            )

            raw_notes = raw.get("notes", [])
            for n in raw_notes:
                entity = n.get("entity", "")
                note_text = n.get("note", "")
                confidence = float(n.get("confidence", 1.0))

                if entity and note_text:
                    ctx.add_note(entity, note_text, confidence)

            logger.info(
                f"Context extraction: {len(ctx.notes)} note(s) for claim "
                f"'{claim.text[:60]}...'"
            )

            # Phase 2: Optional web lookup (disabled by default)
            if self.enable_web_lookup and self.search_service:
                self._web_enrich(ctx, raw_notes)

        except Exception as e:
            logger.error(f"Context enrichment failed for claim '{claim.text[:60]}': {e}")
            ctx.context_summary = ""
            ctx.enriched_claim_text = claim.text  # always fall back to original

        return ctx

    def _validate_enriched_claim(self, original: str, enriched: str) -> str:
        """Guard against the enriched claim overwriting specific facts.

        Extracts numbers and 4-digit years from both strings and falls
        back to the original if any are missing or changed in the enriched
        version.  This prevents the context agent from silently changing
        "2012" to "2013", or "1.82" to some other figure.

        Args:
            original: The original claim text.
            enriched: The LLM-produced enriched rewrite.

        Returns:
            enriched if safe, original otherwise.
        """
        import re

        # Extract all numbers (integers and decimals) from both strings
        original_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', original))
        enriched_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', enriched))

        # If any number from original is missing in enriched, fall back
        missing = original_numbers - enriched_numbers
        if missing:
            logger.warning(
                f"Enriched claim dropped numbers {missing} from original — "
                f"falling back to original claim text to prevent fact corruption."
            )
            return original

        return enriched

    def _web_enrich(self, ctx: ClaimContext, raw_notes: list) -> None:
        """Run targeted web searches for entities flagged as needing lookup.

        WARNING: This method is disabled by default (enable_web_lookup=False).
        Web lookups can inject incorrect metadata.  Only enable if you have
        validated that the search results are trustworthy for your use case.

        Updates ctx.notes in-place with additional information from the web.
        """
        lookup_count = 0

        for raw_note in raw_notes:
            if lookup_count >= self.max_web_lookups_per_claim:
                break

            needs_lookup = raw_note.get("needs_web_lookup", False)
            confidence = float(raw_note.get("confidence", 1.0))
            query = raw_note.get("suggested_search_query", "").strip()
            entity = raw_note.get("entity", "")

            if not needs_lookup or confidence >= self.min_confidence_for_lookup or not query:
                continue

            try:
                logger.debug(f"Web lookup for context entity '{entity}': {query}")
                results = self.search_service.search(query=query, num_results=2)

                if results:
                    snippets = " | ".join(r.snippet[:200] for r in results if r.snippet)
                    if snippets:
                        for note in ctx.notes:
                            if note.entity == entity:
                                note.note = f"{note.note} [Web: {snippets[:300]}]"
                                # Do NOT raise confidence here — web results may be wrong
                                break

                lookup_count += 1

            except Exception as e:
                logger.warning(f"Web lookup failed for entity '{entity}': {e}")

    def enrich_batch(self, claims: List[Claim]) -> List[ClaimContext]:
        """Enrich multiple claims."""
        return [self.enrich(claim) for claim in claims]

    async def aenrich(self, claim: Claim) -> ClaimContext:
        """Async version of enrich."""
        import asyncio
        return await asyncio.to_thread(self.enrich, claim)