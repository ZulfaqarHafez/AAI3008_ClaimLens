"""Verifier Agent for claim verification against evidence."""

import logging
from typing import List, Optional
import re
from urllib.parse import urlparse

from ..models.schemas import Claim, Evidence, VerificationResult, Verdict
from ..models.nli_placeholder import BaseVerifier, get_verifier
from ..services.llm_service import LLMService
from ..config import settings
from ..agents.event_frame import EventFrameAgent

logger = logging.getLogger(__name__)


class VerifierAgent:
    """Agent responsible for verifying claims against evidence.
    
    This agent uses either HuggingFace NLI models or OpenAI for
    determining if evidence supports, refutes, or is insufficient
    for verifying a claim.
    
    Designed with dependency injection for easy swapping of the
    underlying verification model.
    """
    
    def __init__(
        self,
        verifier: Optional[BaseVerifier] = None,
        verifier_type: Optional[str] = None,
        llm_service: Optional[LLMService] = None
    ):
        """Initialize the Verifier Agent.
        
        Args:
            verifier: Custom verifier instance (takes precedence)
            verifier_type: Type of verifier to use if no instance provided
            llm_service: LLM service for OpenAI verifier
        """
        if verifier:
            self.llm_service = llm_service
            self.verifier = verifier
        else:
            verifier_type = verifier_type or settings.VERIFIER_TYPE
            
            if verifier_type == "openai":
                self.llm_service = llm_service or LLMService()
                self.verifier = get_verifier("openai", llm_service=self.llm_service)
            elif verifier_type == "claimlens":
                self.llm_service = llm_service
                self.verifier = get_verifier("claimlens")
            else:
                self.llm_service = llm_service
                self.verifier = get_verifier(verifier_type)
        
        self.event_frame_agent = EventFrameAgent(self.llm_service or LLMService())
        logger.info(f"Initialized VerifierAgent with {type(self.verifier).__name__}")
    
    def verify(
        self,
        claim: Claim,
        evidence: List[Evidence]
    ) -> VerificationResult:
        """Verify a single claim against evidence.
        
        Args:
            claim: The claim to verify
            evidence: List of evidence to check against
            
        Returns:
            VerificationResult with verdict, confidence, and reasoning
        """
        logger.info(f"Verifying claim: {claim.text[:50]}...")
        logger.debug(f"Evidence count: {len(evidence)}")
        
        try:
            result = self.verifier.verify(claim, evidence)

            if result.verdict != Verdict.NOT_ENOUGH_INFO and evidence:
                # Event-frame gating: require same-event match across dimensions
                event_check = self._event_frame_check(claim, evidence, result.verdict)
                if event_check == Verdict.NOT_ENOUGH_INFO:
                    result.verdict = Verdict.NOT_ENOUGH_INFO
                    result.confidence = min(result.confidence, 0.4)
                    result.reasoning = (
                        f"{result.reasoning} "
                        "Adjusted to NOT_ENOUGH_INFO because evidence did not confirm the full event."
                    )
                elif event_check == Verdict.REFUTED:
                    result.verdict = Verdict.REFUTED
                    result.confidence = min(max(result.confidence, 0.5), 0.8)
                    result.reasoning = (
                        f"{result.reasoning} "
                        "Adjusted to REFUTED because evidence contradicts a key event dimension."
                    )

            if result.verdict != Verdict.NOT_ENOUGH_INFO and evidence:
                # Direct/contradiction match gating
                if result.verdict == Verdict.REFUTED:
                    has_match = self._has_contradiction_match(claim, evidence)
                    match_reason = "no evidence snippet directly contradicts the claim"
                else:
                    has_match = self._has_direct_match(claim, evidence)
                    match_reason = "no evidence snippet directly matches the claim"

                if not has_match:
                    result.verdict = Verdict.NOT_ENOUGH_INFO
                    result.confidence = min(result.confidence, 0.4)
                    result.reasoning = (
                        f"{result.reasoning} "
                        f"Adjusted to NOT_ENOUGH_INFO because {match_reason}."
                    )

            if result.verdict != Verdict.NOT_ENOUGH_INFO and evidence:
                # Cross-source agreement gating (direct match for SUPPORT, contradiction for REFUTE)
                if not self._has_cross_source_agreement(claim, evidence, result.verdict):
                    result.verdict = Verdict.NOT_ENOUGH_INFO
                    result.confidence = min(result.confidence, 0.4)
                    result.reasoning = (
                        f"{result.reasoning} "
                        "Adjusted to NOT_ENOUGH_INFO because there was no cross-source agreement."
                    )

            # Update claim status
            claim.status = "completed"
            result.claim = claim
            
            logger.info(
                f"Verification result: {result.verdict.value} "
                f"(confidence: {result.confidence:.2f})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            
            # Return NOT_ENOUGH_INFO on failure
            claim.status = "failed"
            return VerificationResult(
                claim=claim,
                evidence_list=evidence,
                verdict=Verdict.NOT_ENOUGH_INFO,
                confidence=0.0,
                reasoning=f"Verification failed: {str(e)}"
            )
    
    def verify_with_retry(
        self,
        claim: Claim,
        evidence: List[Evidence],
        iteration: int = 1
    ) -> VerificationResult:
        """Verify with iteration tracking.
        
        Args:
            claim: The claim to verify
            evidence: List of evidence
            iteration: Current iteration number
            
        Returns:
            VerificationResult with iteration count
        """
        result = self.verify(claim, evidence)
        result.iterations_used = iteration
        return result
    
    def verify_with_context(
        self,
        claim: Claim,
        evidence: List[Evidence],
        iteration: int = 1,
        context_hint: str = "",
        enriched_claim_text: str = ""
    ) -> VerificationResult:
        """Verify with an optional context hint prepended to reasoning.

        For NLI-based verifiers (DeBERTa), the context_hint is appended
        to the reasoning string post-hoc. For LLM-based verifiers it can
        be injected into the prompt.

        Args:
            claim: The claim to verify
            evidence: Evidence list
            iteration: Current iteration number
            context_hint: Context summary from ContextAgent

        Returns:
            VerificationResult, reasoning enriched with context
        """
        from ..models.nli_placeholder import ClaimLensVerifier
        if isinstance(self.verifier, ClaimLensVerifier) and enriched_claim_text:
            result = self.verifier.verify(claim, evidence, enriched_claim_text=enriched_claim_text)
            result.iterations_used = iteration
        else:
            result = self.verify_with_retry(claim, evidence, iteration)

        if context_hint and result.reasoning:
            result.reasoning = f"[Context: {context_hint}] {result.reasoning}"

        return result

    def should_continue_searching(
        self,
        result: VerificationResult,
        current_iteration: int,
        max_iterations: int = None
    ) -> bool:
        """Determine if more evidence should be gathered.
        
        Args:
            result: Current verification result
            current_iteration: Current iteration number
            max_iterations: Maximum allowed iterations
            
        Returns:
            True if more searching should be done
        """
        max_iterations = max_iterations or settings.MAX_VERIFICATION_ITERATIONS
        confidence_threshold = settings.CONFIDENCE_THRESHOLD
        
        # Stop if max iterations reached
        if current_iteration >= max_iterations:
            logger.debug("Max iterations reached, stopping search")
            return False
        
        # Stop if confidence is high enough
        if result.confidence >= confidence_threshold:
            logger.debug(
                f"Confidence {result.confidence:.2f} >= threshold "
                f"{confidence_threshold}, stopping search"
            )
            return False
        
        # Stop if verdict is clear (supported or refuted with reasonable confidence)
        if result.verdict != Verdict.NOT_ENOUGH_INFO and result.confidence >= 0.5:
            logger.debug("Clear verdict reached, stopping search")
            return False
        
        # Continue searching
        logger.debug(
            f"Confidence {result.confidence:.2f} < threshold, "
            f"continuing search (iteration {current_iteration + 1})"
        )
        return True
    
    def get_evidence_gap(
        self,
        claim: Claim,
        evidence: List[Evidence],
        current_result: VerificationResult
    ) -> str:
        """Identify what additional evidence is needed.
        
        Args:
            claim: The claim being verified
            evidence: Current evidence collected
            current_result: Current verification result
            
        Returns:
            Description of what evidence is still needed
        """
        if current_result.verdict == Verdict.NOT_ENOUGH_INFO:
            return "Need more direct evidence addressing the specific claim"
        
        if current_result.confidence < 0.5:
            return "Need more authoritative sources to increase confidence"
        
        if current_result.verdict == Verdict.SUPPORTED:
            return "Consider searching for potential counter-evidence"
        
        return "Need additional corroborating evidence"
    
    async def averify(
        self,
        claim: Claim,
        evidence: List[Evidence]
    ) -> VerificationResult:
        """Async version of verify.
        
        Args:
            claim: The claim to verify
            evidence: List of evidence
            
        Returns:
            VerificationResult
        """
        import asyncio
        return await asyncio.to_thread(self.verify, claim, evidence)
    
    def batch_verify(
        self,
        claims_evidence: List[tuple]
    ) -> List[VerificationResult]:
        """Verify multiple claims in batch.
        
        Args:
            claims_evidence: List of (claim, evidence_list) tuples
            
        Returns:
            List of VerificationResults
        """
        results = []
        
        for claim, evidence in claims_evidence:
            result = self.verify(claim, evidence)
            results.append(result)
        
        return results
    
    def set_verifier(self, verifier: BaseVerifier):
        """Swap the underlying verifier implementation.
        
        This allows runtime switching of verification backends.
        
        Args:
            verifier: New verifier instance to use
        """
        self.verifier = verifier
        logger.info(f"Switched to verifier: {type(verifier).__name__}")

    def _event_frame_check(
        self,
        claim: Claim,
        evidence_list: List[Evidence],
        verdict: Verdict,
    ) -> Verdict:
        """Compare claim event frame to evidence frames to enforce same-event matching."""
        claim_frame = getattr(getattr(claim, "context", None), "event_frame", None)
        if not claim_frame:
            return verdict

        if not self._should_apply_event_gating(claim_frame):
            return verdict

        match_domains = set()
        contradiction_domains = set()

        for ev in evidence_list:
            ev_frame = ev.event_frame
            if not ev_frame:
                continue
            comparison = self.event_frame_agent.compare_frames(claim, claim_frame, ev)
            status = comparison.get("verdict", "partial")
            if settings.DEBUG_MODE:
                logger.debug(
                    "Event frame compare: claim=%s evidence=%s verdict=%s failed=%s reason=%s",
                    claim_frame.model_dump(),
                    ev_frame.model_dump(),
                    status,
                    comparison.get("failed_dimensions"),
                    comparison.get("reason"),
                )
            domain = urlparse(ev.url).netloc.lower().replace("www.", "")
            if status == "match":
                match_domains.add(domain)
            elif status == "contradict":
                contradiction_domains.add(domain)

        if verdict == Verdict.REFUTED:
            if len(contradiction_domains) >= 1:
                return Verdict.REFUTED
            return Verdict.NOT_ENOUGH_INFO

        if len(match_domains) >= 1:
            return Verdict.SUPPORTED
        return Verdict.NOT_ENOUGH_INFO

    def _should_apply_event_gating(self, claim_frame) -> bool:
        """Skip event gating for timeless/attribute claims lacking core event dimensions."""
        filled = 0
        for d in ("person", "action", "location", "time", "context"):
            if getattr(claim_frame, d, None):
                filled += 1
        # Require at least 2 concrete dimensions to treat as an event
        return filled >= 2

    def _has_direct_match(self, claim: Claim, evidence_list: List[Evidence]) -> bool:
        """Check if any evidence snippet directly matches key terms from the claim."""
        claim_keywords, claim_numbers = self._extract_keywords_from_claim(claim)

        for ev in evidence_list:
            snippet = ev.snippet or ""
            snippet_l = snippet.lower()

            # If claim has numbers, require at least one number match
            if claim_numbers and not any(n in snippet for n in claim_numbers):
                continue

            matches = sum(1 for kw in claim_keywords if kw in snippet_l)
            required = 2 if len(claim_keywords) >= 3 else 1
            if matches >= required:
                return True

        return False

    def _has_cross_source_agreement(
        self,
        claim: Claim,
        evidence_list: List[Evidence],
        verdict: Verdict,
    ) -> bool:
        """Require at least two distinct domains with matching evidence."""
        matched_domains = set()
        for ev in evidence_list:
            if verdict == Verdict.REFUTED:
                matched = self._has_contradiction_match(claim, [ev])
            else:
                matched = self._has_direct_match(claim, [ev])

            if matched:
                domain = urlparse(ev.url).netloc.lower().replace("www.", "")
                matched_domains.add(domain)
        return len(matched_domains) >= 2

    def _has_contradiction_match(self, claim: Claim, evidence_list: List[Evidence]) -> bool:
        """Check if evidence explicitly contradicts the claim."""
        claim_keywords, claim_numbers = self._extract_keywords_from_claim(claim)
        contradiction_cues = {
            "not", "no", "cannot", "can't", "unable", "false", "myth",
            "debunk", "incorrect", "untrue", "contradict", "refute",
        }

        for ev in evidence_list:
            snippet = (ev.snippet or "").lower()

            # Keyword overlap requirement
            matches = sum(1 for kw in claim_keywords if kw in snippet)
            required = 2 if len(claim_keywords) >= 3 else 1
            if matches < required:
                continue

            # If claim has numbers, consider a different number as potential contradiction
            if claim_numbers:
                snippet_numbers = re.findall(r"\d+(?:\.\d+)?", snippet)
                if snippet_numbers and any(n not in claim_numbers for n in snippet_numbers):
                    return True

            # Look for explicit contradiction cues
            if any(cue in snippet for cue in contradiction_cues):
                return True

        return False

    def _extract_keywords(self, text: str) -> tuple[list[str], list[str]]:
        """Extract simple keyword set and numbers from claim text."""
        text_l = text.lower()
        numbers = re.findall(r"\d+(?:\.\d+)?", text_l)
        tokens = re.findall(r"[a-z][a-z0-9']+", text_l)
        stopwords = {
            "the", "a", "an", "and", "or", "but", "if", "then", "than", "to", "of", "in",
            "on", "for", "with", "by", "from", "as", "is", "was", "were", "be", "been",
            "it", "this", "that", "these", "those", "he", "she", "they", "we", "you", "i",
            "at", "about", "over", "under", "into", "after", "before", "just", "more",
            "most", "some", "any", "all", "not", "no", "nor", "so",
        }
        keywords = [t for t in tokens if t not in stopwords and len(t) >= 4]
        # De-duplicate while preserving order
        seen = set()
        deduped = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                deduped.append(kw)
        return deduped, numbers

    def _extract_keywords_from_claim(self, claim: Claim) -> tuple[list[str], list[str]]:
        """Extract keywords from claim text plus enriched context and aliases."""
        keywords, numbers = self._extract_keywords(claim.text)
        context = getattr(claim, "context", None)

        def _merge(text: Optional[str]):
            if not text:
                return
            extra_kw, extra_num = self._extract_keywords(text)
            keywords.extend(extra_kw)
            numbers.extend(extra_num)

        if context:
            _merge(context.normalized_claim)
            _merge(context.enriched_claim_text)
            _merge(context.context_summary)
            _merge(context.temporal_context)
            _merge(context.venue_context)
            for alias in context.entity_aliases or []:
                _merge(alias)
            for note in context.context_notes or []:
                _merge(note.entity)
                _merge(note.note)

        # Add lightweight equivalence expansions
        expansions = {
            "dpm": ["deputy", "prime", "minister"],
            "deputy prime minister": ["dpm"],
            "committee of supply": ["parliament", "parliamentary", "parliament of singapore"],
        }
        text_l = f"{claim.text} {context.context_summary if context else ''}".lower()
        for key, vals in expansions.items():
            if key in text_l:
                for v in vals:
                    keywords.extend(self._extract_keywords(v)[0])

        # De-duplicate
        seen = set()
        deduped = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                deduped.append(kw)
        numbers = list(dict.fromkeys(numbers))
        return deduped, numbers
