"""Verifier Agent for claim verification against evidence."""

import logging
from typing import List, Optional

from ..models.schemas import Claim, Evidence, VerificationResult, Verdict
from ..models.nli_placeholder import BaseVerifier, get_verifier
from ..services.llm_service import LLMService
from ..config import settings

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
        self.llm_service = llm_service or LLMService()
        
        if verifier:
            self.verifier = verifier
        else:
            verifier_type = verifier_type or settings.VERIFIER_TYPE
            
            if verifier_type == "openai":
                self.verifier = get_verifier("openai", llm_service=self.llm_service)
            else:
                self.verifier = get_verifier(verifier_type)
        
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
