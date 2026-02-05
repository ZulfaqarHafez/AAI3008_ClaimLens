"""Example of using ClaimLens with a custom verifier."""

import os
import sys
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from claimlens.models.schemas import Claim, Evidence, VerificationResult, Verdict
from claimlens.models.nli_placeholder import BaseVerifier
from claimlens.agents.verifier import VerifierAgent
from claimlens.graph import ClaimLensGraph


class CustomDummyVerifier(BaseVerifier):
    """
    Example custom verifier implementation.
    
    This demonstrates how to create a custom verifier that can be
    swapped in for the ClaimLens DeBERTa-v3 model when it's ready.
    """
    
    def __init__(self, model_path: str = None):
        """Initialize the custom verifier.
        
        Args:
            model_path: Path to your custom model weights
        """
        self.model_path = model_path
        # TODO: Load your custom model here
        # self.model = load_model(model_path)
        print(f"CustomDummyVerifier initialized (model_path: {model_path})")
    
    def verify(
        self, 
        claim: Claim, 
        evidence: List[Evidence]
    ) -> VerificationResult:
        """Verify a claim against evidence.
        
        This is where you would implement your custom NLI logic.
        
        Args:
            claim: The claim to verify
            evidence: List of evidence pieces
            
        Returns:
            VerificationResult with verdict and confidence
        """
        # Placeholder logic - replace with actual model inference
        if not evidence:
            return VerificationResult(
                claim=claim,
                evidence_list=[],
                verdict=Verdict.NOT_ENOUGH_INFO,
                confidence=0.3,
                reasoning="No evidence available"
            )
        
        # Dummy scoring based on evidence count and relevance
        avg_relevance = sum(e.relevance_score for e in evidence) / len(evidence)
        
        if avg_relevance > 0.7:
            verdict = Verdict.SUPPORTED
            confidence = 0.6 + avg_relevance * 0.3
        elif avg_relevance < 0.3:
            verdict = Verdict.REFUTED
            confidence = 0.5 + (1 - avg_relevance) * 0.3
        else:
            verdict = Verdict.NOT_ENOUGH_INFO
            confidence = 0.4
        
        return VerificationResult(
            claim=claim,
            evidence_list=evidence,
            verdict=verdict,
            confidence=min(confidence, 0.95),
            reasoning=f"Custom verifier result based on {len(evidence)} evidence pieces"
        )
    
    def batch_verify(
        self, 
        claims_evidence: List[Tuple[Claim, List[Evidence]]]
    ) -> List[VerificationResult]:
        """Verify multiple claims in batch.
        
        For efficiency, implement batch inference here.
        
        Args:
            claims_evidence: List of (claim, evidence) tuples
            
        Returns:
            List of VerificationResults
        """
        return [
            self.verify(claim, evidence) 
            for claim, evidence in claims_evidence
        ]


def main():
    """Demonstrate custom verifier usage."""
    
    print("=" * 60)
    print("ClaimLens - Custom Verifier Example")
    print("=" * 60)
    
    # Create custom verifier instance
    custom_verifier = CustomDummyVerifier(model_path="/path/to/your/model")
    
    # Create verifier agent with custom verifier
    verifier_agent = VerifierAgent(verifier=custom_verifier)
    
    # Create graph with custom verifier agent
    graph = ClaimLensGraph(verifier_agent=verifier_agent)
    
    # Sample text
    sample_text = """
    Python was created by Guido van Rossum and first released in 1991.
    It is one of the most popular programming languages in the world.
    """
    
    print(f"\nInput: {sample_text.strip()}\n")
    print("-" * 60)
    
    # Run verification with custom verifier
    report = graph.run(sample_text)
    
    print(f"\nTrust Score: {report.overall_trust_score:.1%}")
    print(f"Summary: {report.summary}")
    
    for result in report.verification_results:
        print(f"\n- {result.claim.text}")
        print(f"  Verdict: {result.verdict.value} ({result.confidence:.1%})")


if __name__ == "__main__":
    main()
