"""Tests for ClaimLens pipeline."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from claimlens.models.schemas import (
    Claim,
    ClaimStatus,
    Evidence,
    VerificationResult,
    Verdict,
    FinalReport,
)
from claimlens.agents.decomposition import DecompositionAgent
from claimlens.agents.search_architect import SearchArchitectAgent
from claimlens.agents.verifier import VerifierAgent
from claimlens.models.nli_placeholder import BaseVerifier


class TestDataModels:
    """Test Pydantic data models."""
    
    def test_claim_creation(self):
        """Test Claim model creation."""
        claim = Claim(
            text="The Eiffel Tower is 330 meters tall.",
            source_sentence="The Eiffel Tower is 330 meters tall and beautiful."
        )
        
        assert claim.text == "The Eiffel Tower is 330 meters tall."
        assert claim.status == ClaimStatus.PENDING
        assert claim.id is not None
    
    def test_evidence_creation(self):
        """Test Evidence model creation."""
        evidence = Evidence(
            url="https://example.com",
            title="Example Source",
            snippet="The tower stands at 330 meters.",
            relevance_score=0.9
        )
        
        assert evidence.url == "https://example.com"
        assert evidence.relevance_score == 0.9
        assert evidence.source_quality is None
    
    def test_verification_result_creation(self):
        """Test VerificationResult model creation."""
        claim = Claim(
            text="Test claim",
            source_sentence="Test sentence"
        )
        
        result = VerificationResult(
            claim=claim,
            evidence_list=[],
            verdict=Verdict.SUPPORTED,
            confidence=0.85,
            reasoning="Test reasoning"
        )
        
        assert result.verdict == Verdict.SUPPORTED
        assert result.confidence == 0.85
    
    def test_final_report_trust_score(self):
        """Test FinalReport trust score calculation."""
        claim = Claim(text="Test", source_sentence="Test")
        
        results = [
            VerificationResult(
                claim=claim,
                evidence_list=[],
                verdict=Verdict.SUPPORTED,
                confidence=0.9,
                reasoning="Supported"
            ),
            VerificationResult(
                claim=claim,
                evidence_list=[],
                verdict=Verdict.SUPPORTED,
                confidence=0.8,
                reasoning="Supported"
            ),
        ]
        
        report = FinalReport(
            original_text="Test text",
            claims=[claim, claim],
            verification_results=results
        )
        
        trust_score = report.calculate_trust_score()
        assert 0.0 <= trust_score <= 1.0


class TestDecompositionAgent:
    """Test DecompositionAgent."""
    
    def test_validate_claims_filters_short(self):
        """Test that short claims are filtered out."""
        agent = DecompositionAgent.__new__(DecompositionAgent)
        
        claims = [
            Claim(text="Short", source_sentence="Short sentence"),
            Claim(text="This is a valid claim with enough content.", source_sentence="Test"),
        ]
        
        filtered = agent.validate_claims(claims)
        
        assert len(filtered) == 1
        assert "valid claim" in filtered[0].text
    
    def test_validate_claims_filters_subjective(self):
        """Test that subjective claims are filtered out."""
        agent = DecompositionAgent.__new__(DecompositionAgent)
        
        claims = [
            Claim(text="I think this is the best thing ever.", source_sentence="Test"),
            Claim(text="The population of Paris is 2.1 million.", source_sentence="Test"),
        ]
        
        filtered = agent.validate_claims(claims)
        
        assert len(filtered) == 1
        assert "population" in filtered[0].text


class TestSearchArchitectAgent:
    """Test SearchArchitectAgent."""
    
    def test_fallback_queries(self):
        """Test fallback query generation."""
        agent = SearchArchitectAgent.__new__(SearchArchitectAgent)
        
        claim = Claim(
            text="The Eiffel Tower is in Paris.",
            source_sentence="Test sentence"
        )
        
        queries = agent._fallback_queries(claim)
        
        assert len(queries) == 3
        assert claim.text in queries[0]
        assert "fact check" in queries[1]


class TestVerifierAgent:
    """Test VerifierAgent."""
    
    def test_should_continue_searching_max_iterations(self):
        """Test iteration limit stops searching."""
        # Create mock verifier
        mock_verifier = Mock(spec=BaseVerifier)
        agent = VerifierAgent(verifier=mock_verifier)
        
        claim = Claim(text="Test", source_sentence="Test")
        result = VerificationResult(
            claim=claim,
            evidence_list=[],
            verdict=Verdict.NOT_ENOUGH_INFO,
            confidence=0.3,
            reasoning="Test"
        )
        
        # Should stop at max iterations
        should_continue = agent.should_continue_searching(result, 3, max_iterations=3)
        assert not should_continue
    
    def test_should_continue_searching_high_confidence(self):
        """Test high confidence stops searching."""
        mock_verifier = Mock(spec=BaseVerifier)
        agent = VerifierAgent(verifier=mock_verifier)
        
        claim = Claim(text="Test", source_sentence="Test")
        result = VerificationResult(
            claim=claim,
            evidence_list=[],
            verdict=Verdict.SUPPORTED,
            confidence=0.85,
            reasoning="Test"
        )
        
        # High confidence should stop searching
        should_continue = agent.should_continue_searching(result, 1, max_iterations=3)
        assert not should_continue
    
    def test_should_continue_searching_low_confidence(self):
        """Test low confidence continues searching."""
        mock_verifier = Mock(spec=BaseVerifier)
        agent = VerifierAgent(verifier=mock_verifier)
        
        claim = Claim(text="Test", source_sentence="Test")
        result = VerificationResult(
            claim=claim,
            evidence_list=[],
            verdict=Verdict.NOT_ENOUGH_INFO,
            confidence=0.3,
            reasoning="Test"
        )
        
        # Low confidence should continue
        should_continue = agent.should_continue_searching(result, 1, max_iterations=3)
        assert should_continue


class TestCustomVerifier:
    """Test custom verifier implementation."""
    
    def test_custom_verifier_interface(self):
        """Test that custom verifier implements interface correctly."""
        
        class TestVerifier(BaseVerifier):
            def verify(self, claim, evidence):
                return VerificationResult(
                    claim=claim,
                    evidence_list=evidence,
                    verdict=Verdict.SUPPORTED,
                    confidence=0.9,
                    reasoning="Custom verification"
                )
            
            def batch_verify(self, claims_evidence):
                return [self.verify(c, e) for c, e in claims_evidence]
        
        verifier = TestVerifier()
        claim = Claim(text="Test claim", source_sentence="Test")
        
        result = verifier.verify(claim, [])
        
        assert result.verdict == Verdict.SUPPORTED
        assert result.confidence == 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
