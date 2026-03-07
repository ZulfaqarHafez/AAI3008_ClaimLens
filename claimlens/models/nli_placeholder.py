"""NLI placeholder models for verification."""

from abc import ABC, abstractmethod
from typing import List, Tuple
import logging

from ..models.schemas import Evidence, VerificationResult, Verdict, Claim

logger = logging.getLogger(__name__)


class BaseVerifier(ABC):
    """Abstract base class for claim verifiers.
    
    This interface allows easy swapping of verification backends.
    Implement this interface for custom NLI models.
    """
    
    @abstractmethod
    def verify(
        self, 
        claim: Claim, 
        evidence: List[Evidence]
    ) -> VerificationResult:
        """Verify a claim against provided evidence.
        
        Args:
            claim: The claim to verify
            evidence: List of evidence snippets
            
        Returns:
            VerificationResult with verdict, confidence, and reasoning
        """
        pass
    
    @abstractmethod
    def batch_verify(
        self, 
        claims_evidence: List[Tuple[Claim, List[Evidence]]]
    ) -> List[VerificationResult]:
        """Verify multiple claims in batch.
        
        Args:
            claims_evidence: List of (claim, evidence_list) tuples
            
        Returns:
            List of VerificationResults
        """
        pass


class HuggingFaceNLIVerifier(BaseVerifier):
    """Verifier using Hugging Face's BART-large-MNLI model.
    
    This is a placeholder implementation using zero-shot NLI classification.
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        """Initialize the HuggingFace NLI verifier.
        
        Args:
            model_name: Name of the HuggingFace model to use
        """
        self.model_name = model_name
        self._pipeline = None
        
    def _load_model(self):
        """Lazy load the model pipeline."""
        if self._pipeline is None:
            try:
                from transformers import pipeline
                logger.info(f"Loading NLI model: {self.model_name}")
                self._pipeline = pipeline(
                    "zero-shot-classification",
                    model=self.model_name
                )
            except ImportError:
                raise ImportError(
                    "transformers package required. Install with: pip install transformers torch"
                )
        return self._pipeline
    
    def _classify_nli(
        self, 
        premise: str, 
        hypothesis: str
    ) -> Tuple[str, float]:
        """Classify the relationship between premise and hypothesis.
        
        Args:
            premise: The evidence text (premise)
            hypothesis: The claim text (hypothesis)
            
        Returns:
            Tuple of (label, confidence_score)
        """
        pipeline = self._load_model()
        
        # NLI labels
        labels = ["entailment", "contradiction", "neutral"]
        
        result = pipeline(
            premise,
            candidate_labels=labels,
            hypothesis_template="This text {} the claim: {}".format("{}", hypothesis)
        )
        
        # Get the top prediction
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        
        return top_label, top_score
    
    def _aggregate_evidence_nli(
        self, 
        claim: str, 
        evidence_list: List[Evidence]
    ) -> Tuple[Verdict, float, str]:
        """Aggregate NLI results from multiple evidence pieces.
        
        Args:
            claim: The claim text
            evidence_list: List of evidence to check against
            
        Returns:
            Tuple of (verdict, confidence, reasoning)
        """
        if not evidence_list:
            return Verdict.NOT_ENOUGH_INFO, 0.3, "No evidence available for verification."
        
        entailment_scores = []
        contradiction_scores = []
        neutral_scores = []
        
        for evidence in evidence_list:
            label, score = self._classify_nli(evidence.snippet, claim)
            
            if label == "entailment":
                entailment_scores.append(score * evidence.relevance_score)
            elif label == "contradiction":
                contradiction_scores.append(score * evidence.relevance_score)
            else:
                neutral_scores.append(score * evidence.relevance_score)
        
        # Aggregate scores
        avg_entailment = sum(entailment_scores) / len(entailment_scores) if entailment_scores else 0
        avg_contradiction = sum(contradiction_scores) / len(contradiction_scores) if contradiction_scores else 0
        avg_neutral = sum(neutral_scores) / len(neutral_scores) if neutral_scores else 0
        
        # Determine verdict
        max_score = max(avg_entailment, avg_contradiction, avg_neutral)
        
        if max_score < 0.4:
            verdict = Verdict.NOT_ENOUGH_INFO
            confidence = 0.3 + max_score * 0.2
            reasoning = "Evidence is inconclusive or not directly relevant to the claim."
        elif avg_entailment >= avg_contradiction and avg_entailment >= avg_neutral:
            verdict = Verdict.SUPPORTED
            confidence = min(0.95, 0.5 + avg_entailment * 0.5)
            reasoning = f"Evidence supports the claim with {len(entailment_scores)} supporting sources."
        elif avg_contradiction > avg_entailment and avg_contradiction > avg_neutral:
            verdict = Verdict.REFUTED
            confidence = min(0.95, 0.5 + avg_contradiction * 0.5)
            reasoning = f"Evidence contradicts the claim with {len(contradiction_scores)} refuting sources."
        else:
            verdict = Verdict.NOT_ENOUGH_INFO
            confidence = 0.4
            reasoning = "Evidence is mixed or neutral regarding the claim."
        
        return verdict, confidence, reasoning
    
    def verify(
        self, 
        claim: Claim, 
        evidence: List[Evidence]
    ) -> VerificationResult:
        """Verify a single claim against evidence."""
        verdict, confidence, reasoning = self._aggregate_evidence_nli(
            claim.text, 
            evidence
        )
        
        return VerificationResult(
            claim=claim,
            evidence_list=evidence,
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def batch_verify(
        self, 
        claims_evidence: List[Tuple[Claim, List[Evidence]]]
    ) -> List[VerificationResult]:
        """Verify multiple claims in batch."""
        results = []
        for claim, evidence in claims_evidence:
            result = self.verify(claim, evidence)
            results.append(result)
        return results


class OpenAIVerifier(BaseVerifier):
    """Verifier using OpenAI GPT models with structured output.
    
    Uses zero-shot prompting for NLI-style verification.
    """
    
    def __init__(self, llm_service=None):
        """Initialize the OpenAI verifier.
        
        Args:
            llm_service: LLMService instance for API calls
        """
        self._llm_service = llm_service
        
    @property
    def llm_service(self):
        """Lazy load LLM service."""
        if self._llm_service is None:
            from ..services.llm_service import LLMService
            self._llm_service = LLMService()
        return self._llm_service
    
    def verify(
        self, 
        claim: Claim, 
        evidence: List[Evidence]
    ) -> VerificationResult:
        """Verify a claim using OpenAI."""
        if not evidence:
            return VerificationResult(
                claim=claim,
                evidence_list=[],
                verdict=Verdict.NOT_ENOUGH_INFO,
                confidence=0.3,
                reasoning="No evidence available for verification."
            )
        
        # Format evidence for the prompt
        evidence_text = "\n\n".join([
            f"Source {i+1} ({e.url}):\n{e.snippet}"
            for i, e in enumerate(evidence)
        ])
        
        system_prompt = """You are an expert fact-checker performing Natural Language Inference (NLI).
Your task is to determine if the provided evidence SUPPORTS, REFUTES, or provides NOT_ENOUGH_INFO for the given claim.

Rules:
1. SUPPORTED: The evidence clearly confirms the claim is true
2. REFUTED: The evidence clearly contradicts the claim
3. NOT_ENOUGH_INFO: The evidence is insufficient, irrelevant, or inconclusive

Respond with a JSON object containing:
- verdict: "SUPPORTED", "REFUTED", or "NOT_ENOUGH_INFO"
- confidence: A float between 0.0 and 1.0
- reasoning: A brief explanation (1-2 sentences)"""

        user_prompt = f"""Claim to verify:
"{claim.text}"

Evidence:
{evidence_text}

Analyze the evidence and provide your verdict."""

        try:
            result = self.llm_service.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_schema={
                    "type": "object",
                    "properties": {
                        "verdict": {
                            "type": "string",
                            "enum": ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "reasoning": {
                            "type": "string"
                        }
                    },
                    "required": ["verdict", "confidence", "reasoning"]
                }
            )
            
            return VerificationResult(
                claim=claim,
                evidence_list=evidence,
                verdict=Verdict(result["verdict"]),
                confidence=result["confidence"],
                reasoning=result["reasoning"]
            )
            
        except Exception as e:
            logger.error(f"OpenAI verification failed: {e}")
            return VerificationResult(
                claim=claim,
                evidence_list=evidence,
                verdict=Verdict.NOT_ENOUGH_INFO,
                confidence=0.3,
                reasoning=f"Verification failed: {str(e)}"
            )
    
    def batch_verify(
        self, 
        claims_evidence: List[Tuple[Claim, List[Evidence]]]
    ) -> List[VerificationResult]:
        """Verify multiple claims in batch."""
        results = []
        for claim, evidence in claims_evidence:
            result = self.verify(claim, evidence)
            results.append(result)
        return results


class ClaimLensVerifier(BaseVerifier):
    """Verifier using the fine-tuned ClaimLens DeBERTa-v3 NLI model.
    
    Labels: SUPPORTED (0), REFUTED (1), NEI (2).
    Model: Zulfhagez/claimlens-deberta-v3-nli
    """
    
    LABEL_MAP = {
        0: Verdict.SUPPORTED,
        1: Verdict.REFUTED,
        2: Verdict.NOT_ENOUGH_INFO,
    }
    
    def __init__(self, model_path: str = "Zulfhagez/claimlens-deberta-v3-nli"):
        """Initialize the ClaimLens verifier.
        
        Args:
            model_path: HuggingFace model ID or local path
        """
        self.model_path = model_path
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """Lazy-load the model and tokenizer on first use."""
        if self._model is None:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                import torch  # noqa: F401 – ensure torch is available
            except ImportError:
                raise ImportError(
                    "transformers and torch are required. "
                    "Install with: pip install transformers torch sentencepiece accelerate"
                )
            logger.info(f"Loading ClaimLens NLI model: {self.model_path}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self._model.eval()
    
    def _run_nli(
        self,
        claim_text: str,
        evidence_snippet: str,
    ) -> Tuple[Verdict, float]:
        """Run NLI inference for a single evidence-claim pair.
        
        Args:
            claim_text: The claim text (hypothesis)
            evidence_snippet: The evidence text (premise)
            
        Returns:
            Tuple of (verdict_label, confidence)
        """
        import torch
        
        self._load_model()
        
        inputs = self._tokenizer(
            evidence_snippet,
            claim_text,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            logits = self._model(**inputs).logits
        
        probs = torch.softmax(logits, dim=-1).squeeze()
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])
        label = self.LABEL_MAP[pred_idx]
        
        return label, confidence
    
    def verify(
        self, 
        claim: Claim, 
        evidence: List[Evidence]
    ) -> VerificationResult:
        """Verify a claim against evidence using weighted voting."""
        if not evidence:
            return VerificationResult(
                claim=claim,
                evidence_list=[],
                verdict=Verdict.NOT_ENOUGH_INFO,
                confidence=0.3,
                reasoning="No evidence available for verification.",
            )
        
        # Weighted voting across all evidence pieces
        weighted_scores: dict[Verdict, float] = {
            Verdict.SUPPORTED: 0.0,
            Verdict.REFUTED: 0.0,
            Verdict.NOT_ENOUGH_INFO: 0.0,
        }
        
        for ev in evidence:
            label, conf = self._run_nli(claim.text, ev.snippet)
            weighted_scores[label] += conf * ev.relevance_score
        
        # Determine verdict from highest weighted score
        verdict = max(weighted_scores, key=weighted_scores.get)
        total_weight = sum(weighted_scores.values())
        confidence = (
            weighted_scores[verdict] / total_weight
            if total_weight > 0
            else 0.0
        )
        confidence = min(confidence, 0.99)
        
        label_counts = {}
        for v in Verdict:
            if weighted_scores[v] > 0:
                label_counts[v.value] = round(weighted_scores[v], 3)
        
        reasoning = (
            f"Aggregated {len(evidence)} evidence piece(s) via weighted voting. "
            f"Weighted scores: {label_counts}."
        )
        
        return VerificationResult(
            claim=claim,
            evidence_list=evidence,
            verdict=verdict,
            confidence=round(confidence, 4),
            reasoning=reasoning,
        )
    
    def batch_verify(
        self, 
        claims_evidence: List[Tuple[Claim, List[Evidence]]]
    ) -> List[VerificationResult]:
        """Verify multiple claims in batch."""
        return [self.verify(claim, ev) for claim, ev in claims_evidence]


def get_verifier(verifier_type: str = "openai", **kwargs) -> BaseVerifier:
    """Factory function to get the appropriate verifier.
    
    Args:
        verifier_type: Type of verifier ("huggingface", "openai", "claimlens")
        **kwargs: Additional arguments passed to the verifier constructor
        
    Returns:
        An instance of BaseVerifier
    """
    verifiers = {
        "huggingface": HuggingFaceNLIVerifier,
        "openai": OpenAIVerifier,
        "claimlens": ClaimLensVerifier,
    }
    
    if verifier_type not in verifiers:
        raise ValueError(
            f"Unknown verifier type: {verifier_type}. "
            f"Choose from: {list(verifiers.keys())}"
        )
    
    return verifiers[verifier_type](**kwargs)
