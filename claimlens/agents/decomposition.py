"""Decomposition Agent for breaking text into atomic claims."""

import logging
from typing import List, Optional

from ..models.schemas import Claim
from ..services.llm_service import LLMService

logger = logging.getLogger(__name__)


class DecompositionAgent:
    """Agent responsible for decomposing text into atomic, verifiable claims.
    
    This agent takes a paragraph of text and breaks it down into simple,
    self-contained factual claims that can be independently verified.
    """
    
    SYSTEM_PROMPT = """You are an expert claim decomposition agent. Your task is to break down complex text into atomic, verifiable factual claims.

RULES:
1. Each claim must be self-contained and independently verifiable
2. Each claim should express exactly ONE factual assertion
3. Remove opinions, subjective statements, and value judgments
4. Preserve the original meaning - don't add or remove information
5. Convert relative references to absolute ones when possible (e.g., "last year" → specific year)
6. Ignore filler phrases, transitions, and rhetorical devices
7. If a sentence contains multiple facts, split them into separate claims
8. Keep claims concise but complete enough to verify

EXAMPLES:

Input: "The Eiffel Tower, which was built in 1889 by Gustave Eiffel, stands at 330 meters and attracts millions of visitors annually."
Output claims:
- "The Eiffel Tower was built in 1889."
- "Gustave Eiffel built the Eiffel Tower."
- "The Eiffel Tower is 330 meters tall."
- "The Eiffel Tower attracts millions of visitors annually."

Input: "I think Paris is the most beautiful city, with its amazing Louvre Museum housing over 35,000 artworks."
Output claims:
- "The Louvre Museum is located in Paris."
- "The Louvre Museum houses over 35,000 artworks."
(Note: "most beautiful" is subjective and excluded)

Respond with a JSON object containing:
- claims: Array of claim objects, each with "text" and "source_sentence" fields"""

    def __init__(self, llm_service: Optional[LLMService] = None):
        """Initialize the Decomposition Agent.
        
        Args:
            llm_service: LLM service instance for API calls
        """
        self.llm_service = llm_service or LLMService()
    
    def decompose(self, text: str) -> List[Claim]:
        """Decompose text into atomic claims.
        
        Args:
            text: Input text paragraph to decompose
            
        Returns:
            List of Claim objects
        """
        if not text or not text.strip():
            return []
        
        user_prompt = f"""Decompose the following text into atomic, verifiable factual claims:

TEXT:
\"\"\"
{text}
\"\"\"

Extract all factual claims, ensuring each is self-contained and verifiable."""

        try:
            response_schema = {
                "type": "object",
                "properties": {
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "The atomic claim text"
                                },
                                "source_sentence": {
                                    "type": "string",
                                    "description": "The original sentence this claim was extracted from"
                                }
                            },
                            "required": ["text", "source_sentence"]
                        }
                    }
                },
                "required": ["claims"]
            }
            
            result = self.llm_service.generate_structured(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_schema=response_schema,
                temperature=0.1  # Low temperature for consistency
            )
            
            claims = []
            for i, claim_data in enumerate(result.get("claims", [])):
                claim = Claim(
                    text=claim_data["text"],
                    source_sentence=claim_data.get("source_sentence", text[:200])
                )
                claims.append(claim)
                logger.debug(f"Extracted claim {i+1}: {claim.text[:50]}...")
            
            logger.info(f"Decomposed text into {len(claims)} claims")
            return claims
            
        except Exception as e:
            logger.error(f"Claim decomposition failed: {e}")
            raise
    
    async def adecompose(self, text: str) -> List[Claim]:
        """Async version of decompose.
        
        Args:
            text: Input text paragraph to decompose
            
        Returns:
            List of Claim objects
        """
        import asyncio
        return await asyncio.to_thread(self.decompose, text)
    
    def validate_claims(self, claims: List[Claim]) -> List[Claim]:
        """Validate and filter claims.
        
        Args:
            claims: List of claims to validate
            
        Returns:
            Filtered list of valid claims
        """
        valid_claims = []
        
        for claim in claims:
            # Basic validation rules
            if len(claim.text) < 10:
                logger.debug(f"Skipping too short claim: {claim.text}")
                continue
            
            if len(claim.text) > 500:
                logger.debug(f"Skipping too long claim: {claim.text[:50]}...")
                continue
            
            # Check for subjective language patterns
            subjective_patterns = [
                "i think", "i believe", "in my opinion", "probably",
                "might be", "could be", "seems like", "arguably",
                "best", "worst", "most beautiful", "most ugly"
            ]
            
            claim_lower = claim.text.lower()
            if any(pattern in claim_lower for pattern in subjective_patterns):
                logger.debug(f"Skipping subjective claim: {claim.text[:50]}...")
                continue
            
            valid_claims.append(claim)
        
        return valid_claims
