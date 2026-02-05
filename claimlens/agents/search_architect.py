"""Search Architect Agent for generating optimized search queries."""

import logging
from typing import List, Optional

from ..models.schemas import Claim
from ..services.llm_service import LLMService

logger = logging.getLogger(__name__)


class SearchArchitectAgent:
    """Agent responsible for generating optimized search queries for claims.
    
    This agent takes an atomic claim and generates diverse search queries
    designed to find both supporting and potentially refuting evidence.
    """
    
    SYSTEM_PROMPT = """You are an expert search query architect for fact-checking. Your task is to generate diverse, effective search queries to verify factual claims.

RULES:
1. Generate 2-3 search queries per claim
2. Include queries likely to find SUPPORTING evidence
3. Include at least one query that might find COUNTER-EVIDENCE or alternative viewpoints
4. Use specific keywords, names, dates, and numbers from the claim
5. Vary query structure: some specific, some broader context
6. Avoid overly broad queries that won't yield relevant results
7. Include authoritative source hints when relevant (e.g., "site:wikipedia.org" concepts)

QUERY STRATEGIES:
- Direct verification: Search for the exact fact
- Context search: Search for related context that would confirm/deny
- Counter-evidence: Search for potential contradictions
- Source search: Target authoritative sources

EXAMPLES:

Claim: "The Eiffel Tower was built in 1889."
Queries:
1. "Eiffel Tower construction date 1889" (direct verification)
2. "Eiffel Tower history when was it built" (context search)
3. "Eiffel Tower construction year Wikipedia" (authoritative source)

Claim: "Amazon rainforest produces 20% of world's oxygen."
Queries:
1. "Amazon rainforest oxygen production percentage" (direct verification)
2. "world oxygen sources rainforest contribution" (context search)  
3. "Amazon 20% oxygen myth debunked" (counter-evidence search)

Respond with a JSON object containing:
- queries: Array of search query strings"""

    def __init__(self, llm_service: Optional[LLMService] = None):
        """Initialize the Search Architect Agent.
        
        Args:
            llm_service: LLM service instance for API calls
        """
        self.llm_service = llm_service or LLMService()
    
    def generate_queries(
        self, 
        claim: Claim, 
        num_queries: int = 3
    ) -> List[str]:
        """Generate search queries for a claim.
        
        Args:
            claim: The claim to generate queries for
            num_queries: Number of queries to generate (2-3 recommended)
            
        Returns:
            List of search query strings
        """
        user_prompt = f"""Generate {num_queries} diverse search queries to verify this claim:

CLAIM: "{claim.text}"

CONTEXT (original sentence): "{claim.source_sentence}"

Generate queries that will help find evidence to support OR refute this claim."""

        try:
            response_schema = {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 5,
                        "description": "List of search queries"
                    },
                    "strategy_notes": {
                        "type": "string",
                        "description": "Brief notes on query strategies used"
                    }
                },
                "required": ["queries"]
            }
            
            result = self.llm_service.generate_structured(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_schema=response_schema,
                temperature=0.3  # Some creativity for diverse queries
            )
            
            queries = result.get("queries", [])[:num_queries]
            
            logger.info(f"Generated {len(queries)} queries for claim: {claim.text[:50]}...")
            for i, q in enumerate(queries):
                logger.debug(f"  Query {i+1}: {q}")
            
            return queries
            
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            # Fallback: generate simple queries from claim text
            return self._fallback_queries(claim)
    
    def _fallback_queries(self, claim: Claim) -> List[str]:
        """Generate simple fallback queries if LLM fails.
        
        Args:
            claim: The claim to generate queries for
            
        Returns:
            List of basic search queries
        """
        # Extract key terms (simple approach)
        text = claim.text
        
        # Basic query: the claim itself
        queries = [text]
        
        # Add "fact check" variant
        queries.append(f"{text} fact check")
        
        # Add "true or false" variant
        queries.append(f"is it true that {text}")
        
        return queries[:3]
    
    async def agenerate_queries(
        self, 
        claim: Claim, 
        num_queries: int = 3
    ) -> List[str]:
        """Async version of generate_queries.
        
        Args:
            claim: The claim to generate queries for
            num_queries: Number of queries to generate
            
        Returns:
            List of search query strings
        """
        import asyncio
        return await asyncio.to_thread(self.generate_queries, claim, num_queries)
    
    def generate_refined_queries(
        self,
        claim: Claim,
        previous_queries: List[str],
        evidence_gap: str
    ) -> List[str]:
        """Generate refined queries based on previous search results.
        
        Args:
            claim: The claim being verified
            previous_queries: Queries that were already tried
            evidence_gap: Description of what evidence is still needed
            
        Returns:
            List of new, refined search queries
        """
        user_prompt = f"""Generate 2 new search queries to find missing evidence for this claim.

CLAIM: "{claim.text}"

PREVIOUS QUERIES TRIED:
{chr(10).join(f'- {q}' for q in previous_queries)}

WHAT'S STILL NEEDED: {evidence_gap}

Generate different queries that might find the missing evidence."""

        try:
            response_schema = {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 3
                    }
                },
                "required": ["queries"]
            }
            
            result = self.llm_service.generate_structured(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_schema=response_schema,
                temperature=0.4
            )
            
            return result.get("queries", [])[:2]
            
        except Exception as e:
            logger.error(f"Refined query generation failed: {e}")
            return []
